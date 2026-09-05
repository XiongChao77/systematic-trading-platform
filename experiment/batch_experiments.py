#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch experiment runner (prep -> train -> sim) with resume support.

Key design goals
- Deterministic task spec (tasks_spec.json) and stable param hashing
- Resume by skipping reports already present in reports.jsonl
- Simple process model: prep workers + sim workers; train runs in main process
- Clean separation: path/layout, spec I/O, worker loops, main orchestration
"""

from __future__ import annotations

import argparse
import atexit
import copy
import fcntl
import importlib
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from collections import defaultdict

SNAPSHOT_PATH_ENV = "FINANCIAL_ML_SNAPSHOT_PATH"
ORIGINAL_PROJECT_DIR_ENV = "FINANCIAL_ML_ORIGINAL_PROJECT_DIR"
SOURCE_REVISION_ENV = "FINANCIAL_ML_SOURCE_REVISION"

# Repository-relative files required at runtime but intentionally not tracked.
# Keep this list small and explicit so ignored data, artifacts, and environments
# are never copied accidentally.
essential_files_list = [
    "data_process/config.py",
    "data_process/analyse/volatility_prediction_heatmap.py",
    "experiment/task_constructors.py",
]


def _run_git(repo_root: str, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", repo_root, *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8", errors="surrogateescape")


def _require_clean_worktree(repo_root: str) -> None:
    status = _run_git(repo_root, "status", "--short", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "Git working tree is not clean; commit or remove all staged, unstaged, " f"and untracked changes before starting an experiment:\n{status}"
        )


def _safe_repository_path(repo_root: str, relative_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(relative_path)
    if not relative_path or os.path.isabs(relative_path) or normalized == os.pardir or normalized.startswith(os.pardir + os.sep):
        raise ValueError(f"Snapshot path must stay inside the repository: {relative_path!r}")

    root = os.path.realpath(repo_root)
    source = os.path.realpath(os.path.join(root, normalized))
    if os.path.commonpath([root, source]) != root:
        raise ValueError(f"Snapshot path resolves outside the repository: {relative_path!r}")
    return normalized, source


def _copy_snapshot_file(
    repo_root: str,
    snapshot_root: str,
    relative_path: str,
    *,
    required: bool,
) -> None:
    normalized, source = _safe_repository_path(repo_root, relative_path)
    if not os.path.lexists(source):
        if required:
            raise FileNotFoundError(f"Essential snapshot file does not exist: {normalized}")
        return
    if os.path.isdir(source) and not os.path.islink(source):
        raise ValueError(f"Snapshot entries must be files, not directories: {normalized}")

    destination = os.path.join(snapshot_root, normalized)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)


def _create_source_snapshot(repo_root: str, snapshot_root: str) -> None:
    tracked_output = subprocess.run(
        ["git", "-C", repo_root, "ls-files", "-z", "--cached"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    tracked_files = {os.fsdecode(raw_path) for raw_path in tracked_output.split(b"\0") if raw_path}
    for relative_path in sorted(tracked_files):
        _copy_snapshot_file(
            repo_root,
            snapshot_root,
            relative_path,
            required=False,
        )
    for relative_path in essential_files_list:
        _copy_snapshot_file(
            repo_root,
            snapshot_root,
            relative_path,
            required=True,
        )


def _replace_snapshot_directory(staging_dir: str, snapshot_dir: str) -> None:
    if os.path.islink(snapshot_dir) or os.path.isfile(snapshot_dir):
        os.unlink(snapshot_dir)
    elif os.path.isdir(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    os.replace(staging_dir, snapshot_dir)


def _is_debugging() -> bool:
    """Detect tracing debuggers and Python 3.12+ monitoring debuggers."""
    if sys.gettrace() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    return monitoring is not None and monitoring.get_tool(monitoring.DEBUGGER_ID) is not None


def _clean_check_requested(argv: List[str]) -> bool:
    if _is_debugging():
        return False
    require_clean = True
    for argument in argv:
        if argument == "--check-git-clean":
            require_clean = True
        elif argument == "--no-check-git-clean":
            require_clean = False
    return require_clean


def _restart_from_source_snapshot() -> None:
    if _is_debugging():
        print("Debugger detected: running in the current source tree without Git clean checks or a source snapshot.", flush=True)
        return
    if os.environ.get(SNAPSHOT_PATH_ENV) or any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        return

    script_path = os.path.realpath(__file__)
    repo_root = _run_git(os.path.dirname(script_path), "rev-parse", "--show-toplevel").strip()
    revision = _run_git(repo_root, "rev-parse", "HEAD").strip()
    require_clean = _clean_check_requested(sys.argv[1:])
    if require_clean:
        _require_clean_worktree(repo_root)

    repo_name = os.path.basename(os.path.normpath(repo_root))
    snapshot_dir = os.path.join("/dev/shm", repo_name)
    lock_path = os.path.join("/dev/shm", f".{repo_name}.batch-experiments.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise RuntimeError(f"Another batch experiment is already using snapshot {snapshot_dir}") from exc
    os.set_inheritable(lock_fd, True)

    staging_dir = tempfile.mkdtemp(prefix=f".{repo_name}.snapshot-", dir="/dev/shm")
    try:
        _create_source_snapshot(repo_root, staging_dir)
        if require_clean:
            _require_clean_worktree(repo_root)
            current_revision = _run_git(repo_root, "rev-parse", "HEAD").strip()
            if current_revision != revision:
                raise RuntimeError("Git HEAD changed while creating the source snapshot")
        _replace_snapshot_directory(staging_dir, snapshot_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    relative_script = os.path.relpath(script_path, repo_root)
    snapshot_script = os.path.join(snapshot_dir, relative_script)
    os.environ[SNAPSHOT_PATH_ENV] = snapshot_dir
    os.environ[ORIGINAL_PROJECT_DIR_ENV] = repo_root
    os.environ[SOURCE_REVISION_ENV] = revision
    os.chdir(snapshot_dir)
    print(f"Executing experiment from source snapshot: {snapshot_dir}", flush=True)
    os.execv(sys.executable, [sys.executable, snapshot_script, *sys.argv[1:]])


# Create the snapshot before importing project modules. Spawned multiprocessing
# children execute this file as ``__mp_main__`` and skip this branch.
if __name__ == "__main__":
    _restart_from_source_snapshot()

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

from model import train_config
from data_process import common, preparation

try:
    experiment_tasks = importlib.import_module("experiment.task_constructors")
except ModuleNotFoundError as exc:
    if exc.name != "experiment.task_constructors":
        raise
    experiment_tasks = importlib.import_module("experiment.task_constructors_example")
from data_process.utils import (
    TaskIdentity,
    config_from_dict_train,
    json_safe,
    load_selected_configs,
)
from trade.runner.config import ExperimentContext

# NOTE: train/simulation are imported lazily inside the process that needs them.
#       This avoids CUDA / heavy imports in workers.

TASKS_SPEC_FILE = "tasks_spec.json"
REPORTS_FILE = "reports.jsonl"
TRAIN_REPORTS_FILE = "train_reports.jsonl"
SELECTED_FILE = "selected_configs.jsonl"
PRE_OUTPUT_DIR = "pre_output"
TRAIN_OUTPUT_DIR = "train_output"
PREDICTION_CACHE_DIR = "prediction_cache"
SIM_OUTPUT_DIR = "sim_output"
BACKTEST_REPRODUCTION_DIR = "backtest_reproduction"
ORIGINAL_PREDICTION_CACHE_DIR = "original_prediction_cache"
BACKTEST_REPRODUCTION_DEVICE = "auto"

construct_experiment_tasks = experiment_tasks.construct_experiment_tasks
MAX_PREP = experiment_tasks.MAX_PREP
MAX_TRAIN = experiment_tasks.MAX_TRAIN
MAX_SIM = experiment_tasks.MAX_SIM
INFERENCE_BATCH_SIZE = experiment_tasks.INFERENCE_BATCH_SIZE
SYMBOL = experiment_tasks.SYMBOL
INTERVAL = experiment_tasks.INTERVAL
TRAIN_MODE = experiment_tasks.TRAIN_MODE


class ExperimentTaskError(RuntimeError):
    """A child stage failed; the complete experiment must stop."""


# -----------------------------------------------------------------------------
# Path layout helpers
# -----------------------------------------------------------------------------
def _prep_output_dir(exp_dir: str, pre_h: str) -> str:
    return os.path.join(exp_dir, PRE_OUTPUT_DIR, pre_h)


def _train_output_dir(exp_dir: str, pre_h: str, tr_h: str) -> str:
    return os.path.join(exp_dir, TRAIN_OUTPUT_DIR, f"{pre_h}_{tr_h}")


def _prediction_cache_output_dir(exp_dir: str, pre_h: str, tr_h: str) -> str:
    return os.path.join(exp_dir, PREDICTION_CACHE_DIR, f"{pre_h}_{tr_h}")


def _sim_output_dir(exp_dir: str, full_h: str) -> str:
    return os.path.join(exp_dir, SIM_OUTPUT_DIR, full_h)


def _selected_model_artifact_dir(
    selected_configs_path: str,
    pre_h: str,
    tr_h: str,
) -> str:
    """Resolve a model copied from its original experiment during selection."""
    for name, value in (("preparation", pre_h), ("training", tr_h)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(value)):
            raise ValueError(f"Unsafe {name} artifact hash: {value!r}")

    archive_root = os.path.abspath(os.path.join(os.path.dirname(selected_configs_path), "train"))
    artifact_dir = os.path.abspath(os.path.join(archive_root, f"pre_{pre_h}", f"train_{tr_h}"))
    if os.path.commonpath([archive_root, artifact_dir]) != archive_root:
        raise ValueError("Selected model artifact escaped its archive root: " f"path={artifact_dir}")

    required_files = ("model.pt", "meta.json", "train_config.json")
    missing_files = [filename for filename in required_files if not os.path.isfile(os.path.join(artifact_dir, filename))]
    if missing_files:
        raise FileNotFoundError("Original selected model artifact is incomplete: " f"path={artifact_dir}, missing={missing_files}")
    return artifact_dir


def _cleanup_ephemeral_outputs(exp_dir: str, logger: logging.Logger) -> None:
    for directory_name in (PREDICTION_CACHE_DIR, PRE_OUTPUT_DIR):
        path = os.path.join(exp_dir, directory_name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            logger.info("Removed ephemeral output directory: %s", path)


# -----------------------------------------------------------------------------
# Spec build / load
# -----------------------------------------------------------------------------
def build_task_spec(
    preparation_task: List[Any],
    training_task: List[Any],
    simulation_task: List[Any],
) -> Dict[str, Any]:
    """
    Build a tree spec:
      pre_hash -> {params, train: train_hash -> {params, sim_tasks:[{hash, params}, ...]}}
    NOTE: prep_output_dir/save_dir are NOT written to spec; they are derived from hash layout.
    """
    from trade.runner import backtest_runner

    spec: Dict[str, Any] = {}
    for pre in preparation_task:
        pre_d = asdict(pre)
        pre_d.pop("prep_output_dir", None)
        pre_h = TaskIdentity.prep_hash_for(pre_d)
        _assert_hash_roundtrip("prep", pre_d, common.BaseDefine(**pre_d), pre_h)

        node_pre = spec.setdefault(pre_h, {"params": json_safe(pre_d), "train": {}})
        pre_strategy_configs = [backtest_runner.strategy_config_for_preparation(strategy_config, pre) for strategy_config in simulation_task]

        for tr in training_task:
            tr_d = asdict(tr)
            tr_d.pop("save_dir", None)
            tr_h = TaskIdentity.train_hash_for(tr_d)
            _config_from_dict_train(json_safe(tr_d), expected_hash=tr_h)

            node_tr = node_pre["train"].setdefault(tr_h, {"params": json_safe(tr_d), "sim_tasks": []})

            # de-dup sim tasks by hash
            existing = {s["hash"] for s in node_tr["sim_tasks"]}
            for strategy_config in pre_strategy_configs:
                sim_d = _simulation_params(strategy_config)
                identity = TaskIdentity.from_params(prep=pre_d, train=tr_d, sim=sim_d)
                sim_h = identity.sim_hash
                _assert_sim_hash_roundtrip(
                    sim_d,
                    strategy_config=backtest_runner.strategy_config_from_dict(
                        sim_d["strategy_config"],
                    ),
                    expected_hash=sim_h,
                )
                if sim_h in existing:
                    continue
                node_tr["sim_tasks"].append(
                    {
                        "hash": sim_h,
                        "identity": identity.as_dict(),
                        "params": json_safe(sim_d),
                    }
                )
                existing.add(sim_h)
    return spec


def _count_spec_tasks(task_spec: Dict[str, Any]) -> Tuple[int, int, int]:
    n_prep = len(task_spec)
    n_train = sum(len(n["train"]) for n in task_spec.values())
    n_sim = sum(len(tr["sim_tasks"]) for pre in task_spec.values() for tr in pre["train"].values())
    return n_prep, n_train, n_sim


def load_done_set(reports_path: str) -> set[str]:
    """
    Read reports.jsonl and collect completed params.hash.
    """
    done: set[str] = set()
    if not os.path.exists(reports_path):
        raise RuntimeError(f"{reports_path}")
    with open(reports_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = ((d or {}).get("params") or {}).get("hash")
            if isinstance(h, str) and h:
                done.add(h)
    return done


def _assert_hash_roundtrip(kind: str, original: Dict[str, Any], restored_obj: Any, expected_hash: str) -> None:
    """
    Recompute param_hash from the restored dataclass and check it against the hash the
    dict was stored/looked-up under (pre_h/tr_h/sim_h). This is the same identity the
    rest of the pipeline relies on (dir naming, dedup, done_set matching), so a mismatch
    here means the dict<->dataclass round-trip silently changed the effective params
    (dropped/defaulted field) -- exactly the class of bug that hid in the old
    model_cfg/data_cfg skip.
    """
    restored_d = json_safe(asdict(restored_obj))
    hash_functions = {
        "prep": TaskIdentity.prep_hash_for,
        "train": TaskIdentity.train_hash_for,
    }
    actual_hash = hash_functions[kind](restored_d)
    if actual_hash != expected_hash:
        before = json_safe(original)
        mismatched = {k: (before.get(k), restored_d.get(k)) for k in set(before) | set(restored_d) if before.get(k) != restored_d.get(k)}
        raise ValueError(
            f"{kind} params failed hash round-trip: expected {expected_hash!r}, " f"restored config hashes to {actual_hash!r}. Mismatched fields: {mismatched}"
        )


def _simulation_params(strategy_config: Any) -> Dict[str, Any]:
    from trade.runner import backtest_runner

    return {
        "strategy_config": backtest_runner.strategy_config_to_dict(strategy_config),
        "broker_config": asdict(backtest_runner.BrokerConfig()),
    }


def _assert_sim_hash_roundtrip(
    original: Dict[str, Any],
    *,
    strategy_config: Any,
    expected_hash: str,
) -> None:
    restored = json_safe(_simulation_params(strategy_config))
    actual_hash = TaskIdentity.sim_hash_for(restored)
    if actual_hash != expected_hash:
        raise ValueError(
            f"sim params failed hash round-trip: expected {expected_hash!r}, "
            f"restored config hashes to {actual_hash!r}; "
            f"original={json_safe(original)!r}, restored={restored!r}",
        )


def _config_from_dict_train(train_params: Dict[str, Any], expected_hash: Optional[str] = None):
    """
    Restore TrainConfig from dict stored in task spec, including nested
    model_cfg/data_cfg dataclasses. If expected_hash (the tr_h the dict is
    stored/keyed under) is given, verifies the restored config re-hashes to it.
    """
    t_cfg = config_from_dict_train(train_params)
    if expected_hash is not None:
        _assert_hash_roundtrip("train", train_params, t_cfg, expected_hash)
    return t_cfg


def filter_pending_from_spec(task_spec: Dict[str, Any], done_set: set[str]) -> Dict[str, Any]:
    """
    Filter sim leaf tasks that are already present in reports.jsonl.
    """
    from trade.runner import backtest_runner

    pending: Dict[str, Any] = {}
    for pre_h, pre_node in task_spec.items():
        pre_params = pre_node["params"]
        train_pending: Dict[str, Any] = {}

        for tr_h, tr_node in pre_node["train"].items():
            train_params = tr_node["params"]
            sim_pending = []
            for sim in tr_node.get("sim_tasks", []):
                sim_params = sim["params"]
                identity = TaskIdentity.from_params(
                    prep=pre_params,
                    train=train_params,
                    sim=sim_params,
                )
                saved_identity = sim.get("identity")
                if saved_identity is not None and saved_identity != identity.as_dict():
                    raise ValueError(f"Task identity mismatch for {pre_h}/{tr_h}/{sim['hash']}: " f"stored={saved_identity}, calculated={identity.as_dict()}")
                if identity.full_hash not in done_set:
                    sim_pending.append(sim)

            if sim_pending:
                train_pending[tr_h] = {"params": train_params, "sim_tasks": sim_pending}

        if train_pending:
            pending[pre_h] = {"params": pre_params, "train": train_pending}

    return pending


def load_pending_tasks(exp_dir: str, done_set: set[str]) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
    """
    Load tasks_spec.json then filter already-finished tasks based on reports.jsonl.
    """
    tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
    if not os.path.exists(tasks_spec_path):
        raise FileNotFoundError(f"Tasks spec not found: {tasks_spec_path}")
    with open(tasks_spec_path, "r", encoding="utf-8") as f:
        task_spec = json.load(f)
    total_counts = _count_spec_tasks(task_spec)
    pending = filter_pending_from_spec(task_spec, done_set)
    return pending, total_counts


def _create_output_dirs(task_spec: Dict[str, Any], exp_dir: str) -> None:
    """
    Create prep/train output dirs for all pending tasks.
    """
    for pre_h, pre_node in task_spec.items():
        os.makedirs(_prep_output_dir(exp_dir, pre_h), exist_ok=True)
        for tr_h in pre_node["train"]:
            os.makedirs(_train_output_dir(exp_dir, pre_h, tr_h), exist_ok=True)


# -----------------------------------------------------------------------------
# Worker logging
# -----------------------------------------------------------------------------
def _worker_logger(log_file: str) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s"))
    root.addHandler(fh)
    return root


SWEEPABLE_TYPES = (int, float, str, bool, type(None))


def collect_from_any(
    obj: Any,
    out: Dict[str, Set[Any]],
    prefix: str = "",
):
    if isinstance(obj, SWEEPABLE_TYPES):
        out[prefix].add(obj)
        return

    if is_dataclass(obj):
        for f in fields(obj):
            value = getattr(obj, f.name)
            key = f"{prefix}.{f.name}" if prefix else f.name
            collect_from_any(value, out, key)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            collect_from_any(v, out, key)
        return


def log_param_sweep(logger, sweep):
    logger.info("📌 Experiment parameter sweep:")

    for stage in ["pre", "train", "sim"]:
        if not sweep[stage]:
            continue
        logger.info(f"  [{stage}]")
        for k, v in sweep[stage].items():
            logger.info(f"    {k}: {v}")


def collect_param_sweep(task_spec):
    sweep = {
        "pre": defaultdict(set),
        "train": defaultdict(set),
        "sim": defaultdict(set),
    }

    for pre_node in task_spec.values():
        # pre params
        collect_from_any(pre_node["params"], sweep["pre"])

        for tr_node in pre_node["train"].values():
            collect_from_any(tr_node["params"], sweep["train"])

            for sim in tr_node.get("sim_tasks", []):
                collect_from_any(sim["params"], sweep["sim"])

    def value_sort_key(value):
        if isinstance(value, bool):
            return (1, int(value))
        if isinstance(value, (int, float)):
            return (0, float(value))
        if isinstance(value, str):
            return (2, value)
        if value is None:
            return (3, "")
        return (4, f"{type(value).__name__}:{value!r}")

    def finalize(d):
        return {k: sorted(v, key=value_sort_key) for k, v in d.items() if len(v) > 1}

    return {
        "pre": finalize(sweep["pre"]),
        "train": finalize(sweep["train"]),
        "sim": finalize(sweep["sim"]),
    }


def create_task_spec(logger, exp_dir, done_set: set[str]):
    is_combo = TRAIN_MODE in train_config.COMBO_SUB_TASKS
    preparation_task, training_task, simulation_task = construct_experiment_tasks(
        SYMBOL,
        INTERVAL,
        TRAIN_MODE,
    )

    # In combo_model mode a sub-model is not a tradable strategy on its own, so no sim_tasks are attached here --
    # the real backtest tasks are generated after training for every fused pair,
    # see run_combo_fusion_and_backtest. The real simulation_task list is returned
    # to the caller unchanged, for the fuse stage to use.
    task_spec = build_task_spec(preparation_task, training_task, [] if is_combo else simulation_task)
    # task_spec is already ready
    sweep = collect_param_sweep(task_spec)
    log_param_sweep(logger, sweep)

    tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
    with open(tasks_spec_path, "w", encoding="utf-8") as f:
        json.dump(task_spec, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"📄 Tasks spec saved: {tasks_spec_path}")

    n_prep_total, n_train_total, n_sim_total = _count_spec_tasks(task_spec)
    total_all = n_prep_total + n_train_total + n_sim_total
    logger.info(f"📊 Total: {total_all} (prep={n_prep_total}, train={n_train_total}, sim={n_sim_total})")
    if done_set:
        task_spec = filter_pending_from_spec(task_spec, done_set)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        total_pending = n_prep + n_train + n_sim
        logger.info(f"📊 Pending: {total_pending} (prep={n_prep}, train={n_train}, sim={n_sim}), done: {total_all - total_pending}")
    return task_spec, (simulation_task if is_combo else None)


# -----------------------------------------------------------------------------
# Worker loops
# -----------------------------------------------------------------------------
def _worker_prep(worker_log_file: str, task_queue: mp.Queue, train_queue: mp.Queue, exp_dir: str):
    logger = _worker_logger(worker_log_file)
    while True:
        try:
            msg = task_queue.get(timeout=0.5)
        except Empty:
            continue
        if msg is None:
            break

        task_spec = msg
        for pre_h, pre_task in task_spec.items():
            para = common.BaseDefine(**pre_task["params"])
            t0 = time.time()
            try:
                prep_dir = _prep_output_dir(exp_dir, pre_h)
                preparation.main(logger, para=para, prep_output_dir=prep_dir)
            except Exception:
                logger.exception(f"Prep failed: {pre_h}")
                train_queue.put(("prep_failed", pre_h, time.time() - t0, traceback.format_exc()))
                return

            elapsed = time.time() - t0

            # Build train items for this prep hash (pass only json-safe dicts across processes)
            train_items = []
            for tr_h, tr_node in pre_task.get("train", {}).items():
                save_dir = _train_output_dir(exp_dir, pre_h, tr_h)
                train_items.append(
                    {
                        "pre_h": pre_h,
                        "tr_h": tr_h,
                        "pre_params": copy.deepcopy(pre_task["params"]),
                        "train_params": copy.deepcopy(tr_node["params"]),
                        "sim_tasks": copy.deepcopy(tr_node.get("sim_tasks", [])),
                        "prep_output_dir": prep_dir,
                        "save_dir": save_dir,
                        "prediction_cache_dir": _prediction_cache_output_dir(
                            exp_dir,
                            pre_h,
                            tr_h,
                        ),
                    }
                )

            train_queue.put(("prep_done", pre_h, elapsed, train_items))


def _precompute_backtest_predictions(
    logger: logging.Logger,
    *,
    prep_output_dir: str,
    train_output_dir: str,
    prediction_cache_dir: str,
    train_cfg: Any,
    device: str,
) -> None:
    from trade.runner import backtest_runner

    for period in ("long", "forward"):
        data_config = backtest_runner.ModelDataConfig(
            prep_output_dir=prep_output_dir,
            train_output_dir=train_output_dir,
            prediction_cache_dir=prediction_cache_dir,
            device=device,
            use_prediction_cache=True,
        )
        cache_path = backtest_runner.precompute_prediction_cache(
            logger,
            data_config,
            train_cfg,
            period,
            inference_batch_size=INFERENCE_BATCH_SIZE,
        )
        logger.info("Precomputed %s prediction cache: %s", period, cache_path)


def _train_task(
    worker_log_file: str,
    item: Dict[str, Any],
    sim_task_queue: mp.Queue,
    train_result_queue: mp.Queue,
):
    """Run a single train in its own process, then enqueue sims."""
    logger = _worker_logger(worker_log_file)
    import model.train as train

    pre_h = item["pre_h"]
    tr_h = item["tr_h"]
    pre_params = item["pre_params"]
    train_params = item["train_params"]
    sim_tasks = item.get("sim_tasks", [])
    prep_output_dir = item["prep_output_dir"]
    save_dir = item["save_dir"]
    prediction_cache_dir = item["prediction_cache_dir"]

    t0 = time.time()
    try:
        pre_para = common.BaseDefine(**pre_params)
        t_cfg = _config_from_dict_train(train_params, expected_hash=tr_h)
        # Single model mode (DIRECT_3CLASS/LONG_OVR/...) and the combo_model sub-model mode share one
        # worker: which task is trained is decided by the train_task field of the TrainConfig itself
        # tagged every row when it was built), nothing is hard coded to one task type here.
        result = train.train(
            logger=logger,
            config=t_cfg,
            prep_output_dir=prep_output_dir,
            save_dir=save_dir,
        )

        if sim_tasks:
            # Re-load the just-saved checkpoint in this same training process. If CUDA
            # is available this reuses the process's existing CUDA context; sim workers
            # remain CPU-only and are forbidden from filling a missing cache.
            _precompute_backtest_predictions(
                logger,
                prep_output_dir=prep_output_dir,
                train_output_dir=save_dir,
                prediction_cache_dir=prediction_cache_dir,
                train_cfg=t_cfg,
                device="auto",
            )

        # IMPORTANT: enqueue sims BEFORE reporting train_done (so main can safely send None after last train_done)
        for sim in sim_tasks:
            sim_task_queue.put(
                (
                    pre_h,
                    pre_params,
                    tr_h,
                    train_params,
                    sim,
                    save_dir,
                    prediction_cache_dir,
                    {},
                )
            )
        # The training report (task_type/metrics/save_dir etc.) is also appended to train_reports.jsonl.
        # In single model mode this is just an extra archive; in combo_model mode it is the only data source
        # the later fuse stage uses to group by (pre_key, train_compatibility) and pair the sub-models.
        train_report = {
            "pre_h": pre_h,
            "tr_h": tr_h,
            "task_type": t_cfg.train_task,
            "model_type": t_cfg.model_cfg.model_type,
            "model_version": t_cfg.model_cfg.model_version,
            "train_compatibility": t_cfg.train_compatibility,
            "metrics": result,
            "pre_params": pre_params,
            "train_params": train_params,
            "save_dir": save_dir,
        }
        train_result_queue.put(("train_done", pre_h, tr_h, time.time() - t0, train_report))
    except Exception:
        logger.exception(f"Train failed: {pre_h}/{tr_h}")
        train_result_queue.put(
            (
                "train_failed",
                pre_h,
                tr_h,
                time.time() - t0,
                None,
                traceback.format_exc(),
            )
        )


def _original_model_task(
    worker_log_file: str,
    item: Dict[str, Any],
    sim_task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Load an archived original model and enqueue backtests without training."""
    logger = _worker_logger(worker_log_file)
    pre_h = item["pre_h"]
    tr_h = item["tr_h"]
    train_params = item["train_params"]
    train_output_dir = item["train_output_dir"]
    prediction_cache_dir = item["prediction_cache_dir"]
    t0 = time.time()

    try:
        train_cfg = _config_from_dict_train(train_params, expected_hash=tr_h)
        _precompute_backtest_predictions(
            logger,
            prep_output_dir=item["prep_output_dir"],
            train_output_dir=train_output_dir,
            prediction_cache_dir=prediction_cache_dir,
            train_cfg=train_cfg,
            device=BACKTEST_REPRODUCTION_DEVICE,
        )

        for sim in item.get("sim_tasks", []):
            sim_task_queue.put(
                (
                    pre_h,
                    item["pre_params"],
                    tr_h,
                    train_params,
                    sim,
                    train_output_dir,
                    prediction_cache_dir,
                    {"validation_mode": "original_model_backtest"},
                )
            )

        result_queue.put(("original_model_ready", pre_h, tr_h, time.time() - t0, None))
    except Exception:
        logger.exception("Original model preparation failed: %s/%s", pre_h, tr_h)
        result_queue.put(
            (
                "original_model_failed",
                pre_h,
                tr_h,
                time.time() - t0,
                traceback.format_exc(),
            )
        )


def _worker_sim(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    reports_path: str,
    exp_dir: str,
    experiment_context: ExperimentContext,
    prep_exp_dir: Optional[str] = None,
):
    logger = _worker_logger(worker_log_file)

    from trade.runner import backtest_runner

    while True:
        try:
            msg = task_queue.get(timeout=0.5)
        except Empty:
            continue
        if msg is None:
            break

        # train_output_dir is given explicitly by the producer (the single model _train_task or combo_model's
        # run_combo_fusion_and_backtest), the worker no longer derives the path from (pre_h, tr_h)
        # -- so the same worker pool can run both the single model training output directory and the
        # combined model directory produced by the combo_model fuse; the backtest stage is identical for both.
        (
            pre_h,
            pre_params,
            tr_h,
            train_params,
            sim,
            train_output_dir,
            prediction_cache_dir,
            extra_report_fields,
        ) = msg
        sim_h = sim["hash"]
        sim_params = sim["params"]
        strategy_config = backtest_runner.strategy_config_from_dict(
            sim_params["strategy_config"],
        )
        broker_config = backtest_runner.BrokerConfig(
            **sim_params["broker_config"],
        )
        prep_dir = _prep_output_dir(prep_exp_dir or exp_dir, pre_h)

        t0 = time.time()
        try:
            identity = TaskIdentity.from_params(
                prep=pre_params,
                train=train_params,
                sim=sim_params,
            )
            if identity.prep_hash != pre_h or identity.sim_hash != sim_h:
                raise ValueError(f"Queued task identity mismatch: queued={pre_h}/{tr_h}/{sim_h}, " f"calculated={identity.as_dict()}")
            stored_identity = sim.get("identity")
            if stored_identity is not None and stored_identity != identity.as_dict():
                raise ValueError(f"Stored task identity mismatch: stored={stored_identity}, " f"calculated={identity.as_dict()}")
            report_stat = None

            def run_period(period):
                runner_config = backtest_runner.RunnerConfig(
                    strategy_config=strategy_config,
                    broker_config=broker_config,
                    save_dir=os.path.join(
                        _sim_output_dir(exp_dir, identity.full_hash),
                        period,
                    ),
                    data_config=backtest_runner.ModelDataConfig(
                        prep_output_dir=prep_dir,
                        train_output_dir=train_output_dir,
                        prediction_cache_dir=prediction_cache_dir,
                        device="cpu",
                        use_prediction_cache=True,
                    ),
                    experiment_context=experiment_context,
                )
                return backtest_runner.main(logger, runner_config, period)

            period_results = {}
            period_details = {}
            shared_params = None
            for period in ("long", "forward"):
                period_output = run_period(period)
                period_report = period_output["report"]
                period_detail = period_output["report_details"]
                params = period_report["params"]
                report_identity = params.get("identity")
                if report_identity != identity.as_dict():
                    raise ValueError(f"{period} report identity mismatch: report={report_identity}, " f"task={identity.as_dict()}")
                if shared_params is None:
                    shared_params = params
                elif params != shared_params:
                    raise ValueError("Long and forward report parameters differ")
                period_results[period] = period_report["results"][period]
                period_details[period] = period_detail["results"][period]

            report = {
                **(extra_report_fields or {}),
                "params": shared_params,
                "results": period_results,
            }
            _write_report_details(
                reports_path,
                identity,
                {"results": period_details},
            )
            report_stat = report
        except Exception:
            logger.exception(f"Sim failed: {pre_h}/{tr_h}/{sim_h}")
            result_queue.put(
                (
                    "sim_failed",
                    pre_h,
                    tr_h,
                    sim_h,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return

        elapsed = time.time() - t0
        result_queue.put(("sim_done", pre_h, tr_h, sim_h, elapsed, report_stat, reports_path, train_output_dir))


def _write_report_details(
    output_path: str,
    identity: TaskIdentity,
    report_details: Dict[str, Any],
) -> str:
    detail_output_path = os.path.join(
        _sim_output_dir(os.path.dirname(output_path), identity.full_hash),
        "report_details.json",
    )
    os.makedirs(os.path.dirname(detail_output_path), exist_ok=True)
    temporary_path = f"{detail_output_path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output:
            json.dump(
                report_details,
                output,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        os.replace(temporary_path, detail_output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return detail_output_path


# -----------------------------------------------------------------------------
# Result handling
# -----------------------------------------------------------------------------
def _hardlink_tree(source_dir: str, target_dir: str) -> None:
    """Materialize a strategy artifact tree without duplicating model files."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir, copy_function=os.link)


def _drain_sim_results(
    sim_result_queue: mp.Queue,
    stats: Dict[str, Any],
    logger: logging.Logger,
    eta_msg,
    pending_sim_hashes: Dict[Tuple[str, str], Set[str]],
    valid: bool,
) -> None:
    while True:
        try:
            msg = sim_result_queue.get_nowait()
        except Empty:
            break

        if not msg:
            continue
        typ = msg[0]
        if typ == "sim_failed":
            _, pre_h, tr_h, sim_h, elapsed, error = msg
            raise ExperimentTaskError(f"Simulation failed after {elapsed:.2f}s for " f"{pre_h}/{tr_h}/{sim_h}:\n{error}")
        if typ != "sim_done":
            raise ExperimentTaskError(f"Unexpected simulation result: {msg!r}")

        _, pre_h, tr_h, sim_h, elapsed, report_stat, rp, train_dir = msg
        stats["simulation"]["time"] += elapsed
        stats["simulation"]["count"] += 1

        if report_stat is not None:
            common.append_jsonl(rp, report_stat)
            if valid == True:
                strategy_hash = report_stat["params"]["hash"]
                target_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "valid_train_out", strategy_hash)
                _hardlink_tree(train_dir, target_dir)
                logger.info(f"🚀 Hardlinked artifacts: {tr_h} -> {strategy_hash} {target_dir}")
        # 2. Hash-based cleanup and deletion logic
        train_key = (pre_h, tr_h)
        if train_key in pending_sim_hashes:
            # Remove current finished sim_h from the pending set
            pending_sim_hashes[train_key].discard(sim_h)

            # If all sim tasks for this train task have been removed
            # if not pending_sim_hashes[train_key]:
            #     if os.path.exists(train_dir):
            #         try:
            #             if valid == False:
            #                 shutil.rmtree(train_dir)
            #                 logger.info(f"🧹 All sims finished for Train {tr_h}. Deleted: {train_dir}")
            #         except Exception as e:
            #             logger.error(f"❌ Failed to handle {train_dir}: {e}")

        logger.info(f"    Sim {pre_h}/{tr_h}/{sim_h} done in {elapsed:.2f}s")
        em = eta_msg()
        if em:
            logger.info(f"    {em}")


def _send_none_to_workers(q: mp.Queue, n: int) -> None:
    for _ in range(n):
        q.put(None)


# -----------------------------------------------------------------------------
# Reporting: compare old/new (valid mode)
# -----------------------------------------------------------------------------
def compare_old_new_reports(
    old_reports_path: str,
    new_reports_path: str,
    output_dir: str,
    logger: logging.Logger,
    output_filename: str = "compare_reports.jsonl",
):
    """
    Enhanced comparison between old (selected_configs) and new (reports) files.
    Supports CAGR precision comparison for \"long\" and \"forward\" periods (rounded to 1 decimal place).
    """
    logger.info("\n" + "=" * 40)
    logger.info("📊 Starting Multi-Period Comparison...")

    periods = ["long", "forward"]

    # --- 1. Define internal loader to avoid redundant I/O ---
    def load_records_by_hash(path: str, is_selected_config: bool = False) -> Dict[str, Dict[str, Any]]:
        """Load records from file and index them by hash, preserving all period information."""
        data_map = {}
        if not os.path.exists(path):
            return data_map

        # Support both selected_configs (list) and reports (jsonl)
        try:
            if is_selected_config:
                # Assume load_selected_configs is the existing helper
                records = load_selected_configs(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    records = [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"❌ Failed to load {path}: {e}")
            return data_map
        for record in records:
            h = record.get("params", {}).get("hash")
            data_map[h] = record
        return data_map

    # --- 2. Load data ---
    old_data = load_records_by_hash(old_reports_path, is_selected_config=True)
    new_data = load_records_by_hash(new_reports_path, is_selected_config=False)

    logger.info(f"📥 Loaded {len(old_data)} old records and {len(new_data)} new records")

    # --- 3. Core comparison logic ---
    compare_results = []
    hashes_only_in_old = []
    hashes_only_in_new = list(set(new_data.keys()) - set(old_data.keys()))

    for h, old_record in old_data.items():
        if h not in new_data:
            hashes_only_in_old.append(h)
            continue

        new_record = new_data[h]
        # Initialize comparison entry
        comparison_entry = {"hash": h, "verify_all_passed": True, "period_details": {}}

        # Iterate over three periods
        for p in periods:
            old_p = (old_record.get("results") or {}).get(p)
            new_p = (new_record.get("results") or {}).get(p)

            # Case A: both reports contain this period
            if old_p and new_p:
                old_cagr = old_p.get("performance", {}).get("cagr")
                new_cagr = new_p.get("performance", {}).get("cagr")

                # Only compare when both CAGR values are numeric
                if isinstance(old_cagr, (int, float)) and isinstance(new_cagr, (int, float)):
                    # Compare with one decimal place (e.g., 0.1556 represents 15.6%)
                    v1 = round(old_cagr, 1)
                    v2 = round(new_cagr, 1)
                    is_match = v1 == v2

                    if not is_match:
                        comparison_entry["verify_all_passed"] = False

                    comparison_entry["period_details"][p] = {"status": "match" if is_match else "mismatch", "old_cagr": v1, "new_cagr": v2}
                else:
                    comparison_entry["period_details"][p] = {"status": "missing_performance_data"}

            # Case B: one side is missing this period
            elif old_p or new_p:
                comparison_entry["period_details"][p] = {"status": "period_not_in_both"}
                # If the period should exist in the strategy but is missing, mark as failed
                comparison_entry["verify_all_passed"] = False

        # Keep params in results for easier inspection
        comparison_entry["params"] = old_record.get("params")
        compare_results.append(comparison_entry)

    # --- 4. Save and summarize ---
    if not compare_results:
        logger.warning("⚠️ No matching hashes found to compare.")
        return None, 0, len(hashes_only_in_old), len(hashes_only_in_new)

    output_path = os.path.join(output_dir, output_filename)
    failed_count = sum(1 for r in compare_results if not r["verify_all_passed"])

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in compare_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"✅ Comparison finished. Result saved to: {output_path}")
    logger.info(f"📊 Matched: {len(compare_results)} | Failed: {failed_count}")
    logger.info(f"ℹ️  Only in old: {len(hashes_only_in_old)} | Only in new: {len(hashes_only_in_new)}")

    return output_path, len(compare_results), len(hashes_only_in_old), len(hashes_only_in_new)


# -----------------------------------------------------------------------------
# ETA helper
# -----------------------------------------------------------------------------
def _make_eta_fn(n_prep: int, n_train: int, n_sim: int, stats: Dict[str, Any]):
    def phase_eta(total: int, count: int, elapsed: float, workers: int) -> Optional[float]:
        if total <= 0 or count >= total:
            return 0.0
        if count <= 0:
            return None
        return (elapsed / count) * (total - count) / max(1, workers)

    def fmt(seconds: Optional[float]) -> str:
        if seconds is None:
            return "—"
        if seconds <= 0:
            return "0h"
        hours = seconds / 3600
        return f"{seconds:.0f}s" if hours < 0.1 else f"{hours:.2f}h"

    def eta_msg() -> str:
        prep_eta = phase_eta(n_prep, stats["preparation"]["count"], stats["preparation"]["time"], MAX_PREP)
        train_eta = phase_eta(n_train, stats["train"]["count"], stats["train"]["time"], 1)
        sim_eta = phase_eta(n_sim, stats["simulation"]["count"], stats["simulation"]["time"], MAX_SIM)

        parts = []
        if n_prep > 0:
            parts.append(f"prep:{fmt(prep_eta)}")
        if n_train > 0:
            parts.append(f"train:{fmt(train_eta)}")
        if n_sim > 0:
            parts.append(f"sim:{fmt(sim_eta)}")

        if not parts:
            return ""

        total_eta = None
        if (
            (n_prep == 0 or stats["preparation"]["count"] > 0)
            and (n_train == 0 or stats["train"]["count"] > 0)
            and (n_sim == 0 or stats["simulation"]["count"] > 0)
        ):
            total_eta = (prep_eta or 0) + (train_eta or 0) + (sim_eta or 0)

        msg = "[ETA] " + ", ".join(parts)
        if total_eta is not None and total_eta > 0:
            msg += f" | total ~{fmt(total_eta)}"
        return msg

    return eta_msg


# -----------------------------------------------------------------------------
# CLI / entry
# -----------------------------------------------------------------------------
def _setup_root_logger(exp_dir: str) -> logging.Logger:
    log_file_path = os.path.join(exp_dir, "experiment.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    return logger


def _assert_experiment_revision(
    context: ExperimentContext,
    *,
    check_git_clean: bool = True,
) -> None:
    current_commit = common.git_revision(require_clean=check_git_clean)
    if current_commit != context.git_commit:
        raise RuntimeError("Git state changed while the experiment was running: " f"started={context.git_commit}, current={current_commit}")


def run_combo_fusion_and_backtest(
    logger: logging.Logger,
    train_reports_path: str,
    exp_dir: str,
    simulation_task: List[Any],
    reports_path: str,
    sim_task_queue: mp.Queue,
    sim_result_queue: mp.Queue,
    sim_workers: List[mp.Process],
    valid: bool = False,
):
    """
    Fuse + backtest stage that follows the combo_model training:

    1. Read every finished sub-model record back from train_reports_path (one line appended per finished
       sub-model, see _train_task / run_task_spec's _drain_train_results).
    2. Group them by (pre_h, train_compatibility) -- only two sub-models built on the same data preparation (pre_h)
       and with identical seq_len/stride/feature set (train_compatibility) may be paired, otherwise the input
       windows the two sub-models consume do not line up and fusing them is meaningless. This constraint follows
       the same idea as build_model_registry_from_reports/select_fusion_pairs in
       batch_simulation.py.
    3. Inside a group, take the cartesian product of the two roles of TRAIN_MODE (COMBO_SUB_TASKS[TRAIN_MODE]):
       M role_1 sub-models x N role_2 sub-models = M*N combinations.
    4. Every pair calls fusion_trigger_dir / fusion_long_short_ovr to write the combined model's
       task_description.json, then runs the long/forward backtests through simulation_task and
       writes to reports_path. In the backtest report, model_metrics (overall 3-class metrics after fusion) and
       sub_model_metrics (metrics of both sub-models on the backtest data) are attached automatically by
       backtest_runner.main -> ModelHandler.evaluate_sub_models, nothing has to be computed here.
    """
    import model.train as train

    role_1, role_2 = train_config.COMBO_SUB_TASKS[TRAIN_MODE]

    if not os.path.exists(train_reports_path):
        logger.warning(f"No train reports found at {train_reports_path}, nothing to fuse.")
        return

    records = []
    with open(train_reports_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    groups: Dict[Tuple[str, str], Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[(r["pre_h"], r["train_compatibility"])][r["task_type"]].append(r)

    fusion_pairs = []
    for (pre_h, compat), by_role in groups.items():
        role_1_items = by_role.get(role_1, [])
        role_2_items = by_role.get(role_2, [])
        logger.info(
            f"[combo fuse] pre={pre_h}, compat={compat}: "
            f"{role_1}={len(role_1_items)}, {role_2}={len(role_2_items)} -> "
            f"{len(role_1_items) * len(role_2_items)} pairs"
        )
        for r1 in role_1_items:
            for r2 in role_2_items:
                fusion_pairs.append((pre_h, compat, r1, r2))

    logger.info(f"[combo fuse] total fusion pairs: {len(fusion_pairs)}")

    # Coarse grained resume safety: skip when the whole fusion_hash (not per sim_task) already appears
    # in reports_path -- not full --resume support, just a guard so an accidentally repeated script run
    # does not write the same batch of backtests twice.
    done_fusion_hashes = set()
    if os.path.exists(reports_path):
        with open(reports_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fh = d.get("fusion_hash")
                if fh:
                    done_fusion_hashes.add(fh)

    pending_sim_hashes: Dict[Tuple[str, str], Set[str]] = {}
    n_sim_total = 0

    for pre_h, compat, r1, r2 in fusion_pairs:
        payload = {
            "pre_h": pre_h,
            "compat": compat,
            f"{role_1}_tr_h": r1["tr_h"],
            f"{role_2}_tr_h": r2["tr_h"],
        }
        fusion_hash = TaskIdentity.train_hash_for({"fusion": payload})

        if fusion_hash in done_fusion_hashes:
            logger.info(f"[combo fuse] skip fusion_hash={fusion_hash} (already in reports)")
            continue

        fusion_dir = _train_output_dir(exp_dir, pre_h, fusion_hash)
        os.makedirs(fusion_dir, exist_ok=True)

        logger.info(f"[combo fuse] fusing {role_1}={r1['tr_h']} + {role_2}={r2['tr_h']} -> {fusion_hash}")
        if TRAIN_MODE == train_config.TrainTask.TRIGGER_DIRECTION:
            train.fusion_trigger_dir(logger, r1["save_dir"], r2["save_dir"], fusion_dir)
        else:
            train.fusion_long_short_ovr(logger, r1["save_dir"], r2["save_dir"], fusion_dir)

        # A fused model only exists after both training processes finish. Build
        # its cache here on CPU before any sim worker can observe the task.
        _precompute_backtest_predictions(
            logger,
            prep_output_dir=_prep_output_dir(exp_dir, pre_h),
            train_output_dir=fusion_dir,
            prediction_cache_dir=_prediction_cache_output_dir(
                exp_dir,
                pre_h,
                fusion_hash,
            ),
            train_cfg=_config_from_dict_train(r1["train_params"]),
            device="cpu",
        )

        # The backtest stage is identical to the single model mode (a fused directory is just an ordinary
        # train_output_dir as far as backtest_runner.main is concerned), so instead of calling backtest_runner.main
        # sequentially here, every sim_task is pushed into the same sim_task_queue the single model mode uses and
        # handled in parallel by the MAX_SIM _worker_sim processes already running -- in combo mode those worker
        # processes are idle anyway (the sub-model tasks attach no sim_tasks), so they are put to good use.
        extra_report_fields = {
            "fusion_hash": fusion_hash,
            "train_compatibility": compat,
            role_1: {"tr_h": r1["tr_h"], "metrics": r1["metrics"], "train_params": r1["train_params"]},
            role_2: {"tr_h": r2["tr_h"], "metrics": r2["metrics"], "train_params": r2["train_params"]},
        }
        for strategy_config in simulation_task:
            pre_para = common.BaseDefine(**r1["pre_params"])
            effective_strategy_config = backtest_runner.strategy_config_for_preparation(
                strategy_config,
                pre_para,
            )
            sim_d = json_safe(_simulation_params(effective_strategy_config))
            identity = TaskIdentity.from_params(
                prep=r1["pre_params"],
                train=r1["train_params"],
                sim=sim_d,
            )
            sim_h = identity.sim_hash
            pending_sim_hashes.setdefault((pre_h, fusion_hash), set()).add(sim_h)
            sim_task_queue.put(
                (
                    pre_h,
                    r1["pre_params"],
                    fusion_hash,
                    r1["train_params"],
                    {"hash": sim_h, "identity": identity.as_dict(), "params": sim_d},
                    fusion_dir,
                    _prediction_cache_output_dir(exp_dir, pre_h, fusion_hash),
                    extra_report_fields,
                )
            )
            n_sim_total += 1

    logger.info(f"[combo fuse] enqueued {n_sim_total} backtest tasks across {len(fusion_pairs)} fusion pairs")

    if n_sim_total == 0:
        return

    # Reuses the same _drain_sim_results as the single model mode: writes reports_path and deletes the matching
    # fusion_dir once every sim finished (mirroring the cleanup of the single model training directories), except
    # that there is no prep/train stage to coordinate with here, so a simple polling loop is enough.
    stats = {"simulation": {"time": 0.0, "count": 0}}

    def _no_eta():
        return ""

    while stats["simulation"]["count"] < n_sim_total:
        for process in sim_workers:
            if process.exitcode not in (None, 0):
                raise ExperimentTaskError(f"combo simulation process {process.name} exited " f"with code {process.exitcode}")
        _drain_sim_results(
            sim_result_queue,
            stats,
            logger,
            _no_eta,
            pending_sim_hashes,
            valid,
        )
        if stats["simulation"]["count"] < n_sim_total:
            time.sleep(0.5)
    logger.info(f"[combo fuse] all {n_sim_total} backtest tasks done.")


def main():
    parser = argparse.ArgumentParser(description="Batch experiments: prep -> train -> sim (with resume)")
    parser.add_argument("-a", "--add", type=str, help="add more to exist expirement")
    parser.add_argument("-v", "--valid", action="store_true", default=False, help="Rerun selected_configs.jsonl then compare")
    parser.add_argument("-r", "--resume", type=str, help="Resume experiment from specified directory name under PERSISTENCE_DIR")
    parser.add_argument("-l", "--load", action="store_true", default=False, help="load condidate configs for verification,befor applying to market")
    parser.add_argument(
        "--check-git-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Require a clean Git working tree when creating the source snapshot " "(disable with --no-check-git-clean; always disabled under a debugger)"),
    )

    args = parser.parse_args()
    if _is_debugging():
        args.check_git_clean = False
    experiment_context = ExperimentContext(
        git_commit=common.git_revision(require_clean=args.check_git_clean),
    )

    if TRAIN_MODE in train_config.COMBO_SUB_TASKS and (args.resume or args.add or args.valid or args.load):
        raise ValueError(
            "combo_model 模式 (TRAIN_MODE=TRIGGER_DIRECTION/LONG_SHORT_OVR) 暂不支持 "
            "--resume/--add/--valid/--load，这些流程假设的是单模型模式下 "
            "sim_tasks 直接挂在训练节点上的 spec 形状。请用不带这些参数的全新实验跑一遍。"
        )

    # ---------------- resolve exp_dir ----------------
    if args.resume:
        exp_dir = os.path.join(common.PERSISTENCE_DIR, args.resume)
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Resume directory not found: {exp_dir}")
    elif args.add:
        exp_dir = os.path.join(common.PERSISTENCE_DIR, args.add)
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Add directory not found: {exp_dir}")
    elif args.valid:
        selected_configs_source = os.path.join(
            common.PERSISTENCE_DIR,
            "batch_experiments",
            "selected_configs",
            SELECTED_FILE,
        )
        if not os.path.exists(selected_configs_source):
            raise FileNotFoundError(f"Valid config file not found: {selected_configs_source}")
        exp_dir = os.path.join(
            common.PERSISTENCE_DIR,
            "batch_experiments",
            "valid_train_out",
        )
        shutil.rmtree(exp_dir, ignore_errors=True)
        os.makedirs(exp_dir, exist_ok=True)
        selected_configs = os.path.join(exp_dir, SELECTED_FILE)
        shutil.copy2(selected_configs_source, selected_configs)
    elif args.load:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        records = common.load_selected_configs(selected_configs)  # just to validate file and format
        from trade.runner import backtest_runner
        import model.train as train

        exp_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "load_configs")
        os.makedirs(exp_dir, exist_ok=True)
        logger = _setup_root_logger(exp_dir)
        logger.info("Experiment Git commit: %s", experiment_context.git_commit)
        logger.info("Experiment source snapshot: %s", os.environ.get(SNAPSHOT_PATH_ENV))
        if not args.check_git_clean:
            logger.warning("The startup Git clean-worktree check is disabled")
        begin_time = time.time()
        results = []
        for r in records:
            params = r["params"]
            strategy_config = backtest_runner.strategy_config_from_dict(
                params["strategy"],
            )
            broker_config = backtest_runner.BrokerConfig(
                **params["broker"],
            )
            pre_para = common.BaseDefine(**params["common"])
            train_cfg = _config_from_dict_train(params["train"])
            load_prep_output_dir = os.path.join(common.TEMPORARY_DIR, "batch_experiments", "load_configs", "prep", f"{pre_para.symbol}_{pre_para.interval}")
            strategy_hash = params["hash"]
            # prepare train output for market
            train_save_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "valid_train_out", strategy_hash)
            if not os.path.exists(train_save_dir):
                raise FileNotFoundError(f"Training artifact not found for {strategy_hash}: {train_save_dir}")
            preparation.main(logger, para=pre_para, prep_output_dir=load_prep_output_dir)
            data_config = backtest_runner.ModelDataConfig(
                prep_output_dir=load_prep_output_dir,
                train_output_dir=train_save_dir,
                device="cpu",
                use_prediction_cache=True,
            )
            for period in ("long", "forward"):
                backtest_runner.precompute_prediction_cache(
                    logger,
                    data_config,
                    train_cfg,
                    period,
                    inference_batch_size=INFERENCE_BATCH_SIZE,
                )
            last_cagr = 0
            for risk_per_trade_pct in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                result = {}
                strategy_config.risk_per_trade_pct = risk_per_trade_pct
                result[strategy_hash] = {risk_per_trade_pct: {"cagr": {}}}

                def run_selected_period(period):
                    runner_config = backtest_runner.RunnerConfig(
                        strategy_config=strategy_config,
                        broker_config=broker_config,
                        save_dir=os.path.join(
                            exp_dir,
                            "trade_risk_test",
                            strategy_hash,
                            str(risk_per_trade_pct),
                            period,
                        ),
                        data_config=data_config,
                        experiment_context=experiment_context,
                    )
                    output = backtest_runner.main(logger, runner_config, period)
                    return output["report"]["results"][period]

                long_result = run_selected_period("long")
                forward_result = run_selected_period("forward")
                result[strategy_hash][risk_per_trade_pct]["cagr"]["long"] = long_result["performance"]["cagr"]
                result[strategy_hash][risk_per_trade_pct]["cagr"]["forward"] = forward_result["performance"]["cagr"]
                result[strategy_hash][risk_per_trade_pct]["long"] = long_result
                result[strategy_hash][risk_per_trade_pct]["forward"] = forward_result
                results.append(result)
                if long_result["performance"]["cagr"] < last_cagr:
                    break
                last_cagr = long_result["performance"]["cagr"]
        output_path = os.path.join(exp_dir, "trade_risk_test", "loaded_reports.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for report in results:
                f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
        logger.info(f"✅ Completed in {time.time() - begin_time:.2f}s , saved to {output_path}")
        _assert_experiment_revision(
            experiment_context,
            check_git_clean=args.check_git_clean,
        )
        exit(0)
    else:
        exp_dir = common.create_experiment_dir(
            os.path.join(common.PERSISTENCE_DIR, "batch_experiments"),
            SYMBOL,
            INTERVAL,
        )

    logger = _setup_root_logger(exp_dir)
    atexit.register(_cleanup_ephemeral_outputs, exp_dir, logger)
    logger.info("Experiment Git commit: %s", experiment_context.git_commit)
    logger.info("Experiment source snapshot: %s", os.environ.get(SNAPSHOT_PATH_ENV))
    if args.valid:
        logger.info(
            "Validation selected-config snapshot: %s | sha256=%s",
            selected_configs,
            common.sha256_file(selected_configs),
        )
    if not args.check_git_clean:
        logger.warning("The startup Git clean-worktree check is disabled")

    begin_time = time.time()
    reports_path = os.path.join(exp_dir, REPORTS_FILE)
    train_reports_path = os.path.join(exp_dir, TRAIN_REPORTS_FILE)

    combo_simulation_task = None

    # ---------------- build/load spec ----------------
    if args.resume:
        done_set = load_done_set(reports_path)
        task_spec, (n_prep_total, n_train_total, n_sim_total) = load_pending_tasks(exp_dir, done_set)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        total_all = n_prep_total + n_train_total + n_sim_total
        total_pending = n_prep + n_train + n_sim
        logger.info(f"📥 Loaded from {exp_dir}")
        logger.info(f"📊 Total: {total_all} (prep={n_prep_total}, train={n_train_total}, sim={n_sim_total})")
        logger.info(f"📊 Pending: {total_pending} (prep={n_prep}, train={n_train}, sim={n_sim}), done: {total_all - total_pending}")
    elif args.add:
        done_set = load_done_set(reports_path)
        task_spec, combo_simulation_task = create_task_spec(logger, exp_dir, done_set)
    elif args.valid:
        task_spec = _load_task_from_configs(selected_configs)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        logger.info(f"📥 Loaded from {selected_configs}")
        logger.info(f"📊 Pending: prep={n_prep}, train={n_train}, sim={n_sim}")
    else:
        task_spec, combo_simulation_task = create_task_spec(logger, exp_dir, None)
    if not task_spec:
        logger.info("✅ No pending tasks.")
        _cleanup_ephemeral_outputs(exp_dir, logger)
        _assert_experiment_revision(
            experiment_context,
            check_git_clean=args.check_git_clean,
        )
        return

    logger.info(f"🚀 Pipeline: MAX_PREP={MAX_PREP}, train={MAX_TRAIN}, MAX_SIM={MAX_SIM}")

    # ---------------- queues & workers ----------------
    prep_task_queue: mp.Queue = mp.Manager().Queue()
    train_task_queue: mp.Queue = mp.Manager().Queue()
    train_result_queue: mp.Queue = mp.Manager().Queue()
    sim_task_queue: mp.Queue = mp.Manager().Queue()
    sim_result_queue: mp.Queue = mp.Manager().Queue()

    # start workers
    prep_workers = []
    for i in range(MAX_PREP):
        worker_log = os.path.join(exp_dir, f"prep_{i}.log")
        p = mp.Process(target=_worker_prep, args=(worker_log, prep_task_queue, train_task_queue, exp_dir))
        p.start()
        prep_workers.append(p)

    is_combo = TRAIN_MODE in train_config.COMBO_SUB_TASKS

    # In combo_model mode the sub-models of the training stage attach no sim_tasks, so the sim_task_queue/
    # sim_workers here are unused during training -- and run_task_spec pushes None into sim_task_queue as soon
    # as training finishes, shutting the sim workers down (which is correct for the single model mode,
    # where the sims were queued during training; in combo mode there is not a single backtest task at that
    # point). So combo mode does not start the sim workers early: a fresh batch is started once every sub-model
    # finished training and the fuse stage really has backtests to run (see the
    # run_combo_fusion_and_backtest call below), so they cannot be shut down prematurely.
    sim_workers = []
    if not is_combo:
        for i in range(MAX_SIM):
            worker_log = os.path.join(exp_dir, f"sim_{i}.log")
            p = mp.Process(
                target=_worker_sim,
                args=(
                    worker_log,
                    sim_task_queue,
                    sim_result_queue,
                    reports_path,
                    exp_dir,
                    experiment_context,
                ),
            )
            p.start()
            sim_workers.append(p)
    run_task_spec(
        task_spec,
        exp_dir,
        prep_task_queue,
        train_task_queue,
        train_result_queue,
        sim_task_queue,
        sim_result_queue,
        logger,
        prep_workers,
        sim_workers,
        valid=args.valid,
        train_reports_path=train_reports_path,
    )

    _send_none_to_workers(prep_task_queue, MAX_PREP)
    if not is_combo:
        _send_none_to_workers(sim_task_queue, MAX_SIM)

    if args.valid:
        compare_old_new_reports(selected_configs, reports_path, exp_dir, logger)
        original_reports_path = run_original_model_backtests(
            task_spec=task_spec,
            exp_dir=exp_dir,
            selected_configs_path=selected_configs_source,
            experiment_context=experiment_context,
            logger=logger,
        )
        compare_old_new_reports(
            selected_configs,
            original_reports_path,
            os.path.dirname(original_reports_path),
            logger,
        )

    # combo_model mode: once every sub-model finished training, group by (pre_key, train_compatibility),
    # fuse the pairs into combined models and backtest them; the results also go to reports_path.
    # A brand new queue + sim worker set is used (instead of reusing the training stage one), to avoid racing
    # with the None shutdown signal run_task_spec sends as soon as training completes.
    if is_combo:
        combo_sim_task_queue: mp.Queue = mp.Manager().Queue()
        combo_sim_result_queue: mp.Queue = mp.Manager().Queue()
        combo_sim_workers = []
        for i in range(MAX_SIM):
            worker_log = os.path.join(exp_dir, f"combo_sim_{i}.log")
            p = mp.Process(
                target=_worker_sim,
                args=(
                    worker_log,
                    combo_sim_task_queue,
                    combo_sim_result_queue,
                    reports_path,
                    exp_dir,
                    experiment_context,
                ),
            )
            p.start()
            combo_sim_workers.append(p)

        try:
            run_combo_fusion_and_backtest(
                logger=logger,
                train_reports_path=train_reports_path,
                exp_dir=exp_dir,
                simulation_task=combo_simulation_task or [],
                reports_path=reports_path,
                sim_task_queue=combo_sim_task_queue,
                sim_result_queue=combo_sim_result_queue,
                sim_workers=combo_sim_workers,
                valid=args.valid,
            )
        except BaseException:
            for p in combo_sim_workers:
                if p.is_alive():
                    p.terminate()
            raise
        finally:
            _send_none_to_workers(combo_sim_task_queue, MAX_SIM)
            for p in combo_sim_workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

    _cleanup_ephemeral_outputs(exp_dir, logger)

    logger.info("\n" + "=" * 40)
    elapsed = time.time() - begin_time

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    _assert_experiment_revision(
        experiment_context,
        check_git_clean=args.check_git_clean,
    )
    logger.info(f"✅ Completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    logger.info("=" * 40)


def run_task_spec(
    task_spec,
    exp_dir,
    prep_task_queue,
    train_task_queue,
    train_result_queue,
    sim_task_queue,
    sim_result_queue,
    logger,
    prep_workers,
    sim_workers,
    valid=False,
    train_reports_path=None,
):
    stats = {"preparation": {"time": 0.0, "count": 0}, "train": {"time": 0.0, "count": 0}, "simulation": {"time": 0.0, "count": 0}}
    # key: (pre_h, tr_h), value: set of sim_h
    pending_sim_hashes = {}
    for pre_h, pre_node in task_spec.items():
        for tr_h, tr_node in pre_node["train"].items():
            sim_ids = {sim["hash"] for sim in tr_node.get("sim_tasks", [])}
            if sim_ids:
                pending_sim_hashes[(pre_h, tr_h)] = sim_ids
    _create_output_dirs(task_spec, exp_dir)
    prep_task_queue.put(task_spec)

    # ETA printer
    n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
    eta_msg = _make_eta_fn(n_prep, n_train, n_sim, stats)

    # If no train stage exists, we should stop sim workers after we enqueue all sims.
    sim_nones_sent = n_train == 0

    pending_train_items: List[Dict[str, Any]] = []
    train_procs: List[mp.Process] = []
    train_idx = 0

    def _reap_train_procs():
        nonlocal train_procs
        alive = []
        for p in train_procs:
            if p.is_alive():
                alive.append(p)
            else:
                p.join(timeout=0)
                if p.exitcode:
                    raise ExperimentTaskError(f"Training process {p.name} exited with code {p.exitcode}")
        train_procs = alive

    def _check_worker_processes():
        for stage, processes in (("prep", prep_workers), ("simulation", sim_workers)):
            for process in processes:
                if process.exitcode not in (None, 0):
                    raise ExperimentTaskError(f"{stage} process {process.name} exited with code {process.exitcode}")

    def _drain_train_results():
        nonlocal sim_nones_sent
        while True:
            try:
                msg = train_result_queue.get_nowait()
            except Empty:
                break
            if not msg:
                continue
            typ, pre_h, tr_h, elapsed, train_report, *details = msg
            if typ == "train_failed":
                error = details[0] if details else "no traceback returned"
                raise ExperimentTaskError(f"Training failed after {elapsed:.2f}s for {pre_h}/{tr_h}:\n{error}")
            if typ != "train_done":
                raise ExperimentTaskError(f"Unexpected training result: {msg!r}")

            stats["train"]["time"] += float(elapsed)
            stats["train"]["count"] += 1
            logger.info(f"  {pre_h}/{tr_h}  Train done in {elapsed:.2f}s")
            if train_reports_path and train_report is not None:
                common.append_jsonl(train_reports_path, train_report)

            # safe to stop sim workers only after ALL train_done received
            if stats["train"]["count"] >= n_train and not sim_nones_sent:
                _send_none_to_workers(sim_task_queue, MAX_SIM)
                sim_nones_sent = True

    def _try_start_train_procs():
        nonlocal train_idx
        _reap_train_procs()
        while pending_train_items and len(train_procs) < MAX_TRAIN:
            item = pending_train_items.pop(0)
            worker_log = os.path.join(exp_dir, f"train_{train_idx%MAX_TRAIN}.log")
            p = mp.Process(target=_train_task, args=(worker_log, item, sim_task_queue, train_result_queue))
            p.start()
            train_procs.append(p)
            train_idx += 1

    # main loop: consume prep_done -> spawn train processes -> enqueue sims; also drain sim results
    try:
        while stats["preparation"]["count"] < n_prep or stats["train"]["count"] < n_train or stats["simulation"]["count"] < n_sim:
            _check_worker_processes()
            # consume prep->train messages
            try:
                msg = train_task_queue.get(timeout=0.2)
            except Empty:
                msg = None

            if msg is not None:
                typ, pre_h, elapsed, train_items = msg
                if typ == "prep_failed":
                    raise ExperimentTaskError(f"Preparation failed after {elapsed:.2f}s for {pre_h}:\n{train_items}")
                if typ != "prep_done":
                    raise ExperimentTaskError(f"Unexpected preparation result: {msg!r}")

                stats["preparation"]["time"] += float(elapsed)
                stats["preparation"]["count"] += 1
                logger.info(f"  {pre_h}  Prep done in {elapsed:.2f}s")

                pending_train_items.extend(train_items or [])

            _try_start_train_procs()
            _drain_train_results()
            _drain_sim_results(
                sim_result_queue,
                stats,
                logger,
                eta_msg,
                pending_sim_hashes,
                valid,
            )

        # final drains
        _drain_train_results()
        _drain_sim_results(
            sim_result_queue,
            stats,
            logger,
            eta_msg,
            pending_sim_hashes,
            valid,
        )
    finally:
        # Stop the entire process tree immediately on either success or failure.
        for p in train_procs + prep_workers + sim_workers:
            if p.is_alive():
                p.terminate()
        for p in train_procs + prep_workers + sim_workers:
            p.join(timeout=5)


def run_original_model_backtests(
    task_spec: Dict[str, Any],
    exp_dir: str,
    selected_configs_path: str,
    experiment_context: ExperimentContext,
    logger: logging.Logger,
) -> str:
    """Reproduce backtests from archived original models without retraining."""
    reproduction_dir = os.path.join(exp_dir, BACKTEST_REPRODUCTION_DIR)
    reports_path = os.path.join(reproduction_dir, REPORTS_FILE)
    prediction_cache_root = os.path.join(
        reproduction_dir,
        ORIGINAL_PREDICTION_CACHE_DIR,
    )
    os.makedirs(reproduction_dir, exist_ok=True)

    model_items: List[Dict[str, Any]] = []
    pending_sim_hashes: Dict[Tuple[str, str], Set[str]] = {}
    for pre_h, pre_node in task_spec.items():
        for tr_h, tr_node in pre_node["train"].items():
            sim_tasks = copy.deepcopy(tr_node.get("sim_tasks", []))
            model_items.append(
                {
                    "pre_h": pre_h,
                    "tr_h": tr_h,
                    "pre_params": copy.deepcopy(pre_node["params"]),
                    "train_params": copy.deepcopy(tr_node["params"]),
                    "sim_tasks": sim_tasks,
                    "prep_output_dir": _prep_output_dir(exp_dir, pre_h),
                    "train_output_dir": _selected_model_artifact_dir(
                        selected_configs_path,
                        pre_h,
                        tr_h,
                    ),
                    "prediction_cache_dir": os.path.join(
                        prediction_cache_root,
                        f"{pre_h}_{tr_h}",
                    ),
                }
            )
            pending_sim_hashes[(pre_h, tr_h)] = {sim["hash"] for sim in sim_tasks}

    total_models = len(model_items)
    total_sims = sum(len(item["sim_tasks"]) for item in model_items)
    if total_models == 0 or total_sims == 0:
        raise RuntimeError("Backtest reproduction has no original model tasks")

    manager = mp.Manager()
    model_result_queue = manager.Queue()
    sim_task_queue = manager.Queue()
    sim_result_queue = manager.Queue()
    sim_workers: List[mp.Process] = []
    model_processes: List[mp.Process] = []

    for index in range(MAX_SIM):
        process = mp.Process(
            target=_worker_sim,
            name=f"OriginalBacktestSim-{index}",
            args=(
                os.path.join(reproduction_dir, f"sim_{index}.log"),
                sim_task_queue,
                sim_result_queue,
                reports_path,
                reproduction_dir,
                experiment_context,
                exp_dir,
            ),
        )
        process.start()
        sim_workers.append(process)

    pending_models = list(model_items)
    model_done = 0
    stats = {"simulation": {"time": 0.0, "count": 0}}
    process_index = 0
    sim_stop_sent = False

    def no_eta() -> str:
        return ""

    try:
        while model_done < total_models or stats["simulation"]["count"] < total_sims:
            active_processes: List[mp.Process] = []
            for process in model_processes:
                if process.is_alive():
                    active_processes.append(process)
                    continue
                process.join(timeout=0)
                if process.exitcode:
                    raise ExperimentTaskError(f"Original model process {process.name} exited " f"with code {process.exitcode}")
            model_processes = active_processes

            while pending_models and len(model_processes) < MAX_TRAIN:
                item = pending_models.pop(0)
                process = mp.Process(
                    target=_original_model_task,
                    name=f"OriginalModel-{process_index}",
                    args=(
                        os.path.join(
                            reproduction_dir,
                            f"model_{process_index % MAX_TRAIN}.log",
                        ),
                        item,
                        sim_task_queue,
                        model_result_queue,
                    ),
                )
                process.start()
                model_processes.append(process)
                process_index += 1

            while True:
                try:
                    message = model_result_queue.get_nowait()
                except Empty:
                    break
                kind, pre_h, tr_h, elapsed, detail = message
                if kind == "original_model_failed":
                    raise ExperimentTaskError("Original model preparation failed after " f"{elapsed:.2f}s for {pre_h}/{tr_h}:\n{detail}")
                if kind != "original_model_ready":
                    raise ExperimentTaskError(f"Unexpected original model result: {message!r}")
                model_done += 1
                logger.info(
                    "Original model %d/%d ready without training: %s/%s in %.2fs",
                    model_done,
                    total_models,
                    pre_h,
                    tr_h,
                    elapsed,
                )

            if model_done >= total_models and not sim_stop_sent:
                _send_none_to_workers(sim_task_queue, MAX_SIM)
                sim_stop_sent = True

            for process in sim_workers:
                if process.exitcode not in (None, 0):
                    raise ExperimentTaskError(f"Original backtest process {process.name} exited " f"with code {process.exitcode}")

            _drain_sim_results(
                sim_result_queue,
                stats,
                logger,
                no_eta,
                pending_sim_hashes,
                False,
            )
            if model_done < total_models or stats["simulation"]["count"] < total_sims:
                time.sleep(0.2)

        logger.info(
            "Original-model backtest reproduction completed: models=%d, simulations=%d",
            total_models,
            total_sims,
        )
        return reports_path
    finally:
        if not sim_stop_sent:
            _send_none_to_workers(sim_task_queue, MAX_SIM)
        for process in model_processes + sim_workers:
            if process.is_alive():
                process.terminate()
        for process in model_processes + sim_workers:
            process.join(timeout=5)
        if os.path.isdir(prediction_cache_root):
            shutil.rmtree(prediction_cache_root)


def _load_task_from_configs(path: str) -> Dict[str, Any]:
    """
    Build a task spec tree from selected_configs.jsonl.
    Each record is a full report with {"params": {"strategy":.., "common":.., "train":.., "hash":..}}
    """
    task_spec: Dict[str, Any] = {}
    records = load_selected_configs(path)
    for r in records:
        params = common.recursive_get(r, "params")

        r_hash = params["hash"]
        pre_conf = common.recursive_get(params, "common")
        tr_conf = common.recursive_get(params, "train")
        strategy_conf = common.recursive_get(params, "strategy")
        broker_conf = common.recursive_get(params, "broker")
        if strategy_conf and "strategy_config" in strategy_conf:
            sim_conf = strategy_conf
        elif strategy_conf and broker_conf:
            sim_conf = {
                "strategy_config": strategy_conf,
                "broker_config": broker_conf,
            }
        else:
            sim_conf = None

        if not pre_conf or not tr_conf or not sim_conf:
            continue

        if "prep_output_dir" in pre_conf or "save_dir" in tr_conf:
            raise ValueError("Unexpected prep_output_dir/save_dir in config params")

        identity = TaskIdentity.from_params(
            prep=pre_conf,
            train=tr_conf,
            sim=sim_conf,
        )
        pre_h = identity.prep_hash
        tr_h = identity.train_hash
        sim_h = identity.sim_hash

        if pre_h in task_spec:
            print(f"⚠️  Warning: duplicate prep config hash {pre_h} in {path}")
        node_pre = task_spec.setdefault(pre_h, {"params": json_safe(pre_conf), "train": {}})
        node_tr = node_pre["train"].setdefault(tr_h, {"params": json_safe(tr_conf), "sim_tasks": []})

        existing = {s["hash"] for s in node_tr["sim_tasks"]}
        if sim_h not in existing:
            node_tr["sim_tasks"].append(
                {
                    "hash": sim_h,
                    "identity": identity.as_dict(),
                    "params": json_safe(sim_conf),
                    "strategy_hash": r_hash,
                }
            )

    return task_spec


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
