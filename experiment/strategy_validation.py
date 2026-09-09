#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from queue import Empty
from typing import Any, Dict, List, Tuple

current_work_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_work_dir, ".."))

from data_process import common, preparation
from data_process.utils import TaskIdentity, config_from_dict_train, load_selected_configs
from model import train_config
from trade.runner.config import ExperimentContext

SELECTED_FILE = "selected_configs.jsonl"
COMPARE_REPORTS_FILE = "compare_reports.jsonl"
MODEL_ARCHIVE_DIR = "models"
MODEL_ARCHIVE_MANIFEST_FILE = "model_archive_manifest.jsonl"
BACKTEST_COMPARE_REPORTS_FILE = "backtest_compare_reports.jsonl"
BACKTEST_REPORTS_FILE = "backtest_reproduction_reports.jsonl"
REPORT_DETAILS_FILE = "report_details.json"
BACKTEST_REPRODUCTION_ARCHIVE_DIR = "backtest_reproduction"
EQUITY_PLOT_DIR = "equity"
EQUITY_BATCH_SIZE = 3

CROSS_TEST_SYMBOLS = ("DOGEUSDT", "ETHUSDT", "BTCUSDT")
CROSS_TEST_INTERVALS = ("15m", "30m", "1h")

try:
    experiment_tasks = importlib.import_module("experiment.task_constructors")
except ModuleNotFoundError as exc:
    if exc.name != "experiment.task_constructors":
        raise
    experiment_tasks = importlib.import_module("experiment.task_constructors_example")

MAX_PREP = experiment_tasks.MAX_PREP
MAX_TRAIN = experiment_tasks.MAX_TRAIN
MAX_SIM = 4
INFERENCE_BATCH_SIZE = experiment_tasks.INFERENCE_BATCH_SIZE
ENABLE_RETRAINED_TEST = False

MODEL_PERIOD_BY_MODE = {
    "original_model": "long",
    "retrained_model": "forward",
}
ENABLED_MODEL_MODES = (
    "original_model",
    *(("retrained_model",) if ENABLE_RETRAINED_TEST else ()),
)


class CrossTestError(RuntimeError):
    pass


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    formatter = logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s")

    file_handler = logging.FileHandler(
        os.path.join(output_dir, "cross_test.log"),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger = logging.getLogger("cross_test")
    logger.setLevel(logging.INFO)
    return logger


def worker_logger(log_file: str) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s"))
    root.addHandler(handler)
    return root


def cross_test_targets(
    original_symbol: str,
    original_interval: str,
) -> List[Tuple[str, str]]:
    targets = [(symbol, original_interval) for symbol in CROSS_TEST_SYMBOLS if symbol != original_symbol]
    targets.extend((original_symbol, interval) for interval in CROSS_TEST_INTERVALS if interval != original_interval)
    return list(dict.fromkeys(targets))


def target_key(symbol: str, interval: str) -> str:
    return f"{symbol}_{interval}"


def training_validation_root() -> str:
    return os.path.join(
        common.PERSISTENCE_DIR,
        "batch_experiments",
        "valid_train_out",
    )


def original_model_dir(
    selected_configs_path: str,
    params: Dict[str, Any],
) -> str:
    """Resolve the model archived from the original experiment at selection time."""
    identity = params.get("identity") or {}
    prep_hash = str(identity.get("prep_hash", ""))
    train_hash = str(identity.get("train_hash", ""))
    strategy_hash = params.get("hash", "unknown")

    for name, value in (("preparation", prep_hash), ("training", train_hash)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError(
                f"Selected strategy has an unsafe {name} artifact hash: "
                f"strategy_hash={strategy_hash}, value={value!r}"
            )

    archive_root = os.path.abspath(
        os.path.join(os.path.dirname(selected_configs_path), "train")
    )
    model_dir = os.path.abspath(
        os.path.join(
            archive_root,
            f"pre_{prep_hash}",
            f"train_{train_hash}",
        )
    )
    if os.path.commonpath([archive_root, model_dir]) != archive_root:
        raise ValueError(
            f"Original model path escaped its archive root: {model_dir}"
        )

    required_files = ("model.pt", "meta.json", "train_config.json")
    missing_files = [
        filename
        for filename in required_files
        if not os.path.isfile(os.path.join(model_dir, filename))
    ]
    if missing_files:
        raise FileNotFoundError(
            "Original selected model artifact is incomplete: "
            f"strategy_hash={strategy_hash}, path={model_dir}, "
            f"missing={missing_files}"
        )
    return model_dir


def compare_reports_source(selected_configs_path: str) -> str:
    candidates = [
        os.path.join(
            os.path.dirname(os.path.abspath(selected_configs_path)),
            COMPARE_REPORTS_FILE,
        ),
        os.path.join(training_validation_root(), COMPARE_REPORTS_FILE),
    ]
    for candidate in dict.fromkeys(candidates):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Comparison report for selected configs was not found: "
        f"candidates={candidates}"
    )


def backtest_reproduction_source(filename: str) -> str:
    path = os.path.join(
        training_validation_root(),
        "backtest_reproduction",
        filename,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Backtest reproduction file was not found: {path}"
        )
    return path


def report_details_path(
    report: Dict[str, Any],
    reports_path: str,
) -> str:
    """Return the canonical sidecar path for one report record."""
    identity = (report.get("params") or {}).get("identity") or {}
    full_hash = str(identity.get("full_hash", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", full_hash):
        raise ValueError(
            "Report has an unsafe full simulation hash: "
            f"value={full_hash!r}"
        )
    return os.path.join(
        os.path.dirname(os.path.abspath(reports_path)),
        "sim_output",
        full_hash,
        REPORT_DETAILS_FILE,
    )


def copy_report_details(
    records: List[Dict[str, Any]],
    source_reports_path: str,
    destination_reports_path: str,
) -> int:
    """Copy report sidecars while preserving their canonical relative paths."""
    copied_paths = set()
    for report in records:
        source = report_details_path(report, source_reports_path)
        destination = report_details_path(report, destination_reports_path)
        if destination in copied_paths:
            continue
        if not os.path.isfile(source):
            raise FileNotFoundError(
                "Report details file was not found: "
                f"report={source_reports_path}, details={source}"
            )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.abspath(source) != os.path.abspath(destination):
            shutil.copy2(source, destination)
        copied_paths.add(destination)
    return len(copied_paths)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record must be an object: path={path}, "
                    f"line={line_number}"
                )
            records.append(record)
    return records


def archive_cross_test_inputs(
    selected_configs_path: str,
    records: List[Dict[str, Any]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Copy selection inputs and the exact original model artifacts into a run."""
    selected_configs_path = os.path.abspath(selected_configs_path)
    output_dir = os.path.abspath(output_dir)
    compare_reports_path = os.path.abspath(
        compare_reports_source(selected_configs_path)
    )
    backtest_compare_reports_path = os.path.abspath(
        backtest_reproduction_source(COMPARE_REPORTS_FILE)
    )
    backtest_reports_path = os.path.abspath(
        backtest_reproduction_source("reports.jsonl")
    )
    archived_selected_path = os.path.join(output_dir, SELECTED_FILE)
    backtest_archive_dir = os.path.join(
        output_dir,
        BACKTEST_REPRODUCTION_ARCHIVE_DIR,
    )
    archived_backtest_reports_path = os.path.join(
        backtest_archive_dir,
        "reports.jsonl",
    )
    archived_backtest_compare_reports_path = os.path.join(
        backtest_archive_dir,
        COMPARE_REPORTS_FILE,
    )

    file_sources = (
        (selected_configs_path, archived_selected_path),
        (
            compare_reports_path,
            os.path.join(output_dir, COMPARE_REPORTS_FILE),
        ),
        (
            backtest_compare_reports_path,
            os.path.join(output_dir, BACKTEST_COMPARE_REPORTS_FILE),
        ),
        (
            backtest_reports_path,
            os.path.join(output_dir, BACKTEST_REPORTS_FILE),
        ),
        (
            backtest_compare_reports_path,
            archived_backtest_compare_reports_path,
        ),
        (
            backtest_reports_path,
            archived_backtest_reports_path,
        ),
    )
    for source, destination in file_sources:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if source != os.path.abspath(destination):
            shutil.copy2(source, destination)

    selected_details_count = copy_report_details(
        records,
        selected_configs_path,
        archived_selected_path,
    )
    backtest_records = load_jsonl(backtest_reports_path)
    backtest_details_count = copy_report_details(
        backtest_records,
        backtest_reports_path,
        archived_backtest_reports_path,
    )

    copied_hashes = set()
    archive_manifest: List[Dict[str, Any]] = []
    for record in records:
        strategy_hash = str(validate_record(record)["hash"])
        if strategy_hash in copied_hashes:
            continue

        params = validate_record(record)
        source_dir = original_model_dir(selected_configs_path, params)
        destination_dir = os.path.join(
            output_dir,
            MODEL_ARCHIVE_DIR,
            strategy_hash,
        )
        shutil.copytree(
            source_dir,
            destination_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("prediction_cache"),
        )
        source_model_path = os.path.join(source_dir, "model.pt")
        destination_model_path = os.path.join(destination_dir, "model.pt")
        source_model_sha256 = common.sha256_file(source_model_path)
        destination_model_sha256 = common.sha256_file(destination_model_path)
        if destination_model_sha256 != source_model_sha256:
            raise RuntimeError(
                "Archived model checksum mismatch: "
                f"strategy_hash={strategy_hash}, source={source_model_sha256}, "
                f"destination={destination_model_sha256}"
            )
        archive_manifest.append(
            {
                "strategy_hash": strategy_hash,
                "identity": params["identity"],
                "selected_archive_source": source_dir,
                "original_experiment_source": (params.get("data") or {}).get(
                    "train_output_dir"
                ),
                "destination": destination_dir,
                "model_sha256": source_model_sha256,
            }
        )
        copied_hashes.add(strategy_hash)

    write_jsonl(
        os.path.join(output_dir, MODEL_ARCHIVE_MANIFEST_FILE),
        archive_manifest,
    )

    logger.info(
        "Archived cross-test inputs | selected_configs=%s | "
        "compare_reports=%s | backtest_compare_reports=%s | "
        "backtest_reports=%s | selected_report_details=%d | "
        "backtest_report_details=%d | models=%d | destination=%s",
        selected_configs_path,
        compare_reports_path,
        backtest_compare_reports_path,
        backtest_reports_path,
        selected_details_count,
        backtest_details_count,
        len(copied_hashes),
        output_dir,
    )


def prep_output_dir(
    output_dir: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
) -> str:
    return os.path.join(
        output_dir,
        "prep",
        strategy_hash,
        target_key(symbol, interval),
    )


def retrain_output_dir(
    output_dir: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
) -> str:
    return os.path.join(
        output_dir,
        "retrained_models",
        strategy_hash,
        target_key(symbol, interval),
    )


def backtest_output_dir(
    output_dir: str,
    mode: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
) -> str:
    return os.path.join(
        output_dir,
        "backtests",
        mode,
        strategy_hash,
        target_key(symbol, interval),
    )


def cross_test_report_details_path(
    output_dir: str,
    mode: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
) -> str:
    return os.path.join(
        backtest_output_dir(
            output_dir,
            mode,
            strategy_hash,
            symbol,
            interval,
        ),
        REPORT_DETAILS_FILE,
    )


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output:
            json.dump(
                payload,
                output,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    params = common.recursive_get(record, "params")
    if not params:
        raise ValueError("Selected config record has no params")

    required = ("hash", "common", "train", "strategy", "broker")
    missing = [field for field in required if common.recursive_get(params, field) is None]
    if missing:
        raise ValueError(f"Selected config record is missing required params: {missing}")
    return params


def validate_preparation(
    pre_para: common.BaseDefine,
    output_dir: str,
) -> None:
    expected_hash = TaskIdentity.prep_hash_for(asdict(pre_para))
    manifest = common.load_data_manifest_from_dir(output_dir)

    actual_hash = manifest.get("configuration_hash")
    if actual_hash != expected_hash:
        raise RuntimeError("Preparation configuration mismatch: " f"path={output_dir}, expected={expected_hash}, actual={actual_hash}")

    stored_pre_para = common.load_pre_params_from_dir(output_dir)
    stored_hash = TaskIdentity.prep_hash_for(asdict(stored_pre_para))
    if stored_hash != expected_hash:
        raise RuntimeError("Preparation metadata mismatch: " f"path={output_dir}, expected={expected_hash}, actual={stored_hash}")

    source_path = common.market_data_path(pre_para)
    expected_source = {
        "filename": os.path.basename(source_path),
        "size_bytes": os.path.getsize(source_path),
        "sha256": common.sha256_file(source_path),
    }
    actual_source = manifest.get("source") or {}

    mismatches = {key: (expected_value, actual_source.get(key)) for key, expected_value in expected_source.items() if actual_source.get(key) != expected_value}
    if mismatches:
        raise RuntimeError("Preparation source mismatch: " f"path={output_dir}, mismatches={mismatches}")

    required_paths = (
        common.get_train_data_path_in_dir(output_dir),
        common.get_test_data_path_in_dir(output_dir),
    )
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise RuntimeError(f"Preparation is incomplete: missing={missing_paths}")


def build_jobs(
    records: List[Dict[str, Any]],
    output_dir: str,
    selected_configs_path: str,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    for record in records:
        params = validate_record(record)
        strategy_hash = params["hash"]
        original_pre_para = common.BaseDefine(**params["common"])

        for symbol, interval in cross_test_targets(
            original_pre_para.symbol,
            original_pre_para.interval,
        ):
            target_pre_para = copy.deepcopy(original_pre_para)
            target_pre_para.symbol = symbol
            target_pre_para.interval = interval

            jobs.append(
                {
                    "strategy_hash": strategy_hash,
                    "params": copy.deepcopy(params),
                    "target_symbol": symbol,
                    "target_interval": interval,
                    "target_pre_params": asdict(target_pre_para),
                    "original_model_dir": original_model_dir(
                        selected_configs_path,
                        params,
                    ),
                    "prep_dir": prep_output_dir(
                        output_dir,
                        strategy_hash,
                        symbol,
                        interval,
                    ),
                    "retrain_dir": retrain_output_dir(
                        output_dir,
                        strategy_hash,
                        symbol,
                        interval,
                    ),
                }
            )

    return jobs


def prep_worker(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    logger = worker_logger(worker_log_file)

    while True:
        try:
            job = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if job is None:
            return

        t0 = time.time()
        try:
            pre_para = common.BaseDefine(**job["target_pre_params"])
            prep_dir = job["prep_dir"]

            manifest_path = common.get_data_manifest_path_in_dir(prep_dir)
            if not os.path.isfile(manifest_path):
                preparation.main(
                    logger,
                    para=pre_para,
                    prep_output_dir=prep_dir,
                )

            validate_preparation(pre_para, prep_dir)

            result_queue.put(
                (
                    "prep_done",
                    job,
                    time.time() - t0,
                )
            )
        except Exception:
            result_queue.put(
                (
                    "prep_failed",
                    job,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return


def sim_worker(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: str,
    experiment_context: ExperimentContext,
) -> None:
    logger = worker_logger(worker_log_file)

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logger.info(
        "Simulation worker CPU threads limited to one " "(intra_op=%d, inter_op=%d)",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )

    from trade.runner import backtest_runner

    while True:
        try:
            payload = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if payload is None:
            return

        mode = payload["mode"]
        job = payload["job"]
        train_metrics = payload.get("train_metrics")
        t0 = time.time()

        try:
            params = job["params"]
            strategy_hash = job["strategy_hash"]
            symbol = job["target_symbol"]
            interval = job["target_interval"]

            train_cfg = config_from_dict_train(params["train"])
            strategy_config = backtest_runner.strategy_config_from_dict(params["strategy"])
            broker_config = backtest_runner.BrokerConfig(**params["broker"])

            if mode == "original_model":
                train_output_dir = job["original_model_dir"]
            elif mode == "retrained_model":
                train_output_dir = job["retrain_dir"]
            else:
                raise ValueError(f"Unknown simulation mode: {mode}")

            period = MODEL_PERIOD_BY_MODE[mode]

            data_config = backtest_runner.ModelDataConfig(
                prep_output_dir=job["prep_dir"],
                train_output_dir=train_output_dir,
                device="auto",
                use_prediction_cache=True,
            )

            backtest_runner.precompute_prediction_cache(
                logger,
                data_config,
                train_cfg,
                period,
                inference_batch_size=INFERENCE_BATCH_SIZE,
            )

            runner_config = backtest_runner.RunnerConfig(
                strategy_config=strategy_config,
                broker_config=broker_config,
                save_dir=backtest_output_dir(
                    output_dir,
                    mode,
                    strategy_hash,
                    symbol,
                    interval,
                ),
                data_config=data_config,
                experiment_context=experiment_context,
            )

            backtest_result = backtest_runner.main(
                logger,
                runner_config,
                period,
            )
            report = backtest_result["report"]
            report_details = backtest_result["report_details"]

            validate_report_market(
                report,
                symbol,
                interval,
            )
            validate_report_period(report, period)
            validate_report_period(report_details, period)

            details_path = cross_test_report_details_path(
                output_dir,
                mode,
                strategy_hash,
                symbol,
                interval,
            )
            write_json(details_path, report_details)

            result_queue.put(
                (
                    "sim_done",
                    mode,
                    job,
                    time.time() - t0,
                    train_metrics,
                    report,
                    details_path,
                )
            )
        except Exception:
            result_queue.put(
                (
                    "sim_failed",
                    mode,
                    job,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return


def retrain_worker(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    logger = worker_logger(worker_log_file)

    import model.train as train

    while True:
        try:
            job = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if job is None:
            return

        t0 = time.time()
        try:
            train_cfg = config_from_dict_train(job["params"]["train"])
            save_dir = job["retrain_dir"]
            os.makedirs(save_dir, exist_ok=True)

            train_metrics = train.train(
                logger=logger,
                config=train_cfg,
                prep_output_dir=job["prep_dir"],
                save_dir=save_dir,
            )

            result_queue.put(
                (
                    "retrain_done",
                    job,
                    time.time() - t0,
                    train_metrics,
                )
            )
        except Exception:
            result_queue.put(
                (
                    "retrain_failed",
                    job,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return


def validate_report_market(
    report: Dict[str, Any],
    expected_symbol: str,
    expected_interval: str,
) -> None:
    report_common = report["params"]["common"]

    if report_common.get("symbol") != expected_symbol or report_common.get("interval") != expected_interval:
        raise RuntimeError("Cross-test report market mismatch: " f"expected={expected_symbol}_{expected_interval}, " f"report_common={report_common}")


def validate_report_period(
    report: Dict[str, Any],
    expected_period: str,
) -> None:
    report_periods = set((report.get("results") or {}).keys())
    if report_periods != {expected_period}:
        raise RuntimeError("Cross-test report period mismatch: " f"expected={[expected_period]}, actual={sorted(report_periods)}")


def send_none(queue: mp.Queue, count: int) -> None:
    for _ in range(count):
        queue.put(None)


def ensure_workers_alive(
    stage_name: str,
    workers: List[mp.Process],
) -> None:
    for process in workers:
        if process.exitcode not in (None, 0):
            raise CrossTestError(f"{stage_name} worker {process.name} exited " f"with code {process.exitcode}")


def terminate_workers(workers: List[mp.Process]) -> None:
    for process in workers:
        if process.is_alive():
            process.terminate()

    for process in workers:
        process.join(timeout=5)


def job_id(job: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        job["strategy_hash"],
        job["target_symbol"],
        job["target_interval"],
    )


def run_parallel_pipeline(
    logger: logging.Logger,
    jobs: List[Dict[str, Any]],
    output_dir: str,
    experiment_context: ExperimentContext,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    manager = mp.Manager()

    prep_queue = manager.Queue()
    prep_result_queue = manager.Queue()

    retrain_queue = manager.Queue()
    retrain_result_queue = manager.Queue()

    sim_queue = manager.Queue()
    sim_result_queue = manager.Queue()

    prep_workers: List[mp.Process] = []
    retrain_workers: List[mp.Process] = []
    sim_workers: List[mp.Process] = []

    for index in range(MAX_PREP):
        process = mp.Process(
            target=prep_worker,
            name=f"CrossPrep-{index}",
            args=(
                os.path.join(output_dir, f"prep_{index}.log"),
                prep_queue,
                prep_result_queue,
            ),
        )
        process.start()
        prep_workers.append(process)

    if ENABLE_RETRAINED_TEST:
        for index in range(MAX_TRAIN):
            process = mp.Process(
                target=retrain_worker,
                name=f"CrossRetrain-{index}",
                args=(
                    os.path.join(output_dir, f"retrain_{index}.log"),
                    retrain_queue,
                    retrain_result_queue,
                ),
            )
            process.start()
            retrain_workers.append(process)

    for index in range(MAX_SIM):
        process = mp.Process(
            target=sim_worker,
            name=f"CrossSim-{index}",
            args=(
                os.path.join(output_dir, f"sim_{index}.log"),
                sim_queue,
                sim_result_queue,
                output_dir,
                experiment_context,
            ),
        )
        process.start()
        sim_workers.append(process)

    all_workers = prep_workers + retrain_workers + sim_workers

    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    total_jobs = len(jobs)
    total_sim_tasks = total_jobs * len(ENABLED_MODEL_MODES)

    prep_done = 0
    retrain_done = 0
    sim_done = 0

    for job in jobs:
        results[job_id(job)] = {
            "job": job,
            **{model_mode: None for model_mode in ENABLED_MODEL_MODES},
        }
        prep_queue.put(job)

    try:
        while sim_done < total_sim_tasks:
            ensure_workers_alive("prep", prep_workers)
            ensure_workers_alive("retrain", retrain_workers)
            ensure_workers_alive("simulation", sim_workers)

            try:
                message = prep_result_queue.get(timeout=0.1)
            except Empty:
                message = None

            if message is not None:
                kind, job, elapsed, *payload = message

                if kind == "prep_failed":
                    raise CrossTestError("Preparation failed for " f"{job_id(job)} after {elapsed:.2f}s:\n" f"{payload[0]}")
                if kind != "prep_done":
                    raise CrossTestError(f"Unexpected prep result: {message!r}")

                prep_done += 1
                logger.info(
                    "Prep %d/%d done: %s in %.2fs",
                    prep_done,
                    total_jobs,
                    job_id(job),
                    elapsed,
                )

                sim_queue.put(
                    {
                        "mode": "original_model",
                        "job": job,
                    }
                )
                if ENABLE_RETRAINED_TEST:
                    retrain_queue.put(job)

            if ENABLE_RETRAINED_TEST:
                while True:
                    try:
                        message = retrain_result_queue.get_nowait()
                    except Empty:
                        break

                    kind, job, elapsed, *payload = message

                    if kind == "retrain_failed":
                        raise CrossTestError("Retraining failed for " f"{job_id(job)} after {elapsed:.2f}s:\n" f"{payload[0]}")
                    if kind != "retrain_done":
                        raise CrossTestError(f"Unexpected retrain result: {message!r}")

                    train_metrics = payload[0]
                    retrain_done += 1

                    logger.info(
                        "Retrain %d/%d done: %s in %.2fs",
                        retrain_done,
                        total_jobs,
                        job_id(job),
                        elapsed,
                    )

                    sim_queue.put(
                        {
                            "mode": "retrained_model",
                            "job": job,
                            "train_metrics": train_metrics,
                        }
                    )

            while True:
                try:
                    message = sim_result_queue.get_nowait()
                except Empty:
                    break

                kind, mode, job, elapsed, *payload = message

                if kind == "sim_failed":
                    raise CrossTestError(f"{mode} simulation failed for " f"{job_id(job)} after {elapsed:.2f}s:\n" f"{payload[0]}")
                if kind != "sim_done":
                    raise CrossTestError(f"Unexpected simulation result: {message!r}")

                train_metrics, report, details_path = payload
                period = MODEL_PERIOD_BY_MODE[mode]
                result_entry = {
                    "cagr": report["results"][period]["performance"]["cagr"],
                    "report": report,
                    "report_details_path": os.path.relpath(
                        details_path,
                        output_dir,
                    ),
                }
                if mode == "retrained_model":
                    result_entry["train_metrics"] = train_metrics

                results[job_id(job)][mode] = result_entry
                sim_done += 1

                logger.info(
                    "Simulation %d/%d done: mode=%s job=%s in %.2fs",
                    sim_done,
                    total_sim_tasks,
                    mode,
                    job_id(job),
                    elapsed,
                )

        return results

    finally:
        send_none(prep_queue, MAX_PREP)
        send_none(retrain_queue, len(retrain_workers))
        send_none(sim_queue, MAX_SIM)
        terminate_workers(all_workers)


def _test_type(
    original_symbol: str,
    original_interval: str,
    target_symbol: str,
    target_interval: str,
) -> str:
    if target_symbol != original_symbol and target_interval == original_interval:
        return "cross_asset"
    if target_symbol == original_symbol and target_interval != original_interval:
        return "cross_period"
    raise ValueError(
        "Cross-test target must change exactly one market dimension: "
        f"original={original_symbol}_{original_interval}, "
        f"target={target_symbol}_{target_interval}"
    )


def _summary_metrics_from_report(
    report: Dict[str, Any],
    period: str,
) -> Dict[str, Any]:
    results = report.get("results") or {}
    period_result = results.get(period) or {}

    performance = period_result.get("performance") or {}
    trades = period_result.get("trades") or {}

    return {
        "cagr": performance.get("cagr"),
        "avg_pct_gross": trades.get("avg_pct_gross"),
    }


def _aggregate_model_summary(
    grouped_targets: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    cagr_values: List[float] = []
    avg_pct_gross_values: List[float] = []

    for test_group in ("cross_period", "cross_asset"):
        for metrics in grouped_targets[test_group].values():
            cagr = metrics.get("cagr")
            avg_pct_gross = metrics.get("avg_pct_gross")

            if isinstance(cagr, (int, float)):
                cagr_values.append(float(cagr))

            if isinstance(avg_pct_gross, (int, float)):
                avg_pct_gross_values.append(float(avg_pct_gross))

    def mean(values: List[float]) -> Any:
        if not values:
            return None
        return sum(values) / len(values)

    def median(values: List[float]) -> Any:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    positive_ratio = None
    if cagr_values:
        positive_ratio = sum(1 for value in cagr_values if value > 0) / len(cagr_values)

    return {
        "overall": {
            "mean_cagr": mean(cagr_values),
            "median_cagr": median(cagr_values),
            "positive_ratio": positive_ratio,
            "mean_avg_pct_gross": mean(avg_pct_gross_values),
        },
        "cross_period": grouped_targets["cross_period"],
        "cross_asset": grouped_targets["cross_asset"],
    }


def aggregate_results(
    records: List[Dict[str, Any]],
    flat_results: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregated: List[Dict[str, Any]] = []

    for record in records:
        params = validate_record(record)
        strategy_hash = params["hash"]
        original_symbol = params["common"]["symbol"]
        original_interval = params["common"]["interval"]

        summary_targets: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
            model_mode: {
                "cross_period": {},
                "cross_asset": {},
            }
            for model_mode in ENABLED_MODEL_MODES
        }

        reports: Dict[str, Dict[str, Dict[str, Any]]] = {
            model_mode: {
                "cross_period": {},
                "cross_asset": {},
            }
            for model_mode in ENABLED_MODEL_MODES
        }

        for (
            result_strategy_hash,
            target_symbol,
            target_interval,
        ), result in flat_results.items():
            if result_strategy_hash != strategy_hash:
                continue

            test_type = _test_type(
                original_symbol,
                original_interval,
                target_symbol,
                target_interval,
            )
            key = target_key(target_symbol, target_interval)

            for model_mode in ENABLED_MODEL_MODES:
                model_result = result.get(model_mode)
                if not model_result:
                    continue

                report = model_result["report"]

                period = MODEL_PERIOD_BY_MODE[model_mode]
                summary_targets[model_mode][test_type][key] = _summary_metrics_from_report(
                    report,
                    period,
                )

                report_entry: Dict[str, Any] = {
                    "report": report,
                    "report_details_path": model_result[
                        "report_details_path"
                    ],
                }

                if model_mode == "retrained_model":
                    report_entry["train_metrics"] = model_result.get("train_metrics")

                reports[model_mode][test_type][key] = report_entry

        strategy_result: Dict[str, Any] = {
            "strategy": {
                "hash": strategy_hash,
                "symbol": original_symbol,
                "interval": original_interval,
            },
            "summary": {model_mode: _aggregate_model_summary(summary_targets[model_mode]) for model_mode in ENABLED_MODEL_MODES},
            "reports": reports,
        }

        aggregated.append(strategy_result)

    return aggregated


def _load_cross_test_plot_rows(
    flat_results: Dict[Tuple[str, str, str], Dict[str, Any]],
    output_dir: str,
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Group detailed cross-test reports by model mode and target market."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for (strategy_hash, symbol, interval), result in flat_results.items():
        for mode in ENABLED_MODEL_MODES:
            model_result = result.get(mode)
            if not model_result:
                continue
            details_path = os.path.join(
                output_dir,
                model_result["report_details_path"],
            )
            with open(details_path, "r", encoding="utf-8") as source:
                report_details = json.load(source)
            grouped.setdefault((mode, symbol, interval), []).append(
                {
                    "raw": model_result["report"],
                    "report_details": report_details,
                    "_strategy_label": strategy_hash,
                }
            )
    return grouped


def regenerate_equity_images(
    records: List[Dict[str, Any]],
    flat_results: Dict[Tuple[str, str, str], Dict[str, Any]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Regenerate selected-strategy and cross-test equity images for the run."""
    from experiment import reports_view

    equity_root = os.path.join(output_dir, EQUITY_PLOT_DIR)
    selected_reports_path = os.path.join(output_dir, SELECTED_FILE)
    selected_rows = [
        reports_view.extract_row(report, selected_reports_path)
        for report in records
    ]
    selected_output_dir = os.path.join(equity_root, "selected")
    reports_view.show_performance(
        selected_rows,
        selected_output_dir,
        batch_size=EQUITY_BATCH_SIZE,
        equity_scale=reports_view.EQUITY_SCALE,
        plot_ood=True,
    )
    selected_indicators_path = os.path.join(
        selected_output_dir,
        reports_view.KEY_STRATEGY_INDICATORS_FILE,
    )
    shutil.copy2(
        selected_indicators_path,
        os.path.join(
            output_dir,
            reports_view.KEY_STRATEGY_INDICATORS_FILE,
        ),
    )

    grouped_rows = _load_cross_test_plot_rows(flat_results, output_dir)
    for (mode, symbol, interval), rows in grouped_rows.items():
        reports_view.plot_in_batches(
            rows,
            os.path.join(
                equity_root,
                "cross_test",
                mode,
                target_key(symbol, interval),
            ),
            batch_size=EQUITY_BATCH_SIZE,
            equity_scale=reports_view.EQUITY_SCALE,
            plot_ood=True,
        )

    logger.info(
        "Regenerated equity images | selected_strategies=%d | "
        "cross_test_groups=%d | destination=%s",
        len(selected_rows),
        len(grouped_rows),
        equity_root,
    )


def write_jsonl(
    path: str,
    records: List[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


LIVE_REPRODUCTION_METRICS = (
    "performance.gross_return",
    "performance.cagr",
    "performance.end_value",
    "performance.sharpe",
    "performance.calmar",
    "drawdown.max_dd_pct",
)


def build_live_reproduction_jobs(live_config_path, strategy_ids=None, settings="original"):
    """Resolve each configured hash in its own report without connecting to venues."""
    from pathlib import Path

    if settings not in {"original", "live"}:
        raise ValueError("settings must be original or live")
    config_path = Path(live_config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    requested = set(strategy_ids or [])
    entries = payload["strategy"]
    unknown = requested - entries.keys()
    if unknown:
        raise ValueError(f"Unknown strategy IDs: {sorted(unknown)}")
    records_by_path = {}
    jobs = []
    for strategy_id, entry in entries.items():
        if requested and strategy_id not in requested:
            continue
        if not isinstance(entry["run_live"], bool):
            raise TypeError(f"run_live must be boolean: {strategy_id}")
        if not requested and not entry["run_live"]:
            continue
        source = (config_path.parent / entry["config_path"]).resolve()
        if source.suffix.lower() != ".jsonl":
            raise ValueError(f"config_path must reference JSONL: {source}")
        if source not in records_by_path:
            records_by_path[source] = load_jsonl(str(source))
        matches = [record for record in records_by_path[source] if record["params"]["hash"] == entry["hash"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one report for {strategy_id}, hash={entry['hash']}, found={len(matches)} in {source}")
        original = matches[0]
        params = validate_record(original)
        model_path = (config_path.parent / entry["model_path"]).resolve()
        for filename in ("model.pt", "meta.json", "train_config.json"):
            file_path = model_path / filename
            if not file_path.is_file() or file_path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing model artifact: {file_path}")
        saved_train = json.loads((model_path / "train_config.json").read_text())
        saved_meta = json.loads((model_path / "meta.json").read_text())
        for key in ("model_cfg", "feature_conf_list"):
            if saved_train[key] != params["train"][key]:
                raise ValueError(f"Model/report {key} mismatch: {strategy_id}")
        if saved_meta["model_type"] != params["train"]["model_cfg"]["model_type"]:
            raise ValueError(f"Model/report type mismatch: {strategy_id}")
        live_broker = entry["broker_config"]
        compound = entry.get("compound", True)
        if not isinstance(compound, bool):
            raise TypeError(f"compound must be boolean: {strategy_id}")
        strategy = copy.deepcopy(params["strategy"])
        if settings == "live":
            strategy["compound"] = compound
        differences = {
            f"broker.{key}": {"original": params["broker"].get(key), "live": value}
            for key, value in live_broker.items() if params["broker"].get(key) != value
        }
        if compound != params["strategy"].get("compound"):
            differences["strategy.compound"] = {"original": params["strategy"].get("compound"), "live": compound}
        jobs.append({
            "strategy_id": strategy_id,
            "hash": entry["hash"],
            "config_path": str(source),
            "model_path": str(model_path),
            "device": entry.get("device", "auto"),
            "settings": settings,
            "configuration_differences": differences,
            "broker": copy.deepcopy(params["broker"] if settings == "original" else live_broker),
            "strategy": strategy,
            "original": original,
        })
    if not jobs:
        raise ValueError("No live-config strategies selected for reproduction")
    return jobs


def compare_reproduction_metrics(original, reproduced, period, *, rtol=1e-6, atol=1e-8):
    """Compare fixed return/risk metrics; absent or nonfinite numbers do not pass."""
    import math
    import pandas as pd

    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise ValueError("Comparison tolerances must be finite and non-negative")
    expected = original["results"][period]
    actual = reproduced["results"][period]
    comparisons = {}
    for path in LIVE_REPRODUCTION_METRICS:
        section, name = path.split(".")
        old, new = expected.get(section, {}).get(name), actual.get(section, {}).get(name)
        finite = all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                     for value in (old, new))
        comparisons[path] = {
            "original": old,
            "reproduced": new,
            "delta": new - old if finite else None,
            "matches": finite and math.isclose(old, new, rel_tol=rtol, abs_tol=atol),
        }
    times = [(pd.to_datetime(expected.get("time", {}).get(key), utc=True, errors="coerce"),
              pd.to_datetime(actual.get("time", {}).get(key), utc=True, errors="coerce"))
             for key in ("start", "end")]
    same_range = all(not pd.isna(old) and not pd.isna(new) and old == new for old, new in times)
    return {"matches": same_range and all(item["matches"] for item in comparisons.values()),
            "period_range_matches": same_range, "metrics": comparisons}


def run_live_reproduction(jobs, output_dir, logger, experiment_context, *, rtol=1e-6, atol=1e-8):
    """Replay only the forward period offline and persist per-strategy comparisons.

    Original mode keeps original broker/compound settings for a like-for-like
    reproduction. Live mode applies live broker/compound overrides and reports
    differences; neither mode connects to a broker or retrains the model.
    Preparation and prediction caches are owned by this output directory.
    """
    import hashlib
    import math
    from trade.runner import backtest_runner

    period = "forward"
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise ValueError("Comparison tolerances must be finite and non-negative")
    os.makedirs(output_dir, exist_ok=True)
    prepared = {}
    source_hashes = {}
    executions = {}
    comparisons = []
    for index, job in enumerate(jobs, start=1):
        params = job["original"]["params"]
        result = {key: job[key] for key in ("strategy_id", "hash", "config_path", "model_path", "settings", "configuration_differences")}
        result.update(periods={}, status="error", rtol=rtol, atol=atol)
        try:
            pre_para = common.BaseDefine(**params["common"])
            source_path = common.market_data_path(pre_para)
            if source_path not in source_hashes:
                source_hashes[source_path] = common.sha256_file(source_path)
            if source_hashes[source_path] != params["data_manifest"]["sha256"]:
                raise ValueError(f"Market data differs from the original report: {source_path}")
            prep_key = TaskIdentity.prep_hash_for(asdict(pre_para))
            prep_dir = os.path.join(output_dir, "preparation", prep_key)
            if prep_key not in prepared:
                if not os.path.isfile(common.get_data_manifest_path_in_dir(prep_dir)):
                    preparation.main(logger, para=pre_para, prep_output_dir=prep_dir)
                validate_preparation(pre_para, prep_dir)
                prepared[prep_key] = prep_dir
            # Reuse identical backtests across venues without losing strategy IDs.
            execution_key = hashlib.sha256(json.dumps(
                {"prep": prep_key, "model": job["model_path"], "broker": job["broker"],
                 "strategy": job["strategy"], "period": period, "device": job["device"]},
                sort_keys=True, default=str,
            ).encode()).hexdigest()[:20]
            if execution_key not in executions:
                run_dir = os.path.join(output_dir, "backtests", execution_key)
                os.makedirs(run_dir, exist_ok=True)
                data_config = backtest_runner.ModelDataConfig(
                    prep_output_dir=prep_dir, train_output_dir=job["model_path"],
                    prediction_cache_dir=os.path.join(output_dir, "prediction_cache", prep_key,
                                                      hashlib.sha256(job["model_path"].encode()).hexdigest()[:20]),
                    device=job["device"], use_prediction_cache=True,
                )
                backtest_runner.precompute_prediction_cache(
                    logger, data_config, config_from_dict_train(params["train"]), period,
                    inference_batch_size=INFERENCE_BATCH_SIZE,
                )
                runner_config = backtest_runner.RunnerConfig(
                    strategy_config=backtest_runner.strategy_config_from_dict(job["strategy"]),
                    broker_config=backtest_runner.BrokerConfig(**job["broker"]),
                    data_config=data_config, save_dir=run_dir, experiment_context=experiment_context,
                )
                logger.info("Replaying %d/%d | strategy=%s period=%s model=%s", index, len(jobs), job["strategy_id"], period, job["model_path"])
                output = backtest_runner.main(logger, runner_config, period)
                report = output["report"]
                validate_report_market(report, pre_para.symbol, pre_para.interval)
                validate_report_period(report, period)
                report_path = os.path.join(run_dir, "report.json")
                write_json(report_path, report)
                write_json(os.path.join(run_dir, REPORT_DETAILS_FILE), output["report_details"])
                executions[execution_key] = (report, report_path)
            report, report_path = executions[execution_key]
            comparison = compare_reproduction_metrics(job["original"], report, period, rtol=rtol, atol=atol)
            comparison["report_path"] = report_path
            result["periods"][period] = comparison
            result["status"] = "matched" if comparison["matches"] else "mismatch"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Reproduction failed: %s", job["strategy_id"])
        logger.info(
            "Live reproduction %d/%d | strategy=%s period=%s | %s | status=%s",
            index, len(jobs), job["strategy_id"], period,
            "PASS" if result["status"] == "matched" else "FAIL", result["status"],
        )
        comparisons.append(result)
        write_jsonl(os.path.join(output_dir, "live_reproduction_comparisons.jsonl"), comparisons)
    summary = {status: sum(item["status"] == status for item in comparisons) for status in ("matched", "mismatch", "error")}
    summary.update(total=len(comparisons), settings=jobs[0]["settings"])
    write_json(os.path.join(output_dir, "live_reproduction_summary.json"), summary)
    import csv

    with open(os.path.join(output_dir, "live_reproduction_metrics.csv"), "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["strategy_id", "hash", "settings", "status", "period", "metric", "original", "reproduced", "delta", "matches"])
        writer.writeheader()
        for result in comparisons:
            for period, comparison in result["periods"].items():
                for metric, values in comparison["metrics"].items():
                    writer.writerow({**{key: result[key] for key in ("strategy_id", "hash", "settings", "status")},
                                     "period": period, "metric": metric, **values})
    logger.info("Reproduction summary: %s", summary)
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel cross-test runner. For every selected strategy and " "target market, run the model modes enabled by the module-level " "feature flags."
        )
    )
    parser.add_argument(
        "--selected-configs",
        default=os.path.join(
            common.PERSISTENCE_DIR,
            "batch_experiments",
            "selected_configs",
            SELECTED_FILE,
        ),
        help="Path to selected_configs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory",
    )
    parser.add_argument(
        "--check-git-clean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a clean Git working tree",
    )
    parser.add_argument("--live-config", help="Replay strategies and model paths from a live_config.json instead of cross-testing")
    parser.add_argument("--strategy-id", action="append", help="Select a live strategy ID; repeat to select several")
    parser.add_argument("--settings", choices=("original", "live"), default="original",
                        help="Keep original broker/compound settings for reproduction, or apply live overrides")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--dry-run", action="store_true", help="Validate live sources and model metadata without running backtests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.live_config:
        jobs = build_live_reproduction_jobs(args.live_config, args.strategy_id, args.settings)
        if args.dry_run:
            for job in jobs:
                print(json.dumps({key: job[key] for key in ("strategy_id", "hash", "config_path", "model_path", "settings", "configuration_differences")}))
            print(f"Validated {len(jobs)} live reproduction jobs; no backtests executed")
            return
        output_dir = os.path.abspath(args.output_dir or os.path.join(
            common.PERSISTENCE_DIR, "live_reproduction", time.strftime("%Y%m%d_%H%M%S")))
        logger = setup_logger(output_dir)
        context = ExperimentContext(git_commit=common.git_revision(require_clean=args.check_git_clean))
        results = run_live_reproduction(jobs, output_dir, logger, context,
                                       rtol=args.rtol, atol=args.atol)
        if any(item["status"] != "matched" for item in results):
            raise SystemExit(1)
        return
    if args.dry_run or args.strategy_id:
        raise ValueError("--dry-run and --strategy-id require --live-config")

    if not os.path.isfile(args.selected_configs):
        raise FileNotFoundError(f"Selected config file not found: {args.selected_configs}")

    records = load_selected_configs(args.selected_configs)
    if not records:
        raise RuntimeError(f"No selected configs found: {args.selected_configs}")

    first_params = validate_record(records[0])
    first_common = common.BaseDefine(**first_params["common"])

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = common.create_experiment_dir(
            os.path.join(
                common.PERSISTENCE_DIR,
                "batch_experiments",
                "cross_test",
            ),
            first_common.symbol,
            first_common.interval,
        )

    logger = setup_logger(output_dir)

    experiment_context = ExperimentContext(git_commit=common.git_revision(require_clean=args.check_git_clean))

    jobs = build_jobs(records, output_dir, args.selected_configs)

    logger.info(
        "Cross-test jobs=%d | MAX_PREP=%d | active_train_workers=%d | " "MAX_SIM=%d | model_modes=%s",
        len(jobs),
        MAX_PREP,
        MAX_TRAIN if ENABLE_RETRAINED_TEST else 0,
        MAX_SIM,
        ENABLED_MODEL_MODES,
    )
    logger.info("Output directory: %s", output_dir)
    logger.info("Git commit: %s", experiment_context.git_commit)

    archive_cross_test_inputs(
        args.selected_configs,
        records,
        output_dir,
        logger,
    )

    begin_time = time.time()

    flat_results = run_parallel_pipeline(
        logger=logger,
        jobs=jobs,
        output_dir=output_dir,
        experiment_context=experiment_context,
    )

    aggregated = aggregate_results(
        records,
        flat_results,
    )

    output_path = os.path.join(
        output_dir,
        "cross_test_reports.jsonl",
    )
    write_jsonl(output_path, aggregated)
    regenerate_equity_images(
        records,
        flat_results,
        output_dir,
        logger,
    )

    logger.info(
        "Completed in %.2fs. Reports saved to %s",
        time.time() - begin_time,
        output_path,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
