#!/usr/bin/env python3
"""Measure a selected strategy against uniformly random model predictions."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as datetime_module
import json
import logging
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

current_work_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_work_dir, "..", "..")))

from data_process import common, preparation
from data_process.utils import TaskIdentity, load_selected_configs

SELECTED_CONFIGS_FILE = "selected_configs.jsonl"
DEFAULT_RUN_COUNT = 1000
DEFAULT_BIN_COUNT = 10
DEFAULT_EQUITY_BATCH_SIZE = 10
PERIODS = ("long", "forward")


def default_cross_test_root() -> str:
    return os.path.join(
        common.PERSISTENCE_DIR,
        "batch_experiments",
        "cross_test",
    )


def resolve_cross_test_folder(folder: str | None) -> tuple[str, str]:
    """Resolve a cross-test run and its selected configuration file."""
    candidate = Path(folder or default_cross_test_root()).expanduser().resolve()
    if candidate.is_file():
        if candidate.name != SELECTED_CONFIGS_FILE:
            raise ValueError(f"Expected {SELECTED_CONFIGS_FILE}, received file: {candidate}")
        return str(candidate.parent), str(candidate)

    if not candidate.is_dir():
        raise FileNotFoundError(f"Cross-test folder does not exist: {candidate}")

    direct_file = candidate / SELECTED_CONFIGS_FILE
    if direct_file.is_file():
        return str(candidate), str(direct_file)

    matches = [path for path in candidate.rglob(SELECTED_CONFIGS_FILE) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No {SELECTED_CONFIGS_FILE} found below: {candidate}")
    selected_path = max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return str(selected_path.parent), str(selected_path)


def record_params(record: dict[str, Any]) -> dict[str, Any]:
    params = common.recursive_get(record, "params")
    if not isinstance(params, dict):
        raise ValueError("Selected configuration record has no params object")
    required = ("hash", "common", "train", "strategy", "broker")
    missing = [name for name in required if params.get(name) is None]
    if missing:
        raise ValueError(f"Selected configuration is missing required params: {missing}")
    return params


def select_record(
    records: Iterable[dict[str, Any]],
    strategy_hash: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = [(record, record_params(record)) for record in records]
    if not validated:
        raise RuntimeError("The selected configuration file is empty")

    if strategy_hash is None:
        if len(validated) == 1:
            return validated[0]
        hashes = ", ".join(str(params["hash"]) for _, params in validated)
        raise ValueError("--hash is required when the selected configuration file contains " f"multiple strategies. Available hashes: {hashes}")

    exact = [item for item in validated if str(item[1]["hash"]) == strategy_hash]
    if len(exact) == 1:
        return exact[0]

    prefix = [item for item in validated if str(item[1]["hash"]).startswith(strategy_hash)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        matches = ", ".join(str(params["hash"]) for _, params in prefix)
        raise ValueError(f"Strategy hash prefix is ambiguous: {matches}")
    raise ValueError(f"Strategy hash was not found: {strategy_hash}")


def archived_model_dir(cross_test_folder: str, strategy_hash: str) -> str:
    model_dir = os.path.abspath(os.path.join(cross_test_folder, "models", strategy_hash))
    required_files = ("meta.json", "train_config.json")
    missing = [filename for filename in required_files if not os.path.isfile(os.path.join(model_dir, filename))]
    if missing:
        raise FileNotFoundError(f"Archived model metadata is incomplete: path={model_dir}, missing={missing}")
    return model_dir


def load_model_metadata(model_dir: str) -> dict[str, Any]:
    path = os.path.join(model_dir, "meta.json")
    with open(path, "r", encoding="utf-8") as source:
        metadata = json.load(source)
    required = ("feature_cols", "label_col", "seq_len")
    missing = [name for name in required if metadata.get(name) is None]
    if missing:
        raise ValueError(f"Model metadata is missing required fields: {missing}")
    return metadata


def create_output_dir(
    cross_test_folder: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
    requested: str | None,
) -> str:
    if requested:
        output_dir = os.path.abspath(os.path.expanduser(requested))
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    root = os.path.join(
        cross_test_folder,
        "random_prediction",
        strategy_hash,
    )
    return common.create_experiment_dir(root, symbol, interval)


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("random_prediction_validation")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "random_prediction_validation.log"),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def worker_logger(path: str) -> logging.Logger:
    logger = logging.getLogger(f"random_prediction_worker_{os.getpid()}")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def write_json(path: str, payload: Any) -> None:
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


def write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output:
            for record in records:
                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def validate_preparation(pre_para: common.BaseDefine, prep_dir: str) -> None:
    expected_hash = TaskIdentity.prep_hash_for(asdict(pre_para))
    manifest = common.load_data_manifest_from_dir(prep_dir)
    if manifest.get("configuration_hash") != expected_hash:
        raise RuntimeError("Preparation configuration hash does not match the selected strategy")
    for path in (
        common.get_train_data_path_in_dir(prep_dir),
        common.get_test_data_path_in_dir(prep_dir),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prepared dataset was not created: {path}")


def random_prediction_payload(
    *,
    backtest_runner,
    data_loader,
    data_config,
    train_cfg,
    frame,
    period: str,
    metadata: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    feature_cols = list(metadata["feature_cols"])
    label_col = str(metadata["label_col"])
    seq_len = int(metadata["seq_len"])
    active_indices = data_loader.valid_window_end_indices(
        frame,
        feature_cols=feature_cols,
        label_col=label_col,
        seq_len=seq_len,
        stride=1,
    )
    if len(active_indices) == 0:
        raise RuntimeError(f"No valid prediction windows found for period={period}")

    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet(np.ones(3), size=len(active_indices))
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    prediction_columns = list(backtest_runner._PREDICTION_COLUMNS)
    predicted_frame = frame.loc[:, []].copy()
    for column in prediction_columns:
        predicted_frame[column] = np.nan
    predicted_frame.loc[active_indices, "pred"] = predictions
    predicted_frame.loc[active_indices, "pred_prob"] = probabilities.max(axis=1)
    predicted_frame.loc[active_indices, "prob_short"] = probabilities[:, 0]
    predicted_frame.loc[active_indices, "prob_neutral"] = probabilities[:, 1]
    predicted_frame.loc[active_indices, "prob_long"] = probabilities[:, 2]
    predicted_frame.loc[active_indices, "net_score"] = probabilities[:, 2] - probabilities[:, 0]

    handler_metadata = SimpleNamespace(
        feature_cols=feature_cols,
        label_col=label_col,
        seq_len=seq_len,
    )
    time_regions = backtest_runner._model_time_regions(
        data_config,
        train_cfg,
        handler_metadata,
        frame,
        period,
    )
    counts = np.bincount(predictions, minlength=3)
    return {
        "schema": backtest_runner._PREDICTION_CACHE_SCHEMA,
        "predictions": predicted_frame,
        "first_valid_idx": active_indices[0],
        "model_stats": {
            "random_baseline": True,
            "seed": int(seed),
            "sample_count": int(len(predictions)),
            "label_distribution_pred": {str(index): int(count) for index, count in enumerate(counts)},
        },
        "sub_model_stats": {},
        "time_regions": time_regions,
    }


def run_random_simulation(task: dict[str, Any]) -> dict[str, Any]:
    import torch

    from model import data_loader
    from trade.runner import backtest_runner
    from trade.runner.config import ExperimentContext

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    run_index = int(task["run_index"])
    run_dir = os.path.join(task["output_dir"], "runs", f"run_{run_index:03d}")
    os.makedirs(run_dir, exist_ok=True)
    logger = worker_logger(os.path.join(run_dir, "simulation.log"))
    strategy_config = backtest_runner.strategy_config_from_dict(task["params"]["strategy"])
    broker_config = backtest_runner.BrokerConfig(**task["params"]["broker"])
    data_config = backtest_runner.ModelDataConfig(
        prep_output_dir=task["prep_dir"],
        train_output_dir=task["model_dir"],
        device="cpu",
        use_prediction_cache=False,
    )
    original_inference = backtest_runner._infer_prediction_payload
    period_results: dict[str, Any] = {}
    period_details: dict[str, Any] = {}
    shared_params = None
    period_seeds: dict[str, int] = {}

    try:
        for period_index, period in enumerate(task["periods"], start=1):
            period_seed = int(np.random.SeedSequence([int(task["base_seed"]), run_index, period_index]).generate_state(1, dtype=np.uint32)[0])
            period_seeds[period] = period_seed

            def infer_random(
                logger,
                data_config,
                train_cfg,
                frame,
                train_output_dir,
                interval_ms,
                requested_period,
                inference_batch_size=512,
                *,
                _seed=period_seed,
            ):
                del logger, train_output_dir, interval_ms, inference_batch_size
                return random_prediction_payload(
                    backtest_runner=backtest_runner,
                    data_loader=data_loader,
                    data_config=data_config,
                    train_cfg=train_cfg,
                    frame=frame,
                    period=requested_period,
                    metadata=task["model_metadata"],
                    seed=_seed,
                )

            backtest_runner._infer_prediction_payload = infer_random
            runner_config = backtest_runner.RunnerConfig(
                strategy_config=strategy_config,
                broker_config=broker_config,
                save_dir=os.path.join(run_dir, period),
                data_config=data_config,
                experiment_context=ExperimentContext(git_commit=task["git_commit"]),
            )
            output = backtest_runner.main(logger, runner_config, period)
            period_report = output["report"]
            current_params = period_report["params"]
            if shared_params is None:
                shared_params = current_params
            elif current_params != shared_params:
                raise RuntimeError("Period reports produced different parameter snapshots")
            period_results[period] = period_report["results"][period]
            period_details[period] = output["report_details"]["results"][period]
            del output
    finally:
        backtest_runner._infer_prediction_payload = original_inference

    random_metadata = {
        "run_index": run_index,
        "base_seed": int(task["base_seed"]),
        "period_seeds": period_seeds,
        "distribution": "symmetric_dirichlet_argmax",
    }
    report = {
        "random_prediction": random_metadata,
        "params": shared_params,
        "results": period_results,
    }
    details = {
        "random_prediction": random_metadata,
        "results": period_details,
    }
    report_path = os.path.join(run_dir, "report.json")
    details_path = os.path.join(run_dir, "report_details.json")
    write_json(report_path, report)
    write_json(details_path, details)

    metrics = {
        period: {
            "cagr": period_results[period]["performance"].get("cagr"),
            "avg_pct_gross": period_results[period]["trades"].get("avg_pct_gross"),
        }
        for period in task["periods"]
    }
    return {
        "run_index": run_index,
        "report_path": report_path,
        "details_path": details_path,
        "metrics": metrics,
        "period_seeds": period_seeds,
    }


def finite_metric_values(
    completed_runs: Iterable[dict[str, Any]],
    period: str,
    metric: str,
) -> list[float]:
    values = []
    for result in completed_runs:
        value = (result.get("metrics", {}).get(period, {}) or {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                values.append(value)
    return values


def histogram_summary(values: list[float], bin_count: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "bins": []}
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bin_count)
    total = len(values)
    return {
        "count": total,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bins": [
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "upper_inclusive": index == len(counts) - 1,
                "count": int(count),
                "fraction": float(count / total),
            }
            for index, count in enumerate(counts)
        ],
    }


def build_distribution_summary(
    completed_runs: list[dict[str, Any]],
    periods: Iterable[str],
    bin_count: int,
) -> dict[str, Any]:
    return {
        period: {
            metric: histogram_summary(
                finite_metric_values(completed_runs, period, metric),
                bin_count,
            )
            for metric in ("cagr", "avg_pct_gross")
        }
        for period in periods
    }


def print_distribution_summary(summary: dict[str, Any]) -> None:
    for period, metrics in summary.items():
        for metric, stats in metrics.items():
            print(
                f"\n{period.upper()} {metric} | count={stats['count']} "
                f"mean={stats.get('mean', float('nan')):.6g} "
                f"median={stats.get('median', float('nan')):.6g} "
                f"min={stats.get('min', float('nan')):.6g} "
                f"max={stats.get('max', float('nan')):.6g}"
            )
            print(f"{'bucket':>31} {'count':>8} {'fraction':>10}")
            for bucket in stats["bins"]:
                right = "]" if bucket["upper_inclusive"] else ")"
                label = f"[{bucket['lower']:.6g}, {bucket['upper']:.6g}{right}"
                print(f"{label:>31} {bucket['count']:>8d} " f"{bucket['fraction']:>9.2%}")


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def plot_all_equities(
    completed_runs: list[dict[str, Any]],
    output_dir: str,
    batch_size: int,
    equity_scale: str,
) -> list[str]:
    from trade.runner.analyze_backtest_report import plot_equity_curves

    equity_dir = os.path.join(output_dir, "equity")
    os.makedirs(equity_dir, exist_ok=True)
    paths = []
    for start in range(0, len(completed_runs), batch_size):
        group = completed_runs[start : start + batch_size]
        payloads = []
        for completed in group:
            report = load_json(completed["report_path"])
            details = load_json(completed["details_path"])
            report["params"] = dict(report["params"])
            report["params"]["hash"] = f"R{completed['run_index']:03d}"
            payloads.append({"report": report, "report_details": details})
        figure_index = start // batch_size + 1
        filename = f"batch_{figure_index:03d}.png"
        path = plot_equity_curves(
            payloads,
            equity_dir,
            filename,
            equity_scale=equity_scale,
            include_ood=True,
        )
        if path:
            paths.append(path)
            print(f"Equity image saved: {path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run a selected cross-test strategy repeatedly with uniformly random " "three-class predictions."))
    parser.add_argument(
        "--folder",
        default=None,
        help=("Cross-test run folder or an ancestor containing cross-test runs. " "The newest selected_configs.jsonl is used when an ancestor is given."),
    )
    parser.add_argument(
        "--hash",
        dest="strategy_hash",
        default=None,
        help="Exact strategy hash or a unique hash prefix",
    )
    parser.add_argument(
        "--list-hashes",
        action="store_true",
        help="List available strategy hashes and exit",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUN_COUNT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(25, os.cpu_count() or 1),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=DEFAULT_BIN_COUNT)
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=PERIODS,
        default=list(PERIODS),
        help="Backtest periods included in every random simulation task",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--equity-batch-size",
        type=int,
        default=DEFAULT_EQUITY_BATCH_SIZE,
        help="Number of random equity curves per image",
    )
    parser.add_argument(
        "--equity-scale",
        choices=("linear", "log", "both"),
        default="linear",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("runs", "workers", "bins", "equity_batch_size"):
        if getattr(args, name) <= 0:
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be positive")
    args.periods = list(dict.fromkeys(args.periods))


def main() -> None:
    args = parse_args()
    validate_args(args)
    cross_test_folder, selected_configs_path = resolve_cross_test_folder(args.folder)
    records = load_selected_configs(selected_configs_path)

    if args.list_hashes:
        for record in records:
            params = record_params(record)
            pre_para = common.BaseDefine(**params["common"])
            print(f"{params['hash']}\t{pre_para.symbol}\t{pre_para.interval}")
        return

    _, params = select_record(records, args.strategy_hash)
    strategy_hash = str(params["hash"])
    pre_para = common.BaseDefine(**params["common"])
    model_dir = archived_model_dir(cross_test_folder, strategy_hash)
    model_metadata = load_model_metadata(model_dir)
    output_dir = create_output_dir(
        cross_test_folder,
        strategy_hash,
        pre_para.symbol,
        pre_para.interval,
        args.output_dir,
    )
    logger = setup_logger(output_dir)
    logger.info("Cross-test folder: %s", cross_test_folder)
    logger.info("Selected configurations: %s", selected_configs_path)
    logger.info("Strategy hash: %s", strategy_hash)
    logger.info("Output directory: %s", output_dir)

    prep_dir = os.path.join(output_dir, "pre_output")
    logger.info("Running preparation: %s", prep_dir)
    preparation.main(logger, para=pre_para, prep_output_dir=prep_dir)
    validate_preparation(pre_para, prep_dir)

    git_commit = common.git_revision(require_clean=False)
    run_manifest = {
        "schema_version": 1,
        "created_at": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "cross_test_folder": cross_test_folder,
        "selected_configs_path": selected_configs_path,
        "strategy_hash": strategy_hash,
        "model_dir": model_dir,
        "prep_dir": prep_dir,
        "runs": args.runs,
        "workers": args.workers,
        "base_seed": args.seed,
        "periods": args.periods,
        "bins": args.bins,
        "git_commit": git_commit,
        "random_prediction_distribution": "symmetric_dirichlet_argmax",
    }
    write_json(os.path.join(output_dir, "run_manifest.json"), run_manifest)

    common_task = {
        "params": params,
        "prep_dir": prep_dir,
        "model_dir": model_dir,
        "model_metadata": model_metadata,
        "output_dir": output_dir,
        "base_seed": args.seed,
        "periods": args.periods,
        "git_commit": git_commit,
    }
    completed_runs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, args.runs)) as executor:
        future_to_index = {
            executor.submit(
                run_random_simulation,
                {**common_task, "run_index": run_index},
            ): run_index
            for run_index in range(1, args.runs + 1)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            run_index = future_to_index[future]
            try:
                completed = future.result()
            except Exception:
                logger.exception("Random simulation failed: run=%03d", run_index)
                for pending in future_to_index:
                    pending.cancel()
                raise
            completed_runs.append(completed)
            logger.info(
                "Random simulation completed: %d/%d (run=%03d)",
                len(completed_runs),
                args.runs,
                run_index,
            )

    completed_runs.sort(key=lambda item: item["run_index"])
    reports = [load_json(item["report_path"]) for item in completed_runs]
    write_jsonl(os.path.join(output_dir, "reports.jsonl"), reports)
    write_jsonl(os.path.join(output_dir, "run_results.jsonl"), completed_runs)

    distribution_summary = build_distribution_summary(
        completed_runs,
        args.periods,
        args.bins,
    )
    summary_path = os.path.join(output_dir, "distribution_summary.json")
    write_json(summary_path, distribution_summary)
    print_distribution_summary(distribution_summary)

    equity_paths = plot_all_equities(
        completed_runs,
        output_dir,
        args.equity_batch_size,
        args.equity_scale,
    )
    logger.info("Distribution summary: %s", summary_path)
    logger.info("Equity images generated: %d", len(equity_paths))
    logger.info("Random prediction validation completed: %s", output_dir)


# Run from the repository root: python -m test.offline.test_random_prediction_validation --help
if __name__ == "__main__":
    main()
