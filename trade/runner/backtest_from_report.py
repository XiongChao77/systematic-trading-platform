#!/usr/bin/env python3
"""Re-run one backtest from a report stored in a JSONL file.

Edit REPORT_PATH, REPORT_LINE_INDEX and PERIOD before running this script.
The referenced preprocessing data and trained-model artifacts must still exist.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data_process import common
from data_process.utils import config_from_dict_train
from trade.runner import backtest_runner
from trade.runner.analyze_backtest_report import plot_equity_curves


# Modify these values manually before running.
REPORT_PATH = (
    "/home/chao/work/quant_output/batch_experiments/selected_configs/"
    "selected_configs.jsonl"
)
REPORT_LINE_INDEX = 0  # Zero-based index of the non-empty JSONL record.
PERIOD = "long"  # short | forward | long | all


def load_jsonl_record(path: str, record_index: int) -> dict[str, Any]:
    """Load one non-empty JSONL record without reading the whole file."""
    if record_index < 0:
        raise ValueError("REPORT_LINE_INDEX must be zero or greater")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Report file not found: {path}")

    current_index = 0
    with open(path, "r", encoding="utf-8") as report_file:
        for file_line_number, line in enumerate(report_file, start=1):
            if not line.strip():
                continue
            if current_index != record_index:
                current_index += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{file_line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected a JSON object at record {record_index}, "
                    f"got {type(record).__name__}"
                )
            return record

    raise IndexError(
        f"REPORT_LINE_INDEX={record_index} is outside the report file; "
        f"found {current_index} non-empty records"
    )


def select_period_report(record: dict[str, Any], period: str) -> dict[str, Any]:
    """Materialize one period from a canonical summary report."""
    if period not in {"short", "forward", "long", "all"}:
        raise ValueError(f"Unsupported PERIOD: {period!r}")

    source_period = "long" if period == "all" else period
    params = _required_dict(record, "params")
    results = _required_dict(record, "results")
    period_result = _required_dict(results, source_period)
    return {"params": params, **period_result}


def _required_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing report parameter object: params.{key}")
    return value


def build_runner_config(
    period_report: dict[str, Any],
    *,
    save_dir: str,
) -> tuple[backtest_runner.RunnerConfig, common.BaseDefine, Any, str | None]:
    """Restore backtest_runner configuration objects from a report snapshot."""
    params = _required_dict(period_report, "params")
    strategy_params = _required_dict(params, "strategy")
    broker_params = _required_dict(params, "broker")
    common_params = _required_dict(params, "common")
    train_params = _required_dict(params, "train")
    data_params = _required_dict(params, "data")

    strategy_config = backtest_runner.strategy_config_from_dict(strategy_params)
    broker_config = backtest_runner.BrokerConfig(**broker_params)
    market_config = common.preparation_config_from_dict(common_params)
    train_config = config_from_dict_train(train_params)

    prep_output_dir = data_params.get("prep_output_dir")
    train_output_dir = data_params.get("train_output_dir")
    if not prep_output_dir:
        raise ValueError("Missing params.data.prep_output_dir")
    if not train_output_dir:
        raise ValueError("Missing params.data.train_output_dir")

    data_config = backtest_runner.ModelDataConfig(
        prep_output_dir=str(prep_output_dir),
        train_output_dir=str(train_output_dir),
        device=str(data_params.get("device", "cpu")),
    )
    runner_config = backtest_runner.RunnerConfig(
        strategy_config=strategy_config,
        broker_config=broker_config,
        data_config=data_config,
        save_dir=save_dir,
        experiment_context=backtest_runner.ExperimentContext(
            git_commit=common.git_revision(),
        ),
    )
    return runner_config, market_config, train_config, params.get("hash")


def preparation_artifacts_exist(prep_output_dir: str) -> bool:
    """Return whether preparation produced all files consumed downstream."""
    data_suffix = "csv" if common.CONF_DF == "to_csv" else "feather"
    required_paths = (
        common.get_data_config_path_in_dir(prep_output_dir),
        os.path.join(prep_output_dir, f"train_data.{data_suffix}"),
        os.path.join(prep_output_dir, f"test_data.{data_suffix}"),
    )
    return all(os.path.isfile(path) for path in required_paths)


def model_artifacts_exist(train_output_dir: str) -> bool:
    """Validate the model artifact graph referenced by task_description.json."""
    visited: set[str] = set()

    def validate_artifact_dir(artifact_dir: str) -> bool:
        artifact_dir = os.path.abspath(artifact_dir)
        if artifact_dir in visited:
            return False
        visited.add(artifact_dir)

        task_path = os.path.join(artifact_dir, "task_description.json")
        data_config_path = common.get_data_config_path_in_dir(artifact_dir)
        if not os.path.isfile(task_path) or not os.path.isfile(data_config_path):
            return False
        try:
            with open(task_path, "r", encoding="utf-8") as task_file:
                task_description = json.load(task_file)
        except (OSError, json.JSONDecodeError):
            return False

        models = task_description.get("models")
        if not isinstance(models, dict) or not models:
            return False
        for model_spec in models.values():
            if isinstance(model_spec, dict):
                model_path = model_spec.get("model")
                meta_path = model_spec.get("meta")
                if not model_path or not meta_path:
                    return False
                if not os.path.isfile(os.path.join(artifact_dir, model_path)):
                    return False
                if not os.path.isfile(os.path.join(artifact_dir, meta_path)):
                    return False
            elif isinstance(model_spec, str):
                if not validate_artifact_dir(os.path.join(artifact_dir, model_spec)):
                    return False
            else:
                return False
        return True

    return validate_artifact_dir(train_output_dir)


def ensure_backtest_artifacts(
    config: backtest_runner.RunnerConfig,
    market_config: common.BaseDefine,
    train_config: Any,
    logger: logging.Logger,
) -> bool:
    """Prepare data and train from restored report params when artifacts are absent.

    Returns True when a model was trained and False when the existing model was
    reused. Missing preparation data alone is regenerated without retraining a
    complete model.
    """
    data_config = config.data_config
    prep_ready = preparation_artifacts_exist(data_config.prep_output_dir)
    model_ready = model_artifacts_exist(data_config.train_output_dir)

    if model_ready and prep_ready:
        logger.info("Reusing complete training artifacts: %s", data_config.train_output_dir)
        return False

    from data_process import preparation

    if not model_ready:
        logger.warning(
            "Model artifacts are missing or incomplete; rebuilding from report params: %s",
            data_config.train_output_dir,
        )
        if train_config is None:
            raise ValueError("The report does not contain params.train")

        from model import train

        if train_config.train_task in train.TrainTask.COMBO_TASKS:
            raise ValueError(
                f"Cannot rebuild composite task {train_config.train_task!r} from one "
                "report config; its two sub-model training configs are required"
            )

        # Rebuild preparation as well so the model and its data metadata come
        # from the same report snapshot and current market-data source.
        preparation.main(
            logger,
            para=market_config,
            prep_output_dir=data_config.prep_output_dir,
        )
        os.makedirs(data_config.train_output_dir, exist_ok=True)
        train.train(
            logger=logger,
            config=train_config,
            prep_output_dir=data_config.prep_output_dir,
            save_dir=data_config.train_output_dir,
        )
        if not model_artifacts_exist(data_config.train_output_dir):
            raise RuntimeError(
                "Training completed without producing a loadable artifact set at "
                f"{data_config.train_output_dir}"
            )
        logger.info("Model training artifacts created: %s", data_config.train_output_dir)
        return True

    logger.warning(
        "Preparation artifacts are missing; rebuilding them at %s",
        data_config.prep_output_dir,
    )
    preparation.main(
        logger,
        para=market_config,
        prep_output_dir=data_config.prep_output_dir,
    )
    if not preparation_artifacts_exist(data_config.prep_output_dir):
        raise RuntimeError(
            "Preparation completed without producing all required files at "
            f"{data_config.prep_output_dir}"
        )
    return False


def main() -> None:
    started_at = time.time()
    record = load_jsonl_record(REPORT_PATH, REPORT_LINE_INDEX)
    period_report = select_period_report(record, PERIOD)

    params = _required_dict(period_report, "params")
    common_params = _required_dict(params, "common")
    symbol = str(common_params["symbol"])
    interval = str(common_params["interval"])
    output_dir = common.create_experiment_dir(
        os.path.join(common.PERSISTENCE_DIR, "simulation"),
        symbol,
        interval,
    )
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(output_dir, "experiment.log"),
        console_level=logging.INFO,
        file_level=logging.INFO,
    )
    runner_config, market_config, train_config, params_hash = build_runner_config(
        period_report,
        save_dir=output_dir,
    )

    logger.info("Source report: %s", REPORT_PATH)
    logger.info("Report record: %d", REPORT_LINE_INDEX)
    logger.info("Period: %s", PERIOD)
    logger.info("Params hash: %s", params_hash)
    logger.info("Prep output: %s", runner_config.data_config.prep_output_dir)
    logger.info("Train output: %s", runner_config.data_config.train_output_dir)
    ensure_backtest_artifacts(runner_config, market_config, train_config, logger)
    result = backtest_runner.main(logger, runner_config, PERIOD)
    reports_path = os.path.join(output_dir, "reports.jsonl")
    common.append_jsonl(reports_path, result["report"])
    plot_equity_curves(
        result,
        output_dir,
        file_name=f"equity_{PERIOD}.png",
        logger=logger,
    )
    logger.info(
        "Run time: %.4f s, report saved to %s",
        time.time() - started_at,
        reports_path,
    )


if __name__ == "__main__":
    main()
