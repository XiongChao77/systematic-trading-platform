#!/usr/bin/env python3
"""Trigger+Direction fusion model-family comparison on one rolling split.

Every arm trains the same two heads (TRIGGER + DIRECTION) with identical
prepared data, rolling windows, sequence length and stride. Only the model
family changes, so the backtest differences are attributable to the model.

Arms:
    xgboost      current rolling baseline (TriggerModel / DirectionModel)
    logistic     both heads use LogisticConfig
    transformer  both heads use TransformerConfig

Each arm is backtested with period=development, which reports the Train
(fold0 in-sample) and OOS stages separately as well as the combined run.

Outputs (under experiment/sweep_outputs/fusion_model_compare/):
    <arm>/train, <arm>/backtest, <arm>/experiment.log
    summary.json / summary.csv / comparison.md

Usage:
    python experiment/fusion_model_comparison.py
    python experiment/fusion_model_comparison.py --arms xgboost,logistic
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import replace

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data_process import common
from data_process import preparation
from data_process.config import DOGE_15m_TBM
from model.train import train_windows
from model.train_config import (
    DirectionModel,
    LogisticConfig,
    RollingConfig,
    TransformerConfig,
    TriggerModel,
)
from trade.runner.analyze_backtest_report import plot_equity_curves
from trade.runner.backtest_entry import main as run_backtest
from trade.runner.config import (
    BrokerConfig,
    ExperimentContext,
    ModelDataConfig,
    RunnerConfig,
)
from trade.strategy.strategy_bbm import BbmStrategyConfig

SWEEP_ROOT = os.path.join(os.path.dirname(__file__), "sweep_outputs")
OUT_ROOT = os.path.join(SWEEP_ROOT, "fusion_model_compare")
SHARED_DATA_DIR = os.path.join(SWEEP_ROOT, "doge", "data")
TARGET_SYMBOL = "DOGEUSDT"

ARMS = ("xgboost", "logistic", "transformer")

STAGE_KEYS = (
    "cagr_pct", "sharpe", "calmar", "maxdd_pct",
    "trades", "winrate_pct",
)
MODEL_KEYS = (
    "accuracy", "accuracy_minus_baseline", "macro_f1",
    "coverage", "directional_accuracy",
)


def build_configs(arm: str, seq_len: int, epochs: int, patience: int):
    """Return (trigger, direction) TrainConfigs for one model family."""
    if arm == "xgboost":
        return TriggerModel, DirectionModel
    if arm == "logistic":
        trigger_cfg = LogisticConfig(seq_len=seq_len)
        direction_cfg = LogisticConfig(seq_len=seq_len)
    elif arm == "transformer":
        trigger_cfg = TransformerConfig(seq_len=seq_len)
        direction_cfg = TransformerConfig(seq_len=seq_len)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return (
        replace(TriggerModel, model_cfg=trigger_cfg, epochs=epochs, patience=patience),
        replace(DirectionModel, model_cfg=direction_cfg, epochs=epochs, patience=patience),
    )


def ensure_data(data_dir: str, split_time: str, logger: logging.Logger) -> str:
    """Reuse an existing preparation, or build one at the requested split."""
    manifest_path = os.path.join(data_dir, "data_manifest.json")
    if os.path.isfile(manifest_path):
        manifest = common.load_data_manifest_from_dir(data_dir)
        saved_split = str(manifest.get("split_time"))
        if saved_split != split_time:
            raise ValueError(
                f"Prepared data split_time={saved_split} does not match "
                f"requested split_time={split_time}"
            )
        logger.info("Reusing prepared data | %s | data_id=%s", data_dir, manifest.get("data_id"))
        return data_dir
    logger.info("Preparing data | %s", data_dir)
    preparation.main(
        logger,
        common.FEATURE_GROUP_LIST,
        [],
        para=replace(DOGE_15m_TBM, symbols=(TARGET_SYMBOL,)),
        prep_output_dir=data_dir,
        split_time=split_time,
    )
    return data_dir


def _stage_block(stage_summary: dict, stage: str) -> dict:
    block = (stage_summary.get("stages") or {}).get(stage) or {}
    return block


def collect_metrics(backtest_dir: str, logger: logging.Logger) -> dict:
    path = os.path.join(backtest_dir, "stage_summary.json")
    if not os.path.isfile(path):
        logger.warning("Missing stage summary: %s", path)
        return {}
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)

    metrics = {}
    for stage in ("train", "oos"):
        block = _stage_block(summary, stage)
        strategy = block.get("strategy") or {}
        perf = strategy.get("performance") or {}
        drawdown = strategy.get("drawdown") or {}
        trades = strategy.get("trades") or {}
        model = block.get("model") or {}
        metrics[f"{stage}_cagr_pct"] = round(float(perf.get("cagr") or 0.0) * 100, 2)
        metrics[f"{stage}_sharpe"] = round(float(perf.get("sharpe") or 0.0), 3)
        metrics[f"{stage}_calmar"] = round(float(perf.get("calmar") or 0.0), 3)
        metrics[f"{stage}_maxdd_pct"] = round(float(drawdown.get("max_dd_pct") or 0.0), 2)
        metrics[f"{stage}_trades"] = int(trades.get("total") or 0)
        metrics[f"{stage}_winrate_pct"] = round(float(trades.get("win_rate") or 0.0) * 100, 2)
        for key in MODEL_KEYS:
            value = model.get(key)
            metrics[f"{stage}_{key}"] = None if value is None else round(float(value), 4)

    overall = summary.get("overall") or {}
    perf = overall.get("performance") or {}
    drawdown = overall.get("drawdown") or {}
    trades = overall.get("trades") or {}
    metrics["overall_cagr_pct"] = round(float(perf.get("cagr") or 0.0) * 100, 2)
    metrics["overall_sharpe"] = round(float(perf.get("sharpe") or 0.0), 3)
    metrics["overall_calmar"] = round(float(perf.get("calmar") or 0.0), 3)
    metrics["overall_maxdd_pct"] = round(float(drawdown.get("max_dd_pct") or 0.0), 2)
    metrics["overall_trades"] = int(trades.get("total") or 0)
    metrics["overall_winrate_pct"] = round(float(trades.get("win_rate") or 0.0) * 100, 2)
    return metrics


def run_arm(
    arm: str,
    *,
    data_dir: str,
    out_root: str,
    mode: str,
    period: str,
    oos_months: int,
    seq_len: int,
    epochs: int,
    patience: int,
    stop_on_error: bool,
) -> dict:
    arm_dir = os.path.join(out_root, arm)
    train_dir = os.path.join(arm_dir, "train")
    backtest_dir = os.path.join(arm_dir, "backtest")
    if os.path.isdir(arm_dir):
        shutil.rmtree(arm_dir)
    os.makedirs(arm_dir, exist_ok=True)

    trigger_cfg, direction_cfg = build_configs(arm, seq_len, epochs, patience)
    record = {
        "arm": arm,
        "model_type": trigger_cfg.model_cfg.model_type,
        "model_version": trigger_cfg.model_cfg.model_version,
        "seq_len": seq_len,
        "trigger_stride": trigger_cfg.stride,
        "direction_stride": direction_cfg.stride,
        "mode": mode,
        "period": period,
        "oos_months": oos_months,
        "status": "ok",
        "train_dir": train_dir,
        "backtest_dir": backtest_dir,
    }

    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(arm_dir, "experiment.log"),
        console_level=logging.INFO,
        file_level=logging.INFO,
    )
    try:
        logger.info(
            "=== [%s] fusion trigger+direction | model=%s | mode=%s oos_months=%s ===",
            arm, trigger_cfg.model_cfg.model_type, mode, oos_months,
        )
        train_windows(
            configs=[trigger_cfg, direction_cfg],
            rolling_config=RollingConfig(mode=mode, oos_months=oos_months),
            prep_output_dir=data_dir,
            save_dir=train_dir,
            logger=logger,
        )

        prep = common.load_pre_params_from_dir(train_dir)
        config = RunnerConfig(
            strategy_config=BbmStrategyConfig(fixed_hold_bars=prep.predict_num),
            broker_config=BrokerConfig(),
            save_dir=backtest_dir,
            experiment_context=ExperimentContext(git_commit=common.git_revision()),
            data_config=ModelDataConfig(
                prep_output_dir=data_dir,
                train_output_dir=train_dir,
                device="auto",
                symbol=TARGET_SYMBOL,
            ),
        )
        result = run_backtest(logger, config, period=period, evaluate_stages=False)
        record["equity_path"] = plot_equity_curves(
            result,
            backtest_dir,
            file_name="equity_curve.png",
            normalize_equity=False,
            include_oos=True,
            logger=logger,
        )
        record["metrics"] = collect_metrics(backtest_dir, logger)
        logger.info("=== [%s] DONE ===", arm)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("=== [%s] FAILED ===", arm)
        if stop_on_error:
            raise
    return record


def write_summary(records: list[dict], out_root: str) -> tuple[str, str, str]:
    os.makedirs(out_root, exist_ok=True)
    json_path = os.path.join(out_root, "summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    metric_columns = [
        *(f"train_{key}" for key in STAGE_KEYS),
        *(f"oos_{key}" for key in STAGE_KEYS),
        *(f"train_{key}" for key in MODEL_KEYS),
        *(f"oos_{key}" for key in MODEL_KEYS),
        "overall_cagr_pct", "overall_sharpe", "overall_calmar", "overall_maxdd_pct",
        "overall_trades", "overall_winrate_pct",
    ]
    columns = [
        "arm", "model_type", "model_version", "seq_len",
        "trigger_stride", "direction_stride", "mode", "period",
        "oos_months", "status", *metric_columns,
        "equity_path", "train_dir", "backtest_dir", "error",
    ]
    csv_path = os.path.join(out_root, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in (
                "arm", "model_type", "model_version", "seq_len",
                "trigger_stride", "direction_stride", "mode", "period",
                "oos_months", "status",
            )}
            row.update(record.get("metrics") or {})
            row.update({
                "equity_path": record.get("equity_path"),
                "train_dir": record["train_dir"],
                "backtest_dir": record["backtest_dir"],
                "error": record.get("error"),
            })
            writer.writerow(row)

    md_path = os.path.join(out_root, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Trigger+Direction fusion model comparison\n\n")
        handle.write(
            "| arm | model | OOS CAGR | OOS Sharpe | OOS MaxDD | OOS trades | "
            "OOS accuracy | OOS acc-baseline | OOS macro F1 | overall CAGR |\n"
        )
        handle.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for record in records:
            m = record.get("metrics") or {}
            handle.write(
                "| {arm} | {model} | {oos_cagr}% | {oos_sharpe} | {oos_dd}% | "
                "{oos_trades} | {oos_acc} | {oos_lift} | {oos_f1} | {all_cagr}% |\n".format(
                    arm=record["arm"],
                    model=f"{record['model_type']}_v{record['model_version']}",
                    oos_cagr=m.get("oos_cagr_pct"),
                    oos_sharpe=m.get("oos_sharpe"),
                    oos_dd=m.get("oos_maxdd_pct"),
                    oos_trades=m.get("oos_trades"),
                    oos_acc=m.get("oos_accuracy"),
                    oos_lift=m.get("oos_accuracy_minus_baseline"),
                    oos_f1=m.get("oos_macro_f1"),
                    all_cagr=m.get("overall_cagr_pct"),
                )
            )
    return json_path, csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--data-dir", default=SHARED_DATA_DIR)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--split-time", default=None,
                        help="Only used when the data dir has to be prepared")
    parser.add_argument("--mode", choices=("expanding", "sliding"), default="expanding")
    parser.add_argument("--period", default="development")
    parser.add_argument("--oos-months", type=int, default=1,
                        help="OOS window length; raise it only for smoke tests")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epoch budget for torch arms (xgboost ignores it)")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError(f"Unsupported arms: {unknown}; choose from {ARMS}")

    split_time = args.split_time or _reference_split_time()
    os.makedirs(args.out_root, exist_ok=True)
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(args.out_root, "experiment.log"),
        console_level=logging.INFO,
        file_level=logging.INFO,
    )
    data_dir = ensure_data(args.data_dir, split_time, logger)

    print(
        f"fusion model comparison | arms={arms} | data={data_dir} | "
        f"mode={args.mode} period={args.period} oos_months={args.oos_months} "
        f"seq_len={args.seq_len}"
    )

    records = []
    for arm in arms:
        started = time.perf_counter()
        record = run_arm(
            arm,
            data_dir=data_dir,
            out_root=args.out_root,
            mode=args.mode,
            period=args.period,
            oos_months=args.oos_months,
            seq_len=args.seq_len,
            epochs=args.epochs,
            patience=args.patience,
            stop_on_error=args.stop_on_error,
        )
        record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        records.append(record)
        print(f"[{arm}] {record['status']} ({record['elapsed_seconds']}s)")
        write_summary(records, args.out_root)

    json_path, csv_path, md_path = write_summary(records, args.out_root)
    ok = sum(1 for record in records if record["status"] == "ok")
    print(f"\n{ok}/{len(records)} arms succeeded.")
    print(f"Summary JSON : {json_path}")
    print(f"Summary CSV  : {csv_path}")
    print(f"Comparison   : {md_path}")


def _reference_split_time() -> str:
    """Same split boundary as the existing multi-symbol sweep."""
    import pandas as pd

    source = replace(DOGE_15m_TBM, symbols=(TARGET_SYMBOL,))
    path = common.market_data_path(source)
    times = pd.to_datetime(
        pd.read_csv(path, usecols=["open_time_ms_utc"])["open_time_ms_utc"],
        unit="ms",
        utc=True,
    )
    return (times.max() - pd.DateOffset(months=3)).isoformat()


if __name__ == "__main__":
    main()
