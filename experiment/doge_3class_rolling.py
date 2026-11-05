#!/usr/bin/env python3
"""DOGE single-asset DIRECT_3CLASS rolling training + backtest.

Runs the full pipeline for a SINGLE-HEAD three-class model on one symbol:
  data prep -> rolling train (configs=[DIRECT_3CLASS]) -> backtest -> metrics.

This is the 3-class counterpart of the existing binary fusion baseline
(Trigger + Direction) produced by multi_symbol_sweep.py's `doge` combo, so
the two can be compared on the same split boundary.

Because `train_rolling`'s CLI hardcodes the Trigger+Direction fusion heads,
the three-class config must be passed explicitly here.

Outputs (under experiment/sweep_outputs/<tag>_3class/):
  data/ train/ backtest/ experiment.log
  summary.json / summary.csv             -- this run's metrics
  <tag>_3class_vs_fusion.csv             -- 3-class vs binary fusion

Usage:
    python experiment/doge_3class_rolling.py [--symbol DOGEUSDT]
                                             [--mode expanding|sliding]
                                             [--period long|oos|forward]
                                             [--seq-len 32] [--stride 4]
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
from model.train import train_rolling
from model.train_config import (
    RollingConfig,
    TrainConfig,
    TrainTask,
    XGBoostConfig,
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

from multi_symbol_sweep import (
    SWEEP_ROOT,
    check_symbols,
    parse_backtest_metrics,
    reference_split_time,
)

METRIC_KEYS = (
    "cagr_pct", "calmar", "sharpe", "maxdd_pct",
    "grossret_pct", "trades", "winrate_pct",
)


def three_class_tag(symbol: str) -> str:
    return f"{symbol.replace('USDT', '').lower()}_3class"


def build_three_class_config(seq_len: int, stride: int) -> TrainConfig:
    """Single-head XGBoost DIRECT_3CLASS config (mirrors TriggerModelLogistic)."""
    return TrainConfig(
        model_cfg=XGBoostConfig(seq_len=seq_len),
        feature_conf_list=None,
        train_task=TrainTask.DIRECT_3CLASS,
        stride=stride,
    )


def run(symbol: str, *, mode: str, period: str, split_time: str,
        seq_len: int, stride: int, stop_on_error: bool) -> dict:
    tag = three_class_tag(symbol)
    base_dir = os.path.join(SWEEP_ROOT, tag)
    data_dir = os.path.join(base_dir, "data")
    train_dir = os.path.join(base_dir, "train")
    backtest_dir = os.path.join(base_dir, "backtest")
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    record = {
        "tag": tag,
        "symbol": symbol,
        "task": TrainTask.DIRECT_3CLASS,
        "mode": mode,
        "period": period,
        "split_time": split_time,
        "seq_len": seq_len,
        "stride": stride,
        "status": "ok",
        "data_dir": data_dir,
        "train_dir": train_dir,
        "backtest_dir": backtest_dir,
    }

    logger, log_path = common.setup_session_logger(
        log_file_path=os.path.join(base_dir, "experiment.log"),
        console_level=logging.INFO,
        file_level=logging.INFO,
    )

    try:
        logger.info(
            "=== [%s] DIRECT_3CLASS symbol=%s mode=%s period=%s seq_len=%s stride=%s ===",
            tag, symbol, mode, period, seq_len, stride,
        )
        check_symbols((symbol,))

        # 1. Prepare single-symbol data at the shared split boundary.
        para = replace(DOGE_15m_TBM, symbols=(symbol,))
        logger.info("=== [%s] Preparing data ===", tag)
        preparation.main(
            logger,
            common.FEATURE_GROUP_LIST,
            [],
            para=para,
            prep_output_dir=data_dir,
            split_time=split_time,
        )

        # 2. Rolling training with a single three-class head (no fusion).
        logger.info("=== [%s] Training (mode=%s) ===", tag, mode)
        train_rolling(
            configs=[build_three_class_config(seq_len, stride)],
            rolling_config=RollingConfig(mode=mode),
            prep_output_dir=data_dir,
            save_dir=train_dir,
            logger=logger,
        )

        # 3. Backtest the same symbol.
        logger.info("=== [%s] Backtesting %s ===", tag, symbol)
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
                symbol=symbol,
            ),
        )
        result = run_backtest(logger, config, period=period, evaluate_stages=False)
        equity_path = plot_equity_curves(
            result,
            backtest_dir,
            file_name="equity_curve.png",
            normalize_equity=False,
            include_oos=False,
            logger=logger,
        )
        record["equity_path"] = equity_path
        record["metrics"] = parse_backtest_metrics(log_path)
        logger.info("=== [%s] DONE ===", tag)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("=== [%s] FAILED ===", tag)
        if stop_on_error:
            raise
    return record


def write_summary(record: dict) -> tuple[str, str]:
    base_dir = os.path.dirname(record["backtest_dir"])
    json_path = os.path.join(base_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)

    columns = [
        "tag", "symbol", "task", "mode", "period", "split_time",
        "seq_len", "stride", "status", *METRIC_KEYS,
        "equity_path", "data_dir", "train_dir", "backtest_dir", "error",
    ]
    csv_path = os.path.join(base_dir, "summary.csv")
    metrics = record.get("metrics") or {}
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        row = {key: record.get(key) for key in (
            "tag", "symbol", "task", "mode", "period", "split_time",
            "seq_len", "stride", "status",
        )}
        row.update({key: metrics.get(key) for key in METRIC_KEYS})
        row.update({
            "equity_path": record.get("equity_path"),
            "data_dir": record["data_dir"],
            "train_dir": record["train_dir"],
            "backtest_dir": record["backtest_dir"],
            "error": record.get("error"),
        })
        writer.writerow(row)
    return json_path, csv_path


def write_comparison(record: dict) -> str:
    """Compare this 3-class run against the binary fusion `doge` combo."""
    base_dir = os.path.dirname(record["backtest_dir"])
    three_metrics = record.get("metrics") or {}
    rows = []
    joint_path = os.path.join(SWEEP_ROOT, "matrix_summary.json")
    if os.path.isfile(joint_path):
        with open(joint_path, encoding="utf-8") as handle:
            for entry in json.load(handle):
                if (
                    entry.get("status") == "ok"
                    and entry.get("tag") == "doge"
                    and entry.get("target_symbol") == record["symbol"]
                ):
                    metrics = entry.get("metrics") or {}
                    rows.append({
                        "kind": "binary_fusion(doge)",
                        "task": "TRIGGER+DIRECTION",
                        **{key: metrics.get(key) for key in METRIC_KEYS},
                    })
                    break
    rows.append({
        "kind": "three_class",
        "task": TrainTask.DIRECT_3CLASS,
        **{key: three_metrics.get(key) for key in METRIC_KEYS},
    })

    columns = ["kind", "task", *METRIC_KEYS]
    csv_path = os.path.join(base_dir, f"{record['tag']}_vs_fusion.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--mode", choices=("expanding", "sliding"), default="expanding")
    parser.add_argument("--period", default="long")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    split_time = reference_split_time()
    print(
        f"DIRECT_3CLASS symbol={args.symbol} split_time={split_time} "
        f"mode={args.mode} period={args.period} seq_len={args.seq_len} stride={args.stride}"
    )

    started = time.perf_counter()
    record = run(
        args.symbol,
        mode=args.mode,
        period=args.period,
        split_time=split_time,
        seq_len=args.seq_len,
        stride=args.stride,
        stop_on_error=args.stop_on_error,
    )
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    metrics = record.get("metrics") or {}
    print(
        "[%s] -> %s | CAGR=%s Sharpe=%s MaxDD=%s (%.1fs)"
        % (
            record["tag"], record["status"],
            metrics.get("cagr_pct"), metrics.get("sharpe"),
            metrics.get("maxdd_pct"), record["elapsed_seconds"],
        )
    )

    json_path, csv_path = write_summary(record)
    comparison_path = write_comparison(record)
    print(f"Summary JSON      : {json_path}")
    print(f"Summary CSV       : {csv_path}")
    print(f"3class vs fusion  : {comparison_path}")


if __name__ == "__main__":
    main()
