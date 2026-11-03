#!/usr/bin/env python3
"""Multi-symbol joint-training sweep.

Runs, for a cumulative sequence of symbol sets, the full pipeline:
    data preparation -> rolling train -> DOGE backtest -> equity curve.

Each combo keeps its own prep/train/backtest directories under
`experiment/sweep_outputs/<tag>/` so results can be compared later. A compact
`summary.csv` and `summary.json` are written at the sweep root.

Usage:
    python experiment/multi_symbol_sweep.py [--mode expanding|sliding]
                                            [--period long|oos|forward]
                                            [--split-time <iso>]
                                            [--stop-on-error]
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import replace

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data_process import common
from data_process import preparation
from data_process.config import DOGE_15m_TBM
from model.train import train_rolling
from model.train_config import RollingConfig
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
TARGET_SYMBOL = "DOGEUSDT"

# Cumulative symbol sets. "xml" in the request maps to XRPUSDT.
COMBOS = [
    ("DOGEUSDT",),
    ("DOGEUSDT", "AAVEUSDT"),
    ("DOGEUSDT", "AAVEUSDT", "XRPUSDT"),
    ("DOGEUSDT", "AAVEUSDT", "XRPUSDT", "ETHUSDT"),
    ("DOGEUSDT", "AAVEUSDT", "XRPUSDT", "ETHUSDT", "ADAUSDT"),
]


def combo_tag(symbols) -> str:
    return "_".join(symbol.replace("USDT", "").lower() for symbol in symbols)


def reference_split_time() -> str:
    """Single authoritative split so every combo shares the same OOS boundary."""
    source = replace(DOGE_15m_TBM, symbols=("DOGEUSDT",))
    path = common.market_data_path(source)
    times = pd.to_datetime(
        pd.read_csv(path, usecols=["open_time_ms_utc"])["open_time_ms_utc"],
        unit="ms",
        utc=True,
    )
    return (times.max() - pd.DateOffset(months=3)).isoformat()


def check_symbols(symbols) -> None:
    missing = []
    for symbol in symbols:
        source = replace(DOGE_15m_TBM, symbols=(symbol,))
        if not os.path.isfile(common.market_data_path(source)):
            missing.append(symbol)
    if missing:
        raise FileNotFoundError(
            f"Missing market data for symbols: {missing}; "
            f"expected under {common.PROJECT_DATA_DIR}"
        )


_METRIC_PATTERNS = {
    "cagr_pct": r"CAGR:\s*(-?[\d.]+)%",
    "calmar": r"Calmar:\s*(-?[\d.]+)",
    "sharpe": r"Sharpe:\s*(-?[\d.]+)",
    "maxdd_pct": r"MaxDD:\s*([\d.]+)%",
    "grossret_pct": r"GrossRet:\s*(-?[\d.]+)%",
    "trades": r"TRADES\s*\|\s*Total:\s*(\d+)",
    "winrate_pct": r"WinRate:\s*([\d.]+)%",
}


def parse_backtest_metrics(log_path: str) -> dict:
    if not os.path.isfile(log_path):
        return {}
    text = open(log_path, encoding="utf-8", errors="ignore").read()
    metrics = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            metrics[key] = int(value) if key == "trades" else value
    return metrics


def run_combo(symbols, *, mode: str, period: str, split_time: str, stop_on_error: bool) -> dict:
    tag = combo_tag(symbols)
    base_dir = os.path.join(SWEEP_ROOT, tag)
    data_dir = os.path.join(base_dir, "data")
    train_dir = os.path.join(base_dir, "train")
    backtest_dir = os.path.join(base_dir, "backtest")
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    record = {
        "tag": tag,
        "symbols": list(symbols),
        "mode": mode,
        "period": period,
        "split_time": split_time,
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
        logger.info("=== [%s] symbols=%s mode=%s period=%s ===", tag, list(symbols), mode, period)
        check_symbols(symbols)

        # 1. Prepare (single-symbol path or multi-symbol joint path).
        para = replace(DOGE_15m_TBM, symbols=symbols)
        logger.info("=== [%s] Preparing data ===", tag)
        preparation.main(
            logger,
            common.FEATURE_GROUP_LIST,
            [],
            para=para,
            prep_output_dir=data_dir,
            split_time=split_time,
        )

        # 2. Rolling training (expanding or sliding).
        logger.info("=== [%s] Training (mode=%s) ===", tag, mode)
        train_rolling(
            rolling_config=RollingConfig(mode=mode),
            prep_output_dir=data_dir,
            save_dir=train_dir,
            logger=logger,
        )

        # 3. Backtest DOGE.
        logger.info("=== [%s] Backtesting %s ===", tag, TARGET_SYMBOL)
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


def write_summary(records: list[dict]) -> tuple[str, str]:
    os.makedirs(SWEEP_ROOT, exist_ok=True)

    json_path = os.path.join(SWEEP_ROOT, "summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    flat_columns = [
        "tag", "symbols", "mode", "period", "split_time", "status",
        "cagr_pct", "sharpe", "maxdd_pct", "grossret_pct", "trades",
        "winrate_pct", "equity_path", "data_dir", "train_dir", "backtest_dir",
        "error",
    ]
    csv_path = os.path.join(SWEEP_ROOT, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_columns)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics") or {}
            row = {
                "tag": record["tag"],
                "symbols": "|".join(record["symbols"]),
                "mode": record["mode"],
                "period": record["period"],
                "split_time": record["split_time"],
                "status": record["status"],
                "cagr_pct": metrics.get("cagr_pct"),
                "sharpe": metrics.get("sharpe"),
                "maxdd_pct": metrics.get("maxdd_pct"),
                "grossret_pct": metrics.get("grossret_pct"),
                "trades": metrics.get("trades"),
                "winrate_pct": metrics.get("winrate_pct"),
                "equity_path": record.get("equity_path"),
                "data_dir": record["data_dir"],
                "train_dir": record["train_dir"],
                "backtest_dir": record["backtest_dir"],
                "error": record.get("error"),
            }
            writer.writerow(row)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("expanding", "sliding"), default="expanding")
    parser.add_argument("--period", default="long")
    parser.add_argument("--split-time", default=None, help="ISO split time; defaults to DOGE end minus 3 months")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    split_time = args.split_time or reference_split_time()
    print(f"Sweep split_time={split_time} mode={args.mode} period={args.period}")

    records = []
    for symbols in COMBOS:
        started = time.perf_counter()
        record = run_combo(
            symbols,
            mode=args.mode,
            period=args.period,
            split_time=split_time,
            stop_on_error=args.stop_on_error,
        )
        record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        records.append(record)
        print(
            "[%s] %s -> %s (%.1fs)"
            % (record["tag"], list(record["symbols"]), record["status"], record["elapsed_seconds"])
        )

    json_path, csv_path = write_summary(records)
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n{ok}/{len(records)} combos succeeded.")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV : {csv_path}")


if __name__ == "__main__":
    main()