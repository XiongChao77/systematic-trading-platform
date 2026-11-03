#!/usr/bin/env python3
"""Single-asset baselines for the DOGE/AAVE/XRP/ETH/ADA sweep.

For each requested symbol, run the full pipeline trained and backtested on
that symbol ALONE (data prep -> rolling train -> backtest), then emit:
  * `solo_summary.csv` / `.json`   -- the solo baselines
  * `solo_vs_joint.csv`            -- solo vs every joint combo that
                                      contains the same target symbol
                                      (read from matrix_summary.json)

This isolates "joint training dilutes the asset" from "the asset is simply
not tradeable under this configuration".

Usage:
    python experiment/single_asset_baseline.py [--mode expanding|sliding]
                                               [--period long|oos|forward]
                                               [--symbols AAVEUSDT,XRPUSDT,...]
                                               [--stop-on-error]
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

from multi_symbol_sweep import (
    SWEEP_ROOT,
    check_symbols,
    parse_backtest_metrics,
    reference_split_time,
)

DEFAULT_SOLO = ("AAVEUSDT", "XRPUSDT", "ETHUSDT", "ADAUSDT")


def solo_tag(symbol: str) -> str:
    return f"solo_{symbol.replace('USDT', '').lower()}"


def run_solo(symbol: str, *, mode: str, period: str, split_time: str, stop_on_error: bool) -> dict:
    tag = solo_tag(symbol)
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
        logger.info("=== [%s] solo symbol=%s mode=%s period=%s ===", tag, symbol, mode, period)
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

        # 2. Rolling training on the single symbol.
        logger.info("=== [%s] Training (mode=%s) ===", tag, mode)
        train_rolling(
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


def write_solo_summary(records: list[dict]) -> tuple[str, str]:
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    json_path = os.path.join(SWEEP_ROOT, "solo_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    columns = [
        "tag", "symbol", "mode", "period", "split_time", "status",
        "cagr_pct", "calmar", "sharpe", "maxdd_pct", "grossret_pct",
        "trades", "winrate_pct", "equity_path", "data_dir", "train_dir",
        "backtest_dir", "error",
    ]
    csv_path = os.path.join(SWEEP_ROOT, "solo_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics") or {}
            row = {
                "tag": record["tag"],
                "symbol": record["symbol"],
                "mode": record["mode"],
                "period": record["period"],
                "split_time": record["split_time"],
                "status": record["status"],
                "equity_path": record.get("equity_path"),
                "data_dir": record["data_dir"],
                "train_dir": record["train_dir"],
                "backtest_dir": record["backtest_dir"],
                "error": record.get("error"),
            }
            for key in ("cagr_pct", "calmar", "sharpe", "maxdd_pct",
                        "grossret_pct", "trades", "winrate_pct"):
                row[key] = metrics.get(key)
            writer.writerow(row)
    return json_path, csv_path


def write_solo_vs_joint(solo_records: list[dict]) -> str:
    """Tidy long table: one row per (target symbol, source model)."""
    joint_path = os.path.join(SWEEP_ROOT, "matrix_summary.json")
    rows = []
    metric_keys = ("cagr_pct", "calmar", "sharpe", "maxdd_pct",
                   "grossret_pct", "trades", "winrate_pct")
    if os.path.isfile(joint_path):
        with open(joint_path, encoding="utf-8") as handle:
            for record in json.load(handle):
                if record.get("status") != "ok":
                    continue
                metrics = record.get("metrics") or {}
                combo = record["combo_symbols"]
                row = {
                    "target_symbol": record["target_symbol"],
                    "kind": "solo" if len(combo) == 1 else "joint",
                    "source": record["tag"],
                    "combo_symbols": "|".join(combo),
                }
                for key in metric_keys:
                    row[key] = metrics.get(key)
                rows.append(row)
    for record in solo_records:
        if record.get("status") != "ok":
            continue
        metrics = record.get("metrics") or {}
        row = {
            "target_symbol": record["symbol"],
            "kind": "solo",
            "source": record["tag"],
            "combo_symbols": record["symbol"],
        }
        for key in metric_keys:
            row[key] = metrics.get(key)
        rows.append(row)

    rows.sort(key=lambda row: (row["target_symbol"], row["kind"], row["source"]))
    columns = ["target_symbol", "kind", "source", "combo_symbols",
               "cagr_pct", "calmar", "sharpe", "maxdd_pct", "grossret_pct",
               "trades", "winrate_pct"]
    csv_path = os.path.join(SWEEP_ROOT, "solo_vs_joint.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("expanding", "sliding"), default="expanding")
    parser.add_argument("--period", default="long")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SOLO))
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    symbols = tuple(sym.strip() for sym in args.symbols.split(",") if sym.strip())
    split_time = reference_split_time()
    print(f"Solo baseline split_time={split_time} mode={args.mode} period={args.period}")

    records = []
    for symbol in symbols:
        started = time.perf_counter()
        record = run_solo(
            symbol,
            mode=args.mode,
            period=args.period,
            split_time=split_time,
            stop_on_error=args.stop_on_error,
        )
        record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        records.append(record)
        metrics = record.get("metrics") or {}
        print(
            "[%s] -> %s | CAGR=%s Sharpe=%s MaxDD=%s (%.1fs)"
            % (
                record["tag"], record["status"],
                metrics.get("cagr_pct"), metrics.get("sharpe"),
                metrics.get("maxdd_pct"), record["elapsed_seconds"],
            )
        )

    json_path, csv_path = write_solo_summary(records)
    comparison_path = write_solo_vs_joint(records)
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n{ok}/{len(records)} solo baselines succeeded.")
    print(f"Solo JSON      : {json_path}")
    print(f"Solo CSV       : {csv_path}")
    print(f"Solo vs joint  : {comparison_path}")


if __name__ == "__main__":
    main()
