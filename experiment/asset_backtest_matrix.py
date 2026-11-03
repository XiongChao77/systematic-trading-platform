#!/usr/bin/env python3
"""Per-asset backtest matrix.

For every already-trained combo under `experiment/sweep_outputs/<tag>/train`,
run the backtest on EACH constituent symbol (not only DOGE) and write a
matrix summary. Training artifacts are reused; nothing is retrained.

Rows: (combo tag, target symbol). Each backtest writes its own
`backtest_<SYMBOL>/` directory (report, log, equity curve) so every cell is
kept for later comparison.

Usage:
    python experiment/asset_backtest_matrix.py [--period long|oos|forward]
                                               [--stop-on-error]
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from data_process import common
from trade.runner.analyze_backtest_report import plot_equity_curves
from trade.runner.backtest_entry import main as run_backtest
from trade.runner.config import (
    BrokerConfig,
    ExperimentContext,
    ModelDataConfig,
    RunnerConfig,
)
from trade.strategy.strategy_bbm import BbmStrategyConfig

from multi_symbol_sweep import COMBOS, SWEEP_ROOT, combo_tag, parse_backtest_metrics


def _combo_paths(combo_symbols):
    tag = combo_tag(combo_symbols)
    base_dir = os.path.join(SWEEP_ROOT, tag)
    data_dir = os.path.join(base_dir, "data")
    train_dir = os.path.join(base_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Trained combo missing: {train_dir}. Run multi_symbol_sweep.py first."
        )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Prepared data missing: {data_dir}")
    return tag, base_dir, data_dir, train_dir


def run_cell(combo_symbols, target, *, period: str, stop_on_error: bool) -> dict:
    tag, base_dir, data_dir, train_dir = _combo_paths(combo_symbols)
    out_dir = os.path.join(base_dir, f"backtest_{target}")
    os.makedirs(out_dir, exist_ok=True)

    record = {
        "tag": tag,
        "combo_symbols": list(combo_symbols),
        "target_symbol": target,
        "period": period,
        "status": "ok",
        "backtest_dir": out_dir,
    }

    logger, log_path = common.setup_session_logger(
        log_file_path=os.path.join(out_dir, "experiment.log"),
        console_level=logging.INFO,
        file_level=logging.INFO,
    )

    try:
        prep = common.load_pre_params_from_dir(train_dir)
        config = RunnerConfig(
            strategy_config=BbmStrategyConfig(fixed_hold_bars=prep.predict_num),
            broker_config=BrokerConfig(),
            save_dir=out_dir,
            experiment_context=ExperimentContext(git_commit=common.git_revision()),
            data_config=ModelDataConfig(
                prep_output_dir=data_dir,
                train_output_dir=train_dir,
                device="auto",
                symbol=target,
            ),
        )
        logger.info(
            "=== [%s] backtest target=%s symbols=%s period=%s ===",
            tag, target, list(combo_symbols), period,
        )
        result = run_backtest(logger, config, period=period, evaluate_stages=False)
        equity_path = plot_equity_curves(
            result,
            out_dir,
            file_name="equity_curve.png",
            normalize_equity=False,
            include_oos=False,
            logger=logger,
        )
        record["equity_path"] = equity_path
        record["metrics"] = parse_backtest_metrics(log_path)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("=== [%s] target=%s FAILED ===", tag, target)
        if stop_on_error:
            raise
    return record


def write_matrix(records: list[dict]) -> tuple[str, str]:
    os.makedirs(SWEEP_ROOT, exist_ok=True)
    json_path = os.path.join(SWEEP_ROOT, "matrix_summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    columns = [
        "tag", "combo_symbols", "target_symbol", "period", "status",
        "cagr_pct", "calmar", "sharpe", "maxdd_pct", "grossret_pct",
        "trades", "winrate_pct", "equity_path", "backtest_dir", "error",
    ]
    csv_path = os.path.join(SWEEP_ROOT, "matrix_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics") or {}
            writer.writerow({
                "tag": record["tag"],
                "combo_symbols": "|".join(record["combo_symbols"]),
                "target_symbol": record["target_symbol"],
                "period": record["period"],
                "status": record["status"],
                "cagr_pct": metrics.get("cagr_pct"),
                "calmar": metrics.get("calmar"),
                "sharpe": metrics.get("sharpe"),
                "maxdd_pct": metrics.get("maxdd_pct"),
                "grossret_pct": metrics.get("grossret_pct"),
                "trades": metrics.get("trades"),
                "winrate_pct": metrics.get("winrate_pct"),
                "equity_path": record.get("equity_path"),
                "backtest_dir": record["backtest_dir"],
                "error": record.get("error"),
            })
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="long")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    records = []
    for combo_symbols in COMBOS:
        for target in combo_symbols:
            started = time.perf_counter()
            record = run_cell(
                combo_symbols, target,
                period=args.period, stop_on_error=args.stop_on_error,
            )
            record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
            records.append(record)
            metrics = record.get("metrics") or {}
            print(
                "[%s] target=%s -> %s | CAGR=%s Sharpe=%s MaxDD=%s (%.1fs)"
                % (
                    record["tag"], target, record["status"],
                    metrics.get("cagr_pct"), metrics.get("sharpe"),
                    metrics.get("maxdd_pct"), record["elapsed_seconds"],
                )
            )

    json_path, csv_path = write_matrix(records)
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n{ok}/{len(records)} cells succeeded.")
    print(f"Matrix JSON: {json_path}")
    print(f"Matrix CSV : {csv_path}")


if __name__ == "__main__":
    main()
