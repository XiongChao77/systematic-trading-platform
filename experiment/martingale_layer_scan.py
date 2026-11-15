"""Scan one martingale ladder per bar over the full DOGE history.

Each bar is an independent start. The scan keeps the outcome (take profit at
layer N, full-grid stop, or unresolved at the end of data) and, for stops, the
bar where the grid broke. The Monte Carlo path in martingale_backtest is
unchanged; this module is a separate experiment.

Outputs under ``--out``:

    cycles_<side>_<K>additions.csv          one row per start bar
    breach_positions_<side>_<K>additions.csv  full-grid exits
    scan_summary.json                        pooled statistics and timing
    manifest.json                            parameters, workers and elapsed time

Fee and slippage rates are decimal fractions, defaulting to zero. Funding is
not modeled. Returns use fixed initial equity; K excludes the base order.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_process import common
from trade.sim import martingale_backtest as mb
from trade.sim import martingale_filters as mf
from trade.sim import martingale_scan as ms

DEFAULT_OUT = str(Path(__file__).resolve().parents[1] / "output" / "martingale_scan" / "DOGEUSDT_5m")
DEFAULT_LAYERS = (2, 3, 4, 5, 6)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="Maximum additions K, excluding the base order",
    )
    parser.add_argument("--side", choices=("long", "short", "both"), default="long")
    parser.add_argument(
        "--step-pct",
        type=float,
        default=0.01,
        help="Absolute Fibonacci offset scale (default: 0.01)",
    )
    parser.add_argument(
        "--initial-take-profit-pct",
        type=float,
        default=0.01,
        help="Independent initial price target fraction (default: 0.01)",
    )
    descriptions = {
        "initial_equity": "Fixed equity used for profit targets and returns",
        "target_profit_pct": "Net profit target as a fraction of initial equity",
        "base_notional": "Base order notional; omitted means initial_equity",
        "max_nominal_leverage": "Startup limit on planned node notional / initial equity",
        "maker_fee_rate": "Resting order fee as a decimal fraction",
        "taker_fee_rate": "Market order fee as a decimal fraction",
        "base_slippage_rate": "Adverse base entry slippage fraction",
        "stop_slippage_rate": "Adverse market exit slippage fraction",
    }
    for name, value in asdict(ms.ScanParams()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=value,
            help=f"{descriptions[name]} (default: {value if value is not None else 'initial_equity'})",
        )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--intrabar-interval", default="1m")
    parser.add_argument("--no-intrabar", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--entry-filter",
        action="store_true",
        help="Block new ladders when any causal exclusion condition matches",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report-data", type=Path, help="Rebuild the interactive report data after scanning")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def _data_params() -> mb.DataParams:
    return mb.DataParams(
        market_category=common.DOGE_5m.market_category,
        data_source=common.DOGE_5m.data_source,
        symbols=common.DOGE_5m.symbols,
        interval=common.DOGE_5m.interval,
        trading_type=common.DOGE_5m.trading_type,
    )


def _as_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "label" in frame.columns:
        frame["label"] = pd.array(frame["label"], dtype="Int64")
    for column in ("start_time", "exit_time", "breach_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _time_breakdown(breaches: pd.DataFrame) -> dict:
    if breaches.empty:
        return {}
    times = pd.to_datetime(breaches["breach_time"], errors="coerce", utc=True)
    valid = times.notna()
    return {
        "by_hour_utc": times[valid].dt.hour.value_counts().sort_index().to_dict(),
        "by_weekday_utc": times[valid]
        .dt.dayofweek.value_counts()
        .sort_index()
        .to_dict(),
        "by_month": times[valid].dt.month.value_counts().sort_index().to_dict(),
        "by_year": times[valid].dt.year.value_counts().sort_index().to_dict(),
    }


def _start_time_breakdown(cycles: pd.DataFrame) -> dict:
    if cycles.empty:
        return {}
    times = pd.to_datetime(cycles["start_time"], errors="raise", utc=True)
    frame = cycles.assign(
        stopped=cycles["terminal"].eq(ms.GRID_STOP),
        unresolved=cycles["terminal"].eq(ms.UNRESOLVED),
        take_profit=cycles["terminal"].eq(ms.TAKE_PROFIT),
    )
    dimensions = {
        "hour": (times.dt.hour, range(24)),
        "weekday": (times.dt.dayofweek, range(7)),
        "day_of_month": (times.dt.day, range(1, 32)),
        "month": (times.dt.month, range(1, 13)),
    }
    result = {"timezone": "UTC", "groups": {}}
    for name, (keys, categories) in dimensions.items():
        grouped = frame.groupby(keys).agg(
            starts=("label", "size"),
            breaches=("stopped", "sum"),
            unresolved=("unresolved", "sum"),
            take_profit=("take_profit", "sum"),
            mean_net_pnl=("net_pnl", "mean"),
            mean_net_return=("net_return_pct", "mean"),
            mean_bars_held=("bars_held", "mean"),
        ).reindex(categories)
        records = []
        for key, row in grouped.iterrows():
            starts = int(row["starts"]) if pd.notna(row["starts"]) else 0
            record = {"key": int(key), "starts": starts}
            for field in ("breaches", "unresolved", "take_profit"):
                record[field] = int(row[field]) if starts else 0
            record["breach_rate"] = record["breaches"] / starts if starts else None
            for field in ("mean_net_pnl", "mean_net_return", "mean_bars_held"):
                record[field] = float(row[field]) if starts else None
            records.append(record)
        result["groups"][name] = records
    return result


def _summarize(
    cycles: pd.DataFrame, breaches: pd.DataFrame, layers: int, seconds: float
) -> dict:
    if cycles.empty:
        return {"layers": layers, "starts": 0, "seconds": seconds}
    label_counts = cycles["label"].value_counts().sort_index()
    total = int(len(cycles))
    resolved = int(cycles["terminal"].ne(ms.UNRESOLVED).sum())
    summary = {
        "layers": layers,
        "starts": total,
        "resolved": resolved,
        "unresolved": total - resolved,
        "take_profit": int((cycles["terminal"] == ms.TAKE_PROFIT).sum()),
        "grid_stop": int((cycles["terminal"] == ms.GRID_STOP).sum()),
        "exit_reasons": cycles["exit_reason"].value_counts().to_dict(),
        "return_basis": "net_pnl / initial_equity (decimal fraction)",
        "mean_net_pnl": float(cycles["net_pnl"].mean()),
        "label_counts": {int(key): int(value) for key, value in label_counts.items()},
        "label_share": {
            int(key): float(value / total) for key, value in label_counts.items()
        },
        "label_null_count": int(cycles["label"].isna().sum()),
        "label_status_counts": cycles["label_status"].value_counts().to_dict(),
        "risk_extrema": "unavailable: minimal label events; full-path extrema are NaN",
        "bars_held": {
            "mean": float(cycles["bars_held"].mean()),
            "median": float(cycles["bars_held"].median()),
            "p90": float(cycles["bars_held"].quantile(0.90)),
            "max": int(cycles["bars_held"].max()),
        },
        "mean_net_return_pct": float(cycles["net_return_pct"].mean()),
        "breach_count": int(len(breaches)),
        "breach_mean_drawdown_pct": (
            float(breaches["breach_drawdown_pct"].mean())
            if not breaches.empty
            else None
        ),
        "seconds": round(seconds, 2),
        "rows_per_second": round(total / seconds, 1) if seconds > 0 else None,
    }
    summary["start_time_breakdown"] = _start_time_breakdown(cycles)
    summary.update(_time_breakdown(breaches))
    return summary


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("martingale-scan")

    started_run = time.perf_counter()
    params = ms.ScanParams(
        **{key: getattr(args, key) for key in asdict(ms.ScanParams())}
    )
    sides = (
        (ms.LONG, ms.SHORT)
        if args.side == "both"
        else ((ms.LONG,) if args.side == "long" else (ms.SHORT,))
    )
    policies = [
        ms.FibonacciGrid(
            max_layers=k,
            side=side,
            step_pct=args.step_pct,
            params=params,
            initial_take_profit_pct=args.initial_take_profit_pct,
        )
        for side in sides
        for k in dict.fromkeys(args.layers)
    ]
    if args.chunk_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("chunk-size and limit must be positive")
    intrabar = None if args.no_intrabar else (args.intrabar_interval or None)
    logger.info(
        "Loading DOGE bars (intrabar=%s, workers=%s, limit=%s)",
        intrabar,
        args.workers,
        args.limit,
    )
    bars = mb.load_bars(logger, _data_params(), intrabar_interval=intrabar)
    main_interval_ms = int(np.median(np.diff(bars.time)) / np.timedelta64(1, "ms"))
    fine_interval_ms = common.get_interval_ms(intrabar or "1m")
    bars.intrabar_step_ms = fine_interval_ms
    bars.agg_trade_prices = common.AggTradeResolver(main_interval_ms, fine_interval_ms)
    bars.require_resolved_path = True
    logger.info(
        "Loaded %s bars; intrabar rows=%s",
        len(bars),
        len(bars.intrabar) if bars.intrabar is not None else 0,
    )

    stamp = datetime.now().strftime("%Y-%m-%d/%H_%M_%S_%f")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "scan.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)

    filter_params = mf.EntryFilterParams() if args.entry_filter else None
    gate = None if filter_params is None else mf.build_entry_gate(bars, filter_params)
    if gate is not None:
        logger.info(
            "Entry filter allows %s of %s bars (%.2f%%)",
            int(gate.sum()),
            len(gate),
            gate.sum() / len(gate) * 100,
        )

    label_metadata = {}
    side_price_params = {}
    side_rows = {}
    side_seconds = {}
    for side in sides:
        side_policies = [policy for policy in policies if policy.side == side]
        side_name = "long" if side == ms.LONG else "short"
        price_params = side_policies[0].price_params
        label_metadata[side_name] = common.martingale_metadata(bars, price_params)
        side_price_params[side_name] = asdict(price_params)
        logger.info("Scanning shared labels for %s (%s starts)", side_name, len(bars))
        started = time.perf_counter()
        side_rows[side] = ms.scan_policies(
            bars,
            side_policies,
            workers=args.workers,
            chunk_size=args.chunk_size,
            limit=args.limit,
            gate=gate,
        )
        side_seconds[side] = time.perf_counter() - started

    all_summaries = []
    for policy in policies:
        layers = policy.max_layers
        side_name = "long" if policy.side == ms.LONG else "short"
        rows = side_rows[policy.side].pop(layers)
        seconds = side_seconds[policy.side]
        cycles = _as_frame(rows)
        breaches = (
            cycles[cycles["terminal"] == ms.GRID_STOP].copy()
            if not cycles.empty
            else cycles.copy()
        )

        cycles.to_csv(
            out_dir / f"cycles_{side_name}_{layers}additions.csv", index=False
        )
        breaches.to_csv(
            out_dir / f"breach_positions_{side_name}_{layers}additions.csv", index=False
        )
        summary = _summarize(cycles, breaches, layers, seconds)
        summary.update(
            side=side_name,
            strategy=asdict(policy),
            timing_basis="shared side scan including all K projections",
            resolved_workers=ms.resolved_workers(
                args.workers, (len(rows) + args.chunk_size - 1) // args.chunk_size
            ),
        )
        all_summaries.append(summary)
        logger.info(
            "layers=%s starts=%s stop=%s unresolved=%s in %.1fs",
            layers,
            summary["starts"],
            summary.get("grid_stop", 0),
            summary.get("unresolved", 0),
            seconds,
        )
        if args.show:
            print(json.dumps(summary, indent=2, ensure_ascii=False))

    (out_dir / "scan_summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "strategy": "FibonacciGrid",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "account": asdict(params),
        "label_metadata": label_metadata,
        "side_price_params": side_price_params,
        "risk_extrema": "unavailable: minimal label events; full-path extrema are NaN",
        "funding_model": "not modeled",
        "rate_units": "decimal fractions",
        "return_basis": "net_pnl / fixed initial_equity",
        "base_notional_resolved": (
            params.initial_equity
            if params.base_notional is None
            else params.base_notional
        ),
        "bars": len(bars),
        "entry_filter": None if filter_params is None else asdict(filter_params),
        "entry_filter_allowed_bars": None if gate is None else int(gate.sum()),
        "elapsed_seconds": time.perf_counter() - started_run,
        "runs": all_summaries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "Fees maker=%s taker=%s; funding not modeled; summary saved to %s",
        params.maker_fee_rate,
        params.taker_fee_rate,
        out_dir / "scan_summary.json",
    )
    if args.report_data:
        from experiment.martingale_report import write_report_data

        logger.info("Building report data")
        write_report_data(bars, all_summaries, out_dir, args.report_data)
        logger.info("Report data saved to %s", args.report_data)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
