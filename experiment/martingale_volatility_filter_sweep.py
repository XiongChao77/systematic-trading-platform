#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expected profit of the 1% martingale ladder under causal volatility filters.

Every combo replays the full history once (``run_once`` from bar 0) with the
same account settings, so only the filter changes. The gate is built by
``build_open_gate``: bar ``i`` may open a ladder only if the annualized
volatility measured through bar ``i-1`` sits inside the requested band. No
statistic of the filter reads the bar it decides on, and the thresholds are
absolute numbers rather than full-sample quantiles, which would leak the future.

An open ladder is never interrupted by the filter: it only decides whether a
flat account may start a new cycle.

Usage:
    python experiment/martingale_volatility_filter_sweep.py
    python experiment/martingale_volatility_filter_sweep.py --layers 2 6 --workers 2
    python experiment/martingale_volatility_filter_sweep.py --no-intrabar --out /tmp/sweep

Each worker keeps its own 5m + 1m bar cache, so memory scales with
``--workers``. Four workers on DOGE 5m is a comfortable default.
"""

from __future__ import annotations

import argparse
import numpy as np
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import pandas as pd

from data_process import common
from trade.sim import martingale_backtest as mb
from trade.sim.martingale_engine import AccountParams, MartingaleParams

# ---- experiment grid ------------------------------------------------------
# Ladder: equal 1% spacing, 1% cash take profit, same settings as the
# intrabar scan that produced the breach tables.
LAYERS = (2, 3, 4, 5, 6)
GRID_DEVIATION_PCT = 0.01
TAKE_PROFIT_PCT = 0.01
VOL_WINDOW_HOURS = 4.0
# Upper bound only: skip opening when trailing annualized vol exceeds this.
MAX_ONLY_PCTS = (40.0, 60.0, 80.0, 110.0, 140.0)
# Two-sided bands (min, max), annualized percent.
BAND_PCTS = ((30.0, 80.0), (45.0, 110.0), (60.0, 140.0))

INITIAL_EQUITY = 10_000.0
LEVERAGE = 100.0
MARGIN_CAP = 0.90
RUIN_PCT = 0.40

_WORKER_BARS: "mb.BarSeries | None" = None


def _data_params() -> mb.DataParams:
    return mb.DataParams(
        market_category=common.DOGE_5m.market_category,
        data_source=common.DOGE_5m.data_source,
        symbols=common.DOGE_5m.symbols,
        interval=common.DOGE_5m.interval,
        trading_type=common.DOGE_5m.trading_type,
    )


def _filter_specs() -> list[tuple[str, "mb.VolatilityFilterParams | None"]]:
    """Baseline first, then one-sided caps, then bands."""
    specs: list[tuple[str, "mb.VolatilityFilterParams | None"]] = [("none", None)]
    for pct in MAX_ONLY_PCTS:
        specs.append((
            f"max{int(pct)}",
            mb.VolatilityFilterParams(
                window_hours=VOL_WINDOW_HOURS, max_annualized_pct=pct
            ),
        ))
    for low, high in BAND_PCTS:
        specs.append((
            f"{int(low)}-{int(high)}",
            mb.VolatilityFilterParams(
                window_hours=VOL_WINDOW_HOURS,
                min_annualized_pct=low,
                max_annualized_pct=high,
            ),
        ))
    return specs


def _config(layers: int, volatility_filter) -> mb.BacktestConfig:
    return mb.BacktestConfig(
        strategy=MartingaleParams(
            grid_deviation_pcts=[GRID_DEVIATION_PCT] * layers,
            take_profit_pct=TAKE_PROFIT_PCT,
        ),
        account=AccountParams(
            initial_equity=INITIAL_EQUITY,
            leverage=LEVERAGE,
            margin_usage_cap_pct=MARGIN_CAP,
            ruin_equity_pct=RUIN_PCT,
            taker_fee_pct=0.0,
            maker_fee_pct=0.0,
        ),
        capital=mb.CapitalParams(
            profit_handling="withdraw",
            grid_break_is_ruin=True,
        ),
        data=_data_params(),
        volatility_filter=volatility_filter,
    )


def _init_worker(data_kwargs: dict, intrabar_interval: "str | None") -> None:
    global _WORKER_BARS
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("vol-filter-sweep")
    _WORKER_BARS = mb.load_bars(
        logger, mb.DataParams(**data_kwargs), intrabar_interval=intrabar_interval
    )
    main_interval_ms = int(
        np.median(np.diff(_WORKER_BARS.time)) / np.timedelta64(1, "ms")
    )
    fine_interval_ms = common.get_interval_ms(intrabar_interval or "1m")
    _WORKER_BARS.intrabar_step_ms = fine_interval_ms
    _WORKER_BARS.agg_trade_prices = common.AggTradeResolver(
        main_interval_ms, fine_interval_ms
    )


def _metrics(
    result: dict,
    bars: "mb.BarSeries",
    layers: int,
    label: str,
    volatility_filter,
    extra: dict,
) -> dict:
    summary = result["summary"]
    trades = result["trades"]
    total_bars = len(bars)

    take_profit = [t for t in trades if t["reason"] == "take_profit"]
    grid_break = [t for t in trades if t["reason"] == "grid_break"]
    others = len(trades) - len(take_profit) - len(grid_break)
    tp_pnl = sum(t["net_pnl"] for t in take_profit)
    break_pnl = sum(t["net_pnl"] for t in grid_break)
    net_pnl = sum(t["net_pnl"] for t in trades)
    cycles = len(trades)
    bars_in_market = sum(int(t.get("bars_held", 0)) for t in trades)

    span_seconds = (
        pd.Timestamp(bars.time[-1]) - pd.Timestamp(bars.time[0])
    ).total_seconds()
    years = max(span_seconds / (365.25 * 24.0 * 3600.0), 1e-9)
    days = years * 365.25

    resolved = len(take_profit) + len(grid_break)
    return {
        "layers": layers,
        "filter": label,
        "vol_window_hours": (
            None if volatility_filter is None else volatility_filter.window_hours
        ),
        "min_vol_pct": (
            None if volatility_filter is None else volatility_filter.min_annualized_pct
        ),
        "max_vol_pct": (
            None if volatility_filter is None else volatility_filter.max_annualized_pct
        ),
        "intrabar_interval": extra["intrabar_interval"],
        "bars": total_bars,
        "years": years,
        "cycles": cycles,
        "take_profit_cycles": len(take_profit),
        "grid_break_cycles": len(grid_break),
        "other_exit_cycles": others,
        "take_profit_rate": (
            len(take_profit) / resolved if resolved else float("nan")
        ),
        "tp_pnl_total": tp_pnl,
        "grid_break_pnl_total": break_pnl,
        "net_profit_total": net_pnl,
        "ev_per_cycle": net_pnl / cycles if cycles else float("nan"),
        "cycles_per_day": cycles / days if days > 0.0 else float("nan"),
        "net_profit_per_day": net_pnl / days if days > 0.0 else float("nan"),
        "net_profit_per_year": net_pnl / years,
        "exposure_pct": (
            bars_in_market / total_bars * 100.0 if total_bars else float("nan")
        ),
        "volatility_blocked_pct": (
            summary.get("volatility_blocked_bars", 0) / total_bars * 100.0
            if total_bars else float("nan")
        ),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "sharpe": summary.get("sharpe"),
        "final_total_wealth": summary.get("final_total_wealth"),
    }


def _run_combo(payload: tuple[int, str, int]) -> dict:
    layers, label, index = payload
    volatility_filter = _filter_specs_by_label()[label]
    bars = _WORKER_BARS
    started = time.time()
    result = mb.run_once(bars, _config(layers, volatility_filter))
    row = _metrics(
        result,
        bars,
        layers,
        label,
        volatility_filter,
        {"intrabar_interval": None if bars.intrabar is None else "1m"},
    )
    row["order"] = index
    row["seconds"] = time.time() - started
    return row


def _filter_specs_by_label() -> dict:
    return {label: params for label, params in _filter_specs()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=list(LAYERS),
        help="ladder depths to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="parallel processes; each keeps its own bar cache (default: 4)",
    )
    parser.add_argument(
        "--no-intrabar", action="store_true",
        help="disable the 1m ambiguity resolution (faster, unresolved paths raise an error)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output directory (default: quant_output/martingale_vol_filter/<timestamp>)",
    )
    parser.add_argument(
        "--show", action="store_true", help="print the per-combo table",
    )
    return parser.parse_args()


def _default_out_dir() -> str:
    stamp = time.strftime("%Y-%m-%d/%H_%M_%S")
    return os.path.join(
        common.PERSISTENCE_DIR, "martingale_vol_filter", stamp
    )


def main() -> None:
    args = _parse_args()
    out_dir = args.out or _default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(out_dir, "sweep.log"), mode="w"),
        ],
    )
    logger = logging.getLogger("vol-filter-sweep")
    logger.info(
        f"Sweep start | layers {args.layers} | workers {args.workers} | "
        f"intrabar {'off' if args.no_intrabar else '1m'} | out {out_dir}"
    )

    intrabar_interval = None if args.no_intrabar else "1m"
    combos = []
    index = 0
    for layers in args.layers:
        for label, _ in _filter_specs():
            combos.append((layers, label, index))
            index += 1

    rows: list[dict] = []
    payload = asdict(_data_params())
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(payload, intrabar_interval),
    ) as pool:
        futures = {pool.submit(_run_combo, combo): combo for combo in combos}
        for done, future in enumerate(as_completed(futures), start=1):
            layers, label, _ = futures[future]
            try:
                row = future.result()
            except Exception as error:  # noqa: BLE001 - keep the sweep alive
                logger.exception(f"{layers}L {label} failed")
                row = {"layers": layers, "filter": label, "error": str(error)}
            rows.append(row)
            if "error" not in row:
                logger.info(
                    f"[{done}/{len(combos)}] {layers}L {label:<10} "
                    f"ev {row['ev_per_cycle']:.2f} | "
                    f"net {row['net_profit_total']:.0f} | "
                    f"cycles {row['cycles']} | {row['seconds']:.0f}s"
                )

    frame = pd.DataFrame(rows).sort_values(["layers", "order"])
    csv_path = os.path.join(out_dir, "vol_filter_sweep.csv")
    frame.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    valid = (
        frame[frame["error"].isna()] if "error" in frame.columns else frame
    )
    if len(valid):
        for value, title in (
            ("ev_per_cycle", "EV per cycle"),
            ("net_profit_total", "Total net profit"),
            ("net_profit_per_year", "Net profit per year"),
            ("cycles_per_day", "Cycles per day"),
            ("exposure_pct", "Exposure %"),
        ):
            table = valid.pivot(index="layers", columns="filter", values=value)
            logger.info(f"\n{title}\n{table.round(2).to_string()}")
        if args.show:
            print(valid.drop(columns=["order", "seconds"], errors="ignore").to_string())
    logger.info("Sweep done")


if __name__ == "__main__":
    main()
