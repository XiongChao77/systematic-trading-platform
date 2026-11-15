"""Standalone backtest and Monte Carlo driver for the martingale simulator.

Reads raw OHLC straight from the canonical market-data CSV, replays it through
:mod:`trade.sim.martingale_engine` and aggregates the run statistics.  Nothing
here touches backtrader, the venue layer or the strategy layer.

Monte Carlo starts at ``runs`` random bars and trades each market path through
its configured horizon. A grid break is only a realized full-ladder exit unless
the capital rules classify it as ruin. Only ruin resets trading capital and
starts a new strategy-run segment on the same Monte Carlo path. Grid
infeasibility is unrecoverable and terminates that path.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", ".."))

from data_process import common
from trade.sim.martingale_engine import (
    SIDE_NAME,
    AccountParams,
    MartingaleParams,
    MartingaleSimulator,
)


# ============================================================
# Configuration
# ============================================================
@dataclass(kw_only=True)
class DataParams(common.MarketDataSourceConfig):
    """Which raw OHLC file to read, and which slice of it to use."""

    test_start_date: Optional[str] = None  # YYYY-MM-DD, inclusive
    test_end_date: Optional[str] = None    # YYYY-MM-DD, inclusive
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    root_dir: Optional[str] = None


@dataclass(frozen=True)
class MonteCarloParams:
    runs: int = 200
    seed: int = 42
    min_bars: int = 2_000          # bars that must remain ahead of a random start
    max_bars: Optional[int] = None  # None: run until ruin or end of data
    start_fraction: float = 0.30    # random starts are limited to the first 30%
    # Runs are fully independent, so they fan out over processes.  0 picks
    # cpu_count - 1, 1 stays in this process.  Start offsets are drawn in the
    # parent, so the result never depends on how many workers ran it.
    workers: int = 0

    def __post_init__(self):
        if not 0.0 < self.start_fraction <= 1.0:
            raise ValueError("start_fraction must be in (0, 1]")


@dataclass(frozen=True)
class CapitalParams:
    """How realized profits are handled during one Monte Carlo run.

    ``withdraw`` moves every positive closed-trade PnL to an isolated reserve,
    so the next grid continues with the same trading capital. ``compound``
    leaves profits in the trading account. ``both`` is accepted by the
    top-level driver and replays identical random starts once per mode.
    """

    profit_handling: str = "compound"  # "withdraw", "compound", or "both"
    stop_at_first_grid_breach: bool = False
    grid_break_is_ruin: bool = False
    double_target_multiple: float = 2.0

    def __post_init__(self):
        if self.profit_handling not in ("withdraw", "compound", "both"):
            raise ValueError(
                "profit_handling must be 'withdraw', 'compound', or 'both'"
            )
        if self.double_target_multiple <= 1.0:
            raise ValueError("double_target_multiple must be > 1")


@dataclass(frozen=True)
class VolatilityFilterParams:
    """Causal gate on new base orders, expressed in trailing annualized vol.

    The gate is evaluated at each bar's open using the volatility that was
    already closed on the previous bar, so the filter never reads the bar it
    decides on. Thresholds are absolute percentages: deriving them from
    quantiles of the full sample would leak the future even though the
    comparison itself is trailing.

    ``None`` on either bound means "no constraint on that side". An open cycle
    is never filtered: the gate only decides whether a flat account may start
    a new ladder.
    """

    window_hours: float = 4.0
    min_annualized_pct: Optional[float] = None
    max_annualized_pct: Optional[float] = None

    def __post_init__(self):
        if self.window_hours <= 0.0:
            raise ValueError("window_hours must be positive")
        if self.min_annualized_pct is None and self.max_annualized_pct is None:
            raise ValueError("at least one volatility bound must be set")
        if self.min_annualized_pct is not None and self.min_annualized_pct < 0.0:
            raise ValueError("min_annualized_pct must be >= 0")
        if self.max_annualized_pct is not None and self.max_annualized_pct <= 0.0:
            raise ValueError("max_annualized_pct must be > 0")
        if (
            self.min_annualized_pct is not None
            and self.max_annualized_pct is not None
            and self.min_annualized_pct >= self.max_annualized_pct
        ):
            raise ValueError("min_annualized_pct must be below max_annualized_pct")


@dataclass(frozen=True)
class BacktestConfig:
    strategy: MartingaleParams = field(default_factory=MartingaleParams)
    account: AccountParams = field(default_factory=AccountParams)
    data: DataParams = field(default_factory=DataParams)
    monte_carlo: MonteCarloParams = field(default_factory=MonteCarloParams)
    capital: CapitalParams = field(default_factory=CapitalParams)
    gap_analysis: "GapAnalysisParams" = field(default_factory=lambda: GapAnalysisParams())
    # Finer interval used to settle ambiguous bars of the primary series.
    intrabar_interval: Optional[str] = "1m"
    # None: every bar may start a new ladder (the original behaviour).
    volatility_filter: Optional[VolatilityFilterParams] = None
    max_cycle_days: Optional[float] = None
    save_path: Optional[str] = None
    trades_path: Optional[str] = None   # one row per closed trade
    fills_path: Optional[str] = None    # one row per filled layer
    plot_path: Optional[str] = None      # price chart with every failure marked
    equity_plot_path: Optional[str] = None  # market + all Monte Carlo equity paths
    equity_curves_per_plot: int = 10     # split paths over figures of this size
    adverse_plot_path: Optional[str] = None  # market + per-layer adverse hits
    grid_break_plot_dir: Optional[str] = None  # one market chart per grid break
    grid_break_context_bars: int = 200
    pressure_path: Optional[str] = None  # time x window-scale pressure heatmap
    cluster_path: Optional[str] = None   # gap allowance x cluster-size heatmap
    pressure_min_scale: int = 8
    pressure_max_scale: Optional[int] = None
    pressure_scale_count: int = 40
    cluster_min_scale: int = 1
    cluster_max_scale: int = 5_000
    cluster_scale_count: int = 40
    # Every grid break on the price/volatility timeline plus the conditional
    # break probability per trailing-volatility bucket.
    breach_analysis_dir: Optional[str] = None
    breach_vol_window_hours: float = 4.0
    breach_horizons_hours: Sequence[float] = (1.0, 4.0, 24.0)
    breach_buckets: int = 10

    def __post_init__(self):
        if self.equity_curves_per_plot < 1:
            raise ValueError("equity_curves_per_plot must be >= 1")
        if self.grid_break_context_bars < 0:
            raise ValueError("grid_break_context_bars must be >= 0")
        if self.max_cycle_days is not None and self.max_cycle_days <= 0.0:
            raise ValueError("max_cycle_days must be positive or None")
        if self.breach_vol_window_hours <= 0.0:
            raise ValueError("breach_vol_window_hours must be positive")
        if not self.breach_horizons_hours or any(
            hours <= 0.0 for hours in self.breach_horizons_hours
        ):
            raise ValueError("breach_horizons_hours must be a non-empty list of positive hours")
        if self.breach_buckets < 2:
            raise ValueError("breach_buckets must be >= 2")


# ============================================================
# Data
# ============================================================
@dataclass
class BarSeries:
    """Column-wise OHLC: cheap to slice and to index inside the hot loop."""

    time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    # Bar frequency is a property of the series, not of a single run: computed
    # once in load_bars instead of re-derived from the whole time column on
    # every Monte Carlo run.
    periods_per_year: float = 0.0
    # Optional finer bars per bar, shape (bars, steps, 4): open/high/low/close.
    # Missing rows are NaN and fall back to the pessimistic two-path choice.
    intrabar: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.open)


def _periods_per_year(times: np.ndarray) -> float:
    """Annualization factor from the median positive bar spacing."""
    if len(times) < 2:
        return 0.0
    # ``.dt.total_seconds()`` is unit aware; pandas may store timestamps in
    # microseconds rather than nanoseconds, so int64 arithmetic is not safe.
    deltas = pd.to_datetime(times).to_series().diff().dt.total_seconds().to_numpy()
    positive = deltas[deltas > 0]
    if not len(positive):
        return 0.0
    median_seconds = float(np.median(positive))
    if median_seconds <= 0.0:
        return 0.0
    return 365.25 * 24.0 * 3600.0 / median_seconds


def rolling_annualized_volatility(
    bars: BarSeries,
    *,
    window_hours: float = 4.0,
) -> tuple[np.ndarray, int]:
    """Trailing annualized close-to-close volatility of the primary bars.

    The lookback is given in hours, so the same setting means the same window
    on 1m, 5m or 1h series. A value at index ``i`` uses closes up to and
    including bar ``i``; callers that decide on bar ``i``'s open must shift it
    by one bar (see ``build_open_gate``).
    """
    bars_per_hour = (
        bars.periods_per_year / (365.25 * 24.0)
        if bars.periods_per_year > 0.0
        else 0.0
    )
    window_bars = max(2, int(round(window_hours * bars_per_hour))) if bars_per_hour else 8
    close = np.asarray(bars.close, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(close), prepend=np.nan)
    series = pd.Series(returns, dtype=float).rolling(
        window_bars, min_periods=window_bars
    ).std(ddof=1)
    return series.to_numpy() * np.sqrt(bars.periods_per_year), window_bars


def build_open_gate(
    bars: BarSeries,
    params: "VolatilityFilterParams",
) -> np.ndarray:
    """Bars where a flat account may open a base order, strictly causal.

    Bar ``i`` opens at its own open, before that bar's close is known, so the
    gate uses volatility from bar ``i-1``. Warm-up bars (no trailing window
    yet) stay flat rather than guess.
    """
    volatility, _ = rolling_annualized_volatility(bars, window_hours=params.window_hours)
    shifted = np.concatenate(([np.nan], volatility[:-1]))
    percentage = shifted * 100.0
    allowed = np.isfinite(percentage)
    if params.min_annualized_pct is not None:
        allowed &= percentage >= params.min_annualized_pct
    if params.max_annualized_pct is not None:
        allowed &= percentage <= params.max_annualized_pct
    return allowed


def _days_to_bars(days: Optional[float], periods_per_year: float) -> Optional[int]:
    if days is None:
        return None
    if days <= 0.0:
        raise ValueError("max_cycle_days must be positive or None")
    bars_per_day = periods_per_year / 365.25 if periods_per_year > 0.0 else 0.0
    if bars_per_day <= 0.0 or not np.isfinite(bars_per_day):
        raise ValueError("cannot convert max_cycle_days without a valid bar frequency")
    return max(1, int(math.ceil(days * bars_per_day)))


def _parse_ymd_date(value: Optional[str], name: str) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"{name} must use YYYY-MM-DD, got {value!r}") from error
    if text != parsed.strftime("%Y-%m-%d"):
        raise ValueError(f"{name} must use YYYY-MM-DD, got {value!r}")
    return pd.Timestamp(parsed, tz="UTC")


def _interval_seconds(interval: str) -> int:
    """Seconds per bar for intervals such as 1m, 5m, 1h."""
    unit = interval[-1].lower()
    value = int(interval[:-1])
    factor = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit)
    if factor is None or value <= 0:
        raise ValueError(f"unsupported interval: {interval!r}")
    return value * factor


def _load_intrabar(
    logger: logging.Logger,
    data: DataParams,
    times: np.ndarray,
    intrabar_interval: str,
) -> Optional[np.ndarray]:
    """Align a finer interval onto the primary bars as (bars, steps, 4)."""
    sub_data = replace(data, interval=intrabar_interval)
    path = common.market_data_path(sub_data, root_dir=data.root_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Intrabar file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8", usecols=[
        "open_time_date_utc", "open", "high", "low", "close",
    ])
    frame["open_time_date_utc"] = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    frame = frame.sort_values("open_time_date_utc")
    ohlc = frame[["open", "high", "low", "close"]].astype(float)
    valid = (
        ohlc.notna().all(axis=1)
        & (ohlc > 0.0).all(axis=1)
        & (ohlc["high"] >= ohlc[["open", "low", "close"]].max(axis=1))
        & (ohlc["low"] <= ohlc[["open", "high", "close"]].min(axis=1))
    )
    frame = frame[valid]
    sub_times = frame["open_time_date_utc"].to_numpy()

    main_seconds = int(round(np.median(np.diff(times)) / np.timedelta64(1, "s")))
    sub_seconds = _interval_seconds(intrabar_interval)
    steps = int(round(main_seconds / sub_seconds))
    if steps < 2:
        raise ValueError(
            f"intrabar interval {intrabar_interval!r} is not finer than the primary bars"
        )

    lower = times[0]
    upper = times[-1] + np.timedelta64(main_seconds, "s")
    inside = (sub_times >= lower) & (sub_times < upper)
    sub_times = sub_times[inside]
    sub_ohlc = frame.loc[inside, ["open", "high", "low", "close"]].to_numpy(dtype=float)

    bars = np.full((len(times), steps, 4), np.nan)
    slot = np.searchsorted(times, sub_times, side="right") - 1
    offset = ((sub_times - times[slot]) / np.timedelta64(sub_seconds, "s")).astype(int)
    keep = (slot >= 0) & (offset >= 0) & (offset < steps)
    bars[slot[keep], offset[keep]] = sub_ohlc[keep]
    coverage = float(np.isfinite(bars).all(axis=(1, 2)).mean())
    logger.info(
        f"Loaded {len(sub_times)} {intrabar_interval} rows into {steps} steps/bar | "
        f"complete bars {coverage * 100:.1f}%"
    )
    return bars


def load_bars(
    logger: logging.Logger,
    data: DataParams,
    intrabar_interval: Optional[str] = None,
) -> BarSeries:
    """Read the canonical raw CSV and drop rows that are not valid candles."""
    path = common.market_data_path(data, root_dir=data.root_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8")
    if "open_time_date_utc" not in frame.columns:
        raise ValueError("'open_time_date_utc' column missing from market data")
    frame["open_time_date_utc"] = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    frame = frame.sort_values("open_time_date_utc")

    start_date = data.test_start_date or data.from_date
    end_date = data.test_end_date or data.to_date
    start_ts = _parse_ymd_date(start_date, "test_start_date")
    end_ts = _parse_ymd_date(end_date, "test_end_date")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            f"test_start_date {start_date!r} must be <= test_end_date {end_date!r}"
        )
    if start_ts is not None:
        frame = frame[frame["open_time_date_utc"] >= start_ts]
    if end_ts is not None:
        # End date is inclusive for a YYYY-MM-DD user setting.
        frame = frame[frame["open_time_date_utc"] < end_ts + pd.Timedelta(days=1)]
    if start_ts is not None or end_ts is not None:
        logger.info(
            "TEST RANGE | "
            f"{start_ts.date() if start_ts is not None else 'data-start'} -> "
            f"{end_ts.date() if end_ts is not None else 'data-end'} inclusive"
        )

    raw_rows = len(frame)
    ohlc = frame[["open", "high", "low", "close"]].astype(float)
    valid = (
        ohlc.notna().all(axis=1)
        & (ohlc > 0.0).all(axis=1)
        & (ohlc["high"] >= ohlc[["open", "low", "close"]].max(axis=1))
        & (ohlc["low"] <= ohlc[["open", "high", "close"]].min(axis=1))
    )
    frame = frame[valid]
    if not len(frame):
        raise ValueError(f"No usable bars left after filtering {path}")
    times = frame["open_time_date_utc"].to_numpy()
    periods_per_year = _periods_per_year(times)
    logger.info(
        f"Loaded {len(frame)} bars from {path} "
        f"({raw_rows - len(frame)} malformed rows dropped) | "
        f"{frame['open_time_date_utc'].iloc[0]} -> {frame['open_time_date_utc'].iloc[-1]} "
        f"| {periods_per_year:.0f} bars/year"
    )
    intrabar = (
        None if intrabar_interval is None
        else _load_intrabar(logger, data, times, intrabar_interval)
    )
    return BarSeries(
        time=times,
        open=frame["open"].to_numpy(dtype=float),
        high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float),
        close=frame["close"].to_numpy(dtype=float),
        periods_per_year=periods_per_year,
        intrabar=intrabar,
    )


# ============================================================
# One run
# ============================================================
def _elapsed_years(start_time, end_time) -> float:
    seconds = (pd.Timestamp(end_time) - pd.Timestamp(start_time)).total_seconds()
    return max(seconds / (365.25 * 24.0 * 3600.0), 0.0)


# Annualizing a window shorter than a day is meaningless and overflows: on 1m
# bars a 2.6x peak reached in one bar compounds to ~1e17 or straight to inf.
MIN_CAGR_YEARS = 1.0 / 365.25


def _cagr(start_value: float, end_value: float, years: float) -> Optional[float]:
    if start_value <= 0.0 or end_value < 0.0 or years < MIN_CAGR_YEARS:
        return None
    if end_value == 0.0:
        return -1.0
    rate = (end_value / start_value) ** (1.0 / years) - 1.0
    return float(rate) if np.isfinite(rate) else None


def _annualized_sharpe(equity_curve: Sequence[float], periods_per_year: float) -> Optional[float]:
    values = np.asarray(equity_curve, dtype=float)
    if len(values) < 3 or periods_per_year <= 0.0:
        return None
    previous = values[:-1]
    current = values[1:]
    valid = (previous > 0.0) & np.isfinite(previous) & np.isfinite(current)
    if valid.sum() < 2:
        return None
    returns = current[valid] / previous[valid] - 1.0
    std = float(np.std(returns, ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return None
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _annualized_sharpe_segments(
    equity_segments: Sequence[Sequence[float]],
    periods_per_year: float,
) -> Optional[float]:
    """Sharpe over real trading moves, excluding reset-capital injection jumps."""
    returns = []
    for segment in equity_segments:
        values = np.asarray(segment, dtype=float)
        if len(values) < 2:
            continue
        previous = values[:-1]
        current = values[1:]
        valid = (previous > 0.0) & np.isfinite(previous) & np.isfinite(current)
        returns.extend((current[valid] / previous[valid] - 1.0).tolist())
    if len(returns) < 2 or periods_per_year <= 0.0:
        return None
    std = float(np.std(returns, ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return None
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def run_once(
    bars: BarSeries,
    config: BacktestConfig,
    *,
    start_index: int = 0,
    max_bars: Optional[int] = None,
) -> dict:
    """Trade until the limit/data end, resetting after every ruin.

    A grid break is only a realized full-ladder exit. Whether it also ruins the
    current strategy run is controlled by ``grid_break_is_ruin`` and the
    peak-balance drawdown threshold. Grid infeasibility is unrecoverable and
    ends the Monte Carlo path.
    """
    total_bars = len(bars)
    if not 0 <= start_index < total_bars:
        raise ValueError(f"start_index {start_index} outside 0..{total_bars - 1}")

    account = config.account
    capital = config.capital
    if capital.profit_handling == "both":
        raise ValueError(
            "run_once needs one profit_handling mode; use main() to compare both"
        )
    simulator = MartingaleSimulator(config.strategy, account)
    state = simulator.new_state()
    initial_equity = account.initial_equity
    end_index = total_bars if max_bars is None else min(total_bars, start_index + max_bars)
    periods_per_year = bars.periods_per_year
    # Built once per run, already shifted so each bar decides on the previous
    # bar's closed volatility. None means the original always-open behaviour.
    open_gate = (
        None if config.volatility_filter is None
        else build_open_gate(bars, config.volatility_filter)
    )

    peak_equity = initial_equity
    peak_index = start_index
    peak_time = bars.time[start_index]
    # Per-bar mark-to-market balance: realized balance plus open PnL at close.
    equity_curve = [initial_equity]
    equity_curve_indices = [start_index]
    performance_equity_segments = [[initial_equity]]
    performance_equity_index_segments = [[start_index]]
    exit_reason = "end_of_data"
    index = start_index
    processed_trades = 0
    reserve_balance = 0.0
    first_grid_breach_reason: Optional[str] = None
    first_grid_breach_index: Optional[int] = None
    first_grid_breach_time = None
    first_ruin_reason: Optional[str] = None
    first_ruin_index: Optional[int] = None
    first_ruin_time = None
    first_ruin_equity: Optional[float] = None
    withdrawn_profit_before_breach: Optional[float] = None
    doubled_before_grid_breach = False
    double_index: Optional[int] = None
    double_time = None
    double_balance: Optional[float] = None
    completed_states = []
    all_trades: List[dict] = []
    grid_failures: List[dict] = []
    strategy_runs: List[dict] = []
    first_account_failure_price: Optional[float] = None
    reset_capital_injected = 0.0
    reset_capital_removed = 0.0
    account_reset_count = 0
    account_start_index = start_index
    account_start_time = bars.time[start_index]
    account_start_trade_pos = 0

    def finish_strategy_run(
        *,
        sequence: int,
        account_state,
        final_account_equity: float,
        end_idx: int,
        end_at,
        reason: str,
        failure_event: Optional[dict],
    ):
        account_trades = all_trades[account_start_trade_pos:]
        if not account_trades and failure_event is None and end_idx < account_start_index:
            return
        segment = performance_equity_segments[-1]
        segment_indices = performance_equity_index_segments[-1]
        peak_pos = int(np.argmax(segment)) if segment else 0
        peak_value = float(segment[peak_pos]) if segment else initial_equity
        peak_idx = (
            segment_indices[min(peak_pos, len(segment_indices) - 1)]
            if segment_indices else account_start_index
        )
        grid_break_trades = [
            trade for trade in account_trades if trade["reason"] == "grid_break"
        ]
        first_grid_break_trade = (
            min(grid_break_trades, key=lambda trade: trade["exit_bar"])
            if grid_break_trades else None
        )
        ruin_reason = (
            None if failure_event is None
            else failure_event.get("account_failure_reason")
        )
        ruined = ruin_reason in ("liquidation", "ruin_threshold", "grid_break_ruin")
        profitable_count = sum(trade["net_pnl"] > 0.0 for trade in account_trades)
        final_total_wealth = final_account_equity
        if capital.profit_handling == "withdraw":
            final_total_wealth += sum(
                trade.get("withdrawn_profit", 0.0) for trade in account_trades
            )
        exit_counts = {}
        for trade in account_trades:
            exit_counts[trade["reason"]] = exit_counts.get(trade["reason"], 0) + 1
        elapsed = _elapsed_years(account_start_time, end_at)
        peak_elapsed = _elapsed_years(account_start_time, bars.time[peak_idx])
        strategy_runs.append({
            "monte_carlo_run_id": None,
            "account_sequence": sequence,
            "start_index": account_start_index,
            "start_time": account_start_time,
            "end_index": end_idx,
            "end_time": end_at,
            "bars_survived": max(end_idx - account_start_index + 1, 0),
            "elapsed_years": elapsed,
            "initial_equity": initial_equity,
            "final_equity": final_account_equity,
            "final_total_wealth": final_total_wealth,
            "final_total_wealth_after_reset_transfers": final_total_wealth,
            "profit_handling": capital.profit_handling,
            "reserve_balance": final_total_wealth - final_account_equity,
            "reset_capital_injected": 0.0,
            "reset_capital_removed": 0.0,
            "reset_capital_net_injected": 0.0,
            "withdrawn_profit_before_grid_breach": (
                final_total_wealth - final_account_equity
            ),
            "grid_breached": first_grid_break_trade is not None,
            "grid_breach_reason": (
                None if first_grid_break_trade is None
                else first_grid_break_trade["reason"]
            ),
            "grid_breach_index": (
                None if first_grid_break_trade is None
                else first_grid_break_trade["exit_bar"]
            ),
            "grid_breach_time": (
                None if first_grid_break_trade is None
                else first_grid_break_trade["closed_at"]
            ),
            "bars_to_grid_break": (
                None if first_grid_break_trade is None
                else first_grid_break_trade["exit_bar"] - account_start_index + 1
            ),
            "return_before_grid_break_pct": (
                None if first_grid_break_trade is None
                else (
                    first_grid_break_trade["balance_after"] - initial_equity
                ) / initial_equity
            ),
            "trades_before_grid_breach": (
                len(account_trades)
                if first_grid_break_trade is None
                else sum(
                    trade["exit_bar"] <= first_grid_break_trade["exit_bar"]
                    for trade in account_trades
                )
            ),
            "doubled_before_grid_breach": peak_value >= initial_equity * capital.double_target_multiple - 1e-12,
            "double_target_multiple": capital.double_target_multiple,
            "double_index": None,
            "double_time": None,
            "double_balance": None,
            "bars_to_double": None,
            "grid_failure_count": 0 if failure_event is None else 1,
            "grid_failures": [] if failure_event is None else [failure_event],
            "account_reset_count": 0,
            "min_equity": float(account_state.min_equity_seen),
            "peak_equity": peak_value,
            "peak_realized_balance": float(account_state.max_balance_seen),
            "peak_equity_multiple": float(peak_value / initial_equity),
            "peak_return_pct": float(peak_value / initial_equity - 1.0),
            "peak_index": peak_idx,
            "peak_time": bars.time[peak_idx],
            "bars_to_peak": max(peak_idx - account_start_index + 1, 0),
            "years_to_peak": peak_elapsed,
            "peak_cagr": _cagr(initial_equity, peak_value, peak_elapsed),
            "total_return_pct": (final_account_equity - initial_equity) / initial_equity,
            "total_return_after_reset_cost_pct": (
                (final_total_wealth - initial_equity) / initial_equity
            ),
            "cagr": _cagr(initial_equity, final_account_equity, elapsed),
            "sharpe": _annualized_sharpe(segment, periods_per_year),
            "periods_per_year": periods_per_year,
            "max_drawdown_pct": float(account_state.max_drawdown_seen),
            "max_margin_usage": float(account_state.max_margin_usage_seen),
            "exit_reason": reason,
            "ruin_reason": ruin_reason,
            "ruined": ruined,
            "ruin_index": None if failure_event is None or not ruined else failure_event["index"],
            "ruin_time": None if failure_event is None or not ruined else failure_event["time"],
            "bars_to_ruin": None if not ruined else max(end_idx - account_start_index + 1, 0),
            "return_before_ruin_pct": (
                None if not ruined
                else (final_account_equity - initial_equity) / initial_equity
            ),
            "failure_price": None if failure_event is None else failure_event["price"],
            "total_trades": len(account_trades),
            "profitable_trades": int(profitable_count),
            "losing_trades": int(len(account_trades) - profitable_count),
            "win_rate": (
                float(profitable_count / len(account_trades)) if account_trades else 0.0
            ),
            "total_net_profit_abs": sum(trade["net_pnl"] for trade in account_trades),
            "total_fees": sum(trade["fees"] for trade in account_trades),
            "grid_break_trades": len(grid_break_trades),
            "grid_break_net_pnl": sum(trade["net_pnl"] for trade in grid_break_trades),
            "same_bar_trades": int(sum(trade["same_bar"] for trade in account_trades)),
            "multiple_fills_same_bar_trades": int(sum(
                trade["multiple_fills_same_bar"] for trade in account_trades
            )),
            "max_layers_seen": max(
                max((trade["layers"] for trade in account_trades), default=0),
                account_state.cycle.layers,
            ),
            "open_layers_at_end": account_state.cycle.layers,
            "unfilled_layers": account_state.unfilled_layers,
            "cooldown_blocked_bars": account_state.cooldown_blocked_bars,
            "volatility_blocked_bars": account_state.volatility_blocked_bars,
            "reversals": account_state.reversals,
            "final_direction": SIDE_NAME[account_state.next_side],
            "dual_path_bars": account_state.dual_path_bars,
            "up_first_worse_bars": account_state.up_first_worse_bars,
            "exit_counts": exit_counts,
        })

    for index in range(start_index, end_index):
        previous_peak = state.max_equity_seen
        state = simulator.process_bar(
            state,
            index=index,
            time=bars.time[index],
            open_=bars.open[index],
            high=bars.high[index],
            low=bars.low[index],
            close=bars.close[index],
            intrabar=None if bars.intrabar is None else bars.intrabar[index],
            trade_resolver=(
                lambda fine_index, row: bars.agg_trade_prices(
                    bars.time[index] + np.timedelta64(
                        fine_index * bars.intrabar_step_ms, "ms"
                    ),
                    row,
                    True,
                )
                if hasattr(bars, "agg_trade_prices") else None
            ),
            allow_open=True if open_gate is None else bool(open_gate[index]),
        )

        new_trades = state.trades[processed_trades:]
        processed_trades = len(state.trades)
        for offset, trade in enumerate(new_trades, start=1):
            trade["index"] = len(all_trades) + offset
            trade["account_sequence"] = len(completed_states) + 1
        if capital.profit_handling == "withdraw":
            for trade in new_trades:
                withdrawn = max(float(trade["net_pnl"]), 0.0)
                if withdrawn:
                    # The reserve is outside the trading account: it cannot
                    # size the next ladder or be lost by the active grid.
                    state.equity = max(0.0, state.equity - withdrawn)
                    reserve_balance += withdrawn
                trade["withdrawn_profit"] = withdrawn
                trade["reserve_after"] = reserve_balance
                trade["trading_balance_after_withdrawal"] = state.equity
        else:
            for trade in new_trades:
                trade["withdrawn_profit"] = 0.0
                trade["reserve_after"] = 0.0
                trade["trading_balance_after_withdrawal"] = state.equity
        all_trades.extend(new_trades)

        if (
            capital.profit_handling == "compound"
            and first_grid_breach_reason is None
            and not doubled_before_grid_breach
            and state.max_balance_seen
            >= initial_equity * capital.double_target_multiple - 1e-12
        ):
            doubled_before_grid_breach = True
            double_index = index
            double_time = bars.time[index]
            double_balance = float(state.max_balance_seen)

        breach_reason = next(
            (trade["reason"] for trade in new_trades if trade["reason"] == "grid_break"),
            None,
        )
        account_failure_reason = None
        if state.bankrupt:
            account_failure_reason = "liquidation"
        elif breach_reason == "grid_break" and capital.grid_break_is_ruin:
            account_failure_reason = "grid_break_ruin"
            state.ruin_threshold_hit = True
            state.failure_equity = state.equity
            state.failure_price = (
                new_trades[-1]["exit_price"] if new_trades else bars.close[index]
            )
        elif (
            breach_reason == "grid_break"
            and state.equity
            <= state.max_balance_seen * account.ruin_equity_pct + 1e-12
        ):
            # Ruin is checked only after a realized full-grid break. The
            # threshold is a drawdown from the highest realized balance seen in
            # this strategy run, not a fixed share of initial equity.
            account_failure_reason = "ruin_threshold"
            state.ruin_threshold_hit = True
            state.failure_equity = state.equity
            state.failure_price = (
                new_trades[-1]["exit_price"] if new_trades else bars.close[index]
            )
        elif state.grid_infeasible:
            account_failure_reason = "grid_infeasible"
        if breach_reason == "grid_break" and first_grid_breach_reason is None:
            first_grid_breach_reason = breach_reason
            first_grid_breach_index = index
            first_grid_breach_time = bars.time[index]
            withdrawn_profit_before_breach = reserve_balance
        if breach_reason is None:
            breach_reason = account_failure_reason
        if (
            account_failure_reason in ("liquidation", "ruin_threshold", "grid_break_ruin")
            and first_ruin_reason is None
        ):
            first_ruin_reason = account_failure_reason
            first_ruin_index = index
            first_ruin_time = bars.time[index]
            first_ruin_equity = float(state.failure_equity if state.failure_equity is not None else state.equity)

        if state.max_equity_seen > peak_equity + 1e-12:
            peak_equity = state.max_equity_seen
            peak_index = index
            peak_time = bars.time[index]
        elif state.max_equity_seen > previous_peak + 1e-12:
            peak_equity = max(peak_equity, state.max_equity_seen)

        if state.bankrupt:
            current_equity = float(state.failure_equity if state.failure_equity is not None else state.equity)
            exit_reason = "liquidation"
        elif state.ruin_threshold_hit:
            current_equity = float(
                state.failure_equity
                if state.failure_equity is not None
                else state.max_balance_seen * account.ruin_equity_pct
            )
            exit_reason = "ruin_threshold"
        elif state.grid_infeasible:
            current_equity = float(
                state.failure_equity
                if state.failure_equity is not None
                else simulator.equity_at(state, bars.close[index])
            )
            exit_reason = "grid_infeasible"
        else:
            current_equity = simulator.equity_at(state, bars.close[index])

        failure_equity = max(current_equity, 0.0)
        equity_curve.append(failure_equity)
        equity_curve_indices.append(index)
        performance_equity_segments[-1].append(failure_equity)
        performance_equity_index_segments[-1].append(index)

        if breach_reason is not None:
            failure_price = state.failure_price
            if failure_price is None or not np.isfinite(failure_price):
                matching_trade = next(
                    (
                        trade for trade in reversed(new_trades)
                        if trade["reason"] == breach_reason
                    ),
                    None,
                )
                failure_price = (
                    float(matching_trade["exit_price"])
                    if matching_trade is not None
                    else float(bars.close[index])
                )
            ruined_event = account_failure_reason in (
                "liquidation", "ruin_threshold", "grid_break_ruin"
            )
            resettable = ruined_event
            reset_capital_delta = (
                initial_equity - failure_equity if resettable else 0.0
            )
            injected = max(reset_capital_delta, 0.0)
            removed = max(-reset_capital_delta, 0.0)
            grid_failures.append({
                "sequence": len(grid_failures) + 1,
                "index": index,
                "time": bars.time[index],
                "reason": breach_reason,
                "account_failure_reason": account_failure_reason,
                "ruined": ruined_event,
                "peak_balance_before_event": float(state.max_balance_seen),
                "ruin_equity_threshold": (
                    float(state.max_balance_seen * account.ruin_equity_pct)
                    if ruined_event else None
                ),
                "equity_before_reset": failure_equity,
                "reset_equity": initial_equity if resettable else None,
                "reset_capital_delta": reset_capital_delta,
                "reset_capital_injected": injected,
                "reset_capital_removed": removed,
                "price": float(failure_price),
            })
            failure_event = grid_failures[-1]
            reset_capital_injected += injected
            reset_capital_removed += removed
            if state.failed and first_account_failure_price is None:
                first_account_failure_price = float(failure_price)

            if (
                capital.stop_at_first_grid_breach
                or account_failure_reason == "grid_infeasible"
            ):
                finish_strategy_run(
                    sequence=len(completed_states) + 1,
                    account_state=state,
                    final_account_equity=failure_equity,
                    end_idx=index,
                    end_at=bars.time[index],
                    reason=account_failure_reason or breach_reason,
                    failure_event=failure_event,
                )
                exit_reason = "grid_breach"
                break

            if not resettable:
                # A plain grid break is just a closed losing cycle. It stays in
                # the same strategy run and the next cycle can continue after
                # the engine's normal same-bar cooldown.
                continue

            finish_strategy_run(
                sequence=len(completed_states) + 1,
                account_state=state,
                final_account_equity=failure_equity,
                end_idx=index,
                end_at=bars.time[index],
                reason=account_failure_reason or breach_reason,
                failure_event=failure_event,
            )

            # Keep the failed account for aggregate statistics, then make the
            # reset an explicit second point at the same timestamp. The next
            # account may open only on the next bar, never after seeing the
            # remainder of the bar that caused the failure.
            completed_states.append(state)
            state = simulator.new_state()
            processed_trades = 0
            account_reset_count += 1
            exit_reason = "end_of_data"
            equity_curve.append(initial_equity)
            equity_curve_indices.append(index)
            account_start_index = min(index + 1, end_index - 1)
            account_start_time = bars.time[account_start_index]
            account_start_trade_pos = len(all_trades)
            performance_equity_segments.append([initial_equity])
            performance_equity_index_segments.append([account_start_index])

    final_equity = equity_curve[-1]
    end_time = bars.time[index]
    has_open_strategy_activity = (
        account_start_trade_pos < len(all_trades)
        or len(performance_equity_segments[-1]) > 1
        or state.cycle.qty > 0.0
    )
    if (
        has_open_strategy_activity
        and (
            not strategy_runs
            or strategy_runs[-1]["account_sequence"] != len(completed_states) + 1
        )
    ):
        finish_strategy_run(
            sequence=len(completed_states) + 1,
            account_state=state,
            final_account_equity=final_equity,
            end_idx=index,
            end_at=end_time,
            reason=exit_reason,
            failure_event=None,
        )
    elapsed_years = _elapsed_years(bars.time[start_index], end_time)
    peak_years = _elapsed_years(bars.time[start_index], peak_time)
    states = completed_states + [state]
    max_drawdown = max((item.max_drawdown_seen for item in states), default=0.0)
    trades = all_trades
    exit_counts: dict[str, int] = {}
    for trade in trades:
        exit_counts[trade["reason"]] = exit_counts.get(trade["reason"], 0) + 1

    profitable = sum(trade["net_pnl"] > 0.0 for trade in trades)
    grid_breaks = [trade for trade in trades if trade["reason"] == "grid_break"]
    if withdrawn_profit_before_breach is None:
        withdrawn_profit_before_breach = reserve_balance
    final_total_wealth = final_equity + reserve_balance
    final_total_wealth_after_reset_transfers = (
        final_total_wealth - reset_capital_injected + reset_capital_removed
    )
    summary = {
        "start_index": start_index,
        "start_time": bars.time[start_index],
        "end_index": index,
        "end_time": end_time,
        "bars_survived": index - start_index + 1,
        "elapsed_years": elapsed_years,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "final_total_wealth": final_total_wealth,
        "final_total_wealth_after_reset_transfers": (
            final_total_wealth_after_reset_transfers
        ),
        "profit_handling": capital.profit_handling,
        "reserve_balance": reserve_balance,
        "reset_capital_injected": reset_capital_injected,
        "reset_capital_removed": reset_capital_removed,
        "reset_capital_net_injected": (
            reset_capital_injected - reset_capital_removed
        ),
        "withdrawn_profit_before_grid_breach": withdrawn_profit_before_breach,
        "grid_breached": first_grid_breach_reason is not None,
        "grid_breach_reason": first_grid_breach_reason,
        "grid_breach_index": first_grid_breach_index,
        "grid_breach_time": first_grid_breach_time,
        "bars_to_grid_break": (
            None if first_grid_breach_index is None
            else first_grid_breach_index - start_index + 1
        ),
        "return_before_grid_break_pct": (
            None if first_grid_breach_index is None
            else (
                next(
                    (
                        trade["balance_after"]
                        for trade in grid_breaks
                        if trade["exit_bar"] == first_grid_breach_index
                    ),
                    final_equity,
                )
                - initial_equity
            ) / initial_equity
        ),
        "ruined_before_reset": first_ruin_reason is not None,
        "ruin_reason": first_ruin_reason,
        "ruin_index": first_ruin_index,
        "ruin_time": first_ruin_time,
        "bars_to_ruin": (
            None if first_ruin_index is None else first_ruin_index - start_index + 1
        ),
        "return_before_ruin_pct": (
            None if first_ruin_index is None
            else (first_ruin_equity - initial_equity) / initial_equity
        ),
        "trades_before_grid_breach": (
            len(trades)
            if first_grid_breach_index is None
            else sum(
                trade["exit_bar"] <= first_grid_breach_index for trade in trades
            )
        ),
        "doubled_before_grid_breach": doubled_before_grid_breach,
        "double_target_multiple": capital.double_target_multiple,
        "double_index": double_index,
        "double_time": double_time,
        "double_balance": double_balance,
        "bars_to_double": (
            None if double_index is None else double_index - start_index + 1
        ),
        "grid_failure_count": len(grid_failures),
        "grid_failures": grid_failures,
        "account_reset_count": account_reset_count,
        "min_equity": float(min(item.min_equity_seen for item in states)),
        "peak_equity": float(peak_equity),
        "peak_realized_balance": float(max(item.max_balance_seen for item in states)),
        "peak_equity_multiple": float(peak_equity / initial_equity),
        "peak_return_pct": float(peak_equity / initial_equity - 1.0),
        "peak_index": peak_index,
        "peak_time": peak_time,
        "bars_to_peak": peak_index - start_index + 1,
        "years_to_peak": peak_years,
        "peak_cagr": _cagr(initial_equity, peak_equity, peak_years),
        "total_return_pct": (final_equity - initial_equity) / initial_equity,
        "total_return_after_reset_cost_pct": (
            (final_total_wealth_after_reset_transfers - initial_equity)
            / initial_equity
        ),
        "cagr": _cagr(initial_equity, final_equity, elapsed_years),
        "sharpe": _annualized_sharpe_segments(
            performance_equity_segments,
            periods_per_year,
        ),
        "periods_per_year": periods_per_year,
        "max_drawdown_pct": float(max_drawdown),
        "max_margin_usage": float(max(item.max_margin_usage_seen for item in states)),
        "exit_reason": exit_reason,
        "ruined": any(item.bankrupt or item.ruin_threshold_hit for item in states),
        "failure_price": first_account_failure_price,
        "total_trades": len(trades),
        "profitable_trades": int(profitable),
        "losing_trades": int(len(trades) - profitable),
        "win_rate": float(profitable / len(trades)) if trades else 0.0,
        "total_net_profit_abs": sum(trade["net_pnl"] for trade in trades),
        "total_fees": sum(trade["fees"] for trade in trades),
        "grid_break_trades": len(grid_breaks),
        "grid_break_net_pnl": sum(trade["net_pnl"] for trade in grid_breaks),
        "same_bar_trades": int(sum(trade["same_bar"] for trade in trades)),
        "multiple_fills_same_bar_trades": int(sum(
            trade["multiple_fills_same_bar"] for trade in trades
        )),
        "max_layers_seen": max(
            max((trade["layers"] for trade in trades), default=0),
            max((item.cycle.layers for item in states), default=0),
        ),
        "open_layers_at_end": state.cycle.layers,
        "unfilled_layers": sum(item.unfilled_layers for item in states),
        "cooldown_blocked_bars": sum(item.cooldown_blocked_bars for item in states),
        "volatility_blocked_bars": sum(item.volatility_blocked_bars for item in states),
        "reversals": sum(item.reversals for item in states),
        "final_direction": SIDE_NAME[state.next_side],
        "dual_path_bars": sum(item.dual_path_bars for item in states),
        "up_first_worse_bars": sum(item.up_first_worse_bars for item in states),
        "exit_counts": exit_counts,
    }
    return {
        "summary": summary,
        "strategy_runs": strategy_runs,
        "trades": trades,
        "layer_stats": layer_statistics(trades),
        "equity_curve": {
            "indices": equity_curve_indices,
            "values": equity_curve,
            "balances": equity_curve,
            "balance_basis": "mark_to_market_close",
            "performance_values": performance_equity_segments,
            "performance_basis": "mark_to_market_close_without_reset_jumps",
        },
    }


def direction_statistics(trades: Sequence[dict]) -> dict:
    """Split the trades by the side the cycle was opened on.

    With ``reverse_after_grid_break`` on, this is what says whether following
    the break actually paid: compare the two sides' win rate and net PnL.
    """
    buckets: dict[str, dict] = {}
    for trade in trades:
        bucket = buckets.setdefault(
            trade.get("direction", "long"),
            {"trades": 0, "profitable_trades": 0, "entry_notional": 0.0,
             "net_profit_abs": 0.0, "fees": 0.0,
             "grid_breaks": 0},
        )
        bucket["trades"] += 1
        bucket["profitable_trades"] += int(trade["net_pnl"] > 0.0)
        bucket["entry_notional"] += trade["entry_notional"]
        bucket["net_profit_abs"] += trade["net_pnl"]
        bucket["fees"] += trade["fees"]
        bucket["grid_breaks"] += int(trade["reason"] == "grid_break")

    total = len(trades)
    for bucket in buckets.values():
        bucket["trade_pct"] = bucket["trades"] / total if total else 0.0
        bucket["win_rate"] = bucket["profitable_trades"] / bucket["trades"]
        bucket["grid_break_pct"] = bucket["grid_breaks"] / bucket["trades"]
        bucket["net_profit_pct"] = (
            bucket["net_profit_abs"] / bucket["entry_notional"]
            if bucket["entry_notional"] else 0.0
        )
        bucket["avg_net_profit_abs"] = bucket["net_profit_abs"] / bucket["trades"]
    return {name: buckets[name] for name in sorted(buckets)}


def layer_statistics(trades: Sequence[dict]) -> dict:
    """Report terminal PnL and take-profit probability at every reached layer.

    ``take_profit_pct`` uses every cycle that reached the layer as its
    denominator. Its complement is simply "did not take profit at this layer".
    """

    def new_bucket() -> dict:
        return {
            # Cycles whose terminal exit happened at exactly this layer.
            "trades": 0,
            "profitable_trades": 0,
            "entry_notional": 0.0,
            "net_profit_abs": 0.0,
            "fees": 0.0,
            "same_bar_adverse_then_tp": 0,
            "liquidations": 0,
            "grid_breaks": 0,
            "grid_break_net_pnl": 0.0,
            # Cycles that reached this layer and what happened next.
            "reached_trades": 0,
            "take_profits_at_layer": 0,
            # Separate layer PnL: one fill's own entry, exit and fee outcome.
            "layer_fills": 0,
            "layer_notional": 0.0,
            "layer_gross_pnl": 0.0,
            "layer_fees": 0.0,
            "layer_net_pnl": 0.0,
            "layer_take_profit_net_pnl": 0.0,
            "layer_first_adverse_final_net_pnl": 0.0,
            "planned_layer_net_pnl_at_tp": 0.0,
            "layer_bars_held_sum": 0.0,
            "layer_duration_seconds_sum": 0.0,
            "layer_take_profit_bars_sum": 0.0,
            "layer_take_profit_seconds_sum": 0.0,
            "layer_take_profit_count": 0,
            "layer_adverse_bars_sum": 0.0,
            "layer_adverse_seconds_sum": 0.0,
            "layer_adverse_count": 0,
            "layer_take_profit_bars_values": [],
            "layer_adverse_bars_values": [],
            "grid_spacing_sum": 0.0,
            "entry_price_to_base_sum": 0.0,
            "grid_geometry_count": 0,
        }

    def duration_seconds(fill: dict, end_key: str) -> float:
        start, end = fill.get("time"), fill.get(end_key)
        if start is None or end is None:
            return 0.0
        return max((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds(), 0.0)

    buckets: dict[str, dict] = {}
    for trade in trades:
        final_layer = int(trade["layers"])
        terminal = buckets.setdefault(str(final_layer), new_bucket())
        terminal["trades"] += 1
        terminal["profitable_trades"] += int(trade["net_pnl"] > 0.0)
        terminal["entry_notional"] += trade["entry_notional"]
        terminal["net_profit_abs"] += trade["net_pnl"]
        terminal["fees"] += trade["fees"]
        terminal["liquidations"] += int(trade["reason"] == "liquidation")
        if trade["reason"] == "grid_break":
            terminal["grid_breaks"] += 1
            terminal["grid_break_net_pnl"] += trade["net_pnl"]

        fills = trade.get("fills", ())
        same_bar_adverse_then_tp = (
            trade["reason"] == "take_profit"
            and any(
                fill.get("layer", 0) > 1
                and fill.get("bar") == trade.get("exit_bar")
                for fill in fills
            )
        )
        if same_bar_adverse_then_tp:
            terminal["same_bar_adverse_then_tp"] += 1

        for reached_layer in range(1, final_layer + 1):
            bucket = buckets.setdefault(str(reached_layer), new_bucket())
            bucket["reached_trades"] += 1
            if reached_layer == final_layer and trade["reason"] == "take_profit":
                bucket["take_profits_at_layer"] += 1

        base_entry_price = None
        if fills:
            base = fills[0]
            base_entry_price = base.get("planned_entry_price", base.get("price"))

        for fill in fills:
            bucket = buckets.setdefault(str(int(fill["layer"])), new_bucket())
            boundary_kind = fill.get("boundary_kind")
            seconds = duration_seconds(fill, "boundary_time")
            bars_held = float(fill.get(
                "boundary_bars_held",
                fill.get("layer_bars_held", 0.0),
            ))
            bucket["layer_fills"] += 1
            bucket["layer_notional"] += fill.get("notional", 0.0)
            bucket["layer_gross_pnl"] += fill.get("layer_gross_pnl", 0.0)
            bucket["layer_fees"] += fill.get("layer_fees", 0.0)
            bucket["layer_net_pnl"] += fill.get("layer_net_pnl", 0.0)
            bucket["planned_layer_net_pnl_at_tp"] += fill.get(
                "planned_layer_net_pnl_at_tp", 0.0
            )
            spacing = fill.get("planned_take_profit_pct_from_entry")
            if spacing is None:
                spacing = fill.get("planned_next_adverse_pct_from_entry")
            entry_price = fill.get("planned_entry_price", fill.get("price"))
            if (
                spacing is not None
                and entry_price is not None
                and base_entry_price
            ):
                bucket["grid_spacing_sum"] += abs(float(spacing))
                bucket["entry_price_to_base_sum"] += (
                    float(entry_price) / float(base_entry_price)
                )
                bucket["grid_geometry_count"] += 1
            bucket["layer_bars_held_sum"] += bars_held
            bucket["layer_duration_seconds_sum"] += seconds
            if boundary_kind == "take_profit":
                bucket["layer_take_profit_net_pnl"] += fill.get(
                    "layer_net_pnl", 0.0
                )
                bucket["layer_take_profit_bars_sum"] += bars_held
                bucket["layer_take_profit_seconds_sum"] += seconds
                bucket["layer_take_profit_count"] += 1
                bucket["layer_take_profit_bars_values"].append(bars_held)
            elif boundary_kind in (
                "adverse",
                "grid_break",
                "liquidation",
                "ruin_threshold",
                "grid_infeasible",
            ):
                bucket["layer_first_adverse_final_net_pnl"] += fill.get(
                    "layer_net_pnl", 0.0
                )
                bucket["layer_adverse_bars_sum"] += bars_held
                bucket["layer_adverse_seconds_sum"] += seconds
                bucket["layer_adverse_count"] += 1
                bucket["layer_adverse_bars_values"].append(bars_held)

    total = len(trades)
    for bucket in buckets.values():
        closed = bucket["trades"]
        reached = bucket["reached_trades"]
        bucket["trade_pct"] = closed / total if total else 0.0
        bucket["reach_pct"] = reached / total if total else 0.0
        bucket["win_rate"] = (
            bucket["profitable_trades"] / closed if closed else 0.0
        )
        bucket["take_profit_pct"] = (
            bucket["take_profits_at_layer"] / reached if reached else 0.0
        )
        bucket["same_bar_pct"] = (
            bucket["same_bar_adverse_then_tp"] / bucket["take_profits_at_layer"]
            if bucket["take_profits_at_layer"]
            else 0.0
        )
        bucket["net_profit_pct"] = (
            bucket["net_profit_abs"] / bucket["entry_notional"]
            if bucket["entry_notional"]
            else 0.0
        )
        bucket["avg_net_profit_abs"] = (
            bucket["net_profit_abs"] / closed if closed else 0.0
        )
        bucket["avg_grid_break_abs"] = (
            bucket["grid_break_net_pnl"] / bucket["grid_breaks"]
            if bucket["grid_breaks"]
            else 0.0
        )
        bucket["layer_net_pnl_pct"] = (
            bucket["layer_net_pnl"] / bucket["layer_notional"]
            if bucket["layer_notional"]
            else 0.0
        )
        bucket["avg_layer_net_pnl"] = (
            bucket["layer_net_pnl"] / bucket["layer_fills"]
            if bucket["layer_fills"]
            else 0.0
        )
        bucket["avg_planned_layer_net_pnl_at_tp"] = (
            bucket["planned_layer_net_pnl_at_tp"] / bucket["layer_fills"]
            if bucket["layer_fills"]
            else 0.0
        )
        bucket["avg_layer_bars_held"] = (
            bucket["layer_bars_held_sum"] / bucket["layer_fills"]
            if bucket["layer_fills"]
            else 0.0
        )
        bucket["avg_layer_duration_seconds"] = (
            bucket["layer_duration_seconds_sum"] / bucket["layer_fills"]
            if bucket["layer_fills"]
            else 0.0
        )
        bucket["avg_layer_take_profit_bars"] = (
            bucket["layer_take_profit_bars_sum"]
            / bucket["layer_take_profit_count"]
            if bucket["layer_take_profit_count"]
            else 0.0
        )
        bucket["avg_layer_take_profit_seconds"] = (
            bucket["layer_take_profit_seconds_sum"]
            / bucket["layer_take_profit_count"]
            if bucket["layer_take_profit_count"]
            else 0.0
        )
        bucket["avg_layer_adverse_bars"] = (
            bucket["layer_adverse_bars_sum"] / bucket["layer_adverse_count"]
            if bucket["layer_adverse_count"]
            else 0.0
        )
        bucket["avg_layer_adverse_seconds"] = (
            bucket["layer_adverse_seconds_sum"]
            / bucket["layer_adverse_count"]
            if bucket["layer_adverse_count"]
            else 0.0
        )
        bucket["p95_layer_take_profit_bars"] = (
            float(np.percentile(bucket["layer_take_profit_bars_values"], 95))
            if bucket["layer_take_profit_bars_values"]
            else 0.0
        )
        bucket["p95_layer_adverse_bars"] = (
            float(np.percentile(bucket["layer_adverse_bars_values"], 95))
            if bucket["layer_adverse_bars_values"]
            else 0.0
        )
        bucket.pop("layer_take_profit_bars_values")
        bucket.pop("layer_adverse_bars_values")
        bucket["avg_grid_spacing_pct"] = (
            bucket["grid_spacing_sum"] / bucket["grid_geometry_count"]
            if bucket["grid_geometry_count"]
            else 0.0
        )
        bucket["avg_entry_price_to_base"] = (
            bucket["entry_price_to_base_sum"] / bucket["grid_geometry_count"]
            if bucket["grid_geometry_count"]
            else 0.0
        )
    return {layer: buckets[layer] for layer in sorted(buckets, key=int)}


# ============================================================
# Monte Carlo
# ============================================================
def _percentiles(values: Sequence[float], points=(5, 25, 50, 75, 95)) -> dict:
    if not len(values):
        return {f"p{point}": 0.0 for point in points}
    array = np.asarray(values, dtype=float)
    return {f"p{point}": float(np.percentile(array, point)) for point in points}


def _distribution(values: Sequence[Optional[float]]) -> dict:
    clean = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if not len(clean):
        return {"count": 0, "mean": None, "p5": None, "p25": None,
                "p50": None, "p75": None, "p95": None}
    result = {"count": int(len(clean)), "mean": float(np.mean(clean))}
    result.update(_percentiles(clean))
    return result


def bar_range_statistics(bars: BarSeries) -> dict:
    """Single-candle high-low range as a share of that candle's open."""
    valid = (
        np.isfinite(bars.open)
        & np.isfinite(bars.high)
        & np.isfinite(bars.low)
        & (bars.open > 0.0)
        & (bars.high >= bars.low)
    )
    ranges = (bars.high[valid] - bars.low[valid]) / bars.open[valid]
    if not len(ranges):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "percentiles": _percentiles([], points=(1, 5, 25, 50, 75, 95, 99)),
            "basis": "(high - low) / open",
        }
    percentiles = _percentiles(ranges, points=(1, 5, 25, 50, 75, 95, 99))
    return {
        "count": int(len(ranges)),
        "mean": float(np.mean(ranges)),
        "median": float(np.median(ranges)),
        "percentiles": percentiles,
        "basis": "(high - low) / open",
    }


def _merge_layer_stats(runs: Sequence[dict]) -> dict:
    """Pool every run's trades into one layer distribution."""
    return layer_statistics([trade for run in runs for trade in run["trades"]])


_WORKER_BARS: Optional[BarSeries] = None
_WORKER_CONFIG: Optional[BacktestConfig] = None


def _worker_init(bars: BarSeries, config: BacktestConfig):
    """Hand each process the bars and the config once, not once per run."""
    global _WORKER_BARS, _WORKER_CONFIG
    _WORKER_BARS = bars
    _WORKER_CONFIG = config


def _tag_run(result: dict, run_id: int, start_index: int) -> dict:
    result["summary"]["run_id"] = run_id
    result["equity_curve"]["run_id"] = run_id
    for strategy_run in result.get("strategy_runs", ()):
        strategy_run["run_id"] = (
            f"{run_id}.{strategy_run['account_sequence']}"
        )
        strategy_run["monte_carlo_run_id"] = run_id
        strategy_run["monte_carlo_start_index"] = start_index
    strategy_starts = {
        item["account_sequence"]: item["start_index"]
        for item in result.get("strategy_runs", ())
    }
    for trade in result["trades"]:
        trade["run_id"] = run_id
        trade["strategy_run_id"] = f"{run_id}.{trade.get('account_sequence', 1)}"
        trade["strategy_start_index"] = strategy_starts.get(
            trade.get("account_sequence", 1),
            start_index,
        )
        trade["start_index"] = start_index
    return result


def _worker_run(task) -> dict:
    run_id, start_index = task
    return _tag_run(
        run_once(
            _WORKER_BARS,
            _WORKER_CONFIG,
            start_index=start_index,
            max_bars=_WORKER_CONFIG.monte_carlo.max_bars,
        ),
        run_id,
        start_index,
    )


def resolve_workers(requested: int, tasks: int) -> int:
    if requested and requested > 0:
        return max(1, min(requested, tasks))
    return max(1, min((os.cpu_count() or 1) - 1, tasks))


# ============================================================
# Do grid breaks cluster, or are they scattered at random?
# ============================================================
@dataclass(frozen=True)
class GapAnalysisParams:
    """How inter-grid-break spacing is measured and what it is compared against."""

    window_bars: int = 500    # window width for the count dispersion (Fano) test
    short_gap_bars: int = 10  # "another grid break almost immediately" threshold
    repeats: int = 20         # random baselines drawn per run
    seed: int = 17


def _point_process_stats(
    positions: np.ndarray, span: int, params: GapAnalysisParams
) -> Optional[dict]:
    """Spacing statistics of one run's grid-break positions, in bars.

    The reference is a homogeneous random process over the same span with the
    same number of grid breaks, for which gap CV == 1, lag-1 autocorrelation
    == 0 and the Fano factor == 1.  Values above those mean the breaks arrive
    in bursts; below means they are more evenly spaced than chance.
    """
    positions = np.sort(np.asarray(positions, dtype=float))
    if len(positions) < 3 or span <= 1:
        return None
    gaps = np.diff(positions)
    mean_gap = float(gaps.mean())
    if mean_gap <= 0.0:
        return None

    stats = {
        "count": int(len(positions)),
        "span": int(span),
        "gap_mean": mean_gap,
        "gap_cv": float(gaps.std(ddof=1) / mean_gap) if len(gaps) > 1 else 0.0,
        "short_gap_pct": float((gaps <= params.short_gap_bars).mean()),
        "normalized_gaps": gaps / mean_gap,
    }

    if len(gaps) >= 4:
        first, second = gaps[:-1], gaps[1:]
        if first.std() > 0.0 and second.std() > 0.0:
            stats["gap_lag1"] = float(np.corrcoef(first, second)[0, 1])

    # Count dispersion needs several windows to mean anything.
    if span >= params.window_bars * 3:
        counts, _ = np.histogram(
            positions, bins=np.arange(0, span + params.window_bars, params.window_bars)
        )
        mean_count = counts.mean()
        if mean_count > 0.0:
            stats["fano"] = float(counts.var(ddof=1) / mean_count)
    return stats


def _pool_point_process(entries: Sequence[dict]) -> dict:
    """Average the per-run measures and pool the rate-normalized gaps."""
    if not entries:
        return {}
    normalized = np.concatenate([entry["normalized_gaps"] for entry in entries])
    pooled = {
        "runs": len(entries),
        "grid_breaks": int(sum(entry["count"] for entry in entries)),
        # Gaps are normalized by each run's own mean before pooling: runs have
        # different loss rates, and mixing those would inflate the CV on its own
        # and fake a clustering signal.
        "gap_cv_pooled": float(normalized.std(ddof=1) / normalized.mean()),
        "gap_mean_bars": float(np.mean([entry["gap_mean"] for entry in entries])),
        "gap_cv": float(np.mean([entry["gap_cv"] for entry in entries])),
        "short_gap_pct": float(np.mean([entry["short_gap_pct"] for entry in entries])),
    }
    for key in ("gap_lag1", "fano"):
        values = [entry[key] for entry in entries if key in entry]
        pooled[key] = float(np.mean(values)) if values else None
    pooled["gap_percentiles"] = _percentiles(
        np.concatenate([entry["normalized_gaps"] * entry["gap_mean"] for entry in entries])
    )
    return pooled


def _ratio(observed: Optional[float], reference: Optional[float]) -> Optional[float]:
    if observed is None or reference is None or reference == 0.0:
        return None
    return float(observed / reference)


def loss_gap_statistics(
    summaries: Sequence[dict],
    trades: Sequence[dict],
    params: GapAnalysisParams = GapAnalysisParams(),
) -> dict:
    """Compare grid-break spacing against a random null of the same rate."""
    spans = {summary["run_id"]: summary["bars_survived"] for summary in summaries}
    positions_by_run: dict[int, list] = {}
    for trade in trades:
        if trade["reason"] == "grid_break":
            run_key = trade.get("strategy_run_id", trade["run_id"])
            local_start = trade.get("strategy_start_index", trade.get("start_index", 0))
            positions_by_run.setdefault(run_key, []).append(
                trade["exit_bar"] - local_start
            )

    rng = np.random.default_rng(params.seed)
    observed_entries, random_entries = [], []
    skipped = 0
    for run_id, positions in positions_by_run.items():
        span = spans.get(run_id)
        if span is None:
            continue
        observed = _point_process_stats(np.asarray(positions), span, params)
        if observed is None:
            skipped += 1
            continue
        observed_entries.append(observed)
        # Same span, same number of grid breaks, thrown down uniformly at random.
        for _ in range(params.repeats):
            draw = _point_process_stats(rng.random(len(positions)) * span, span, params)
            if draw is not None:
                random_entries.append(draw)

    observed = _pool_point_process(observed_entries)
    reference = _pool_point_process(random_entries)
    ratios = {
        key: _ratio(observed.get(key), reference.get(key))
        for key in ("gap_cv_pooled", "gap_cv", "fano", "short_gap_pct")
    }
    lag1 = observed.get("gap_lag1")

    dispersion = [value for value in (ratios["gap_cv_pooled"], ratios["fano"]) if value]
    strongest = max(dispersion) if dispersion else 1.0
    if strongest >= 1.2 or (lag1 is not None and lag1 >= 0.15):
        verdict = "clustered"
    elif strongest <= 0.85:
        verdict = "more regular than random"
    else:
        verdict = "indistinguishable from random"

    return {
        "params": asdict(params),
        "runs_analyzed": len(observed_entries),
        "runs_skipped": skipped,
        "observed": observed,
        "random": reference,
        "ratios": ratios,
        "gap_lag1_excess": (
            None if lag1 is None or reference.get("gap_lag1") is None
            else float(lag1 - reference["gap_lag1"])
        ),
        "verdict": verdict,
    }


def run_monte_carlo(
    logger: logging.Logger,
    bars: BarSeries,
    config: BacktestConfig,
) -> dict:
    if config.capital.profit_handling == "both":
        raise ValueError(
            "run_monte_carlo needs one profit_handling mode; use main() to compare both"
        )
    monte_carlo = config.monte_carlo
    available_last_start = len(bars) - monte_carlo.min_bars
    if available_last_start < 0:
        raise ValueError(
            f"Only {len(bars)} bars available, need at least "
            f"min_bars={monte_carlo.min_bars}"
        )

    fraction_last_start = max(
        0, int(len(bars) * monte_carlo.start_fraction) - 1
    )
    last_start = min(available_last_start, fraction_last_start)
    logger.info(
        f"Monte Carlo start range | 0..{last_start} "
        f"(first {monte_carlo.start_fraction:.0%} of {len(bars)} bars)"
    )

    # Drawn here, so a 1-worker run and a 16-worker run replay the same starts.
    rng = random.Random(monte_carlo.seed)
    tasks = [
        (run_id, rng.randint(0, last_start)) for run_id in range(monte_carlo.runs)
    ]
    workers = resolve_workers(monte_carlo.workers, len(tasks))
    report_every = max(1, monte_carlo.runs // 10)
    runs: List[dict] = []

    def progress():
        if len(runs) % report_every == 0 or len(runs) == len(tasks):
            ruined = sum(run["summary"]["ruined"] for run in runs)
            logger.info(
                f"Monte Carlo | {len(runs)}/{len(tasks)} runs "
                f"| ruined so far: {ruined}"
            )

    if workers == 1:
        _worker_init(bars, config)
        for task in tasks:
            runs.append(_worker_run(task))
            progress()
    else:
        logger.info(f"Monte Carlo | {len(tasks)} runs over {workers} processes")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(bars, config),
        ) as pool:
            futures = [pool.submit(_worker_run, task) for task in tasks]
            for future in as_completed(futures):
                runs.append(future.result())
                progress()
        runs.sort(key=lambda run: run["summary"]["run_id"])

    monte_carlo_summaries = [run["summary"] for run in runs]
    summaries = [
        strategy_run
        for run in runs
        for strategy_run in run.get("strategy_runs", ())
    ]
    ruined = [s for s in summaries if s["ruined"]]
    breached = [s for s in summaries if s["grid_breached"]]
    doubled = [s for s in summaries if s["doubled_before_grid_breach"]]
    all_trades = [trade for run in runs for trade in run["trades"]]
    all_grid_failures = [
        failure for summary in summaries
        for failure in summary["grid_failures"]
    ]

    total_trades = len(all_trades)
    profitable = sum(trade["net_pnl"] > 0.0 for trade in all_trades)
    cycle_same_bar = sum(trade["same_bar"] for trade in all_trades)
    same_bar = sum(
        trade["reason"] == "take_profit"
        and any(
            fill.get("layer", 0) > 1
            and fill.get("bar") == trade.get("exit_bar")
            for fill in trade.get("fills", ())
        )
        for trade in all_trades
    )
    multi_fill = sum(trade["multiple_fills_same_bar"] for trade in all_trades)
    net_profit = sum(trade["net_pnl"] for trade in all_trades)
    fees = sum(trade["fees"] for trade in all_trades)
    entry_notional = sum(trade["entry_notional"] for trade in all_trades)

    exit_counts: dict[str, int] = {}
    for trade in all_trades:
        exit_counts[trade["reason"]] = exit_counts.get(trade["reason"], 0) + 1

    # A full-layer grid break is a large, recurring realized loss.  The point
    # of these numbers is the exchange rate between the two: how many
    # take-profit cycles one break gives back, and how often it fires.
    stop_trades = [trade for trade in all_trades if trade["reason"] == "grid_break"]
    winners = [trade for trade in all_trades if trade["net_pnl"] > 0.0]
    stop_net = sum(trade["net_pnl"] for trade in stop_trades)
    avg_stop = stop_net / len(stop_trades) if stop_trades else 0.0
    avg_win = (
        sum(trade["net_pnl"] for trade in winners) / len(winners) if winners else 0.0
    )

    aggregate = {
        "runs": len(summaries),
        "monte_carlo_runs": len(monte_carlo_summaries),
        "profit_handling": config.capital.profit_handling,
        "stop_at_first_grid_breach": config.capital.stop_at_first_grid_breach,
        "grid_break_is_ruin": config.capital.grid_break_is_ruin,
        "ruin_equity_pct_of_peak_balance": config.account.ruin_equity_pct,
        "grid_breached_runs": len(breached),
        "grid_breach_rate": len(breached) / len(summaries) if summaries else 0.0,
        "grid_failures": len(all_grid_failures),
        "account_resets": sum(s["account_reset_count"] for s in monte_carlo_summaries),
        "reset_capital_injected": sum(
            s["reset_capital_injected"] for s in monte_carlo_summaries
        ),
        "reset_capital_removed": sum(
            s["reset_capital_removed"] for s in monte_carlo_summaries
        ),
        "reset_capital_net_injected": sum(
            s["reset_capital_net_injected"] for s in monte_carlo_summaries
        ),
        "grid_failures_per_run": (
            len(all_grid_failures) / len(summaries) if summaries else 0.0
        ),
        "grid_breach_reasons": {
            reason: sum(
                failure["reason"] == reason for failure in all_grid_failures
            )
            for reason in sorted({
                failure["reason"] for failure in all_grid_failures
            })
        },
        "withdrawn_profit_before_grid_breach_mean": float(np.mean([
            s["withdrawn_profit_before_grid_breach"] for s in summaries
        ])),
        "withdrawn_profit_before_grid_breach": _percentiles([
            s["withdrawn_profit_before_grid_breach"] for s in summaries
        ]),
        "withdrawn_profit_on_breached_runs": _distribution([
            s["withdrawn_profit_before_grid_breach"] for s in breached
        ]),
        "doubled_before_grid_breach_runs": len(doubled),
        "double_before_grid_breach_rate": (
            len(doubled) / len(summaries) if summaries else 0.0
        ),
        "double_before_grid_breach_rate_on_breached_runs": (
            sum(s["doubled_before_grid_breach"] for s in breached) / len(breached)
            if breached else 0.0
        ),
        "bars_to_double": _distribution([s["bars_to_double"] for s in doubled]),
        "ruined_runs": len(ruined),
        "ruin_rate": len(ruined) / len(summaries) if summaries else 0.0,
        "grid_break_ruin_runs": sum(
            any(
                f.get("account_failure_reason") == "grid_break_ruin"
                for f in s["grid_failures"]
            )
            for s in summaries
        ),
        "liquidation_runs": sum(
            any(
                f.get("account_failure_reason", f["reason"]) == "liquidation"
                for f in s["grid_failures"]
            )
            for s in summaries
        ),
        "ruin_threshold_runs": sum(
            any(
                f.get("account_failure_reason", f["reason"]) == "ruin_threshold"
                for f in s["grid_failures"]
            )
            for s in summaries
        ),
        "grid_infeasible_runs": sum(
            any(
                f.get("account_failure_reason", f["reason"]) == "grid_infeasible"
                for f in s["grid_failures"]
            )
            for s in summaries
        ),
        "survived_runs": len(summaries) - len(ruined),
        "bars_survived_mean": float(np.mean([s["bars_survived"] for s in summaries])),
        "bars_survived": _percentiles([s["bars_survived"] for s in summaries]),
        "bars_to_grid_break": _distribution([
            s["bars_to_grid_break"] for s in breached
        ]),
        "return_before_grid_break_pct": _distribution([
            s["return_before_grid_break_pct"] for s in breached
        ]),
        "bars_to_ruin": _distribution([s["bars_to_ruin"] for s in ruined]),
        "return_before_ruin_pct": _distribution([
            s["return_before_ruin_pct"] for s in ruined
        ]),
        "elapsed_years_mean": float(np.mean([s["elapsed_years"] for s in summaries])),
        "elapsed_years": _percentiles([s["elapsed_years"] for s in summaries]),
        "total_return_pct_mean": float(np.mean([s["total_return_pct"] for s in summaries])),
        "total_return_pct": _percentiles([s["total_return_pct"] for s in summaries]),
        "total_return_after_reset_cost_pct_mean": float(np.mean([
            s["total_return_after_reset_cost_pct"] for s in summaries
        ])),
        "total_return_after_reset_cost_pct": _percentiles([
            s["total_return_after_reset_cost_pct"] for s in summaries
        ]),
        "max_drawdown_pct_mean": float(np.mean([s["max_drawdown_pct"] for s in summaries])),
        "max_margin_usage_mean": float(np.mean([s["max_margin_usage"] for s in summaries])),
        "total_trades": total_trades,
        "trades_per_run": total_trades / len(summaries) if summaries else 0.0,
        "trades_per_run_percentiles": _percentiles([s["total_trades"] for s in summaries]),
        "profitable_trades": profitable,
        "losing_trades": total_trades - profitable,
        "win_rate": profitable / total_trades if total_trades else 0.0,
        "total_net_profit_abs": net_profit,
        "total_fees_abs": fees,
        "total_entry_notional": entry_notional,
        "net_profit_pct_of_notional": net_profit / entry_notional if entry_notional else 0.0,
        "fees_pct_of_notional": fees / entry_notional if entry_notional else 0.0,
        "avg_net_profit_abs_per_trade": net_profit / total_trades if total_trades else 0.0,
        # --- grid break ---
        "grid_break_trades": len(stop_trades),
        "grid_break_pct_of_trades": len(stop_trades) / total_trades if total_trades else 0.0,
        "grid_break_runs": sum(s["grid_break_trades"] > 0 for s in summaries),
        "grid_breaks_per_run": len(stop_trades) / len(summaries) if summaries else 0.0,
        "grid_break_net_pnl_abs": stop_net,
        "avg_grid_break_abs": avg_stop,
        "avg_grid_break_pct_of_balance": (
            float(np.mean([trade["return_on_balance"] for trade in stop_trades]))
            if stop_trades else 0.0
        ),
        "avg_win_abs": avg_win,
        "wins_per_grid_break": abs(avg_stop) / avg_win if avg_win > 0.0 and stop_trades else 0.0,
        "cycle_same_bar_trades": cycle_same_bar,
        "cycle_same_bar_trade_pct": cycle_same_bar / total_trades if total_trades else 0.0,
        "same_bar_trades": same_bar,
        "same_bar_trade_pct": same_bar / total_trades if total_trades else 0.0,
        "multiple_fills_same_bar_trades": multi_fill,
        "multiple_fills_same_bar_pct": multi_fill / total_trades if total_trades else 0.0,
        "bar_range_pct": bar_range_statistics(bars),
        "max_layers_seen": max((s["max_layers_seen"] for s in summaries), default=0),
        "unfilled_layers": sum(s["unfilled_layers"] for s in summaries),
        "cooldown_blocked_bars": sum(s["cooldown_blocked_bars"] for s in summaries),
        "volatility_blocked_bars": sum(
            s.get("volatility_blocked_bars", 0) for s in summaries
        ),
        "reversals": sum(s["reversals"] for s in summaries),
        "dual_path_bars": sum(s["dual_path_bars"] for s in summaries),
        "up_first_worse_bars": sum(s["up_first_worse_bars"] for s in summaries),
        "exit_counts": exit_counts,
    }
    aggregate["up_first_worse_pct"] = (
        aggregate["up_first_worse_bars"] / aggregate["dual_path_bars"]
        if aggregate["dual_path_bars"]
        else 0.0
    )
    aggregate["distributions"] = {
        "final_equity": _distribution([s["final_equity"] for s in summaries]),
        "final_total_wealth": _distribution([
            s["final_total_wealth"] for s in summaries
        ]),
        "final_total_wealth_after_reset_transfers": _distribution([
            s["final_total_wealth_after_reset_transfers"] for s in summaries
        ]),
        "reserve_balance": _distribution([
            s["reserve_balance"] for s in summaries
        ]),
        "reset_capital_injected": _distribution([
            s["reset_capital_injected"] for s in summaries
        ]),
        "reset_capital_removed": _distribution([
            s["reset_capital_removed"] for s in summaries
        ]),
        "peak_equity": _distribution([s["peak_equity"] for s in summaries]),
        "peak_realized_balance": _distribution([s["peak_realized_balance"] for s in summaries]),
        "peak_equity_multiple": _distribution([s["peak_equity_multiple"] for s in summaries]),
        "peak_return_pct": _distribution([s["peak_return_pct"] for s in summaries]),
        "bars_to_peak": _distribution([s["bars_to_peak"] for s in summaries]),
        "years_to_peak": _distribution([s["years_to_peak"] for s in summaries]),
        "total_return_pct": _distribution([s["total_return_pct"] for s in summaries]),
        "total_return_after_reset_cost_pct": _distribution([
            s["total_return_after_reset_cost_pct"] for s in summaries
        ]),
        "cagr": _distribution([s["cagr"] for s in summaries]),
        "peak_cagr": _distribution([s["peak_cagr"] for s in summaries]),
        "sharpe": _distribution([s["sharpe"] for s in summaries]),
        "max_drawdown_pct": _distribution([s["max_drawdown_pct"] for s in summaries]),
        "max_margin_usage": _distribution([s["max_margin_usage"] for s in summaries]),
        "bars_survived": _distribution([s["bars_survived"] for s in summaries]),
        "elapsed_years": _distribution([s["elapsed_years"] for s in summaries]),
        "total_trades": _distribution([s["total_trades"] for s in summaries]),
        "grid_break_trades": _distribution([s["grid_break_trades"] for s in summaries]),
    }
    return {
        "aggregate": aggregate,
        "direction_stats": direction_statistics(all_trades),
        "layer_stats": _merge_layer_stats(runs),
        "loss_gaps": loss_gap_statistics(summaries, all_trades, config.gap_analysis),
        "runs": summaries,
        "monte_carlo_runs": monte_carlo_summaries,
        "trades": all_trades,
        "equity_curves": [run["equity_curve"] for run in runs],
        "workers": workers,
    }


# ============================================================
# Trade level export (for downstream statistical analysis)
# ============================================================
TRADE_COLUMNS = [
    "run_id", "strategy_run_id", "start_index", "strategy_start_index",
    "account_sequence", "index", "reason",
    "opened_at", "closed_at", "entry_bar", "exit_bar", "bars_held",
    "duration_seconds", "same_bar", "max_fills_in_one_bar",
    "layers", "full_layers", "qty",
    "entry_notional", "exit_notional",
    "first_entry_price", "last_entry_price", "avg_entry_price", "exit_price",
    "worst_price", "best_price", "mae_pct", "mfe_pct", "exit_pct_from_avg",
    "gross_pnl", "entry_fees", "exit_fee", "fees", "net_pnl", "profit_pct",
    "balance_before", "balance_after", "return_on_balance", "shortfall",
]


def trades_dataframe(trades: Sequence[dict]) -> pd.DataFrame:
    """One row per closed trade: time, price, size and pnl of every cycle."""
    if not trades:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    frame = pd.DataFrame(
        [{key: value for key, value in trade.items() if key != "fills"} for trade in trades]
    )
    if frame["opened_at"].dtype.kind == "M" and frame["closed_at"].dtype.kind == "M":
        frame["duration_seconds"] = (
            frame["closed_at"] - frame["opened_at"]
        ).dt.total_seconds()
    else:
        frame["duration_seconds"] = np.nan
    ordered = [column for column in TRADE_COLUMNS if column in frame.columns]
    return frame[ordered + [c for c in frame.columns if c not in ordered]]


def fills_dataframe(trades: Sequence[dict]) -> pd.DataFrame:
    """One row per filled layer, joinable back on (run_id, trade_index)."""
    rows = [
        {
            "run_id": trade.get("run_id"),
            "strategy_run_id": trade.get("strategy_run_id"),
            "strategy_start_index": trade.get("strategy_start_index"),
            "capital_mode": trade.get("capital_mode"),
            "trade_index": trade["index"],
            "trade_reason": trade["reason"],
            **fill,
        }
        for trade in trades
        for fill in trade.get("fills", ())
    ]
    frame = pd.DataFrame(rows)
    if (
        not frame.empty
        and "time" in frame.columns
        and "exit_time" in frame.columns
    ):
        start = pd.to_datetime(frame["time"], errors="coerce")
        end = pd.to_datetime(frame["exit_time"], errors="coerce")
        frame["layer_duration_seconds"] = (
            end - start
        ).dt.total_seconds().clip(lower=0.0)
    if (
        not frame.empty
        and "time" in frame.columns
        and "boundary_time" in frame.columns
    ):
        start = pd.to_datetime(frame["time"], errors="coerce")
        end = pd.to_datetime(frame["boundary_time"], errors="coerce")
        frame["boundary_duration_seconds"] = (
            end - start
        ).dt.total_seconds().clip(lower=0.0)
    return frame


def write_table(logger: logging.Logger, frame: pd.DataFrame, path: str, label: str):
    """Write parquet when the suffix asks for it, CSV otherwise."""
    if frame.empty:
        logger.warning(f"No {label} to export, skipping {path}")
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if path.endswith(".parquet"):
        try:
            frame.to_parquet(path, index=False)
        except (ImportError, ValueError) as error:
            path = f"{os.path.splitext(path)[0]}.csv"
            logger.warning(f"Parquet unavailable ({error}), writing {path} instead")
            frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)
    logger.info(f"Exported {len(frame)} {label} rows to {path}")


# ============================================================
# Reporting
# ============================================================
def log_direction_table(logger: logging.Logger, direction_stats: dict, aggregate: dict):
    """Only interesting once both sides were traded, i.e. reversal is on."""
    if len(direction_stats) < 2 and not aggregate.get("reversals"):
        return
    logger.info(
        f"{'SIDE':<7}{'TRADES':>9}{'SHARE':>8}{'WINRATE':>9}{'GRIDBRK':>9}"
        f"{'NET_PNL':>18}{'PNL/NOTIONAL':>15}   reversals "
        f"{aggregate.get('reversals', 0)}"
    )
    for name, bucket in direction_stats.items():
        logger.info(
            f"{name:<7}{bucket['trades']:>9}"
            f"{bucket['trade_pct'] * 100:>7.2f}%"
            f"{bucket['win_rate'] * 100:>8.2f}%"
            f"{bucket['grid_break_pct'] * 100:>8.2f}%"
            f"{bucket['net_profit_abs']:>18.2f}"
            f"{bucket['net_profit_pct'] * 100:>14.3f}%"
        )


def log_loss_gaps(logger: logging.Logger, gaps: dict):
    """Observed spacing of grid breaks next to a same-rate random null."""
    observed, reference = gaps.get("observed"), gaps.get("random")
    if not observed or not reference:
        return
    percentiles = observed["gap_percentiles"]
    logger.info(
        f"GRIDGAP | runs {gaps['runs_analyzed']} "
        f"| grid breaks {observed['grid_breaks']} "
        f"| mean gap {observed['gap_mean_bars']:.0f} bars "
        f"| p5 {percentiles['p5']:.0f} | p50 {percentiles['p50']:.0f} "
        f"| p95 {percentiles['p95']:.0f}"
    )
    ratios = gaps["ratios"]
    logger.info(
        f"CLUSTER | gap CV {observed['gap_cv_pooled']:.3f} vs random "
        f"{reference['gap_cv_pooled']:.3f} (x{_fmt(ratios['gap_cv_pooled'], 2)}) "
        f"| Fano {_fmt(observed.get('fano'), 2)} vs {_fmt(reference.get('fano'), 2)} "
        f"(x{_fmt(ratios['fano'], 2)}) "
        f"| lag-1 {_fmt(observed.get('gap_lag1'), 3)} "
        f"| short gaps {observed['short_gap_pct'] * 100:.1f}% vs "
        f"{reference['short_gap_pct'] * 100:.1f}% "
        f"(x{_fmt(ratios['short_gap_pct'], 2)}) -> {gaps['verdict']}"
    )


def log_layer_table(logger: logging.Logger, layer_stats: dict):
    logger.info(
        f"{'LAYERS':<7}{'GRID_GAP':>10}{'ENTRY/BASE':>12}"
        f"{'REACHED':>9}{'REACH%':>8}{'TP%':>8}"
        f"{'TP_CNT':>9}{'ADV_CNT':>9}{'SAMEBAR':>9}{'NET_PNL':>18}"
        f"{'PNL/NOTIONAL':>15}{'LAYER_PNL':>16}{'AVG_LAYER':>12}"
        f"{'AVG_HOLD':>10}{'TP_HOLD':>12}{'TP_P95':>9}"
        f"{'ADV_HOLD':>10}{'ADV_P95':>10}"
    )
    for layer, bucket in layer_stats.items():
        logger.info(
            f"{layer:<7}"
            f"{bucket['avg_grid_spacing_pct'] * 100:>9.3f}%"
            f"{bucket['avg_entry_price_to_base']:>12.5f}"
            f"{bucket['reached_trades']:>9}"
            f"{bucket['reach_pct'] * 100:>7.2f}%"
            f"{bucket['take_profit_pct'] * 100:>7.2f}%"
            f"{bucket['layer_take_profit_count']:>9}"
            f"{bucket['layer_adverse_count']:>9}"
            f"{bucket['same_bar_pct'] * 100:>8.2f}%"
            f"{bucket['net_profit_abs']:>18.2f}"
            f"{bucket['net_profit_pct'] * 100:>14.3f}%"
            f"{bucket['layer_net_pnl']:>16.2f}"
            f"{bucket['avg_layer_net_pnl']:>12.2f}"
            f"{bucket['avg_layer_bars_held']:>10.1f}"
            f"{bucket['avg_layer_take_profit_bars']:>12.1f}"
            f"{bucket['p95_layer_take_profit_bars']:>9.1f}"
            f"{bucket['avg_layer_adverse_bars']:>10.2f}"
            f"{bucket['p95_layer_adverse_bars']:>10.2f}"
        )


def _fmt(value: Optional[float], digits: int = 2, scale: float = 1.0) -> str:
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


def _fmt_dist(distribution: dict, *, digits: int = 2, scale: float = 1.0) -> str:
    if not distribution or not distribution.get("count"):
        return "count 0"
    return (
        f"count {distribution['count']} "
        f"| mean {_fmt(distribution['mean'], digits, scale)} "
        f"| p50 {_fmt(distribution['p50'], digits, scale)} "
        f"| p95 {_fmt(distribution['p95'], digits, scale)}"
    )


def log_report(logger: logging.Logger, result: dict, config: BacktestConfig):
    aggregate = result["aggregate"]
    strategy = config.strategy
    account = config.account
    grid = ", ".join(f"{item:.3%}" for item in strategy.grid_deviation_pcts)
    logger.info("-" * 96)
    logger.info(
        f"STRATEGY| target {strategy.take_profit_pct:.3%} of cycle balance "
        f"| layers {strategy.layer_count} "
        f"| grid [{grid}] "
        f"| max cycle {strategy.max_cycle_bars or 'none'} bars "
        f"| preflight full grid"
    )
    logger.info(
        f"GRIDBRK | full-layer break {strategy.stop_at_full_layers}"
        f" | break is ruin {config.capital.grid_break_is_ruin}"
    )
    logger.info(
        f"ACCOUNT | equity {account.initial_equity:.0f} "
        f"| leverage {account.leverage}x "
        f"| fee taker {account.taker_fee_pct}% / maker {account.maker_fee_pct}% "
        f"| maint {account.maintenance_margin_pct}% "
        f"| margin cap {account.margin_usage_cap_pct:.0%}"
    )
    breach_profit = aggregate["withdrawn_profit_before_grid_breach"]
    logger.info(
        f"CAPITAL | mode {config.capital.profit_handling} "
        f"| stop at first breach {config.capital.stop_at_first_grid_breach} "
        f"| ruin if balance <= peak*{account.ruin_equity_pct:.2f} "
        f"| breached {aggregate['grid_breached_runs']}/{aggregate['runs']} "
        f"({aggregate['grid_breach_rate'] * 100:.2f}%) "
        f"| failures {aggregate['grid_failures']} "
        f"| resets {aggregate['account_resets']} "
        f"| injected {aggregate['reset_capital_injected']:.2f} "
        f"| removed {aggregate['reset_capital_removed']:.2f} "
        f"| reasons {aggregate['grid_breach_reasons']}"
    )
    if config.capital.profit_handling == "withdraw":
        logger.info(
            f"WITHDRAW| accumulated before breach mean "
            f"{aggregate['withdrawn_profit_before_grid_breach_mean']:.2f} "
            f"| p5 {breach_profit['p5']:.2f} | p50 {breach_profit['p50']:.2f} "
            f"| p95 {breach_profit['p95']:.2f}"
        )
    else:
        logger.info(
            f"COMPOUND| reached {config.capital.double_target_multiple:g}x before breach "
            f"{aggregate['doubled_before_grid_breach_runs']}/{aggregate['runs']} "
            f"({aggregate['double_before_grid_breach_rate'] * 100:.2f}%)"
        )
    logger.info(
        f"SURVIVE | MC paths {aggregate['monte_carlo_runs']} "
        f"| strategy runs {aggregate['runs']} "
        f"| ruined {aggregate['ruined_runs']} ({aggregate['ruin_rate'] * 100:.2f}%) "
        f"| liquidation {aggregate['liquidation_runs']} "
        f"| threshold {aggregate['ruin_threshold_runs']} "
        f"| grid-break-ruin {aggregate['grid_break_ruin_runs']} "
        f"| grid infeasible {aggregate['grid_infeasible_runs']} "
        f"| survived {aggregate['survived_runs']}"
    )
    logger.info(
        f"BREAK_T | bars before grid break: "
        f"{_fmt_dist(aggregate['bars_to_grid_break'], digits=0)} "
        f"| return before break: "
        f"{_fmt_dist(aggregate['return_before_grid_break_pct'], digits=2, scale=100)}%"
    )
    logger.info(
        f"RUIN_T  | bars before ruin: "
        f"{_fmt_dist(aggregate['bars_to_ruin'], digits=0)} "
        f"| return before ruin: "
        f"{_fmt_dist(aggregate['return_before_ruin_pct'], digits=2, scale=100)}%"
    )
    bars = aggregate["bars_survived"]
    logger.info(
        f"BARS    | mean {aggregate['bars_survived_mean']:.0f} "
        f"| p5 {bars['p5']:.0f} | p25 {bars['p25']:.0f} | p50 {bars['p50']:.0f} "
        f"| p75 {bars['p75']:.0f} | p95 {bars['p95']:.0f}"
    )
    years = aggregate["elapsed_years"]
    logger.info(
        f"TIME    | survived mean {aggregate['elapsed_years_mean']:.3f}y "
        f"| p5 {years['p5']:.3f}y | p25 {years['p25']:.3f}y "
        f"| p50 {years['p50']:.3f}y | p75 {years['p75']:.3f}y "
        f"| p95 {years['p95']:.3f}y"
    )
    returns = aggregate["total_return_pct"]
    logger.info(
        f"RETURN  | trading acct mean {aggregate['total_return_pct_mean'] * 100:.2f}% "
        f"| p5 {returns['p5'] * 100:.2f}% | p50 {returns['p50'] * 100:.2f}% "
        f"| p95 {returns['p95'] * 100:.2f}% "
        f"| avg MaxDD {aggregate['max_drawdown_pct_mean'] * 100:.2f}% "
        f"| avg peak IM/equity {aggregate['max_margin_usage_mean'] * 100:.1f}%"
    )
    net_returns = aggregate["total_return_after_reset_cost_pct"]
    logger.info(
        f"NETRET  | after reset capital mean "
        f"{aggregate['total_return_after_reset_cost_pct_mean'] * 100:.2f}% "
        f"| p5 {net_returns['p5'] * 100:.2f}% "
        f"| p50 {net_returns['p50'] * 100:.2f}% "
        f"| p95 {net_returns['p95'] * 100:.2f}%"
    )
    distributions = aggregate["distributions"]
    peak = distributions["peak_equity"]
    peak_multiple = distributions["peak_equity_multiple"]
    realized_peak = distributions["peak_realized_balance"]
    logger.info(
        f"PEAK    | equity mean {_fmt(peak['mean'])} | p50 {_fmt(peak['p50'])} "
        f"| p95 {_fmt(peak['p95'])} | realized p50 {_fmt(realized_peak['p50'])} "
        f"| multiple mean {_fmt(peak_multiple['mean'])}x "
        f"| p50 {_fmt(peak_multiple['p50'])}x | p95 {_fmt(peak_multiple['p95'])}x"
    )
    cagr = distributions["cagr"]
    peak_cagr = distributions["peak_cagr"]
    sharpe = distributions["sharpe"]
    logger.info(
        f"RISKADJ | CAGR mean {_fmt(cagr['mean'], scale=100)}% "
        f"| p50 {_fmt(cagr['p50'], scale=100)}% | p95 {_fmt(cagr['p95'], scale=100)}% "
        f"| peak CAGR mean {_fmt(peak_cagr['mean'], scale=100)}% "
        f"| Sharpe mean {_fmt(sharpe['mean'], 3)} "
        f"| p50 {_fmt(sharpe['p50'], 3)} | p95 {_fmt(sharpe['p95'], 3)}"
    )
    trade_counts = aggregate["trades_per_run_percentiles"]
    logger.info(
        f"TRADES  | total {aggregate['total_trades']} "
        f"| per run mean {aggregate['trades_per_run']:.1f} "
        f"| p5 {trade_counts['p5']:.0f} | p25 {trade_counts['p25']:.0f} "
        f"| p50 {trade_counts['p50']:.0f} | p75 {trade_counts['p75']:.0f} "
        f"| p95 {trade_counts['p95']:.0f} "
        f"| win {aggregate['win_rate'] * 100:.2f}%"
    )
    logger.info(
        f"PNL     | net {aggregate['total_net_profit_abs']:.2f} "
        f"| fees {aggregate['total_fees_abs']:.2f} "
        f"({aggregate['fees_pct_of_notional'] * 100:.4f}% of notional) "
        f"| avg {aggregate['avg_net_profit_abs_per_trade']:.4f}/trade"
    )
    logger.info(
        f"GRIDBRK| trades {aggregate['grid_break_trades']} "
        f"({aggregate['grid_break_pct_of_trades'] * 100:.2f}% of trades) "
        f"| runs hit {aggregate['grid_break_runs']}/{aggregate['runs']} "
        f"| per run {aggregate['grid_breaks_per_run']:.2f} "
        f"| avg {aggregate['avg_grid_break_abs']:.2f} "
        f"({aggregate['avg_grid_break_pct_of_balance'] * 100:.2f}% of balance) "
        f"| avg win {aggregate['avg_win_abs']:.4f} "
        f"| wins to recover one break {aggregate['wins_per_grid_break']:.1f}"
    )
    logger.info(
        f"SAMEBAR | adverse fill + TP in one bar {aggregate['same_bar_trades']} "
        f"({aggregate['same_bar_trade_pct'] * 100:.2f}%) "
        f"| cycle open+close in one bar {aggregate['cycle_same_bar_trades']} "
        f"({aggregate['cycle_same_bar_trade_pct'] * 100:.2f}%) "
        f"| multi-layer fills in one bar {aggregate['multiple_fills_same_bar_trades']} "
        f"({aggregate['multiple_fills_same_bar_pct'] * 100:.2f}%)"
    )
    range_stats = aggregate["bar_range_pct"]
    range_pct = range_stats["percentiles"]
    logger.info(
        f"K_RANGE | basis {range_stats['basis']} | bars {range_stats['count']} "
        f"| mean {_fmt(range_stats['mean'], 4, 100)}% "
        f"| median {_fmt(range_stats['median'], 4, 100)}% "
        f"| p1 {_fmt(range_pct['p1'], 4, 100)}% "
        f"| p5 {_fmt(range_pct['p5'], 4, 100)}% "
        f"| p25 {_fmt(range_pct['p25'], 4, 100)}% "
        f"| p75 {_fmt(range_pct['p75'], 4, 100)}% "
        f"| p95 {_fmt(range_pct['p95'], 4, 100)}% "
        f"| p99 {_fmt(range_pct['p99'], 4, 100)}%"
    )
    logger.info(
        f"PATH    | dual-path bars {aggregate['dual_path_bars']} "
        f"| up-first was worse {aggregate['up_first_worse_bars']} "
        f"({aggregate['up_first_worse_pct'] * 100:.2f}%) "
        f"| exits {aggregate['exit_counts']} "
        f"| layers refused by margin {aggregate['unfilled_layers']}"
    )
    log_direction_table(logger, result.get("direction_stats", {}), aggregate)
    log_loss_gaps(logger, result.get("loss_gaps", {}))
    log_layer_table(logger, result["layer_stats"])
    logger.info("-" * 96)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_safe(value):
    """Replace inf/NaN with null so a finished run always writes its report.

    ``allow_nan=False`` is what makes the report readable by strict JSON
    parsers, but it raises at the very end of a run that already cost minutes.
    Sanitizing up front keeps that guarantee without the crash.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


GRID_FAILURE_HINTS = {
    "target_already_met_without_new_order": (
        "the existing position already exceeds the cash target at this layer's "
        "take-profit price; remove later grid_deviation_pcts entries or reduce "
        "the affected layer spacing"
    ),
    "non_positive_required_notional": (
        "the sizing equation produced no positive order; remove later "
        "grid_deviation_pcts entries or inspect existing_net_at_take_profit"
    ),
    "non_positive_incremental_return": (
        "the layer's price edge does not cover its entry and exit fees; increase "
        "the affected grid_deviation_pcts item or reduce fees"
    ),
    "margin_cap_exceeded": (
        "initial margin exceeds margin_usage_cap_pct; reduce take_profit_pct or "
        "remove later grid_deviation_pcts entries, or increase leverage/"
        "margin_usage_cap_pct"
    ),
    "grid_break_equity_non_positive": (
        "the full-grid break loss is greater than the starting equity; reduce "
        "take_profit_pct, reduce cumulative exposure, or make the final break "
        "closer to the last entry"
    ),
    "non_positive_equity_after_order": (
        "mark-to-market equity after the order fee is non-positive; reduce grid "
        "size, take_profit_pct, or remove later grid_deviation_pcts entries"
    ),
    "adverse_step_ge_100pct": (
        "an adverse grid step reached 100%; every grid_deviation_pcts item must "
        "stay below 100%"
    ),
    "take_profit_deviation_ge_100pct": (
        "the take-profit deviation reached 100%; every grid_deviation_pcts item "
        "must stay below 100%"
    ),
    "non_positive_layer_price": (
        "the planned layer price is non-positive; reduce grid spacing or layers"
    ),
}


def _diagnostic_number(value: Optional[float], digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _preflight_base_entry_price(result: dict, fallback: float) -> float:
    return next(
        (
            row.get("entry_price")
            for row in result["layers"]
            if row.get("entry_price") is not None
        ),
        fallback,
    )


def _preflight_grid_break_metrics(
    result: dict,
    config: BacktestConfig,
) -> dict:
    rows = [
        row for row in result["layers"]
        if row.get("grid_break_net_pnl") is not None
    ]
    if not rows:
        return {
            "grid_break_loss_pct": None,
            "grid_break_equity_pct": None,
            "grid_break_net_pnl": None,
        }
    last = rows[-1]
    net_pnl = last["grid_break_net_pnl"]
    equity_after_break = config.account.initial_equity + net_pnl
    return {
        "grid_break_loss_pct": (
            -net_pnl / config.account.initial_equity
        ),
        "grid_break_equity_pct": (
            equity_after_break / config.account.initial_equity
        ),
        "grid_break_net_pnl": net_pnl,
    }


def _preflight_summary_row(
    result: dict,
    config: BacktestConfig,
    base_entry_price: float,
) -> dict:
    rows = [
        row for row in result["layers"]
        if row.get("entry_price") is not None
    ]
    sized_rows = [
        row for row in rows
        if row.get("cumulative_notional_after") is not None
    ]
    if not rows:
        return {
            "side": result["side"],
            "max_layer": 0,
            "final_entry_x": None,
            "cum_bal": None,
            "min_equity": None,
            "peak_im_eq": None,
            "grid_break_loss_pct": None,
            "grid_break_equity_pct": None,
        }
    grid_break = _preflight_grid_break_metrics(result, config)
    deepest = rows[-1]
    equity_base = config.account.initial_equity
    return {
        "side": result["side"],
        "max_layer": deepest.get("layer"),
        "final_entry_x": (
            deepest["entry_price"] / base_entry_price
            if base_entry_price else None
        ),
        "cum_bal": (
            max(row["cumulative_notional_after"] for row in sized_rows) / equity_base
            if sized_rows else None
        ),
        "min_equity": (
            min(row["equity_mtm_before_order"] for row in sized_rows) / equity_base
            if sized_rows else None
        ),
        "peak_im_eq": (
            max(
                row["margin_usage_pct"]
                for row in sized_rows
                if row.get("margin_usage_pct") is not None
            )
            if any(row.get("margin_usage_pct") is not None for row in sized_rows)
            else None
        ),
        "grid_break_loss_pct": grid_break["grid_break_loss_pct"],
        "grid_break_equity_pct": grid_break["grid_break_equity_pct"],
    }


def _log_preflight_failure_diagnostic(
    logger: logging.Logger,
    side: str,
    row: dict,
    base_entry_price: float,
):
    entry = row.get("entry_price")
    take_profit = row.get("take_profit_price")
    entry_x = entry / base_entry_price if entry is not None and base_entry_price else None
    take_profit_x = (
        take_profit / base_entry_price
        if take_profit is not None and base_entry_price
        else None
    )
    logger.error(f"GRID {side.upper()} DIAGNOSTIC L{row['layer']:02d}:")
    logger.error(f"  entry             = {_diagnostic_number(entry_x, 6)}x")
    logger.error(f"  TP                = {_diagnostic_number(take_profit_x, 6)}x")
    logger.error(
        f"  existing net @ TP = "
        f"{_diagnostic_number(row.get('existing_net_at_take_profit'), 6)}"
    )
    logger.error(f"  target            = {_diagnostic_number(row.get('target_cash'), 6)}")
    logger.error(
        f"  required notional = "
        f"{_diagnostic_number(row.get('raw_required_notional'), 6)}"
    )
    logger.error(
        f"  used notional     = "
        f"{_diagnostic_number(row.get('required_notional'), 6)}"
    )
    logger.error(
        f"  cumulative        = "
        f"{_diagnostic_number(row.get('cumulative_notional_after'), 6)}"
    )
    logger.error(f"  order fee         = {_diagnostic_number(row.get('order_fee'), 6)}")
    logger.error(
        f"  equity MTM        = "
        f"{_diagnostic_number(row.get('equity_mtm_before_order'), 6)}"
    )
    logger.error(
        f"  equity after fee  = "
        f"{_diagnostic_number(row.get('equity_after_order'), 6)}"
    )
    logger.error(
        f"  initial margin    = "
        f"{_diagnostic_number(row.get('initial_margin_after'), 6)}"
    )
    logger.error(
        f"  margin cap        = "
        f"{_diagnostic_number(row.get('margin_cap_amount'), 6)}"
    )
    logger.error(f"  reason            = {row.get('failure_reason')}")


def log_grid_preflight(
    logger: logging.Logger,
    diagnostics: dict[str, dict],
    config: BacktestConfig,
    start_price: float,
):
    """Print the grid structure first; print raw sizing diagnostics only on fail."""
    logger.info(
        f"GRID CONFIG | start {start_price:.12g} "
        f"| target {config.strategy.take_profit_pct:.4%} "
        f"(${config.account.initial_equity * config.strategy.take_profit_pct:.2f}) "
        f"| layers {config.strategy.layer_count} "
        f"| gaps {[f'{gap:.3%}' for gap in config.strategy.grid_deviation_pcts]} "
        f"| leverage {config.account.leverage:g}x "
        f"| margin cap {config.account.margin_usage_cap_pct:.2%}"
    )
    summary_rows = []
    for side, result in diagnostics.items():
        outcome = "VALID" if result["executable"] else (
            f"INVALID AT L{result['failure_layer']:02d}"
        )
        base_entry_price = _preflight_base_entry_price(result, start_price)
        summary_rows.append(_preflight_summary_row(result, config, base_entry_price))
        logger.info(
            f"GRID {side.upper()} | target ${result['target_cash']:.2f} "
            f"| {config.strategy.layer_count} layers | {outcome}"
        )
        logger.info(
            "L   GAP      GAP/PREV   ENTRY      TP         ADD/BAL    CUM/BAL    "
            "EQUITY%    IM/EQ%   ADV_LOSS"
        )
        previous_gap = None
        for row in result["layers"]:
            entry = row.get("entry_price")
            take_profit = row.get("take_profit_price")
            entry_x = entry / base_entry_price if entry is not None and base_entry_price else None
            gap = row.get("take_profit_deviation_pct")
            gap_prev_x = (
                1.0
                if previous_gap is None
                else (
                    gap / previous_gap
                    if gap is not None and previous_gap
                    else None
                )
            )
            take_profit_x = (
                take_profit / base_entry_price
                if take_profit is not None and base_entry_price
                else None
            )
            add_bal = (
                row.get("required_notional", 0.0) / config.account.initial_equity
                if row.get("required_notional") is not None else None
            )
            cum_bal = (
                row.get("cumulative_notional_after") / config.account.initial_equity
                if row.get("cumulative_notional_after") is not None else None
            )
            equity_pct = (
                row.get("equity_mtm_before_order") / config.account.initial_equity
                if row.get("equity_mtm_before_order") is not None else None
            )
            im_eq = row.get("margin_usage_pct")
            adverse_loss = row.get("adverse_loss_pct")
            logger.info(
                f"{row['layer']:<3d}"
                f"{_fmt(gap, 2, 100.0):>7}%"
                f"{_diagnostic_number(gap_prev_x, 4):>12}"
                f"{_diagnostic_number(entry_x, 4):>11}"
                f"{_diagnostic_number(take_profit_x, 4):>10}"
                f"{_fmt(add_bal, 2, 100.0):>11}%"
                f"{_fmt(cum_bal, 2, 100.0):>11}%"
                f"{_fmt(equity_pct, 2, 100.0):>10}%"
                f"{_fmt(im_eq, 2, 100.0):>9}%"
                f"{_fmt(adverse_loss, 2, 100.0):>11}%"
            )
            if gap is not None:
                previous_gap = gap
        if not result["executable"]:
            reason = result["failure_reason"]
            logger.error(
                f"GRID FAILURE | {side.upper()} layer {result['failure_layer']} "
                f"| {reason} | {GRID_FAILURE_HINTS.get(reason, 'inspect the failed layer')}"
            )
            failed_row = result["layers"][-1] if result["layers"] else {}
            if failed_row:
                _log_preflight_failure_diagnostic(
                    logger, side, failed_row, base_entry_price
                )

    if summary_rows:
        logger.info("GRID SUMMARY")
        logger.info(
            "SIDE    MAX LAYER   FINAL ENTRY   CUM/BAL   MIN EQUITY   "
            "PEAK IM/EQ   BREAK LOSS   BREAK EQ"
        )
        for row in summary_rows:
            logger.info(
                f"{row['side'].upper():<8}"
                f"{row['max_layer']:>5}"
                f"{_diagnostic_number(row['final_entry_x'], 4):>16}x"
                f"{_fmt(row['cum_bal'], 2, 100.0):>10}%"
                f"{_fmt(row['min_equity'], 2, 100.0):>13}%"
                f"{_fmt(row['peak_im_eq'], 2, 100.0):>13}%"
                f"{_fmt(row['grid_break_loss_pct'], 2, 100.0):>13}%"
                f"{_fmt(row['grid_break_equity_pct'], 2, 100.0):>11}%"
            )


def main(
    logger: logging.Logger,
    config: BacktestConfig,
    *,
    grid_validation_only: bool = False,
) -> dict:
    bars = load_bars(logger, config.data, intrabar_interval=config.intrabar_interval)
    if config.max_cycle_days is not None:
        max_cycle_bars = _days_to_bars(config.max_cycle_days, bars.periods_per_year)
        config = replace(
            config,
            strategy=replace(config.strategy, max_cycle_bars=max_cycle_bars),
        )
        logger.info(
            f"CYCLE LIMIT | {config.max_cycle_days:g} days -> "
            f"{max_cycle_bars} bars"
        )
    start_price = float(bars.open[0])
    simulator = MartingaleSimulator(config.strategy, config.account)
    diagnostics = simulator.grid_diagnostics(start_price)
    log_grid_preflight(logger, diagnostics, config, start_price)
    grid_valid = all(result["executable"] for result in diagnostics.values())
    if grid_validation_only:
        result = {
            "grid_validation_only": True,
            "grid_valid": grid_valid,
            "start_price": start_price,
            "bar_range_pct": bar_range_statistics(bars),
            "grid_validation": diagnostics,
            "params": {
                "strategy": asdict(config.strategy),
                "account": asdict(config.account),
                "data": asdict(config.data),
                "max_cycle_days": config.max_cycle_days,
            },
        }
        logger.info(
            f"GRID VALIDATION ONLY | valid {grid_valid} | backtest skipped"
        )
        if config.save_path:
            os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
            with open(config.save_path, "w", encoding="utf-8") as handle:
                json.dump(
                    json_safe(result),
                    handle,
                    indent=4,
                    default=_json_default,
                    allow_nan=False,
                )
            logger.info(f"Grid validation report saved to {config.save_path}")
        return result
    simulator.validate_initial_grid(start_price, diagnostics)
    modes = (
        ("withdraw", "compound")
        if config.capital.profit_handling == "both"
        else (config.capital.profit_handling,)
    )
    scenario_results = {}
    export_trades = []
    for mode in modes:
        scenario_config = replace(
            config, capital=replace(config.capital, profit_handling=mode)
        )
        logger.info(f"CAPITAL SCENARIO | {mode}")
        scenario_result = run_monte_carlo(logger, bars, scenario_config)
        for trade in scenario_result["trades"]:
            trade["capital_mode"] = mode
        export_trades.extend(scenario_result["trades"])
        scenario_results[mode] = scenario_result
        log_report(logger, scenario_result, scenario_config)
        if config.equity_plot_path:
            from trade.sim.martingale_plot import plot_equity_paths

            equity_path = config.equity_plot_path
            if len(modes) > 1:
                root, extension = os.path.splitext(equity_path)
                equity_path = f"{root}_{mode}{extension or '.png'}"
            scenario_result["equity_plot_paths"] = plot_equity_paths(
                logger,
                bars,
                scenario_result,
                equity_path,
                curves_per_plot=config.equity_curves_per_plot,
                title=(
                    f"{config.data.symbol} {config.data.interval} | {mode} | "
                    f"{config.strategy.layer_count} layers | "
                    f"ruin resets to {config.account.initial_equity:g}"
                ),
            )
        if config.adverse_plot_path:
            from trade.sim.martingale_plot import plot_adverse_boundaries

            adverse_path = config.adverse_plot_path
            if len(modes) > 1:
                root, extension = os.path.splitext(adverse_path)
                adverse_path = f"{root}_{mode}{extension or '.png'}"
            scenario_result["adverse_plot_path"] = plot_adverse_boundaries(
                logger,
                bars,
                scenario_result,
                adverse_path,
                title=(
                    f"{config.data.symbol} {config.data.interval} | {mode} | "
                    f"adverse boundary hits by layer"
                ),
            )
        if config.grid_break_plot_dir:
            from trade.sim.martingale_plot import plot_grid_break_events

            grid_break_dir = config.grid_break_plot_dir
            if len(modes) > 1:
                grid_break_dir = os.path.join(grid_break_dir, mode)
            scenario_result["grid_break_event_plot_dir"] = plot_grid_break_events(
                logger,
                bars,
                scenario_result,
                grid_break_dir,
                context_bars=config.grid_break_context_bars,
                title_prefix=(
                    f"{config.data.symbol} {config.data.interval} | {mode}"
                ),
            )

    # Keep the existing result shape for plots and callers. In comparison mode
    # the primary path is compound, while both complete summary sets are nested.
    primary_mode = "compound" if "compound" in scenario_results else modes[0]
    result = scenario_results[primary_mode]
    if len(scenario_results) > 1:
        result["capital_scenarios"] = {
            mode: {
                key: value
                for key, value in scenario_result.items()
                if key not in ("trades", "equity_curves")
            }
            for mode, scenario_result in scenario_results.items()
        }
        result["capital_comparison"] = {
            "withdraw": {
                "accumulated_before_breach": scenario_results["withdraw"]["aggregate"][
                    "withdrawn_profit_before_grid_breach"
                ],
                "mean": scenario_results["withdraw"]["aggregate"][
                    "withdrawn_profit_before_grid_breach_mean"
                ],
                "on_breached_runs": scenario_results["withdraw"]["aggregate"][
                    "withdrawn_profit_on_breached_runs"
                ],
            },
            "compound": {
                "double_target_multiple": config.capital.double_target_multiple,
                "doubled_runs": scenario_results["compound"]["aggregate"][
                    "doubled_before_grid_breach_runs"
                ],
                "double_rate": scenario_results["compound"]["aggregate"][
                    "double_before_grid_breach_rate"
                ],
                "double_rate_on_breached_runs": scenario_results["compound"][
                    "aggregate"
                ]["double_before_grid_breach_rate_on_breached_runs"],
            },
        }

    if config.trades_path:
        write_table(logger, trades_dataframe(export_trades), config.trades_path, "trade")
    if config.fills_path:
        write_table(logger, fills_dataframe(export_trades), config.fills_path, "fill")
    if config.pressure_path or config.cluster_path:
        from trade.sim.martingale_plot import plot_cluster_scales, plot_pressure_heatmap

        label = (
            f"{config.data.symbol} {config.data.interval} | "
            f"layers {config.strategy.layer_count} "
            f"grid {list(config.strategy.grid_deviation_pcts)} "
            f"cooldown {config.strategy.loss_cooldown_bars}"
        )
        if config.pressure_path:
            plot_pressure_heatmap(
                logger, bars, result, config.pressure_path,
                title=f"Stop-loss pressure Z(t, m) — {label}",
                min_scale=config.pressure_min_scale,
                max_scale=config.pressure_max_scale,
                scale_count=config.pressure_scale_count,
            )
        if config.cluster_path:
            plot_cluster_scales(
                logger, result, config.cluster_path,
                title=f"Stop-loss clusters vs gap allowance — {label}",
                min_scale=config.cluster_min_scale,
                max_scale=config.cluster_max_scale,
                scale_count=config.cluster_scale_count,
            )

    if config.plot_path:
        # Imported here so the headless matplotlib backend is only pulled in
        # when a chart was actually asked for.
        from trade.sim.martingale_plot import plot_failures

        plot_failures(
            logger, bars, result, config.plot_path,
            title=(
                f"{config.data.symbol} {config.data.interval} | "
                f"grid {list(config.strategy.grid_deviation_pcts)} "
                f"tp {config.strategy.take_profit_pct:.2%} "
                f"layers {config.strategy.layer_count} "
                f"preflight grid "
                f"dir {config.strategy.initial_direction}"
                f"{'+reverse' if config.strategy.reverse_after_grid_break else ''} "
                f"full-layer break {config.strategy.stop_at_full_layers} | "
                f"lev {config.account.leverage:g}x "
                f"fee {config.account.taker_fee_pct}/{config.account.maker_fee_pct}%"
            ),
        )

    if config.breach_analysis_dir:
        # Imported here so matplotlib is only imported when a chart is wanted.
        from trade.sim.martingale_plot import plot_breach_analysis

        plot_breach_analysis(
            logger,
            bars,
            result.get("trades", []),
            config.breach_analysis_dir,
            title=(
                f"{config.data.symbol} {config.data.interval} | "
                f"grid {list(config.strategy.grid_deviation_pcts)} "
                f"tp {config.strategy.take_profit_pct:.2%} "
                f"dir {config.strategy.initial_direction}"
            ),
            window_hours=config.breach_vol_window_hours,
            horizons_hours=config.breach_horizons_hours,
            buckets=config.breach_buckets,
        )

    result["params"] = {
        "strategy": asdict(config.strategy),
        "account": asdict(config.account),
        "data": asdict(config.data),
        "monte_carlo": asdict(config.monte_carlo),
        "capital": asdict(config.capital),
        "intrabar_interval": config.intrabar_interval,
        "volatility_filter": (
            None if config.volatility_filter is None
            else asdict(config.volatility_filter)
        ),
        "max_cycle_days": config.max_cycle_days,
        "plots": {
            "equity_plot_path": config.equity_plot_path,
            "equity_curves_per_plot": config.equity_curves_per_plot,
            "adverse_plot_path": config.adverse_plot_path,
            "grid_break_plot_dir": config.grid_break_plot_dir,
            "grid_break_context_bars": config.grid_break_context_bars,
            "breach_analysis_dir": config.breach_analysis_dir,
        },
        "grid_validation_only": grid_validation_only,
    }
    if config.save_path:
        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
        # The per-trade records go to their own table; keeping them here would
        # bloat the summary JSON by orders of magnitude.
        report = json_safe(
            {key: value for key, value in result.items()
             if key not in ("trades", "equity_curves")}
        )
        with open(config.save_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=4, default=_json_default, allow_nan=False)
        logger.info(f"Report saved to {config.save_path}")
    return result


# ============================================================
# Run settings - edit these values directly
# ============================================================
@dataclass
class Args:
    """Every knob of one experiment, in one place.

    Edit the defaults here (or build ``Args(...)`` in a loop for a sweep) and
    run the module; nothing is read from the command line.
    """

    # --- data ---
    symbol: str = common.DOGE_5m.symbol
    interval: str = common.DOGE_5m.interval
    trading_type: str = common.DOGE_5m.trading_type
    market_category: str = common.DOGE_5m.market_category
    data_source: str = common.DOGE_5m.data_source
    # Finer bars used only where the two candidate paths of one bar disagree.
    # None keeps the pessimistic worst-of-two-path model.
    intrabar_interval: Optional[str] = None
    # Causally gate new base orders on trailing volatility. None disables it.
    volatility_filter: Optional[VolatilityFilterParams] = None
    test_start_date: Optional[str] = None  # YYYY-MM-DD, inclusive
    test_end_date: Optional[str] = None    # YYYY-MM-DD, inclusive
    from_date: Optional[str] = None        # legacy alias for test_start_date
    to_date: Optional[str] = None          # legacy alias for test_end_date

    # --- strategy ---
    grid_validation_only: bool = False    # print grid validation and skip backtest
    grid_deviation_pcts: List[float] = field(default_factory=lambda: [0.01,0.012,0.015,0.02,0.025,0.04,0.06,0.08,0.1]) #Fibonacci 1.618
    take_profit_pct: float = 0.001       # fixed net cash target / cycle-start balance
    initial_direction: str = "long"       # "long" or "short"
    reverse_after_grid_break: bool = False  # flip the side after every grid break
    loss_cooldown_bars: int = 0           # bars to sit out after a losing cycle
    max_cycle_days: Optional[float] = None  # force-close one cycle after this many days
    stop_at_full_layers: bool = True      # exit at the rung the ladder cannot reach

    # --- account ---
    initial_equity: float = 10_000.0
    leverage: float = 50.0
    taker_fee_pct: float = 0.05           # percent, base order / grid break / liquidation
    maker_fee_pct: float = 0.02           # percent, ladder and take profit
    maintenance_margin_pct: float = 0.5   # percent of position notional
    margin_usage_cap_pct: float = 0.90    # refuse orders past this share of equity
    liquidation_penalty_pct: float = 0.0
    ruin_equity_pct: float = 0.4         # ruin when balance <= peak balance * this

    # --- profit handling experiment ---
    profit_handling: str = "withdraw"         # "withdraw", "compound", or "both"
    stop_at_first_grid_breach: bool = False  # True: stop path at first grid break
    grid_break_is_ruin: bool = True      # True: every grid break starts a new strategy run
    double_target_multiple: float = 2.0

    # --- monte carlo ---
    runs: int = 30
    seed: int = 42
    min_bars: int = 2_000                 # bars that must remain ahead of a start
    max_bars: Optional[int] = None        # None: run until ruin or end of data
    monte_carlo_start_fraction: float = 0.30
    workers: int = 30                     # 0 = cpu_count - 1, 1 = serial

    # --- loss clustering ---
    gap_window_bars: int = 500            # window for the count dispersion test
    gap_short_bars: int = 10              # "another grid break almost immediately"
    gap_repeats: int = 20                 # random baselines drawn per run
    gap_seed: int = 17

    # --- output ---  (None = a default name inside the experiment directory)
    save_path: Optional[str] = None       # summary JSON
    trades_path: Optional[str] = None     # .csv or .parquet, one row per trade
    fills_path: Optional[str] = None      # .csv or .parquet, one row per layer
    plot_path: Optional[str] = None       # .png loss/failure chart
    equity_plot_path: Optional[str] = None  # .png market + equity path charts
    equity_curves_per_plot: int = 10      # Monte Carlo paths in each market chart
    adverse_plot_path: Optional[str] = None  # .png per-layer adverse hit density
    grid_break_plot_dir: Optional[str] = None  # directory with one chart per grid break
    grid_break_context_bars: int = 200       # bars shown before open / after break
    pressure_path: Optional[str] = None   # .png time x window-scale pressure map
    cluster_path: Optional[str] = None    # .png cluster size vs gap allowance
    # Directory for the grid-break timeline, the volatility curves and their
    # CSV tables. None skips the analysis.
    breach_analysis_dir: Optional[str] = None
    breach_vol_window_hours: float = 4.0
    breach_horizons_hours: List[float] = field(default_factory=lambda: [1.0, 4.0, 24.0])
    breach_buckets: int = 10
    # Window-scale ladders for those two charts, in bars.
    pressure_min_scale: int = 8
    pressure_max_scale: Optional[int] = None   # None: total_bars // 8
    pressure_scale_count: int = 40
    cluster_min_scale: int = 1
    cluster_max_scale: int = 5_000
    cluster_scale_count: int = 40
    no_plot: bool = False
    no_export: bool = False


def config_from_args(args: Args) -> BacktestConfig:
    return BacktestConfig(
        strategy=MartingaleParams(
            grid_deviation_pcts=args.grid_deviation_pcts,
            take_profit_pct=args.take_profit_pct,
            loss_cooldown_bars=args.loss_cooldown_bars,
            max_cycle_bars=None,
            initial_direction=args.initial_direction,
            reverse_after_grid_break=args.reverse_after_grid_break,
            stop_at_full_layers=args.stop_at_full_layers,
        ),
        account=AccountParams(
            initial_equity=args.initial_equity,
            leverage=args.leverage,
            taker_fee_pct=args.taker_fee_pct,
            maker_fee_pct=args.maker_fee_pct,
            maintenance_margin_pct=args.maintenance_margin_pct,
            margin_usage_cap_pct=args.margin_usage_cap_pct,
            liquidation_penalty_pct=args.liquidation_penalty_pct,
            ruin_equity_pct=args.ruin_equity_pct,
        ),
        capital=CapitalParams(
            profit_handling=args.profit_handling,
            stop_at_first_grid_breach=args.stop_at_first_grid_breach,
            grid_break_is_ruin=args.grid_break_is_ruin,
            double_target_multiple=args.double_target_multiple,
        ),
        data=DataParams(
            market_category=args.market_category,
            data_source=args.data_source,
            symbols=(args.symbol,),
            interval=args.interval,
            trading_type=args.trading_type,
            test_start_date=args.test_start_date,
            test_end_date=args.test_end_date,
            from_date=args.from_date,
            to_date=args.to_date,
        ),
        monte_carlo=MonteCarloParams(
            runs=args.runs,
            seed=args.seed,
            min_bars=args.min_bars,
            max_bars=args.max_bars,
            start_fraction=args.monte_carlo_start_fraction,
            workers=args.workers,
        ),
        gap_analysis=GapAnalysisParams(
            window_bars=args.gap_window_bars,
            short_gap_bars=args.gap_short_bars,
            repeats=args.gap_repeats,
            seed=args.gap_seed,
        ),
        max_cycle_days=args.max_cycle_days,
        intrabar_interval=args.intrabar_interval,
        volatility_filter=args.volatility_filter,
        save_path=args.save_path,
        trades_path=None if args.no_export else args.trades_path,
        fills_path=None if args.no_export else args.fills_path,
        plot_path=None if args.no_plot else args.plot_path,
        equity_plot_path=None if args.no_plot else args.equity_plot_path,
        equity_curves_per_plot=args.equity_curves_per_plot,
        adverse_plot_path=None if args.no_plot else args.adverse_plot_path,
        grid_break_plot_dir=None if args.no_plot else args.grid_break_plot_dir,
        grid_break_context_bars=args.grid_break_context_bars,
        pressure_path=None if args.no_plot else args.pressure_path,
        cluster_path=None if args.no_plot else args.cluster_path,
        breach_analysis_dir=None if args.no_plot else args.breach_analysis_dir,
        breach_vol_window_hours=args.breach_vol_window_hours,
        breach_horizons_hours=args.breach_horizons_hours,
        breach_buckets=args.breach_buckets,
        pressure_min_scale=args.pressure_min_scale,
        pressure_max_scale=args.pressure_max_scale,
        pressure_scale_count=args.pressure_scale_count,
        cluster_min_scale=args.cluster_min_scale,
        cluster_max_scale=args.cluster_max_scale,
        cluster_scale_count=args.cluster_scale_count,
    )


def run(args: Optional[Args] = None) -> dict:
    """Set up the experiment directory and logger, then run one experiment."""
    # Not a default argument: Args is mutable, and one shared instance would
    # carry edits from a previous call into the next one.
    args = args if args is not None else Args()
    exp_dir = common.create_experiment_dir(
        os.path.join(common.PERSISTENCE_DIR, "martingale_sim"),
        args.symbol,
        args.interval,
    )
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(exp_dir, "simulation.log"),
        console_level=logging.INFO,
        file_level=logging.INFO,
    )
    config = config_from_args(args)
    defaults = {}
    if config.save_path is None:
        defaults["save_path"] = os.path.join(exp_dir, "monte_carlo_report.json")
    if not args.no_export:
        if config.trades_path is None:
            defaults["trades_path"] = os.path.join(exp_dir, "trades.parquet")
        if config.fills_path is None:
            defaults["fills_path"] = os.path.join(exp_dir, "fills.parquet")
    if not args.no_plot:
        if config.plot_path is None:
            defaults["plot_path"] = os.path.join(exp_dir, "failures.png")
        if config.equity_plot_path is None:
            defaults["equity_plot_path"] = os.path.join(exp_dir, "equity_paths.png")
        if config.adverse_plot_path is None:
            defaults["adverse_plot_path"] = os.path.join(exp_dir, "adverse_boundaries.png")
        if config.grid_break_plot_dir is None:
            defaults["grid_break_plot_dir"] = os.path.join(exp_dir, "grid-break")
        if config.pressure_path is None:
            defaults["pressure_path"] = os.path.join(exp_dir, "pressure.png")
        if config.cluster_path is None:
            defaults["cluster_path"] = os.path.join(exp_dir, "clusters.png")
        if config.breach_analysis_dir is None:
            defaults["breach_analysis_dir"] = os.path.join(exp_dir, "breach-analysis")
    config = replace(config, **defaults)
    started = time.time()
    result = main(logger, config, grid_validation_only=args.grid_validation_only)
    logger.info(f"run_time: {time.time() - started:.2f} s")
    return result


if __name__ == "__main__":
    run(Args())
