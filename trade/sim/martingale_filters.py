"""Causal entry gates for martingale ladder starts.

Every condition is evaluated with bars that closed before the candidate start,
so the gate never reads the bar it decides on. Volatility is annualized and
expressed in percent, matching the report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from data_process import common


def wilder(values, period: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    start = finite[0] + period - 1
    result[start] = np.mean(values[finite[0] : start + 1])
    for i in range(start + 1, len(values)):
        result[i] = result[i - 1] + (values[i] - result[i - 1]) / period
    return result


def clustering_series(bars, window: int) -> pd.Series:
    moves = np.log(pd.Series(bars.close)).diff().abs()
    reference = moves.rolling(2016, min_periods=2016).quantile(0.80).shift(1)
    high = moves.gt(reference).astype(float).where(reference.notna())
    return high.rolling(window, min_periods=window).mean().shift(1)


def indicator_series(bars, clustering_window: int = 12) -> dict[str, pd.Series]:
    close = pd.Series(bars.close)
    clustering = clustering_series(bars, clustering_window)
    high_change = pd.Series(bars.high).diff().to_numpy()
    low_change = -pd.Series(bars.low).diff().to_numpy()
    plus = np.where((high_change > low_change) & (high_change > 0), high_change, 0.0)
    minus = np.where((low_change > high_change) & (low_change > 0), low_change, 0.0)
    previous = close.shift(1).to_numpy()
    tr = np.maximum.reduce(
        [
            bars.high - bars.low,
            np.abs(bars.high - previous),
            np.abs(bars.low - previous),
        ]
    )
    plus[0] = minus[0] = np.nan
    atr = wilder(tr, 14)
    pdi = 100 * wilder(plus, 14) / atr
    mdi = 100 * wilder(minus, 14) / atr
    denominator = pdi + mdi
    dx = np.divide(
        100 * np.abs(pdi - mdi),
        denominator,
        out=np.zeros(len(close)),
        where=denominator > 0,
    )
    dx[~np.isfinite(denominator)] = np.nan
    adx = pd.Series(wilder(dx, 14)).shift(1)
    change = close.diff().to_numpy()
    gain = wilder(np.maximum(change, 0), 14)
    loss = wilder(np.maximum(-change, 0), 14)
    total = gain + loss
    rsi = pd.Series(
        np.divide(100 * gain, total, out=np.full(len(close), 50.0), where=total > 0)
    )
    rsi[~np.isfinite(total)] = np.nan
    highest = pd.Series(bars.high).rolling(14, min_periods=14).max()
    lowest = pd.Series(bars.low).rolling(14, min_periods=14).min()
    spread = highest - lowest
    wr = (-100 * (highest - close) / spread.where(spread > 0)).where(spread != 0, -50.0)
    return {
        "clustering": clustering,
        "adx": adx,
        "rsi": rsi.shift(1),
        "wr": wr.shift(1),
    }


@dataclass(frozen=True)
class EntryFilterParams:
    """Blocking thresholds; a start is rejected when any enabled condition matches.

    ``None`` disables an individual condition. WR is compared on its absolute
    value, so ``max_abs_wr`` refers to distance from the 14-bar extreme.
    """

    max_annualized_vol_pct: Optional[float] = 50.0
    max_adx: Optional[float] = 10.0
    max_abs_wr: Optional[float] = 10.0
    clustering_window: int = 12
    max_clustering: Optional[float] = 0.0
    weekdays: Optional[tuple[int, ...]] = (6,)
    vol_ewma_span: Optional[int] = None

    def __post_init__(self):
        for name in ("max_annualized_vol_pct", "max_adx", "max_abs_wr"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.clustering_window < 1:
            raise ValueError("clustering_window must be positive")
        if self.max_clustering is not None and not 0 <= self.max_clustering <= 1:
            raise ValueError("max_clustering must be in [0, 1]")
        if self.weekdays is not None:
            if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
                raise ValueError("weekdays must contain values in [0, 6]")
        if self.vol_ewma_span is not None and self.vol_ewma_span < 1:
            raise ValueError("vol_ewma_span must be positive")


def annualized_volatility_pct(bars, span: Optional[int]) -> np.ndarray:
    parameters = common.DOGE_5m if span is None else replace(common.DOGE_5m, vol_ewma_span=span)
    frame = pd.DataFrame(
        {name: getattr(bars, name) for name in ("open", "high", "low", "close")}
    )
    expected_vol = common.calculate_thresholds(frame, parameters)["expected_vol"].to_numpy()
    return np.concatenate(([np.nan], expected_vol[:-1])) * np.sqrt(
        bars.periods_per_year
    ) * 100.0


def build_entry_gate(bars, params: EntryFilterParams) -> np.ndarray:
    """Boolean mask aligned to bars; ``True`` means a new ladder may start.

    Every enabled condition blocks entry, so the gate is the complement of the
    union of the individual condition masks.
    """
    blocked = np.zeros(len(bars.close), dtype=bool)
    for mask in condition_masks(bars, params).values():
        blocked |= mask
    return ~blocked


def condition_masks(bars, params: EntryFilterParams) -> dict[str, np.ndarray]:
    """Per-condition blocking masks for diagnostics; keys follow the field names."""
    features = indicator_series(bars, params.clustering_window)
    masks = {}
    if params.max_annualized_vol_pct is not None:
        volatility = annualized_volatility_pct(bars, params.vol_ewma_span)
        masks["max_annualized_vol_pct"] = np.isfinite(volatility) & (
            volatility <= params.max_annualized_vol_pct
        )
    if params.max_adx is not None:
        adx = features["adx"].to_numpy()
        masks["max_adx"] = np.isfinite(adx) & (adx <= params.max_adx)
    if params.max_abs_wr is not None:
        wr = np.abs(features["wr"].to_numpy())
        masks["max_abs_wr"] = np.isfinite(wr) & (wr <= params.max_abs_wr)
    if params.max_clustering is not None:
        clustering = features["clustering"].to_numpy()
        masks["max_clustering"] = np.isfinite(clustering) & (
            clustering <= params.max_clustering
        )
    if params.weekdays is not None:
        weekday = pd.DatetimeIndex(pd.to_datetime(bars.time, utc=True)).dayofweek
        masks["weekdays"] = np.isin(weekday, params.weekdays)
    return masks
