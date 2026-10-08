"""
Shared protocol between the three layers (feed / strategy / venue depend on this file only).

  feed     produces market data and model signals -> packed into an Observation by the venue
  strategy consumes the Observation and produces a TradeIntent
  venue    executes the TradeIntent and feeds account/position back into the next Observation

This file depends on neither backtrader nor any exchange SDK.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional
from datetime import date, datetime, UTC
from math import inf

from data_process.common import Signal


# ============================================================
# Enums (system protocol layer)
# ============================================================
class PositionDir(IntEnum):
    """
    Current / target position direction
    """

    INVALID = -2
    NEGATIVE = -1
    FLAT = 0
    POSITIVE = 1


class ActionType(Enum):
    """
    Trade action emitted by the strategy layer
    """

    NOOP = "noop"  # do nothing, leave every position state untouched
    OPEN = "open"  # open a position
    CLOSE = "close"  # close the position
    PYRAMID = "pyramid"  # add to the position


class OrderType(str, Enum):
    """Supported entry order execution types."""

    MARKET = "market"
    LIMIT = "limit"


class Firm(str, Enum):
    """Supported proprietary trading firms."""

    FTMO = "ftmo"
    THE5ERS = "the5ers"

    @classmethod
    def parse(cls, value: object) -> "Firm":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().casefold()
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(firm.value for firm in cls)
            raise ValueError(
                f"firm must be one of {choices}; got {value!r}"
            ) from exc


@dataclass
class TradeIntent:
    """
    Output of the strategy layer: what to do.

    The single intent type of the whole system (strategy_base / strategy_ml /
    strategy_martingale each used to define their own, with inconsistent fields).
    Every strategy fills in what it needs; the venue only reads the fields it knows:
      - direction / layer semantics only (turtle, ...): target_dir / target_layers / target_pct
      - explicit order size (ML / martingale)        : order_qty / stop_loss_pct / take_profit_pct
      - martingale extras                            : layer (which safety order) / reason (intent origin, for tracing)
    """

    action: ActionType
    price: float = 0
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    target_dir: PositionDir = PositionDir.FLAT
    target_layers: int = 0
    target_pct: Optional[float] = None
    order_qty: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    layer: int = 0
    reason: str = ""

    @property
    def stop_loss_price(self) -> Optional[float]:
        if self.price <= 0 or self.stop_loss_pct <= 0:
            return None
        if self.target_dir == PositionDir.POSITIVE:
            return self.price * (1.0 - self.stop_loss_pct)
        if self.target_dir == PositionDir.NEGATIVE:
            return self.price * (1.0 + self.stop_loss_pct)
        return None

    @property
    def take_profit_price(self) -> Optional[float]:
        if self.price <= 0 or self.take_profit_pct <= 0:
            return None
        if self.target_dir == PositionDir.POSITIVE:
            return self.price * (1.0 + self.take_profit_pct)
        if self.target_dir == PositionDir.NEGATIVE:
            return self.price * (1.0 - self.take_profit_pct)
        return None


@dataclass
class MarketView:
    """From the feed: market data + model signal; account independent and reproducible offline."""

    price: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    signal: Optional[Signal] = None
    pred_prob: float = 0.0
    expected_vol: Optional[float] = None
    # Optional feed value: infinity means a continuous market with no close boundary.
    bars_to_close: Optional[float] = inf


@dataclass
class PositionView:
    """From the venue: the current position."""

    dir: PositionDir = PositionDir.FLAT
    size: float = 0
    price: float = 0


@dataclass
class AccountView:
    """Account equity for risk guards and optional cash balance for margin limits."""

    equity: float = 0.0
    balance: Optional[float] = None


@dataclass
class Observation:
    """
    Input of the strategy layer: everything the strategy can see on one bar.

    Kept in three blocks so the boundaries stay obvious:
      market   <- what the feed can compute on its own
      position <- the position, known only to the venue
      account  <- the equity, known only to the venue
      candle_open_time_utc <- market-event time used for continuity checks
      daily_reset_date     <- venue-defined account risk date
    """

    market: MarketView
    position: PositionView = field(default_factory=PositionView)
    account: AccountView = field(default_factory=AccountView)
    candle_open_time_utc: Optional[datetime] = None
    daily_reset_date: Optional[date] = None
