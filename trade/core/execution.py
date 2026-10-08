"""Venue-neutral execution records for live orders."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional


@dataclass(frozen=True)
class ExecutionFill:
    """One venue deal, or one aggregate fill when deal details are unavailable."""

    price: float
    quantity: float
    order_id: str = ""
    deal_id: str = ""
    client_order_id: str = ""
    executed_at_utc: Optional[datetime] = None
    is_aggregate: bool = False


@dataclass(frozen=True)
class ExecutionOrder:
    """One child order submitted for a strategy execution."""

    submitted_quantity: float
    order_id: str = ""
    client_order_id: str = ""
    status: str = "submitted"


@dataclass(frozen=True)
class ExecutionEvent:
    """One append-only venue state transition for an execution."""

    execution_id: str
    status: str
    event_at_utc: datetime
    order_role: str
    side: str
    order_id: str = ""
    client_order_id: str = ""
    submitted_quantity: float = 0.0
    reason: str = ""
    fill: ExecutionFill | None = None
    event_id: str = ""


@dataclass(frozen=True)
class ExecutionReport:
    """Snapshot of one submission response, not its final lifecycle state.

    completed_at_utc is local report creation time; accepted_at_utc is local
    response receipt time only when the response contains acceptance evidence.
    Venue events and actual fills determine the subsequent order lifecycle.
    """

    side: str
    requested_quantity: float
    decision_price: float
    decision_at_utc: datetime
    bid: float
    ask: float
    spread_pct: float
    status: str
    order_role: str = "entry"
    submitted_quantity: float | None = None
    reason: str = ""
    fills: tuple[ExecutionFill, ...] = ()
    orders: tuple[ExecutionOrder, ...] = ()
    quote_at_utc: Optional[datetime] = None
    submitted_at_utc: Optional[datetime] = None
    accepted_at_utc: Optional[datetime] = None
    completed_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.submitted_quantity is None:
            object.__setattr__(
                self,
                "submitted_quantity",
                float(self.requested_quantity),
            )

    @property
    def filled_quantity(self) -> float:
        return sum(fill.quantity for fill in self.fills)

    @property
    def fill_vwap(self) -> Optional[float]:
        quantity = self.filled_quantity
        if quantity <= 0:
            return None
        return sum(fill.price * fill.quantity for fill in self.fills) / quantity

    @property
    def best_fill_price(self) -> Optional[float]:
        prices = [fill.price for fill in self.fills if not fill.is_aggregate]
        if not prices:
            return None
        return min(prices) if self.side == "buy" else max(prices)

    @property
    def worst_fill_price(self) -> Optional[float]:
        prices = [fill.price for fill in self.fills if not fill.is_aggregate]
        if not prices:
            return None
        return max(prices) if self.side == "buy" else min(prices)

    @property
    def arrival_price(self) -> Optional[float]:
        """Executable quote observed immediately before submission."""

        price = self.ask if self.side == "buy" else self.bid
        if not math.isfinite(price) or price <= 0:
            return None
        return price

    @property
    def first_fill_at_utc(self) -> Optional[datetime]:
        timestamps = [
            fill.executed_at_utc
            for fill in self.fills
            if fill.executed_at_utc is not None
        ]
        return min(timestamps) if timestamps else None

    @property
    def decision_slippage_bps(self) -> Optional[float]:
        vwap = self.fill_vwap
        if (
            vwap is None
            or not math.isfinite(self.decision_price)
            or self.decision_price <= 0
        ):
            return None
        direction = 1.0 if self.side == "buy" else -1.0
        return direction * (vwap - self.decision_price) / self.decision_price * 10_000.0

    @property
    def implementation_shortfall_bps(self) -> Optional[float]:
        arrival = self.arrival_price
        vwap = self.fill_vwap
        if arrival is None or vwap is None:
            return None
        direction = 1.0 if self.side == "buy" else -1.0
        return direction * (vwap - arrival) / arrival * 10_000.0

    @property
    def execution_slippage_vwap_bps(self) -> Optional[float]:
        best = self.best_fill_price
        vwap = self.fill_vwap
        if best is None or vwap is None or best <= 0:
            return None
        direction = 1.0 if self.side == "buy" else -1.0
        return direction * (vwap - best) / best * 10_000.0

    @property
    def execution_slippage_worst_bps(self) -> Optional[float]:
        best = self.best_fill_price
        worst = self.worst_fill_price
        if best is None or worst is None or best <= 0:
            return None
        direction = 1.0 if self.side == "buy" else -1.0
        return direction * (worst - best) / best * 10_000.0

    @property
    def decision_to_first_fill_ms(self) -> Optional[float]:
        first_fill = self.first_fill_at_utc
        if first_fill is None:
            return None
        return (first_fill - self.decision_at_utc).total_seconds() * 1_000.0

    @property
    def submit_to_first_fill_ms(self) -> Optional[float]:
        first_fill = self.first_fill_at_utc
        if first_fill is None or self.submitted_at_utc is None:
            return None
        return (first_fill - self.submitted_at_utc).total_seconds() * 1_000.0

    @property
    def acknowledgement_latency_ms(self) -> Optional[float]:
        if self.submitted_at_utc is None or self.accepted_at_utc is None:
            return None
        return (self.accepted_at_utc - self.submitted_at_utc).total_seconds() * 1_000.0
