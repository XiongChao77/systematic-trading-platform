"""Validated wire models for ephemeral live runner snapshots."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class AccountSnapshot(StrictModel):
    balance: float
    equity: float
    used_margin: float | None = Field(ge=0.0)

    @computed_field
    @property
    def margin_utilization(self) -> float | None:
        """Occupied position margin divided by account equity, as a fraction."""
        if self.used_margin is None or self.equity <= 0:
            return None
        ratio = self.used_margin / self.equity
        return ratio if math.isfinite(ratio) else None


class PositionComponentSnapshot(StrictModel):
    quantity: float
    entry_price: float
    stop_loss_price: float | None = Field(default=None, gt=0.0)
    take_profit_price: float | None = Field(default=None, gt=0.0)


class PositionSnapshot(StrictModel):
    symbol: str
    side: Literal["long", "short"]
    quantity: float
    entry_price: float
    mark_price: float
    notional: float | None = None
    unrealized_pnl: float
    unrealized_pnl_pct: float | None = None
    leverage: float | None = None
    liquidation_price: float | None = None
    margin_mode: Literal["cross", "isolated", "unknown"] = "unknown"
    opened_at: AwareDatetime | None = None
    remaining_holding_seconds: float | None = Field(default=None, ge=0.0)
    stop_loss_price: float | None = Field(default=None, gt=0.0)
    take_profit_price: float | None = Field(default=None, gt=0.0)
    components: list[PositionComponentSnapshot] = Field(default_factory=list)


class SignalSnapshot(StrictModel):
    model_output: str
    raw_model_output: float | None = None
    probability: float | None = None
    net_score: float | None = None
    decision: str
    target_side: str
    quantity: float | None = None
    reason: str = ""
    updated_at: AwareDatetime


class AvailabilitySnapshot(StrictModel):
    account: bool
    position: bool
    latest_signal: bool


class SnapshotError(StrictModel):
    component: str
    message: str


class StrategySnapshot(StrictModel):
    strategy_id: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    risk_per_trade_pct: float = Field(ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(ge=0.0, le=1.0)
    max_holding_seconds: float | None = Field(default=None, ge=0.0)
    status: Literal["running", "disabled", "stopped"]
    dashboard_age_seconds: float | None = Field(default=None, ge=0.0)
    account: AccountSnapshot | None = None
    position: PositionSnapshot | None = None
    latest_signal: SignalSnapshot | None = None
    availability: AvailabilitySnapshot
    errors: list[SnapshotError] = Field(default_factory=list)


class RunnerSnapshot(StrictModel):
    runner_id: str = Field(min_length=1)
    runner_instance_id: str = Field(min_length=1)
    runner_started_at: AwareDatetime
    sequence: int = Field(ge=1)
    sent_at: AwareDatetime
    strategies: list[StrategySnapshot]
