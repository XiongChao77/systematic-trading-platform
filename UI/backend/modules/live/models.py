"""Validated wire models for ephemeral live runner snapshots."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field, model_validator


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


class ConnectionSnapshot(StrictModel):
    status: Literal["ready", "recovering", "paused", "closed"]
    reason: str | None = None
    next_retry_at: AwareDatetime | None = None
    episode: int = Field(default=0, ge=0)


class StrategySnapshot(StrictModel):
    collection_scope: Literal["idle", "overview", "detail"] = "detail"
    instance_id: str = Field(min_length=1)
    strategy_hash: str = Field(min_length=1)
    account_id: str = ""
    trader_login: str = ""
    model_type: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    initial_balance: float | None = Field(default=None, ge=0.0)
    start_time: AwareDatetime | None = None
    risk_per_trade_pct: float = Field(ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(ge=0.0, le=1.0)
    max_holding_seconds: float | None = Field(default=None, ge=0.0)
    status: Literal["running", "disabled", "paused", "stopped"]
    connection: ConnectionSnapshot | None = None
    dashboard_age_seconds: float | None = Field(default=None, ge=0.0)
    account_updated_at: AwareDatetime | None = None
    position_updated_at: AwareDatetime | None = None
    account: AccountSnapshot | None = None
    position: PositionSnapshot | None = None
    latest_signal: SignalSnapshot | None = None
    availability: AvailabilitySnapshot
    errors: list[SnapshotError] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_account_identity(self):
        if self.venue.casefold() == "ctrader":
            if not self.trader_login or self.account_id:
                raise ValueError("cTrader snapshots require trader_login and no account_id")
        elif not self.account_id:
            raise ValueError("Non-cTrader snapshots require account_id")
        return self

    @computed_field
    @property
    def display_name(self) -> str:
        return f"{self.symbol.upper()}-{self.interval}-{self.venue.lower()}-{self.strategy_hash}"


class RunnerSnapshot(StrictModel):
    runner_id: str = Field(min_length=1)
    runner_instance_id: str = Field(min_length=1)
    runner_started_at: AwareDatetime
    sequence: int = Field(ge=1)
    sent_at: AwareDatetime
    strategies: list[StrategySnapshot]
