import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class MarginMode(Enum):
    CROSS = "cross"
    ISOLATED = "isolated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AccountBalance:
    """Account totals; used_margin is None when the venue cannot report it."""

    balance: float
    equity: float
    used_margin: Optional[float]

    def __post_init__(self) -> None:
        if self.used_margin is not None and (
            not math.isfinite(self.used_margin) or self.used_margin < 0
        ):
            object.__setattr__(self, "used_margin", None)


@dataclass(frozen=True)
class AccountPositionComponent:
    quantity: float
    entry_price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None


@dataclass(frozen=True)
class AccountPosition:
    symbol: str
    side: PositionSide
    quantity: float

    entry_price: float
    mark_price: float
    notional: Optional[float]

    unrealized_pnl: float
    unrealized_pnl_pct: Optional[float]

    leverage: Optional[float]
    liquidation_price: Optional[float]

    margin_mode: MarginMode = MarginMode.UNKNOWN
    opened_at: Optional[datetime] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    components: tuple[AccountPositionComponent, ...] = ()


class AccountDashboard(ABC):
    """
    Read-only account information for logging, monitoring, and frontend display.

    Dashboard methods must not participate in strategy decisions or order
    execution. Failures in this interface should not affect the trading path.
    """

    @abstractmethod
    def get_dashboard_balance(self) -> AccountBalance:
        """Return account balance, equity and occupied position margin."""
        raise NotImplementedError

    @abstractmethod
    def get_dashboard_position(self) -> Optional[AccountPosition]:
        """Return the single logical strategy position, if one exists."""
        raise NotImplementedError
