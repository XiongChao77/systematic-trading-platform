"""No-execution venue used by deterministic live-pipeline replay."""

from __future__ import annotations

from trade.core.protocol import OrderType, PositionView
from trade.core.dashboard_base import AccountBalance, AccountDashboard
from trade.core.venue_base import VenueBase


class MockVenue(VenueBase, AccountDashboard):
    """Implement the venue contract without changing account or position state."""

    def __init__(
        self,
        initial_equity: float = 10_000.0,
    ):
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        self._equity = float(initial_equity)

    def get_account_equity(self) -> float:
        return self._equity

    def get_current_state(self) -> PositionView:
        return PositionView()

    def get_dashboard_balance(self) -> AccountBalance:
        return AccountBalance(balance=self._equity, equity=self._equity, used_margin=0.0)

    def get_dashboard_position(self):
        return None

    def get_last_position_open_time(self):
        return None

    def close_position(self, size=None, **kwargs):
        return None

    def submit_order(
        self,
        size,
        is_buy,
        stop_loss_pct=None,
        take_profit_pct=None,
        *,
        order_type=OrderType.MARKET,
        price=None,
        execution_id=None,
    ):
        self.normalize_order_request(order_type, price)
        return None

    def withdraw_cash(self, amount):
        return None

    def shutdown(self) -> None:
        return None
