"""
Venue layer base class: the only boundary between a strategy and "one concrete trading venue".

A venue works in two directions:
  inbound  observe : assemble the feed's market signals plus the account/position only it knows into an Observation
  outbound execute : take a TradeIntent and turn it into real orders (submit_order / close_position)

Backtest (BtVenue) and live trading (BybitVenue / MT5Venue) are its two families of implementation;
switching backtest framework or exchange should only touch this layer.
"""

from abc import abstractmethod
from datetime import UTC, date, datetime, timedelta, timezone
import logging
import math
import uuid
from zoneinfo import ZoneInfo

from trade.core.protocol import Firm, OrderType, PositionView
from trade.core.protocol import TradeIntent, ActionType, PositionDir
from trade.core.execution_status import execution_status
from trade.core.execution import (
    ExecutionEvent,
    ExecutionFill,
    ExecutionOrder,
    ExecutionReport,
)


class VenueBase:
    MAX_ENTRY_SPREAD_PCT = 0.0015
    FIRM_DAILY_RESET_TIMEZONES = {
        Firm.FTMO: ZoneInfo("Europe/Prague"),
        # The5ers defines its reset as a fixed UTC+3 boundary without DST.
        Firm.THE5ERS: timezone(timedelta(hours=3)),
    }

    # ---------------- inbound: queries ----------------
    @abstractmethod
    def get_account_equity(self):
        pass

    @abstractmethod
    def get_current_state(self) -> PositionView:
        pass

    @staticmethod
    def _normalized_candle_time_utc(candle_open_time_utc: datetime) -> datetime:
        if candle_open_time_utc is None or candle_open_time_utc.tzinfo is None:
            raise ValueError("candle_open_time_utc must be timezone-aware")
        return candle_open_time_utc.astimezone(UTC)

    def get_daily_reset_date(self, candle_open_time_utc: datetime) -> date:
        """Return the UTC candle date used by ordinary venues and backtests."""

        return self._normalized_candle_time_utc(candle_open_time_utc).date()

    def get_firm_daily_reset_date(
        self,
        candle_open_time_utc: datetime,
        firm: Firm,
    ) -> date:
        reset_timezone = self.FIRM_DAILY_RESET_TIMEZONES[firm]
        return (
            self._normalized_candle_time_utc(candle_open_time_utc)
            .astimezone(reset_timezone)
            .date()
        )

    @abstractmethod
    def get_last_position_open_time(self):  # return utc
        pass

    # ---------------- outbound: orders ----------------
    @abstractmethod
    def close_position(self, size=None, **kwargs):
        pass

    @staticmethod
    def normalize_order_request(order_type=OrderType.MARKET, price=None):
        """Validate an entry order type and its optional limit price."""
        try:
            normalized_type = (
                order_type
                if isinstance(order_type, OrderType)
                else OrderType(str(order_type).strip().casefold())
            )
        except ValueError as exc:
            raise ValueError("order_type must be 'market' or 'limit'") from exc

        if normalized_type == OrderType.MARKET:
            if price is not None:
                raise ValueError("price must be omitted for market orders")
            return normalized_type, None

        try:
            normalized_price = float(price)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A positive finite price is required for limit orders"
            ) from exc
        if not math.isfinite(normalized_price) or normalized_price <= 0:
            raise ValueError("A positive finite price is required for limit orders")
        return normalized_type, normalized_price

    # size: Order quantity denominated in units of the base asset, not its notional value.
    @abstractmethod
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
        pass

    def normalize_order_quantity(self, size: float) -> float:
        """Return the exact base-asset quantity that will be submitted."""

        quantity = float(size)
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("Order quantity must be a positive finite value")
        return quantity

    def get_execution_account_id(self) -> str:
        """Return a stable, non-secret venue account identifier when available."""

        for name in ("account_id", "login", "trader_login"):
            value = getattr(self, name, None)
            if value not in (None, ""):
                return str(value)
        return ""

    def get_execution_symbol(self) -> str:
        """Return the actual instrument name used by the venue."""

        return str(getattr(self, "symbol", ""))

    def set_execution_event_callback(self, callback) -> None:
        """Register the runner callback used for asynchronous venue events."""

        self._execution_event_callback = callback

    def _emit_execution_event(self, event: ExecutionEvent) -> None:
        callback = getattr(self, "_execution_event_callback", None)
        if callback is not None:
            callback(event)

    def activate_execution_updates(self, execution_id: str) -> None:
        """Allow a venue to flush events buffered before the initial trace row."""

        return None

    def reconcile_execution_events(self, since_utc: datetime) -> int:
        """Emit venue events missed while the runner was offline."""

        return 0

    def get_bid_ask(self) -> tuple[float, float] | None:
        """Return an executable bid/ask pair when the venue supports live quotes."""

        return None

    def _execution_fills(self, result, *, is_buy: bool) -> tuple[ExecutionFill, ...]:
        """Convert a venue response into normalized fills."""

        return ()

    def _execution_orders(
        self,
        result,
        *,
        submitted_quantity: float,
        is_buy: bool,
    ) -> tuple[ExecutionOrder, ...]:
        """Convert a venue response into normalized child orders."""

        return ()

    def _quote_snapshot(self) -> tuple[float, float, float] | None:
        quote = self.get_bid_ask()
        if quote is None:
            return None
        bid, ask = map(float, quote)
        if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask < bid:
            raise RuntimeError(
                f"Venue returned an invalid bid/ask quote: bid={bid}, ask={ask}"
            )
        midpoint = (bid + ask) / 2.0
        return bid, ask, (ask - bid) / midpoint

    def _build_execution_report(
        self,
        *,
        execution_id: str,
        action: TradeIntent,
        order_role: str,
        side: str,
        requested_quantity: float,
        submitted_quantity: float,
        quote: tuple[float, float, float] | None,
        quote_at_utc: datetime | None,
        submitted_at_utc: datetime,
        accepted_at_utc: datetime,
        result,
    ) -> ExecutionReport:
        is_buy = side == "buy"
        try:
            fills = self._execution_fills(result, is_buy=is_buy)
        except Exception:
            logger = getattr(self, "logger", None) or logging.getLogger("trade.venue")
            logger.exception("Failed to extract execution fills")
            fills = ()
        try:
            orders = self._execution_orders(
                result,
                submitted_quantity=submitted_quantity,
                is_buy=is_buy,
            )
        except Exception:
            logger = getattr(self, "logger", None) or logging.getLogger("trade.venue")
            logger.exception("Failed to extract submitted orders")
            orders = ()
        bid, ask, spread_pct = quote or (
            float("nan"),
            float("nan"),
            float("nan"),
        )
        order_statuses = {order.status for order in orders}
        filled_quantity = sum(fill.quantity for fill in fills)
        status = execution_status(
            [order.status for order in orders], submitted_quantity, filled_quantity
        )
        # This is a response snapshot. Receipt alone does not prove acceptance.
        if not (order_statuses & {"accepted", "partially_filled", "filled"} or fills):
            accepted_at_utc = None
        return ExecutionReport(
            execution_id=execution_id,
            order_role=order_role,
            side=side,
            requested_quantity=requested_quantity,
            submitted_quantity=submitted_quantity,
            decision_price=float(action.price),
            decision_at_utc=action.created_at_utc,
            quote_at_utc=quote_at_utc,
            submitted_at_utc=submitted_at_utc,
            accepted_at_utc=accepted_at_utc,
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
            status=status,
            reason=action.reason,
            fills=fills,
            orders=orders,
        )

    def execute_action(self, action: TradeIntent):
        """Normalize and execute one strategy intent."""

        if action.action == ActionType.NOOP:
            return None
        if action.action == ActionType.CLOSE:
            position = self.get_current_state()
            if position.dir not in {PositionDir.POSITIVE, PositionDir.NEGATIVE}:
                return None
            side = "sell" if position.dir == PositionDir.POSITIVE else "buy"
            requested_quantity = (
                float(action.order_qty)
                if float(action.order_qty) > 0
                else abs(float(position.size))
            )
            submitted_quantity = self.normalize_order_quantity(requested_quantity)
            quote = self._quote_snapshot()
            quote_at_utc = datetime.now(UTC)
            execution_id = uuid.uuid4().hex
            submitted_at_utc = datetime.now(UTC)
            result = self.close_position(
                size=submitted_quantity,
                execution_id=execution_id,
            )
            accepted_at_utc = datetime.now(UTC)
            return self._build_execution_report(
                execution_id=execution_id,
                action=action,
                order_role="exit",
                side=side,
                requested_quantity=requested_quantity,
                submitted_quantity=submitted_quantity,
                quote=quote,
                quote_at_utc=quote_at_utc,
                submitted_at_utc=submitted_at_utc,
                accepted_at_utc=accepted_at_utc,
                result=result,
            )
        if action.action == ActionType.OPEN:
            is_buy = action.target_dir == PositionDir.POSITIVE
            requested_quantity = float(action.order_qty)
            submitted_quantity = self.normalize_order_quantity(requested_quantity)
            spread = self._quote_snapshot()
            quote_at_utc = datetime.now(UTC)
            bid = ask = spread_pct = float("nan")
            if spread is not None:
                bid, ask, spread_pct = spread
                if spread_pct > self.MAX_ENTRY_SPREAD_PCT:
                    logger = getattr(self, "logger", None) or logging.getLogger(
                        "trade.venue"
                    )
                    logger.warning(
                        "Entry rejected because spread exceeds limit | "
                        "symbol=%s bid=%g ask=%g spread_pct=%.4f%% limit_pct=%.4f%%",
                        getattr(self, "symbol", "unknown"),
                        bid,
                        ask,
                        spread_pct * 100.0,
                        self.MAX_ENTRY_SPREAD_PCT * 100.0,
                    )
                    return ExecutionReport(
                        side="buy" if is_buy else "sell",
                        requested_quantity=requested_quantity,
                        submitted_quantity=0.0,
                        decision_price=float(action.price),
                        decision_at_utc=action.created_at_utc,
                        quote_at_utc=quote_at_utc,
                        bid=bid,
                        ask=ask,
                        spread_pct=spread_pct,
                        status="rejected",
                        reason="spread_limit",
                    )

            execution_id = uuid.uuid4().hex
            submitted_at_utc = datetime.now(UTC)
            result = self.submit_order(
                submitted_quantity,
                is_buy=is_buy,
                stop_loss_pct=action.stop_loss_pct,
                take_profit_pct=action.take_profit_pct,
                execution_id=execution_id,
            )
            accepted_at_utc = datetime.now(UTC)
            return self._build_execution_report(
                execution_id=execution_id,
                action=action,
                order_role="entry",
                side="buy" if is_buy else "sell",
                requested_quantity=requested_quantity,
                submitted_quantity=submitted_quantity,
                quote=spread,
                quote_at_utc=quote_at_utc,
                submitted_at_utc=submitted_at_utc,
                accepted_at_utc=accepted_at_utc,
                result=result,
            )
        raise ValueError(f"Unsupported trade action: {action.action!r}")

    # ---------------- outbound: cash and life cycle (optional) ----------------
    def withdraw_cash(self, amount):
        """Move cash out of the account (used by the restartable martingale restart cost). Unsupported by default."""
        raise NotImplementedError

    def request_halt(self):
        """The strategy asks to stop this run (e.g. both accounts wiped out). Ignored by default."""
        pass
