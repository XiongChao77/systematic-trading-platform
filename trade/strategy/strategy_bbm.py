import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

from trade.core.protocol import ActionType, Observation, PositionDir, Signal, TradeIntent
from trade.core.strategy_base import StrategyBase
from trade.core.venue_base import VenueBase


@dataclass
class BbmStrategyConfig:
    """Strategy parameters for the Binary Barrier Model label backtest."""

    compound: bool = True
    risk_per_trade_pct: float = 0.015
    fixed_hold_bars: Optional[int] = None
    # Multipliers of MarketView.expected_vol, not absolute percentages.
    threshold_long: float = 1.7
    threshold_short: float = 1.7
    stop_loss_long: float = 1.7
    stop_loss_short: float = 1.7
    allow_long: bool = True
    allow_short: bool = True
    prob_thresh: Optional[float] = None
    min_expected_move_pct: float = 0.01
    max_daily_loss_pct: float = 0.02


class BbmSignalStrategy(StrategyBase):
    """
    Enter from model signal and let the bracket order exit by BBM barriers.

    A configured fixed_hold_bars acts as a time barrier: once a position is
    opened, it is closed after that many observed position bars. Later model
    signals do not refresh the period. Before the time barrier, the attached
    take-profit/stop-loss bracket and the daily loss guard can still exit early.

    Backtrader note: bracket exits are simulated from OHLC bars, so if one bar
    contains both the stop-loss and take-profit prices, the engine cannot know
    the real intrabar order. Backtrader resolves the first executable child
    order according to its broker/order matching flow, then OCO-cancels the
    sibling. That is different from BBM labeling, where same-bar dual touches
    are marked INVALID because the forward path is ambiguous.
    """

    def __init__(
        self,
        config: BbmStrategyConfig,
        init_equity: float,
        data_interval_ms: int,
        exist_hold_bars: int = 0,
        leverage: float = 1.0,
    ):
        super().__init__(data_interval_ms)
        self.logger = logging.getLogger("trade")
        self.config = config
        self.init_equity = float(init_equity)
        self.leverage = max(1.0, float(leverage))
        self.day_start_equity: Optional[float] = None
        self.last_trade_date = None
        self.is_halted_today = False
        self.meltdown_days = 0
        self.unexplained_meltdown = 0
        self.entries = 0
        self.skipped_missing_threshold = 0
        self.skipped_small_threshold = 0
        self.skipped_size = 0
        self.position_hold_bars = max(0, int(exist_hold_bars))
        self.previous_position_dir = PositionDir.FLAT

    def _update_position_hold_bars(self, position_dir: PositionDir):
        """Count from the observed entry without refreshing on later signals."""
        if position_dir == PositionDir.FLAT:
            self.position_hold_bars = 0
        elif self.previous_position_dir == PositionDir.FLAT and self.position_hold_bars > 0:
            self.position_hold_bars += 1
        elif position_dir != self.previous_position_dir:
            self.position_hold_bars = 1
        else:
            self.position_hold_bars += 1
        self.previous_position_dir = position_dir

    def _update_daily_state(self, daily_reset_date: date, account_equity: float):
        if daily_reset_date is None:
            raise ValueError("Observation.daily_reset_date is required")
        if self.last_trade_date != daily_reset_date:
            self.logger.debug(
                f"new daily reset date {daily_reset_date}, equity:{account_equity}"
            )
            self.day_start_equity = account_equity
            self.last_trade_date = daily_reset_date
            self.is_halted_today = False

    def _barrier_pcts(self, target_dir: PositionDir, state: Observation) -> tuple[float, float]:
        expected_vol = state.market.expected_vol
        if not valid_pct(expected_vol):
            self.skipped_missing_threshold += 1
            return 0.0, 0.0

        long_threshold = expected_vol * self.config.threshold_long
        short_threshold = expected_vol * self.config.threshold_short
        stop_loss_long = expected_vol * self.config.stop_loss_long
        stop_loss_short = expected_vol * self.config.stop_loss_short
        if not valid_pct(long_threshold) or not valid_pct(short_threshold):
            self.skipped_missing_threshold += 1
            return 0.0, 0.0
        if long_threshold < self.config.min_expected_move_pct or short_threshold < self.config.min_expected_move_pct:
            self.skipped_small_threshold += 1
            return 0.0, 0.0

        if target_dir == PositionDir.POSITIVE:
            stop_loss = stop_loss_long
            take_profit = long_threshold
        elif target_dir == PositionDir.NEGATIVE:
            stop_loss = stop_loss_short
            take_profit = short_threshold
        else:
            return 0.0, 0.0

        if not valid_pct(stop_loss):
            self.skipped_missing_threshold += 1
            return 0.0, 0.0
        return float(stop_loss), float(take_profit)

    def _risk_equity(self, account_equity: float) -> float:
        return account_equity if self.config.compound else self.init_equity

    def _daily_loss_base(self) -> float:
        return self.day_start_equity if self.config.compound else self.init_equity

    def _daily_loss_limit(self) -> float:
        return self._daily_loss_base() * self.config.max_daily_loss_pct

    def _remaining_daily_loss(self, account_equity: float) -> float:
        """Equity distance to the daily-loss floor, including unrealized PnL."""
        daily_loss_floor = self.day_start_equity - self._daily_loss_limit()
        return account_equity - daily_loss_floor

    def _risk_per_trade(self, account_equity: float) -> float:
        return self._risk_equity(account_equity) * self.config.risk_per_trade_pct

    def _calculate_order_qty(
        self,
        state: Observation,
        stop_loss_pct: float,
    ) -> float:
        if not valid_pct(stop_loss_pct):
            return 0.0

        final_qty = self._risk_per_trade(state.account.equity) / (state.market.price * stop_loss_pct)

        # Keep the risk budget independent from the live cash affordability cap.
        margin_capital = state.account.equity
        if state.account.balance is not None:
            margin_capital = float(state.account.balance)
            if not math.isfinite(margin_capital) or margin_capital < 0:
                raise ValueError("Margin sizing requires a nonnegative finite account balance")

        required_margin = final_qty * state.market.price / self.leverage
        if required_margin > margin_capital:
            final_qty = margin_capital * self.leverage / state.market.price

        if final_qty <= 0 or not math.isfinite(final_qty):
            self.skipped_size += 1
            return 0.0
        return final_qty

    def _process(self, state: Observation) -> TradeIntent:
        signal = state.market.signal
        if signal == Signal.INVALID:
            signal = Signal.NEUTRAL
        if self.config.prob_thresh is not None and state.market.pred_prob < self.config.prob_thresh:
            signal = Signal.NEUTRAL

        if state.position.dir == PositionDir.INVALID or state.account.equity < 0:
            self.logger.error(f"market info loss,position dir:{state.position.dir},equity:{state.account.equity} ")
            return TradeIntent(ActionType.NOOP)

        self._update_position_hold_bars(state.position.dir)
        self._update_daily_state(state.daily_reset_date, state.account.equity)
        daily_loss_abs = max(0.0, self.day_start_equity - state.account.equity)
        remaining_risk_budget = self._remaining_daily_loss(state.account.equity)

        if remaining_risk_budget <= 0.0:
            if not self.is_halted_today:
                daily_loss_pct = daily_loss_abs / self._daily_loss_base()
                self.logger.warning("Daily loss guard triggered: " f"{daily_loss_pct:.2%} / {self.config.max_daily_loss_pct:.2%}")
                self.is_halted_today = True
                self.meltdown_days += 1
                if self.last_action is None:
                    self.logger.error("unknown reason caused meltdown: no previous trade action")
                    self.unexplained_meltdown += 1
                elif self.last_action.target_dir == PositionDir.POSITIVE:
                    if self.last_action.stop_loss_price > state.market.open:
                        self.logger.warning("buy stop loss shft ")
                    elif self.last_action.action == ActionType.OPEN and self.last_action.stop_loss_price > state.market.low:
                        self.logger.warning("buy stop price cross becasue open/stop happened on the same bar ")
                    else:
                        self.logger.error(
                            "unknown reason caused meltdown at candle %s",
                            state.candle_open_time_utc,
                        )
                        self.unexplained_meltdown += 1
                elif self.last_action.target_dir == PositionDir.NEGATIVE:
                    if self.last_action.stop_loss_price < state.market.open:
                        self.logger.warning("sell stop loss shft ")
                    elif self.last_action.action == ActionType.OPEN and self.last_action.stop_loss_price < state.market.high:
                        self.logger.warning("sell stop price cross becasue open/stop happened on the same bar ")
                    else:
                        self.logger.error("unlnow reason cause meltdown")
                        self.unexplained_meltdown += 1
            if state.position.dir != PositionDir.FLAT:
                return TradeIntent(ActionType.CLOSE)
            return TradeIntent(ActionType.NOOP)

        if state.position.dir != PositionDir.FLAT:
            if state.position.dir == PositionDir.INVALID:
                return TradeIntent(ActionType.NOOP)
            elif self.config.fixed_hold_bars is not None and self.position_hold_bars >= self.config.fixed_hold_bars:
                return TradeIntent(ActionType.CLOSE, reason="fixed_hold_expired")
            return TradeIntent(ActionType.NOOP)

        if self.is_halted_today:
            return TradeIntent(ActionType.NOOP)

        target_dir = PositionDir.FLAT
        if signal == Signal.POSITIVE and self.config.allow_long:
            target_dir = PositionDir.POSITIVE
        elif signal == Signal.NEGATIVE and self.config.allow_short:
            target_dir = PositionDir.NEGATIVE

        if target_dir == PositionDir.FLAT:
            return TradeIntent(ActionType.NOOP)

        stop_loss_pct, take_profit_pct = self._barrier_pcts(target_dir, state)
        if not valid_pct(stop_loss_pct) or not valid_pct(take_profit_pct):
            return TradeIntent(ActionType.NOOP)

        risk_per_trade = self._risk_per_trade(state.account.equity)
        if remaining_risk_budget < risk_per_trade:
            return TradeIntent(ActionType.NOOP)

        order_qty = self._calculate_order_qty(state, stop_loss_pct)
        if order_qty <= 0:
            return TradeIntent(ActionType.NOOP)

        action = TradeIntent(
            action=ActionType.OPEN,
            price=state.market.price,
            target_dir=target_dir,
            target_layers=1,
            order_qty=order_qty,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            reason="bbm_signal_entry",
        )
        self.entries += 1
        return action

    def report(self) -> tuple[dict, dict]:
        return {
            "meltdown_days": self.meltdown_days,
            "unexplained_meltdown": self.unexplained_meltdown,
            "entries": self.entries,
            "fixed_hold_bars": self.config.fixed_hold_bars,
            "min_expected_move_pct": self.config.min_expected_move_pct,
            "skipped_missing_threshold": self.skipped_missing_threshold,
            "skipped_small_threshold": self.skipped_small_threshold,
            "skipped_size": self.skipped_size,
        }, {}


def valid_pct(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
