import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

import MetaTrader5 as mt5

from trade.core.dashboard_base import (
    AccountBalance,
    AccountDashboard,
    AccountPosition,
    AccountPositionComponent,
    MarginMode,
    PositionSide,
)
from trade.core.venue_base import VenueBase
from trade.core.execution import ExecutionFill, ExecutionOrder
from trade.core.protocol import Firm, OrderType, PositionDir, PositionView

MT5_SYMBOL_FTMO_MAP = {"DOGEUSDT": "DOGEUSD", "ETHUSDT": "ETHUSD", "BTCUSDT": "BTCUSD"}


class MT5Venue(VenueBase, AccountDashboard):
    def __init__(
        self,
        path,
        symbol,
        magic,
        logger,
        login=None,
        password=None,
        server=None,
        *,
        firm: Firm,
    ):
        self.firm = Firm.parse(firm)
        self.symbol = MT5_SYMBOL_FTMO_MAP[symbol]
        self.magic = magic
        self.logger = logger
        self.path = path
        self.login = int(login) if login not in (None, "") else None
        self.password = password
        self.server = server

        self._connect_and_login()
        
        # make sure the symbol exist
        if not mt5.symbol_select(self.symbol, True):
            mt5.shutdown()
            raise RuntimeError(f"{symbol} not support | {self.magic}")

    def _connect_and_login(self):
        mt5.shutdown()
        initialize_args = {
            "path": self.path,
            "timeout": 60_000,
        }
        if self.login is not None:
            initialize_args["login"] = self.login
            if self.password is not None:
                initialize_args["password"] = self.password
            if self.server is not None:
                initialize_args["server"] = self.server

        initialized = mt5.initialize(**initialize_args)
        if initialized and self._is_target_account(mt5.account_info()):
            self.logger.info(
                "MT5 connection ready | login=%s magic=%s",
                self.login,
                self.magic,
            )
            return

        first_error = mt5.last_error()
        if self.login is None:
            self.logger.error(
                "MT5 initialization or account check failed | error=%s",
                first_error,
            )
            mt5.shutdown()
            raise RuntimeError("MT5 initialization failed")

        if not initialized:
            self.logger.warning(
                "MT5 credential initialization failed | error=%s; retrying login",
                first_error,
            )
            mt5.shutdown()
            if not mt5.initialize(path=self.path, timeout=60_000):
                error = mt5.last_error()
                self.logger.error("MT5 initialization retry failed | error=%s", error)
                mt5.shutdown()
                raise RuntimeError("MT5 initialization failed")
        else:
            account = mt5.account_info()
            self.logger.warning(
                "MT5 initialized with unexpected account | current=%s target=%s; "
                "retrying login",
                getattr(account, "login", None),
                self.login,
            )

        login_args = {"login": self.login}
        if self.password is not None:
            login_args["password"] = self.password
        if self.server is not None:
            login_args["server"] = self.server
        if not mt5.login(**login_args):
            error = mt5.last_error()
            self.logger.error(
                "MT5 login failed | login=%s server=%s error=%s",
                self.login,
                self.server,
                error,
            )
            mt5.shutdown()
            raise RuntimeError("MT5 login failed")

        account = mt5.account_info()
        if not self._is_target_account(account):
            self.logger.error(
                "MT5 account verification failed | current=%s target=%s",
                getattr(account, "login", None),
                self.login,
            )
            mt5.shutdown()
            raise RuntimeError("MT5 account verification failed")

        self.logger.info(
            "MT5 login succeeded | login=%s server=%s magic=%s",
            self.login,
            self.server,
            self.magic,
        )

    def _is_target_account(self, account):
        if account is None:
            return False
        if self.login is None:
            return True
        return int(account.login) == self.login

    def _ensure_target_account(self):
        account = mt5.account_info()
        if self._is_target_account(account):
            return account

        self.logger.warning(
            "MT5 account changed or disconnected | current=%s target=%s; reconnecting",
            getattr(account, "login", None),
            self.login,
        )
        self._connect_and_login()
        account = mt5.account_info()
        if not self._is_target_account(account):
            raise RuntimeError("MT5 target account is not active")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"MT5 symbol is unavailable after login: {self.symbol}")
        return account

    def shutdown(self):
        mt5.shutdown()

    def get_account_equity(self):
        """Used by the daily risk audit"""
        return self._ensure_target_account().equity

    def get_dashboard_balance(self) -> AccountBalance:
        account = self._ensure_target_account()
        return AccountBalance(
            balance=float(account.balance),
            equity=float(account.equity),
            used_margin=float(account.margin),
        )

    def _strategy_positions(self):
        self._ensure_target_account()
        return list(mt5.positions_get(symbol=self.symbol, magic=self.magic) or [])

    @staticmethod
    def _aggregate_position_values(positions):
        if not positions:
            return None
        directions = {int(position.type) for position in positions}
        if len(directions) != 1:
            raise RuntimeError(
                "MT5 strategy has simultaneous long and short positions"
            )
        total_volume = sum(float(position.volume) for position in positions)
        if total_volume <= 0:
            return None
        entry_price = sum(
            float(position.volume) * float(position.price_open)
            for position in positions
        ) / total_volume
        return directions.pop(), total_volume, entry_price

    def get_dashboard_position(self) -> AccountPosition | None:
        positions = self._strategy_positions()
        aggregated = self._aggregate_position_values(positions)
        if aggregated is None:
            return None
        direction, total_volume, entry_price = aggregated
        mark_price = sum(
            float(position.volume)
            * float(getattr(position, "price_current", 0.0) or 0.0)
            for position in positions
        ) / total_volume
        if mark_price <= 0:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                raise RuntimeError(f"MT5 returned no current price for {self.symbol}")
            mark_price = float(tick.bid if direction == 0 else tick.ask)
        unrealized_pnl = sum(
            float(getattr(position, "profit", 0.0) or 0.0)
            for position in positions
        )
        symbol_info = mt5.symbol_info(self.symbol)
        contract_size = float(
            getattr(symbol_info, "trade_contract_size", 0.0) or 0.0
        )
        digits = int(getattr(symbol_info, "digits", 0) or 0)
        stop_loss_prices = {
            round(float(getattr(position, "sl", 0.0) or 0.0), digits)
            for position in positions
            if float(getattr(position, "sl", 0.0) or 0.0) > 0
        }
        take_profit_prices = {
            round(float(getattr(position, "tp", 0.0) or 0.0), digits)
            for position in positions
            if float(getattr(position, "tp", 0.0) or 0.0) > 0
        }
        components = tuple(
            AccountPositionComponent(
                quantity=float(position.volume),
                entry_price=float(position.price_open),
                stop_loss_price=(
                    round(float(getattr(position, "sl", 0.0) or 0.0), digits)
                    or None
                ),
                take_profit_price=(
                    round(float(getattr(position, "tp", 0.0) or 0.0), digits)
                    or None
                ),
            )
            for position in positions
        )
        notional = (
            total_volume * mark_price * contract_size
            if contract_size > 0
            else None
        )
        account = self._ensure_target_account()
        cost = total_volume * entry_price * contract_size
        return AccountPosition(
            symbol=self.symbol,
            side=PositionSide.LONG if direction == 0 else PositionSide.SHORT,
            quantity=total_volume,
            entry_price=entry_price,
            mark_price=mark_price,
            notional=notional,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=(
                unrealized_pnl / cost if contract_size > 0 and cost > 0 else None
            ),
            leverage=float(getattr(account, "leverage", 0.0) or 0.0) or None,
            liquidation_price=None,
            margin_mode=MarginMode.UNKNOWN,
            stop_loss_price=(
                stop_loss_prices.pop() if len(stop_loss_prices) == 1 else None
            ),
            take_profit_price=(
                take_profit_prices.pop()
                if len(take_profit_prices) == 1
                else None
            ),
            components=components,
        )

    def get_current_state(self) -> PositionView:
        """Return the current position direction, size, and entry price."""
        aggregated = self._aggregate_position_values(self._strategy_positions())
        if aggregated is None:
            return PositionView()
        position_type, total_volume, entry_price = aggregated
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol not found: {self.symbol}")
        return PositionView(
            dir=(
                PositionDir.POSITIVE
                if position_type == 0
                else PositionDir.NEGATIVE
            ),
            size=total_volume * float(symbol_info.trade_contract_size),
            price=entry_price,
        )

    def submit_order(
        self,
        size,
        is_buy,
        stop_loss_pct=None,
        take_profit_pct=None,
        interval_ms=500,
        *,
        order_type=OrderType.MARKET,
        price=None,
        execution_id=None,
    ):
        """
        size: Order quantity in base-asset units
        is_buy: Boolean, True for BUY, False for SELL
        stop_loss_pct: Percentage value, e.g. 0.02 for 2%
        take_profit_pct: Percentage value, e.g. 0.04 for 4%
        interval_ms: Delay between split orders in milliseconds
        """
        order_type, price = self.normalize_order_request(order_type, price)
        self._ensure_target_account()
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol not found: {self.symbol}")
            return

        # 2. Convert base-asset quantity to lots and floor it to the venue step.
        contract_size = Decimal(str(symbol_info.trade_contract_size))
        volume_step = Decimal(str(symbol_info.volume_step))
        total_lots_decimal = (
            Decimal(str(size)) / contract_size / volume_step
        ).to_integral_value(rounding=ROUND_DOWN) * volume_step
        total_lots = float(total_lots_decimal)

        if total_lots < symbol_info.volume_min:
            self.logger.warning(f"Total lots {total_lots} below minimum {symbol_info.volume_min}")
            return

        self.logger.info(
            f"Start execution: total_lots={total_lots} | max_per_order={symbol_info.volume_max}"
        )

        remaining_lots = total_lots_decimal
        maximum_lots = Decimal(str(symbol_info.volume_max))
        minimum_lots = Decimal(str(symbol_info.volume_min))
        order_count = 0
        benchmark_price = None
        responses = []

        # 3. Execute split orders
        while remaining_lots > 0:
            current_batch_lots_decimal = min(remaining_lots, maximum_lots)
            current_batch_lots_decimal = (
                current_batch_lots_decimal / volume_step
            ).to_integral_value(rounding=ROUND_DOWN) * volume_step

            if current_batch_lots_decimal < minimum_lots:
                break
            current_batch_lots = float(current_batch_lots_decimal)

            if order_type == OrderType.MARKET:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    self.logger.error("Tick fetch failed, aborting")
                    break
                order_price = tick.ask if is_buy else tick.bid
            else:
                order_price = price
            order_price = round(order_price, symbol_info.digits)
            if benchmark_price is None:
                benchmark_price = order_price

            # Stop Loss
            if stop_loss_pct is not None:
                sl_price = (
                    order_price * (1.0 - stop_loss_pct)
                    if is_buy else order_price * (1.0 + stop_loss_pct)
                )
                sl_price = round(sl_price, symbol_info.digits)
            else:
                sl_price = 0.0

            # Take Profit
            if take_profit_pct is not None:
                tp_price = (
                    order_price * (1.0 + take_profit_pct)
                    if is_buy else order_price * (1.0 - take_profit_pct)
                )
                tp_price = round(tp_price, symbol_info.digits)
            else:
                tp_price = 0.0

            request = {
                "action": (
                    mt5.TRADE_ACTION_DEAL
                    if order_type == OrderType.MARKET
                    else mt5.TRADE_ACTION_PENDING
                ),
                "symbol": self.symbol,
                "volume": current_batch_lots,
                "type": (
                    mt5.ORDER_TYPE_BUY
                    if order_type == OrderType.MARKET and is_buy
                    else mt5.ORDER_TYPE_SELL
                    if order_type == OrderType.MARKET
                    else mt5.ORDER_TYPE_BUY_LIMIT
                    if is_buy
                    else mt5.ORDER_TYPE_SELL_LIMIT
                ),
                "price": order_price,
                "sl": sl_price,
                "tp": tp_price,
                "magic": self.magic,
                "comment": (
                    f"exec_{str(execution_id)[-12:]}_{order_count}"
                    if execution_id
                    else f"Split_Order_{order_count}"
                ),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": (
                    mt5.ORDER_FILLING_IOC
                    if order_type == OrderType.MARKET
                    else mt5.ORDER_FILLING_RETURN
                ),
            }

            self._ensure_target_account()
            res = mt5.order_send(request)
            responses.append(res)

            success_codes = {
                mt5.TRADE_RETCODE_DONE,
                getattr(mt5, "TRADE_RETCODE_PLACED", mt5.TRADE_RETCODE_DONE),
            }
            if res is None or res.retcode not in success_codes:
                err_msg = res.comment if res else "Order failed"
                self.logger.error(f"Batch {order_count} failed: {err_msg}")
                break

            self.logger.info(
                f"Batch {order_count} executed | "
                f"lots={current_batch_lots} | req_price={order_price} | "
                f"SL={sl_price} | TP={tp_price}"
            )

            remaining_lots -= current_batch_lots_decimal
            order_count += 1

            if remaining_lots > 0:
                time.sleep(interval_ms / 1000.0)

        if order_type == OrderType.LIMIT:
            return responses

        # 4. Wait briefly to ensure position is updated
        time.sleep(0.2)

        # 5. Get current positions and compute weighted average price
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)

        if not positions:
            self.logger.error("No positions found after execution")
            return

        total_volume = 0.0
        weighted_price_sum = 0.0

        for p in positions:
            total_volume += p.volume
            weighted_price_sum += p.volume * p.price_open

        if total_volume == 0:
            self.logger.error("Total position volume is zero")
            return

        avg_price = weighted_price_sum / total_volume

        # 6. Calculate slippage
        slippage = (avg_price - benchmark_price) if is_buy else (benchmark_price - avg_price)
        slippage_pct = slippage / benchmark_price

        self.logger.info(
            f"Execution finished: batches={order_count} | "
            f"benchmark_price={benchmark_price} | avg_price={avg_price:.6f} | "
            f"slippage={slippage_pct * 100:.4f}%"
        )
        return responses

    def get_bid_ask(self) -> tuple[float, float]:
        self._ensure_target_account()
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"MT5 tick is unavailable for {self.symbol}")
        return float(tick.bid), float(tick.ask)

    def normalize_order_quantity(self, size: float) -> float:
        self._ensure_target_account()
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol not found: {self.symbol}")
        raw_lots = Decimal(str(size)) / Decimal(
            str(symbol_info.trade_contract_size)
        )
        step = Decimal(str(symbol_info.volume_step))
        lots = (
            raw_lots / step
        ).to_integral_value(rounding=ROUND_DOWN) * step
        if lots < Decimal(str(symbol_info.volume_min)):
            raise ValueError(
                f"MT5 order volume {lots} is below minimum "
                f"{symbol_info.volume_min}"
            )
        return float(lots * Decimal(str(symbol_info.trade_contract_size)))

    def _execution_fills(self, result, *, is_buy: bool) -> tuple[ExecutionFill, ...]:
        responses = result if isinstance(result, list) else [result]
        fills = []
        symbol_info = mt5.symbol_info(self.symbol)
        contract_size = float(symbol_info.trade_contract_size)
        for response in responses:
            if response is None:
                continue
            price = float(getattr(response, "price", 0.0) or 0.0)
            quantity = float(getattr(response, "volume", 0.0) or 0.0)
            if price <= 0 or quantity <= 0:
                continue
            fills.append(
                ExecutionFill(
                    price=price,
                    quantity=quantity * contract_size,
                    order_id=str(getattr(response, "order", "") or ""),
                    deal_id=str(getattr(response, "deal", "") or ""),
                )
            )
        return tuple(fills)

    def _execution_orders(
        self,
        result,
        *,
        submitted_quantity: float,
        is_buy: bool,
    ) -> tuple[ExecutionOrder, ...]:
        responses = result if isinstance(result, list) else [result]
        symbol_info = mt5.symbol_info(self.symbol)
        contract_size = float(symbol_info.trade_contract_size)
        orders = []
        for response in responses:
            if response is None:
                continue
            request = getattr(response, "request", None)
            volume = float(
                getattr(request, "volume", 0.0)
                or getattr(response, "volume", 0.0)
                or 0.0
            )
            orders.append(
                ExecutionOrder(
                    order_id=str(getattr(response, "order", "") or ""),
                    client_order_id=str(
                        getattr(request, "comment", "") or ""
                    ),
                    submitted_quantity=volume * contract_size,
                    status=(
                        "filled"
                        if int(getattr(response, "deal", 0) or 0) > 0
                        else "submitted"
                    ),
                )
            )
        return tuple(orders)
        
    def close_position(self, size=None, execution_id=None, **kwargs):
        """Close every position of the current magic number"""
        if kwargs:
            raise TypeError(f"Unsupported MT5 close arguments: {sorted(kwargs)}")
        self._ensure_target_account()
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        symbol_info = mt5.symbol_info(self.symbol)
        remaining_lots = (
            None
            if size is None
            else float(size) / float(symbol_info.trade_contract_size)
        )
        responses = []
        for pos in positions:
            close_lots = (
                float(pos.volume)
                if remaining_lots is None
                else min(float(pos.volume), remaining_lots)
            )
            if close_lots <= 0:
                break
            self._ensure_target_account()
            tick = mt5.symbol_info_tick(self.symbol)
            response = mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "position": pos.ticket,
                "volume": close_lots,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": tick.bid if pos.type == 0 else tick.ask,
                "magic": self.magic,
                "comment": (
                    f"exec_{str(execution_id)[-12:]}_close"
                    if execution_id
                    else "strategy_close"
                ),
                # "type_filling": mt5.ORDER_FILLING_IOC,
            })
            responses.append(response)
            if remaining_lots is not None:
                remaining_lots -= close_lots
        self.logger.info(f"order close {self.magic}")
        return responses

    def get_last_position_open_time(self):
        try:
            self._ensure_target_account()
            positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
            
            # No position
            if not positions:
                return None
            
            pos = positions[0]
            
            # MT5 returns a second level timestamp (int)
            open_time = pos.time
            
            if open_time is None or open_time == 0:
                return None
            
            return datetime.fromtimestamp(open_time, tz=timezone.utc)

        except Exception as e:
            self.logger.error(f"Failed to get last position open time: {e}")
            return None

    def get_daily_reset_date(self, candle_open_time_utc: datetime):
        return self.get_firm_daily_reset_date(
            candle_open_time_utc,
            self.firm,
        )
