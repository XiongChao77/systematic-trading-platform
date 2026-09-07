import hashlib
import itertools
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

# Path setup
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", "..", ".."))

from trade.venue.live.bybit.bybit_engine import BybitEngine
from trade.core.dashboard_base import (
    AccountBalance,
    AccountDashboard,
    AccountPosition,
    AccountPositionComponent,
    MarginMode,
    PositionSide,
)
from trade.core.execution import ExecutionFill, ExecutionOrder
from trade.core.protocol import ActionType, OrderType, PositionDir, PositionView
from trade.core.venue_base import VenueBase


class BybitVenue(VenueBase, AccountDashboard):
    def __init__(
        self,
        key_path,
        symbol: str,
        magic: str | None = None,
        *,
        logger: logging.Logger | None = None,
    ):
        self.engine = BybitEngine(key_path)
        self.symbol = symbol
        self.magic = str(magic or "financial-ml-system")
        self._order_id_sequence = itertools.count()
        self.logger = logger or logging.getLogger("BybitVenue")
        self.logger.info(f"BybitVenue key_path:{key_path} symbol {symbol}")
        
        # Initialize precision info
        self.qty_step = 0.0
        self.tick_size = 0.0
        self.min_qty = 0.0
        self._init_symbol_info()

    def get_execution_account_id(self) -> str:
        api_key = str(getattr(self.engine, "api_key", "") or "")
        if not api_key:
            return ""
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"api-key-{digest}"

    def _new_order_link_id(
        self,
        action: str,
        execution_id: str | None = None,
    ) -> str:
        safe_magic = re.sub(r"[^A-Za-z0-9_-]", "_", self.magic)
        nonce_source = (
            str(execution_id)
            if execution_id
            else f"{time.time_ns():x}{next(self._order_id_sequence):x}"
        )
        nonce = re.sub(r"[^A-Za-z0-9_-]", "_", nonce_source)[-14:]
        suffix = f"_{action}_{nonce}"
        return f"{safe_magic[:36 - len(suffix)]}{suffix}"

    def normalize_order_quantity(self, size: float) -> float:
        quantity = Decimal(str(size))
        step = Decimal(str(self.qty_step))
        normalized = (
            quantity / step
        ).to_integral_value(rounding=ROUND_DOWN) * step
        minimum = Decimal(str(self.min_qty))
        if normalized < minimum:
            raise ValueError(
                f"Bybit order quantity {normalized} is below minimum {minimum}"
            )
        return float(normalized)

    def _init_symbol_info(self):
        """Sync exchange precision settings to prevent Invalid Volume errors."""
        try:
            res = self.engine.http.get_instruments_info(category="linear", symbol=self.symbol)
            if res['retCode'] == 0:
                info = res['result']['list'][0]
                self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                self.min_qty = float(info['lotSizeFilter']['minOrderQty'])
                self.tick_size = float(info['priceFilter']['tickSize'])
                self.logger.info(f"✅ Precision synced: QtyStep={self.qty_step}, Tick={self.tick_size}")
        except Exception as e:
            self.logger.error(f"Failed to sync precision info: {e}")

    def get_account_equity(self):
        """Get USDT account equity."""
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            return float(res['result']['list'][0]['coin'][0]['equity'])
        return 0.0

    def get_dashboard_balance(self) -> AccountBalance:
        response = self.engine.http.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT",
        )
        if response.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit wallet request failed: {response.get('retMsg', 'unknown error')}"
            )
        accounts = response.get("result", {}).get("list", [])
        coins = accounts[0].get("coin", []) if accounts else []
        coin = next((item for item in coins if item.get("coin") == "USDT"), None)
        if coin is None:
            raise RuntimeError("Bybit wallet response has no USDT balance")
        balance = float(coin.get("walletBalance", 0.0))
        equity = float(coin.get("equity", 0.0))
        if not all(math.isfinite(value) for value in (balance, equity)):
            raise RuntimeError("Bybit returned invalid dashboard balance data")
        margin = coin.get("totalPositionIM")
        return AccountBalance(
            balance=balance,
            equity=equity,
            used_margin=None if margin in (None, "") else float(margin),
        )

    def _dashboard_position_payload(self):
        response = self.engine.http.get_positions(
            category="linear",
            symbol=self.symbol,
        )
        if response.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit position request failed: {response.get('retMsg', 'unknown error')}"
            )
        active = [
            position
            for position in response.get("result", {}).get("list", [])
            if float(position.get("size", 0.0) or 0.0) > 0
        ]
        if len(active) > 1:
            raise RuntimeError(
                f"Bybit returned multiple active positions for {self.symbol}"
            )
        return active[0] if active else None

    def get_dashboard_position(self) -> AccountPosition | None:
        position = self._dashboard_position_payload()
        if position is None:
            return None
        quantity = float(position["size"])
        entry_price = float(position.get("avgPrice", 0.0) or 0.0)
        mark_price = float(position.get("markPrice", 0.0) or 0.0)
        if mark_price <= 0:
            mark_price = self._latest_price()
        unrealized_pnl = float(position.get("unrealisedPnl", 0.0) or 0.0)
        cost = entry_price * quantity
        trade_mode = str(position.get("tradeMode", ""))
        margin_mode = (
            MarginMode.CROSS
            if trade_mode == "0"
            else MarginMode.ISOLATED
            if trade_mode
            else MarginMode.UNKNOWN
        )
        stop_loss_price = float(position.get("stopLoss", 0.0) or 0.0) or None
        take_profit_price = float(position.get("takeProfit", 0.0) or 0.0) or None
        return AccountPosition(
            symbol=self.symbol,
            side=(
                PositionSide.LONG
                if position.get("side") == "Buy"
                else PositionSide.SHORT
            ),
            quantity=quantity,
            entry_price=entry_price,
            mark_price=mark_price,
            notional=float(position.get("positionValue", 0.0) or 0.0)
            or quantity * mark_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl / cost if cost > 0 else None,
            leverage=float(position.get("leverage", 0.0) or 0.0) or None,
            liquidation_price=float(position.get("liqPrice", 0.0) or 0.0)
            or None,
            margin_mode=margin_mode,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            components=(
                AccountPositionComponent(
                    quantity=quantity,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                ),
            ),
        )

    def get_current_state(self) -> PositionView:
        """Return the current position direction, size, and average price."""
        try:
            pos = self._dashboard_position_payload()
            if pos is None:
                return PositionView()
            size = float(pos['size'])
            avg_price = float(pos['avgPrice']) if size > 0 else 0.0
            side = pos['side'] # 'Buy' or 'Sell'

            if size == 0:
                return PositionView()
            
            direction = PositionDir.POSITIVE  if side == 'Buy' else PositionDir.NEGATIVE
            return PositionView(dir=direction, size=size, price=avg_price)

        except Exception as e:
            self.logger.error(f"Failed to get position state: {e}")
            return PositionView()

    def _latest_price(self) -> float:
        response = self.engine.http.get_tickers(
            category="linear",
            symbol=self.symbol,
        )
        price = float(response["result"]["list"][0]["lastPrice"])
        if not math.isfinite(price) or price <= 0:
            raise RuntimeError("Bybit returned an invalid ticker price")
        return price

    def get_bid_ask(self) -> tuple[float, float]:
        response = self.engine.http.get_tickers(
            category="linear",
            symbol=self.symbol,
        )
        ticker = response["result"]["list"][0]
        return float(ticker["bid1Price"]), float(ticker["ask1Price"])

    def _execution_fills(self, result, *, is_buy: bool) -> tuple[ExecutionFill, ...]:
        results = result if isinstance(result, list) else [result]
        fills = []
        for item in results:
            if not isinstance(item, dict) or item.get("retCode") != 0:
                continue
            order_id = item.get("result", {}).get("orderId")
            client_order_id = str(item.get("_trace_client_order_id", ""))
            if not order_id:
                continue
            response = self.engine.http.get_executions(
                category="linear",
                symbol=self.symbol,
                orderId=order_id,
                limit=100,
            )
            fills.extend(
                ExecutionFill(
                    price=float(execution["execPrice"]),
                    quantity=float(execution["execQty"]),
                    order_id=str(execution.get("orderId", order_id)),
                    deal_id=str(execution.get("execId", "")),
                    client_order_id=client_order_id,
                    executed_at_utc=(
                        datetime.fromtimestamp(
                            int(execution["execTime"]) / 1000.0,
                            tz=timezone.utc,
                        )
                        if execution.get("execTime") is not None
                        else None
                    ),
                )
                for execution in response.get("result", {}).get("list", [])
                if execution.get("execType") == "Trade"
            )
        return tuple(fills)

    def _execution_orders(
        self,
        result,
        *,
        submitted_quantity: float,
        is_buy: bool,
    ) -> tuple[ExecutionOrder, ...]:
        results = result if isinstance(result, list) else [result]
        orders = []
        for item in results:
            if not isinstance(item, dict) or item.get("retCode") != 0:
                continue
            payload = item.get("result", {})
            order_id = str(payload.get("orderId", "") or "")
            client_order_id = str(item.get("_trace_client_order_id", ""))
            if not order_id and not client_order_id:
                continue
            orders.append(
                ExecutionOrder(
                    order_id=order_id,
                    client_order_id=client_order_id,
                    submitted_quantity=float(
                        item.get("_trace_submitted_quantity")
                        or submitted_quantity / max(1, len(results))
                    ),
                    status="submitted",
                )
            )
        return tuple(orders)

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
        """
        Place an order.

        size: base coin quantity
        is_buy: True = Buy/Long, False = Sell/Short
        stop_loss_pct: stop-loss ratio, e.g. 0.05 means 5%
        take_profit_pct: take-profit ratio, e.g. 0.10 means 10%
        """
        order_type, price = self.normalize_order_request(order_type, price)

        # 1. Align quantity to exchange precision
        qty = self.normalize_order_quantity(float(size))
        qty_str = str(qty)

        reference_price = self._latest_price() if price is None else price
        if order_type == OrderType.LIMIT:
            reference_price = round(reference_price / self.tick_size) * self.tick_size

        # 3. Compute stop-loss / take-profit concrete prices.
        # Bybit expects concrete price, while the strategy layer provides ratios.
        sl_price = 0.0
        tp_price = 0.0

        if stop_loss_pct:
            if is_buy:
                raw_sl = reference_price * (1 - stop_loss_pct)
            else:
                raw_sl = reference_price * (1 + stop_loss_pct)

            sl_price = round(raw_sl / self.tick_size) * self.tick_size

        if take_profit_pct:
            if is_buy:
                raw_tp = reference_price * (1 + take_profit_pct)
            else:
                raw_tp = reference_price * (1 - take_profit_pct)

            tp_price = round(raw_tp / self.tick_size) * self.tick_size

        side = "Buy" if is_buy else "Sell"

        self.logger.info(
            f"🐢 Placing {order_type.value} order: {side} {qty_str} | "
            f"SL: {sl_price} | TP: {tp_price}"
        )

        try:
            # 4. Place order via engine.
            # Market orders do not need a limit price.
            order_params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market" if order_type == OrderType.MARKET else "Limit",
                "qty": qty_str,
                "positionIdx": 0,  # one-way position mode
                "reduceOnly": False,
                "orderLinkId": self._new_order_link_id(
                    "open",
                    execution_id,
                ),
            }
            if order_type == OrderType.LIMIT:
                order_params["price"] = str(reference_price)
                order_params["timeInForce"] = "GTC"

            if sl_price > 0:
                order_params["stopLoss"] = str(sl_price)

            if tp_price > 0:
                order_params["takeProfit"] = str(tp_price)

            # Optional but often recommended:
            # Use MarkPrice to reduce wick-trigger noise.
            # If you prefer LastPrice, remove these two lines.
            if sl_price > 0:
                order_params["slTriggerBy"] = "MarkPrice"

            if tp_price > 0:
                order_params["tpTriggerBy"] = "MarkPrice"

            res = self.engine.http.place_order(**order_params)
            res["_trace_client_order_id"] = order_params["orderLinkId"]
            res["_trace_submitted_quantity"] = qty

            if res["retCode"] == 0:
                self.logger.info(
                    f"✅ Order placed successfully: ID {res['result']['orderId']}"
                )
            else:
                self.logger.error(f"❌ Order failed: {res['retMsg']}")
            return res

        except Exception as e:
            self.logger.error(f"Order exception: {e}")
            raise

    def close_position(self, size=None, execution_id=None):
        """Close all open positions for this symbol."""
        try:
            res = self.engine.http.get_positions(category="linear", symbol=self.symbol)
            responses = []
            remaining = None if size is None else abs(float(size))
            for pos in res['result']['list']:
                position_size = float(pos['size'])
                close_size = (
                    position_size
                    if remaining is None
                    else min(position_size, remaining)
                )
                if close_size > 0:
                    side = "Sell" if pos['side'] == "Buy" else "Buy"
                    self.logger.info(
                        f"Closing position: {pos['side']} {close_size}"
                    )
                    client_order_id = self._new_order_link_id(
                        "close",
                        execution_id,
                    )
                    response = self.engine.http.place_order(
                        category="linear",
                        symbol=self.symbol,
                        side=side,
                        orderType="Market",
                        qty=str(close_size),
                        positionIdx=0,
                        reduceOnly=True,
                        orderLinkId=client_order_id,
                    )
                    response["_trace_client_order_id"] = client_order_id
                    response["_trace_submitted_quantity"] = close_size
                    responses.append(response)
                    if remaining is not None:
                        remaining -= close_size
                        if remaining <= 0:
                            break
            if not responses:
                return None
            return responses[0] if len(responses) == 1 else responses
        except Exception as e:
            self.logger.error(f"Close position exception: {e}")
            raise

    def get_last_position_open_time(self):
        try:
            res = self.engine.http.get_executions(
                category="linear",
                symbol=self.symbol,
                limit=50
            )
            
            if res.get("retCode") != 0:
                return None
            
            executions = res["result"]["list"]
            if not executions:
                return None
            
            # Current position direction
            position = self.get_current_state()
            if position.dir == PositionDir.FLAT:
                return None
            
            target_side = "Buy" if position.dir == PositionDir.POSITIVE else "Sell"
            
            # Find the most recent fill in the opening direction
            for exe in executions:
                if exe["execType"] == "Trade" and exe["side"] == target_side:
                    ts = int(exe["execTime"])
                    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            
            return None

        except Exception as e:
            self.logger.error(f"Failed to get position open time: {e}")
            return None
