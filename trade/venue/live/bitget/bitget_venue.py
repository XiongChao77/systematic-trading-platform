"""Bitget Classic Account USDT perpetual futures venue (REST/WebSocket v2)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
from pathlib import Path
import queue
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode

import requests
import websocket

from trade.core.dashboard_base import (
    AccountBalance,
    AccountDashboard,
    AccountPosition,
    AccountPositionComponent,
    MarginMode,
    PositionSide,
)
from trade.core.execution import ExecutionEvent, ExecutionFill, ExecutionOrder
from trade.core.protocol import OrderType, PositionDir, PositionView
from trade.core.venue_base import VenueBase


class BitgetAPIError(RuntimeError):
    """An exchange rejection with a safe, inspectable error code."""

    def __init__(self, path: str, code: str, message: str):
        self.code = code
        super().__init__(
            f"Bitget request failed for {path}: code={code}, message={message}"
        )


class BitgetVenue(VenueBase, AccountDashboard):
    """One logical position per symbol, with strategy-owned exchange protection."""

    BASE_URL = "https://api.bitget.com"
    USER_STREAM_URL = "wss://ws.bitget.com/v2/ws/private"
    PRODUCT_TYPE = "USDT-FUTURES"
    MARGIN_COIN = "USDT"
    PAGE_SIZE = 100
    ORDER_WAIT_SECONDS = 5.0
    ORDER_POLL_SECONDS = 0.2
    REQUEST_INTERVAL_SECONDS = 0.11
    USER_STREAM_RECONNECT_SECONDS = 5.0
    USER_STREAM_PING_SECONDS = 25.0
    USER_STREAM_PONG_TIMEOUT_SECONDS = 55.0
    RECONCILIATION_INTERVAL_SECONDS = 60.0
    TERMINAL_STATES = {"filled", "canceled", "cancelled", "rejected", "expired"}

    def __init__(
        self,
        key_path: str,
        symbol: str,
        magic: str | None = None,
        *,
        logger: logging.Logger | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
        enable_user_stream: bool = True,
        read_only: bool = False,
    ):
        self.symbol = str(symbol).strip().upper()
        self.magic = str(magic or "financial-ml-system")
        self.logger = logger or logging.getLogger("trade.bitget")
        self.timeout = float(timeout)
        self.read_only = bool(read_only)
        self.session = session or requests.Session()
        self._request_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self._stream_lock = threading.RLock()
        self._last_request: dict[str, float] = {}
        self._execution_ids: dict[str, str] = {}
        self._protective_clients_by_order_id: dict[str, str] = {}
        self._protective_orders: dict[str, dict[str, Any]] = {}
        self._stream_stop = threading.Event()
        self._stream_ready = threading.Event()
        self._stream_error: str | None = None
        self._stream_app: websocket.WebSocketApp | None = None
        self._stream_threads: list[threading.Thread] = []
        self._stream_messages: queue.Queue = queue.Queue()
        self._subscribed_channels: set[str] = set()
        self._last_pong = time.monotonic()
        self._reconcile_since = datetime.now(timezone.utc)
        try:
            self.api_key = self._load_credential(key_path, "apikey")
            self.api_secret = self._load_credential(key_path, "secret_key")
            self.passphrase = self._load_credential(key_path, "passphrase")
            self._load_filters()
            self._load_account_mode()
            self._sync_protective_orders()
            if enable_user_stream:
                self._start_user_stream()
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _load_credential(directory: str, filename: str) -> str:
        path = Path(directory) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Bitget credential file: {path}")
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"Empty Bitget credential file: {path}")
        return value

    def _redact(self, message: Any) -> str:
        text = str(message)
        for name in ("api_key", "api_secret", "passphrase"):
            secret = getattr(self, name, "")
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    def _sign(self, message: str) -> str:
        return base64.b64encode(
            hmac.new(
                self.api_secret.encode(), message.encode(), hashlib.sha256
            ).digest()
        ).decode("ascii")

    def _require_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("Bitget venue is read-only")

    def _request(self, method, path, params=None, *, signed=False):
        method = method.upper()
        if method != "GET":
            self._require_writable()
        payload = dict(params or {})
        query = urlencode(payload) if method == "GET" else ""
        body = json.dumps(payload, separators=(",", ":")) if method != "GET" else ""
        request_path = path + (f"?{query}" if query else "")
        with self._request_lock:
            if self._stream_stop.is_set():
                raise RuntimeError("Bitget venue has been shut down")
            delay = self.REQUEST_INTERVAL_SECONDS - (
                time.monotonic() - self._last_request.get(path, 0.0)
            )
            if delay > 0:
                time.sleep(delay)
            headers = {"Content-Type": "application/json", "locale": "en-US"}
            if signed:
                timestamp = str(int(time.time() * 1000))
                headers.update(
                    {
                        "ACCESS-KEY": self.api_key,
                        "ACCESS-PASSPHRASE": self.passphrase,
                        "ACCESS-TIMESTAMP": timestamp,
                        "ACCESS-SIGN": self._sign(
                            timestamp + method + request_path + body
                        ),
                    }
                )
            self._last_request[path] = time.monotonic()
            try:
                response = self.session.request(
                    method,
                    self.BASE_URL + request_path,
                    data=body or None,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                # Requests exceptions retain headers; do not propagate credential objects.
                raise RuntimeError(
                    f"Bitget transport failed for {path}: {type(exc).__name__}"
                ) from None
            try:
                result = response.json()
            except ValueError:
                raise RuntimeError(
                    f"Bitget returned non-JSON data for {path}: HTTP {response.status_code}"
                ) from None
            if not isinstance(result, dict):
                raise RuntimeError(f"Bitget returned invalid response data for {path}")
            if not 200 <= response.status_code < 300 or result.get("code") != "00000":
                raise BitgetAPIError(
                    path,
                    str(result.get("code", response.status_code)),
                    self._redact(result.get("msg", "Request rejected")),
                )
            if "data" not in result:
                raise RuntimeError(f"Bitget response has no data for {path}")
            return result["data"]

    def _symbol_params(self, **extra) -> dict[str, Any]:
        return {"symbol": self.symbol, "productType": self.PRODUCT_TYPE, **extra}

    @staticmethod
    def _positive(value, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"Bitget {name} must be positive and finite")
        return number

    @staticmethod
    def _floor_to_step(value, step: Decimal) -> Decimal:
        return (Decimal(str(value)) / step).to_integral_value(
            rounding=ROUND_DOWN
        ) * step

    @staticmethod
    def _decimal_string(value: Decimal) -> str:
        return format(value.normalize(), "f")

    def _load_filters(self) -> None:
        contracts = self._request(
            "GET", "/api/v2/mix/market/contracts", self._symbol_params()
        )
        if not isinstance(contracts, list):
            raise RuntimeError("Bitget returned invalid contract data")
        contract = next(
            (row for row in contracts if row.get("symbol") == self.symbol), None
        )
        if contract is None or contract.get("symbolStatus") != "normal":
            raise ValueError(
                f"Bitget symbol is not available for trading: {self.symbol}"
            )
        if contract.get(
            "symbolType"
        ) != "perpetual" or self.MARGIN_COIN not in contract.get(
            "supportMarginCoins", []
        ):
            raise ValueError("Bitget venue requires a USDT perpetual contract")
        self.quantity_step = Decimal(str(contract["sizeMultiplier"]))
        self.minimum_quantity = Decimal(str(contract["minTradeNum"]))
        self.price_tick = Decimal(str(contract["priceEndStep"])) * Decimal(10) ** -int(
            contract["pricePlace"]
        )
        self.minimum_notional = Decimal(str(contract["minTradeUSDT"]))
        self.maximum_market_quantity = Decimal(str(contract["maxMarketOrderQty"]))
        self.maximum_limit_quantity = Decimal(str(contract["maxOrderQty"]))
        for value in (
            self.quantity_step,
            self.minimum_quantity,
            self.price_tick,
            self.minimum_notional,
            self.maximum_market_quantity,
            self.maximum_limit_quantity,
        ):
            if not value.is_finite() or value <= 0:
                raise RuntimeError("Bitget returned invalid contract filters")

    def _account(self) -> dict[str, Any]:
        account = self._request(
            "GET",
            "/api/v2/mix/account/account",
            self._symbol_params(marginCoin=self.MARGIN_COIN),
            signed=True,
        )
        if not isinstance(account, dict):
            raise RuntimeError("Bitget returned invalid account data")
        return account

    def _load_account_mode(self) -> None:
        account = self._account()
        if account.get("posMode") not in {"one_way_mode", "hedge_mode"}:
            raise RuntimeError("Bitget returned an invalid position mode")
        if account.get("marginMode") not in {"isolated", "crossed"}:
            raise RuntimeError("Bitget returned an invalid margin mode")
        self.hedge_mode = account["posMode"] == "hedge_mode"
        self.margin_mode = account["marginMode"]

    def get_execution_account_id(self) -> str:
        return hashlib.sha256(self.api_key.encode()).hexdigest()[:16]

    def _client_order_prefix(self, action: str) -> str:
        owner = hashlib.sha256(f"{self.magic}:{self.symbol}".encode()).hexdigest()[:12]
        return f"fml-{owner}-{action}-"

    def _new_client_order_id(self, action: str, execution_id=None) -> str:
        client_id = self._client_order_prefix(action) + uuid.uuid4().hex[:12]
        if execution_id:
            self._execution_ids[client_id] = str(execution_id)
        return client_id

    def _order_action(self, client_id: str) -> str | None:
        return next(
            (
                action
                for action in ("open", "close", "sl", "tp")
                if client_id.startswith(self._client_order_prefix(action))
            ),
            None,
        )

    def normalize_order_quantity(self, size: float) -> float:
        quantity = self._floor_to_step(
            self._positive(size, "quantity"), self.quantity_step
        )
        if quantity < self.minimum_quantity:
            raise ValueError(
                f"Bitget quantity {quantity} is below minimum {self.minimum_quantity}"
            )
        return float(quantity)

    def _position(self) -> dict[str, Any] | None:
        rows = self._request(
            "GET",
            "/api/v2/mix/position/single-position",
            self._symbol_params(marginCoin=self.MARGIN_COIN),
            signed=True,
        )
        if not isinstance(rows, list):
            raise RuntimeError("Bitget returned invalid position data")
        active = []
        for row in rows:
            if str(row.get("symbol", "")).upper() != self.symbol:
                continue
            quantity = float(row["total"])
            if not math.isfinite(quantity) or quantity < 0:
                raise RuntimeError("Bitget returned invalid position quantity")
            if quantity:
                if row.get("holdSide") not in {"long", "short"}:
                    raise RuntimeError("Bitget returned invalid position direction")
                self._positive(row["openPriceAvg"], "entry price")
                active.append(row)
        if len(active) > 1:
            raise RuntimeError(
                f"Bitget has simultaneous LONG and SHORT positions: {self.symbol}"
            )
        return active[0] if active else None

    def get_account_equity(self) -> float:
        return self._positive(self._account()["accountEquity"], "account equity")

    def get_dashboard_balance(self) -> AccountBalance:
        account = self._account()
        equity = float(account["accountEquity"])
        balance = equity - float(account["unrealizedPL"] or 0)
        if not all(math.isfinite(value) for value in (equity, balance)):
            raise RuntimeError("Bitget returned invalid dashboard balance")
        margins = [account.get("crossedMargin"), account.get("isolatedMargin")]
        used_margin = (
            None
            if any(value in (None, "") for value in margins)
            else sum(float(value) for value in margins)
        )
        return AccountBalance(balance=balance, equity=equity, used_margin=used_margin)

    def get_current_state(self) -> PositionView:
        with self._operation_lock:
            position = self._position()
            if position is None:
                if self._protective_orders and not self.read_only:
                    self._cancel_owned_protective_orders()
                return PositionView()
            return PositionView(
                dir=(
                    PositionDir.POSITIVE
                    if position["holdSide"] == "long"
                    else PositionDir.NEGATIVE
                ),
                size=float(position["total"]),
                price=float(position["openPriceAvg"]),
            )

    @staticmethod
    def _timestamp(value) -> datetime:
        timestamp = int(value)
        if timestamp <= 0:
            raise ValueError("Bitget timestamp must be positive")
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

    def get_last_position_open_time(self):
        position = self._position()
        return self._timestamp(position["cTime"]) if position else None

    def _ticker(self) -> dict[str, Any]:
        rows = self._request("GET", "/api/v2/mix/market/ticker", self._symbol_params())
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or rows[0].get("symbol") != self.symbol
        ):
            raise RuntimeError("Bitget returned invalid ticker data")
        return rows[0]

    def _latest_price(self) -> float:
        return self._positive(self._ticker()["lastPr"], "ticker price")

    def get_bid_ask(self) -> tuple[float, float]:
        ticker = self._ticker()
        bid = self._positive(ticker["bidPr"], "bid")
        ask = self._positive(ticker["askPr"], "ask")
        if ask < bid:
            raise RuntimeError("Bitget returned a crossed bid/ask quote")
        return bid, ask

    def _pages(self, path: str, params: dict, list_key: str):
        cursor = None
        seen = set()
        while True:
            query = {**params, "limit": str(self.PAGE_SIZE)}
            if cursor:
                query["idLessThan"] = cursor
            data = self._request("GET", path, query, signed=True)
            # Bitget represents an empty page with a null list and no next ID.
            if (
                isinstance(data, dict)
                and list_key in data
                and data[list_key] is None
                and "endId" in data
                and data["endId"] in (None, "")
            ):
                return
            if not isinstance(data, dict) or not isinstance(data.get(list_key), list):
                list_type = (
                    type(data[list_key]).__name__
                    if isinstance(data, dict) and list_key in data
                    else "missing"
                )
                raise RuntimeError(
                    f"Bitget returned invalid paginated data for {path}: "
                    f"expected data.{list_key} to be a list, "
                    f"data_type={type(data).__name__}, list_type={list_type}"
                )
            rows = data[list_key]
            yield from rows
            if len(rows) < self.PAGE_SIZE:
                return
            cursor = str(data.get("endId") or "")
            if not cursor or cursor in seen:
                raise RuntimeError(f"Bitget pagination did not advance for {path}")
            seen.add(cursor)

    def _pending_plans(self, plan_type="profit_loss") -> list[dict]:
        return list(
            self._pages(
                "/api/v2/mix/order/orders-plan-pending",
                self._symbol_params(planType=plan_type),
                "entrustedList",
            )
        )

    def _sync_protective_orders(self) -> list[dict]:
        orders = [
            row
            for row in self._pending_plans()
            if self._order_action(str(row.get("clientOid", ""))) in {"sl", "tp"}
        ]
        self._protective_orders = {str(row["orderId"]): row for row in orders}
        return orders

    def _restore_triggered_order_owners(self, **filters) -> list[dict]:
        """Resolve plan IDs to executable order IDs, including after a restart."""
        plans = list(
            self._pages(
                "/api/v2/mix/order/orders-plan-history",
                self._symbol_params(planType="profit_loss", **filters),
                "entrustedList",
            )
        )
        owned = []
        for plan in plans:
            client_id = str(plan.get("clientOid") or "")
            child_id = str(plan.get("executeOrderId") or "")
            if (
                self._order_action(client_id) in {"sl", "tp"}
                and child_id
                and child_id != "0"
            ):
                self._protective_clients_by_order_id[child_id] = client_id
                owned.append(plan)
        return owned

    def _attribute_order(self, order: dict) -> dict:
        client_id = self._protective_clients_by_order_id.get(str(order.get("orderId")))
        return {**order, "clientOid": client_id} if client_id else order

    def _cancel_triggered_protective_orders(self) -> None:
        """Release positions reserved by triggered take-profit limit orders."""
        pending = list(
            self._pages(
                "/api/v2/mix/order/orders-pending",
                self._symbol_params(),
                "entrustedList",
            )
        )
        if not pending:
            return
        self._restore_triggered_order_owners()
        for raw_order in pending:
            order = self._attribute_order(raw_order)
            if self._order_action(str(order.get("clientOid") or "")) not in {
                "sl",
                "tp",
            }:
                continue
            self._request(
                "POST",
                "/api/v2/mix/order/cancel-order",
                self._symbol_params(
                    marginCoin=self.MARGIN_COIN, orderId=str(order["orderId"])
                ),
                signed=True,
            )
            self._wait_for_order(order)

    def _cancel_owned_protective_orders(self) -> None:
        self._require_writable()
        failures = []
        for order in self._sync_protective_orders():
            try:
                result = self._request(
                    "POST",
                    "/api/v2/mix/order/cancel-plan-order",
                    self._symbol_params(
                        marginCoin=self.MARGIN_COIN,
                        planType=order["planType"],
                        orderIdList=[{"orderId": str(order["orderId"])}],
                    ),
                    signed=True,
                )
                if (
                    not isinstance(result, dict)
                    or result.get("failureList")
                    or not result.get("successList")
                ):
                    raise RuntimeError(
                        "Bitget did not confirm protective order cancellation"
                    )
                self._protective_orders.pop(str(order["orderId"]), None)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(
                f"Failed to cancel {len(failures)} Bitget protective orders"
            ) from failures[0]

    def _assert_no_open_orders(self) -> None:
        regular = list(
            self._pages(
                "/api/v2/mix/order/orders-pending",
                self._symbol_params(),
                "entrustedList",
            )
        )
        plans = [
            row
            for kind in ("profit_loss", "normal_plan", "track_plan")
            for row in self._pending_plans(kind)
        ]
        if regular or plans:
            raise RuntimeError(
                f"Refusing to open with existing Bitget orders: {self.symbol}"
            )

    def _protective_price(self, action: str) -> float | None:
        prices = {
            float(order["triggerPrice"])
            for order in self._protective_orders.values()
            if self._order_action(str(order.get("clientOid", ""))) == action
        }
        return prices.pop() if len(prices) == 1 else None

    def get_dashboard_position(self) -> AccountPosition | None:
        with self._operation_lock:
            position = self._position()
            if position is None:
                return None
            self._sync_protective_orders()
            quantity = float(position["total"])
            entry = float(position["openPriceAvg"])
            mark = self._positive(position["markPrice"], "mark price")
            pnl = float(position["unrealizedPL"])
            sl, tp = self._protective_price("sl"), self._protective_price("tp")
            liquidation = float(position.get("liquidationPrice") or 0)
            return AccountPosition(
                symbol=self.symbol,
                side=(
                    PositionSide.LONG
                    if position["holdSide"] == "long"
                    else PositionSide.SHORT
                ),
                quantity=quantity,
                entry_price=entry,
                mark_price=mark,
                notional=quantity * mark,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl / (quantity * entry),
                leverage=float(position["leverage"]),
                liquidation_price=liquidation if liquidation > 0 else None,
                margin_mode=(
                    MarginMode.CROSS
                    if position["marginMode"] == "crossed"
                    else MarginMode.ISOLATED
                ),
                opened_at=self._timestamp(position["cTime"]),
                stop_loss_price=sl,
                take_profit_price=tp,
                components=(AccountPositionComponent(quantity, entry, sl, tp),),
            )

    def _order_detail(self, order: dict) -> dict:
        identifier = (
            {"orderId": str(order["orderId"])}
            if order.get("orderId")
            else {"clientOid": order["clientOid"]}
        )
        data = self._request(
            "GET",
            "/api/v2/mix/order/detail",
            self._symbol_params(**identifier),
            signed=True,
        )
        if not isinstance(data, dict):
            raise RuntimeError("Bitget returned invalid order detail")
        return data

    def _wait_for_order(self, order: dict) -> dict:
        deadline = time.monotonic() + self.ORDER_WAIT_SECONDS
        while True:
            detail = self._order_detail(order)
            if detail.get("state") in self.TERMINAL_STATES:
                return {**order, **detail}
            if time.monotonic() >= deadline:
                raise RuntimeError("Bitget market order did not reach a terminal state")
            time.sleep(self.ORDER_POLL_SECONDS)

    def _place_order(self, params: dict, *, wait: bool) -> dict:
        data = self._request(
            "POST", "/api/v2/mix/order/place-order", params, signed=True
        )
        if not isinstance(data, dict):
            raise RuntimeError("Bitget returned invalid order acknowledgement")
        order = {
            **data,
            "clientOid": params["clientOid"],
            "_trace_submitted_quantity": float(params["size"]),
        }
        return self._wait_for_order(order) if wait else order

    def _cancel_entry_and_flatten(self, client_id: str) -> None:
        """Resolve uncertain submissions by client ID before flattening any fills."""
        detail = self._order_detail({"clientOid": client_id})
        if detail.get("state") not in self.TERMINAL_STATES:
            self._request(
                "POST",
                "/api/v2/mix/order/cancel-order",
                self._symbol_params(marginCoin=self.MARGIN_COIN, clientOid=client_id),
                signed=True,
            )
            detail = self._wait_for_order({"clientOid": client_id})
        self.close_position()
        if self._position() is not None:
            raise RuntimeError("Bitget emergency close left an active position")

    def _trigger_price(self, price: float) -> str:
        rounded = self._floor_to_step(
            self._positive(price, "trigger price"), self.price_tick
        )
        if rounded <= 0:
            raise ValueError("Bitget trigger price rounded to zero")
        return self._decimal_string(rounded)

    def _place_protection(self, action, trigger, is_buy, execution_id):
        client_id = self._new_client_order_id(action, execution_id)
        params = self._symbol_params(
            marginCoin=self.MARGIN_COIN,
            planType="pos_loss" if action == "sl" else "pos_profit",
            triggerPrice=trigger,
            triggerType="mark_price",
            executePrice="0" if action == "sl" else trigger,
            holdSide=(
                ("long" if is_buy else "short")
                if self.hedge_mode
                else ("buy" if is_buy else "sell")
            ),
            clientOid=client_id,
        )
        data = self._request(
            "POST", "/api/v2/mix/order/place-tpsl-order", params, signed=True
        )
        if not isinstance(data, dict) or not data.get("orderId"):
            raise RuntimeError("Bitget did not acknowledge protective order")
        self._protective_orders[str(data["orderId"])] = {**params, **data}

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
        self._require_writable()
        order_type, price = self.normalize_order_request(order_type, price)
        for value in (stop_loss_pct, take_profit_pct):
            if value is not None and (
                not math.isfinite(float(value)) or not 0 <= float(value) < 1
            ):
                raise ValueError(
                    "Bitget protection percentages must be finite and in [0, 1)"
                )
        if order_type == OrderType.LIMIT and (stop_loss_pct or take_profit_pct):
            raise ValueError(
                "Bitget resting limit entries with protection are not supported"
            )
        quantity = Decimal(str(self.normalize_order_quantity(size)))
        maximum = (
            self.maximum_market_quantity
            if order_type == OrderType.MARKET
            else self.maximum_limit_quantity
        )
        if quantity > maximum:
            raise ValueError("Bitget order quantity exceeds the contract maximum")
        reference = (
            self._latest_price() if price is None else float(self._trigger_price(price))
        )
        if quantity * Decimal(str(reference)) < self.minimum_notional:
            raise ValueError("Bitget order notional is below the contract minimum")
        # Validate rounded protection before sending any entry to the exchange.
        for pct, direction in (
            (stop_loss_pct, -1 if is_buy else 1),
            (take_profit_pct, 1 if is_buy else -1),
        ):
            if pct:
                self._trigger_price(reference * (1 + direction * float(pct)))
        with self._operation_lock:
            self._load_account_mode()
            if self._position() is not None:
                raise RuntimeError(
                    f"Refusing to open over an existing Bitget position: {self.symbol}"
                )
            self._cancel_owned_protective_orders()
            self._assert_no_open_orders()
            client_id = self._new_client_order_id("open", execution_id)
            params = self._symbol_params(
                marginCoin=self.MARGIN_COIN,
                marginMode=self.margin_mode,
                size=self._decimal_string(quantity),
                side="buy" if is_buy else "sell",
                orderType=order_type.value,
                clientOid=client_id,
            )
            if self.hedge_mode:
                params["tradeSide"] = "open"
            if order_type == OrderType.LIMIT:
                params.update(price=self._trigger_price(price), force="gtc")
            try:
                result = self._place_order(params, wait=order_type == OrderType.MARKET)
                if order_type == OrderType.LIMIT:
                    return result
                filled = Decimal(str(result.get("baseVolume") or "0"))
                if filled <= 0:
                    raise RuntimeError("Bitget entry completed without a fill")
                executed_price = self._positive(
                    result["priceAvg"], "average fill price"
                )
                for action, pct, direction in (
                    ("sl", stop_loss_pct, -1 if is_buy else 1),
                    ("tp", take_profit_pct, 1 if is_buy else -1),
                ):
                    if pct:
                        self._place_protection(
                            action,
                            self._trigger_price(
                                executed_price * (1 + direction * float(pct))
                            ),
                            is_buy,
                            execution_id,
                        )
                return result
            except Exception:
                self.logger.error(
                    "Bitget entry/protection failed; reconciling entry and closing its fills"
                )
                try:
                    self._cancel_entry_and_flatten(client_id)
                except Exception as cleanup_error:
                    self.logger.error(
                        "Bitget entry recovery failed: %s", self._redact(cleanup_error)
                    )
                    raise RuntimeError(
                        "Bitget entry failed and recovery could not confirm a flat position"
                    ) from cleanup_error
                raise

    def close_position(self, size=None, execution_id=None):
        self._require_writable()
        if size is not None:
            self._positive(size, "close quantity")
        with self._operation_lock:
            position = self._position()
            if position is None:
                self._cancel_owned_protective_orders()
                return None
            self._load_account_mode()
            self._cancel_triggered_protective_orders()
            # A triggered protection may have filled while its cancellation was in flight.
            position = self._position()
            if position is None:
                self._cancel_owned_protective_orders()
                return None
            available = Decimal(str(position["total"]))
            quantity = self._floor_to_step(
                available if size is None else min(available, Decimal(str(size))),
                self.quantity_step,
            )
            if quantity <= 0 or quantity > self.maximum_market_quantity:
                raise ValueError("Bitget close quantity is outside the contract limits")
            is_long = position["holdSide"] == "long"
            params = self._symbol_params(
                marginCoin=self.MARGIN_COIN,
                marginMode=position["marginMode"],
                size=self._decimal_string(quantity),
                orderType="market",
                clientOid=self._new_client_order_id("close", execution_id),
            )
            if self.hedge_mode:
                params.update(side="buy" if is_long else "sell", tradeSide="close")
            else:
                params.update(side="sell" if is_long else "buy", reduceOnly="YES")
            result = self._place_order(params, wait=True)
            if float(result.get("baseVolume") or 0) <= 0:
                raise RuntimeError("Bitget close completed without a fill")
            # Preserve protection if a partial close leaves any exposure.
            if self._position() is None:
                self._cancel_owned_protective_orders()
            return result

    @staticmethod
    def _status(raw: str) -> str:
        return {"live": "accepted", "canceled": "cancelled"}.get(
            raw, raw or "submitted"
        )

    def _execution_orders(self, result, *, submitted_quantity, is_buy):
        if not isinstance(result, dict):
            return ()
        return (
            ExecutionOrder(
                order_id=str(result.get("orderId") or ""),
                client_order_id=str(result.get("clientOid") or ""),
                submitted_quantity=float(
                    result.get("_trace_submitted_quantity", submitted_quantity)
                ),
                status=self._status(str(result.get("state", ""))),
            ),
        )

    def _fill(self, trade: dict, client_id: str) -> ExecutionFill:
        return ExecutionFill(
            price=self._positive(trade["price"], "fill price"),
            quantity=self._positive(trade["baseVolume"], "fill quantity"),
            order_id=str(trade["orderId"]),
            deal_id=str(trade["tradeId"]),
            client_order_id=client_id,
            executed_at_utc=self._timestamp(trade["cTime"]),
        )

    def _execution_fills(self, result, *, is_buy):
        if not isinstance(result, dict):
            return ()
        client_id = str(result.get("clientOid") or "")
        order_id = result.get("orderId")
        if order_id:
            try:
                fills = tuple(
                    self._fill(trade, client_id)
                    for trade in self._pages(
                        "/api/v2/mix/order/fills",
                        self._symbol_params(orderId=str(order_id)),
                        "fillList",
                    )
                )
                if fills:
                    return fills
            except Exception as exc:
                self.logger.warning("Bitget fill lookup failed: %s", self._redact(exc))
        quantity = float(result.get("baseVolume") or 0)
        price = float(result.get("priceAvg") or 0)
        if quantity <= 0 or price <= 0:
            return ()
        return (
            ExecutionFill(
                price=price,
                quantity=quantity,
                order_id=str(order_id or ""),
                client_order_id=client_id,
                executed_at_utc=self._timestamp(result["uTime"]),
                is_aggregate=True,
            ),
        )

    def _emit_order_event(self, order: dict, fill: ExecutionFill | None = None) -> bool:
        order = self._attribute_order(order)
        client_id = str(order.get("clientOid") or "")
        action = self._order_action(client_id)
        if not action:
            return False
        order_id = str(order["orderId"])
        status = self._status(str(order.get("state") or order.get("status") or ""))
        timestamp = fill.executed_at_utc if fill else self._timestamp(order["uTime"])
        # Bitget's hedge-mode request side denotes the held direction; traces use execution side.
        if action in {"sl", "tp"} or str(order.get("tradeSide")) == "close":
            pos_side = order.get("posSide")
            side = (
                "sell"
                if pos_side == "long"
                else "buy" if pos_side == "short" else str(order["side"])
            )
        else:
            side = str(order["side"])
        account_id = self.get_execution_account_id()
        identity = (
            f"fill:{fill.deal_id}"
            if fill
            else f"{status}:{int(timestamp.timestamp() * 1000)}"
        )
        self._emit_execution_event(
            ExecutionEvent(
                event_id=f"bitget:{account_id}:{self.symbol}:{order_id}:{identity}",
                execution_id=self._execution_ids.get(
                    client_id, f"bitget-{account_id}-{order_id}"
                ),
                status=status,
                event_at_utc=timestamp,
                order_role="entry" if action == "open" else "exit",
                side=side,
                order_id=order_id,
                client_order_id=client_id,
                submitted_quantity=float(order["size"]),
                reason={"sl": "stop_loss", "tp": "take_profit"}.get(action, ""),
                fill=fill,
            )
        )
        return True

    def reconcile_execution_events(self, since_utc: datetime) -> int:
        if since_utc.tzinfo is None:
            raise ValueError("since_utc must be timezone-aware")
        orders = {}
        emitted = 0
        self._restore_triggered_order_owners()
        for trade in self._pages(
            "/api/v2/mix/order/fills",
            self._symbol_params(
                startTime=str(int(since_utc.timestamp() * 1000)),
                endTime=str(int(time.time() * 1000)),
            ),
            "fillList",
        ):
            order_id = str(trade["orderId"])
            if order_id not in orders:
                orders[order_id] = self._order_detail({"orderId": order_id})
            order = self._attribute_order(orders[order_id])
            if self._order_action(str(order.get("clientOid") or "")):
                emitted += self._emit_order_event(
                    order, self._fill(trade, str(order["clientOid"]))
                )
        return emitted

    def _start_user_stream(self):
        for target, name in (
            (self._run_user_stream, "socket"),
            (self._run_heartbeat, "heartbeat"),
            (self._run_stream_events, "events"),
        ):
            thread = threading.Thread(
                target=target, name=f"bitget-{name}-{self.symbol}", daemon=True
            )
            self._stream_threads.append(thread)
            thread.start()

    def wait_for_user_stream(self, timeout: float = 15.0) -> bool:
        return self._stream_ready.wait(timeout)

    def _run_user_stream(self):
        while not self._stream_stop.is_set():
            app = websocket.WebSocketApp(
                self.USER_STREAM_URL,
                on_open=self._on_stream_open,
                on_message=self._on_stream_message,
                on_error=self._on_stream_error,
                on_close=self._on_stream_close,
            )
            with self._stream_lock:
                if self._stream_stop.is_set():
                    return
                self._stream_app = app
            try:
                app.run_forever(ping_interval=0)
            except Exception as exc:
                self._on_stream_error(app, exc)
            finally:
                self._stream_ready.clear()
                with self._stream_lock:
                    self._stream_app = None
            self._stream_stop.wait(self.USER_STREAM_RECONNECT_SECONDS)

    def _on_stream_open(self, app):
        self._subscribed_channels.clear()
        self._stream_error = None
        self._last_pong = time.monotonic()
        timestamp = str(int(time.time() * 1000))
        app.send(
            json.dumps(
                {
                    "op": "login",
                    "args": [
                        {
                            "apiKey": self.api_key,
                            "passphrase": self.passphrase,
                            "timestamp": timestamp,
                            "sign": self._sign(timestamp + "GET/user/verify"),
                        }
                    ],
                }
            )
        )

    def _on_stream_message(self, app, raw_message):
        if raw_message == "pong":
            self._last_pong = time.monotonic()
            return
        try:
            payload = json.loads(raw_message)
            event = payload.get("event")
            if event == "error" or (
                event == "login" and str(payload.get("code")) != "0"
            ):
                self._stream_error = (
                    f"Bitget user stream rejected: code={payload.get('code')}"
                )
                self.logger.error(self._stream_error)
                app.close()
            elif event == "login":
                app.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "args": [
                                {
                                    "instType": self.PRODUCT_TYPE,
                                    "channel": channel,
                                    "instId": "default",
                                }
                                for channel in ("orders", "positions", "orders-algo")
                            ],
                        }
                    )
                )
            elif event == "subscribe":
                self._subscribed_channels.add(payload.get("arg", {}).get("channel"))
                if self._subscribed_channels >= {"orders", "positions", "orders-algo"}:
                    self._stream_ready.set()
                    self._stream_messages.put({"reconcile": True})
            elif "data" in payload:
                self._stream_messages.put(payload)
        except Exception as exc:
            self.logger.warning(
                "Bitget user stream message failed: %s", self._redact(exc)
            )

    def _on_stream_error(self, _app, error):
        self._stream_error = self._redact(error)
        if not self._stream_stop.is_set():
            self.logger.warning("Bitget user stream error: %s", self._stream_error)

    def _on_stream_close(self, _app, _status, _message):
        self._stream_ready.clear()

    def _run_heartbeat(self):
        while not self._stream_stop.wait(self.USER_STREAM_PING_SECONDS):
            with self._stream_lock:
                app = self._stream_app
            if app is None or not app.sock or not app.sock.connected:
                continue
            try:
                if (
                    time.monotonic() - self._last_pong
                    > self.USER_STREAM_PONG_TIMEOUT_SECONDS
                ):
                    app.close()
                else:
                    app.send("ping")
            except Exception as exc:
                self._on_stream_error(app, exc)
                app.close()

    def _run_stream_events(self):
        last_reconciliation = time.monotonic()
        while not self._stream_stop.is_set():
            try:
                payload = self._stream_messages.get(timeout=0.5)
            except queue.Empty:
                payload = None
            try:
                if payload is not None:
                    self._handle_user_stream_event(payload)
                if self._stream_ready.is_set() and (
                    time.monotonic() - last_reconciliation
                    >= self.RECONCILIATION_INTERVAL_SECONDS
                ):
                    last_reconciliation = time.monotonic()
                    self._handle_user_stream_event({"reconcile": True})
            except Exception as exc:
                self.logger.error(
                    "Bitget event reconciliation failed: %s", self._redact(exc)
                )

    def _handle_user_stream_event(self, payload):
        with self._operation_lock:
            if payload.get("reconcile"):
                now = datetime.now(timezone.utc)
                self.reconcile_execution_events(self._reconcile_since)
                # Overlap avoids losing fills that reach REST just after the previous scan.
                self._reconcile_since = now - timedelta(seconds=5)
                self._sync_protective_orders()
                relevant = True
            else:
                channel = payload.get("arg", {}).get("channel")
                relevant = False
                for order in payload.get("data", []):
                    if (
                        str(order.get("instId") or order.get("symbol") or "").upper()
                        != self.symbol
                    ):
                        continue
                    relevant = True
                    if channel == "orders":
                        if not self._order_action(
                            str(order.get("clientOid") or "")
                        ) and (
                            "profit" in str(order.get("orderSource"))
                            or "loss" in str(order.get("orderSource"))
                        ):
                            self._restore_triggered_order_owners()
                        order = self._attribute_order(order)
                        fill = None
                        if float(order.get("baseVolume") or 0) > 0 and order.get(
                            "tradeId"
                        ):
                            fill = self._fill(
                                {
                                    **order,
                                    "price": order["fillPrice"],
                                    "cTime": order["fillTime"],
                                },
                                str(order.get("clientOid") or ""),
                            )
                        self._emit_order_event(order, fill)
                    elif channel == "orders-algo":
                        if order.get("status") == "executed" and self._order_action(
                            str(order.get("clientOid") or "")
                        ) in {"sl", "tp"}:
                            for plan in self._restore_triggered_order_owners(
                                orderId=str(order["orderId"])
                            ):
                                child = self._attribute_order(
                                    self._order_detail(
                                        {"orderId": plan["executeOrderId"]}
                                    )
                                )
                                for fill in self._execution_fills(
                                    child, is_buy=child["side"] == "buy"
                                ):
                                    self._emit_order_event(child, fill)
                        self._sync_protective_orders()
            if relevant and not self.read_only and self._position() is None:
                self._cancel_owned_protective_orders()

    def shutdown(self):
        self._stream_stop.set()
        self._stream_ready.clear()
        with self._stream_lock:
            app = self._stream_app
        if app is not None:
            app.close()
        for thread in self._stream_threads:
            if thread is not threading.current_thread():
                thread.join(timeout=self.timeout + 1)
        self._stream_threads.clear()
        self.session.close()
