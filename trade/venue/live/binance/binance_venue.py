"""Binance USD-M futures implementation of the shared live venue protocol."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import hashlib
import hmac
import itertools
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Iterable
from urllib.parse import urlencode

import requests
import websocket

from trade.core.dashboard_base import (
    collect_dashboard,
    AccountBalance,
    AccountDashboard,
    AccountPosition,
    AccountPositionComponent,
    MarginMode,
    PositionSide,
)
from trade.core.execution import (
    ExecutionEvent,
    ExecutionFill,
    ExecutionOrder,
)
from trade.core.protocol import OrderType, PositionDir, PositionView
from trade.core.venue_base import VenueBase
from trade.monitoring.request_budget import binance_dashboard_budget


class BinanceVenue(VenueBase, AccountDashboard):
    """Binance USD-M futures venue with fail-closed bracket creation."""

    BASE_URL = "https://fapi.binance.com"
    API_KEY_FILES = ("hmac_api_key", "binance_api_key")
    API_SECRET_FILES = ("hmac_secret", "binance_api_secret")
    USER_STREAM_URL = "wss://fstream.binance.com/ws/{listen_key}"
    USER_STREAM_RECONNECT_SECONDS = 5.0
    USER_STREAM_KEEPALIVE_SECONDS = 30.0 * 60.0
    TRADE_HISTORY_LIMIT = 1000
    TRADE_HISTORY_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
    TRADE_HISTORY_LOOKBACK_MS = 180 * 24 * 60 * 60 * 1000
    POSITION_HISTORY_RETRIES = 2

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
    ):
        self.symbol = str(symbol).upper()
        self.magic = str(magic or "financial-ml-system")
        self._order_id_sequence = itertools.count()
        self._execution_ids_by_client_order_id: dict[str, str] = {}
        self.logger = logger or logging.getLogger("trade.binance")
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        self._request_lock = threading.RLock()
        self._dashboard_local = threading.local()
        self._dashboard_sessions = []
        self._dashboard_sessions_lock = threading.Lock()
        self._dashboard_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="binance-dashboard")
        self._protective_order_lock = threading.RLock()
        self._protective_orders: dict[str, dict[str, Any]] = {}
        self._user_stream_enabled = bool(enable_user_stream)
        self._user_stream_stop = threading.Event()
        self._user_stream_lock = threading.RLock()
        self._user_stream_listen_key: str | None = None
        self._user_stream_app: websocket.WebSocketApp | None = None
        self._user_stream_thread: threading.Thread | None = None
        self._user_stream_keepalive_thread: threading.Thread | None = None
        try:
            self.api_key = self._load_first(key_path, self.API_KEY_FILES)
            self.api_secret = self._load_first(key_path, self.API_SECRET_FILES)
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})
            self.quantity_step, self.minimum_quantity, self.price_tick = (
                self._load_filters()
            )
            self.hedge_mode = self._load_hedge_mode()
            self._restore_owned_protective_orders()
            if self._user_stream_enabled:
                self._start_user_stream()
        except Exception:
            self._dashboard_pool.shutdown(wait=True, cancel_futures=True)
            self._stop_user_stream()
            self.session.close()
            raise

    @staticmethod
    def _load_first(directory: str, candidates: Iterable[str]) -> str:
        for filename in candidates:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    value = handle.read().strip()
                if value:
                    return value
        raise FileNotFoundError(
            f"No Binance credential file found in {directory}: {tuple(candidates)}"
        )

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = False,
    ) -> Any:
        started = time.monotonic()
        acquired = request_started = response_received = None
        response = None
        success = False
        dashboard_session = getattr(getattr(self, "_dashboard_local", None), "session", None)
        dashboard = dashboard_session is not None
        session = dashboard_session if dashboard else self.session
        try:
            if dashboard and method.upper() != "GET":
                raise RuntimeError("Dashboard transport only accepts read requests")
            # Header observations include traffic from other accounts on this IP.
            binance_dashboard_budget.reserve(5 if dashboard else 1, dashboard=dashboard)
            with nullcontext() if dashboard else self._request_lock:
                acquired = time.monotonic()
                payload = dict(params or {})
                if signed:
                    payload.setdefault("recvWindow", 5000)
                    payload["timestamp"] = int(time.time() * 1000)
                    query = urlencode(payload)
                    payload["signature"] = hmac.new(
                        self.api_secret.encode("utf-8"),
                        query.encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest()
                request_started = time.monotonic()
                response = session.request(
                    method,
                    f"{self.BASE_URL}{path}",
                    params=payload if method.upper() in {"GET", "DELETE"} else None,
                    data=payload if method.upper() not in {"GET", "DELETE"} else None,
                    timeout=self.timeout,
                )
                response_received = time.monotonic()
                binance_dashboard_budget.observe(response.headers, response.status_code)
                try:
                    body = response.json()
                except ValueError as exc:
                    raise RuntimeError(
                        f"Binance returned a non-JSON response for {path}: "
                        f"HTTP {response.status_code}"
                    ) from exc
                if not response.ok:
                    raise RuntimeError(
                        f"Binance request failed for {path}: HTTP {response.status_code}, "
                        f"code={body.get('code')}, message={body.get('msg')}"
                    )
                success = True
                return body
        finally:
            finished = time.monotonic()
            self.logger.info(
                "Binance API timing | method=%s endpoint=%s symbol=%s thread=%s "
                "success=%s status=%s total_ms=%.1f lock_wait_ms=%.1f "
                "http_ms=%.1f decode_ms=%.1f",
                method, path, self.symbol, threading.current_thread().name,
                success, None if response is None else response.status_code,
                (finished - started) * 1000,
                ((finished if acquired is None else acquired) - started) * 1000,
                0.0 if request_started is None else
                ((finished if response_received is None else response_received) - request_started) * 1000,
                0.0 if response_received is None else (finished - response_received) * 1000,
            )

    def _load_filters(self) -> tuple[Decimal, Decimal, Decimal]:
        payload = self._request("GET", "/fapi/v1/exchangeInfo")
        binance_dashboard_budget.configure(payload.get("rateLimits", []))
        symbol_info = next(
            (
                item
                for item in payload.get("symbols", [])
                if item.get("symbol") == self.symbol
            ),
            None,
        )
        if symbol_info is None:
            raise ValueError(f"Binance USD-M symbol is not available: {self.symbol}")
        filters = {item["filterType"]: item for item in symbol_info.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        price = filters.get("PRICE_FILTER")
        if not lot or not price:
            raise RuntimeError(f"Binance symbol filters are incomplete: {self.symbol}")
        return (
            Decimal(str(lot["stepSize"])),
            Decimal(str(lot["minQty"])),
            Decimal(str(price["tickSize"])),
        )

    def _load_hedge_mode(self) -> bool:
        """Return whether the account uses Hedge Mode."""
        mode = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        dual_side = mode.get("dualSidePosition")
        if isinstance(dual_side, bool):
            return dual_side
        normalized = str(dual_side).casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise RuntimeError("Binance returned an invalid position mode response")

    @staticmethod
    def _floor_to_step(value: float, step: Decimal) -> Decimal:
        amount = Decimal(str(value))
        return (amount / step).to_integral_value(rounding=ROUND_DOWN) * step

    @staticmethod
    def _decimal_string(value: Decimal) -> str:
        return format(value.normalize(), "f")

    def _client_order_prefix(self, action: str) -> str:
        safe_magic = re.sub(r"[^.A-Za-z0-9_:/-]", "_", self.magic)
        suffix_length = len(f":{action}:") + 14
        owner_length = 36 - suffix_length
        if len(safe_magic) > owner_length:
            digest = hashlib.sha256(self.magic.encode("utf-8")).hexdigest()[:8]
            safe_magic = f"{safe_magic[:owner_length - len(digest) - 1]}-{digest}"
        return f"{safe_magic}:{action}:"

    def _new_client_order_id(
        self,
        action: str,
        execution_id: str | None = None,
    ) -> str:
        nonce_source = (
            str(execution_id)
            if execution_id
            else f"{time.time_ns():x}{next(self._order_id_sequence):x}"
        )
        nonce = re.sub(r"[^A-Za-z0-9_-]", "_", nonce_source)[-14:]
        client_order_id = f"{self._client_order_prefix(action)}{nonce}"
        if execution_id:
            self._execution_ids_by_client_order_id[client_order_id] = str(execution_id)
        return client_order_id

    def _owns_client_order_id(self, client_order_id: str) -> bool:
        return any(
            client_order_id.startswith(self._client_order_prefix(action))
            for action in ("open", "close", "sl", "tp")
        )

    def get_execution_account_id(self) -> str:
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16]

    def normalize_order_quantity(self, size: float) -> float:
        quantity = self._floor_to_step(float(size), self.quantity_step)
        if quantity < self.minimum_quantity:
            raise ValueError(
                f"Binance order quantity {quantity} is below minimum "
                f"{self.minimum_quantity}"
            )
        return float(quantity)

    def _owns_protective_order(self, order: dict[str, Any]) -> bool:
        client_algo_id = str(order.get("clientAlgoId", ""))
        return client_algo_id.startswith(
            self._client_order_prefix("sl")
        ) or client_algo_id.startswith(self._client_order_prefix("tp"))

    def _remember_protective_order(
        self,
        response: dict[str, Any],
        *,
        role: str,
        client_algo_id: str,
        trigger_price: str,
        limit_price: str | None = None,
    ) -> None:
        algo_id = response.get("algoId")
        returned_client_id = str(response.get("clientAlgoId") or client_algo_id)
        if algo_id is None and not returned_client_id:
            raise RuntimeError(
                "Binance protective order response has no order identifier"
            )
        key = str(algo_id) if algo_id is not None else returned_client_id
        with self._protective_order_lock:
            self._protective_orders[key] = {
                "algoId": algo_id,
                "clientAlgoId": returned_client_id,
                "role": role,
                "triggerPrice": str(response.get("triggerPrice") or trigger_price),
                "price": str(response.get("price") or limit_price or ""),
            }

    def _open_owned_protective_orders(self) -> list[dict[str, Any]]:
        orders = self._request(
            "GET",
            "/fapi/v1/openAlgoOrders",
            {"symbol": self.symbol, "algoType": "CONDITIONAL"},
            signed=True,
        )
        if not isinstance(orders, list):
            raise RuntimeError("Binance returned an invalid open algo order response")
        return [order for order in orders if self._owns_protective_order(order)]

    def _sync_protective_orders(self, orders: Iterable[dict[str, Any]]) -> None:
        synchronized = {}
        for order in orders:
            algo_id = order.get("algoId")
            client_algo_id = str(order.get("clientAlgoId", ""))
            key = str(algo_id) if algo_id is not None else client_algo_id
            if not key:
                continue
            synchronized[key] = {
                "algoId": algo_id,
                "clientAlgoId": client_algo_id,
                "role": (
                    "stop_loss"
                    if str(order.get("orderType", "")).upper().startswith("STOP")
                    else "take_profit"
                ),
                "triggerPrice": str(order.get("triggerPrice") or ""),
                "price": str(order.get("price") or ""),
            }
        with self._protective_order_lock:
            self._protective_orders = synchronized

    def _cancel_owned_protective_orders(
        self,
        *,
        reason: str,
        suppress_errors: bool = False,
    ) -> int:
        with self._protective_order_lock:
            open_orders = self._open_owned_protective_orders()
            self._sync_protective_orders(open_orders)
            canceled = 0
            failures = []
            for order in open_orders:
                params: dict[str, Any]
                if order.get("algoId") is not None:
                    params = {"algoId": order["algoId"]}
                elif order.get("clientAlgoId"):
                    params = {"clientAlgoId": order["clientAlgoId"]}
                else:
                    self.logger.error(
                        "Cannot cancel Binance protective order without an identifier | order=%s",
                        order,
                    )
                    continue
                try:
                    self._request(
                        "DELETE",
                        "/fapi/v1/algoOrder",
                        params,
                        signed=True,
                    )
                except Exception as exc:
                    failures.append((order, exc))
                    self.logger.warning(
                        "Failed to cancel Binance protective order | symbol=%s "
                        "algo_id=%s client_algo_id=%s reason=%s error=%s",
                        self.symbol,
                        order.get("algoId"),
                        order.get("clientAlgoId"),
                        reason,
                        exc,
                    )
                    continue
                canceled += 1
                self.logger.info(
                    "Binance protective order canceled | symbol=%s algo_id=%s "
                    "client_algo_id=%s reason=%s",
                    self.symbol,
                    order.get("algoId"),
                    order.get("clientAlgoId"),
                    reason,
                )
            self._sync_protective_orders(order for order, _ in failures)
            if failures and not suppress_errors:
                raise RuntimeError(
                    f"Failed to cancel {len(failures)} Binance protective order(s) "
                    f"for {self.symbol}"
                ) from failures[0][1]
            return canceled

    def _restore_owned_protective_orders(self) -> None:
        orders = self._open_owned_protective_orders()
        self._sync_protective_orders(orders)
        if orders and self._position() is None:
            self._cancel_owned_protective_orders(reason="startup_flat_reconcile")

    def _start_listen_key(self) -> str:
        response = self._request("POST", "/fapi/v1/listenKey")
        listen_key = (
            str(response.get("listenKey", "")) if isinstance(response, dict) else ""
        )
        if not listen_key:
            raise RuntimeError("Binance returned an invalid user stream listen key")
        return listen_key

    def _start_user_stream(self) -> None:
        listen_key = self._start_listen_key()
        with self._user_stream_lock:
            self._user_stream_listen_key = listen_key
        self._user_stream_thread = threading.Thread(
            target=self._run_user_stream,
            name=f"binance-user-stream-{self.symbol}",
            daemon=True,
        )
        self._user_stream_keepalive_thread = threading.Thread(
            target=self._run_user_stream_keepalive,
            name=f"binance-user-stream-keepalive-{self.symbol}",
            daemon=True,
        )
        self._user_stream_thread.start()
        self._user_stream_keepalive_thread.start()

    def _run_user_stream(self) -> None:
        while not self._user_stream_stop.is_set():
            with self._user_stream_lock:
                listen_key = self._user_stream_listen_key
            if not listen_key:
                try:
                    listen_key = self._start_listen_key()
                    with self._user_stream_lock:
                        self._user_stream_listen_key = listen_key
                except Exception:
                    self.logger.exception(
                        "Failed to create Binance user stream listen key"
                    )
                    self._user_stream_stop.wait(self.USER_STREAM_RECONNECT_SECONDS)
                    continue
            app = websocket.WebSocketApp(
                self.USER_STREAM_URL.format(listen_key=listen_key),
                on_open=self._on_user_stream_open,
                on_message=self._on_user_stream_message,
                on_error=self._on_user_stream_error,
                on_close=self._on_user_stream_close,
            )
            with self._user_stream_lock:
                self._user_stream_app = app
            try:
                app.run_forever(ping_interval=180, ping_timeout=30)
            except Exception:
                if not self._user_stream_stop.is_set():
                    self.logger.exception("Binance user stream failed")
            finally:
                with self._user_stream_lock:
                    if self._user_stream_app is app:
                        self._user_stream_app = None
            if not self._user_stream_stop.wait(self.USER_STREAM_RECONNECT_SECONDS):
                try:
                    replacement = self._start_listen_key()
                    with self._user_stream_lock:
                        self._user_stream_listen_key = replacement
                except Exception:
                    self.logger.exception(
                        "Failed to refresh Binance user stream listen key"
                    )

    def _run_user_stream_keepalive(self) -> None:
        while not self._user_stream_stop.wait(self.USER_STREAM_KEEPALIVE_SECONDS):
            try:
                self._request("PUT", "/fapi/v1/listenKey")
            except Exception:
                self.logger.exception("Binance user stream keepalive failed")
                try:
                    replacement = self._start_listen_key()
                except Exception:
                    self.logger.exception(
                        "Failed to replace Binance user stream listen key"
                    )
                    continue
                with self._user_stream_lock:
                    self._user_stream_listen_key = replacement
                    app = self._user_stream_app
                if app is not None:
                    app.close()

    def _on_user_stream_open(self, _app: websocket.WebSocketApp) -> None:
        self.logger.info("Binance user stream connected | symbol=%s", self.symbol)

    def _on_user_stream_error(self, _app: websocket.WebSocketApp, error: Any) -> None:
        if not self._user_stream_stop.is_set():
            self.logger.warning(
                "Binance user stream error | symbol=%s error=%s",
                self.symbol,
                error,
            )

    def _on_user_stream_close(
        self,
        _app: websocket.WebSocketApp,
        status_code: int | None,
        message: str | None,
    ) -> None:
        if not self._user_stream_stop.is_set():
            self.logger.warning(
                "Binance user stream closed | symbol=%s status=%s message=%s",
                self.symbol,
                status_code,
                message,
            )

    def _on_user_stream_message(
        self,
        _app: websocket.WebSocketApp,
        raw_message: str,
    ) -> None:
        try:
            payload = json.loads(raw_message)
            self._handle_user_stream_event(payload)
        except Exception:
            self.logger.exception("Failed to process Binance user stream event")

    def _handle_user_stream_event(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("e", ""))
        if event_type == "listenKeyExpired":
            with self._user_stream_lock:
                self._user_stream_listen_key = None
                app = self._user_stream_app
            if app is not None:
                app.close()
            return
        relevant = False
        if event_type == "ACCOUNT_UPDATE":
            positions = payload.get("a", {}).get("P", [])
            relevant = any(position.get("s") == self.symbol for position in positions)
        elif event_type == "ORDER_TRADE_UPDATE":
            order = payload.get("o", {})
            self._record_user_stream_execution(payload, order)
            relevant = order.get("s") == self.symbol and order.get("X") == "FILLED"
        if not relevant or self._position() is not None:
            return
        self._cancel_owned_protective_orders(
            reason=f"user_stream_{event_type.lower()}",
            suppress_errors=True,
        )

    def _record_user_stream_execution(
        self,
        payload: dict[str, Any],
        order: dict[str, Any],
    ) -> None:
        if order.get("s") != self.symbol:
            return
        client_order_id = str(order.get("c", ""))
        if not self._owns_client_order_id(client_order_id):
            return
        action = next(
            (
                candidate
                for candidate in ("open", "close", "sl", "tp")
                if client_order_id.startswith(self._client_order_prefix(candidate))
            ),
            "",
        )
        raw_status = str(order.get("X", "")).upper()
        status = {
            "NEW": "accepted",
            "PARTIALLY_FILLED": "partially_filled",
            "FILLED": "filled",
            "CANCELED": "cancelled",
            "EXPIRED": "expired",
            "REJECTED": "rejected",
        }.get(raw_status, raw_status.casefold() or "unknown")
        order_id = str(order.get("i", ""))
        execution_id = self._execution_ids_by_client_order_id.get(
            client_order_id,
            f"binance-{self.get_execution_account_id()}-{order_id}",
        )
        event_time_ms = int(order.get("T") or payload.get("E") or 0)
        event_at_utc = (
            datetime.fromtimestamp(event_time_ms / 1000.0, tz=timezone.utc)
            if event_time_ms > 0
            else datetime.now(timezone.utc)
        )
        last_quantity = float(order.get("l", 0.0) or 0.0)
        last_price = float(order.get("L", 0.0) or 0.0)
        trade_id = str(order.get("t", "") or "")
        fill = None
        if (
            str(order.get("x", "")).upper() == "TRADE"
            and last_quantity > 0
            and last_price > 0
        ):
            fill = ExecutionFill(
                price=last_price,
                quantity=last_quantity,
                order_id=order_id,
                deal_id=trade_id,
                client_order_id=client_order_id,
                executed_at_utc=event_at_utc,
            )
        self._emit_execution_event(
            ExecutionEvent(
                event_id=(
                    f"binance:{self.get_execution_account_id()}:"
                    f"{self.symbol}:{order_id}:{trade_id or status}:"
                    f"{event_time_ms}"
                ),
                execution_id=execution_id,
                status=status,
                event_at_utc=event_at_utc,
                order_role="entry" if action == "open" else "exit",
                side=str(order.get("S", "")).casefold(),
                order_id=order_id,
                client_order_id=client_order_id,
                submitted_quantity=float(order.get("q", 0.0) or 0.0),
                reason={"sl": "stop_loss", "tp": "take_profit"}.get(
                    action,
                    "",
                ),
                fill=fill,
            )
        )

    def reconcile_execution_events(self, since_utc: datetime) -> int:
        if since_utc.tzinfo is None:
            raise ValueError("since_utc must be timezone-aware")
        start_ms = int(since_utc.astimezone(timezone.utc).timestamp() * 1000)
        cursor_end_ms = int(time.time() * 1000)
        collected: dict[tuple[str, ...], dict[str, Any]] = {}
        while cursor_end_ms >= start_ms:
            trades = self._request(
                "GET",
                "/fapi/v1/userTrades",
                {
                    "symbol": self.symbol,
                    "startTime": start_ms,
                    "endTime": cursor_end_ms,
                    "limit": self.TRADE_HISTORY_LIMIT,
                },
                signed=True,
            )
            if not isinstance(trades, list):
                raise RuntimeError("Binance returned invalid account trade history")
            if not trades:
                break
            for trade in trades:
                collected[self._trade_identity(trade)] = trade
            if len(trades) < self.TRADE_HISTORY_LIMIT:
                break
            earliest = min(int(trade.get("time", 0) or 0) for trade in trades)
            if earliest <= start_ms:
                break
            cursor_end_ms = earliest - 1

        orders: dict[str, dict[str, Any]] = {}
        emitted = 0
        for trade in sorted(collected.values(), key=self._trade_sort_key):
            order_id = str(trade.get("orderId", "") or "")
            if not order_id:
                continue
            order = orders.get(order_id)
            if order is None:
                order = self._request(
                    "GET",
                    "/fapi/v1/order",
                    {"symbol": self.symbol, "orderId": order_id},
                    signed=True,
                )
                if not isinstance(order, dict):
                    continue
                orders[order_id] = order
            client_order_id = str(order.get("clientOrderId", ""))
            if not self._owns_client_order_id(client_order_id):
                continue
            action = next(
                candidate
                for candidate in ("open", "close", "sl", "tp")
                if client_order_id.startswith(self._client_order_prefix(candidate))
            )
            trade_time_ms = int(trade.get("time", 0) or 0)
            executed_at_utc = datetime.fromtimestamp(
                trade_time_ms / 1000.0,
                tz=timezone.utc,
            )
            deal_id = str(trade.get("id", "") or "")
            execution_id = self._execution_ids_by_client_order_id.get(
                client_order_id,
                f"binance-{self.get_execution_account_id()}-{order_id}",
            )
            fill = ExecutionFill(
                price=float(trade["price"]),
                quantity=float(trade["qty"]),
                order_id=order_id,
                deal_id=deal_id,
                client_order_id=client_order_id,
                executed_at_utc=executed_at_utc,
            )
            self._emit_execution_event(
                ExecutionEvent(
                    event_id=(
                        f"binance:{self.get_execution_account_id()}:"
                        f"{self.symbol}:{order_id}:{deal_id}:{trade_time_ms}"
                    ),
                    execution_id=execution_id,
                    status=(
                        "filled"
                        if str(order.get("status", "")).upper() == "FILLED"
                        else "partially_filled"
                    ),
                    event_at_utc=executed_at_utc,
                    order_role="entry" if action == "open" else "exit",
                    side=str(trade.get("side", "")).casefold(),
                    order_id=order_id,
                    client_order_id=client_order_id,
                    submitted_quantity=float(
                        order.get("origQty") or trade.get("qty") or 0.0
                    ),
                    reason={
                        "sl": "stop_loss",
                        "tp": "take_profit",
                    }.get(action, "offline_reconciliation"),
                    fill=fill,
                )
            )
            emitted += 1
        return emitted

    def _stop_user_stream(self) -> None:
        self._user_stream_stop.set()
        with self._user_stream_lock:
            app = self._user_stream_app
        if app is not None:
            app.close()
        current = threading.current_thread()
        for thread in (
            self._user_stream_thread,
            self._user_stream_keepalive_thread,
        ):
            if thread is not None and thread is not current:
                thread.join(timeout=5.0)
        self._user_stream_thread = None
        self._user_stream_keepalive_thread = None

    def _position(self) -> dict[str, Any] | None:
        positions = self._request(
            "GET",
            "/fapi/v3/positionRisk",
            {"symbol": self.symbol},
            signed=True,
        )
        if not isinstance(positions, list):
            raise RuntimeError("Binance returned an invalid position response")
        active_positions = [
            position
            for position in positions
            if position.get("symbol") == self.symbol
            and float(position.get("positionAmt", 0.0)) != 0.0
        ]
        if len(active_positions) > 1:
            raise RuntimeError(
                "Binance account has simultaneous LONG and SHORT positions for "
                f"{self.symbol}; this strategy supports one active position"
            )
        if not active_positions:
            return None

        position = active_positions[0]
        if self.hedge_mode:
            position_side = position.get("positionSide")
            quantity = float(position.get("positionAmt", 0.0))
            if position_side not in {"LONG", "SHORT"}:
                raise RuntimeError(
                    "Binance Hedge Mode position has no valid positionSide"
                )
            if (position_side == "LONG" and quantity <= 0) or (
                position_side == "SHORT" and quantity >= 0
            ):
                raise RuntimeError(
                    "Binance Hedge Mode position direction is inconsistent"
                )
        return position

    def get_account_equity(self) -> float:
        account = self._request("GET", "/fapi/v3/account", signed=True)
        equity = float(account.get("totalMarginBalance", 0.0))
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError("Binance returned invalid account equity")
        return equity

    def _dashboard_read(self, operation):
        if not hasattr(self._dashboard_local, "session"):
            session = requests.Session()
            session.headers.update({"X-MBX-APIKEY": self.api_key})
            self._dashboard_local.session = session
            with self._dashboard_sessions_lock:
                self._dashboard_sessions.append(session)
        return operation()

    def get_dashboard_snapshot(self):
        balance = self._dashboard_pool.submit(self._dashboard_read, self.get_dashboard_balance)
        position = self._dashboard_pool.submit(self._dashboard_read, self.get_dashboard_position)
        return collect_dashboard(balance.result, position.result)

    def get_dashboard_position_open_time(self, position):
        return self._dashboard_pool.submit(self._dashboard_read, self.get_last_position_open_time).result()

    def get_dashboard_balance(self) -> AccountBalance:
        account = self._request("GET", "/fapi/v3/account", signed=True)
        balance = float(account.get("totalWalletBalance", 0.0))
        equity = float(account.get("totalMarginBalance", 0.0))
        if not all(math.isfinite(value) for value in (balance, equity)):
            raise RuntimeError("Binance returned invalid dashboard balance data")
        margin = account.get("totalPositionInitialMargin")
        return AccountBalance(
            balance=balance,
            equity=equity,
            used_margin=None if margin in (None, "") else float(margin),
        )

    def get_dashboard_position(self) -> AccountPosition | None:
        position = self._position()
        if position is None:
            return None
        quantity = abs(float(position["positionAmt"]))
        entry_price = float(position.get("entryPrice", 0.0))
        mark_price = float(position.get("markPrice", 0.0) or 0.0)
        if mark_price <= 0:
            mark_price = self._latest_price()
        unrealized_pnl = float(position.get("unRealizedProfit", 0.0) or 0.0)
        cost = entry_price * quantity
        raw_margin_mode = str(position.get("marginType", "")).casefold()
        margin_mode = {
            "cross": MarginMode.CROSS,
            "crossed": MarginMode.CROSS,
            "isolated": MarginMode.ISOLATED,
        }.get(raw_margin_mode, MarginMode.UNKNOWN)
        if self.hedge_mode:
            side = (
                PositionSide.LONG
                if position["positionSide"] == "LONG"
                else PositionSide.SHORT
            )
        else:
            side = (
                PositionSide.LONG
                if float(position["positionAmt"]) > 0
                else PositionSide.SHORT
            )
        notional = abs(float(position.get("notional", mark_price * quantity)))
        leverage = float(position.get("leverage", 0.0) or 0.0) or None
        liquidation_price = float(position.get("liquidationPrice", 0.0) or 0.0) or None
        stop_loss_price = self._cached_protective_price("stop_loss")
        take_profit_price = self._cached_protective_price("take_profit")
        return AccountPosition(
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            mark_price=mark_price,
            notional=notional,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl / cost if cost > 0 else None,
            leverage=leverage,
            liquidation_price=liquidation_price,
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

    def _cached_protective_price(self, role: str) -> float | None:
        with self._protective_order_lock:
            orders = [
                dict(order)
                for order in self._protective_orders.values()
                if order.get("role") == role
            ]
        prices = set()
        for order in orders:
            raw_price = (
                order.get("price")
                if role == "take_profit"
                else order.get("triggerPrice")
            )
            if not raw_price and role == "take_profit":
                raw_price = order.get("triggerPrice")
            price = float(raw_price or 0.0)
            if math.isfinite(price) and price > 0:
                prices.add(price)
        return prices.pop() if len(prices) == 1 else None

    def get_current_state(self) -> PositionView:
        position = self._position()
        if position is None:
            with self._protective_order_lock:
                has_tracked_orders = bool(self._protective_orders)
            if has_tracked_orders:
                self._cancel_owned_protective_orders(reason="flat_position_reconcile")
            return PositionView()
        if self.hedge_mode:
            direction = (
                PositionDir.POSITIVE
                if position["positionSide"] == "LONG"
                else PositionDir.NEGATIVE
            )
        else:
            quantity = float(position["positionAmt"])
            direction = PositionDir.POSITIVE if quantity > 0 else PositionDir.NEGATIVE
        return PositionView(
            dir=direction,
            size=abs(float(position["positionAmt"])),
            price=float(position.get("entryPrice", 0.0)),
        )

    def get_last_position_open_time(self):
        for attempt in range(1, self.POSITION_HISTORY_RETRIES + 1):
            position_before = self._position()
            if position_before is None:
                return None

            trades, opening_trade, opening_quantity = (
                self._load_current_position_trade_cycle(position_before)
            )

            position_after = self._position()
            if position_after is None:
                self.logger.warning(
                    "Binance position closed while reconstructing its opening time "
                    "| symbol=%s attempt=%s",
                    self.symbol,
                    attempt,
                )
                continue
            if self._position_history_signature(
                position_before
            ) != self._position_history_signature(position_after):
                self.logger.warning(
                    "Binance position changed while reconstructing its opening time "
                    "| symbol=%s attempt=%s",
                    self.symbol,
                    attempt,
                )
                continue

            self._validate_reconstructed_position(
                position_after,
                trades,
                opening_trade,
                opening_quantity,
            )
            self._check_opening_order_ownership(opening_trade)
            opening_time_ms = int(opening_trade.get("time", 0) or 0)
            if opening_time_ms <= 0:
                raise RuntimeError(
                    "Binance opening trade has no valid execution timestamp"
                )
            return datetime.fromtimestamp(
                opening_time_ms / 1000.0,
                tz=timezone.utc,
            )

        final_position = self._position()
        if final_position is None:
            return None
        raise RuntimeError(
            "Binance position kept changing while reconstructing its opening time: "
            f"{self.symbol}"
        )

    @staticmethod
    def _position_history_signature(position: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(position.get("positionSide", "BOTH")),
            str(position.get("positionAmt", "0")),
            str(position.get("entryPrice", "0")),
            str(position.get("updateTime", "0")),
        )

    def _trade_applies_to_position(
        self,
        trade: dict[str, Any],
        position: dict[str, Any],
    ) -> bool:
        if str(trade.get("symbol", self.symbol)).upper() != self.symbol:
            return False
        if not self.hedge_mode:
            return True
        return str(trade.get("positionSide", "")) == str(
            position.get("positionSide", "")
        )

    @staticmethod
    def _trade_sort_key(trade: dict[str, Any]) -> tuple[int, int]:
        return (
            int(trade.get("time", 0) or 0),
            int(trade.get("id", 0) or 0),
        )

    @staticmethod
    def _trade_identity(trade: dict[str, Any]) -> tuple[str, ...]:
        return (
            str(trade.get("id", "")),
            str(trade.get("orderId", "")),
            str(trade.get("time", "")),
            str(trade.get("side", "")),
            str(trade.get("positionSide", "")),
            str(trade.get("qty", "")),
            str(trade.get("price", "")),
        )

    @staticmethod
    def _signed_trade_quantity(trade: dict[str, Any]) -> Decimal:
        quantity = Decimal(str(trade.get("qty", "0")))
        if quantity <= 0:
            raise RuntimeError("Binance trade history contains an invalid quantity")
        side = str(trade.get("side", "")).upper()
        if side == "BUY":
            return quantity
        if side == "SELL":
            return -quantity
        raise RuntimeError("Binance trade history contains an invalid side")

    def _quantity_sign(self, quantity: Decimal) -> int:
        tolerance = self.quantity_step / Decimal("2")
        if abs(quantity) <= tolerance:
            return 0
        return 1 if quantity > 0 else -1

    def _find_position_opening_trade(
        self,
        current_quantity: Decimal,
        trades: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], Decimal] | None:
        current_sign = self._quantity_sign(current_quantity)
        quantity_after = current_quantity
        for trade in reversed(trades):
            quantity_before = quantity_after - self._signed_trade_quantity(trade)
            if (
                self._quantity_sign(quantity_after) == current_sign
                and self._quantity_sign(quantity_before) != current_sign
            ):
                return trade, quantity_after
            quantity_after = quantity_before
        return None

    def _load_current_position_trade_cycle(
        self,
        position: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], Decimal]:
        current_quantity = Decimal(str(position.get("positionAmt", "0")))
        if self._quantity_sign(current_quantity) == 0:
            raise RuntimeError("Cannot reconstruct a flat Binance position")

        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - self.TRADE_HISTORY_LOOKBACK_MS
        cursor_end_ms = now_ms
        collected: dict[tuple[str, ...], dict[str, Any]] = {}

        while cursor_end_ms >= cutoff_ms:
            response = self._request(
                "GET",
                "/fapi/v1/userTrades",
                {
                    "symbol": self.symbol,
                    "endTime": cursor_end_ms,
                    "limit": self.TRADE_HISTORY_LIMIT,
                },
                signed=True,
            )
            if not isinstance(response, list):
                raise RuntimeError("Binance returned invalid account trade history")

            response_times: list[int] = []
            for trade in response:
                if not isinstance(trade, dict):
                    raise RuntimeError(
                        "Binance account trade history contains an invalid item"
                    )
                trade_time = int(trade.get("time", 0) or 0)
                if trade_time <= 0:
                    raise RuntimeError(
                        "Binance account trade history contains an invalid timestamp"
                    )
                response_times.append(trade_time)
                if self._trade_applies_to_position(trade, position):
                    collected[self._trade_identity(trade)] = trade

            ordered_trades = sorted(
                collected.values(),
                key=self._trade_sort_key,
            )
            opening = self._find_position_opening_trade(
                current_quantity,
                ordered_trades,
            )
            if opening is not None:
                return ordered_trades, opening[0], opening[1]

            if response_times:
                next_cursor = min(response_times) - 1
                if next_cursor >= cursor_end_ms:
                    next_cursor = cursor_end_ms - self.TRADE_HISTORY_WINDOW_MS
                cursor_end_ms = next_cursor
            else:
                cursor_end_ms -= self.TRADE_HISTORY_WINDOW_MS

        raise RuntimeError(
            "Unable to reconstruct Binance position opening time from the "
            f"available trade history: {self.symbol}"
        )

    def _apply_trade_to_position(
        self,
        quantity: Decimal,
        average_price: Decimal,
        trade_quantity: Decimal,
        trade_price: Decimal,
    ) -> tuple[Decimal, Decimal]:
        quantity_sign = self._quantity_sign(quantity)
        trade_sign = self._quantity_sign(trade_quantity)
        if trade_sign == 0:
            raise RuntimeError("Cannot replay a zero-quantity Binance trade")
        if quantity_sign == 0:
            return trade_quantity, trade_price

        new_quantity = quantity + trade_quantity
        new_sign = self._quantity_sign(new_quantity)
        if trade_sign == quantity_sign:
            weighted_price = (
                abs(quantity) * average_price + abs(trade_quantity) * trade_price
            ) / abs(new_quantity)
            return new_quantity, weighted_price
        if new_sign == 0:
            return Decimal("0"), Decimal("0")
        if new_sign == quantity_sign:
            return new_quantity, average_price
        return new_quantity, trade_price

    def _validate_reconstructed_position(
        self,
        position: dict[str, Any],
        trades: list[dict[str, Any]],
        opening_trade: dict[str, Any],
        opening_quantity: Decimal,
    ) -> None:
        try:
            opening_index = next(
                index for index, trade in enumerate(trades) if trade is opening_trade
            )
        except StopIteration as exc:
            raise RuntimeError(
                "Binance opening trade is missing from the reconstructed cycle"
            ) from exc

        opening_price = Decimal(str(opening_trade.get("price", "0")))
        if opening_price <= 0:
            raise RuntimeError("Binance opening trade has an invalid price")
        quantity = opening_quantity
        average_price = opening_price
        for trade in trades[opening_index + 1 :]:
            trade_price = Decimal(str(trade.get("price", "0")))
            if trade_price <= 0:
                raise RuntimeError("Binance trade history contains an invalid price")
            quantity, average_price = self._apply_trade_to_position(
                quantity,
                average_price,
                self._signed_trade_quantity(trade),
                trade_price,
            )

        expected_quantity = Decimal(str(position.get("positionAmt", "0")))
        if self._quantity_sign(quantity - expected_quantity) != 0:
            raise RuntimeError(
                "Binance reconstructed position quantity does not match the "
                f"exchange | symbol={self.symbol} reconstructed={quantity} "
                f"exchange={expected_quantity}"
            )

        expected_price = Decimal(str(position.get("entryPrice", "0")))
        if expected_price <= 0:
            raise RuntimeError("Binance position has an invalid entry price")
        if abs(average_price - expected_price) > self.price_tick:
            raise RuntimeError(
                "Binance reconstructed entry price does not match the exchange "
                f"| symbol={self.symbol} reconstructed={average_price} "
                f"exchange={expected_price}"
            )

    def _check_opening_order_ownership(
        self,
        opening_trade: dict[str, Any],
    ) -> None:
        order_id = opening_trade.get("orderId")
        if order_id is None:
            self.logger.error(
                "Unable to verify Binance position ownership because the opening "
                "trade has no order ID | symbol=%s magic=%s",
                self.symbol,
                self.magic,
            )
            return
        try:
            order = self._request(
                "GET",
                "/fapi/v1/order",
                {"symbol": self.symbol, "orderId": order_id},
                signed=True,
            )
        except Exception:
            self.logger.error(
                "Unable to verify Binance position ownership; continuing with the "
                "reconstructed opening time | symbol=%s magic=%s order_id=%s",
                self.symbol,
                self.magic,
                order_id,
                exc_info=True,
            )
            return

        client_order_id = (
            str(order.get("clientOrderId", "")) if isinstance(order, dict) else ""
        )
        if not client_order_id.startswith(self._client_order_prefix("open")):
            self.logger.error(
                "Binance current position was not opened by this strategy; "
                "continuing with the reconstructed opening time | symbol=%s "
                "magic=%s order_id=%s client_order_id=%s",
                self.symbol,
                self.magic,
                order_id,
                client_order_id or "<missing>",
            )

    def _latest_price(self) -> float:
        payload = self._request(
            "GET",
            "/fapi/v1/ticker/price",
            {"symbol": self.symbol},
        )
        price = float(payload["price"])
        if price <= 0 or not math.isfinite(price):
            raise RuntimeError("Binance returned invalid ticker price")
        return price

    def get_bid_ask(self) -> tuple[float, float]:
        ticker = self._request(
            "GET",
            "/fapi/v1/ticker/bookTicker",
            {"symbol": self.symbol},
        )
        return float(ticker["bidPrice"]), float(ticker["askPrice"])

    def _execution_fills(self, result, *, is_buy: bool) -> tuple[ExecutionFill, ...]:
        if not isinstance(result, dict):
            return ()
        order_id = result.get("orderId")
        client_order_id = str(
            result.get("clientOrderId") or result.get("_trace_client_order_id") or ""
        )
        if order_id is not None:
            try:
                trades = self._request(
                    "GET",
                    "/fapi/v1/userTrades",
                    {"symbol": self.symbol, "orderId": order_id},
                    signed=True,
                )
                fills = tuple(
                    ExecutionFill(
                        price=float(trade["price"]),
                        quantity=float(trade["qty"]),
                        order_id=str(trade.get("orderId", order_id)),
                        deal_id=str(trade.get("id", "")),
                        client_order_id=client_order_id,
                        executed_at_utc=(
                            datetime.fromtimestamp(
                                int(trade["time"]) / 1000.0,
                                tz=timezone.utc,
                            )
                            if trade.get("time") is not None
                            else None
                        ),
                    )
                    for trade in trades
                    if float(trade.get("price", 0.0) or 0.0) > 0
                    and float(trade.get("qty", 0.0) or 0.0) > 0
                )
                if fills:
                    return fills
            except Exception:
                self.logger.warning(
                    "Failed to load Binance trade fills; using aggregate order result",
                    exc_info=True,
                )
        price = float(result.get("avgPrice", 0.0) or 0.0)
        quantity = float(result.get("executedQty", 0.0) or 0.0)
        if price <= 0 or quantity <= 0:
            return ()
        return (
            ExecutionFill(
                price=price,
                quantity=quantity,
                order_id=str(order_id or ""),
                client_order_id=client_order_id,
                executed_at_utc=(
                    datetime.fromtimestamp(
                        int(result.get("updateTime", 0)) / 1000.0,
                        tz=timezone.utc,
                    )
                    if int(result.get("updateTime", 0) or 0) > 0
                    else None
                ),
                is_aggregate=True,
            ),
        )

    def _execution_orders(
        self,
        result,
        *,
        submitted_quantity: float,
        is_buy: bool,
    ) -> tuple[ExecutionOrder, ...]:
        if not isinstance(result, dict):
            return ()
        order_id = str(result.get("orderId", "") or "")
        client_order_id = str(
            result.get("clientOrderId") or result.get("_trace_client_order_id") or ""
        )
        if not order_id and not client_order_id:
            return ()
        raw_status = str(result.get("status", "")).upper()
        status = {
            "NEW": "accepted",
            "PARTIALLY_FILLED": "partially_filled",
            "FILLED": "filled",
            "CANCELED": "cancelled",
            "EXPIRED": "expired",
            "REJECTED": "rejected",
        }.get(raw_status)
        if status is None:
            status = (
                "filled"
                if float(result.get("executedQty", 0.0) or 0.0) > 0
                else "submitted"
            )
        quantity = float(
            result.get("origQty")
            or result.get("_trace_submitted_quantity")
            or submitted_quantity
        )
        return (
            ExecutionOrder(
                order_id=order_id,
                client_order_id=client_order_id,
                submitted_quantity=quantity,
                status=status,
            ),
        )

    def _trigger_price(self, raw_price: float) -> str:
        price = self._floor_to_step(raw_price, self.price_tick)
        if price <= 0:
            raise ValueError("Protective trigger price rounded to zero")
        return self._decimal_string(price)

    def _place_stop_market_order(
        self,
        side: str,
        trigger_price: str,
        position_side: str | None = None,
        execution_id: str | None = None,
    ):
        client_algo_id = self._new_client_order_id("sl", execution_id)
        params = {
            "algoType": "CONDITIONAL",
            "symbol": self.symbol,
            "side": side,
            "type": "STOP_MARKET",
            "triggerPrice": trigger_price,
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "clientAlgoId": client_algo_id,
        }
        if position_side is not None:
            params["positionSide"] = position_side
        response = self._request(
            "POST",
            "/fapi/v1/algoOrder",
            params,
            signed=True,
        )
        self._remember_protective_order(
            response,
            role="stop_loss",
            client_algo_id=client_algo_id,
            trigger_price=trigger_price,
        )
        return response

    def _place_take_profit_limit_order(
        self,
        side: str,
        trigger_price: str,
        quantity: Decimal,
        position_side: str | None = None,
        execution_id: str | None = None,
    ):
        quantity_string = self._decimal_string(quantity)
        client_algo_id = self._new_client_order_id("tp", execution_id)
        params = {
            "algoType": "CONDITIONAL",
            "symbol": self.symbol,
            "side": side,
            "type": "TAKE_PROFIT",
            "timeInForce": "GTC",
            "quantity": quantity_string,
            "price": trigger_price,
            "triggerPrice": trigger_price,
            "workingType": "MARK_PRICE",
            "clientAlgoId": client_algo_id,
        }
        if position_side is not None:
            params["positionSide"] = position_side
        else:
            params["reduceOnly"] = "true"
        response = self._request(
            "POST",
            "/fapi/v1/algoOrder",
            params,
            signed=True,
        )
        self._remember_protective_order(
            response,
            role="take_profit",
            client_algo_id=client_algo_id,
            trigger_price=trigger_price,
            limit_price=trigger_price,
        )
        return response

    def _assert_no_open_orders(self) -> None:
        regular = self._request(
            "GET",
            "/fapi/v1/openOrders",
            {"symbol": self.symbol},
            signed=True,
        )
        conditional = self._request(
            "GET",
            "/fapi/v1/openAlgoOrders",
            {"symbol": self.symbol},
            signed=True,
        )
        if regular or conditional:
            raise RuntimeError(
                f"Refusing to open with existing Binance orders: {self.symbol}"
            )

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
        order_type, price = self.normalize_order_request(order_type, price)
        if order_type == OrderType.LIMIT and (stop_loss_pct or take_profit_pct):
            raise ValueError(
                "Binance resting limit entries with protective orders require "
                "fill monitoring, which is not supported"
            )
        if self._position() is not None:
            raise RuntimeError(
                f"Refusing to open over an existing Binance position: {self.symbol}"
            )
        self._cancel_owned_protective_orders(reason="pre_open_flat_reconcile")
        self._assert_no_open_orders()

        quantity = Decimal(str(self.normalize_order_quantity(float(size))))
        side = "BUY" if is_buy else "SELL"
        exit_side = "SELL" if is_buy else "BUY"
        position_side = ("LONG" if is_buy else "SHORT") if self.hedge_mode else None
        order_params = {
            "symbol": self.symbol,
            "side": side,
            "type": "MARKET" if order_type == OrderType.MARKET else "LIMIT",
            "quantity": self._decimal_string(quantity),
            "newOrderRespType": "RESULT",
            "newClientOrderId": self._new_client_order_id(
                "open",
                execution_id,
            ),
        }
        if position_side is not None:
            order_params["positionSide"] = position_side
        if order_type == OrderType.LIMIT:
            limit_price = self._floor_to_step(price, self.price_tick)
            if limit_price <= 0:
                raise ValueError("Binance limit price rounded to zero")
            order_params.update(
                price=self._decimal_string(limit_price),
                timeInForce="GTC",
            )
        order = self._request(
            "POST",
            "/fapi/v1/order",
            order_params,
            signed=True,
        )
        order["_trace_client_order_id"] = order_params["newClientOrderId"]
        order["_trace_submitted_quantity"] = float(quantity)
        executed_price = None
        if stop_loss_pct or take_profit_pct:
            executed_price = float(order.get("avgPrice", 0.0) or 0.0)
            if not math.isfinite(executed_price) or executed_price <= 0:
                executed_price = self._latest_price()

        try:
            if stop_loss_pct:
                stop_price = executed_price * (
                    1.0 - stop_loss_pct if is_buy else 1.0 + stop_loss_pct
                )
                self._place_stop_market_order(
                    exit_side,
                    self._trigger_price(stop_price),
                    position_side,
                    execution_id,
                )
            if take_profit_pct:
                take_profit_price = executed_price * (
                    1.0 + take_profit_pct if is_buy else 1.0 - take_profit_pct
                )
                self._place_take_profit_limit_order(
                    exit_side,
                    self._trigger_price(take_profit_price),
                    quantity,
                    position_side,
                    execution_id,
                )
        except Exception:
            self.logger.exception(
                "Protective order creation failed; closing the new Binance position"
            )
            self.close_position()
            raise
        return order

    def close_position(self, size=None, execution_id=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported Binance close arguments: {sorted(kwargs)}")
        position = self._position()
        if position is None:
            self._cancel_owned_protective_orders(reason="explicit_close_already_flat")
            return None
        position_quantity = float(position["positionAmt"])
        close_quantity = (
            abs(position_quantity)
            if size is None
            else min(
                abs(position_quantity),
                abs(float(size)),
            )
        )
        quantity = self._floor_to_step(close_quantity, self.quantity_step)
        if quantity < self.minimum_quantity:
            raise ValueError("Binance close quantity is below the symbol minimum")
        order_params = {
            "symbol": self.symbol,
            "side": "SELL" if position_quantity > 0 else "BUY",
            "type": "MARKET",
            "quantity": self._decimal_string(quantity),
            "newOrderRespType": "RESULT",
            "newClientOrderId": self._new_client_order_id(
                "close",
                execution_id,
            ),
        }
        if self.hedge_mode:
            order_params["positionSide"] = position["positionSide"]
        else:
            order_params["reduceOnly"] = "true"
        result = self._request(
            "POST",
            "/fapi/v1/order",
            order_params,
            signed=True,
        )
        result["_trace_client_order_id"] = order_params["newClientOrderId"]
        result["_trace_submitted_quantity"] = float(quantity)
        self._cancel_owned_protective_orders(reason="explicit_close")
        return result

    def shutdown(self):
        self._dashboard_pool.shutdown(wait=True, cancel_futures=True)
        for session in self._dashboard_sessions:
            session.close()
        self._stop_user_stream()
        self.session.close()
