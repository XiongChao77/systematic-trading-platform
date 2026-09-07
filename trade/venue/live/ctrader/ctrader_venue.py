"""Synchronous venue facade for the asynchronous cTrader Open API SDK."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace
from typing import Any, Iterable

from trade.core.dashboard_base import (
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
from trade.core.protocol import Firm, OrderType, PositionDir, PositionView
from trade.core.venue_base import VenueBase
from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
from ctrader_open_api.messages import (
    OpenApiMessages_pb2,
    OpenApiModelMessages_pb2,
)

# Binance USDT market-data symbols to FTMO's current cTrader crypto CFD symbols.
CTRADER_SYMBOL_MAP = {
    "AAVEUSDT": "AAVUSD",
    "ADAUSDT": "ADAUSD",
    "ALGOUSDT": "ALGUSD",
    "AVAXUSDT": "AVAUSD",
    "BCHUSDT": "BCHUSD",
    "BNBUSDT": "BNBUSD",
    "BTCUSDT": "BTCUSD",
    "DASHUSDT": "DASHUSD",
    "DOGEUSDT": "DOGUSD",
    "DOTUSDT": "DOTUSD",
    "ETCUSDT": "ETCUSD",
    "ETHUSDT": "ETHUSD",
    "FETUSDT": "FETUSD",
    "GALAUSDT": "GALUSD",
    "GRTUSDT": "GRTUSD",
    "HBARUSDT": "BARUSD",
    "ICPUSDT": "ICPUSD",
    "IMXUSDT": "IMXUSD",
    "LINKUSDT": "LNKUSD",
    "LTCUSDT": "LTCUSD",
    "MANAUSDT": "MANUSD",
    "MKRUSDT": "MKRUSD",
    "NEARUSDT": "NERUSD",
    "NEOUSDT": "NEOUSD",
    "SANDUSDT": "SANUSD",
    "SOLUSDT": "SOLUSD",
    "UNIUSDT": "UNIUSD",
    "VETUSDT": "VECUSD",
    "XLMUSDT": "XLMUSD",
    "XMRUSDT": "XMRUSD",
    "XRPUSDT": "XRPUSD",
    "XTZUSDT": "XTZUSD",
}


class _CTraderIsolatedTcpProtocol(TcpProtocol):
    """Keep SDK send state isolated between Live and Demo clients."""

    def __init__(self):
        super().__init__()
        self._send_queue = deque()
        self._send_task = None
        self._lastSendMessageTime = None


class CTraderOpenApiConnection:
    """Run one shared cTrader Open API connection for multiple trading accounts."""

    CLIENT_ID_FILES = ("client_id", "ctrader_client_id")
    CLIENT_SECRET_FILES = ("client_secret", "ctrader_client_secret")
    ACCESS_TOKEN_FILES = ("access_token", "ctrader_access_token")
    REFRESH_TOKEN_FILES = ("refresh_token", "ctrader_refresh_token")
    ACCESS_TOKEN_EXPIRES_AT_FILES = (
        "access_token_expires_at",
        "ctrader_access_token_expires_at",
    )

    TOKEN_ENDPOINT = "https://openapi.ctrader.com/apps/token"
    TOKEN_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
    TOKEN_REFRESH_MARGIN_SECONDS = 48 * 60 * 60
    HEARTBEAT_INTERVAL_SECONDS = 10.0

    _reactor_lock = threading.Lock()
    _reactor_ready = threading.Event()
    _reactor_thread: threading.Thread | None = None

    _AUTHENTICATION_MESSAGES = {
        "ProtoOAApplicationAuthReq",
        "ProtoOAGetAccountListByAccessTokenReq",
        "ProtoOAAccountAuthReq",
    }

    def __init__(
        self,
        key_path: str,
        *,
        environment: str = "demo",
        timeout: float = 10.0,
        logger: logging.Logger | None = None,
    ):
        self._protobuf = Protobuf
        self._messages = OpenApiMessages_pb2
        self._timeout = float(timeout)
        self._logger = logger or logging.getLogger("trade.ctrader.connection")
        self._key_path = os.path.realpath(key_path)

        self._connected = threading.Event()
        self._application_authenticated = threading.Event()
        self._authentication_lock = threading.RLock()
        self._token_refresh_lock = threading.Lock()
        self._closed = False
        self._ever_authenticated = False

        self._granted_account_ids: set[int] = set()
        self._account_ids_by_trader_login: dict[int, tuple[int, ...]] = {}
        self._account_environments_by_id: dict[int, str] = {}
        self._authenticated_account_ids: set[int] = set()
        self._registered_account_ids: set[int] = set()
        self._account_ref_counts: dict[int, int] = {}

        self._quotes: dict[tuple[int, int], dict[str, float | int]] = {}
        self._quote_events: dict[tuple[int, int], threading.Event] = {}
        self._quote_lock = threading.Lock()
        self._execution_listeners: set[Any] = set()
        self._execution_listener_lock = threading.Lock()

        self._heartbeat_timer: threading.Timer | None = None
        self._token_check_timer: threading.Timer | None = None

        self._client_id = self._load_first(self._key_path, self.CLIENT_ID_FILES)
        self._client_secret = self._load_first(
            self._key_path,
            self.CLIENT_SECRET_FILES,
        )
        self._access_token = self._load_first(self._key_path, self.ACCESS_TOKEN_FILES)
        self._refresh_token = self._load_first(
            self._key_path,
            self.REFRESH_TOKEN_FILES,
            required=False,
        )
        self._access_token_expires_at = self._load_expires_at()

        normalized_environment = str(environment).strip().casefold()
        if normalized_environment not in {"demo", "live"}:
            raise ValueError("cTrader environment must be 'demo' or 'live'")
        self.environment = normalized_environment
        host = (
            EndPoints.PROTOBUF_LIVE_HOST
            if normalized_environment == "live"
            else EndPoints.PROTOBUF_DEMO_HOST
        )

        self._check_access_token(force_if_expiry_unknown=True)

        self._ensure_reactor()
        self._client = Client(
            host,
            EndPoints.PROTOBUF_PORT,
            _CTraderIsolatedTcpProtocol,
        )
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)
        self._call_in_reactor(self._client.startService)
        if not self._connected.wait(self._timeout):
            self.shutdown()
            raise TimeoutError(
                f"Timed out connecting to cTrader {normalized_environment}"
            )

        try:
            self._authenticate_application()
            self._reload_granted_accounts()
            self._ever_authenticated = True
            self._start_heartbeat_timer()
            self._start_token_check_timer()
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _load_first(
        directory: str,
        candidates: Iterable[str],
        *,
        required: bool = True,
    ) -> str | None:
        for filename in candidates:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    value = handle.read().strip()
                if value:
                    return value
        if required:
            raise FileNotFoundError(
                f"No cTrader credential file found in {directory}: {tuple(candidates)}"
            )
        return None

    def _load_expires_at(self) -> float | None:
        value = self._load_first(
            self._key_path,
            self.ACCESS_TOKEN_EXPIRES_AT_FILES,
            required=False,
        )
        if value is None:
            return None
        try:
            expires_at = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid cTrader access token expiry timestamp: {value!r}"
            ) from exc
        if not math.isfinite(expires_at) or expires_at <= 0:
            raise ValueError(
                f"Invalid cTrader access token expiry timestamp: {value!r}"
            )
        return expires_at

    @staticmethod
    def _atomic_write(path: str, value: str) -> None:
        temp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def _write_primary(self, candidates: Iterable[str], value: str) -> None:
        filename = next(iter(candidates))
        self._atomic_write(os.path.join(self._key_path, filename), value)

    def _check_access_token(self, *, force_if_expiry_unknown: bool = False) -> None:
        now = time.time()
        expiry_unknown = self._access_token_expires_at is None
        refresh_needed = (
            expiry_unknown
            or self._access_token_expires_at - now <= self.TOKEN_REFRESH_MARGIN_SECONDS
        )

        if expiry_unknown and not force_if_expiry_unknown:
            return
        if not refresh_needed:
            return
        if not self._refresh_token:
            if expiry_unknown:
                self._logger.warning(
                    "cTrader access token expiry is unknown and no refresh token is available"
                )
                return
            raise RuntimeError(
                "cTrader access token is near expiry but no refresh token is available"
            )
        self.refresh_access_token()

    def refresh_access_token(self) -> None:
        """Refresh, persist, and validate the cTrader access token."""

        import json
        import urllib.parse
        import urllib.request

        with self._token_refresh_lock:
            disk_access_token = self._load_first(
                self._key_path,
                self.ACCESS_TOKEN_FILES,
            )
            disk_refresh_token = self._load_first(
                self._key_path,
                self.REFRESH_TOKEN_FILES,
                required=False,
            )
            disk_expires_at = self._load_expires_at()

            if (
                disk_access_token != self._access_token
                or disk_refresh_token != self._refresh_token
                or disk_expires_at != self._access_token_expires_at
            ):
                self._access_token = disk_access_token
                self._refresh_token = disk_refresh_token
                self._access_token_expires_at = disk_expires_at
                if (
                    self._access_token_expires_at is not None
                    and self._access_token_expires_at - time.time()
                    > self.TOKEN_REFRESH_MARGIN_SECONDS
                ):
                    return

            if not self._refresh_token:
                raise RuntimeError("cTrader refresh token is not available")

            query = urllib.parse.urlencode(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                }
            )
            request = urllib.request.Request(
                f"{self.TOKEN_ENDPOINT}?{query}",
                method="POST",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                raise RuntimeError("Failed to refresh cTrader access token") from exc

            if payload.get("errorCode"):
                raise RuntimeError(
                    "cTrader token refresh failed: "
                    f"{payload.get('errorCode')}: {payload.get('description', '')}"
                )

            access_token = str(payload.get("accessToken") or "").strip()
            refresh_token = str(payload.get("refreshToken") or "").strip()
            try:
                expires_in = int(payload.get("expiresIn"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "cTrader token refresh returned an invalid expiresIn value"
                ) from exc

            if not access_token or not refresh_token or expires_in <= 0:
                raise RuntimeError(
                    "cTrader token refresh returned incomplete token data"
                )

            expires_at = time.time() + expires_in

            # Persist immediately because a successful refresh invalidates the old token pair.
            self._write_primary(self.ACCESS_TOKEN_FILES, access_token)
            self._write_primary(self.REFRESH_TOKEN_FILES, refresh_token)
            self._write_primary(
                self.ACCESS_TOKEN_EXPIRES_AT_FILES,
                f"{expires_at:.6f}",
            )

            self._access_token = access_token
            self._refresh_token = refresh_token
            self._access_token_expires_at = expires_at

            with self._authentication_lock:
                registered_ids = tuple(self._registered_account_ids)
                self._authenticated_account_ids.clear()

            if self._connected.is_set():
                self._application_authenticated.clear()
                self._authenticate_application()
                self._reload_granted_accounts()
                for account_id in registered_ids:
                    self._authenticate_account(account_id)

            self._logger.info(
                "cTrader access token refreshed | expires_at=%s",
                datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
            )

    @classmethod
    def _ensure_reactor(cls) -> None:
        from twisted.internet import reactor

        with cls._reactor_lock:
            if cls._reactor_thread is not None:
                return
            if reactor.running:
                cls._reactor_thread = threading.current_thread()
                cls._reactor_ready.set()
                return
            if getattr(reactor, "_startedBefore", False):
                raise RuntimeError("The Twisted reactor cannot be restarted")

            def run_reactor() -> None:
                reactor.run(installSignalHandlers=False)

            cls._reactor_thread = threading.Thread(
                target=run_reactor,
                name="ctrader-open-api-reactor",
                daemon=True,
            )
            cls._reactor_thread.start()
        deadline = time.monotonic() + 5.0
        while not reactor.running and time.monotonic() < deadline:
            time.sleep(0.01)
        if not reactor.running:
            raise RuntimeError("Failed to start the cTrader Open API reactor")
        cls._reactor_ready.set()

    @staticmethod
    def _call_in_reactor(callback, *args) -> None:
        from twisted.internet import reactor

        reactor.callFromThread(callback, *args)

    def _on_connected(self, _client) -> None:
        self._connected.set()
        self._logger.info(
            "cTrader Open API connection ready | environment=%s",
            self.environment,
        )
        if self._ever_authenticated and not self._closed:
            threading.Thread(
                target=self._reauthenticate,
                name="ctrader-reauth",
                daemon=True,
            ).start()

    def _on_disconnected(self, _client, reason) -> None:
        self._connected.clear()
        self._application_authenticated.clear()
        with self._authentication_lock:
            self._authenticated_account_ids.clear()
        if not self._closed:
            self._logger.warning("cTrader Open API disconnected: %s", reason)

    def _on_message(self, _client, envelope) -> None:
        try:
            message = self._protobuf.extract(envelope)
        except Exception:
            self._logger.exception("Failed to decode a cTrader Open API message")
            return
        message_name = message.__class__.__name__
        if message_name == "ProtoOAExecutionEvent":
            with self._execution_listener_lock:
                listeners = tuple(self._execution_listeners)
            client_message_id = str(getattr(envelope, "clientMsgId", "") or "")
            for listener in listeners:
                try:
                    listener(message, client_message_id)
                except Exception:
                    self._logger.exception(
                        "Failed to process a cTrader execution event"
                    )
            return
        if message_name != "ProtoOASpotEvent":
            return
        account_id = int(message.ctidTraderAccountId)
        symbol_id = int(message.symbolId)
        with self._authentication_lock:
            if account_id not in self._registered_account_ids:
                return
        quote_key = (account_id, symbol_id)
        with self._quote_lock:
            quote = self._quotes.setdefault(quote_key, {})
            if message.HasField("bid"):
                quote["bid"] = int(message.bid) / 100_000.0
            if message.HasField("ask"):
                quote["ask"] = int(message.ask) / 100_000.0
            if message.HasField("timestamp"):
                quote["timestamp"] = int(message.timestamp)
            self._quote_events.setdefault(quote_key, threading.Event()).set()

    def add_execution_listener(self, listener) -> None:
        with self._execution_listener_lock:
            self._execution_listeners.add(listener)

    def remove_execution_listener(self, listener) -> None:
        with self._execution_listener_lock:
            self._execution_listeners.discard(listener)

    @staticmethod
    def _set_message_fields(message, fields: dict[str, Any]) -> None:
        for name, value in fields.items():
            target = getattr(message, name)
            if isinstance(value, (list, tuple)) and hasattr(target, "extend"):
                target.extend(value)
            else:
                setattr(message, name, value)

    def _authenticate_application(self) -> None:
        with self._authentication_lock:
            self._send_request(
                "ProtoOAApplicationAuthReq",
                clientId=self._client_id,
                clientSecret=self._client_secret,
            )
            self._application_authenticated.set()

    def _reload_granted_accounts(self) -> set[int]:
        if not self._application_authenticated.wait(self._timeout):
            raise ConnectionError("cTrader application session is not authorized")
        accounts = self._send_request(
            "ProtoOAGetAccountListByAccessTokenReq",
            accessToken=self._access_token,
        )
        account_ids_by_login: dict[int, set[int]] = {}
        account_environments_by_id: dict[int, str] = {}
        granted_ids: set[int] = set()
        for item in accounts.ctidTraderAccount:
            account_id = int(item.ctidTraderAccountId)
            granted_ids.add(account_id)
            if self._message_has_field(item, "isLive"):
                account_environment = "live" if bool(item.isLive) else "demo"
            else:
                account_environment = self.environment
            account_environments_by_id[account_id] = account_environment
            if not self._message_has_field(item, "traderLogin"):
                continue
            trader_login = int(item.traderLogin)
            account_ids_by_login.setdefault(trader_login, set()).add(account_id)
        self._granted_account_ids = granted_ids
        self._account_environments_by_id = account_environments_by_id
        self._account_ids_by_trader_login = {
            trader_login: tuple(sorted(account_ids))
            for trader_login, account_ids in account_ids_by_login.items()
        }
        return granted_ids

    @staticmethod
    def _message_has_field(message, field_name: str) -> bool:
        has_field = getattr(message, "HasField", None)
        if callable(has_field):
            try:
                return bool(has_field(field_name))
            except (TypeError, ValueError):
                pass
        return getattr(message, field_name, None) is not None

    def resolve_account(self, trader_login: int | str) -> tuple[int, str]:
        """Resolve a UI-visible trader login to its account ID and endpoint."""

        try:
            normalized_login = int(trader_login)
        except (TypeError, ValueError) as exc:
            raise ValueError("cTrader trader_login must be an integer") from exc
        if normalized_login <= 0:
            raise ValueError("cTrader trader_login must be positive")

        with self._authentication_lock:
            account_ids = self._account_ids_by_trader_login.get(
                normalized_login,
                (),
            )
            if not account_ids:
                self._reload_granted_accounts()
                account_ids = self._account_ids_by_trader_login.get(
                    normalized_login,
                    (),
                )
            if not account_ids:
                raise ValueError(
                    f"cTrader trader login {normalized_login} is not granted by the access token"
                )
            if len(account_ids) > 1:
                candidates = ", ".join(str(account_id) for account_id in account_ids)
                raise ValueError(
                    f"cTrader trader login {normalized_login} is ambiguous; "
                    f"account IDs: {candidates}"
                )
            account_id = account_ids[0]
            return account_id, self._account_environments_by_id[account_id]

    def resolve_account_id(self, trader_login: int | str) -> int:
        """Resolve a UI-visible trader login to an Open API account ID."""

        account_id, _ = self.resolve_account(trader_login)
        return account_id

    def _authenticate_account(self, account_id: int) -> None:
        with self._authentication_lock:
            if self._closed or account_id not in self._registered_account_ids:
                return
            if account_id in self._authenticated_account_ids:
                return

            if not self._application_authenticated.wait(self._timeout):
                raise ConnectionError("cTrader application session is not authorized")

            if account_id not in self._granted_account_ids:
                self._reload_granted_accounts()
            if account_id not in self._granted_account_ids:
                raise ValueError(
                    f"cTrader account {account_id} is not granted by the access token"
                )

            self._send_request(
                "ProtoOAAccountAuthReq",
                ctidTraderAccountId=account_id,
                accessToken=self._access_token,
            )
            self._authenticated_account_ids.add(account_id)
            self._logger.info("cTrader account authenticated | account=%s", account_id)

    def authenticate(self, account_id: int | str) -> int:
        """Acquire one venue reference to an authenticated trading account."""

        account_id = int(account_id)
        with self._authentication_lock:
            if self._closed:
                raise RuntimeError("cTrader connection is closed")
            self._account_ref_counts[account_id] = (
                self._account_ref_counts.get(account_id, 0) + 1
            )
            self._registered_account_ids.add(account_id)
            try:
                self._authenticate_account(account_id)
            except Exception:
                remaining = self._account_ref_counts[account_id] - 1
                if remaining > 0:
                    self._account_ref_counts[account_id] = remaining
                else:
                    self._account_ref_counts.pop(account_id, None)
                    self._registered_account_ids.discard(account_id)
                raise
            return account_id

    def release_account(self, account_id: int | str) -> None:
        """Release one venue reference and close the connection when unused."""

        account_id = int(account_id)
        close_connection = False
        remove_account_quotes = False
        with self._authentication_lock:
            references = self._account_ref_counts.get(account_id, 0)
            if references == 0:
                return
            if references > 1:
                self._account_ref_counts[account_id] = references - 1
                return

            self._account_ref_counts.pop(account_id, None)
            self._registered_account_ids.discard(account_id)
            self._authenticated_account_ids.discard(account_id)
            remove_account_quotes = True
            if not self._account_ref_counts and not self._closed:
                self._closed = True
                close_connection = True

        if remove_account_quotes:
            with self._quote_lock:
                quote_keys = [key for key in self._quotes if key[0] == account_id]
                for quote_key in quote_keys:
                    self._quotes.pop(quote_key, None)
                    self._quote_events.pop(quote_key, None)

        if close_connection:
            self._stop_connection()

    def _reauthenticate(self) -> None:
        try:
            self._authenticate_application()
            self._reload_granted_accounts()
            with self._authentication_lock:
                registered_ids = tuple(self._registered_account_ids)
            for account_id in registered_ids:
                self._authenticate_account(account_id)
            self._ever_authenticated = True
        except Exception:
            self._logger.exception("cTrader Open API reauthentication failed")

    def _send_request(
        self,
        message_name: str,
        *,
        client_message_id: str | None = None,
        **fields,
    ):
        if self._closed:
            raise RuntimeError("cTrader connection is closed")
        if not self._connected.wait(self._timeout):
            raise ConnectionError("cTrader Open API is disconnected")

        message_class = getattr(self._messages, message_name)
        message = message_class()
        self._set_message_fields(message, fields)
        completed = threading.Event()
        outcome: dict[str, Any] = {}

        def send() -> None:
            deferred = self._client.send(
                message,
                clientMsgId=client_message_id,
                responseTimeoutInSeconds=self._timeout,
            )

            def succeeded(envelope):
                try:
                    outcome["response"] = self._protobuf.extract(envelope)
                except Exception as exc:
                    outcome["error"] = exc
                completed.set()
                return envelope

            def failed(failure):
                outcome["error"] = failure
                completed.set()
                return None

            deferred.addCallbacks(succeeded, failed)

        self._call_in_reactor(send)
        if not completed.wait(self._timeout + 1.0):
            raise TimeoutError(f"Timed out waiting for {message_name}")
        if "error" in outcome:
            raise RuntimeError(
                f"cTrader request {message_name} failed: {outcome['error']}"
            )
        response = outcome["response"]
        response_name = response.__class__.__name__
        if response_name.endswith("ErrorRes") or response_name.endswith("ErrorEvent"):
            raise RuntimeError(
                f"cTrader request {message_name} failed: "
                f"{getattr(response, 'errorCode', response_name)}: "
                f"{getattr(response, 'description', '')}"
            )
        return response

    def request(
        self,
        message_name: str,
        *,
        client_message_id: str | None = None,
        **fields,
    ):
        if message_name not in self._AUTHENTICATION_MESSAGES:
            if not self._application_authenticated.wait(self._timeout):
                raise ConnectionError("cTrader application session is not authorized")
            account_id = fields.get("ctidTraderAccountId")
            if (
                account_id is not None
                and int(account_id) not in self._authenticated_account_ids
            ):
                raise ConnectionError(
                    f"cTrader account {int(account_id)} is not authenticated on this connection"
                )
        return self._send_request(
            message_name,
            client_message_id=client_message_id,
            **fields,
        )

    def _start_heartbeat_timer(self) -> None:
        if self._closed:
            return
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
        self._heartbeat_timer = threading.Timer(
            self.HEARTBEAT_INTERVAL_SECONDS,
            self._heartbeat_timer_handler,
        )
        self._heartbeat_timer.daemon = True
        self._heartbeat_timer.start()

    def _heartbeat_timer_handler(self) -> None:
        if self._closed:
            return
        try:
            if self._connected.is_set():

                def send() -> None:
                    connected = self._client.whenConnected(failAfterFailures=1)
                    connected.addCallbacks(
                        lambda protocol: protocol.heartbeat(),
                        lambda failure: self._logger.warning(
                            "cTrader heartbeat failed: %s",
                            failure,
                        ),
                    )

                self._call_in_reactor(send)
        except Exception:
            self._logger.exception("Failed to send cTrader heartbeat")
        finally:
            self._start_heartbeat_timer()

    def _start_token_check_timer(self) -> None:
        if self._closed:
            return
        if self._token_check_timer is not None:
            self._token_check_timer.cancel()
        self._token_check_timer = threading.Timer(
            self.TOKEN_CHECK_INTERVAL_SECONDS,
            self._token_check_timer_handler,
        )
        self._token_check_timer.daemon = True
        self._token_check_timer.start()

    def _token_check_timer_handler(self) -> None:
        if self._closed:
            return
        try:
            self._check_access_token(force_if_expiry_unknown=True)
        except Exception:
            self._logger.exception("cTrader access token check failed")
        finally:
            self._start_token_check_timer()

    def subscribe_spots(self, account_id: int, symbol_id: int) -> None:
        account_id = int(account_id)
        symbol_id = int(symbol_id)
        if account_id not in self._authenticated_account_ids:
            raise ConnectionError(
                f"cTrader account {account_id} is not authenticated on this connection"
            )
        quote_key = (account_id, symbol_id)
        with self._quote_lock:
            event = self._quote_events.setdefault(quote_key, threading.Event())
        self.request(
            "ProtoOASubscribeSpotsReq",
            ctidTraderAccountId=account_id,
            symbolId=[symbol_id],
            subscribeToSpotTimestamp=True,
        )
        if not event.wait(self._timeout):
            raise TimeoutError(
                f"Timed out waiting for cTrader quote for account {account_id}, "
                f"symbol {symbol_id}"
            )

    def latest_price(self, account_id: int, symbol_id: int, *, is_buy: bool) -> float:
        bid, ask = self.latest_bid_ask(account_id, symbol_id)
        return ask if is_buy else bid

    def latest_bid_ask(self, account_id: int, symbol_id: int) -> tuple[float, float]:
        quote_key = (int(account_id), int(symbol_id))
        with self._quote_lock:
            event = self._quote_events.setdefault(quote_key, threading.Event())
        if not event.wait(self._timeout):
            raise TimeoutError(
                f"No cTrader quote available for account {quote_key[0]}, "
                f"symbol {quote_key[1]}"
            )
        with self._quote_lock:
            quote = self._quotes.get(quote_key, {})
            bid = quote.get("bid")
            ask = quote.get("ask")
        if bid is None or ask is None:
            raise RuntimeError(
                f"cTrader quote has no complete bid/ask for account {quote_key[0]}, "
                f"symbol {quote_key[1]}"
            )
        return float(bid), float(ask)

    @property
    def granted_account_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._granted_account_ids))

    @property
    def authenticated_account_ids(self) -> tuple[int, ...]:
        with self._authentication_lock:
            return tuple(sorted(self._authenticated_account_ids))

    def _stop_connection(self) -> None:
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None
        if self._token_check_timer is not None:
            self._token_check_timer.cancel()
            self._token_check_timer = None
        client = getattr(self, "_client", None)
        if client is not None:
            self._call_in_reactor(client.stopService)

    def shutdown(self) -> None:
        """Force-close the connection and discard every account reference."""

        with self._authentication_lock:
            if self._closed:
                return
            self._closed = True
            self._account_ref_counts.clear()
            self._registered_account_ids.clear()
            self._authenticated_account_ids.clear()
        self._stop_connection()


class CTraderVenue(VenueBase, AccountDashboard):
    """cTrader venue scoped to one account, symbol, and strategy label."""

    MARKET_ORDER = 1
    LIMIT_ORDER = 2
    BUY = 1
    SELL = 2

    def __init__(
        self,
        key_path: str,
        symbol: str,
        magic: str | None = None,
        *,
        logger: logging.Logger | None = None,
        trader_login: int | str | None = None,
        environment: str = "live",
        timeout: float = 10.0,
        api: Any = None,
        firm: Firm,
    ):
        self.firm = Firm.parse(firm)
        self.logger = logger
        self.label = str(magic)[:100]
        source_symbol = str(symbol).upper()
        mapped_symbol = CTRADER_SYMBOL_MAP.get(source_symbol, str(symbol))
        standard_usd_symbol = (
            source_symbol[:-1] if source_symbol.endswith("USDT") else source_symbol
        )
        self._symbol_candidates = tuple(
            dict.fromkeys((mapped_symbol, standard_usd_symbol))
        )
        self.symbol = mapped_symbol
        if trader_login in (None, ""):
            raise ValueError("trader_login is required for CTraderVenue")
        self.api = api or CTraderOpenApiConnection(
            key_path,
            environment=environment,
            timeout=timeout,
            logger=self.logger,
        )
        self.trader_login = int(trader_login)
        self.account_id = self.api.resolve_account_id(self.trader_login)
        self._account_acquired = False
        self._closed = False
        self._execution_event_lock = threading.RLock()
        self._execution_ids_by_client_message_id: dict[str, str] = {}
        self._execution_ids_by_client_order_id: dict[str, str] = {}
        self._execution_ids_by_order_id: dict[str, str] = {}
        self._execution_roles: dict[str, str] = {}
        self._buffered_execution_events: dict[
            str,
            list[ExecutionEvent],
        ] = {}
        self._active_execution_ids: set[str] = set()
        try:
            self.api.authenticate(self.account_id)
            self._account_acquired = True
            self._load_account_and_symbol()
            register = getattr(self.api, "add_execution_listener", None)
            if callable(register):
                register(self._handle_execution_event)
        except Exception:
            if self._account_acquired:
                self.api.release_account(self.account_id)
                self._account_acquired = False
            elif api is None:
                self.api.shutdown()
            raise

    @staticmethod
    def _normalized_symbol(value: str) -> str:
        return "".join(character for character in value.upper() if character.isalnum())

    def _load_account_and_symbol(self) -> None:
        trader_response = self.api.request(
            "ProtoOATraderReq",
            ctidTraderAccountId=self.account_id,
        )
        trader = trader_response.trader
        response_account_id = int(
            getattr(trader, "ctidTraderAccountId", self.account_id)
        )
        if response_account_id != self.account_id:
            raise RuntimeError(
                "cTrader trader response account ID does not match the resolved "
                f"account: expected {self.account_id}, got {response_account_id}"
            )
        if CTraderOpenApiConnection._message_has_field(trader, "traderLogin"):
            response_trader_login = int(trader.traderLogin)
            if response_trader_login != self.trader_login:
                raise RuntimeError(
                    "cTrader trader response login does not match the configured "
                    f"trader_login: expected {self.trader_login}, "
                    f"got {response_trader_login}"
                )
        self._limited_risk = bool(getattr(trader, "isLimitedRisk", False))

        symbols = self.api.request(
            "ProtoOASymbolsListReq",
            ctidTraderAccountId=self.account_id,
            includeArchivedSymbols=False,
        )
        symbols_by_name = {
            self._normalized_symbol(
                str(getattr(item, "symbolName", getattr(item, "name", "")))
            ): item
            for item in symbols.symbol
        }
        light_symbol = next(
            (
                symbols_by_name[self._normalized_symbol(candidate)]
                for candidate in self._symbol_candidates
                if self._normalized_symbol(candidate) in symbols_by_name
            ),
            None,
        )
        if light_symbol is None:
            raise ValueError(
                "cTrader symbol is not available: "
                f"candidates={self._symbol_candidates}"
            )
        self.symbol = str(
            getattr(
                light_symbol,
                "symbolName",
                getattr(light_symbol, "name", self.symbol),
            )
        )
        self.symbol_id = int(light_symbol.symbolId)

        details = self.api.request(
            "ProtoOASymbolByIdReq",
            ctidTraderAccountId=self.account_id,
            symbolId=[self.symbol_id],
        )
        if not details.symbol:
            raise RuntimeError(f"cTrader returned no details for symbol {self.symbol}")
        symbol_info = details.symbol[0]
        self.volume_min = max(1, int(getattr(symbol_info, "minVolume", 0) or 1))
        self.volume_step = max(1, int(getattr(symbol_info, "stepVolume", 0) or 1))
        self.volume_max = int(getattr(symbol_info, "maxVolume", 0) or 0)
        self.digits = int(symbol_info.digits)
        self._guaranteed_stop_loss = bool(
            getattr(symbol_info, "guaranteedStopLoss", False)
        )
        self.api.subscribe_spots(self.account_id, self.symbol_id)
        self.logger.info(
            "cTrader venue ready | trader_login=%s account=%s symbol=%s "
            "symbol_id=%s label=%s",
            self.trader_login,
            self.account_id,
            self.symbol,
            self.symbol_id,
            self.label,
        )

    @staticmethod
    def _money(value: int, digits: int) -> float:
        return float(value) / (10 ** int(digits))

    def get_account_equity(self) -> float:
        trader_response = self.api.request(
            "ProtoOATraderReq",
            ctidTraderAccountId=self.account_id,
        )
        trader = trader_response.trader
        balance = self._money(trader.balance, getattr(trader, "moneyDigits", 0))
        pnl_response = self.api.request(
            "ProtoOAGetPositionUnrealizedPnLReq",
            ctidTraderAccountId=self.account_id,
        )
        pnl = sum(
            self._money(item.netUnrealizedPnL, pnl_response.moneyDigits)
            for item in pnl_response.positionUnrealizedPnL
        )
        equity = balance + pnl
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError("cTrader returned invalid account equity")
        return equity

    def get_dashboard_balance(self) -> AccountBalance:
        trader_response = self.api.request(
            "ProtoOATraderReq",
            ctidTraderAccountId=self.account_id,
        )
        trader = trader_response.trader
        balance = self._money(
            trader.balance,
            getattr(trader, "moneyDigits", 0),
        )
        pnl_response = self.api.request(
            "ProtoOAGetPositionUnrealizedPnLReq",
            ctidTraderAccountId=self.account_id,
        )
        unrealized_pnl = sum(
            self._money(item.netUnrealizedPnL, pnl_response.moneyDigits)
            for item in pnl_response.positionUnrealizedPnL
        )
        equity = balance + unrealized_pnl
        if not all(math.isfinite(value) for value in (balance, equity)):
            raise RuntimeError("cTrader returned invalid dashboard balance data")
        return AccountBalance(
            balance=balance,
            equity=equity,
            used_margin=self._dashboard_used_margin(trader),
        )

    def _dashboard_used_margin(self, trader) -> float | None:
        """Aggregate account margin per symbol using the broker's hedge rule."""
        response = self.api.request(
            "ProtoOAReconcileReq",
            ctidTraderAccountId=self.account_id,
        )
        margins_by_symbol: dict[int, dict[int, float]] = {}
        for position in response.position:
            if not CTraderOpenApiConnection._message_has_field(position, "usedMargin"):
                return None
            margin = self._money(position.usedMargin, position.moneyDigits)
            if not math.isfinite(margin) or margin < 0:
                return None
            sides = margins_by_symbol.setdefault(
                int(position.tradeData.symbolId), {self.BUY: 0.0, self.SELL: 0.0}
            )
            sides[int(position.tradeData.tradeSide)] += margin

        total = 0.0
        for sides in margins_by_symbol.values():
            long_margin, short_margin = sides[self.BUY], sides[self.SELL]
            if trader.totalMarginCalculationType == 0:  # MAX
                total += max(long_margin, short_margin)
            elif trader.totalMarginCalculationType == 1:  # SUM
                total += long_margin + short_margin
            elif trader.totalMarginCalculationType == 2:  # NET
                total += abs(long_margin - short_margin)
            else:
                return None
        return total

    def _aggregate_protection_price(
        self,
        positions: list[Any],
        field_name: str,
    ) -> float | None:
        weighted_price = 0.0
        total_volume = 0
        for position in positions:
            volume = int(position.tradeData.volume)
            price = float(getattr(position, field_name, 0.0) or 0.0)
            if volume <= 0 or not math.isfinite(price) or price <= 0:
                return None
            weighted_price += volume * price
            total_volume += volume
        if total_volume <= 0:
            return None
        return round(weighted_price / total_volume, self.digits)

    def get_dashboard_position(self) -> AccountPosition | None:
        positions = self._positions()
        if not positions:
            return None
        sides = {int(position.tradeData.tradeSide) for position in positions}
        if len(sides) != 1:
            raise RuntimeError(
                "cTrader strategy label has simultaneous long and short positions"
            )
        side_value = sides.pop()
        total_volume = sum(int(position.tradeData.volume) for position in positions)
        if total_volume <= 0:
            return None
        quantity = total_volume / 100.0
        entry_price = (
            sum(
                int(position.tradeData.volume) * float(position.price)
                for position in positions
            )
            / total_volume
        )
        mark_price = self.api.latest_price(
            self.account_id,
            self.symbol_id,
            is_buy=side_value == self.SELL,
        )
        pnl_response = self.api.request(
            "ProtoOAGetPositionUnrealizedPnLReq",
            ctidTraderAccountId=self.account_id,
        )
        position_ids = {int(position.positionId) for position in positions}
        unrealized_pnl = sum(
            self._money(item.netUnrealizedPnL, pnl_response.moneyDigits)
            for item in pnl_response.positionUnrealizedPnL
            if int(getattr(item, "positionId", -1)) in position_ids
        )
        stop_loss_price = self._aggregate_protection_price(
            positions,
            "stopLoss",
        )
        take_profit_price = self._aggregate_protection_price(
            positions,
            "takeProfit",
        )
        components = tuple(
            AccountPositionComponent(
                quantity=int(position.tradeData.volume) / 100.0,
                entry_price=float(position.price),
                stop_loss_price=(
                    round(
                        float(getattr(position, "stopLoss", 0.0) or 0.0),
                        self.digits,
                    )
                    or None
                ),
                take_profit_price=(
                    round(
                        float(getattr(position, "takeProfit", 0.0) or 0.0),
                        self.digits,
                    )
                    or None
                ),
            )
            for position in positions
        )
        cost = quantity * entry_price
        return AccountPosition(
            symbol=self.symbol,
            side=(PositionSide.LONG if side_value == self.BUY else PositionSide.SHORT),
            quantity=quantity,
            entry_price=entry_price,
            mark_price=mark_price,
            notional=quantity * mark_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl / cost if cost > 0 else None,
            leverage=None,
            liquidation_price=None,
            margin_mode=MarginMode.UNKNOWN,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            components=components,
        )

    def _positions(self) -> list[Any]:
        response = self.api.request(
            "ProtoOAReconcileReq",
            ctidTraderAccountId=self.account_id,
        )
        return [
            position
            for position in response.position
            if int(position.tradeData.symbolId) == self.symbol_id
            and str(getattr(position.tradeData, "label", "")) == self.label
        ]

    def get_current_state(self) -> PositionView:
        positions = self._positions()
        if not positions:
            return PositionView()
        sides = {int(position.tradeData.tradeSide) for position in positions}
        if len(sides) != 1:
            raise RuntimeError(
                "cTrader strategy label has simultaneous long and short positions"
            )
        total_volume = sum(int(position.tradeData.volume) for position in positions)
        if total_volume <= 0:
            return PositionView()
        average_price = (
            sum(
                int(position.tradeData.volume) * float(position.price)
                for position in positions
            )
            / total_volume
        )
        direction = (
            PositionDir.POSITIVE if sides == {self.BUY} else PositionDir.NEGATIVE
        )
        return PositionView(
            dir=direction,
            size=total_volume / 100.0,
            price=average_price,
        )

    def get_last_position_open_time(self):
        timestamps = [
            int(getattr(position.tradeData, "openTimestamp", 0) or 0)
            for position in self._positions()
        ]
        timestamps = [timestamp for timestamp in timestamps if timestamp > 0]
        if not timestamps:
            return None
        return datetime.fromtimestamp(max(timestamps) / 1000.0, tz=timezone.utc)

    def get_daily_reset_date(self, candle_open_time_utc: datetime):
        daily_reset_date = self.get_firm_daily_reset_date(
            candle_open_time_utc,
            self.firm,
        )
        self.logger.info(
            "cTrader daily reset date | trader_login=%s firm=%s "
            "candle_open_time_utc=%s daily_reset_date=%s",
            self.trader_login,
            self.firm.value,
            candle_open_time_utc.isoformat(),
            daily_reset_date.isoformat(),
        )
        return daily_reset_date

    def _normalize_volume(self, size: float) -> int:
        requested = int(
            (Decimal(str(size)) * 100).to_integral_value(rounding=ROUND_DOWN)
        )
        volume = requested // self.volume_step * self.volume_step
        if volume < self.volume_min:
            raise ValueError(
                f"cTrader order volume {volume / 100:g} is below minimum "
                f"{self.volume_min / 100:g}"
            )
        return volume

    def _volume_batches(self, total_volume: int) -> list[int]:
        maximum = self.volume_max or total_volume
        maximum = maximum // self.volume_step * self.volume_step
        if maximum < self.volume_min:
            raise RuntimeError("cTrader symbol has invalid maximum volume metadata")
        batches = []
        remaining = total_volume
        while remaining > maximum:
            batch = maximum
            remainder = remaining - batch
            if remainder < self.volume_min:
                adjustment = (
                    (self.volume_min - remainder + self.volume_step - 1)
                    // self.volume_step
                    * self.volume_step
                )
                batch -= adjustment
            if batch < self.volume_min:
                raise ValueError("cTrader volume cannot be split into valid orders")
            batches.append(batch)
            remaining -= batch
        if remaining < self.volume_min:
            raise ValueError(
                "cTrader order remainder is below the symbol minimum volume"
            )
        batches.append(remaining)
        return batches

    @staticmethod
    def _order_percentage(value: Any, name: str) -> float | None:
        if value is None:
            return None
        percentage = float(value)
        if not math.isfinite(percentage) or percentage <= 0:
            raise ValueError(f"{name} must be a positive finite ratio")
        return percentage

    def get_bid_ask(self) -> tuple[float, float]:
        return self.api.latest_bid_ask(self.account_id, self.symbol_id)

    def normalize_order_quantity(self, size: float) -> float:
        return self._normalize_volume(float(size)) / 100.0

    @staticmethod
    def _normalized_order_status(raw_status: int) -> str:
        return {
            OpenApiModelMessages_pb2.ORDER_STATUS_ACCEPTED: "accepted",
            OpenApiModelMessages_pb2.ORDER_STATUS_FILLED: "filled",
            OpenApiModelMessages_pb2.ORDER_STATUS_REJECTED: "rejected",
            OpenApiModelMessages_pb2.ORDER_STATUS_EXPIRED: "expired",
            OpenApiModelMessages_pb2.ORDER_STATUS_CANCELLED: "cancelled",
        }.get(int(raw_status), "unknown")

    @staticmethod
    def _normalized_execution_status(execution_type: int) -> str:
        return {
            OpenApiModelMessages_pb2.ORDER_ACCEPTED: "accepted",
            OpenApiModelMessages_pb2.ORDER_FILLED: "filled",
            OpenApiModelMessages_pb2.ORDER_PARTIAL_FILL: "partially_filled",
            OpenApiModelMessages_pb2.ORDER_REPLACED: "replaced",
            OpenApiModelMessages_pb2.ORDER_CANCELLED: "cancelled",
            OpenApiModelMessages_pb2.ORDER_EXPIRED: "expired",
            OpenApiModelMessages_pb2.ORDER_REJECTED: "rejected",
            OpenApiModelMessages_pb2.ORDER_CANCEL_REJECTED: "cancel_rejected",
        }.get(int(execution_type), "unknown")

    @staticmethod
    def _event_trade_data(message):
        order = getattr(message, "order", None)
        trade_data = getattr(order, "tradeData", None)
        if trade_data is not None:
            return trade_data
        position = getattr(message, "position", None)
        return getattr(position, "tradeData", None)

    def _handle_execution_event(
        self,
        message,
        client_message_id: str,
    ) -> None:
        if int(getattr(message, "ctidTraderAccountId", 0) or 0) != self.account_id:
            return
        order = getattr(message, "order", None)
        deal = getattr(message, "deal", None)
        trade_data = self._event_trade_data(message)
        event_symbol_id = int(
            getattr(deal, "symbolId", 0) or getattr(trade_data, "symbolId", 0) or 0
        )
        if event_symbol_id and event_symbol_id != self.symbol_id:
            return
        client_order_id = str(getattr(order, "clientOrderId", "") or "")
        order_id = str(
            getattr(order, "orderId", "") or getattr(deal, "orderId", "") or ""
        )
        with self._execution_event_lock:
            execution_id = (
                self._execution_ids_by_client_order_id.get(client_order_id)
                or self._execution_ids_by_client_message_id.get(client_message_id)
                or self._execution_ids_by_order_id.get(order_id)
            )
        label = str(getattr(trade_data, "label", "") or "")
        if execution_id is None and label != self.label:
            return
        deal_id = str(getattr(deal, "dealId", "") or "")
        if execution_id is None:
            stable_id = order_id or deal_id
            if not stable_id:
                return
            execution_id = f"ctrader-{self.account_id}-{stable_id}"
        if order_id:
            with self._execution_event_lock:
                self._execution_ids_by_order_id[order_id] = execution_id

        is_closing = bool(getattr(order, "closingOrder", False))
        order_type = int(getattr(order, "orderType", 0) or 0)
        with self._execution_event_lock:
            order_role = self._execution_roles.get(execution_id)
        if order_role is None:
            order_role = (
                "exit"
                if is_closing
                or order_type == OpenApiModelMessages_pb2.STOP_LOSS_TAKE_PROFIT
                else "entry"
            )
        trade_side = int(
            getattr(deal, "tradeSide", 0) or getattr(trade_data, "tradeSide", 0) or 0
        )
        side = (
            "buy"
            if trade_side == self.BUY
            else "sell" if trade_side == self.SELL else ""
        )
        raw_volume = int(
            getattr(trade_data, "volume", 0) or getattr(deal, "volume", 0) or 0
        )
        submitted_quantity = raw_volume / 100.0
        timestamp_ms = int(
            getattr(deal, "executionTimestamp", 0)
            or getattr(order, "utcLastUpdateTimestamp", 0)
            or getattr(deal, "utcLastUpdateTimestamp", 0)
            or 0
        )
        event_at_utc = (
            datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            if timestamp_ms > 0
            else datetime.now(timezone.utc)
        )
        price = float(getattr(deal, "executionPrice", 0.0) or 0.0)
        filled_volume = int(
            getattr(deal, "filledVolume", 0) or getattr(deal, "volume", 0) or 0
        )
        fill = None
        if deal_id and price > 0 and filled_volume > 0:
            fill = ExecutionFill(
                price=price,
                quantity=filled_volume / 100.0,
                order_id=order_id,
                deal_id=deal_id,
                client_order_id=client_order_id,
                executed_at_utc=event_at_utc,
            )
        status = self._normalized_execution_status(
            int(getattr(message, "executionType", 0) or 0)
        )
        reason = str(getattr(message, "errorCode", "") or "")
        if not reason and order_role == "exit" and not client_order_id:
            reason = "broker_protection"
        event = ExecutionEvent(
            event_id=(
                f"ctrader:{self.account_id}:{self.symbol_id}:"
                f"{order_id}:{deal_id or status}:{timestamp_ms}"
            ),
            execution_id=execution_id,
            status=status,
            event_at_utc=event_at_utc,
            order_role=order_role,
            side=side,
            order_id=order_id,
            client_order_id=client_order_id,
            submitted_quantity=submitted_quantity,
            reason=reason,
            fill=fill,
        )
        with self._execution_event_lock:
            if (
                execution_id in self._execution_roles
                and execution_id not in self._active_execution_ids
            ):
                self._buffered_execution_events.setdefault(
                    execution_id,
                    [],
                ).append(event)
                return
        self._emit_execution_event(event)

    def activate_execution_updates(self, execution_id: str) -> None:
        with self._execution_event_lock:
            self._active_execution_ids.add(execution_id)
            buffered = self._buffered_execution_events.pop(
                execution_id,
                [],
            )
        for event in buffered:
            self._emit_execution_event(event)

    def reconcile_execution_events(self, since_utc: datetime) -> int:
        if since_utc.tzinfo is None:
            raise ValueError("since_utc must be timezone-aware")
        from_timestamp = int(since_utc.astimezone(timezone.utc).timestamp() * 1000)
        to_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        order_response = self.api.request(
            "ProtoOAOrderListReq",
            ctidTraderAccountId=self.account_id,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
        )
        owned_orders = {}
        for order in getattr(order_response, "order", ()):
            trade_data = getattr(order, "tradeData", None)
            if (
                int(getattr(trade_data, "symbolId", 0) or 0) == self.symbol_id
                and str(getattr(trade_data, "label", "") or "") == self.label
            ):
                owned_orders[int(order.orderId)] = order

        deal_response = self.api.request(
            "ProtoOADealListReq",
            ctidTraderAccountId=self.account_id,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
            maxRows=1000,
        )
        emitted = 0
        order_ids_with_deals = set()
        for deal in getattr(deal_response, "deal", ()):
            order_id = int(getattr(deal, "orderId", 0) or 0)
            order = owned_orders.get(order_id)
            if order is None:
                continue
            order_ids_with_deals.add(order_id)
            deal_status = int(getattr(deal, "dealStatus", 0) or 0)
            execution_type = (
                OpenApiModelMessages_pb2.ORDER_FILLED
                if deal_status == OpenApiModelMessages_pb2.FILLED
                else (
                    OpenApiModelMessages_pb2.ORDER_PARTIAL_FILL
                    if deal_status == OpenApiModelMessages_pb2.PARTIALLY_FILLED
                    else OpenApiModelMessages_pb2.ORDER_REJECTED
                )
            )
            self._handle_execution_event(
                SimpleNamespace(
                    ctidTraderAccountId=self.account_id,
                    executionType=execution_type,
                    errorCode=(
                        ""
                        if execution_type != OpenApiModelMessages_pb2.ORDER_REJECTED
                        else "deal_rejected"
                    ),
                    order=order,
                    deal=deal,
                ),
                "",
            )
            emitted += 1

        order_status_to_execution_type = {
            OpenApiModelMessages_pb2.ORDER_STATUS_ACCEPTED: (
                OpenApiModelMessages_pb2.ORDER_ACCEPTED
            ),
            OpenApiModelMessages_pb2.ORDER_STATUS_FILLED: (
                OpenApiModelMessages_pb2.ORDER_FILLED
            ),
            OpenApiModelMessages_pb2.ORDER_STATUS_REJECTED: (
                OpenApiModelMessages_pb2.ORDER_REJECTED
            ),
            OpenApiModelMessages_pb2.ORDER_STATUS_EXPIRED: (
                OpenApiModelMessages_pb2.ORDER_EXPIRED
            ),
            OpenApiModelMessages_pb2.ORDER_STATUS_CANCELLED: (
                OpenApiModelMessages_pb2.ORDER_CANCELLED
            ),
        }
        for order_id, order in owned_orders.items():
            if order_id in order_ids_with_deals:
                continue
            execution_type = order_status_to_execution_type.get(
                int(getattr(order, "orderStatus", 0) or 0)
            )
            if execution_type is None:
                continue
            self._handle_execution_event(
                SimpleNamespace(
                    ctidTraderAccountId=self.account_id,
                    executionType=execution_type,
                    errorCode="",
                    order=order,
                    deal=None,
                ),
                "",
            )
            emitted += 1
        if bool(getattr(deal_response, "hasMore", False)):
            self.logger.warning(
                "cTrader execution reconciliation reached the 1000-deal limit | "
                "account=%s symbol=%s",
                self.account_id,
                self.symbol,
            )
        return emitted

    def _execution_fills(self, result, *, is_buy: bool) -> tuple[ExecutionFill, ...]:
        responses = result if isinstance(result, list) else [result]
        fills = []
        for response in responses:
            deal = getattr(response, "deal", None)
            if deal is None:
                continue
            price = float(getattr(deal, "executionPrice", 0.0) or 0.0)
            raw_volume = int(
                getattr(deal, "filledVolume", 0) or getattr(deal, "volume", 0) or 0
            )
            if price <= 0 or raw_volume <= 0:
                continue
            timestamp_ms = int(getattr(deal, "executionTimestamp", 0) or 0)
            order = getattr(response, "order", None)
            fills.append(
                ExecutionFill(
                    price=price,
                    quantity=raw_volume / 100.0,
                    order_id=str(getattr(deal, "orderId", "") or ""),
                    deal_id=str(getattr(deal, "dealId", "") or ""),
                    client_order_id=str(getattr(order, "clientOrderId", "") or ""),
                    executed_at_utc=(
                        datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
                        if timestamp_ms > 0
                        else None
                    ),
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
        orders = []
        for response in responses:
            order = getattr(response, "order", None)
            if order is None:
                continue
            trade_data = getattr(order, "tradeData", None)
            raw_volume = int(getattr(trade_data, "volume", 0) or 0)
            orders.append(
                ExecutionOrder(
                    order_id=str(getattr(order, "orderId", "") or ""),
                    client_order_id=str(getattr(order, "clientOrderId", "") or ""),
                    submitted_quantity=(
                        raw_volume / 100.0
                        if raw_volume > 0
                        else submitted_quantity / max(1, len(responses))
                    ),
                    status=self._normalized_order_status(
                        int(getattr(order, "orderStatus", 0) or 0)
                    ),
                )
            )
        return tuple(orders)

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
        order_type, price = self.normalize_order_request(order_type, price)
        stop_loss_pct = self._order_percentage(stop_loss_pct, "stop_loss_pct")
        take_profit_pct = self._order_percentage(
            take_profit_pct,
            "take_profit_pct",
        )
        if self._limited_risk and stop_loss_pct is None:
            raise ValueError("cTrader limited-risk accounts require a stop loss")

        total_volume = self._normalize_volume(float(size))
        batches = self._volume_batches(total_volume)
        reference_price = (
            self.api.latest_price(
                self.account_id,
                self.symbol_id,
                is_buy=bool(is_buy),
            )
            if price is None
            else round(price, self.digits)
        )
        relative_stop = (
            max(1, int(round(reference_price * stop_loss_pct * 100_000)))
            if stop_loss_pct is not None
            else None
        )
        relative_take_profit = (
            max(1, int(round(reference_price * take_profit_pct * 100_000)))
            if take_profit_pct is not None
            else None
        )
        responses = []
        for index, volume in enumerate(batches):
            client_message_id = f"{execution_id}-{index + 1}" if execution_id else ""
            if execution_id:
                with self._execution_event_lock:
                    self._execution_ids_by_client_message_id[client_message_id] = str(
                        execution_id
                    )
                    self._execution_ids_by_client_order_id[client_message_id] = str(
                        execution_id
                    )
                    self._execution_roles[str(execution_id)] = "entry"
            fields = {
                "ctidTraderAccountId": self.account_id,
                "symbolId": self.symbol_id,
                "orderType": (
                    self.MARKET_ORDER
                    if order_type == OrderType.MARKET
                    else self.LIMIT_ORDER
                ),
                "tradeSide": self.BUY if is_buy else self.SELL,
                "volume": volume,
                "label": self.label,
                "comment": f"financial-ml-system batch {index + 1}/{len(batches)}",
            }
            if client_message_id:
                fields["clientOrderId"] = client_message_id
            if order_type == OrderType.LIMIT:
                fields["limitPrice"] = reference_price
            if relative_stop is not None:
                fields["relativeStopLoss"] = relative_stop
            if relative_take_profit is not None:
                fields["relativeTakeProfit"] = relative_take_profit
            if self._limited_risk:
                if not self._guaranteed_stop_loss:
                    raise RuntimeError(
                        "cTrader limited-risk account requires a symbol with guaranteed stops"
                    )
                fields["guaranteedStopLoss"] = True
            if client_message_id:
                response = self.api.request(
                    "ProtoOANewOrderReq",
                    client_message_id=client_message_id,
                    **fields,
                )
            else:
                response = self.api.request("ProtoOANewOrderReq", **fields)
            responses.append(response)
            if index + 1 < len(batches):
                time.sleep(max(0.0, float(interval_ms) / 1000.0))
        return responses

    def close_position(self, size=None, execution_id=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported cTrader close arguments: {sorted(kwargs)}")
        positions = self._positions()
        if not positions:
            return []
        remaining = None if size is None else self._normalize_volume(abs(float(size)))
        responses = []
        for index, position in enumerate(positions):
            available = int(position.tradeData.volume)
            volume = available if remaining is None else min(available, remaining)
            if volume <= 0:
                break
            client_message_id = f"{execution_id}-{index + 1}" if execution_id else ""
            if execution_id:
                with self._execution_event_lock:
                    self._execution_ids_by_client_message_id[client_message_id] = str(
                        execution_id
                    )
                    self._execution_roles[str(execution_id)] = "exit"
            request_fields = {
                "ctidTraderAccountId": self.account_id,
                "positionId": int(position.positionId),
                "volume": volume,
            }
            if client_message_id:
                response = self.api.request(
                    "ProtoOAClosePositionReq",
                    client_message_id=client_message_id,
                    **request_fields,
                )
            else:
                response = self.api.request(
                    "ProtoOAClosePositionReq",
                    **request_fields,
                )
            responses.append(response)
            if remaining is not None:
                remaining -= volume
        return responses

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        remove_listener = getattr(
            self.api,
            "remove_execution_listener",
            None,
        )
        if callable(remove_listener):
            remove_listener(self._handle_execution_event)
        if self._account_acquired:
            self._account_acquired = False
            self.api.release_account(self.account_id)
