"""Contract tests for Bitget wire formats, position safety and execution events."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlsplit

import pytest
import requests

from trade.core.protocol import ActionType, OrderType, PositionDir, TradeIntent
from trade.venue.live.bitget.bitget_venue import BitgetAPIError, BitgetVenue


class Response:
    def __init__(self, data=None, *, code="00000", message="success", status=200):
        self.status_code = status
        self.body = {"code": code, "msg": message, "data": data}

    def json(self):
        return self.body


class Session:
    """A stateful exchange simulator; all mutations are driven by HTTP payloads."""

    def __init__(self, *, hedge=False):
        self.hedge = hedge
        self.calls = []
        self.orders = {}
        self.plans = []
        self.plan_history = []
        self.positions = []
        self.closed = False
        self.fail_protection = None
        self.fail_cancel = False
        self.reject_entry = False
        self.timeout_after_entry = False
        self.partial_entry = False
        self.fill_ratio = 1.0
        self.contract = {
            "symbol": "BTCUSDT",
            "symbolStatus": "normal",
            "symbolType": "perpetual",
            "supportMarginCoins": ["USDT"],
            "sizeMultiplier": "0.001",
            "minTradeNum": "0.001",
            "priceEndStep": "5",
            "pricePlace": "1",
            "minTradeUSDT": "5",
            "maxMarketOrderQty": "100",
            "maxOrderQty": "200",
        }

    def position(self, size=0.01, side="long"):
        return {
            "symbol": "BTCUSDT",
            "total": str(size),
            "holdSide": side,
            "openPriceAvg": "60100",
            "markPrice": "60200",
            "unrealizedPL": "1",
            "leverage": "10",
            "liquidationPrice": "0",
            "marginMode": "crossed",
            "cTime": "1700000000000",
            "uTime": "1700000000001",
        }

    def request(
        self,
        method,
        url,
        *,
        data=None,
        headers=None,
        timeout=None,
        allow_redirects=None,
    ):
        parsed = urlsplit(url)
        path = parsed.path
        params = dict(parse_qsl(parsed.query)) if method == "GET" else json.loads(data)
        self.calls.append((method, path, params, headers, url, data))
        assert allow_redirects is False
        if path.endswith("/market/contracts"):
            return Response([self.contract])
        if path.endswith("/account/account"):
            return Response(
                {
                    "accountEquity": "1001",
                    "unrealizedPL": "1",
                    "crossedMargin": "200",
                    "isolatedMargin": "50",
                    "marginMode": "crossed",
                    "posMode": "hedge_mode" if self.hedge else "one_way_mode",
                }
            )
        if path.endswith("/position/single-position"):
            return Response(list(self.positions))
        if path.endswith("/market/ticker"):
            return Response(
                [
                    {
                        "symbol": "BTCUSDT",
                        "lastPr": "60000",
                        "bidPr": "60000",
                        "askPr": "60001",
                    }
                ]
            )
        if path.endswith("/orders-plan-pending"):
            return Response(
                {
                    "entrustedList": (
                        list(self.plans) if params["planType"] == "profit_loss" else []
                    ),
                    "endId": "",
                }
            )
        if path.endswith("/orders-plan-history"):
            return Response({"entrustedList": self.plan_history, "endId": ""})
        if path.endswith("/orders-pending"):
            return Response(
                {
                    "entrustedList": [
                        order
                        for order in self.orders.values()
                        if order["state"] == "live"
                    ],
                    "endId": "",
                }
            )
        if path.endswith("/place-order"):
            closing = (
                params.get("reduceOnly") == "YES" or params.get("tradeSide") == "close"
            )
            if self.reject_entry and not closing:
                return Response(code="40762", message="Insufficient balance")
            order_id = str(len(self.orders) + 1)
            quantity = float(params["size"])
            filled = (
                quantity * self.fill_ratio if params["orderType"] == "market" else 0.0
            )
            state = "filled" if filled == quantity else "canceled" if filled else "live"
            if closing:
                position = self.positions[0]
                expected_side = (
                    ("buy" if position["holdSide"] == "long" else "sell")
                    if self.hedge
                    else ("sell" if position["holdSide"] == "long" else "buy")
                )
                assert params["side"] == expected_side
                remaining = float(position["total"]) - filled
                self.positions = (
                    [{**position, "total": str(remaining)}] if remaining > 1e-12 else []
                )
            elif filled:
                self.positions = [
                    self.position(
                        filled, "long" if params["side"] == "buy" else "short"
                    )
                ]
            if self.partial_entry and not closing:
                state = "partially_filled"
            order = {
                **params,
                "orderId": order_id,
                "state": state,
                "priceAvg": "60100",
                "baseVolume": str(filled),
                "uTime": "1700000000001",
                "posSide": "long" if params["side"] == "buy" else "short",
            }
            self.orders[order_id] = order
            if self.timeout_after_entry and not closing:
                raise requests.Timeout("Read timeout")
            return Response({"orderId": order_id, "clientOid": params["clientOid"]})
        if path.endswith("/order/detail"):
            order = next(
                (
                    order
                    for order in self.orders.values()
                    if order["orderId"] == params.get("orderId")
                    or order["clientOid"] == params.get("clientOid")
                ),
                None,
            )
            return (
                Response(dict(order))
                if order
                else Response(code="43001", message="Order does not exist")
            )
        if path.endswith("/order/fills"):
            trades = [
                {
                    "tradeId": order_id,
                    "orderId": order_id,
                    "price": "60100",
                    "baseVolume": order["baseVolume"],
                    "cTime": "1700000000001",
                }
                for order_id, order in self.orders.items()
                if float(order["baseVolume"]) > 0
                and (not params.get("orderId") or params["orderId"] == order_id)
            ]
            return Response({"fillList": trades, "endId": ""})
        if path.endswith("/place-tpsl-order"):
            if params["planType"] == self.fail_protection:
                return Response(code="43023", message="Insufficient position")
            order_id = str(100 + len(self.plans))
            self.plans.append({**params, "orderId": order_id})
            return Response({"orderId": order_id, "clientOid": params["clientOid"]})
        if path.endswith("/cancel-plan-order"):
            identifiers = params["orderIdList"]
            if self.fail_cancel:
                return Response({"successList": [], "failureList": identifiers})
            self.plans = [
                order
                for order in self.plans
                if {"orderId": order["orderId"]} not in identifiers
            ]
            return Response({"successList": identifiers, "failureList": []})
        if path.endswith("/cancel-order"):
            for order in self.orders.values():
                if order["clientOid"] == params.get("clientOid") or order[
                    "orderId"
                ] == params.get("orderId"):
                    order["state"] = "canceled"
            return Response({"orderId": params.get("orderId")})
        raise AssertionError(f"Unexpected request: {method} {path}")

    def close(self):
        self.closed = True


@pytest.fixture
def credentials(tmp_path):
    for name, value in (
        ("apikey", "test-key"),
        ("secret_key", "test-secret"),
        ("passphrase", "test-passphrase"),
    ):
        (tmp_path / name).write_text(value, encoding="utf-8")
    return tmp_path


@pytest.fixture
def venue_factory(credentials, monkeypatch):
    monkeypatch.setattr(BitgetVenue, "REQUEST_INTERVAL_SECONDS", 0)
    venues = []

    def create(session=None, **kwargs):
        venue = BitgetVenue(
            str(credentials),
            "BTCUSDT",
            "strategy:abc",
            session=session or Session(),
            enable_user_stream=False,
            **kwargs,
        )
        venues.append(venue)
        return venue

    yield create
    for venue in venues:
        venue.shutdown()


def test_missing_passphrase_fails_before_network(credentials):
    (credentials / "passphrase").unlink()
    session = Session()
    with pytest.raises(FileNotFoundError, match="passphrase"):
        BitgetVenue(str(credentials), "BTCUSDT", session=session)
    assert session.closed and not session.calls


def test_signatures_cover_exact_query_and_json_body(venue_factory):
    venue = venue_factory()
    venue.submit_order(0.01, True)
    for method, path, params, headers, url, body in venue.session.calls:
        if "/market/" in path:
            assert "ACCESS-KEY" not in headers
            continue
        parsed = urlsplit(url)
        content = headers["ACCESS-TIMESTAMP"] + method + parsed.path
        if parsed.query:
            content += "?" + parsed.query
        content += body or ""
        expected = base64.b64encode(
            hmac.new(b"test-secret", content.encode(), hashlib.sha256).digest()
        ).decode()
        assert headers["ACCESS-SIGN"] == expected
        assert headers["ACCESS-PASSPHRASE"] == "test-passphrase"


def test_filters_account_and_dashboard(venue_factory):
    venue = venue_factory()
    assert venue.price_tick == Decimal("0.5")
    assert venue.normalize_order_quantity(0.0109) == 0.01
    assert venue.get_account_equity() == 1001
    assert venue.get_dashboard_balance().balance == 1000
    assert venue.get_dashboard_balance().used_margin == 250
    assert venue.get_bid_ask() == (60000, 60001)
    assert venue.get_last_position_open_time() is None
    venue.session.positions = [venue.session.position()]
    assert venue.get_current_state().dir == PositionDir.POSITIVE
    dashboard = venue.get_dashboard_position()
    assert dashboard.quantity == 0.01
    assert dashboard.liquidation_price is None
    assert dashboard.opened_at == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert venue.get_last_position_open_time() == dashboard.opened_at


@pytest.mark.parametrize("quantity", [0, -1, float("inf"), float("nan"), 0.0009])
def test_invalid_quantities(venue_factory, quantity):
    venue = venue_factory()
    with pytest.raises(ValueError):
        venue.submit_order(quantity, True)
    assert not any(call[0] == "POST" for call in venue.session.calls)


@pytest.mark.parametrize("hedge", [False, True])
@pytest.mark.parametrize("is_buy", [False, True])
def test_entry_protection_and_close_in_both_modes(venue_factory, hedge, is_buy):
    session = Session(hedge=hedge)
    venue = venue_factory(session)
    result = venue.submit_order(0.0109, is_buy, 0.01, 0.02, execution_id="entry")
    assert result["baseVolume"] == "0.01"
    assert session.plans[0]["planType"] == "pos_loss"
    assert session.plans[0]["executePrice"] == "0"
    assert "size" not in session.plans[0]
    assert session.plans[1]["planType"] == "pos_profit"
    assert "size" not in session.plans[1]
    assert session.plans[1]["executePrice"] == session.plans[1]["triggerPrice"]
    assert session.plans[0]["triggerPrice"] == str(59499 if is_buy else 60701)
    expected_hold = (
        ("long" if is_buy else "short") if hedge else ("buy" if is_buy else "sell")
    )
    assert all(order["holdSide"] == expected_hold for order in session.plans)
    venue.close_position()
    assert session.positions == []
    assert session.plans == []
    assert session.orders["2"].get("tradeSide") == ("close" if hedge else None)
    assert session.orders["2"].get("reduceOnly") == (None if hedge else "YES")


def test_protection_failure_closes_new_position(venue_factory):
    venue = venue_factory()
    venue.session.fail_protection = "pos_profit"
    with pytest.raises(BitgetAPIError, match="43023"):
        venue.submit_order(0.01, True, 0.01, 0.02)
    assert venue.session.positions == []
    assert venue.session.plans == []
    assert len(venue.session.orders) == 2


def test_ambiguous_entry_is_not_retried_and_its_fill_is_closed(venue_factory):
    venue = venue_factory()
    venue.session.timeout_after_entry = True
    with pytest.raises(RuntimeError, match="transport failed"):
        venue.submit_order(0.01, True, 0.01)
    assert len(venue.session.orders) == 2
    assert venue.session.positions == []


def test_partial_entry_timeout_cancels_remainder_before_flattening(
    venue_factory, monkeypatch
):
    venue = venue_factory()
    venue.session.partial_entry = True
    monkeypatch.setattr(venue, "ORDER_WAIT_SECONDS", 0)
    with pytest.raises(RuntimeError, match="terminal state"):
        venue.submit_order(0.01, True, 0.01)
    paths = [call[1] for call in venue.session.calls if call[0] == "POST"]
    assert paths == [
        "/api/v2/mix/order/place-order",
        "/api/v2/mix/order/cancel-order",
        "/api/v2/mix/order/place-order",
    ]
    assert venue.session.positions == []


def test_partial_close_preserves_stop_loss(venue_factory):
    venue = venue_factory()
    venue.submit_order(0.01, True, 0.01, 0.02)
    venue.close_position(0.004)
    assert venue.get_current_state().size == pytest.approx(0.006)
    assert any(order["planType"] == "pos_loss" for order in venue.session.plans)


def test_simultaneous_positions_are_rejected(venue_factory):
    venue = venue_factory(Session(hedge=True))
    venue.session.positions = [
        venue.session.position(),
        venue.session.position(side="short"),
    ]
    with pytest.raises(RuntimeError, match="simultaneous"):
        venue.get_current_state()


def test_foreign_orders_are_never_cancelled(venue_factory):
    venue = venue_factory()
    venue.session.plans = [
        {"orderId": "foreign", "clientOid": "another-strategy", "planType": "pos_loss"}
    ]
    with pytest.raises(RuntimeError, match="existing Bitget orders"):
        venue.submit_order(0.01, True)
    assert len(venue.session.plans) == 1
    assert not any(call[0] == "POST" for call in venue.session.calls)


def test_cancel_failure_is_not_silently_accepted(venue_factory):
    venue = venue_factory()
    venue.submit_order(0.01, True, 0.01)
    venue.session.positions = []
    venue.session.fail_cancel = True
    with pytest.raises(RuntimeError, match="Failed to cancel"):
        venue.get_current_state()
    assert venue._protective_orders


def test_limit_entry_and_validation(venue_factory):
    venue = venue_factory()
    with pytest.raises(ValueError, match="resting limit"):
        venue.submit_order(0.01, True, 0.01, order_type=OrderType.LIMIT, price=60000)
    result = venue.submit_order(0.01, True, order_type="limit", price=59000.49)
    assert venue.session.orders[result["orderId"]]["price"] == "59000"
    assert venue.session.orders[result["orderId"]]["force"] == "gtc"
    assert venue._execution_fills(result, is_buy=True) == ()
    with pytest.raises(RuntimeError, match="existing Bitget orders"):
        venue.submit_order(0.01, True)


@pytest.mark.parametrize("pct", [-0.1, float("nan"), float("inf"), 1])
def test_protection_validation_precedes_entry(venue_factory, pct):
    venue = venue_factory()
    with pytest.raises(ValueError, match="percentages"):
        venue.submit_order(0.01, True, pct)
    assert not any(call[0] == "POST" for call in venue.session.calls)


def test_read_only_mode_blocks_mutations(venue_factory):
    venue = venue_factory(read_only=True)
    with pytest.raises(RuntimeError, match="read-only"):
        venue.submit_order(0.01, True)
    with pytest.raises(RuntimeError, match="read-only"):
        venue.close_position()
    with pytest.raises(RuntimeError, match="read-only"):
        venue._request("POST", "/api/v2/mix/order/place-order", {}, signed=True)
    assert venue.get_current_state().dir == PositionDir.FLAT


def test_execution_report_and_reconciliation(venue_factory):
    venue = venue_factory()
    events = []
    venue.set_execution_event_callback(events.append)
    intent = TradeIntent(
        action=ActionType.OPEN,
        target_dir=PositionDir.POSITIVE,
        order_qty=0.01,
        price=60000,
    )
    report = venue.execute_action(intent)
    assert report.status == "filled"
    assert report.fill_vwap == 60100
    assert report.fills[0].deal_id == "1"
    assert not report.fills[0].is_aggregate
    assert venue.reconcile_execution_events(datetime.now(timezone.utc)) == 1
    assert events[0].execution_id == report.execution_id


def test_websocket_login_subscription_and_fill_identity(venue_factory):
    venue = venue_factory()
    app = SimpleNamespace(sent=[], close=lambda: None)
    app.send = app.sent.append
    venue._on_stream_open(app)
    login = json.loads(app.sent[0])["args"][0]
    assert login["sign"] == venue._sign(login["timestamp"] + "GET/user/verify")
    venue._on_stream_message(app, json.dumps({"event": "login", "code": "0"}))
    channels = json.loads(app.sent[1])["args"]
    assert all(channel["instId"] == "default" for channel in channels)
    for arg in channels:
        venue._on_stream_message(app, json.dumps({"event": "subscribe", "arg": arg}))
    assert venue.wait_for_user_stream(0)
    events = []
    venue.set_execution_event_callback(events.append)
    result = venue.submit_order(0.01, True, execution_id="trace")
    order = venue.session.orders[result["orderId"]]
    payload = {
        "arg": {"channel": "orders"},
        "data": [
            {
                **order,
                "instId": "BTCUSDT",
                "status": "filled",
                "fillPrice": "60100",
                "tradeId": "1",
                "fillTime": "1700000000001",
            }
        ],
    }
    venue._handle_user_stream_event(payload)
    venue.reconcile_execution_events(datetime.now(timezone.utc))
    assert len(events) == 2
    assert events[0].event_id == events[1].event_id
    assert events[0].fill.quantity == 0.01
    assert events[0].execution_id == "trace"
    venue._on_stream_message(app, "pong")


def test_client_ids_are_unique_and_scoped(venue_factory):
    venue = venue_factory()
    first = venue._new_client_order_id("open", "same-execution")
    second = venue._new_client_order_id("open", "same-execution")
    assert first != second
    assert len(first) <= 64
    assert venue._order_action(first) == "open"
    venue.magic = "other"
    assert venue._order_action(first) is None


@pytest.mark.parametrize(
    ("data", "data_type", "list_type"),
    [
        (None, "NoneType", "missing"),
        ([], "list", "missing"),
        ({}, "dict", "missing"),
        ({"entrustedList": None}, "dict", "NoneType"),
        ({"endId": None}, "dict", "missing"),
        ({"entrustedList": None, "endId": "123"}, "dict", "NoneType"),
        ({"entrustedList": "private-order-data"}, "dict", "str"),
        ({"entrustedList": {}}, "dict", "dict"),
    ],
)
def test_invalid_pagination_reports_types_without_values(
    venue_factory, monkeypatch, data, data_type, list_type
):
    venue = venue_factory()
    monkeypatch.setattr(venue, "_request", lambda *args, **kwargs: data)
    with pytest.raises(RuntimeError, match="invalid paginated data") as error:
        venue._pending_plans()
    message = str(error.value)
    assert "/api/v2/mix/order/orders-plan-pending" in message
    assert "expected data.entrustedList to be a list" in message
    assert f"data_type={data_type}, list_type={list_type}" in message
    assert "private-order-data" not in message


@pytest.mark.parametrize("end_id", [None, ""])
def test_startup_accepts_null_pending_plans(venue_factory, monkeypatch, end_id):
    session = Session()
    original_request = session.request

    def request(method, url, **kwargs):
        if urlsplit(url).path.endswith("/orders-plan-pending"):
            return Response({"entrustedList": None, "endId": end_id})
        return original_request(method, url, **kwargs)

    monkeypatch.setattr(session, "request", request)
    venue = venue_factory(session)
    assert venue._protective_orders == {}
    assert venue._pending_plans() == []


def test_pagination_ends_on_null_page_after_full_page(venue_factory, monkeypatch):
    venue = venue_factory()
    monkeypatch.setattr(venue, "PAGE_SIZE", 2)
    rows = [{"orderId": "3"}, {"orderId": "2"}]
    pages = [
        {"entrustedList": rows, "endId": "2"},
        {"entrustedList": None, "endId": None},
    ]
    queries = []

    def request(method, path, params, **kwargs):
        queries.append(params)
        return pages.pop(0)

    monkeypatch.setattr(venue, "_request", request)
    assert venue._pending_plans() == rows
    assert len(queries) == 2
    assert queries[1]["idLessThan"] == "2"


def test_pagination_does_not_skip_equal_timestamps(venue_factory, monkeypatch):
    venue = venue_factory()
    monkeypatch.setattr(venue, "PAGE_SIZE", 2)
    queries = []
    pages = [
        {"fillList": [{"tradeId": "3"}, {"tradeId": "2"}], "endId": "2"},
        {"fillList": [{"tradeId": "1"}], "endId": "1"},
    ]

    def request(method, path, params, **kwargs):
        queries.append(params)
        return pages.pop(0)

    monkeypatch.setattr(venue, "_request", request)
    assert len(list(venue._pages("/fills", {}, "fillList"))) == 3
    assert queries[1]["idLessThan"] == "2"


def test_runner_parses_and_constructs_bitget(credentials, monkeypatch):
    from trade.runner.live_runner import LiveRunner, _parse_venue_config
    from trade.venue.live.bitget import bitget_venue

    config = _parse_venue_config(
        str(credentials / "live.json"), {"venue": "Bitget", "bitget": {"path": "."}}
    )
    assert config.venue == "bitget"
    captured = {}

    def factory(path, symbol, magic, **kwargs):
        captured.update(path=path, symbol=symbol, magic=magic, **kwargs)
        return "venue"

    monkeypatch.setattr(bitget_venue, "BitgetVenue", factory)
    spec = SimpleNamespace(
        venue_config=config,
        strategy_id="one",
        hash_id="hash",
        base_define=SimpleNamespace(symbol="BTCUSDT"),
    )
    assert (
        object.__new__(LiveRunner)._create_venue(spec, logging.getLogger("test"))
        == "venue"
    )
    assert captured["magic"] == "one:hash"
    assert captured["path"] == str(credentials)


def test_triggered_protection_fills_restore_strategy_ownership(venue_factory):
    venue = venue_factory(Session(hedge=True))
    session = venue.session
    venue.submit_order(0.01, True, 0.01, 0.02, execution_id="entry-trace")
    plan = session.plans[1]
    session.orders["99"] = {
        "orderId": "99",
        "clientOid": "exchange-generated-id",
        "state": "filled",
        "size": "0.01",
        "baseVolume": "0.01",
        "priceAvg": "61302",
        "side": "buy",
        "tradeSide": "close",
        "posSide": "long",
        "uTime": "1700000000001",
    }
    session.plan_history = [{**plan, "executeOrderId": "99", "planStatus": "executed"}]
    events = []
    venue.set_execution_event_callback(events.append)
    assert venue.reconcile_execution_events(datetime.now(timezone.utc)) == 2
    exit_event = next(event for event in events if event.order_id == "99")
    assert exit_event.reason == "take_profit"
    assert exit_event.order_role == "exit"
    assert exit_event.side == "sell"
    assert exit_event.fill.client_order_id == plan["clientOid"]


def test_close_cancels_owned_triggered_limit_before_market_order(venue_factory):
    venue = venue_factory(Session(hedge=True))
    session = venue.session
    venue.submit_order(0.01, True, 0.01, 0.02)
    plan = session.plans.pop()
    session.orders["99"] = {
        "orderId": "99",
        "clientOid": "exchange-child",
        "state": "live",
        "size": "0.01",
        "baseVolume": "0",
        "side": "buy",
        "posSide": "long",
        "uTime": "1700000000001",
    }
    session.plan_history = [{**plan, "executeOrderId": "99", "planStatus": "executed"}]
    venue.close_position()
    assert session.orders["99"]["state"] == "canceled"
    assert session.positions == []
    posts = [call for call in session.calls if call[0] == "POST"]
    cancel_index = next(
        index for index, call in enumerate(posts) if call[1].endswith("/cancel-order")
    )
    assert posts[cancel_index + 1][1].endswith("/place-order")


def test_websocket_rejection_never_marks_stream_ready(venue_factory):
    venue = venue_factory()
    closed = []
    app = SimpleNamespace(close=lambda: closed.append(True))
    venue._on_stream_message(app, json.dumps({"event": "login", "code": "30005"}))
    assert closed == [True]
    assert not venue.wait_for_user_stream(0)
    assert "30005" in venue._stream_error


def test_error_responses_redact_credentials(venue_factory, monkeypatch):
    venue = venue_factory()
    monkeypatch.setattr(
        venue.session,
        "request",
        lambda *args, **kwargs: Response(
            code="40009", message="test-key test-secret test-passphrase"
        ),
    )
    with pytest.raises(BitgetAPIError) as failure:
        venue._account()
    assert "test-key" not in str(failure.value)
    assert "test-secret" not in str(failure.value)
    assert "test-passphrase" not in str(failure.value)


def test_unsupported_contract_and_bad_filters_fail_closed(venue_factory):
    session = Session()
    session.contract["sizeMultiplier"] = "0"
    with pytest.raises(RuntimeError, match="contract filters"):
        venue_factory(session)
    assert session.closed
    assert not any(call[0] == "POST" for call in session.calls)


def test_different_symbol_events_do_not_change_this_venue(venue_factory):
    venue = venue_factory()
    before = len(venue.session.calls)
    venue._handle_user_stream_event(
        {"arg": {"channel": "orders"}, "data": [{"instId": "ETHUSDT"}]}
    )
    assert len(venue.session.calls) == before
