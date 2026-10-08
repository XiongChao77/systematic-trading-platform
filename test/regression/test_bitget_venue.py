"""Offline startup and trading main-flow checks using simulated exchange I/O."""

from __future__ import annotations








import json


import logging


from datetime import datetime, timezone




from types import SimpleNamespace


from urllib.parse import parse_qsl, urlsplit


import pytest


import requests


from trade.core.protocol import ActionType, PositionDir, TradeIntent


from trade.venue.live.bitget.bitget_venue import BitgetVenue


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
                if not self.positions:
                    # Position TP/SL is cancelled by the exchange on full closure.
                    self.plans = []
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
        transport = session or Session()
        factory = kwargs.pop("session_factory", lambda: SimpleNamespace(request=transport.request, close=lambda: None))
        venue = BitgetVenue(
            str(credentials),
            "BTCUSDT",
            "strategy:abc",
            session=transport,
            session_factory=factory,
            enable_user_stream=False,
            **kwargs,
        )
        venues.append(venue)
        return venue

    yield create
    for venue in venues:
        venue.shutdown()


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
    assert not any(call[1].endswith("/cancel-plan-order") for call in session.calls)
    assert session.orders["2"].get("tradeSide") == ("close" if hedge else None)
    assert session.orders["2"].get("reduceOnly") == (None if hedge else "YES")


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


def test_runner_parses_and_constructs_bitget(credentials, monkeypatch):
    from trade.runner.live_runner import LiveRunner, _parse_venue_config
    from trade.venue.live.bitget import bitget_venue

    config = _parse_venue_config(
        str(credentials / "live.json"), {"venue": "Bitget", "bitget": {"path": "."}},
        telegram_token_path="telegram.json",
    )
    assert config.venue == "bitget"
    (credentials / "telegram.json").write_text('{"bot_token":"test-token","chat_id":"test-chat"}')
    notifier = LiveRunner._create_notifier(SimpleNamespace(venue_config=config), logging.getLogger("test"))
    assert notifier is not None
    captured = {}

    def factory(path, symbol, magic, **kwargs):
        captured.update(path=path, symbol=symbol, magic=magic, **kwargs)
        return "venue"

    monkeypatch.setattr(bitget_venue, "BitgetVenue", factory)
    spec = SimpleNamespace(
        venue_config=config,
        instance_id="one",
        hash_id="hash",
        base_define=SimpleNamespace(symbol="BTCUSDT"),
    )
    assert (
        object.__new__(LiveRunner)._create_venue(spec, logging.getLogger("test"))
        == "venue"
    )
    assert captured["magic"] == "one"
    assert captured["path"] == str(credentials)

