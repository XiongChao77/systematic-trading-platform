"""Regression checks for independent dashboard collection and trading priority."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from trade.core.dashboard_base import AccountBalance, AccountDashboard
from trade.monitoring.live_monitoring import LiveMonitoringConfig, LiveMonitoringService, LiveStateRegistry
from trade.monitoring.request_budget import DashboardRequestBudget
from trade.venue.live.binance import binance_venue as binance
from trade.venue.live.ctrader import ctrader_venue as ctrader
from UI.backend.modules.live.models import RunnerSnapshot
from UI.backend.modules.live.store import LiveSnapshotStore


class Dashboard(AccountDashboard):
    def __init__(self, gate=None):
        self.gate = gate
        self.started = threading.Event()

    def get_dashboard_balance(self):
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(3)
        return AccountBalance(100, 100, 0)

    def get_dashboard_position(self):
        return None


def pipeline(name, venue):
    return NS(venue=venue, enable=True, spec=NS(
        strategy_id=name, venue_config=NS(venue="mock"),
        strategy_config=NS(risk_per_trade_pct=0.01, max_daily_loss_pct=0.1),
        base_define=NS(symbol="BTCUSDT", interval="15m"),
        train_config=NS(model_cfg=NS(model_type="lstm")),
    ))


def test_slow_strategy_does_not_block_publishing_or_fast_strategy():
    release = threading.Event()
    slow = Dashboard(release)
    registry = LiveStateRegistry([pipeline("slow", slow), pipeline("fast", Dashboard())])
    published = []
    ready = threading.Event()
    session = Mock()

    def post(*args, json, **kwargs):
        published.append(json)
        items = {s["strategy_id"]: s for s in json["strategies"]}
        if len(published) >= 2 and items["fast"]["availability"]["account"]:
            assert not items["slow"]["availability"]["account"]
            ready.set()
        return NS(raise_for_status=lambda: None, json=lambda: {"accepted": True})

    session.post.side_effect = post
    service = LiveMonitoringService(
        LiveMonitoringConfig("http://test/snapshots", "test", publish_interval_seconds=0.05),
        registry, logger=logging.getLogger("test"), session=session,
    )
    try:
        service.start()
        assert slow.started.wait(1)
        assert ready.wait(2)
    finally:
        release.set()
        service.stop()
    assert published[-1]["strategies"][0]["status"] == "stopped"
    assert all(not thread.is_alive() for thread in registry._collectors)
    assert [p["sequence"] for p in published] == list(range(1, len(published) + 1))


def test_heartbeat_does_not_refresh_stale_dashboard_data():
    registry = LiveStateRegistry([pipeline("test", Dashboard())])
    strategy = registry._snapshot_pipeline(registry._pipelines["test"], None, "running")
    strategy["dashboard_age_seconds"] = 4.0
    now = datetime.now(UTC)
    store = LiveSnapshotStore()
    store._now = lambda: now
    payload = RunnerSnapshot.model_validate(dict(
        runner_id="runner", runner_instance_id="instance", runner_started_at=now,
        sent_at=now, sequence=1, strategies=[strategy],
    ))
    store.update(payload)
    assert store.strategy("test")["available"]
    now += timedelta(seconds=2)
    assert not store.strategy("test")["available"]
    payload.strategies[0].dashboard_age_seconds = 6.0
    payload.sequence = 2
    store.update(payload)
    assert store.strategies()["runners"]["available"] == 1
    assert store.strategies()["items"][0]["balance"] is None
    payload.strategies[0].dashboard_age_seconds = 0.2
    payload.sequence = 3
    store.update(payload)
    assert store.strategy("test")["available"]


@pytest.mark.parametrize("trader_fails", [False, True])
def test_ctrader_batches_three_reads_and_reuses_position_and_pnl(trader_fails):
    venue = object.__new__(ctrader.CTraderVenue)
    venue._dashboard_local = threading.local()
    venue.account_id = 1
    venue.symbol_id = 10
    venue.label = "test"
    venue.symbol = "BTCUSD"
    venue.digits = 2
    position = NS(positionId=1, price=10, usedMargin=100, moneyDigits=2,
                  tradeData=NS(symbolId=10, label="test", tradeSide=1, volume=10000,
                               openTimestamp=1_000_000_000_000))
    responses = {
        "ProtoOATraderReq": NS(trader=NS(balance=10000, moneyDigits=2, totalMarginCalculationType=0)),
        "ProtoOAReconcileReq": NS(position=[position]),
        "ProtoOAGetPositionUnrealizedPnLReq": NS(moneyDigits=2, positionUnrealizedPnL=[NS(positionId=1, netUnrealizedPnL=100)]),
    }
    barrier = threading.Barrier(3)
    calls = []

    def request(name, **kwargs):
        calls.append(name)
        assert kwargs["client_message_id"].startswith("dashboard-")
        barrier.wait(timeout=2)
        if trader_fails and name == "ProtoOATraderReq":
            raise TimeoutError("Trader data timeout")
        return responses[name]

    venue.api = NS(request=request, latest_price=lambda *a, **kw: 11)
    snapshot = venue.get_dashboard_snapshot()
    assert snapshot.errors == ([{"component": "account", "message": "Trader data timeout"}] if trader_fails else [])
    assert sorted(calls) == sorted(responses)
    assert snapshot.account == (None if trader_fails else AccountBalance(100, 101, 1))
    assert snapshot.position.quantity == 100
    assert snapshot.position.unrealized_pnl == 1
    assert snapshot.position.opened_at == datetime.fromtimestamp(1_000_000_000, UTC)


def test_ctrader_dashboard_preserves_capacity_and_priority(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(ctrader.time, "monotonic", lambda: clock[0])
    protocol = ctrader._CTraderIsolatedTcpProtocol()
    protocol.factory = NS(numberOfMessagesToSendPerSecond=5)
    protocol.heartbeat = Mock()
    sent = []
    protocol.sendString = sent.append
    for i in range(6):
        protocol.send(bytes([i]), clientMsgId=f"dashboard-{i}")
    protocol._sendStrings()
    assert sent == [bytes([i]) for i in range(4)]
    protocol.send(b"order", clientMsgId="order-1")
    assert sent[-1] == b"order"
    protocol.send(b"next-order", clientMsgId="order-2")
    clock[0] = 10.99
    protocol._sendStrings()
    assert len(sent) == 5
    clock[0] = 11.0
    protocol._sendStrings()
    assert sent[5:] == [b"next-order", bytes([4]), bytes([5])]


def test_binance_dashboard_requests_overlap_without_trading_lock(monkeypatch):
    venue = object.__new__(binance.BinanceVenue)
    venue._dashboard_local = threading.local()
    venue._dashboard_sessions = []
    venue._dashboard_sessions_lock = threading.Lock()
    venue._dashboard_pool = ThreadPoolExecutor(max_workers=2)
    venue.api_key = "test"
    venue.api_secret = "test"
    venue.symbol = "BTCUSDT"
    venue.hedge_mode = False
    venue.timeout = 1
    venue.logger = logging.getLogger("test")
    venue._request_lock = threading.Lock()
    venue.session = Mock()
    budget = DashboardRequestBudget()
    budget.limit = 2400
    monkeypatch.setattr(binance, "binance_dashboard_budget", budget)
    barrier = threading.Barrier(2)
    sessions = []

    def new_session():
        session = Mock(headers={})
        sessions.append(session)

        def request(method, url, **kwargs):
            barrier.wait(timeout=2)
            body = {"totalWalletBalance": 100, "totalMarginBalance": 100, "totalPositionInitialMargin": 0} if url.endswith("/account") else []
            return NS(status_code=200, ok=True, headers={}, json=lambda: body)

        session.request.side_effect = request
        return session

    monkeypatch.setattr(binance.requests, "Session", new_session)
    try:
        with venue._request_lock:
            snapshot = venue.get_dashboard_snapshot()
        assert snapshot.errors == []
        assert snapshot.account.balance == 100
        assert snapshot.position_available and snapshot.position is None
        assert len(sessions) == 2
        venue.session.request.assert_not_called()
    finally:
        venue._dashboard_pool.shutdown()


def test_dashboard_weight_budget_reserves_trading_capacity(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("trade.monitoring.request_budget.time.time", lambda: clock[0])
    monkeypatch.setattr("trade.monitoring.request_budget.time.monotonic", lambda: clock[0])
    budget = DashboardRequestBudget()
    budget.limit = 100
    budget.observe({"X-MBX-USED-WEIGHT-1M": "78"}, 200)
    with pytest.raises(RuntimeError):
        budget.reserve(5, dashboard=True)
    budget.reserve(5, dashboard=False)
    clock[0] = 60
    budget.reserve(5, dashboard=True)
    budget.observe({"Retry-After": "120"}, 429)
    clock[0] = 120
    with pytest.raises(RuntimeError):
        budget.reserve(5, dashboard=True)
    budget.reserve(5, dashboard=False)
