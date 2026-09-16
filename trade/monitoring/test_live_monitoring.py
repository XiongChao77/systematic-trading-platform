"""Deterministic checks for monitoring delay diagnostics."""

import logging
import threading
from unittest.mock import Mock

import pytest
import requests

from trade.monitoring import live_monitoring as monitoring
from trade.core.dashboard_base import collect_dashboard
from trade.core.dashboard_reads import DashboardReads
from trade.monitoring.live_monitoring import LiveMonitoringConfig, LiveMonitoringService, LiveStateRegistry


def test_component_timing_includes_failed_calls(monkeypatch):
    registry = monitoring.LiveStateRegistry([])
    clock = iter([10.0, 18.0])
    monkeypatch.setattr(monitoring.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        with registry._time_component("BTC-binance-1", "account"):
            raise TimeoutError("Exchange timeout")
    assert registry.collection_timings == {"BTC-binance-1/account": 8.0}
    assert registry.timing_snapshot() == {"BTC-binance-1/account": 8.0}


@pytest.mark.parametrize("upload_fails", [False, True])
def test_publish_distinguishes_collection_and_upload_delay(monkeypatch, caplog, upload_fails):
    clock = [100.0]
    monkeypatch.setattr(monitoring.time, "monotonic", lambda: clock[0])
    registry = monitoring.LiveStateRegistry([])

    def collect(status):
        clock[0] += 8.0
        registry.collection_timings = {"BTC-binance-1/account": 8.0}
        return []

    monkeypatch.setattr(registry, "strategy_snapshots", collect)
    session = Mock()

    def post(*args, **kwargs):
        clock[0] += 0.25
        if upload_fails:
            raise requests.Timeout("Upload timeout")
        response = Mock()
        response.json.return_value = {"accepted": True}
        return response

    session.post.side_effect = post
    service = monitoring.LiveMonitoringService(
        monitoring.LiveMonitoringConfig("http://localhost/internal/live/snapshots", "test"),
        registry,
        logger=logging.getLogger("monitoring-test"),
        session=session,
    )
    with caplog.at_level(logging.INFO):
        assert service.publish_once() is not upload_fails
        assert "total=8.250s collect=8.000s post=0.250s" in caplog.text
        assert "BTC-binance-1/account=8.000s" in caplog.text
        assert f"accepted={not upload_fails}" in caplog.text
        service.publish_once()
    assert caplog.text.count("Live monitoring timing") == 2


def test_blocked_read_preserves_other_results_and_never_queues_retries():
    reads = DashboardReads()
    release = threading.Event()
    calls = []

    def blocked():
        calls.append(True)
        release.wait()
        return "late result"

    try:
        result = reads.batch({"account": blocked, "position": lambda: None}, timeout=0.05)
        snapshot = collect_dashboard(lambda: result("account"), lambda: result("position"))
        assert not snapshot.account_available
        assert snapshot.position_available and snapshot.position is None
        assert "timed out" in snapshot.errors[0]["message"]
        result = reads.batch({"account": blocked, "position": lambda: "fresh"}, timeout=0.05)
        with pytest.raises(TimeoutError, match="still in progress"):
            result("account")
        assert result("position") == "fresh"
        assert len(calls) == 1
    finally:
        release.set()
    reads._pending["account"].result(timeout=1)
    assert reads.batch({"account": lambda: "new"})("account") == "new"


def test_publisher_preserves_notification_received_during_upload():
    registry = LiveStateRegistry([])
    service = LiveMonitoringService(
        LiveMonitoringConfig("http://test/snapshots", "test", publish_interval_seconds=30),
        registry, logger=logging.getLogger("test"), session=Mock(),
    )
    published = []
    done = threading.Event()

    def publish():
        published.append(True)
        if len(published) == 1:
            registry.updated.set()
        else:
            service._stop_event.set()
            done.set()

    service.publish_once = publish
    registry.updated.set()
    worker = threading.Thread(target=service._run, daemon=True)
    worker.start()
    try:
        assert done.wait(1)
        assert len(published) == 2
    finally:
        service._stop_event.set()
        registry.updated.set()
        worker.join(1)
