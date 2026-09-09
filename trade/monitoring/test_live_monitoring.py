"""Deterministic checks for monitoring delay diagnostics."""

import logging
from unittest.mock import Mock

import pytest
import requests

from trade.monitoring import live_monitoring as monitoring


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
