"""Check API timing attribution without making exchange requests."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from trade.venue.live.binance import binance_venue as binance
from trade.venue.live.ctrader import ctrader_venue as ctrader


@pytest.mark.parametrize("fails", [False, True])
def test_binance_logs_lock_and_http_time(monkeypatch, caplog, fails):
    clock = [0.0]
    monkeypatch.setattr(binance.time, "monotonic", lambda: clock[0])

    class DelayedLock:
        def __enter__(self):
            clock[0] += 0.2

        def __exit__(self, *args):
            pass

    venue = object.__new__(binance.BinanceVenue)
    venue._request_lock = DelayedLock()
    venue.symbol = "BTCUSDT"
    venue.timeout = 10
    venue.logger = logging.getLogger("test.binance")
    venue.session = Mock()

    def request(*args, **kwargs):
        clock[0] += 0.4
        if fails:
            raise requests.Timeout("test")
        return SimpleNamespace(status_code=200, headers={}, ok=True, json=lambda: {"ok": True})

    venue.session.request.side_effect = request
    with caplog.at_level(logging.INFO):
        if fails:
            with pytest.raises(requests.Timeout):
                venue._request("GET", "/fapi/v3/account")
        else:
            assert venue._request("GET", "/fapi/v3/account") == {"ok": True}
    assert "total_ms=600.0 lock_wait_ms=200.0 http_ms=400.0" in caplog.text
    assert f"success={not fails}" in caplog.text


@pytest.mark.parametrize("canceled", [False, True])
def test_ctrader_logs_queue_time_without_changing_cancellation(monkeypatch, caplog, canceled):
    clock = [0.0]
    monkeypatch.setattr(ctrader.time, "monotonic", lambda: clock[0])
    protocol = ctrader._CTraderIsolatedTcpProtocol()
    protocol.factory = SimpleNamespace(numberOfMessagesToSendPerSecond=5)
    protocol.sendString = Mock()
    protocol.heartbeat = Mock()
    protocol.send(b"test", clientMsgId="dashboard-request-1", isCanceled=lambda: canceled)
    clock[0] = 0.9
    with caplog.at_level(logging.INFO):
        protocol._sendStrings()
    assert "request_id=dashboard-request-1 queue_wait_ms=900.0" in caplog.text
    assert f"canceled={canceled}" in caplog.text
    assert protocol.sendString.call_count == (0 if canceled else 1)


def test_ctrader_logs_request_round_trip(monkeypatch, caplog):
    clock = [0.0]
    monkeypatch.setattr(ctrader.time, "monotonic", lambda: clock[0])
    connection = object.__new__(ctrader.CTraderOpenApiConnection)
    connection._logger = logging.getLogger("test.ctrader")
    connection._closed = False
    connection._timeout = 10
    connection._connected = Mock()
    connection._messages = SimpleNamespace(ProtoOATraderReq=lambda: SimpleNamespace())
    connection._set_message_fields = Mock()
    connection._protobuf = SimpleNamespace(extract=lambda value: value)

    def dispatch(callback):
        clock[0] += 0.1
        callback()

    def add_callbacks(success, failure):
        clock[0] += 0.95
        success(SimpleNamespace())

    connection._call_in_reactor = dispatch
    connection._client = SimpleNamespace(send=lambda *a, **kw: SimpleNamespace(addCallbacks=add_callbacks))
    with caplog.at_level(logging.INFO):
        connection._send_request("ProtoOATraderReq", client_message_id="request-1")
    assert "request_id=request-1" in caplog.text
    assert "success=True total_ms=1050.0" in caplog.text
    assert "dispatch_ms=100.0 sdk_round_trip_ms=950.0" in caplog.text
