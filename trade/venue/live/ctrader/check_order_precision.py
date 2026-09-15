"""Offline regression checks for protection precision and broker rejection traces."""

import csv
import logging
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trade.core.protocol import ActionType, OrderType, PositionDir, TradeIntent
from trade.recording.execution_trace import (
    ExecutionTraceConfig,
    LiveExecutionTraceRecorder,
)
from trade.venue.live.ctrader.ctrader_venue import (
    CTraderOpenApiConnection,
    CTraderRequestError,
    CTraderVenue,
    OpenApiMessages_pb2,
    OpenApiModelMessages_pb2 as messages,
)


def test_broker_error_response_has_explicit_exception_type():
    connection = CTraderOpenApiConnection.__new__(CTraderOpenApiConnection)
    connection._closed = False
    connection._connected = threading.Event()
    connection._connected.set()
    connection._timeout = 1
    connection._messages = OpenApiMessages_pb2
    connection._logger = logging.getLogger("precision-test")
    connection._call_in_reactor = lambda callback: callback()
    response = OpenApiMessages_pb2.ProtoOAErrorRes(
        errorCode="INVALID_REQUEST",
        description="Relative stop loss has invalid precision",
    )
    connection._protobuf = Mock()
    connection._protobuf.extract.return_value = response
    deferred = Mock()
    deferred.addCallbacks.side_effect = lambda succeeded, failed: succeeded(object())
    connection._client = Mock()
    connection._client.send.return_value = deferred
    with pytest.raises(
        CTraderRequestError, match="Relative stop loss has invalid precision"
    ):
        connection._send_request("ProtoOANewOrderReq")


@pytest.fixture
def venue():
    instance = CTraderVenue.__new__(CTraderVenue)
    instance.digits = 2
    instance.volume_min = 1
    instance.volume_step = 1
    instance.volume_max = 10000
    instance._limited_risk = False
    instance.account_id = 123
    instance.symbol_id = 102
    instance.label = "test"
    instance.logger = logging.getLogger("precision-test")
    instance._execution_event_lock = threading.Lock()
    instance._execution_ids_by_client_message_id = {}
    instance._execution_ids_by_client_order_id = {}
    instance._execution_roles = {}
    instance.api = Mock()
    instance.api.latest_price.return_value = 2345.67
    instance.api.request.return_value = SimpleNamespace()
    instance._quote_snapshot = Mock(return_value=(2345.66, 2345.67, 0.000004))
    return instance


@pytest.mark.parametrize(
    "digits,price,ratio,expected",
    [
        (2, 2345.67, 0.0064749, 1519000),
        (2, 2345.67, 0.021583, 5063000),
        (3, 150.123, 0.0001, 1500),
        (5, 0.09117, 0.01, 91),
        (0, 2345.67, 0.01, 2300000),
        (2, 1, 0.000001, 1000),
        (2, 1, 0.015, 2000),
        (6, 1, 0.000015, 2),
    ],
)
def test_distance_quantization(venue, digits, price, ratio, expected):
    venue.digits = digits
    assert venue._relative_protection_distance(price, ratio) == expected


@pytest.mark.parametrize("is_buy", [True, False])
@pytest.mark.parametrize("order_type", [OrderType.MARKET, OrderType.LIMIT])
def test_submitted_protections_use_symbol_precision(venue, is_buy, order_type):
    venue.submit_order(
        6.18,
        is_buy,
        stop_loss_pct=0.0064749,
        take_profit_pct=0.021583,
        order_type=order_type,
        price=2345.67 if order_type == OrderType.LIMIT else None,
    )
    fields = venue.api.request.call_args.kwargs
    assert fields["relativeStopLoss"] == 1519000
    assert fields["relativeTakeProfit"] == 5063000
    assert fields["tradeSide"] == (venue.BUY if is_buy else venue.SELL)


def test_optional_protections_remain_optional(venue):
    venue.submit_order(1, True)
    fields = venue.api.request.call_args.kwargs
    assert "relativeStopLoss" not in fields
    assert "relativeTakeProfit" not in fields


def intent(quantity=6.18):
    return TradeIntent(
        action=ActionType.OPEN,
        target_dir=PositionDir.POSITIVE,
        order_qty=quantity,
        price=2345.67,
        stop_loss_pct=0.0064749,
        take_profit_pct=0.021583,
        reason="bbm_signal_entry",
    )


def test_explicit_rejection_is_written_to_execution_csv(venue, tmp_path):
    reason = "cTrader request ProtoOANewOrderReq failed: INVALID_REQUEST: Relative stop loss has invalid precision"
    venue.api.request.side_effect = CTraderRequestError(reason)
    report = venue.execute_action(intent())
    assert report.status == "rejected"
    assert report.reason == reason
    assert report.accepted_at_utc is None
    assert report.submitted_at_utc is not None
    assert report.filled_quantity == 0
    assert report.orders[0].status == "rejected"
    assert report.orders[0].submitted_quantity == 6.18
    venue.api.request.assert_called_once()
    recorder = LiveExecutionTraceRecorder(
        ExecutionTraceConfig(str(tmp_path)), runner_id="test", run_id="test"
    )
    try:
        recorder.record(
            report,
            strategy_id="ETH",
            strategy_hash="test",
            venue="CTraderVenue",
            account_id="123",
            strategy_symbol="ETHUSDT",
            venue_symbol="ETHUSD",
        )
    finally:
        recorder.close()
    with open(recorder.executions_path) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "rejected"
    assert rows[0]["reason"] == reason


def test_partial_batch_fill_survives_later_rejection(venue):
    venue.volume_max = 100
    venue.api.request.side_effect = [
        SimpleNamespace(
            order=SimpleNamespace(
                orderId=7,
                clientOrderId="first",
                tradeData=SimpleNamespace(volume=100),
                orderStatus=messages.ORDER_STATUS_FILLED,
            ),
            deal=SimpleNamespace(executionPrice=2345.67, filledVolume=100),
        ),
        CTraderRequestError("INVALID_REQUEST: rejected second child"),
    ]
    report = venue.execute_action(intent(3))
    assert venue.api.request.call_count == 2
    assert report.status == "partially_filled"
    assert report.requested_quantity == 3
    assert report.submitted_quantity == 2
    assert report.filled_quantity == 1
    assert [order.status for order in report.orders] == ["filled", "rejected"]
    assert "rejected second child" in report.reason


@pytest.mark.parametrize(
    "error", [TimeoutError("unknown result"), RuntimeError("transport failed")]
)
def test_transport_failures_are_not_reported_as_rejected_or_retried(venue, error):
    venue.api.request.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        venue.execute_action(intent())
    venue.api.request.assert_called_once()
