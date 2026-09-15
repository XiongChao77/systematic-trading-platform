"""Margin precheck regression coverage; all broker and Telegram calls are mocked."""

import logging
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ctrader_open_api.messages import OpenApiMessages_pb2

from trade.core.execution import ExecutionReport
from trade.core.protocol import ActionType, PositionDir, TradeIntent
from trade.runner.live_runner import StrategyPipeline
from trade.venue.live.ctrader.ctrader_venue import CTraderVenue


@pytest.fixture
def pipeline():
    def normalize(size):
        quantity = math.floor(size * 100) / 100
        if quantity < 1:
            raise ValueError("Below minimum")
        return quantity

    venue = Mock()
    venue.logger = logging.getLogger("margin-check")
    venue.get_execution_account_id.return_value = "12345"
    venue.get_available_margin.return_value = 1000
    venue.get_expected_margin.side_effect = lambda size, **_: size * 10
    venue.normalize_order_quantity.side_effect = normalize
    venue.execute_action.side_effect = lambda intent: ExecutionReport(
        side="buy", requested_quantity=intent.order_qty, status="accepted",
        decision_price=intent.price, decision_at_utc=intent.created_at_utc,
        bid=10, ask=10, spread_pct=0,
    )
    return StrategyPipeline(
        spec=SimpleNamespace(
            strategy_id="strategy-7", hash_id="hash-7",
            base_define=SimpleNamespace(symbol="DOGEUSDT"),
            broker_config=SimpleNamespace(initial_equity=10000, commission_pct=0),
            venue_config=SimpleNamespace(venue="ctrader", max_loss=.1, profit_target=.1),
        ),
        model=None, venue=venue, strategy=None, feature_factory=None,
        interval_ms=1000, notifier=Mock(),
    )


def execute(pipeline, **kwargs):
    intent = TradeIntent(
        action=ActionType.OPEN, target_dir=PositionDir.POSITIVE,
        order_qty=100, price=10, stop_loss_pct=.01, take_profit_pct=.01,
    )
    for key, value in kwargs.items():
        setattr(intent, key, value)
    return pipeline._execute_intent(SimpleNamespace(account=SimpleNamespace(equity=10000)), intent)


def test_sufficient_margin_keeps_quantity(pipeline):
    result = execute(pipeline)
    assert result.submitted_quantity == 100
    pipeline.notifier.send.assert_not_called()


def test_reduction_rechecks_and_warns_each_entry(pipeline, caplog):
    pipeline.venue.get_available_margin.return_value = 253.17
    with caplog.at_level(logging.WARNING):
        result = execute(pipeline)
        execute(pipeline)
    assert result.requested_quantity == 100
    assert result.submitted_quantity == 25.31
    assert pipeline.notifier.send.call_count == 2
    message = pipeline.notifier.send.call_args.args[0]
    for text in ("strategy_id=strategy-7", "required_margin=1000", "available_margin=253.17", "scale_ratio=0.2531"):
        assert text in message
        assert text in caplog.text
    assert pipeline.venue.get_expected_margin.call_args.args == (25.31,)


def test_nonlinear_margin_is_requeried(pipeline):
    pipeline.venue.get_available_margin.return_value = 200
    pipeline.venue.get_expected_margin.side_effect = lambda q, **_: 100 + q * 10
    result = execute(pipeline)
    assert result.status == "accepted"
    assert result.submitted_quantity < 18.18
    assert 100 + result.submitted_quantity * 10 <= 200


@pytest.mark.parametrize("available", [0, 5])
def test_unaffordable_minimum_rejects_without_disabling(pipeline, available):
    pipeline.venue.get_available_margin.return_value = available
    result = execute(pipeline)
    assert result.status == "rejected"
    assert result.reason == "insufficient_margin_below_minimum"
    assert result.requested_quantity == 100
    pipeline.venue.execute_action.assert_not_called()
    pipeline.notifier.send.assert_called_once()
    assert pipeline.enable


def test_invalid_precheck_never_submits(pipeline):
    pipeline.venue.get_expected_margin.return_value = float("nan")
    pipeline.venue.get_expected_margin.side_effect = None
    assert execute(pipeline).reason == "margin_precheck_failed"
    pipeline.venue.execute_action.assert_not_called()


def test_notification_failure_does_not_restore_oversized_order(pipeline):
    pipeline.venue.get_available_margin.return_value = 200
    pipeline.notifier.send.side_effect = RuntimeError("Telegram unavailable")
    assert execute(pipeline).submitted_quantity == 20


def test_close_bypasses_margin_precheck(pipeline):
    execute(pipeline, action=ActionType.CLOSE)
    pipeline.venue.get_available_margin.assert_not_called()
    pipeline.venue.get_expected_margin.assert_not_called()


def test_venue_margin_uses_matching_volume_side_and_money_digits():
    venue = object.__new__(CTraderVenue)
    venue.account_id, venue.symbol_id = 123, 456
    venue.volume_min, venue.volume_step = 100, 100
    venue._margin_according_to_gsl = False
    response = OpenApiMessages_pb2.ProtoOAExpectedMarginRes(
        ctidTraderAccountId=123, moneyDigits=3,
    )
    response.margin.add(volume=200, buyMargin=12500, sellMargin=18500)
    venue.api = Mock()
    venue.api.request.return_value = response
    assert venue.get_expected_margin(2, is_buy=True) == 12.5
    assert venue.get_expected_margin(2, is_buy=False) == 18.5
    venue.api.request.assert_called_with(
        "ProtoOAExpectedMarginReq", ctidTraderAccountId=123, symbolId=456, volume=[200],
    )
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        venue.get_expected_margin(3, is_buy=True)
    venue._margin_according_to_gsl = True
    with pytest.raises(RuntimeError, match="GSL"):
        venue.get_expected_margin(2, is_buy=True)


def test_free_margin_subtracts_used_margin_and_rejects_missing_values():
    venue = object.__new__(CTraderVenue)
    venue.get_dashboard_balance = Mock(return_value=SimpleNamespace(equity=1000, used_margin=750))
    assert venue.get_available_margin() == 250
    venue.get_dashboard_balance.return_value.used_margin = None
    with pytest.raises(RuntimeError, match="unavailable"):
        venue.get_available_margin()
