"""Offline startup and trading main-flow checks using simulated exchange I/O."""



import logging


import threading


from types import SimpleNamespace


from unittest.mock import Mock


import pytest


from trade.core.protocol import OrderType




from trade.venue.live.ctrader.ctrader_venue import CTraderVenue, CTraderOpenApiConnection


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
    instance.api = Mock(spec=CTraderOpenApiConnection)
    instance.api.accepts_market.return_value = True
    instance.api.trading_ready = True
    instance.api.last_recovered_monotonic = 0.0
    instance.api.latest_price.return_value = 2345.67
    instance.api.request.return_value = SimpleNamespace()
    instance._quote_snapshot = Mock(return_value=(2345.66, 2345.67, 0.000004))
    return instance


@pytest.mark.parametrize("is_buy", [True, False])
@pytest.mark.parametrize("order_type", [OrderType.MARKET, OrderType.LIMIT])
def test_submit_order_with_protection(venue, is_buy, order_type):
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



@pytest.mark.parametrize(
    "risk_pct,balance,equity,margin_per_unit,expected_quantity",
    [
        (0.01, 4000.0, 4800.0, 50.0, 5.0),
        (0.5, 4000.0, 4800.0, 50.0, 40.0),
        (0.5, 5200.0, 5100.0, 50.0, 52.0),
        (0.5, 4000.0, 4800.0, 200.0, 21.6),
        (0.02, 7490.0, 7490.0, 50.0, 10.0),
        (0.02, 7490.0, 7490.0, 1000.0, 6.74),
        (0.01, 7500.0, 7500.0, 50.0, 0.0),
        (0.01, 7600.0, 7600.0, 50.0, 0.0),
    ],
)
@pytest.mark.parametrize("signal", ["POSITIVE", "NEGATIVE"])
@pytest.mark.parametrize("commission_pct", [0.0, 0.1])
def test_live_entry_keeps_initial_risk_and_caps_margin_by_balance(
    venue, risk_pct, balance, equity, margin_per_unit, expected_quantity, signal, commission_pct
):
    from datetime import datetime, timezone
    from ctrader_open_api.messages import OpenApiModelMessages_pb2 as messages
    from trade.core.dashboard_base import AccountBalance
    from trade.core.protocol import ActionType, MarketView, Signal
    from trade.runner.live_runner import LiveRunner, StrategyPipeline
    from trade.strategy.strategy_bbm import BbmSignalStrategy, BbmStrategyConfig

    venue.account_id = 123
    venue.symbol = "ETHUSD"
    venue._margin_according_to_gsl = False
    venue.get_daily_reset_date = lambda timestamp: timestamp.date()
    venue.get_dashboard_balance = Mock(return_value=AccountBalance(balance, equity, 0.0))
    venue.api.latest_price.return_value = 100.0
    venue._quote_snapshot.return_value = (100.0, 100.0, 0.0)
    submitted = []

    def request(message, **fields):
        if message == "ProtoOAReconcileReq":
            return SimpleNamespace(position=[])
        if message == "ProtoOATraderReq":
            return SimpleNamespace(trader=SimpleNamespace(balance=round(balance * 100), moneyDigits=2))
        if message == "ProtoOAGetPositionUnrealizedPnLReq":
            return SimpleNamespace(moneyDigits=2, positionUnrealizedPnL=[
                SimpleNamespace(netUnrealizedPnL=round((equity - balance) * 100))
            ])
        if message == "ProtoOAExpectedMarginReq":
            volume = fields["volume"][0]
            return SimpleNamespace(ctidTraderAccountId=123, moneyDigits=2, margin=[
                SimpleNamespace(volume=volume, buyMargin=round(volume * margin_per_unit), sellMargin=round(volume * margin_per_unit))
            ])
        assert message == "ProtoOANewOrderReq"
        submitted.append(fields)
        return SimpleNamespace(
            executionType=messages.ORDER_FILLED,
            order=SimpleNamespace(orderId=1, clientOrderId=fields["clientOrderId"],
                orderStatus=messages.ORDER_STATUS_FILLED,
                tradeData=SimpleNamespace(volume=fields["volume"])),
            deal=SimpleNamespace(orderId=1, dealId=2, executionPrice=100.0,
                filledVolume=fields["volume"]),
        )

    venue.api.request.side_effect = request
    strategy = BbmSignalStrategy(
        BbmStrategyConfig(compound=False, risk_per_trade_pct=risk_pct, max_daily_loss_pct=0.99,
            threshold_long=1.0, threshold_short=1.0, stop_loss_long=1.0,
            stop_loss_short=1.0),
        init_equity=5000.0, data_interval_ms=900000, leverage=1.0,
    )
    spec = SimpleNamespace(instance_id="balance-entry", hash_id="strategy",
        base_define=SimpleNamespace(symbol="ETHUSDT"),
        broker_config=SimpleNamespace(initial_equity=5000.0, commission_pct=commission_pct),
        venue_config=SimpleNamespace(venue="ctrader", trader_login="123",
            max_loss=0.5, profit_target=0.5))
    pipeline = StrategyPipeline(spec, None, venue, strategy, None, 900000,
        notifier=Mock(send=Mock(return_value=True)))
    runner = LiveRunner.__new__(LiveRunner)
    runner.logger = logging.getLogger("test.balance-entry")
    runner._record_execution = Mock()
    market = MarketView(price=100.0, signal=Signal[signal], expected_vol=0.1)
    candle_at = datetime(2026, 9, 26, tzinfo=timezone.utc)
    venue.api.accepts_market.return_value = False
    assert runner._dispatch(pipeline, market, candle_at) is None
    assert pipeline.enable
    venue.api.request.assert_not_called()
    assert submitted == []
    venue.api.accepts_market.return_value = True
    intent = runner._dispatch(pipeline, market, candle_at)

    assert intent.action == ActionType.OPEN
    assert strategy.last_state.account.balance == balance
    assert strategy.last_state.account.equity == equity
    report = runner._record_execution.call_args.args[1]
    if expected_quantity == 0:
        assert submitted == []
        assert report.status == "rejected"
        assert report.reason == "profit_target_reached"
        assert not pipeline.enable
        pipeline.notifier.send.assert_called_once()
        assert "event=profit_target_reached" in pipeline.notifier.send.call_args.args[0]
        return
    assert pipeline.enable
    expected_take_profit = min(0.1, (7550.0 - equity) / (100.0 * expected_quantity) + 2 * commission_pct / 100)
    assert intent.take_profit_pct == pytest.approx(expected_take_profit)
    assert len(submitted) == 1
    assert submitted[0]["relativeTakeProfit"] == round(100 * expected_take_profit, 2) * 100000
    assert submitted[0]["volume"] == round(expected_quantity * 100)
    report = runner._record_execution.call_args.args[1]
    assert report.status == "filled"
    assert report.filled_quantity == pytest.approx(expected_quantity)
