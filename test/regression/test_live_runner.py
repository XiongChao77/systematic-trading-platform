"""Offline startup and trading main-flow checks using simulated exchange I/O."""

import logging






import threading


from types import SimpleNamespace


from unittest.mock import Mock


import pandas as pd


import pytest


from trade.core.protocol import ActionType, PositionView, Signal, TradeIntent


from trade.core.strategy_base import StrategyBase


from trade.runner import live_runner
from test.regression.test_bitget_venue import credentials, venue_factory


INTERVAL_MS = 900_000


OPEN_TIME_MS = 1_788_786_000_000


class RecordingStrategy(StrategyBase):
    def __init__(self):
        super().__init__(INTERVAL_MS)
        self.observations = []
        self.failure = None

    def _process(self, state):
        self.observations.append(state)
        if self.failure is not None:
            raise self.failure
        return TradeIntent(
            action=ActionType.NOOP if state.market.signal == Signal.INVALID else ActionType.OPEN,
            price=state.market.price,
            order_qty=1,
            target_dir=live_runner.PositionDir.POSITIVE,
        )


@pytest.fixture
def candle_runner(monkeypatch):
    def pipeline(instance_id):
        return SimpleNamespace(
            enable=True,
            spec=SimpleNamespace(
                instance_id=instance_id,
                hash_id=instance_id,
                account_id="shared-account",
                venue_config=SimpleNamespace(venue="mock"),
                base_define=SimpleNamespace(symbol="DOGEUSDT"),
            ),
            strategy=RecordingStrategy(),
            feature_factory=SimpleNamespace(generate=Mock(side_effect=lambda frame: frame)),
            venue=SimpleNamespace(
                get_current_state=Mock(return_value=PositionView()),
                get_account_equity=Mock(return_value=100),
                get_daily_reset_date=Mock(side_effect=lambda timestamp: timestamp.date()),
            ),
            _execute_intent=Mock(return_value=None),
            notifier=Mock(send=Mock(return_value=True)),
        )

    frame = pd.DataFrame([{"open_time_ms_utc": OPEN_TIME_MS, "close": 60000}])
    first, second = pipeline("first"), pipeline("second")
    group = live_runner.FeedGroup(
        market_config=SimpleNamespace(symbol="DOGEUSDT", interval="15m"),
        interval_ms=INTERVAL_MS,
        feed=SimpleNamespace(get_latest_data=Mock(return_value=frame)),
        required_bars=1,
        pipelines=[first, second],
        last_processed_candle_open_time_ms=OPEN_TIME_MS - INTERVAL_MS,
    )
    runner = object.__new__(live_runner.LiveRunner)
    runner.logger = logging.getLogger("test.live_runner")
    runner._closed = False
    runner._calculation_lock = threading.RLock()
    runner._account_execution = live_runner.AccountExecution(runner.logger)
    runner._failure_notifications = live_runner.ExecutionFailureNotifications("test-runner", runner.logger)
    runner._prediction_callback = None
    runner._prediction_trace = None
    runner._predict = Mock(side_effect=lambda pipeline, frame: frame.assign(pred=Signal.POSITIVE.value))
    monkeypatch.setattr(live_runner, "_prepare_market_frame", lambda frame, base_define: frame.copy())
    runner._process_event_async = runner._process_event

    def process_and_wait(event):
        result = runner._process_event_async(event)
        runner.wait_for_execution()
        return result

    runner._process_event = process_and_wait
    yield runner, group, first, second
    runner._account_execution.close()
    runner._failure_notifications.close()



def test_closed_candle_reaches_filled_exchange_order(candle_runner, venue_factory):
    runner, group, first, second = candle_runner
    second.enable = False
    first.venue = venue_factory()
    reports = []

    def execute(observation, intent, *, expired):
        assert not expired()
        report = first.venue.execute_action(intent)
        reports.append(report)
        return report

    first._execute_intent = execute
    assert runner._process_event(live_runner.RunnerEvent(
        live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS,
    ))
    runner._predict.assert_called_once()
    assert len(reports) == 1
    assert reports[0].status == "filled"
    assert reports[0].filled_quantity == 1
    assert len(first.venue.session.orders) == 1


@pytest.mark.parametrize("failure", ["rejected", "precheck", "async", "partial", "exception", "unknown", "expired"])
def test_closed_candle_order_failure_notifies_telegram_without_tracing(candle_runner, failure):
    from datetime import datetime, timezone
    from trade.core.execution import ExecutionEvent, ExecutionFill, ExecutionOrder, ExecutionReport

    runner, group, first, second = candle_runner
    second.enable = False
    now = datetime.now(timezone.utc)
    child = ExecutionOrder(1.0, order_id="failed-child", client_order_id="client-1", status="rejected")
    report = ExecutionReport(
        side="buy", requested_quantity=1.0, decision_price=100.0, decision_at_utc=now,
        bid=100.0, ask=100.0, spread_pct=0.0, execution_id="execution-1",
        status="rejected", reason="NOT_ENOUGH_MONEY", orders=(child,),
    )
    if failure == "precheck":
        report = live_runner.replace(report, orders=(), submitted_quantity=0.0, reason="margin_precheck_failed")
    elif failure == "async":
        report = live_runner.replace(report, status="accepted", reason="",
            orders=(live_runner.replace(child, status="accepted"),))
    elif failure == "partial":
        report = live_runner.replace(report, status="partially_filled", requested_quantity=2.0,
            submitted_quantity=2.0, fills=(ExecutionFill(100.0, 1.0),),
            orders=(ExecutionOrder(1.0, order_id="filled-child", status="filled"), child))
    elif failure == "expired":
        report = live_runner.replace(report, status="expired", reason="Order expired",
            orders=(live_runner.replace(child, status="expired"),))
    elif failure == "unknown":
        report = None
    if failure == "exception":
        first._execute_intent = Mock(side_effect=ConnectionError("Transport lost"))
    else:
        first._execute_intent = Mock(return_value=report)

    if failure == "async":
        first.venue.activate_execution_updates = lambda execution_id: runner._record_execution_event(
            first, ExecutionEvent(execution_id=execution_id, status="rejected", event_at_utc=now,
                order_role="entry", side="buy", order_id="failed-child", client_order_id="client-1",
                submitted_quantity=1.0, reason="NOT_ENOUGH_MONEY"))
    assert runner._process_event(live_runner.RunnerEvent(
        live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS,
    ))
    if failure in {"rejected", "partial", "async"}:
        event = ExecutionEvent(execution_id="execution-1", status="rejected", event_at_utc=now,
            order_role="entry", side="buy", order_id="failed-child", client_order_id="client-1",
            submitted_quantity=1.0, reason="NOT_ENOUGH_MONEY")
        runner._record_execution_event(first, event)
        runner._record_execution_event(first, event)
    runner._failure_notifications.close()
    first.notifier.send.assert_called_once()
    message = first.notifier.send.call_args.args[0]
    assert "event=order_execution_failed" in message
    assert "instance_id=first" in message
    if failure in {"rejected", "partial", "async"}:
        assert "NOT_ENOUGH_MONEY" in message
        assert "order_id=failed-child" in message
