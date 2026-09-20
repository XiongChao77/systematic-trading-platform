"""Regression tests for single-dispatch handling of closed candles."""

import logging
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from trade.core.protocol import ActionType, PositionView, Signal, TradeIntent
from trade.core.strategy_base import StrategyBase
from trade.runner import live_runner


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
        )


@pytest.fixture
def candle_runner(monkeypatch):
    def pipeline(strategy_id):
        return SimpleNamespace(
            enable=True,
            spec=SimpleNamespace(
                strategy_id=strategy_id,
                hash_id=strategy_id,
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

    frame = pd.DataFrame([{"open_time_ms_utc": OPEN_TIME_MS, "close": 0.1}])
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
    runner._prediction_callback = None
    runner._prediction_trace = None
    runner._predict = Mock(side_effect=lambda pipeline, frame: frame.assign(pred=Signal.POSITIVE.value))
    monkeypatch.setattr(live_runner, "_prepare_market_frame", lambda frame, base_define: frame.copy())
    return runner, group, first, second


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("delivery", [True, False, RuntimeError("Telegram unavailable")])
def test_data_check_warns_without_blocking_missing_event(candle_runner, monkeypatch, caplog, missing, delivery):
    runner, group, first, second = candle_runner
    runner._closed = False
    runner.feed_groups = [group]
    runner._events = queue.Queue()
    runner._data_check_timer_interval_ms = INTERVAL_MS
    runner._next_min_expect_candle_open_time = OPEN_TIME_MS
    fire_time = OPEN_TIME_MS + INTERVAL_MS + live_runner.DATA_CHECK_TIMER_DELAY_MS
    runner._next_data_check_timer_time_ms = fire_time
    runner._start_data_check_timer = Mock()
    monkeypatch.setattr(live_runner.time, "time", lambda: fire_time / 1000)
    if not missing:
        group.last_processed_candle_open_time_ms = OPEN_TIME_MS

    def send(message):
        # Telegram must not delay enqueuing the event or scheduling the next check.
        assert runner._events.qsize() == 1
        runner._start_data_check_timer.assert_called_once()
        assert "WARNING: Candle missing at DATA_CHECK" in message
        assert f"check_delay_ms={live_runner.DATA_CHECK_TIMER_DELAY_MS}" in message
        if isinstance(delivery, Exception):
            raise delivery
        return delivery

    first.notifier.send.side_effect = send
    runner._data_check_timer_handler()
    assert first.notifier.send.call_count == int(missing)
    assert second.notifier.send.call_count == int(missing)
    assert runner._events.qsize() == int(missing)
    if missing:
        event = runner._events.get_nowait()
        assert event.e_type == live_runner.RunnerEventType.DATA_CHECK
        assert event.timestamp_ms == OPEN_TIME_MS
        if delivery is not True:
            assert "DATA_CHECK warning delivery failed" in caplog.text


@pytest.mark.parametrize("age, skipped", [(29.999, False), (30.0, False), (30.001, True)])
def test_market_age_includes_queue_wait_and_uses_strict_limit(candle_runner, monkeypatch, caplog, age, skipped):
    runner, group, first, second = candle_runner
    monkeypatch.setattr(live_runner.time, "monotonic", lambda: 100.0 + age)
    event = live_runner.RunnerEvent(
        live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS,
        received_monotonic=100.0,
    )
    assert runner._process_event(event)
    for item in (first, second):
        assert item._execute_intent.call_count == (0 if skipped else 1)
        assert item.notifier.send.call_count == (1 if skipped else 0)
        assert len(item.strategy.observations) == (0 if skipped else 1)
    if skipped:
        assert "receipt_age_seconds=30.001" in caplog.text
        runner._predict.assert_not_called()
    assert not runner._process_event(event)
    assert first.notifier.send.call_count == (1 if skipped else 0)


@pytest.mark.parametrize("stage", ["prediction", "account", "strategy", "previous_execution"])
def test_market_age_is_rechecked_after_slow_work(candle_runner, monkeypatch, stage):
    runner, group, first, second = candle_runner
    clock = [100.0]
    monkeypatch.setattr(live_runner.time, "monotonic", lambda: clock[0])

    def delayed(result):
        def call(*args, **kwargs):
            clock[0] = 130.001
            return result(*args, **kwargs) if callable(result) else result
        return call

    if stage == "prediction":
        runner._predict.side_effect = delayed(lambda pipeline, frame: frame.assign(pred=Signal.POSITIVE.value))
    elif stage == "account":
        first.venue.get_account_equity.side_effect = delayed(100)
    elif stage == "strategy":
        first.strategy._process = delayed(first.strategy._process)
    else:
        first._execute_intent.side_effect = delayed(None)
    event = live_runner.RunnerEvent(
        live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS,
        received_monotonic=100.0,
    )
    runner._process_event(event)
    assert first._execute_intent.call_count == (1 if stage == "previous_execution" else 0)
    second._execute_intent.assert_not_called()
    second.notifier.send.assert_called_once()
    assert not second.strategy.observations
    if stage in {"prediction", "account"}:
        assert not first.strategy.observations


@pytest.mark.parametrize("delivery", [False, RuntimeError("Telegram unavailable")])
def test_notification_failure_does_not_allow_expired_execution(candle_runner, monkeypatch, caplog, delivery):
    runner, group, first, second = candle_runner
    monkeypatch.setattr(live_runner.time, "monotonic", lambda: 131.0)
    if isinstance(delivery, Exception):
        first.notifier.send.side_effect = delivery
    else:
        first.notifier.send.return_value = delivery
    runner._process_event(live_runner.RunnerEvent(
        live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS,
        received_monotonic=100.0,
    ))
    first._execute_intent.assert_not_called()
    second._execute_intent.assert_not_called()
    second.notifier.send.assert_called_once()
    assert "Market age warning delivery failed" in caplog.text


def test_expired_market_cannot_execute_through_invalid_cache_path(candle_runner, monkeypatch):
    runner, group, first, second = candle_runner
    group.feed.get_latest_data.return_value = None
    monkeypatch.setattr(live_runner.time, "monotonic", lambda: 131.0)
    runner._process_event(live_runner.RunnerEvent(
        live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS,
        received_monotonic=100.0,
    ))
    first._execute_intent.assert_not_called()
    second._execute_intent.assert_not_called()
    first.notifier.send.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["execution", "strategy", "recording"])
def test_failed_cycle_is_not_redispatched_and_next_candle_runs(candle_runner, caplog, failure_stage):
    runner, group, first, second = candle_runner
    error = RuntimeError("Simulated cycle failure")
    if failure_stage == "execution":
        first._execute_intent.side_effect = error
    elif failure_stage == "strategy":
        first.strategy.failure = error
    else:
        runner._record_live_cycle = Mock(side_effect=[error, None, None, None])

    event = live_runner.RunnerEvent(live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS)
    assert runner._process_event(event)
    assert len(first.strategy.observations) == 1
    assert first.strategy.observations[0].market.signal == Signal.POSITIVE
    assert first._execute_intent.call_count == (0 if failure_stage == "strategy" else 1)
    assert len(second.strategy.observations) == 1
    assert second._execute_intent.call_count == 1
    assert "Simulated cycle failure" in caplog.text
    assert "strategy_id=first hash=first symbol=DOGEUSDT" in caplog.text
    assert "K-line discontinuity" not in caplog.text

    # Neither a repeated close event nor its data-check timer may retry the bar.
    assert not runner._process_event(event)
    assert not runner._process_event(
        live_runner.RunnerEvent(live_runner.RunnerEventType.DATA_CHECK, group, OPEN_TIME_MS)
    )
    assert len(first.strategy.observations) == 1
    assert len(second.strategy.observations) == 1

    first._execute_intent.side_effect = None
    first.strategy.failure = None
    next_time = OPEN_TIME_MS + INTERVAL_MS
    group.feed.get_latest_data.return_value = pd.DataFrame(
        [{"open_time_ms_utc": next_time, "close": 0.1}]
    )
    assert runner._process_event(
        live_runner.RunnerEvent(live_runner.RunnerEventType.CLOSED_CANDLE, group, next_time)
    )
    assert len(first.strategy.observations) == 2
    assert len(second.strategy.observations) == 2
    assert first.strategy.observations[-1].candle_open_time_utc == pd.Timestamp(next_time, unit="ms", tz="UTC")


@pytest.mark.parametrize("failure_stage", ["features", "prediction"])
@pytest.mark.parametrize("execution_fails", [False, True])
def test_preparation_failure_dispatches_invalid_once(candle_runner, caplog, failure_stage, execution_fails):
    runner, group, first, second = candle_runner
    if failure_stage == "features":
        first.feature_factory.generate.side_effect = RuntimeError("Simulated feature failure")
    else:
        runner._predict.side_effect = [
            RuntimeError("Simulated prediction failure"),
            pd.DataFrame([{"close": 0.1, "pred": Signal.POSITIVE.value}]),
        ]
    if execution_fails:
        first._execute_intent.side_effect = RuntimeError("Simulated execution failure")

    event = live_runner.RunnerEvent(live_runner.RunnerEventType.CLOSED_CANDLE, group, OPEN_TIME_MS)
    assert runner._process_event(event)
    assert len(first.strategy.observations) == 1
    assert first.strategy.observations[0].market.signal == Signal.INVALID
    assert "strategy_id=first hash=first symbol=DOGEUSDT" in caplog.text
    first._execute_intent.assert_called_once()
    assert len(second.strategy.observations) == 1
    assert second.strategy.observations[0].market.signal == Signal.POSITIVE
    assert "K-line discontinuity" not in caplog.text
    assert not runner._process_event(event)
    assert len(first.strategy.observations) == 1


@pytest.fixture
def build_runner():
    runner = object.__new__(live_runner.LiveRunner)
    runner.logger = logging.getLogger("test.live_runner.build")
    runner.runner_id = "test-runner"
    runner.strategy_pipelines = []
    runner.feed_groups = []
    runner._load_model = Mock(return_value=SimpleNamespace(seq_len=1))
    runner._create_feature_generator = Mock(
        return_value=SimpleNamespace(get_global_min_history=lambda: 1)
    )
    runner._venue_factory = Mock(side_effect=lambda spec, logger: SimpleNamespace(logger=logger))
    runner._notify_factory = Mock(return_value=None)
    runner._create_strategy = Mock(side_effect=lambda spec, venue: RecordingStrategy())
    runner._feed_factory = Mock(return_value=SimpleNamespace())
    specs = [
        SimpleNamespace(
            strategy_id=strategy_id,
            hash_id=f"hash-{strategy_id}",
            base_define=live_runner.common.MarketDataSourceConfig(symbol="DOGEUSDT", interval="15m"),
            venue_config=SimpleNamespace(venue="mock"),
            run_live=True,
            enable=True,
        )
        for strategy_id in ("first", "second")
    ]
    return runner, specs


def test_venue_and_strategy_errors_have_independent_context(build_runner, caplog):
    runner, specs = build_runner
    runner._build(specs)
    for pipeline in runner.strategy_pipelines:
        pipeline.venue.logger.error("Entry recovery failed: %s", "order not found")
        pipeline.strategy.strategy_warning("Simulated strategy warning")
        try:
            raise RuntimeError("Simulated venue failure")
        except RuntimeError:
            pipeline.venue.logger.exception("Venue operation failed")

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 6
    for index, record in enumerate(errors):
        strategy_id = "first" if index < 3 else "second"
        assert record.strategy_id == strategy_id
        assert f"strategy_id={strategy_id} hash=hash-{strategy_id} symbol=DOGEUSDT" in record.getMessage()
    assert "Entry recovery failed: order not found" in errors[0].getMessage()
    assert errors[2].exc_info[0] is RuntimeError


@pytest.mark.parametrize("stage", ["_load_model", "_venue_factory", "_create_strategy"])
def test_initialization_error_identifies_strategy(build_runner, caplog, stage):
    runner, specs = build_runner
    getattr(runner, stage).side_effect = RuntimeError("Simulated initialization failure")
    with pytest.raises(RuntimeError, match="Simulated initialization failure"):
        runner._build(specs)
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].strategy_id == "first"
    assert "strategy_id=first" in errors[0].getMessage()
    assert errors[0].exc_info[0] is RuntimeError
