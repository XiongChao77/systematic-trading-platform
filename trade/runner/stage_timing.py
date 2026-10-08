"""Monotonic stage timings for the serial calculation and account workers."""

import threading
import time


class StageTiming:
    def __init__(self, logger, phase, identity, candle_at, *, received_monotonic=None, queued_monotonic=None):
        self.logger = logger
        self.phase = phase
        self.identity = identity
        self.candle_at = candle_at
        self.received = received_monotonic
        self.queued = queued_monotonic
        self.stages = {}
        self.outcome = "completed"

    def __enter__(self):
        self.started = time.monotonic()
        return self

    def call(self, name, operation, *args, **kwargs):
        started = time.monotonic()
        try:
            return operation(*args, **kwargs)
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.monotonic() - started) * 1000

    def __exit__(self, exc_type, exc_value, traceback):
        ended = time.monotonic()
        values = {"total_ms": (ended - self.started) * 1000}
        if self.received is not None:
            values["receipt_age_start_ms"] = (self.started - self.received) * 1000
            values["receipt_age_end_ms"] = (ended - self.received) * 1000
        if self.queued is not None:
            values["account_queue_ms"] = (self.started - self.queued) * 1000
        values.update({f"{name}_ms": elapsed for name, elapsed in self.stages.items()})
        self.logger.info(
            "Runner stage timing | phase=%s identity=%s candle_open_time_utc=%s thread=%s outcome=%s %s",
            self.phase, self.identity, self.candle_at.isoformat(), threading.current_thread().name,
            "failed" if exc_type is not None else self.outcome,
            " ".join(f"{key}={value:.1f}" for key, value in values.items()),
        )
        return False
