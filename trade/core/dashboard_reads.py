"""Bounded, isolated workers for best-effort monitoring reads."""

import logging
import threading
import time
from concurrent.futures import Future
from datetime import UTC, datetime
from queue import Queue


DASHBOARD_TIMEOUT_SECONDS = 1.0


class DashboardReads:
    """Keep one daemon worker per read; never queue behind an unfinished call."""

    def __init__(self, max_age_seconds=3.0):
        self.max_age_seconds = max_age_seconds
        self._lock = threading.Lock()
        self._workers = {}
        self._pending = {}

    @staticmethod
    def _work(queue):
        while True:
            future, operation = queue.get()
            future.started_monotonic = time.monotonic()
            try:
                value = operation()
                future.completed_at = datetime.now(UTC)
                future.completed_monotonic = time.monotonic()
                future.set_result(value)
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, key, operation, *, request_id=None):
        with self._lock:
            previous = self._pending.get(key)
            if previous is not None:
                if not previous.done():
                    return previous
                if (previous.exception() is None and self.max_age_seconds > 0
                        and time.monotonic() - previous.completed_monotonic <= self.max_age_seconds):
                    return previous
            future = Future()
            future.created_monotonic = time.monotonic()
            future.request_id = request_id
            if key not in self._workers:
                queue = Queue(maxsize=1)
                self._workers[key] = queue
                threading.Thread(target=self._work, args=(queue,), name=f"dashboard-read-{key}", daemon=True).start()
            self._pending[key] = future
            self._workers[key].put_nowait((future, operation))
            return future

    def batch(self, operations, timeout=DASHBOARD_TIMEOUT_SECONDS, *, logger=None, context=None, request_ids=None):
        started = time.monotonic()
        deadline = started + timeout
        pending = {key: self.submit(key, operation, request_id=(request_ids or {}).get(key))
                   for key, operation in operations.items()}
        sources = {key: ("new" if future.created_monotonic >= started else
                         "cache" if future.done() else "inflight") for key, future in pending.items()}
        logged = set()

        def result(key):
            waited_at = time.monotonic()
            status = "completed"
            future = pending[key]
            try:
                return future.result(timeout=max(0.0, deadline - time.monotonic()))
            except TimeoutError as exc:
                status = "timeout"
                if str(exc):
                    raise
                raise TimeoutError(f"{key} read timed out or is still in progress") from exc
            except BaseException:
                status = "failed"
                raise
            finally:
                if logger is not None and key not in logged:
                    logged.add(key)
                    now = time.monotonic()
                    worker_started = getattr(future, "started_monotonic", None)
                    completed = getattr(future, "completed_monotonic", None)
                    logger.log(
                        logging.INFO if status != "completed" else logging.DEBUG,
                        "Dashboard read timing | context=%s request=%s request_id=%s source=%s "
                        "status=%s pending=%s batch_ms=%.1f deadline_ms=%.1f wait_ms=%.1f "
                        "request_age_ms=%.1f worker_queue_ms=%.1f work_ms=%.1f cache_age_ms=%.1f",
                        context, key, future.request_id, sources[key], status, not future.done(),
                        (now - started) * 1000, timeout * 1000, (now - waited_at) * 1000,
                        (now - future.created_monotonic) * 1000,
                        ((worker_started or now) - future.created_monotonic) * 1000,
                        0.0 if worker_started is None else ((completed or now) - worker_started) * 1000,
                        0.0 if completed is None else (now - completed) * 1000,
                    )

        result.updated_at = lambda key: pending[key].completed_at
        result.request_id = lambda key: pending[key].request_id
        return result
