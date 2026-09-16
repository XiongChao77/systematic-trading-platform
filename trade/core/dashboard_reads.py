"""Bounded, isolated workers for best-effort monitoring reads."""

import threading
import time
from concurrent.futures import Future
from queue import Queue


DASHBOARD_TIMEOUT_SECONDS = 1.0


class DashboardReads:
    """Keep one daemon worker per read; never queue behind an unfinished call."""

    def __init__(self):
        self._lock = threading.Lock()
        self._workers = {}
        self._pending = {}

    @staticmethod
    def _work(queue):
        while True:
            future, operation = queue.get()
            try:
                future.set_result(operation())
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, key, operation):
        with self._lock:
            previous = self._pending.get(key)
            future = Future()
            if previous is not None and not previous.done():
                future.set_exception(TimeoutError(f"{key} read is still in progress"))
                return future
            if key not in self._workers:
                queue = Queue(maxsize=1)
                self._workers[key] = queue
                threading.Thread(target=self._work, args=(queue,), name=f"dashboard-read-{key}", daemon=True).start()
            self._pending[key] = future
            self._workers[key].put_nowait((future, operation))
            return future

    def batch(self, operations, timeout=DASHBOARD_TIMEOUT_SECONDS):
        deadline = time.monotonic() + timeout
        pending = {key: self.submit(key, operation) for key, operation in operations.items()}

        def result(key):
            try:
                return pending[key].result(timeout=max(0.0, deadline - time.monotonic()))
            except TimeoutError as exc:
                if str(exc):
                    raise
                raise TimeoutError(f"{key} read timed out or is still in progress") from exc

        return result
