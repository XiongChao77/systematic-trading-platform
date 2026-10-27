"""Bounded account queues for serial strategy execution outside model inference."""

import queue
import threading
from concurrent.futures import Future


class AccountExecution:
    """Run one task at a time per account while allowing accounts to overlap."""

    def __init__(self, logger, capacity=64):
        self.logger = logger
        self.capacity = capacity
        self._lock = threading.Lock()
        self._workers = {}
        self._closed = False

    def submit(self, account, operation):
        with self._lock:
            if self._closed:
                raise RuntimeError("Account execution is closed")
            if account not in self._workers:
                pending = queue.Queue(maxsize=self.capacity)
                thread = threading.Thread(
                    target=self._run, args=(pending,),
                    name=f"account-execution-{len(self._workers) + 1}",
                )
                self._workers[account] = (pending, thread)
                thread.start()
            pending, _ = self._workers[account]
            future = Future()
            pending.put_nowait((future, operation))
            return future

    def _run(self, pending):
        while True:
            item = pending.get()
            try:
                if item is None:
                    return
                future, operation = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(operation())
                except BaseException as exc:
                    future.set_exception(exc)
                    self.logger.exception("Account execution task failed")
            finally:
                pending.task_done()

    def wait(self):
        """Wait for already submitted tasks, primarily for offline callers."""
        with self._lock:
            workers = list(self._workers.values())
        for pending, _ in workers:
            pending.join()

    def close(self):
        """Cancel queued tasks and finish active tasks before releasing venues."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = list(self._workers.values())
            for pending, _ in workers:
                while True:
                    try:
                        future, _ = pending.get_nowait()
                    except queue.Empty:
                        break
                    future.cancel()
                    pending.task_done()
                pending.put_nowait(None)
        for _, thread in workers:
            thread.join()
