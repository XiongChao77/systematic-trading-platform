"""Shared IP-weight observations keep dashboard polling below trading capacity."""

import threading
import time


class DashboardRequestBudget:
    def __init__(self):
        self._lock = threading.Lock()
        self.limit = None
        self._minute = None
        self._used = 0
        self._blocked_until = 0.0

    def configure(self, rate_limits):
        for item in rate_limits:
            if item.get("rateLimitType") == "REQUEST_WEIGHT" and item.get("interval") == "MINUTE" and item.get("intervalNum") == 1:
                with self._lock:
                    self.limit = int(item["limit"])

    def _roll(self):
        minute = int(time.time() // 60)
        if minute != self._minute:
            self._minute = minute
            self._used = 0

    def reserve(self, weight, *, dashboard):
        with self._lock:
            self._roll()
            if dashboard and (
                self.limit is None or time.monotonic() < self._blocked_until
                or self._used + weight > self.limit * 0.8
            ):
                raise RuntimeError("Dashboard request deferred to reserve Binance capacity for trading")
            self._used += weight

    def observe(self, headers, status):
        with self._lock:
            self._roll()
            used = headers.get("X-MBX-USED-WEIGHT-1M")
            if used is not None:
                self._used = max(self._used, int(used))
            if status in (418, 429):
                self._blocked_until = time.monotonic() + max(1.0, float(headers.get("Retry-After", 60)))


binance_dashboard_budget = DashboardRequestBudget()
