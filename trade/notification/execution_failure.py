"""Nonblocking Telegram delivery for synchronous and asynchronous order failures."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import re
import threading
import time

from trade.core.execution import ExecutionReport


class ExecutionFailureNotifications:
    FAILURE_STATUSES = frozenset(
        {"rejected", "failed", "error", "expired", "cancel_rejected"}
    )

    def __init__(self, runner_id, logger):
        self.runner_id = runner_id
        self.logger = logger
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="order-failure-telegram"
        )
        self._lock = threading.Lock()
        self._keys = set()

    @staticmethod
    def exception_reason(exc):
        # Extract codes without forwarding HTTP URLs, headers or signed requests.
        details = [type(exc).__name__]
        match = re.search(
            r"(?:code=|failed: )(-?[0-9]+|[A-Z][A-Z_]+)(?:[:, ]|$)", str(exc)
        )
        if match:
            details.append(f"exchange_code={match[1]}")
        code = getattr(exc, "status_code", None)
        if isinstance(code, int):
            details.append(f"status_code={code}")
        return "; ".join(details)

    def notify(
        self,
        pipeline,
        *,
        execution_id,
        status,
        reason,
        order_role="",
        side="",
        order_id="",
        client_order_id="",
        quantity=None,
        event_at=None,
    ):
        # Order IDs distinguish rejected children in a partially filled execution.
        key = (
            pipeline.spec.instance_id,
            order_id or client_order_id or execution_id,
            status,
        )
        with self._lock:
            if key in self._keys:
                return
            self._keys.add(key)
        when = event_at or datetime.now(timezone.utc)
        message = (
            f"ERROR | event=order_execution_failed | runner_id={self.runner_id} | "
            f"instance_id={pipeline.spec.instance_id} | strategy_hash={pipeline.spec.hash_id} | "
            f"symbol={pipeline.spec.base_define.symbol} | "
            f"execution_id={execution_id} | order_id={order_id} | client_order_id={client_order_id} | "
            f"order_role={order_role} | side={side} | status={status} | quantity={quantity} | "
            f"event_at_utc={when.isoformat()} | reason={str(reason)[:1500]}"
        )
        self.logger.error(message)
        try:
            self._executor.submit(self._deliver, pipeline.notifier, message, key)
        except RuntimeError:
            with self._lock:
                self._keys.discard(key)
            self.logger.error(
                "Order failure notification queue unavailable | %s", message
            )

    def connection_transition(self, connection, pipelines, event, reason):
        """Send one transition per recipient and outage; hourly failures stay local."""
        state = connection.recovery_state()
        groups = {}
        for pipeline in pipelines:
            destination = getattr(pipeline.spec.venue_config, "telegram_token_path", None)
            recipient = destination or id(pipeline.notifier)
            groups.setdefault(recipient, []).append(pipeline)
        for recipient, affected in groups.items():
            key = (id(connection), state["episode"], event, recipient)
            with self._lock:
                if key in self._keys:
                    continue
                self._keys.add(key)
            message = (
                f"event=connection_{event} | runner_id={self.runner_id} | "
                f"environment={connection.environment} | "
                f"instance_ids={','.join(p.spec.instance_id for p in affected)} | "
                f"next_retry_at={state['next_retry_at']} | "
                f"event_at_utc={datetime.now(timezone.utc).isoformat()} | reason={reason}"
            )
            self.logger.info(message)
            try:
                self._executor.submit(self._deliver, affected[0].notifier, message, key)
            except RuntimeError:
                with self._lock:
                    self._keys.discard(key)
                self.logger.error("Connection notification queue unavailable | %s", message)

    def _deliver(self, notifier, message, key):
        for attempt in range(3):
            try:
                if notifier is not None and notifier.send(message):
                    return
            except Exception as exc:
                self.logger.error(
                    "Order failure notification attempt failed | error_type=%s",
                    type(exc).__name__,
                )
            if attempt < 2:
                time.sleep(attempt + 1)
        with self._lock:
            self._keys.discard(key)
        self.logger.error(
            "Order failure Telegram delivery exhausted after 3 attempts | %s", message
        )

    def is_failure(self, status, role):
        return status in self.FAILURE_STATUSES or (
            status == "cancelled" and role == "entry"
        )

    def report(self, pipeline, report):
        if not isinstance(report, ExecutionReport):
            return
        failed_orders = [
            order
            for order in report.orders
            if self.is_failure(order.status, report.order_role)
        ]
        for order in failed_orders:
            self.notify(
                pipeline,
                execution_id=report.execution_id,
                status=order.status,
                reason=report.reason or "Exchange child order failed",
                order_role=report.order_role,
                side=report.side,
                order_id=order.order_id,
                client_order_id=order.client_order_id,
                quantity=order.submitted_quantity,
                event_at=report.completed_at_utc,
            )
        if not failed_orders and (
            self.is_failure(report.status, report.order_role)
            or report.status == "unknown"
        ):
            order = report.orders[0] if len(report.orders) == 1 else None
            self.notify(
                pipeline,
                execution_id=report.execution_id,
                status=report.status,
                reason=report.reason or "Execution did not complete successfully",
                order_role=report.order_role,
                side=report.side,
                order_id=order.order_id if order else "",
                client_order_id=order.client_order_id if order else "",
                quantity=report.submitted_quantity,
                event_at=report.completed_at_utc,
            )
        elif (
            not report.orders
            and not report.fills
            and report.submitted_quantity
            and report.status == "submitted"
        ):
            self.notify(
                pipeline,
                execution_id=report.execution_id,
                status="unknown",
                reason="Submission returned no order acknowledgement; verify exchange state before retrying",
                order_role=report.order_role,
                side=report.side,
                quantity=report.submitted_quantity,
                event_at=report.completed_at_utc,
            )

    def event(self, pipeline, event):
        if self.is_failure(event.status, event.order_role):
            self.notify(
                pipeline,
                execution_id=event.execution_id,
                status=event.status,
                reason=event.reason or "Exchange order failed",
                order_role=event.order_role,
                side=event.side,
                order_id=event.order_id,
                client_order_id=event.client_order_id,
                quantity=event.submitted_quantity,
                event_at=event.event_at_utc,
            )

    def close(self):
        self._executor.shutdown(wait=True)
