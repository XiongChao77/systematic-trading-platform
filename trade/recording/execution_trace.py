"""Asynchronous CSV recording for live execution lifecycles."""

from __future__ import annotations

import math
import os
import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from trade.recording.persistent_csv import AppendCsv
from trade.core.execution import ExecutionEvent, ExecutionReport

SCHEMA_VERSION = "4"


@dataclass(frozen=True)
class ExecutionTraceConfig:
    """Destination for persistent execution trace journals."""

    output_dir: str

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("execution_trace.output_dir must not be empty")


def _number(value: Any) -> Any:
    if value is None:
        return ""
    number = float(value)
    return format(number, ".17g") if math.isfinite(number) else ""


def _time(value: datetime | None) -> str:
    return value.astimezone(UTC).isoformat() if value is not None else ""


def _identity_fields(
    *,
    runner_id: str,
    run_id: str,
    instance_id: str,
    strategy_hash: str,
    venue: str,
    account_id: str,
    strategy_symbol: str,
    venue_symbol: str,
    trader_login: str = "",
) -> dict[str, str]:
    if venue.casefold() in {"ctrader", "ctradervenue"} and (not trader_login or account_id):
        raise ValueError("cTrader trace identity requires trader_login and no account_id")
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_id": runner_id,
        "run_id": run_id,
        "instance_id": instance_id,
        "strategy_hash": strategy_hash,
        "venue": venue,
        "account_id": account_id,
        "trader_login": trader_login,
        "strategy_symbol": strategy_symbol,
        "venue_symbol": venue_symbol,
    }


class LiveExecutionTraceRecorder:
    """Write execution reports and asynchronous events off the trading thread."""

    IDENTITY_FIELDS = (
        "schema_version",
        "runner_id",
        "run_id",
        "instance_id",
        "strategy_hash",
        "venue",
        "account_id",
        "trader_login",
        "strategy_symbol",
        "venue_symbol",
    )
    EXECUTION_FIELDS = IDENTITY_FIELDS + (
        "execution_id",
        "order_role",
        "side",
        "status",
        "reason",
        "decision_at_utc",
        "quote_at_utc",
        "submitted_at_utc",
        "accepted_at_utc",
        "first_fill_at_utc",
        "completed_at_utc",
        "requested_quantity",
        "submitted_quantity",
        "filled_quantity",
        "decision_price",
        "arrival_price",
        "best_fill_price",
        "worst_fill_price",
        "fill_vwap",
        "decision_slippage_bps",
        "implementation_shortfall_bps",
        "execution_slippage_vwap_bps",
        "execution_slippage_worst_bps",
        "decision_to_first_fill_ms",
        "submit_to_first_fill_ms",
        "acknowledgement_latency_ms",
        "bid",
        "ask",
        "spread_pct",
        "child_order_count",
        "fill_count",
    )
    CHILD_ORDER_FIELDS = IDENTITY_FIELDS + (
        "execution_id",
        "order_role",
        "side",
        "order_index",
        "order_id",
        "client_order_id",
        "submitted_quantity",
        "status",
    )
    FILL_FIELDS = IDENTITY_FIELDS + (
        "execution_id",
        "order_role",
        "side",
        "fill_index",
        "order_id",
        "client_order_id",
        "deal_id",
        "executed_at_utc",
        "price",
        "quantity",
        "is_aggregate",
    )
    EVENT_FIELDS = IDENTITY_FIELDS + (
        "event_id",
        "execution_id",
        "order_role",
        "side",
        "status",
        "event_at_utc",
        "order_id",
        "client_order_id",
        "submitted_quantity",
        "reason",
        "deal_id",
    )

    def __init__(
        self,
        config: ExecutionTraceConfig,
        *,
        runner_id: str = "live-runner",
        run_id: str | None = None,
        logger=None,
        started_at: datetime | None = None,
    ) -> None:
        timestamp = (started_at or datetime.now(UTC)).astimezone(UTC)
        session = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = os.path.abspath(config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.executions_path = os.path.join(
            output_dir,
            "executions.csv",
        )
        self.orders_path = os.path.join(output_dir, "orders.csv")
        self.fills_path = os.path.join(output_dir, "fills.csv")
        self.events_path = os.path.join(output_dir, "events.csv")
        self.runner_id = str(runner_id)
        self.run_id = str(
            run_id or session
        )
        self._logger = logger
        self._queue: queue.Queue[Any] = queue.Queue()
        self._sentinel = object()
        self._closed = False
        self._error = None
        self._writers = []
        try:
            for path, fields in zip(self.paths, (self.EXECUTION_FIELDS, self.CHILD_ORDER_FIELDS, self.FILL_FIELDS, self.EVENT_FIELDS)):
                self._writers.append(AppendCsv(path, fields))
        except Exception:
            for writer in self._writers:
                writer.close()
            raise
        self._thread = threading.Thread(
            target=self._run,
            name="live-execution-trace",
            daemon=True,
        )
        self._thread.start()

    @property
    def paths(self) -> tuple[str, str, str, str]:
        return (
            self.executions_path,
            self.orders_path,
            self.fills_path,
            self.events_path,
        )

    def record(
        self,
        report: ExecutionReport,
        *,
        instance_id: str,
        strategy_hash: str,
        venue: str,
        account_id: str,
        strategy_symbol: str,
        venue_symbol: str,
        trader_login: str = "",
    ) -> None:
        if self._closed:
            return
        self._queue.put_nowait(
            (
                "report",
                report,
                self._identity(
                    instance_id=instance_id,
                    strategy_hash=strategy_hash,
                    venue=venue,
                    account_id=account_id,
                    trader_login=trader_login,
                    strategy_symbol=strategy_symbol,
                    venue_symbol=venue_symbol,
                ),
            )
        )

    def record_event(
        self,
        event: ExecutionEvent,
        *,
        instance_id: str,
        strategy_hash: str,
        venue: str,
        account_id: str,
        strategy_symbol: str,
        venue_symbol: str,
        trader_login: str = "",
    ) -> None:
        if self._closed:
            return
        self._queue.put_nowait(
            (
                "event",
                event,
                self._identity(
                    instance_id=instance_id,
                    strategy_hash=strategy_hash,
                    venue=venue,
                    account_id=account_id,
                    trader_login=trader_login,
                    strategy_symbol=strategy_symbol,
                    venue_symbol=venue_symbol,
                ),
            )
        )

    def _identity(self, **fields: str) -> dict[str, str]:
        return _identity_fields(
            runner_id=self.runner_id,
            run_id=self.run_id,
            **fields,
        )

    def _run(self) -> None:
        try:
            with self._writers[0], self._writers[1], self._writers[2], self._writers[3]:
                executions_writer, orders_writer, fills_writer, events_writer = self._writers
                handles = self._writers

                while True:
                    item = self._queue.get()
                    if item is self._sentinel:
                        return
                    kind, payload, identity = item
                    if kind == "report":
                        self._write_report(
                            payload,
                            identity,
                            executions_writer,
                            orders_writer,
                            fills_writer,
                        )
                    else:
                        self._write_event(
                            payload,
                            identity,
                            fills_writer,
                            events_writer,
                        )
                    for handle in handles:
                        handle.flush()
        except Exception as exc:
            self._error = exc
            if self._logger is not None:
                self._logger.exception("Live execution trace writer failed")

    @classmethod
    def _write_report(
        cls,
        report,
        identity,
        executions_writer,
        orders_writer,
        fills_writer,
    ) -> None:
        executions_writer.writerow(cls._execution_row(report, identity))
        for index, order in enumerate(report.orders):
            orders_writer.writerow(
                identity
                | {
                    "execution_id": report.execution_id,
                    "order_role": report.order_role,
                    "side": report.side,
                    "order_index": index,
                    "order_id": order.order_id,
                    "client_order_id": order.client_order_id,
                    "submitted_quantity": _number(order.submitted_quantity),
                    "status": order.status,
                }
            )
        for index, fill in enumerate(report.fills):
            fills_writer.writerow(
                cls._fill_row(
                    report.execution_id,
                    report.order_role,
                    report.side,
                    index,
                    fill,
                    identity,
                )
            )
    @classmethod
    def _write_event(
        cls,
        event,
        identity,
        fills_writer,
        events_writer,
    ) -> None:
        deal_id = event.fill.deal_id if event.fill is not None else ""
        event_id = event.event_id or ":".join(
            (
                event.execution_id,
                event.order_id,
                event.status,
                deal_id,
                _time(event.event_at_utc),
            )
        )
        events_writer.writerow(
            identity
            | {
                "event_id": event_id,
                "execution_id": event.execution_id,
                "order_role": event.order_role,
                "side": event.side,
                "status": event.status,
                "event_at_utc": _time(event.event_at_utc),
                "order_id": event.order_id,
                "client_order_id": event.client_order_id,
                "submitted_quantity": _number(event.submitted_quantity),
                "reason": event.reason,
                "deal_id": deal_id,
            }
        )
        if event.fill is not None:
            fills_writer.writerow(
                cls._fill_row(
                    event.execution_id,
                    event.order_role,
                    event.side,
                    "",
                    event.fill,
                    identity,
                )
            )

    @staticmethod
    def _execution_row(report, identity):
        return identity | {
            "execution_id": report.execution_id,
            "order_role": report.order_role,
            "side": report.side,
            "status": report.status,
            "reason": report.reason,
            "decision_at_utc": _time(report.decision_at_utc),
            "quote_at_utc": _time(report.quote_at_utc),
            "submitted_at_utc": _time(report.submitted_at_utc),
            "accepted_at_utc": _time(report.accepted_at_utc),
            "first_fill_at_utc": _time(report.first_fill_at_utc),
            "completed_at_utc": _time(report.completed_at_utc),
            "requested_quantity": _number(report.requested_quantity),
            "submitted_quantity": _number(report.submitted_quantity),
            "filled_quantity": _number(report.filled_quantity),
            "decision_price": _number(report.decision_price),
            "arrival_price": _number(report.arrival_price),
            "best_fill_price": _number(report.best_fill_price),
            "worst_fill_price": _number(report.worst_fill_price),
            "fill_vwap": _number(report.fill_vwap),
            "decision_slippage_bps": _number(report.decision_slippage_bps),
            "implementation_shortfall_bps": _number(
                report.implementation_shortfall_bps
            ),
            "execution_slippage_vwap_bps": _number(report.execution_slippage_vwap_bps),
            "execution_slippage_worst_bps": _number(
                report.execution_slippage_worst_bps
            ),
            "decision_to_first_fill_ms": _number(report.decision_to_first_fill_ms),
            "submit_to_first_fill_ms": _number(report.submit_to_first_fill_ms),
            "acknowledgement_latency_ms": _number(report.acknowledgement_latency_ms),
            "bid": _number(report.bid),
            "ask": _number(report.ask),
            "spread_pct": _number(report.spread_pct),
            "child_order_count": len(report.orders),
            "fill_count": len(report.fills),
        }

    @staticmethod
    def _fill_row(
        execution_id,
        order_role,
        side,
        fill_index,
        fill,
        identity,
    ):
        return identity | {
            "execution_id": execution_id,
            "order_role": order_role,
            "side": side,
            "fill_index": fill_index,
            "order_id": fill.order_id,
            "client_order_id": fill.client_order_id,
            "deal_id": fill.deal_id,
            "executed_at_utc": _time(fill.executed_at_utc),
            "price": _number(fill.price),
            "quantity": _number(fill.quantity),
            "is_aggregate": str(fill.is_aggregate).lower(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._sentinel)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Execution trace writer failed") from self._error
