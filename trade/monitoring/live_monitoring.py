"""Publish best-effort live UI snapshots without affecting trading."""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from urllib.parse import quote, urljoin
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Optional

import requests

from data_process import common
from trade.runner.connection_state import pipeline_connection
from trade.core.dashboard_base import AccountDashboard
from trade.core.dashboard_reads import DashboardReads


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class LiveMonitoringConfig:
    """Runner-level destination for ephemeral live UI snapshots."""

    publish_url: str
    runner_id: str
    publish_interval_seconds: float = 1.0
    request_timeout_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not self.publish_url.startswith(("http://", "https://")):
            raise ValueError("monitoring.publish_url must be an HTTP(S) URL")
        if not self.runner_id.strip():
            raise ValueError("runner_id must not be empty")
        if self.publish_interval_seconds <= 0:
            raise ValueError("monitoring.publish_interval_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("monitoring.request_timeout_seconds must be positive")


class LiveStateRegistry:
    """Thread-safe latest signal and decision state for running strategies."""

    def __init__(self, pipelines) -> None:
        self._lock = threading.Lock()
        self._pipelines = {pipeline.spec.instance_id: pipeline for pipeline in pipelines}
        self._signals: dict[str, dict[str, Any]] = {}
        self._position_open_times: dict[str, tuple[str, datetime]] = {}
        self.collection_timings: dict[str, float] = {}
        self._snapshots: dict[str, tuple[float, dict[str, Any]]] = {}
        self._collection_stop = threading.Event()
        self._collectors: list[threading.Thread] = []
        self.updated = threading.Event()
        self._timing_reads = DashboardReads(max_age_seconds=0)
        self._snapshot_reads = DashboardReads(max_age_seconds=0)
        self._overview_until = 0.0
        self._detail_until = {}
        self._collection_wake = {identity: threading.Event() for identity in self._pipelines}

    def connection_changed(self, pipelines):
        with self._lock:
            for pipeline in pipelines:
                identity = pipeline.spec.instance_id
                connection = pipeline_connection(pipeline)
                if connection is None or connection.trading_ready:
                    self._snapshots.pop(identity, None)
                self._position_open_times.pop(identity, None)
                self._signals.pop(identity, None)
                wake = self._collection_wake.get(identity)
                if wake is not None:
                    wake.set()
        self.updated.set()

    @staticmethod
    def _connection_state(pipeline):
        connection = pipeline_connection(pipeline)
        return None if connection is None else connection.recovery_state()

    @staticmethod
    def _runtime_status(pipeline, status, connection):
        if status == "stopped":
            return status
        if not pipeline.enable:
            return "disabled"
        if connection is not None and connection["status"] != "ready":
            return "paused"
        return status

    def record_prediction(self, pipeline, predicted_row, market, updated_at):
        payload = {
            "model_output": market.signal.name.lower(),
            "raw_model_output": _finite_number(predicted_row.get("pred")),
            "probability": _finite_number(predicted_row.get("pred_prob")),
            "net_score": _finite_number(predicted_row.get("net_score")),
            "decision": "pending", "target_side": "pending", "quantity": None,
            "reason": "Waiting for strategy execution", "updated_at": _iso(updated_at),
        }
        with self._lock:
            self._signals[pipeline.spec.instance_id] = payload

    def record_execution_status(self, pipeline, updated_at, status, reason):
        with self._lock:
            signal = self._signals.get(pipeline.spec.instance_id)
            if signal is not None and signal["updated_at"] == _iso(updated_at):
                signal.update(decision=status, reason=reason)

    def record_cycle(
        self,
        pipeline,
        predicted_row,
        market,
        intent,
        updated_at: datetime,
    ) -> None:
        raw_output = _finite_number(predicted_row.get("pred"))
        payload = {
            "model_output": market.signal.name.lower(),
            "raw_model_output": raw_output,
            "probability": _finite_number(predicted_row.get("pred_prob")),
            "net_score": _finite_number(predicted_row.get("net_score")),
            "decision": intent.action.value,
            "target_side": intent.target_dir.name.lower(),
            "quantity": _finite_number(intent.order_qty),
            "reason": intent.reason or "",
            "updated_at": _iso(updated_at),
        }
        with self._lock:
            current = self._signals.get(pipeline.spec.instance_id)
            if current is not None and current["updated_at"] > payload["updated_at"]:
                return
            self._signals[pipeline.spec.instance_id] = payload
            if intent.action.value in {"open", "close"}:
                self._position_open_times.pop(pipeline.spec.instance_id, None)

    def update_demand(self, demand) -> None:
        now = time.monotonic()
        with self._lock:
            def scope(identity):
                return ("detail" if self._detail_until.get(identity, 0) > now else
                        "overview" if self._overview_until > now else None)
            previous = {identity: scope(identity) for identity in self._pipelines}
            self._overview_until = now + max(0.0, min(10.0, float(demand.get("overview_seconds", 0))))
            self._detail_until = {
                identity: now + max(0.0, min(10.0, float(seconds)))
                for identity, seconds in demand.get("detail_seconds", {}).items()
                if identity in self._pipelines
            }
            for identity, wake in self._collection_wake.items():
                if scope(identity) != previous[identity]:
                    wake.set()

    def collection_scope(self, instance_id):
        now = time.monotonic()
        with self._lock:
            if self._detail_until.get(instance_id, 0) > now:
                return "detail"
            if self._overview_until > now:
                return "overview"
        return None

    def start_collection(self, interval_seconds: float, logger: logging.Logger) -> None:
        if self._collectors:
            return
        self._collection_stop.clear()
        for instance_id, pipeline in self._pipelines.items():
            thread = threading.Thread(
                target=self._collect_strategy, args=(instance_id, pipeline, interval_seconds, logger),
                name=f"dashboard-{instance_id}", daemon=True,
            )
            self._collectors.append(thread)
            thread.start()

    def _collect_strategy(self, instance_id, pipeline, interval_seconds, logger):
        wake = self._collection_wake[instance_id]
        while not self._collection_stop.is_set():
            wake.clear()
            scope = self.collection_scope(instance_id)
            connection = self._connection_state(pipeline)
            if connection is not None and connection["status"] != "ready":
                wake.wait(interval_seconds)
                continue
            if scope is None:
                with self._lock:
                    self._snapshots.pop(instance_id, None)
                wake.wait(interval_seconds)
                continue
            started = time.monotonic()
            try:
                snapshot = self._snapshot_pipeline(pipeline, None, "running", scope=scope)
                with self._lock:
                    if self._collection_stop.is_set():
                        return
                    self._snapshots[instance_id] = (started, snapshot)
            except Exception as exc:
                logger.exception("Dashboard collection failed | strategy=%s", instance_id)
                snapshot = self._snapshot_pipeline(pipeline, None, "running", collect=False)
                snapshot["errors"].append({"component": "dashboard", "message": str(exc) or type(exc).__name__})
                with self._lock:
                    self._snapshots[instance_id] = (started, snapshot)
            self.updated.set()
            elapsed = time.monotonic() - started
            logger.debug("Dashboard collection timing | strategy=%s total_ms=%.1f", instance_id, elapsed * 1000)
            wake.wait(max(0.0, interval_seconds - elapsed))

    def stop_collection(self) -> None:
        self._collection_stop.set()
        for wake in self._collection_wake.values():
            wake.set()
        deadline = time.monotonic() + 2.0
        for thread in self._collectors:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def strategy_snapshots(self, status: str = "running") -> list[dict[str, Any]]:
        """Copy latest data without waiting for any exchange request."""
        now = time.monotonic()
        with self._lock:
            cached = deepcopy(self._snapshots)
            signals = deepcopy(self._signals)
        results = []
        for instance_id, pipeline in self._pipelines.items():
            entry = cached.get(instance_id)
            connection = self._connection_state(pipeline)
            snapshot = entry[1] if entry else self._snapshot_pipeline(pipeline, None, status, collect=False)
            snapshot["dashboard_age_seconds"] = None if entry is None else max(0.0, now - entry[0])
            snapshot["status"] = self._runtime_status(pipeline, status, connection)
            snapshot["connection"] = connection
            if connection is not None and connection["status"] != "ready":
                snapshot["collection_scope"] = "idle"
                snapshot["availability"]["account"] = False
                snapshot["availability"]["position"] = False
            snapshot["latest_signal"] = signals.get(instance_id)
            snapshot["availability"]["latest_signal"] = instance_id in signals
            position = snapshot["position"]
            if position and position.get("opened_at") and snapshot["max_holding_seconds"] is not None:
                opened_at = datetime.fromisoformat(position["opened_at"])
                elapsed = max(0.0, (_utc_now() - opened_at).total_seconds())
                position["remaining_holding_seconds"] = max(0.0, snapshot["max_holding_seconds"] - elapsed)
            results.append(snapshot)
        return results

    def timing_snapshot(self):
        with self._lock:
            return dict(self.collection_timings)

    @contextmanager
    def _time_component(self, instance_id: str, component: str):
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            with self._lock:
                self.collection_timings[f"{instance_id}/{component}"] = elapsed

    def _position_opened_at(self, pipeline, dashboard_position) -> datetime | None:
        instance_id = pipeline.spec.instance_id
        side = dashboard_position.side.value
        with self._lock:
            cached = self._position_open_times.get(instance_id)
        if cached is not None and cached[0] == side:
            return cached[1]

        opened_at = dashboard_position.opened_at
        if opened_at is None:
            result = self._timing_reads.batch({instance_id: lambda: pipeline.venue.get_dashboard_position_open_time(dashboard_position)})
            opened_at = result(instance_id)
        if opened_at is None:
            return None
        if not isinstance(opened_at, datetime):
            raise TypeError("Venue returned a non-datetime position opening time")
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=UTC)
        else:
            opened_at = opened_at.astimezone(UTC)
        with self._lock:
            self._position_open_times[instance_id] = (side, opened_at)
        return opened_at

    def _clear_position_opened_at(self, instance_id: str) -> None:
        with self._lock:
            self._position_open_times.pop(instance_id, None)

    @staticmethod
    def _max_holding_seconds(pipeline) -> float | None:
        fixed_hold_bars = getattr(
            pipeline.spec.strategy_config,
            "fixed_hold_bars",
            None,
        )
        if fixed_hold_bars is None:
            return None
        bars = int(fixed_hold_bars)
        if bars < 0 or bars != fixed_hold_bars:
            raise ValueError("fixed_hold_bars must be a non-negative integer")
        interval_ms = getattr(pipeline, "interval_ms", None)
        if interval_ms is None:
            interval_ms = common.get_interval_ms(pipeline.spec.base_define.interval)
        interval_ms = int(interval_ms)
        return bars * interval_ms / 1000.0

    def _snapshot_pipeline(self, pipeline, signal, status: str, *, collect: bool = True, scope: str = "detail") -> dict[str, Any]:
        account = None
        position = None
        dashboard_position = None
        account_updated_at = position_updated_at = None
        account_available = False
        position_available = False
        errors = []
        connection = self._connection_state(pipeline)
        if connection is not None and connection["status"] != "ready":
            collect = False
        venue = pipeline.venue
        venue_name = str(
            getattr(
                getattr(pipeline.spec, "venue_config", None),
                "venue",
                "",
            )
        ).strip()
        if not venue_name:
            venue_name = type(venue).__name__.removesuffix("Venue").casefold()
        strategy_config = pipeline.spec.strategy_config
        risk_per_trade_pct = _finite_number(getattr(strategy_config, "risk_per_trade_pct", None))
        max_daily_loss_pct = _finite_number(getattr(strategy_config, "max_daily_loss_pct", None))
        max_holding_seconds = self._max_holding_seconds(pipeline)
        if risk_per_trade_pct is None or max_daily_loss_pct is None:
            raise RuntimeError("Live strategy risk configuration must contain finite " "risk_per_trade_pct and max_daily_loss_pct values")

        if collect and not isinstance(venue, AccountDashboard):
            errors.append({"component": "dashboard", "message": f"{type(venue).__name__} has no dashboard interface"})
        elif collect:
            with self._time_component(pipeline.spec.instance_id, "dashboard"):
                result = self._snapshot_reads.batch(
                    {(pipeline.spec.instance_id, scope): (venue.get_dashboard_snapshot if scope == "detail"
                                                        else venue.get_dashboard_overview)}, timeout=1.25,
                    logger=logging.getLogger(__name__) if venue_name.casefold() == "ctrader" else None,
                    context=f"snapshot:{pipeline.spec.instance_id}:{scope}",
                )
                dashboard = result((pipeline.spec.instance_id, scope))
            account = None if dashboard.account is None else asdict(dashboard.account)
            dashboard_position = dashboard.position
            account_updated_at = dashboard.account_updated_at
            position_updated_at = dashboard.position_updated_at
            account_available = dashboard.account_available
            position_available = dashboard.position_available
            errors.extend(dashboard.errors)
            if dashboard_position is not None:
                position = asdict(dashboard_position)
                position["opened_at"] = (
                    _iso(dashboard_position.opened_at)
                    if dashboard_position.opened_at is not None else None
                )
                position["side"] = dashboard_position.side.value
                position["margin_mode"] = dashboard_position.margin_mode.value

        if dashboard_position is None:
            if position_available:
                self._clear_position_opened_at(pipeline.spec.instance_id)
        elif scope == "detail":
            opened_at = None
            try:
                with self._time_component(pipeline.spec.instance_id, "position_timing"):
                    opened_at = self._position_opened_at(
                        pipeline,
                        dashboard_position,
                    )
            except Exception as exc:
                errors.append(
                    {
                        "component": "position_timing",
                        "message": str(exc) or type(exc).__name__,
                    }
                )
            remaining_holding_seconds = None
            if opened_at is not None and max_holding_seconds is not None:
                elapsed_seconds = max(
                    0.0,
                    (_utc_now() - opened_at).total_seconds(),
                )
                remaining_holding_seconds = max(
                    0.0,
                    max_holding_seconds - elapsed_seconds,
                )
            position.update(
                {
                    "opened_at": None if opened_at is None else _iso(opened_at),
                    "remaining_holding_seconds": remaining_holding_seconds,
                }
            )

        return {
            "collection_scope": scope if collect else "idle",
            "instance_id": pipeline.spec.instance_id,
            "strategy_hash": pipeline.spec.hash_id,
            "account_id": "" if venue_name == "ctrader" else pipeline.spec.account_id,
            "trader_login": str(pipeline.spec.venue_config.trader_login) if venue_name == "ctrader" else "",
            "model_type": pipeline.spec.train_config.model_cfg.model_type,
            "venue": venue_name,
            "symbol": pipeline.spec.base_define.symbol,
            "interval": pipeline.spec.base_define.interval,
            "initial_balance": getattr(pipeline, "initial_balance", None),
            "start_time": getattr(pipeline, "start_time", None),
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_holding_seconds": max_holding_seconds,
            "status": self._runtime_status(pipeline, status, connection),
            "connection": connection,
            "account_updated_at": _iso(account_updated_at) if account_updated_at else None,
            "position_updated_at": _iso(position_updated_at) if position_updated_at else None,
            "account": account,
            "position": position,
            "latest_signal": signal,
            "availability": {
                "account": account_available,
                "position": position_available,
                "latest_signal": signal is not None,
            },
            "errors": errors,
        }


class LiveMonitoringService:
    """Collect dashboard data and publish only the newest in-memory snapshot."""

    ERROR_LOG_INTERVAL_SECONDS = 60.0

    def __init__(
        self,
        config: LiveMonitoringConfig,
        registry: LiveStateRegistry,
        *,
        logger: logging.Logger,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.logger = logger
        self.session = session or requests.Session()
        self.runner_instance_id = str(uuid.uuid4())
        self.started_at = _utc_now()
        self._sequence = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._demand_thread: threading.Thread | None = None
        self._last_error_log_at = 0.0
        self._last_publish_started = None
        self._last_component_error_logs: dict[tuple[str, str], float] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self.registry.start_collection(self.config.publish_interval_seconds, self.logger)
        self._demand_thread = threading.Thread(target=self._watch_demand,
                                               name=f"live-demand-{self.config.runner_id}", daemon=True)
        self._demand_thread.start()
        self.registry.updated.set()
        self._thread = threading.Thread(
            target=self._run,
            name=f"live-monitoring-{self.config.runner_id}",
            daemon=True,
        )
        self._thread.start()

    def _watch_demand(self) -> None:
        version = -1
        last_error = 0.0
        url = urljoin(self.config.publish_url, f"demand/{quote(self.config.runner_id, safe='')}")
        with requests.Session() as session:
            while not self._stop_event.is_set():
                try:
                    response = session.get(url, params={"version": version},
                                           timeout=(self.config.request_timeout_seconds, 4.0))
                    response.raise_for_status()
                    result = response.json()
                    if self._stop_event.is_set():
                        break
                    self.registry.update_demand(result["monitoring_demand"])
                    version = int(result["version"])
                except Exception as exc:
                    now = time.monotonic()
                    if now - last_error >= self.ERROR_LOG_INTERVAL_SECONDS:
                        self.logger.warning("Live monitoring demand failed | runner=%s error=%s",
                                            self.config.runner_id, exc)
                        last_error = now
                    self._stop_event.wait(1.0)

    def _run(self) -> None:
        # Publish one combined snapshot per interval, never once per collector.
        interval = self.config.publish_interval_seconds
        next_publish = time.monotonic()
        while not self._stop_event.wait(max(0.0, next_publish - time.monotonic())):
            self.registry.updated.clear()
            self.publish_once()
            next_publish += interval
            if next_publish <= time.monotonic():
                # Skip missed ticks instead of sending catch-up bursts.
                next_publish = time.monotonic() + interval

    def _payload(self, status: str) -> dict[str, Any]:
        self._sequence += 1
        payload = {
            "runner_id": self.config.runner_id,
            "runner_instance_id": self.runner_instance_id,
            "runner_started_at": _iso(self.started_at),
            "sequence": self._sequence,
            "sent_at": _iso(_utc_now()),
            "strategies": self.registry.strategy_snapshots(status=status),
        }
        self._log_component_errors(payload["strategies"])
        return payload

    def _log_component_errors(self, strategies: list[dict[str, Any]]) -> None:
        now = time.monotonic()
        for strategy in strategies:
            for error in strategy.get("errors", []):
                key = (strategy["instance_id"], error["component"])
                last_log = self._last_component_error_logs.get(key, 0.0)
                if now - last_log < self.ERROR_LOG_INTERVAL_SECONDS:
                    continue
                self._last_component_error_logs[key] = now
                self.logger.info(
                    "Live dashboard data unavailable | strategy=%s component=%s error=%s",
                    strategy["instance_id"],
                    error["component"],
                    error["message"],
                )

    def publish_once(self, status: str = "running") -> bool:
        started = time.monotonic()
        publish_gap_ms = (None if self._last_publish_started is None else
                          (started - self._last_publish_started) * 1000)
        self._last_publish_started = started
        collection_finished = None
        accepted = False
        try:
            payload = self._payload(status)
            collection_finished = time.monotonic()
            response = self.session.post(
                self.config.publish_url,
                json=payload,
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            try:
                result = response.json()
            except (TypeError, ValueError):
                result = None
            if isinstance(result, Mapping) and result.get("accepted") is False:
                raise RuntimeError(f"Snapshot rejected: {result.get('reason', 'unknown reason')}")
            accepted = True
            return True
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_error_log_at >= self.ERROR_LOG_INTERVAL_SECONDS:
                self._last_error_log_at = now
                self.logger.warning(
                    "Live monitoring publish failed | runner=%s url=%s error=%s",
                    self.config.runner_id,
                    self.config.publish_url,
                    exc,
                )
            return False
        finally:
            finished = time.monotonic()
            elapsed = finished - started
            collect_seconds = (finished if collection_finished is None else collection_finished) - started
            post_seconds = 0.0 if collection_finished is None else finished - collection_finished
            components = ", ".join(
                f"{name}={duration:.3f}s"
                for name, duration in sorted(
                    self.registry.timing_snapshot().items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            self.logger.log(
                logging.WARNING if elapsed > self.config.publish_interval_seconds else logging.DEBUG,
                "Live monitoring timing | runner=%s accepted=%s total=%.3fs "
                "collect=%.3fs post=%.3fs interval=%.3fs publish_gap_ms=%s components=[%s]",
                self.config.runner_id, accepted, elapsed, collect_seconds,
                post_seconds, self.config.publish_interval_seconds,
                "na" if publish_gap_ms is None else f"{publish_gap_ms:.1f}", components,
            )

    def stop(self) -> None:
        self._stop_event.set()
        self.registry.updated.set()
        self.registry.stop_collection()
        if self._demand_thread is not None:
            self._demand_thread.join(timeout=0.5)
        if self._thread is not None:
            thread = self._thread
            thread.join(timeout=(self.config.publish_interval_seconds + self.config.request_timeout_seconds + 0.5))
            self._thread = None
            if thread.is_alive():
                self.logger.warning(
                    "Live monitoring thread did not stop before shutdown timeout | " "runner=%s",
                    self.config.runner_id,
                )
                return
        self.publish_once(status="stopped")
        self.session.close()


def monitoring_config_from_mapping(
    value: Mapping[str, Any] | None,
    *,
    publish_url: str | None = None,
    runner_id: str | None = None,
) -> LiveMonitoringConfig | None:
    """Build monitoring configuration with explicit overrides."""

    if value is not None and not isinstance(value, Mapping):
        raise TypeError("monitoring must be a JSON object")
    raw = dict(value or {})
    if "runner_id" in raw:
        raise ValueError("runner_id belongs at the configuration root, not inside monitoring")
    enable = raw.get("enable", True)
    if not isinstance(enable, bool):
        raise TypeError("monitoring.enable must be a JSON boolean")
    if not enable:
        return None
    final_url = str(publish_url or raw.get("publish_url") or "").strip()
    if not final_url:
        return None
    final_runner_id = str(runner_id or "").strip()
    return LiveMonitoringConfig(
        publish_url=final_url,
        runner_id=final_runner_id,
        publish_interval_seconds=float(raw.get("publish_interval_seconds", 1.0)),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", 0.5)),
    )
