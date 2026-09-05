#!/usr/bin/env python3
"""Multi-strategy live runner with shared market-data feeds.

Pipeline:

    shared data feed -> feature generation -> model inference -> MarketView
    -> strategy intent -> configured live venue

Each strategy is restored from a shared canonical JSONL backtest report and
selected by its live-config hash.  Live-only settings choose the model artifact,
inference device, and execution venue; strategy and market parameters remain
owned by the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import ntpath
import os, sys
import queue
import time
from numbers import Real
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional
from enum import Enum, auto
import pandas as pd
import threading

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", ".."))
from data_process import common
from data_process.utils import config_from_dict_train
from model.model_loader import ModelHandler
from trade.feed.feed_base import DataFeedBase
from trade.core.protocol import (
    ActionType,
    AccountView,
    Firm,
    MarketView,
    Observation,
    PositionDir,
    Signal,
    TradeIntent,
)
from trade.core.execution import ExecutionEvent, ExecutionReport
from trade.runner.config import BrokerConfig
from trade.monitoring.live_monitoring import (
    LiveMonitoringConfig,
    LiveMonitoringService,
    LiveStateRegistry,
    monitoring_config_from_mapping,
)
from trade.recording.prediction_trace import (
    LivePredictionTraceRecorder,
    PredictionTraceConfig,
)
from trade.recording.execution_trace import (
    ExecutionTraceConfig,
    LiveExecutionTraceRecorder,
)
from trade.notification.notify import Notify
from trade.notification.telegram_notify import TelegramNotify
from trade.venue.live.binance_data_feed import BinanceDataFeed
from trade.core.venue_base import VenueBase
from trade.core.strategy_base import StrategyBase
from trade.venue.live.ctrader.ctrader_venue import CTraderOpenApiConnection
from trade.venue.live.ctrader.ctrader_venue import CTraderVenue

SUPPORTED_BINANCE_DATA_SOURCES = {
    "binance",
    "binance_api",
    "binance_public_data",
}


class RunnerEventType(Enum):
    CLOSED_CANDLE = auto()
    DATA_CHECK = auto()


@dataclass(frozen=True)
class RunnerEvent:
    e_type: RunnerEventType
    group: FeedGroup
    timestamp_ms: int


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigBase:
    """Live-only connection data. Secrets are deliberately omitted from repr."""

    venue: str
    path: str


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigHedge:
    firm: Firm
    hedge: bool = False
    # the rest only required when hedge is true
    cost: Optional[float] = None
    challenge_type: Optional[str] = None
    stage: Optional[str] = None
    hedge_venue: Optional[str] = None
    hedge_key_path: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "firm", Firm.parse(self.firm))


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigMt5(LiveVenueConfigBase, LiveVenueConfigHedge):
    """Live-only connection data. Secrets are deliberately omitted from repr."""

    login: str
    password: str
    server: str

    profit_target: float
    max_loss: float


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigCtrader(LiveVenueConfigBase, LiveVenueConfigHedge):
    """Live-only connection data. Secrets are deliberately omitted from repr."""

    trader_login: str

    profit_target: float
    max_loss: float
    telegram_token_path: str

    def __post_init__(self) -> None:
        super().__post_init__()
        profit_target = float(self.profit_target)
        max_loss = float(self.max_loss)
        if not math.isfinite(profit_target) or profit_target <= 0:
            raise ValueError("cTrader profit_target must be positive and finite")
        if not math.isfinite(max_loss) or not 0 < max_loss < 1:
            raise ValueError("cTrader max_loss must be between zero and one")
        object.__setattr__(self, "profit_target", profit_target)
        object.__setattr__(self, "max_loss", max_loss)


SUPPORTED_VENUES = {
    "mock": LiveVenueConfigBase,
    "mt5": LiveVenueConfigMt5,
    "ctrader": LiveVenueConfigCtrader,
    "bybit": LiveVenueConfigBase,
    "binance": LiveVenueConfigBase,
    "bitget": LiveVenueConfigBase,
}


@dataclass
class LiveStrategySpec:
    strategy_id: str
    hash_id: str
    run_live: bool
    model_path: str
    device: str = "auto"
    compound: bool = True
    enable: bool = True
    base_define: common.BaseDefine = None
    train_config: Any = None
    strategy_config: Any = None
    broker_config: BrokerConfig = None
    venue_config: LiveVenueConfigBase = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_live, bool):
            raise TypeError("Live strategy run_live must be a boolean")
        if not isinstance(self.compound, bool):
            raise TypeError("Live strategy compound must be a boolean")
        if not isinstance(self.enable, bool):
            raise TypeError("Live strategy enable must be a boolean")
        venue_name = str(getattr(self.venue_config, "venue", "")).strip().casefold()
        if venue_name == "ctrader" and self.compound:
            raise ValueError("cTrader live strategies require compound=false")


@dataclass
class StrategyPipeline:
    spec: LiveStrategySpec
    model: ModelHandler
    venue: VenueBase
    strategy: StrategyBase
    feature_factory: Any
    interval_ms: int
    runner_id: str = "live-runner"
    enable: bool = True
    notifier: Notify | None = None
    notification_keys: set[str] = field(default_factory=set, repr=False)

    PROFIT_TARGET_OVERSHOOT_MULTIPLIER: ClassVar[float] = 1.02
    ROUND_TRIP_COMMISSION_SIDES: ClassVar[float] = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.enable, bool):
            raise TypeError("Strategy pipeline enable must be a boolean")

    def set_enabled(self, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise TypeError("Strategy pipeline enable must be a boolean")
        self.enable = enable
        if enable:
            self.notification_keys.clear()

    @property
    def required_bars(self) -> int:
        feature_history = int(self.feature_factory.get_global_min_history())
        model_history = max(1, int(self.model.seq_len)) * 2
        return feature_history + model_history

    def _notify_once(self, key: str, message: str) -> None:
        if key in self.notification_keys:
            return
        logger = getattr(self.venue, "logger", None) or logging.getLogger("trade.live")
        if self.notifier is None:
            logger.error("Notification unavailable | %s", message)
            return
        if self.notifier.send(message):
            self.notification_keys.add(key)
        else:
            logger.error("Notification delivery failed | %s", message)

    @staticmethod
    def _rejected_entry(intent: TradeIntent, reason: str) -> ExecutionReport:
        return ExecutionReport(
            side=("buy" if intent.target_dir == PositionDir.POSITIVE else "sell"),
            requested_quantity=float(intent.order_qty),
            submitted_quantity=0.0,
            decision_price=float(intent.price),
            decision_at_utc=intent.created_at_utc,
            bid=float("nan"),
            ask=float("nan"),
            spread_pct=float("nan"),
            status="rejected",
            reason=reason,
        )

    def _ctrader_entry_quantity(
        self,
        observation: Observation,
        intent: TradeIntent,
    ) -> tuple[float | None, str | None]:
        initial_equity = float(self.spec.broker_config.initial_equity)
        current_equity = float(observation.account.equity)
        price = float(intent.price)
        stop_loss_pct = float(intent.stop_loss_pct)
        take_profit_pct = float(intent.take_profit_pct)
        requested_quantity = float(intent.order_qty)
        config = self.spec.venue_config

        numeric_values = {
            "initial_equity": initial_equity,
            "current_equity": current_equity,
            "price": price,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "requested_quantity": requested_quantity,
        }
        invalid = [name for name, value in numeric_values.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError("cTrader entry risk validation received invalid values: " + ", ".join(invalid))

        commission_pct = float(self.spec.broker_config.commission_pct)
        if not math.isfinite(commission_pct) or commission_pct < 0:
            raise ValueError("cTrader commission_pct must be non-negative and finite")

        loss_floor = initial_equity * (1.0 - float(config.max_loss))
        profit_target_equity = initial_equity * (1.0 + float(config.profit_target))
        profit_ceiling = initial_equity * (1.0 + float(config.profit_target) * self.PROFIT_TARGET_OVERSHOOT_MULTIPLIER)
        if current_equity <= loss_floor:
            return None, "max_loss_reached"
        if current_equity >= profit_target_equity:
            return None, "profit_target_reached"

        commission_rate = max(
            0.0,
            commission_pct / 100.0,
        )
        round_trip_commission = commission_rate * self.ROUND_TRIP_COMMISSION_SIDES
        loss_per_unit = price * (stop_loss_pct + round_trip_commission)
        profit_per_unit = price * max(
            0.0,
            take_profit_pct - round_trip_commission,
        )

        max_loss_quantity = (current_equity - loss_floor) / loss_per_unit
        quantity = min(requested_quantity, max_loss_quantity)
        if profit_per_unit > 0:
            max_profit_quantity = (profit_ceiling - current_equity) / profit_per_unit
            quantity = min(quantity, max_profit_quantity)

        try:
            normalized_quantity = float(self.venue.normalize_order_quantity(quantity))
        except (TypeError, ValueError):
            return None, "entry_quantity_below_minimum"
        if not math.isfinite(normalized_quantity) or normalized_quantity <= 0 or normalized_quantity > quantity:
            return None, "entry_quantity_below_minimum"
        return normalized_quantity, None

    def _execute_intent(self, observation: Observation, intent: TradeIntent):
        if not self.enable:
            return None
        venue_name = str(getattr(self.spec.venue_config, "venue", "")).strip().casefold()
        if venue_name != "ctrader" or intent.action != ActionType.OPEN:
            return self.venue.execute_action(intent)

        original_quantity = float(intent.order_qty)
        quantity, rejection_reason = self._ctrader_entry_quantity(
            observation,
            intent,
        )
        if rejection_reason is not None:
            current_equity = float(observation.account.equity)
            initial_equity = float(self.spec.broker_config.initial_equity)
            self.set_enabled(False)
            message = (
                f"date={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | "
                f"runner_id={self.runner_id} | "
                f"strategy_id={self.spec.strategy_id} | "
                f"event={rejection_reason} | "
                f"initial_equity={initial_equity:.2f} | "
                f"current_equity={current_equity:.2f}"
            )
            self._notify_once(rejection_reason, message)
            return self._rejected_entry(intent, rejection_reason)

        assert quantity is not None
        if quantity < original_quantity:
            logger = getattr(self.venue, "logger", None) or logging.getLogger("trade.live")
            logger.info(
                "cTrader entry quantity reduced by account boundary | " "strategy=%s hash=%s symbol=%s requested=%g submitted=%g " "equity=%.2f",
                self.spec.strategy_id,
                self.spec.hash_id,
                self.spec.base_define.symbol,
                original_quantity,
                quantity,
                float(observation.account.equity),
            )
        intent.order_qty = quantity
        report = self.venue.execute_action(intent)
        if isinstance(report, ExecutionReport) and quantity < original_quantity:
            report = replace(
                report,
                requested_quantity=original_quantity,
            )
        return report


@dataclass
class FeedGroup:
    market_config: common.MarketDataSourceConfig
    interval_ms: int
    feed: DataFeedBase
    required_bars: int
    pipelines: list[StrategyPipeline]
    last_processed_candle_open_time_ms: Optional[int] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def name(self) -> str:
        return f"{self.market_config.symbol}_{self.market_config.interval}"


DATA_CHECK_TIMER_DELAY_MS = 5000


@dataclass(frozen=True)
class LiveRunnerConfiguration:
    strategies: list[LiveStrategySpec]
    runner_id: str
    run_id: str
    output_dir: str
    monitoring: LiveMonitoringConfig | None = None
    prediction_trace: PredictionTraceConfig | None = None
    execution_trace: ExecutionTraceConfig | None = None


def _resolve_data_check_timer_interval_ms(
    feed_groups: Iterable[FeedGroup],
) -> tuple[int, int]:
    """Return the shortest feed interval after validating aligned boundaries."""

    interval_entries = [
        (
            group.market_config.interval,
            common.get_interval_ms(group.market_config.interval),
        )
        for group in feed_groups
    ]
    if not interval_entries:
        raise ValueError("LiveRunner requires at least one feed group")
    invalid = [interval for interval, interval_ms in interval_entries if interval_ms <= 0]
    if invalid:
        raise ValueError(f"Invalid feed intervals: {', '.join(invalid)}")

    data_check_timer_interval_ms = min(interval_ms for _, interval_ms in interval_entries)
    data_check_timer_max_ms = max(interval_ms for _, interval_ms in interval_entries)
    incompatible = [f"{interval} ({interval_ms} ms)" for interval, interval_ms in interval_entries if interval_ms % data_check_timer_interval_ms != 0]
    if incompatible:
        raise ValueError(
            "Every feed interval must be an integer multiple of the shortest "
            f"interval ({data_check_timer_interval_ms} ms); incompatible: " + ", ".join(incompatible)
        )
    return data_check_timer_interval_ms, data_check_timer_max_ms


def _expected_closed_candle_open_time_ms(check_boundary_ms: int, interval_ms: int) -> int:
    """Return the latest candle open time that should be closed at a boundary."""

    return check_boundary_ms // interval_ms * interval_ms - interval_ms


def _format_utc_ms(timestamp_ms: int) -> str:
    """Format an epoch-millisecond timestamp for human-readable logs."""

    return datetime.fromtimestamp(
        int(timestamp_ms) / 1000,
        tz=timezone.utc,
    ).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]


def _resolve_path(config_path: str, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Path must not be empty in {config_path}")
    if os.path.isabs(raw) or ntpath.isabs(raw):
        return os.path.normpath(raw) if os.path.isabs(raw) else raw
    return os.path.normpath(os.path.join(os.path.dirname(config_path), raw))


def _iter_report_records(path: str) -> Iterable[dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Report file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def load_params_from_report(
    strategy_entries: list[LiveStrategySpec],
    report_path: str,
) -> None:
    """Find matching JSONL records and fill strategy parameter configs."""

    from trade.runner.backtest_runner import strategy_config_from_dict

    specs_by_hash: dict[str, list[LiveStrategySpec]] = {}
    for spec in strategy_entries:
        specs_by_hash.setdefault(spec.hash_id, []).append(spec)
    params_by_hash: dict[str, dict[str, Any] | None] = dict.fromkeys(specs_by_hash)

    for record in _iter_report_records(report_path):
        params = record["params"]
        hash_id = params["hash"]
        if hash_id in params_by_hash and params_by_hash[hash_id] is None:
            loaded_params = dict(params)
            market_params = loaded_params["common"]
            train_params = loaded_params["train"]
            strategy_params = loaded_params["strategy"]

            for spec in specs_by_hash[hash_id]:
                spec.base_define = common.BaseDefine(**market_params)
                spec.train_config = config_from_dict_train(train_params)
                spec.strategy_config = strategy_config_from_dict(strategy_params)
            params_by_hash[hash_id] = loaded_params
            if all(params is not None for params in params_by_hash.values()):
                break

    missing = sorted(hash_id for hash_id, params in params_by_hash.items() if params is None)
    if missing:
        missing_text = ", ".join(repr(hash_id) for hash_id in missing)
        raise KeyError(f"Hashes {missing_text} were not found in report file {report_path}")


def _venue_section(entry: Mapping[str, Any], venue_kind: str) -> dict[str, Any]:
    target = venue_kind.casefold()
    for key, value in entry.items():
        if str(key).casefold() == target and isinstance(value, Mapping):
            return dict(value)
    for key, value in entry.items():
        names = {part.strip().casefold() for part in str(key).split("/")}
        if target in names and isinstance(value, Mapping):
            return dict(value)
    raise KeyError(f"Missing {venue_kind} venue configuration object")


def _parse_venue_config(
    config_path: str,
    entry: Mapping[str, Any],
    *,
    telegram_token_path: Any = None,
) -> LiveVenueConfigBase:
    venue_name = str(entry.get("venue", "")).strip().casefold()
    if venue_name not in SUPPORTED_VENUES:
        choices = ", ".join(sorted(SUPPORTED_VENUES))
        raise ValueError(f"venue must be one of {choices}; got {venue_name!r}")
    venue_class = SUPPORTED_VENUES[venue_name]

    section = _venue_section(entry, venue_name)
    path = section.get("path")
    if venue_name != "mt5":
        path = os.path.realpath(_resolve_path(config_path, path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{venue_name} key directory not found: {path}")
    values = {field.name: section.get(field.name) for field in fields(venue_class) if field.name not in {"venue", "path"}}
    if venue_name == "ctrader":
        values["telegram_token_path"] = os.path.realpath(_resolve_path(config_path, telegram_token_path))
    return venue_class(
        venue=venue_name,
        path=path,
        **values,
    )


def _strategy_entries(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    strategies = payload.get("strategy")
    if not isinstance(strategies, Mapping):
        raise TypeError("Live configuration strategy must be an object")
    return strategies


def _validate_runner_id(value: Any) -> str:
    runner_id = str(value or "").strip() or "live-runner"
    if runner_id in {".", ".."} or "/" in runner_id or "\\" in runner_id or "\x00" in runner_id:
        raise ValueError("runner_id must be a single safe directory name without path separators")
    return runner_id


def _runner_identity_from_payload(
    payload: Mapping[str, Any],
    *,
    runner_id: str | None = None,
) -> tuple[str, str]:
    raw_monitoring = payload.get("monitoring")
    if raw_monitoring is not None and not isinstance(raw_monitoring, Mapping):
        raise TypeError("monitoring must be a JSON object")
    if isinstance(raw_monitoring, Mapping) and "runner_id" in raw_monitoring:
        raise ValueError("runner_id belongs at the configuration root, not inside monitoring")
    final_runner_id = _validate_runner_id(runner_id or os.environ.get("LIVE_RUNNER_ID") or payload.get("runner_id"))
    output_dir = os.path.abspath(
        os.path.join(
            common.PERSISTENCE_DIR,
            "live_runner",
            final_runner_id,
        )
    )
    return final_runner_id, output_dir


def _new_live_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _live_run_output_dir(runner_output_dir: str, run_id: str) -> str:
    normalized_run_id = _validate_runner_id(run_id)
    return os.path.join(runner_output_dir, normalized_run_id)


def load_live_runner_identity(
    path: str,
    *,
    runner_id: str | None = None,
) -> tuple[str, str]:
    """Read only the runner identity needed to initialize its session log."""

    config_path = os.path.abspath(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Live configuration root must be an object")
    return _runner_identity_from_payload(payload, runner_id=runner_id)


def _prediction_trace_config(
    enabled: Any,
    output_dir: str,
) -> PredictionTraceConfig | None:
    if not isinstance(enabled, bool):
        raise TypeError("prediction_trace must be a JSON boolean")
    if not enabled:
        return None
    return PredictionTraceConfig(
        output_dir=os.path.join(output_dir, "prediction_traces"),
    )


def _execution_trace_config(
    enabled: Any,
    output_dir: str,
) -> ExecutionTraceConfig | None:
    if not isinstance(enabled, bool):
        raise TypeError("execution_trace must be a JSON boolean")
    if not enabled:
        return None
    return ExecutionTraceConfig(
        output_dir=os.path.join(output_dir, "execution_traces"),
    )


def load_live_runner_configuration(
    path: str,
    *,
    publish_url: str | None = None,
    runner_id: str | None = None,
    run_id: str | None = None,
) -> LiveRunnerConfiguration:
    """Restore a live runner and scope its outputs to one run directory."""

    config_path = os.path.abspath(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Live configuration root must be an object")

    final_runner_id, runner_output_dir = _runner_identity_from_payload(
        payload,
        runner_id=runner_id,
    )
    final_run_id = run_id or _new_live_run_id()
    output_dir = _live_run_output_dir(runner_output_dir, final_run_id)

    report_path = _resolve_path(config_path, payload.get("report"))
    if not report_path.lower().endswith(".jsonl"):
        raise ValueError(f"Live configuration report must be a JSONL file: {report_path}")

    raw_strategy_entries = _strategy_entries(payload)
    telegram_token_path = payload.get("telegram_token")
    strategy_entries: list[LiveStrategySpec] = []
    for raw_id, raw_entry in raw_strategy_entries.items():
        strategy_id = str(raw_id).strip()
        entry = dict(raw_entry)
        run_live = entry["run_live"]
        compound = entry.get("compound", True)
        enable = entry.get("enable", True)
        if not isinstance(run_live, bool):
            raise TypeError(f"Live strategy {strategy_id!r} run_live must be a JSON boolean")
        if not run_live:
            continue
        if not isinstance(compound, bool):
            raise TypeError(f"Live strategy {strategy_id!r} compound must be a JSON boolean")
        if not isinstance(enable, bool):
            raise TypeError(f"Live strategy {strategy_id!r} enable must be a JSON boolean")
        hash_id = entry["hash"]
        model_path = _resolve_path(config_path, entry["model_path"])
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model artifact directory not found: {model_path}")
        strategy_entries.append(
            LiveStrategySpec(
                strategy_id=strategy_id,
                hash_id=hash_id,
                run_live=run_live,
                model_path=model_path,
                device=str(entry.get("device", "auto")),
                compound=compound,
                enable=enable,
                broker_config=BrokerConfig(**entry["broker_config"]),
                venue_config=_parse_venue_config(
                    config_path,
                    entry,
                    telegram_token_path=telegram_token_path,
                ),
            )
        )

    if not strategy_entries:
        raise ValueError("Live configuration contains no run_live strategies")

    load_params_from_report(strategy_entries, report_path)

    monitoring = monitoring_config_from_mapping(
        payload.get("monitoring"),
        publish_url=publish_url or os.environ.get("LIVE_MONITORING_PUBLISH_URL"),
        runner_id=final_runner_id,
    )
    prediction_trace = _prediction_trace_config(
        payload.get("prediction_trace", False),
        output_dir,
    )
    execution_trace = _execution_trace_config(
        payload.get("execution_trace", False),
        output_dir,
    )
    return LiveRunnerConfiguration(
        strategies=strategy_entries,
        runner_id=final_runner_id,
        run_id=final_run_id,
        output_dir=output_dir,
        monitoring=monitoring,
        prediction_trace=prediction_trace,
        execution_trace=execution_trace,
    )


def load_live_strategy_specs(path: str) -> list[LiveStrategySpec]:
    """Restore live strategy specifications without starting monitoring."""

    return load_live_runner_configuration(path).strategies


def _optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _prepare_market_frame(
    frame: pd.DataFrame,
    base_define: common.BaseDefine,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Live data feed returned no closed candles")
    prepared = frame.copy()
    prepared["open_time_ms_utc"] = pd.to_numeric(prepared["open_time_ms_utc"], errors="raise").astype("int64")
    prepared["close_time_ms_utc"] = pd.to_numeric(prepared["close_time_ms_utc"], errors="raise").astype("int64")
    prepared = common.attch_open_time_sn(base_define, prepared)
    prepared = common.calculate_thresholds(prepared, base_define)
    return prepared


def _market_view(
    predicted_frame: pd.DataFrame,
) -> MarketView:
    if predicted_frame is None or predicted_frame.empty:
        raise ValueError("Model returned no live predictions")

    row = predicted_frame.iloc[-1]
    raw_signal = _optional_float(row.get("pred"))
    signal = Signal.INVALID if raw_signal is None else Signal(int(raw_signal))

    close = _optional_float(row.get("close"))
    if close is None or close <= 0:
        raise ValueError("Latest closed candle has no valid close price")

    return MarketView(
        price=close,
        open=_optional_float(row.get("open")),
        high=_optional_float(row.get("high")),
        low=_optional_float(row.get("low")),
        close=close,
        signal=signal,
        pred_prob=_optional_float(row.get("pred_prob"), 0.0) or 0.0,
        expected_vol=_optional_float(row.get("expected_vol")),
        bars_to_close=_optional_float(
            row.get("bars_to_close"),
            math.inf,
        ),
    )


class LiveRunner:
    """Coordinate multiple model strategies while fetching each feed only once."""

    CTRADER_INITIAL_BALANCE_TOLERANCE = 0.20
    EXECUTION_RECONCILIATION_LOOKBACK = timedelta(days=7)

    def __init__(
        self,
        specs: list[LiveStrategySpec],
        *,
        logger: Optional[logging.Logger] = None,
        feed_factory: Optional[Callable[[common.MarketDataSourceConfig, int], DataFeedBase]] = None,
        venue_factory: Optional[Callable[[LiveStrategySpec, logging.Logger], Any]] = None,
        notify_factory: Optional[Callable[[LiveStrategySpec, logging.Logger], Notify | None]] = None,
        prediction_callback: Optional[Callable[[StrategyPipeline, int, pd.Series], None]] = None,
        runner_id: str | None = None,
        output_dir: str | None = None,
        monitoring_config: LiveMonitoringConfig | None = None,
        prediction_trace_config: PredictionTraceConfig | None = None,
        execution_trace_config: ExecutionTraceConfig | None = None,
    ):
        if not specs:
            raise ValueError("LiveRunner requires at least one strategy")
        self.logger = logger or logging.getLogger("trade.live")
        self._feed_factory = feed_factory or self._create_feed
        self._ctrader_connections: dict[str, CTraderOpenApiConnection] = {}
        self._ctrader_connection_path: str | None = None
        self._venue_factory = venue_factory or self._create_venue
        self._notify_factory = notify_factory or self._create_notifier
        self._prediction_callback = prediction_callback
        self.runner_id = _validate_runner_id(runner_id or (monitoring_config.runner_id if monitoring_config is not None else None))
        self.output_dir = os.path.abspath(output_dir) if output_dir is not None else None
        self._monitoring_config = monitoring_config
        self._prediction_trace_config = prediction_trace_config
        self._execution_trace_config = execution_trace_config
        self._live_registry: LiveStateRegistry | None = None
        self._monitoring_service: LiveMonitoringService | None = None
        self._prediction_trace: LivePredictionTraceRecorder | None = None
        self._execution_trace: LiveExecutionTraceRecorder | None = None
        self._prediction_trace_failed = False
        self._initialized = False
        self._closed = False
        self.strategy_pipelines: list[StrategyPipeline] = []
        self.feed_groups: list[FeedGroup] = []
        self._events: queue.Queue[RunnerEvent] = queue.Queue()
        self.data_check_timer = None
        self._data_check_timer_interval_ms = 0
        self._next_min_expect_candle_open_time = 0
        self._next_data_check_timer_time_ms = 0
        self._data_check_timer_max_cycle_ms = 0
        self._data_check_timer_cycle_count = 0
        try:
            self._build(specs)
            if self._prediction_trace_config is not None:
                self._prediction_trace = LivePredictionTraceRecorder(
                    self._prediction_trace_config,
                    self.feed_groups,
                )
            if self._execution_trace_config is not None:
                self._execution_trace = LiveExecutionTraceRecorder(
                    self._execution_trace_config,
                    runner_id=self.runner_id,
                    run_id=(os.path.basename(self.output_dir) if self.output_dir is not None else None),
                    logger=self.logger,
                )
                for pipeline in self.strategy_pipelines:
                    register = getattr(
                        pipeline.venue,
                        "set_execution_event_callback",
                        None,
                    )
                    if callable(register):
                        register(
                            lambda event, current=pipeline: self._record_execution_event(
                                current,
                                event,
                            )
                        )
                self.logger.info(
                    "Live execution trace started | files=%s",
                    self._execution_trace.paths,
                )
                self._reconcile_execution_events()
            self._live_registry = LiveStateRegistry(self.strategy_pipelines)
            if self._monitoring_config is not None:
                self._monitoring_service = LiveMonitoringService(
                    self._monitoring_config,
                    self._live_registry,
                    logger=self.logger,
                )
        except Exception:
            self.close()
            raise

    @classmethod
    def from_config(
        cls,
        path: str,
        logger: Optional[logging.Logger] = None,
        *,
        publish_url: str | None = None,
        runner_id: str | None = None,
    ) -> "LiveRunner":
        configuration = load_live_runner_configuration(
            path,
            publish_url=publish_url,
            runner_id=runner_id,
        )
        return cls.from_configuration(configuration, logger=logger)

    @classmethod
    def from_configuration(
        cls,
        configuration: LiveRunnerConfiguration,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> "LiveRunner":
        return cls(
            configuration.strategies,
            logger=logger,
            runner_id=configuration.runner_id,
            output_dir=configuration.output_dir,
            monitoring_config=configuration.monitoring,
            prediction_trace_config=configuration.prediction_trace,
            execution_trace_config=configuration.execution_trace,
        )

    @staticmethod
    def _create_feed(
        market_config: common.MarketDataSourceConfig,
        max_len: int,
    ):
        if market_config.data_source.casefold() not in SUPPORTED_BINANCE_DATA_SOURCES:
            raise ValueError(f"Unsupported live data source {market_config.data_source!r}; " "only Binance public kline feeds are currently implemented")

        return BinanceDataFeed(
            market_config.symbol,
            market_config.interval,
            market_config.trading_type,
            max_len=max_len,
        )

    @staticmethod
    def _create_feature_generator(spec: LiveStrategySpec):
        return common.FeatureFactory(
            common.get_interval_ms(spec.base_define.interval),
            feature_conf_list=spec.train_config.feature_conf_list,
        )

    @staticmethod
    def _load_model(spec: LiveStrategySpec):
        return ModelHandler(tarin_out_path=spec.model_path, device=spec.device)

    @staticmethod
    def _create_notifier(
        spec: LiveStrategySpec,
        logger: logging.Logger,
    ) -> Notify | None:
        config = spec.venue_config
        if str(getattr(config, "venue", "")).strip().casefold() != "ctrader":
            return None
        return TelegramNotify(
            config.telegram_token_path,
            logger=logger,
        )

    def _create_venue(self, spec: LiveStrategySpec, logger: logging.Logger):
        config = spec.venue_config
        if config.venue == "mock":
            from trade.venue.mock_venue import MockVenue

            return MockVenue(initial_equity=spec.broker_config.initial_equity)
        if config.venue == "mt5":
            if not spec.strategy_id.isascii() or not spec.strategy_id.isdigit():
                raise ValueError("MT5 strategy_id must contain ASCII digits only; " f"got {spec.strategy_id!r}")
            from trade.venue.live.mt5.mt5_venue import MT5Venue

            return MT5Venue(
                config.path,
                spec.base_define.symbol,
                int(spec.strategy_id),
                logger=logger,
                login=config.login,
                password=config.password,
                server=config.server,
                firm=config.firm,
            )
        if config.venue == "bybit":
            from trade.venue.live.bybit.bybit_venue import BybitVenue

            return BybitVenue(
                config.path,
                spec.base_define.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
            )
        if config.venue == "ctrader":
            connection_path = os.path.realpath(config.path)
            if self._ctrader_connection_path is not None and connection_path != self._ctrader_connection_path:
                raise ValueError(
                    "All cTrader venues must use the same credential path; " f"expected {self._ctrader_connection_path!r}, " f"got {connection_path!r}"
                )
            self._ctrader_connection_path = connection_path

            discovery_connection = self._ctrader_connection(
                connection_path,
                "live",
                logger,
            )
            account_id, environment = discovery_connection.resolve_account(config.trader_login)
            logger.info(
                "cTrader account route resolved | trader_login=%s account=%s environment=%s",
                config.trader_login,
                account_id,
                environment,
            )
            account_connection = self._ctrader_connection(
                connection_path,
                environment,
                logger,
            )
            ctrader_venue = CTraderVenue(
                connection_path,
                spec.base_define.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
                trader_login=config.trader_login,
                environment=environment,
                api=account_connection,
                firm=config.firm,
            )
            return ctrader_venue

        if config.venue == "binance":
            from trade.venue.live.binance.binance_venue import BinanceVenue

            return BinanceVenue(
                config.path,
                spec.base_define.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
            )
        if config.venue == "bitget":
            from trade.venue.live.bitget.bitget_venue import BitgetVenue

            return BitgetVenue(
                config.path,
                spec.base_define.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
            )
        raise ValueError(f"Unsupported venue: {config.venue}")

    def _ctrader_connection(
        self,
        connection_path: str,
        environment: str,
        logger: logging.Logger,
    ) -> CTraderOpenApiConnection:
        connection = self._ctrader_connections.get(environment)
        if connection is None:
            connection = CTraderOpenApiConnection(
                connection_path,
                environment=environment,
                logger=logger,
            )
            self._ctrader_connections[environment] = connection
        return connection

    @staticmethod
    def _create_strategy(spec: LiveStrategySpec, venue: VenueBase):
        from trade.strategy.strategy_bbm import BbmSignalStrategy, BbmStrategyConfig

        equity = float(venue.get_account_equity())
        if equity <= 0:
            raise RuntimeError("Venue returned invalid account equity for " f"{spec.strategy_id} ({spec.hash_id})")
        leverage = float(spec.broker_config.leverage)
        data_interval_ms = common.get_interval_ms(spec.base_define.interval)

        held_bars = 0
        position = venue.get_current_state()
        if position.dir in {PositionDir.POSITIVE, PositionDir.NEGATIVE}:
            open_time = venue.get_last_position_open_time()
            if open_time is None:
                logger = logging.getLogger("trade")
                logger.warning(
                    "Open position has no opening timestamp | strategy=%s hash=%s",
                    spec.strategy_id,
                    spec.hash_id,
                )
            else:
                now = datetime.now(timezone.utc)
                if open_time.tzinfo is None:
                    open_time = open_time.replace(tzinfo=timezone.utc)
                else:
                    open_time = open_time.astimezone(timezone.utc)
                elapsed_ms = max(
                    0,
                    int((now - open_time).total_seconds() * 1000),
                )
                held_bars = elapsed_ms // data_interval_ms
        if isinstance(spec.strategy_config, BbmStrategyConfig):
            effective_config = replace(
                spec.strategy_config,
                compound=spec.compound,
            )
            spec.strategy_config = effective_config
            return BbmSignalStrategy(
                config=effective_config,
                init_equity=equity,
                data_interval_ms=data_interval_ms,
                exist_hold_bars=int(held_bars),
                leverage=leverage,
            )
        raise TypeError("Live runner currently supports BbmStrategyConfig, got " f"{type(spec.strategy_config).__name__}")

    @classmethod
    def _validate_ctrader_initial_balance(
        cls,
        spec: LiveStrategySpec,
        venue: VenueBase,
    ) -> None:
        venue_name = str(getattr(spec.venue_config, "venue", "")).strip().casefold()
        if venue_name != "ctrader":
            return

        initial_equity = float(spec.broker_config.initial_equity)
        if not math.isfinite(initial_equity) or initial_equity <= 0:
            raise ValueError("cTrader broker_config.initial_equity must be positive and finite")
        actual_balance = float(venue.get_dashboard_balance().balance)
        if not math.isfinite(actual_balance) or actual_balance <= 0:
            raise RuntimeError("cTrader returned an invalid account balance")

        lower_balance = initial_equity * (1.0 - cls.CTRADER_INITIAL_BALANCE_TOLERANCE)
        upper_balance = initial_equity * (1.0 + cls.CTRADER_INITIAL_BALANCE_TOLERANCE)
        if actual_balance < lower_balance or actual_balance > upper_balance:
            raise ValueError(
                "cTrader account balance is outside the configured initial_equity "
                "tolerance: "
                f"strategy={spec.strategy_id!r}, initial_equity={initial_equity:g}, "
                f"actual_balance={actual_balance:g}, "
                f"allowed_range=[{lower_balance:g}, {upper_balance:g}]"
            )

    def _build(self, specs: list[LiveStrategySpec]) -> None:
        grouped_pipelines: list[tuple[common.MarketDataSourceConfig, list[StrategyPipeline]]] = []

        for spec in specs:
            if not spec.run_live:
                self.logger.info(
                    "Non-live strategy skipped | id=%s hash=%s",
                    spec.strategy_id,
                    spec.hash_id,
                )
                continue

            model = self._load_model(spec)
            feature_generator = self._create_feature_generator(spec)
            market_config = common.MarketDataSourceConfig(
                **{field.name: getattr(spec.base_define, field.name) for field in fields(common.MarketDataSourceConfig)}
            )
            feed_entry = next(
                filter(
                    lambda item: item[0] == market_config,
                    grouped_pipelines,
                ),
                None,
            )
            if feed_entry is None:
                feed_pipelines: list[StrategyPipeline] = []
                grouped_pipelines.append((market_config, feed_pipelines))
            else:
                feed_pipelines = feed_entry[1]
            venue = self._venue_factory(spec, self.logger)
            try:
                self._validate_ctrader_initial_balance(spec, venue)
                notifier = self._notify_factory(spec, self.logger)
                strategy = self._create_strategy(spec, venue)
                self.logger.info(
                    "Strategy created | id=%s hash=%s venue=%s",
                    spec.strategy_id,
                    spec.hash_id,
                    type(venue).__name__,
                )
            except Exception:
                shutdown = getattr(venue, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:
                        self.logger.exception(
                            "Venue cleanup failed during construction: %s",
                            spec.strategy_id,
                        )
                raise

            pipeline = StrategyPipeline(
                spec=spec,
                model=model,
                venue=venue,
                strategy=strategy,
                feature_factory=feature_generator,
                interval_ms=common.get_interval_ms(spec.base_define.interval),
                runner_id=self.runner_id,
                enable=spec.enable,
                notifier=notifier,
            )
            self.strategy_pipelines.append(pipeline)
            feed_pipelines.append(pipeline)

        if not self.strategy_pipelines:
            raise ValueError("LiveRunner requires at least one run_live strategy")

        for market_config, pipelines in grouped_pipelines:
            required_bars = max(pipeline.required_bars for pipeline in pipelines)
            interval_ms = common.get_interval_ms(market_config.interval)

            feed = self._feed_factory(market_config, required_bars + 500)
            self.feed_groups.append(
                FeedGroup(
                    market_config=market_config,
                    interval_ms=interval_ms,
                    feed=feed,
                    required_bars=required_bars,
                    pipelines=pipelines,
                )
            )

        self.logger.info(
            "Live runner built | strategies=%d shared_feeds=%d",
            len(self.strategy_pipelines),
            len(self.feed_groups),
        )

    def set_strategy_enabled(self, strategy_id: str, enable: bool) -> None:
        """Enable or disable one constructed strategy while the runner is active."""

        if not isinstance(enable, bool):
            raise TypeError("Strategy enable must be a boolean")
        matches = [pipeline for pipeline in self.strategy_pipelines if pipeline.spec.strategy_id == strategy_id]
        if not matches:
            raise KeyError(f"Unknown live strategy ID: {strategy_id!r}")
        if len(matches) > 1:
            raise ValueError(f"Duplicate live strategy ID: {strategy_id!r}")
        matches[0].set_enabled(enable)
        self.logger.info(
            "Live strategy runtime state changed | strategy_id=%s enable=%s",
            strategy_id,
            enable,
        )

    def _start_data_check_timer(self):
        if self._closed:
            return
        now_ms = int(time.time() * 1000)
        self._next_min_expect_candle_open_time = (now_ms // self._data_check_timer_interval_ms) * self._data_check_timer_interval_ms
        self._next_data_check_timer_time_ms = self._next_min_expect_candle_open_time + self._data_check_timer_interval_ms + DATA_CHECK_TIMER_DELAY_MS
        delay_seconds = (self._next_data_check_timer_time_ms - now_ms) / 1000.0
        self.logger.debug(
            "DATA_CHECK timer scheduled | now_utc=%s shortest_interval_ms=%d " "expected_shortest_open_time_utc=%s fire_time_utc=%s " "delay_seconds=%.3f",
            _format_utc_ms(now_ms),
            self._data_check_timer_interval_ms,
            _format_utc_ms(self._next_min_expect_candle_open_time),
            _format_utc_ms(self._next_data_check_timer_time_ms),
            delay_seconds,
        )
        self.data_check_timer = threading.Timer(delay_seconds, self._data_check_timer_handler)
        self.data_check_timer.daemon = True
        self.data_check_timer.start()

    def _data_check_timer_handler(self):
        if self._closed:
            return
        now_ms = int(time.time() * 1000)
        drift_ms = now_ms - self._next_data_check_timer_time_ms
        if abs(drift_ms) > 100:
            self.logger.warning(
                "DATA_CHECK timer drift too large | drift_ms=%d " "scheduled_utc=%s actual_utc=%s",
                drift_ms,
                _format_utc_ms(self._next_data_check_timer_time_ms),
                _format_utc_ms(now_ms),
            )
        check_boundary_ms = self._next_min_expect_candle_open_time + self._data_check_timer_interval_ms
        self.logger.debug(
            "DATA_CHECK timer fired | actual_utc=%s scheduled_utc=%s " "drift_ms=%d check_boundary_utc=%s feed_groups=%d",
            _format_utc_ms(now_ms),
            _format_utc_ms(self._next_data_check_timer_time_ms),
            drift_ms,
            _format_utc_ms(check_boundary_ms),
            len(self.feed_groups),
        )
        for group in self.feed_groups:
            with group.lock:
                last_processed_candle_open_time_ms = group.last_processed_candle_open_time_ms
            expected_open_time_ms = _expected_closed_candle_open_time_ms(
                check_boundary_ms,
                group.interval_ms,
            )
            if last_processed_candle_open_time_ms is None:
                self.logger.debug(
                    "DATA_CHECK group skipped before initialization | symbol=%s " "interval=%s expected_open_time_utc=%s",
                    group.market_config.symbol,
                    group.market_config.interval,
                    _format_utc_ms(expected_open_time_ms),
                )
                continue
            lag_ms = expected_open_time_ms - last_processed_candle_open_time_ms
            missing = lag_ms > 0
            self.logger.debug(
                "DATA_CHECK group evaluated | symbol=%s interval=%s interval_ms=%d "
                "check_boundary_utc=%s expected_open_time_utc=%s "
                "last_processed_open_time_utc=%s lag_ms=%d missing=%s",
                group.market_config.symbol,
                group.market_config.interval,
                group.interval_ms,
                _format_utc_ms(check_boundary_ms),
                _format_utc_ms(expected_open_time_ms),
                _format_utc_ms(last_processed_candle_open_time_ms),
                lag_ms,
                missing,
            )
            if missing:
                self.logger.warning(
                    "Candle missing at DATA_CHECK | symbol=%s interval=%s " "expected_open_time_utc=%s last_processed_open_time_utc=%s " "lag_ms=%d",
                    group.market_config.symbol,
                    group.market_config.interval,
                    _format_utc_ms(expected_open_time_ms),
                    _format_utc_ms(last_processed_candle_open_time_ms),
                    lag_ms,
                )
                self._events.put(
                    RunnerEvent(
                        e_type=RunnerEventType.DATA_CHECK,
                        group=group,
                        timestamp_ms=expected_open_time_ms,
                    )
                )
        self._start_data_check_timer()

    def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        if self._initialized:
            return

        self._data_check_timer_interval_ms, self._data_check_timer_max_cycle_ms = _resolve_data_check_timer_interval_ms(self.feed_groups)

        for group in self.feed_groups:
            group.interval_ms = common.get_interval_ms(group.market_config.interval)
            group.feed.initialize_cache(group.required_bars, group.interval_ms)
            initial = group.feed.get_latest_data()
            if initial is None or initial.empty:
                raise RuntimeError(f"Failed to warm live feed: {group.market_config}")
            group.last_processed_candle_open_time_ms = int(initial.iloc[-1]["open_time_ms_utc"])
            self.logger.info(
                "Live feed ready | source=%s symbol=%s interval=%s bars=%d",
                group.market_config.data_source,
                group.market_config.symbol,
                group.market_config.interval,
                len(initial),
            )

        if self._prediction_trace is not None:
            for group in self.feed_groups:
                warmup = group.feed.get_latest_data()
                if warmup is None or warmup.empty:
                    raise RuntimeError("Live feed has no warm-up data for prediction trace")
                self._record_prediction_trace(
                    "record_warmup",
                    group,
                    warmup,
                )
            if self._prediction_trace is not None:
                self.logger.info(
                    "Live prediction trace started | files=%s",
                    self._prediction_trace.paths,
                )

        for group in self.feed_groups:
            group.feed.start(
                lambda open_time_ms, target=group: self._events.put(
                    RunnerEvent(
                        e_type=RunnerEventType.CLOSED_CANDLE,
                        group=target,
                        timestamp_ms=open_time_ms,
                    )
                )
            )
        self._start_data_check_timer()
        self._initialized = True
        if self._monitoring_service is not None:
            self._monitoring_service.start()
        self.logger.info(
            "WebSocket feeds started | data_check_timer_interval_ms=%d delay_ms=%d",
            self._data_check_timer_interval_ms,
            DATA_CHECK_TIMER_DELAY_MS,
        )

    def _record_prediction_trace(self, method: str, *args) -> None:
        recorder = self._prediction_trace
        if recorder is None or self._prediction_trace_failed:
            return
        try:
            getattr(recorder, method)(*args)
        except Exception:
            self._prediction_trace_failed = True
            self.logger.exception("Live prediction trace disabled after a write failure")
            try:
                recorder.close()
            except Exception:
                self.logger.exception("Live prediction trace cleanup failed")
            self._prediction_trace = None

    def _predict(self, pipeline: StrategyPipeline, features: pd.DataFrame) -> pd.DataFrame:
        sequence_length = int(pipeline.model.seq_len)
        if len(features) < sequence_length:
            raise RuntimeError(f"Insufficient live features for model window: " f"need={sequence_length}, actual={len(features)}")
        latest_window = features.iloc[-sequence_length:].copy()
        predicted, _ = pipeline.model.predict(
            latest_window,
            kline_interval_ms=pipeline.interval_ms,
            is_live=True,
            batch_size=1,
            diff_thresh=None,
        )
        return predicted

    def _dispatch_invalid_to_group(self, group: FeedGroup, candle_open_time_ms: int):
        try:
            frame = group.feed.get_latest_data()
        except Exception:
            frame = None
            self.logger.exception(
                "Failed to read cached data for INVALID dispatch | " "symbol=%s interval=%s",
                group.market_config.symbol,
                group.market_config.interval,
            )
        market = self._invalid_market_view(frame)
        candle_open_time_utc = pd.Timestamp(
            candle_open_time_ms,
            unit="ms",
            tz="UTC",
        ).to_pydatetime()
        for pipeline in group.pipelines:
            if not bool(getattr(pipeline, "enable", True)):
                continue
            try:
                observation = Observation(
                    market=market,
                    position=pipeline.venue.get_current_state(),
                    account=AccountView(equity=float(pipeline.venue.get_account_equity())),
                    candle_open_time_utc=candle_open_time_utc,
                    daily_reset_date=pipeline.venue.get_daily_reset_date(candle_open_time_utc),
                )
                intent = pipeline.strategy.process(observation)
                execution_report = pipeline._execute_intent(observation, intent)
                self._record_execution(pipeline, execution_report)
                self._record_live_cycle(
                    pipeline,
                    pd.Series({"pred": Signal.INVALID.value}),
                    market,
                    intent,
                    candle_open_time_utc,
                )
                self.logger.warning(
                    "Invalid candle processed | id=%s hash=%s symbol=%s " "open_time_utc=%s action=%s",
                    pipeline.spec.strategy_id,
                    pipeline.spec.hash_id,
                    pipeline.spec.base_define.symbol,
                    _format_utc_ms(candle_open_time_ms),
                    intent.action.value,
                )
            except Exception:
                self.logger.exception(
                    "Invalid signal dispatch failed | id=%s hash=%s symbol=%s",
                    pipeline.spec.strategy_id,
                    pipeline.spec.hash_id,
                    pipeline.spec.base_define.symbol,
                )

    def _dispatch(
        self,
        pipeline: StrategyPipeline,
        market: MarketView,
        candle_open_time_utc: datetime,
    ) -> TradeIntent:
        observation = Observation(
            market=market,
            position=pipeline.venue.get_current_state(),
            account=AccountView(equity=float(pipeline.venue.get_account_equity())),
            candle_open_time_utc=candle_open_time_utc,
            daily_reset_date=pipeline.venue.get_daily_reset_date(candle_open_time_utc),
        )
        intent = pipeline.strategy.process(observation)
        execution_report = pipeline._execute_intent(observation, intent)
        self._record_execution(pipeline, execution_report)
        self.logger.info(
            "Strategy processed | id=%s hash=%s symbol=%s signal=%s action=%s",
            pipeline.spec.strategy_id,
            pipeline.spec.hash_id,
            pipeline.spec.base_define.symbol,
            market.signal.name,
            intent.action.value,
        )
        return intent

    def _record_execution(
        self,
        pipeline: StrategyPipeline,
        report: Any,
    ) -> None:
        recorder = getattr(self, "_execution_trace", None)
        if recorder is None or not isinstance(report, ExecutionReport):
            return
        try:
            account_id = pipeline.venue.get_execution_account_id()
            venue_symbol = pipeline.venue.get_execution_symbol()
            recorder.record(
                report,
                strategy_id=pipeline.spec.strategy_id,
                strategy_hash=pipeline.spec.hash_id,
                venue=type(pipeline.venue).__name__,
                account_id=account_id,
                strategy_symbol=pipeline.spec.base_define.symbol,
                venue_symbol=venue_symbol,
            )
        except Exception:
            self.logger.exception(
                "Live execution trace enqueue failed | strategy=%s",
                pipeline.spec.strategy_id,
            )
        finally:
            activate = getattr(
                pipeline.venue,
                "activate_execution_updates",
                None,
            )
            if callable(activate) and float(report.submitted_quantity or 0.0) > 0:
                activate(report.execution_id)

    def _record_execution_event(
        self,
        pipeline: StrategyPipeline,
        event: ExecutionEvent,
    ) -> None:
        recorder = getattr(self, "_execution_trace", None)
        if recorder is None:
            return
        try:
            recorder.record_event(
                event,
                strategy_id=pipeline.spec.strategy_id,
                strategy_hash=pipeline.spec.hash_id,
                venue=type(pipeline.venue).__name__,
                account_id=pipeline.venue.get_execution_account_id(),
                strategy_symbol=pipeline.spec.base_define.symbol,
                venue_symbol=pipeline.venue.get_execution_symbol(),
            )
        except Exception:
            self.logger.exception(
                "Live execution event trace enqueue failed | strategy=%s",
                pipeline.spec.strategy_id,
            )

    def _reconcile_execution_events(self) -> None:
        since_utc = datetime.now(timezone.utc) - self.EXECUTION_RECONCILIATION_LOOKBACK
        for pipeline in self.strategy_pipelines:
            reconcile = getattr(
                pipeline.venue,
                "reconcile_execution_events",
                None,
            )
            if not callable(reconcile):
                continue
            try:
                count = int(reconcile(since_utc) or 0)
                if count:
                    self.logger.info(
                        "Live execution history reconciled | strategy=%s " "events=%s since_utc=%s",
                        pipeline.spec.strategy_id,
                        count,
                        since_utc.isoformat(),
                    )
            except Exception:
                self.logger.exception(
                    "Live execution history reconciliation failed | strategy=%s",
                    pipeline.spec.strategy_id,
                )

    def _record_live_cycle(
        self,
        pipeline: StrategyPipeline,
        predicted_row: pd.Series,
        market: MarketView,
        intent: TradeIntent,
        updated_at: datetime,
    ) -> None:
        """Update UI-only memory without affecting the trading path."""

        registry = getattr(self, "_live_registry", None)
        if registry is None:
            return
        try:
            registry.record_cycle(
                pipeline,
                predicted_row,
                market,
                intent,
                updated_at,
            )
        except Exception:
            self.logger.exception(
                "Live UI cycle recording failed | strategy=%s",
                pipeline.spec.strategy_id,
            )

    @staticmethod
    def _invalid_market_view(frame: Optional[pd.DataFrame]) -> MarketView:
        row: Mapping[str, Any] = {}
        if frame is not None and not frame.empty:
            row = frame.iloc[-1]
        last_close = _optional_float(row.get("close"), 0.0) or 0.0
        return MarketView(
            price=last_close,
            open=_optional_float(row.get("open")),
            high=_optional_float(row.get("high")),
            low=_optional_float(row.get("low")),
            close=last_close,
            signal=Signal.INVALID,
            pred_prob=1.0,
            expected_vol=_optional_float(row.get("expected_vol")),
            bars_to_close=_optional_float(
                row.get("bars_to_close"),
                math.inf,
            ),
        )

    def _process_closed_candle(
        self,
        group: FeedGroup,
        last_processed_candle_open_time_ms,
        candle_open_time_ms: int,
    ) -> None:
        expected_open_time_ms = last_processed_candle_open_time_ms + group.interval_ms
        if candle_open_time_ms > expected_open_time_ms:
            missed_count = (candle_open_time_ms - expected_open_time_ms) // group.interval_ms
            self.logger.warning(
                "Closed-kline gap detected | symbol=%s interval=%s " "miss_count=%d last_processed_open_time_utc=%s " "new_candle_open_time_utc=%s",
                group.market_config.symbol,
                group.market_config.interval,
                missed_count,
                _format_utc_ms(last_processed_candle_open_time_ms),
                _format_utc_ms(candle_open_time_ms),
            )

        frame = group.feed.get_latest_data()
        if frame is None or frame.empty:
            self.logger.error(
                "Closed-kline event has no cached data | symbol=%s interval=%s " "candle_open_time_utc=%s",
                group.market_config.symbol,
                group.market_config.interval,
                _format_utc_ms(candle_open_time_ms),
            )
            self._dispatch_invalid_to_group(group, candle_open_time_ms)
        else:
            if frame.empty or int(frame.iloc[-1]["open_time_ms_utc"]) != candle_open_time_ms:
                self.logger.error(
                    "Closed-kline event is absent from cache | symbol=%s interval=%s " "candle_open_time_utc=%s",
                    group.market_config.symbol,
                    group.market_config.interval,
                    _format_utc_ms(candle_open_time_ms),
                )
                self._dispatch_invalid_to_group(group, candle_open_time_ms)
            else:
                trace_predictions: dict[str, pd.Series] = {}
                for pipeline in group.pipelines:
                    if not bool(getattr(pipeline, "enable", True)):
                        continue
                    try:
                        prepared = _prepare_market_frame(frame, pipeline.spec.base_define)
                        features = pipeline.feature_factory.generate(prepared)
                        predicted = self._predict(pipeline, features)
                        latest_prediction = predicted.iloc[-1]
                        trace_predictions[pipeline.spec.strategy_id] = latest_prediction.copy()
                        if self._prediction_callback is not None:
                            self._prediction_callback(pipeline, candle_open_time_ms, latest_prediction.copy())
                        market = _market_view(predicted)
                        candle_open_time_utc = pd.Timestamp(
                            candle_open_time_ms,
                            unit="ms",
                            tz="UTC",
                        ).to_pydatetime()
                        intent = self._dispatch(
                            pipeline,
                            market,
                            candle_open_time_utc,
                        )
                        self._record_live_cycle(
                            pipeline,
                            latest_prediction,
                            market,
                            intent,
                            candle_open_time_utc,
                        )
                    except Exception:
                        self.logger.exception(
                            "Strategy cycle failed; dispatching INVALID | " "id=%s hash=%s symbol=%s",
                            pipeline.spec.strategy_id,
                            pipeline.spec.hash_id,
                            pipeline.spec.base_define.symbol,
                        )
                        try:
                            market = self._invalid_market_view(frame)
                            candle_open_time_utc = pd.Timestamp(
                                candle_open_time_ms,
                                unit="ms",
                                tz="UTC",
                            ).to_pydatetime()
                            intent = self._dispatch(
                                pipeline,
                                market,
                                candle_open_time_utc,
                            )
                            self._record_live_cycle(
                                pipeline,
                                pd.Series({"pred": Signal.INVALID.value}),
                                market,
                                intent,
                                candle_open_time_utc,
                            )
                        except Exception:
                            self.logger.exception(
                                "Fallback INVALID dispatch failed | id=%s hash=%s symbol=%s",
                                pipeline.spec.strategy_id,
                                pipeline.spec.hash_id,
                                pipeline.spec.base_define.symbol,
                            )
                self._record_prediction_trace(
                    "record_live",
                    group,
                    frame.iloc[-1].copy(),
                    trace_predictions,
                )

    def _process_event(self, event: RunnerEvent) -> bool:
        with event.group.lock:
            last_processed_candle_open_time_ms = event.group.last_processed_candle_open_time_ms
            if last_processed_candle_open_time_ms is not None and event.timestamp_ms <= last_processed_candle_open_time_ms:
                self.logger.warning(
                    "Stale runner event ignored | type=%s symbol=%s interval=%s " "last_processed_open_time_utc=%s received_open_time_utc=%s",
                    event.e_type.name,
                    event.group.market_config.symbol,
                    event.group.market_config.interval,
                    _format_utc_ms(last_processed_candle_open_time_ms),
                    _format_utc_ms(event.timestamp_ms),
                )
                return False
            event.group.last_processed_candle_open_time_ms = event.timestamp_ms

        if event.e_type == RunnerEventType.CLOSED_CANDLE:
            self.logger.debug("")
            self._process_closed_candle(
                event.group,
                last_processed_candle_open_time_ms,
                event.timestamp_ms,
            )
        elif event.e_type == RunnerEventType.DATA_CHECK:
            self._dispatch_invalid_to_group(
                event.group,
                event.timestamp_ms,
            )
        else:
            raise ValueError(f"Unsupported runner event type: {event.e_type!r}")
        return True

    def process_pending_events(self) -> int:
        """Process every event currently queued without blocking."""
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        processed_count = 0
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return processed_count
            if self._process_event(event):
                processed_count += 1

    def run_forever(self) -> None:
        try:
            self.initialize()
            self.logger.info("Live runner started")
            while True:
                if self._closed:
                    return
                try:
                    event = self._events.get(
                        timeout=(self._data_check_timer_max_cycle_ms // 1000 + 1),
                    )
                    self.logger.info(
                        "Runner event received | group=%s type=%s time_utc=%s",
                        event.group.name,
                        event.e_type.name,
                        _format_utc_ms(event.timestamp_ms),
                    )
                    self._process_event(event)
                except queue.Empty:
                    pass

        except KeyboardInterrupt:
            self.logger.info("Live runner stopped by user")
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.data_check_timer is not None:
            self.data_check_timer.cancel()
            self.data_check_timer = None
        if getattr(self, "_monitoring_service", None) is not None:
            try:
                self._monitoring_service.stop()
            except Exception:
                self.logger.exception("Live monitoring shutdown failed")
            self._monitoring_service = None
        if getattr(self, "_prediction_trace", None) is not None:
            try:
                self._prediction_trace.close()
            except Exception:
                self.logger.exception("Live prediction trace shutdown failed")
            self._prediction_trace = None
        for pipeline in self.strategy_pipelines:
            try:
                pipeline.strategy.finalize()
            except Exception:
                self.logger.exception(
                    "Strategy finalization failed: %s",
                    pipeline.spec.strategy_id,
                )

        closed = set()
        for pipeline in self.strategy_pipelines:
            venue = pipeline.venue
            if id(venue) in closed:
                continue
            closed.add(id(venue))
            shutdown = getattr(venue, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    self.logger.exception(
                        "Venue shutdown failed: %s",
                        pipeline.spec.strategy_id,
                    )

        for environment, connection in getattr(
            self,
            "_ctrader_connections",
            {},
        ).items():
            try:
                connection.shutdown()
            except Exception:
                self.logger.exception(
                    "cTrader %s connection shutdown failed",
                    environment,
                )
        self._ctrader_connections.clear()

        if getattr(self, "_execution_trace", None) is not None:
            try:
                self._execution_trace.close()
            except Exception:
                self.logger.exception("Live execution trace shutdown failed")
            self._execution_trace = None

        for feed_group in self.feed_groups:
            shutdown = getattr(feed_group.feed, "shutdown", None)
            if not callable(shutdown):
                shutdown = getattr(feed_group.feed, "close", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    self.logger.exception(
                        "Live feed shutdown failed: %s",
                        feed_group.market_config,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple live strategies")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "../../LiveTrading/live_config.json"),
        help="Path to the live strategy JSON configuration",
    )
    parser.add_argument(
        "--publish-url",
        default=None,
        help="Override monitoring.publish_url from the live configuration",
    )
    parser.add_argument(
        "--runner-id",
        default=None,
        help="Override the runner ID used for monitoring and output paths",
    )
    args = parser.parse_args()

    runner_id, runner_output_dir = load_live_runner_identity(
        args.config,
        runner_id=args.runner_id,
    )
    run_id = _new_live_run_id()
    output_dir = _live_run_output_dir(runner_output_dir, run_id)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(log_dir, "session.log"),
        console_level=logging.DEBUG,
    )
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("websocket").setLevel(logging.INFO)
    logger.info(
        "Live runner output directory | runner_id=%s path=%s",
        runner_id,
        output_dir,
    )
    configuration = load_live_runner_configuration(
        args.config,
        publish_url=args.publish_url,
        runner_id=runner_id,
        run_id=run_id,
    )
    runner = LiveRunner.from_configuration(configuration, logger=logger)
    runner.run_forever()


if __name__ == "__main__":
    main()
