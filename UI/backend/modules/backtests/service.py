"""Read-only services for canonical backtest detail payloads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa

from experiment.report_service import (
    ReportRecord,
    json_safe_value,
    load_report_details,
    materialize_period_report,
    report_details_path,
    resolve_under_root,
    validate_period,
)
from trade.runner.frontend_report_store import LATEST_BACKTEST_REPORT_PATH
from UI.backend.modules.experiments import service as experiment_service


_TIME_COLUMN_CANDIDATES = ("open_time_date_utc", "open_time_ms_utc")
_MARKET_COLUMNS = ("open", "high", "low", "close", "volume")
_OPTIONAL_COLUMNS = ("label", "pred")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Resolved files needed to assemble one saved experiment detail."""

    prepared_data: Path


def latest_backtest_path() -> Path:
    """Return the fixed standalone backtest payload path."""

    return LATEST_BACKTEST_REPORT_PATH


def load_experiment_backtest(
    dataset_id: str,
    record_id: str,
    period: str,
) -> dict[str, Any]:
    """Assemble the standard detail payload from saved experiment artifacts."""

    if period == "all":
        return _load_complete_experiment_backtest(dataset_id, record_id)
    validate_period(period)
    record = experiment_service.registry.record(dataset_id, record_id)
    return _load_period_backtest(dataset_id, record_id, record, period)


def _load_period_backtest(
    dataset_id: str,
    record_id: str,
    record: ReportRecord,
    period: str,
) -> dict[str, Any]:
    period_report = _period_report(record, period)
    artifacts = resolve_experiment_artifacts(record, period, period_report)
    root = experiment_service.get_reports_root()
    details = load_report_details(record, root)
    materialized = materialize_period_report(record.raw, period, details)
    candles = _load_candles(artifacts.prepared_data, materialized)
    report = {
        "params": json_safe_value(deepcopy(record.raw["params"])),
        "results": {
            period: json_safe_value(
                deepcopy(record.raw["results"][period])
            ),
        },
    }
    report_details = {
        "results": {
            period: json_safe_value(
                deepcopy(details["results"][period])
            ),
        },
    }
    report_file = resolve_under_root(root, record.source)
    details_file = report_details_path(record, root)
    return {
        "schema_version": 2,
        "source": {
            "kind": "experiment",
            "dataset_id": dataset_id,
            "record_id": record_id,
            "strategy_number": record.strategy_number,
            "period": period,
            "report_file": str(report_file.relative_to(root)),
            "report_details_file": str(details_file.relative_to(root)),
            "data_file": str(artifacts.prepared_data.relative_to(root)),
        },
        "candles": candles,
        "report": report,
        "report_details": report_details,
    }


def _load_complete_experiment_backtest(
    dataset_id: str,
    record_id: str,
) -> dict[str, Any]:
    record = experiment_service.registry.record(dataset_id, record_id)
    long_payload = _load_period_backtest(dataset_id, record_id, record, "long")
    forward_payload = _load_period_backtest(dataset_id, record_id, record, "forward")
    long_report = materialize_period_report(
        long_payload["report"],
        "long",
        long_payload["report_details"],
    )
    forward_report = materialize_period_report(
        forward_payload["report"],
        "forward",
        forward_payload["report_details"],
    )
    combined = _combine_reports(long_report, forward_report)
    params = combined.pop("params")
    daily_account = (
        combined.get("raw_analyzer", {})
        .get("customize", {})
        .get("daily_account", [])
    )
    for detail_key in ("raw_analyzer", "strategy_detail", "trade_logs"):
        combined.pop(detail_key, None)
    long_details = long_payload["report_details"]["results"]["long"]
    forward_details = forward_payload["report_details"]["results"]["forward"]
    combined_details = _combine_additional(long_details, forward_details)
    combined_details.setdefault("raw_analyzer", {}).setdefault(
        "customize",
        {},
    )["daily_account"] = daily_account
    return {
        "schema_version": 2,
        "source": {
            "kind": "experiment",
            "dataset_id": dataset_id,
            "record_id": record_id,
            "strategy_number": record.strategy_number,
            "period": "all",
            "report_files": list(dict.fromkeys([
                long_payload["source"]["report_file"],
                forward_payload["source"]["report_file"],
            ])),
            "data_files": [
                long_payload["source"]["data_file"],
                forward_payload["source"]["data_file"],
            ],
        },
        "candles": _combine_candles(
            long_payload["candles"],
            forward_payload["candles"],
        ),
        "report": {
            "params": params,
            "results": {"all": combined},
        },
        "report_details": {
            "results": {"all": combined_details},
        },
    }


def _combine_candles(*segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_time: dict[int, dict[str, Any]] = {}
    for segment in segments:
        for candle in segment:
            timestamp = candle.get("time")
            if isinstance(timestamp, int):
                by_time[timestamp] = candle
    return [by_time[timestamp] for timestamp in sorted(by_time)]


def _combine_additional(
    long_additional: Mapping[str, Any],
    forward_additional: Mapping[str, Any],
) -> dict[str, Any]:
    combined = deepcopy(dict(long_additional))
    for key in ("trade_logs",):
        long_values = long_additional.get(key)
        forward_values = forward_additional.get(key)
        if isinstance(long_values, list) or isinstance(forward_values, list):
            values = [
                *deepcopy(long_values if isinstance(long_values, list) else []),
                *deepcopy(forward_values if isinstance(forward_values, list) else []),
            ]
            if key == "trade_logs":
                values.sort(key=lambda item: item.get("dt", item.get("time", 0)))
            combined[key] = values
    return combined


def _combine_reports(
    long_report: Mapping[str, Any],
    forward_report: Mapping[str, Any],
) -> dict[str, Any]:
    from trade.runner.analyze_backtest_report import (
        build_continuous_equity_path,
        simulate_ftmo_challenges,
    )

    combined = deepcopy(dict(long_report))
    equity_frame, _ = build_continuous_equity_path(
        [("long", dict(long_report)), ("forward", dict(forward_report))],
    )
    if equity_frame.empty:
        raise ValueError("Long and forward reports contain no usable equity data")

    equity = pd.to_numeric(
        equity_frame["continuous_equity"], errors="coerce"
    ).dropna()
    if equity.empty:
        raise ValueError("Combined strategy equity contains no finite values")
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown_fraction = abs(float(drawdown.min()))
    elapsed_days = max(
        (equity.index[-1] - equity.index[0]).total_seconds() / 86_400.0,
        1.0,
    )
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    gross_return = end_value / start_value - 1.0
    cagr = (end_value / start_value) ** (365.25 / elapsed_days) - 1.0
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std(ddof=1)) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(365.0))

    daily_account = _combine_daily_account(long_report, forward_report)

    combined.setdefault("time", {})["start"] = long_report.get("time", {}).get("start")
    combined["time"]["end"] = forward_report.get("time", {}).get("end")
    regions = {}
    for report in (long_report, forward_report):
        report_regions = report.get("time", {}).get("regions")
        if isinstance(report_regions, Mapping):
            regions.update(deepcopy(dict(report_regions)))
    oos_region = regions.setdefault("oos", {})
    forward_time = forward_report.get("time", {})
    if forward_time.get("start") is not None:
        oos_region["start"] = forward_time["start"]
    if forward_time.get("end") is not None:
        oos_region["end"] = forward_time["end"]
    combined["time"]["regions"] = regions
    combined.setdefault("raw_analyzer", {}).setdefault(
        "customize",
        {},
    )["daily_account"] = daily_account
    combined.setdefault("performance", {}).update(
        {
            "start_value": start_value,
            "end_value": end_value,
            "gross_return": gross_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "calmar": cagr / max_drawdown_fraction
            if max_drawdown_fraction > 0
            else 0.0,
        }
    )
    worst_daily = min(
        daily_account,
        key=lambda row: row["intraday_drawdown_pct"],
        default=None,
    )
    worst_intraday_drawdown = (
        float(worst_daily["intraday_drawdown_pct"])
        if worst_daily is not None
        else 0.0
    )
    combined.setdefault("drawdown", {}).update(
        {
            "max_dd_pct": max_drawdown_fraction * 100.0,
            "max_dd_amt": float((running_peak - equity).max()),
            "max_daily_dd": worst_intraday_drawdown,
            "max_daily_date": worst_daily["date"] if worst_daily else None,
            "dd_3_pct_days": sum(
                row["intraday_drawdown_pct"] < -0.03
                for row in daily_account
            ),
            "dd_4_pct_days": sum(
                row["intraday_drawdown_pct"] < -0.04
                for row in daily_account
            ),
            "dd_5_pct_days": sum(
                row["intraday_drawdown_pct"] < -0.05
                for row in daily_account
            ),
        }
    )
    combined["ftmo_challenge"] = simulate_ftmo_challenges(combined, "all")
    return combined


def _combine_daily_account(
    *reports: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Chain period-relative daily account values into one equity scale."""

    combined = []
    current_equity = None
    for report in reports:
        rows = (
            report.get("raw_analyzer", {})
            .get("customize", {})
            .get("daily_account")
        )
        if not isinstance(rows, list) or not rows:
            continue
        start_value = report.get("performance", {}).get("start_value")
        if not _is_number(start_value) or float(start_value) <= 0:
            continue
        if current_equity is None:
            current_equity = float(start_value)
        scale = current_equity / float(start_value)
        last_end_equity = None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            date = row.get("date")
            if date is None:
                continue
            values = {
                name: row.get(name)
                for name in ("start_equity", "minimum_equity", "end_equity")
            }
            if not all(_is_number(value) for value in values.values()):
                continue
            start_equity = float(values["start_equity"]) * scale
            minimum_equity = float(values["minimum_equity"]) * scale
            end_equity = float(values["end_equity"]) * scale
            combined.append(
                {
                    "date": str(date),
                    "start_equity": start_equity,
                    "minimum_equity": minimum_equity,
                    "end_equity": end_equity,
                    "intraday_drawdown_pct": (
                        minimum_equity / start_equity - 1.0
                        if start_equity > 0
                        else 0.0
                    ),
                }
            )
            last_end_equity = end_equity
        if last_end_equity is not None:
            current_equity = last_end_equity
    return combined


def resolve_experiment_artifacts(
    record: ReportRecord,
    period: str,
    period_report: Mapping[str, Any] | None = None,
) -> ExperimentArtifacts:
    """Resolve saved artifacts without allowing access outside REPORTS_ROOT."""

    validate_period(period)
    root = experiment_service.get_reports_root()
    source_file = resolve_under_root(root, record.source)
    source_file = _within_root(source_file, root)
    report_value = period_report or _period_report(record, period)
    params = report_value.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("The selected report has no params object")
    identity = params.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("The selected report has no params.identity object")

    prep_hash = _artifact_component(identity.get("prep_hash"), "prep_hash")

    run_root = source_file.parent
    canonical_prep = run_root / "train" / f"pre_{prep_hash}"
    prep_candidates = [canonical_prep]

    data_params = params.get("data")
    if isinstance(data_params, Mapping):
        _append_safe_reported_directory(
            prep_candidates,
            data_params.get("prep_output_dir"),
            root,
        )

    data_stem = "train_data" if period == "long" else "test_data"
    prepared_candidates = [
        prep_dir / f"{data_stem}.{suffix}"
        for prep_dir in prep_candidates
        for suffix in ("feather", "csv")
    ]
    prepared_data = _first_existing_file(
        prepared_candidates,
        root,
        description=f"{period} prepared data",
    )
    return ExperimentArtifacts(
        prepared_data=prepared_data,
    )


def _period_report(record: ReportRecord, period: str) -> Mapping[str, Any]:
    return materialize_period_report(record.raw, period)


def _artifact_component(value: Any, field_name: str) -> str:
    component = str(value or "")
    if not component or _SAFE_COMPONENT.fullmatch(component) is None:
        raise ValueError(f"Invalid experiment artifact component: {field_name}")
    return component


def _append_safe_reported_directory(
    candidates: list[Path],
    raw_path: Any,
    root: Path,
) -> None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return
    try:
        candidate = _within_root(Path(raw_path).expanduser(), root)
    except ValueError:
        return
    if candidate not in candidates:
        candidates.append(candidate)


def _within_root(path: Path, root: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Artifact path is outside report root: {path}") from exc
    return resolved


def _first_existing_file(
    candidates: list[Path],
    root: Path,
    *,
    description: str,
) -> Path:
    checked = []
    for candidate in candidates:
        try:
            safe_candidate = _within_root(candidate, root)
        except ValueError:
            continue
        checked.append(str(safe_candidate))
        if safe_candidate.is_file():
            return safe_candidate
    raise FileNotFoundError(
        f"Saved {description} was not found below the report root; "
        f"checked: {checked}"
    )


def _load_candles(path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    available_columns = _data_columns(path)
    time_column = next(
        (name for name in _TIME_COLUMN_CANDIDATES if name in available_columns),
        None,
    )
    if time_column is None:
        raise ValueError(f"Prepared data has no supported time column: {path}")

    missing = [name for name in _MARKET_COLUMNS if name not in available_columns]
    if missing:
        raise ValueError(f"Prepared data is missing required columns: {missing}")
    columns = [time_column, *_MARKET_COLUMNS]
    columns.extend(name for name in _OPTIONAL_COLUMNS if name in available_columns)
    frame = _read_prepared_frame(path, columns)

    if time_column == "open_time_ms_utc":
        times = pd.to_datetime(frame[time_column], unit="ms", utc=True, errors="coerce")
    else:
        times = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if times.isna().any():
        raise ValueError(f"Prepared data contains invalid timestamps: {path}")

    start, end = _report_time_bounds(report)
    selection = (times >= start) & (times <= end)
    frame = frame.loc[selection].copy()
    times = times.loc[selection]
    if frame.empty:
        raise ValueError(
            f"Prepared data contains no candles inside report range {start} -> {end}"
        )

    nanosecond_times = times.astype("datetime64[ns, UTC]")
    frame.insert(
        0,
        "time",
        (nanosecond_times.astype("int64") // 1_000_000_000).astype("int64"),
    )
    frame.drop(columns=[time_column], inplace=True)
    return [
        {key: _json_scalar(value) for key, value in candle.items()}
        for candle in frame.to_dict(orient="records")
    ]


def _data_columns(path: Path) -> set[str]:
    if path.suffix == ".csv":
        return set(pd.read_csv(path, nrows=0).columns)
    if path.suffix == ".feather":
        with pa.memory_map(str(path), "r") as source:
            return set(pa.ipc.open_file(source).schema.names)
    raise ValueError(f"Unsupported prepared data format: {path.suffix}")


def _read_prepared_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    if path.suffix == ".feather":
        return pd.read_feather(path, columns=columns)
    raise ValueError(f"Unsupported prepared data format: {path.suffix}")


def _report_time_bounds(report: Mapping[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    time_value = report.get("time")
    if not isinstance(time_value, Mapping):
        raise ValueError("The saved report has no time object")
    start = pd.to_datetime(time_value.get("start"), utc=True, errors="coerce")
    end = pd.to_datetime(time_value.get("end"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("The saved report has invalid start or end time")
    if start > end:
        raise ValueError("The saved report start time is after its end time")
    return start, end


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
