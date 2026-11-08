"""Pure report loading and analysis primitives for CLI and web frontends.

The module deliberately has no FastAPI, plotting, or project-runtime imports.
Reports are queried through period-relative dotted field paths, for example
``params.train.model_cfg.seq_len`` or ``performance.cagr``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence


PERIODS = ("development", "forward")
DEFAULT_COLUMNS = (
    "params.hash",
    "params.train.model_cfg.model_type",
    "params.train.model_cfg.model_version",
    "performance.cagr",
    "performance.sharpe",
    "performance.calmar",
    "drawdown.max_dd_pct",
    "trades.daily_freq",
    "trades.win_rate",
    "performance.rc_summary.rc_median",
    "performance.rc_summary.rc_pos_ratio",
    "drawdown.max_hwm_duration_days",
)
SUPPORTED_OPERATORS = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "in",
    "not_in",
    "contains",
    "is_null",
    "not_null",
)
SUPPORTED_AGGREGATIONS = (
    "count",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
)

_MISSING = object()


@dataclass
class ReportRecord:
    strategy_number: int
    record_id: str
    source: str
    line_number: int
    raw: dict[str, Any]
    _flat_cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def values(self, period: str) -> dict[str, Any]:
        validate_period(period)
        if period not in self._flat_cache:
            period_report = materialize_period_report(self.raw, period)
            flattened = flatten_mapping(period_report)
            flattened["_meta.record_id"] = self.record_id
            flattened["_meta.source"] = self.source
            flattened["_meta.line_number"] = self.line_number
            self._flat_cache[period] = flattened
        return self._flat_cache[period]


@dataclass
class ReportDataset:
    records: list[ReportRecord]
    report_files: list[str]
    malformed_lines: int = 0
    duplicate_records: int = 0


def validate_period(period: str) -> None:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}")


def resolve_under_root(root: str | Path, requested: str | Path) -> Path:
    """Resolve a user-selected path and reject traversal outside ``root``."""
    root_path = Path(root).expanduser().resolve()
    requested_path = Path(requested).expanduser()
    candidate = (
        requested_path.resolve()
        if requested_path.is_absolute()
        else (root_path / requested_path).resolve()
    )
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Path is outside report root: {requested}") from exc
    return candidate


def find_report_files(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Report path does not exist: {candidate}")
    if candidate.is_file():
        if candidate.name != "reports.jsonl":
            raise ValueError(f"Expected reports.jsonl, got: {candidate.name}")
        return [candidate]
    return sorted(candidate.rglob("reports.jsonl"))


def load_report_dataset(
    paths: Sequence[str | Path],
    *,
    allowed_root: str | Path,
    deduplicate: bool = True,
) -> ReportDataset:
    """Load reports from files/directories selected below ``allowed_root``."""
    if not paths:
        raise ValueError("At least one report path is required")

    files: list[Path] = []
    seen_files: set[Path] = set()
    for selected in paths:
        resolved = resolve_under_root(allowed_root, selected)
        for report_file in find_report_files(resolved):
            resolved_file = report_file.resolve()
            if resolved_file not in seen_files:
                files.append(resolved_file)
                seen_files.add(resolved_file)
    files.sort()
    if not files:
        raise FileNotFoundError("No reports.jsonl files found under selected paths")

    records: list[ReportRecord] = []
    malformed_lines = 0
    duplicate_records = 0
    identities: set[tuple[str, str]] = set()
    root_path = Path(allowed_root).expanduser().resolve()

    for report_file in files:
        source = str(report_file.relative_to(root_path))
        with report_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(raw, dict):
                    malformed_lines += 1
                    continue

                params = raw.get("params", {})
                task_hash = str(params.get("hash") or "")
                git_commit = str(params.get("git_commit") or "")
                identity = (task_hash, git_commit)
                if deduplicate and task_hash and identity in identities:
                    duplicate_records += 1
                    continue
                identities.add(identity)

                id_material = f"{source}\0{line_number}\0{task_hash}\0{git_commit}"
                record_id = sha256(id_material.encode("utf-8")).hexdigest()[:16]
                records.append(
                    ReportRecord(
                        strategy_number=len(records),
                        record_id=record_id,
                        source=source,
                        line_number=line_number,
                        raw=raw,
                    )
                )

    return ReportDataset(
        records=records,
        report_files=[str(path.relative_to(root_path)) for path in files],
        malformed_lines=malformed_lines,
        duplicate_records=duplicate_records,
    )


def materialize_period_report(
    report: Mapping[str, Any],
    period: str,
    report_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine shared parameters, one period result, and optional details."""
    validate_period(period)
    params = report.get("params")
    results = report.get("results")
    if not isinstance(params, Mapping):
        raise ValueError("Report has no params object")
    if not isinstance(results, Mapping) or not isinstance(results.get(period), Mapping):
        raise ValueError(f"Report has no {period!r} result")

    materialized = {"params": dict(params), **dict(results[period])}
    if report_details is not None:
        detail_results = report_details.get("results")
        if not isinstance(detail_results, Mapping):
            raise ValueError("report_details has no results object")
        period_details = detail_results.get(period)
        if not isinstance(period_details, Mapping):
            raise ValueError(f"report_details has no {period!r} result")
        materialized.update(dict(period_details))
    return materialized


def report_details_path(record: ReportRecord, reports_root: str | Path) -> Path:
    """Resolve the per-task detail artifact associated with a summary record."""
    root = Path(reports_root).expanduser().resolve()
    report_file = resolve_under_root(root, record.source)
    params = record.raw.get("params")
    identity = params.get("identity") if isinstance(params, Mapping) else None
    if not isinstance(identity, Mapping):
        raise ValueError("Report params has no identity object")

    full_hash = str(identity.get("full_hash") or "")
    if not full_hash or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in full_hash
    ):
        raise ValueError("Invalid report identity component: full_hash")
    return report_file.parent / "sim_output" / full_hash / "report_details.json"


def load_report_details(
    record: ReportRecord,
    reports_root: str | Path,
) -> dict[str, Any]:
    path = report_details_path(record, reports_root)
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"report_details must contain an object: {path}")
    return payload


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten scalar report leaves without copying large record lists."""
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_mapping(child, path))
        return result
    if isinstance(value, list):
        if all(_is_scalar(item) for item in value):
            result[prefix] = [_json_scalar(item) for item in value]
        else:
            result[f"{prefix}.__count"] = len(value)
        return result
    if prefix:
        result[prefix] = _json_scalar(value)
    return result


def discover_schema(
    records: Sequence[ReportRecord],
    period: str,
    *,
    distinct_limit: int = 60,
) -> list[dict[str, Any]]:
    validate_period(period)
    observations: dict[str, list[Any]] = {}
    for record in records:
        for path, value in record.values(period).items():
            if path.startswith("_meta."):
                continue
            observations.setdefault(path, []).append(value)

    schema: list[dict[str, Any]] = []
    total = len(records)
    for path, values in observations.items():
        non_null = [value for value in values if value is not None]
        field_type = _infer_field_type(non_null)
        distinct = _distinct_values(non_null, distinct_limit)
        item: dict[str, Any] = {
            "path": path,
            "label": path.rsplit(".", 1)[-1],
            "category": path.split(".", 1)[0],
            "type": field_type,
            "role": "parameter" if path.startswith("params.") else "metric",
            "present": len(values),
            "missing": total - len(non_null),
            "groupable": distinct is not None,
        }
        if distinct is not None:
            item["values"] = distinct
        if field_type == "number" and non_null:
            numeric = [float(value) for value in non_null if _is_number(value)]
            if numeric:
                item["min"] = min(numeric)
                item["max"] = max(numeric)
        schema.append(item)
    return sorted(schema, key=lambda item: (item["role"] != "parameter", item["path"]))


def filter_records(
    records: Sequence[ReportRecord],
    period: str,
    filters: Sequence[Mapping[str, Any]] | None = None,
) -> list[ReportRecord]:
    validate_period(period)
    filters = filters or []
    for spec in filters:
        operator = spec.get("operator")
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {operator!r}")
        if not spec.get("field"):
            raise ValueError("Each filter requires a field")

    selected: list[ReportRecord] = []
    for record in records:
        values = record.values(period)
        if all(_matches_filter(values, spec) for spec in filters):
            selected.append(record)
    return selected


def query_records(
    records: Sequence[ReportRecord],
    period: str,
    *,
    filters: Sequence[Mapping[str, Any]] | None = None,
    columns: Sequence[str] | None = None,
    sort_by: str | None = "performance.cagr",
    descending: bool = True,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    if page < 1:
        raise ValueError("page must be >= 1")
    if not 1 <= page_size <= 500:
        raise ValueError("page_size must be between 1 and 500")
    columns = tuple(columns or DEFAULT_COLUMNS)
    selected = filter_records(records, period, filters)
    if sort_by:
        selected = _sort_records(selected, period, sort_by, descending)

    start = (page - 1) * page_size
    page_records = selected[start : start + page_size]
    items = []
    for record in page_records:
        values = record.values(period)
        items.append(
            {
                "strategy_number": record.strategy_number,
                "record_id": record.record_id,
                "source": record.source,
                "values": {column: values.get(column) for column in columns},
            }
        )
    return {
        "total": len(selected),
        "page": page,
        "page_size": page_size,
        "columns": list(columns),
        "items": items,
    }


def aggregate_records(
    records: Sequence[ReportRecord],
    period: str,
    *,
    group_by: Sequence[str],
    metric: str,
    aggregations: Sequence[str] = ("count", "median", "mean"),
    filters: Sequence[Mapping[str, Any]] | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    validate_period(period)
    if not 1 <= len(group_by) <= 2:
        raise ValueError("group_by must contain one or two fields")
    invalid = [name for name in aggregations if name not in SUPPORTED_AGGREGATIONS]
    if invalid:
        raise ValueError(f"Unsupported aggregations: {invalid}")

    selected = filter_records(records, period, filters)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in selected:
        values = record.values(period)
        raw_key = tuple(values.get(path) for path in group_by)
        key = tuple(_hashable(value) for value in raw_key)
        group = grouped.setdefault(
            key,
            {"group": raw_key, "values": [], "record_count": 0},
        )
        group["record_count"] += 1
        metric_value = values.get(metric)
        if _is_number(metric_value):
            group["values"].append(float(metric_value))

    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        numeric = group["values"]
        row = {
            "group": {
                path: _json_scalar(value)
                for path, value in zip(group_by, group["group"], strict=True)
            },
            "count": group["record_count"],
            "metric_count": len(numeric),
        }
        for aggregation in aggregations:
            if aggregation != "count":
                row[aggregation] = _aggregate(numeric, aggregation)
        rows.append(row)

    sort_metric = next((name for name in aggregations if name != "count"), "count")
    rows.sort(
        key=lambda row: (
            row.get(sort_metric) is not None,
            row.get(sort_metric) if row.get(sort_metric) is not None else -math.inf,
        ),
        reverse=True,
    )
    truncated = len(rows) > limit
    return {
        "period": period,
        "group_by": list(group_by),
        "metric": metric,
        "aggregations": list(aggregations),
        "matched_records": len(selected),
        "group_count": len(rows),
        "truncated": truncated,
        "rows": rows[:limit],
    }


def equity_series(
    records: Sequence[ReportRecord],
    record_ids: Sequence[str],
    period: str,
    *,
    reports_root: str | Path,
) -> list[dict[str, Any]]:
    validate_period(period)
    requested = set(record_ids)
    if len(requested) > 20:
        raise ValueError("At most 20 equity series can be requested")
    result = []
    for record in records:
        if record.record_id not in requested:
            continue
        details = load_report_details(record, reports_root)
        period_report = materialize_period_report(record.raw, period, details)
        daily = _get_nested(
            period_report,
            "raw_analyzer.customize.daily_account",
        )
        points = []
        if isinstance(daily, list):
            for item in daily:
                if not isinstance(item, Mapping):
                    continue
                date = item.get("date")
                equity = item.get("end_equity")
                if date is not None and _is_number(equity):
                    points.append({"time": str(date)[:10], "value": float(equity)})
        values = record.values(period)
        model_type = values.get("params.train.model_cfg.model_type") or "model"
        result.append(
            {
                "strategy_number": record.strategy_number,
                "record_id": record.record_id,
                "label": f"#{record.strategy_number} {model_type}",
                "points": points,
            }
        )
    missing = requested - {item["record_id"] for item in result}
    if missing:
        raise KeyError(f"Unknown record ids: {sorted(missing)}")
    return result


def record_detail(records: Sequence[ReportRecord], record_id: str) -> dict[str, Any]:
    for record in records:
        if record.record_id == record_id:
            return {
                "strategy_number": record.strategy_number,
                "record_id": record.record_id,
                "source": record.source,
                "line_number": record.line_number,
                "report": json_safe_value(record.raw),
            }
    raise KeyError(f"Unknown record id: {record_id}")


def _matches_filter(values: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    current = values.get(str(spec["field"]), _MISSING)
    operator = spec["operator"]
    expected = spec.get("value")
    is_null = current is _MISSING or current is None
    if operator == "is_null":
        return is_null
    if operator == "not_null":
        return not is_null
    if is_null:
        return False
    if operator == "eq":
        return _comparable(current) == _comparable(expected)
    if operator == "ne":
        return _comparable(current) != _comparable(expected)
    if operator in {"gt", "gte", "lt", "lte"}:
        return _ordered_compare(current, expected, operator)
    if operator == "between":
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)) or len(expected) != 2:
            raise ValueError("between requires [min, max]")
        return _ordered_compare(current, expected[0], "gte") and _ordered_compare(
            current, expected[1], "lte"
        )
    if operator in {"in", "not_in"}:
        candidates = expected if isinstance(expected, list) else [expected]
        matched = _comparable(current) in {_comparable(item) for item in candidates}
        return matched if operator == "in" else not matched
    if operator == "contains":
        if isinstance(current, list):
            return _comparable(expected) in {_comparable(item) for item in current}
        return str(expected).casefold() in str(current).casefold()
    raise ValueError(f"Unsupported filter operator: {operator!r}")


def _sort_records(
    records: Sequence[ReportRecord], period: str, field: str, descending: bool
) -> list[ReportRecord]:
    present = []
    missing = []
    for record in records:
        value = record.values(period).get(field)
        (missing if value is None else present).append(record)
    present.sort(
        key=lambda record: _sort_value(record.values(period).get(field)),
        reverse=descending,
    )
    return present + missing


def _aggregate(values: Sequence[float], name: str) -> float | int | None:
    if name == "count":
        return len(values)
    if not values:
        return None
    if name == "mean":
        return mean(values)
    if name == "median":
        return median(values)
    if name == "min":
        return min(values)
    if name == "max":
        return max(values)
    if name == "std":
        return pstdev(values)
    if name.startswith("p") and name[1:].isdigit():
        return _percentile(values, int(name[1:]))
    raise ValueError(f"Unsupported aggregation: {name}")


def _percentile(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _infer_field_type(values: Sequence[Any]) -> str:
    if not values:
        return "null"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(_is_number(value) for value in values):
        return "number"
    if all(isinstance(value, list) for value in values):
        return "list"
    return "string"


def _distinct_values(values: Sequence[Any], limit: int) -> list[Any] | None:
    distinct: dict[Any, Any] = {}
    for value in values:
        key = _hashable(value)
        distinct.setdefault(key, _json_scalar(value))
        if len(distinct) > limit:
            return None
    return sorted(distinct.values(), key=lambda value: str(value))


def _ordered_compare(current: Any, expected: Any, operator: str) -> bool:
    try:
        if _is_number(current):
            left, right = float(current), float(expected)
        else:
            left, right = str(current), str(expected)
        return {
            "gt": left > right,
            "gte": left >= right,
            "lt": left < right,
            "lte": left <= right,
        }[operator]
    except (TypeError, ValueError):
        return False


def _get_nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_scalar(value: Any) -> Any:
    return json_safe_value(value)


def json_safe_value(value: Any) -> Any:
    """Return a recursively sanitized value accepted by strict JSON encoders."""

    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    return value


def _comparable(value: Any) -> Any:
    return _hashable(value)


def _sort_value(value: Any) -> tuple[int, Any]:
    if _is_number(value):
        return (0, float(value))
    if isinstance(value, bool):
        return (1, value)
    return (2, json.dumps(value, sort_keys=True) if isinstance(value, list) else str(value))
