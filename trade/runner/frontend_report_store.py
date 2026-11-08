"""Persistent hand-off between the standalone backtest and the detail UI.

The backtest process owns writes. The UI backend only serves the completed
file, so it never needs to import model or backtest execution code.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATEST_BACKTEST_REPORT_PATH = (
    PROJECT_ROOT.parent / "quant_output" / "latest_backtest_report.json"
)
LATEST_BACKTEST_REPORT_PATH = Path(
    os.environ.get(
        "BACKTEST_FRONTEND_REPORT_PATH",
        DEFAULT_LATEST_BACKTEST_REPORT_PATH,
    )
).expanduser().resolve()


def validate_frontend_report(payload: Any) -> None:
    """Reject files that cannot be consumed by the single-strategy UI."""

    if not isinstance(payload, Mapping):
        raise ValueError("The frontend report must be a JSON object")
    if not isinstance(payload.get("candles"), list):
        raise ValueError("The frontend report must contain a candles list")
    report = payload.get("report")
    if not isinstance(report, Mapping):
        raise ValueError("The frontend payload must contain a report object")
    if not isinstance(report.get("params"), Mapping):
        raise ValueError("The report must contain a params object")
    results = report.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("The report must contain a results object")
    period_keys = {"development", "long", "forward", "all"}.intersection(results)
    if len(period_keys) != 1:
        raise ValueError(
            "The frontend report must contain exactly one period result"
        )
    period = next(iter(period_keys))
    if not isinstance(results[period], Mapping):
        raise ValueError("The period result must be a JSON object")

    report_details = payload.get("report_details")
    if not isinstance(report_details, Mapping):
        raise ValueError("The frontend payload must contain report_details")
    detail_results = report_details.get("results")
    if not isinstance(detail_results, Mapping):
        raise ValueError("report_details must contain a results object")
    if not isinstance(detail_results.get(period), Mapping):
        raise ValueError("report_details must contain the matching period")


def write_latest_backtest_report(
    payload: Mapping[str, Any],
    path: Path | str = LATEST_BACKTEST_REPORT_PATH,
) -> Path:
    """Atomically publish a complete backtest payload for the detail UI."""

    validate_frontend_report(payload)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")

    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(
                _json_safe(payload),
                output,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
                allow_nan=False,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return destination


def load_latest_backtest_report(
    path: Path | str = LATEST_BACKTEST_REPORT_PATH,
) -> dict[str, Any]:
    """Load and validate the payload, primarily for diagnostics and tests."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    validate_frontend_report(payload)
    return payload


def _json_safe(value: Any) -> Any:
    """Convert nested report values to strict JSON without changing its shape."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None

    item_method = getattr(value, "item", None)
    if callable(item_method) and type(value).__module__.startswith("numpy"):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return value
