"""Label generation and compact chart-data loading services."""

from __future__ import annotations

from dataclasses import asdict
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
from threading import RLock
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from data_process import common, feature


LOGGER = logging.getLogger("strategy_center.labels")
LABEL_OUTPUT_DIR = Path(
    os.environ.get("LABEL_OUTPUT_DIR", common.DATA_OUT_DIR)
).expanduser()
LABEL_DATA_LOCK = RLock()

SafePathComponent = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
Interval = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[1-9][0-9]*[smhdwM]$",
    ),
]
PositiveFiniteFloat = Annotated[
    float,
    Field(gt=0, allow_inf_nan=False),
]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]


class LabelConfig(BaseModel):
    """Strict API representation of ``data_process.common.BaseDefine``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    market_category: Literal["Cryptocurrency", "Forex", "Stock"] = (
        "Cryptocurrency"
    )
    data_source: SafePathComponent = "binance_public_data"
    symbol: SafePathComponent = "DOGEUSDT"
    interval: Interval = "30m"
    trading_type: Literal["spot", "um", "cm"] = "um"
    label_type: Literal["FTHL", "TBM", "TBM_TREND", "BBM"] = "FTHL"
    vol_ewma_span: int = Field(default=20, gt=0)
    predict_num: int = Field(default=32, gt=0)
    vol_multiplier_long: PositiveFiniteFloat = 1.7
    stop_multiplier_rate_long: NonNegativeFiniteFloat | None = None
    vol_multiplier_short: PositiveFiniteFloat = 1.7
    stop_multiplier_rate_short: NonNegativeFiniteFloat | None = None
    tbm_take_profit_price: Literal["close", "high_low"] | None = None
    min_expected_move_pct: NonNegativeFiniteFloat = 0.01
    version: NonNegativeFiniteFloat = 0.1

    @model_validator(mode="after")
    def validate_label_specific_fields(self) -> "LabelConfig":
        if (
            self.label_type in {"TBM", "TBM_TREND"}
            and self.tbm_take_profit_price is None
        ):
            raise ValueError(
                "tbm_take_profit_price is required for TBM and TBM_TREND labels"
            )
        return self

    def to_base_define(self) -> common.BaseDefine:
        return common.BaseDefine(**self.model_dump())


class LabelServiceError(RuntimeError):
    """Base class for errors surfaced by the Labels service."""


class LabelConfigurationError(LabelServiceError):
    """The requested configuration cannot locate usable market data."""


class LabelDataError(LabelServiceError):
    """Persisted label output is missing required or readable content."""


class LabelGenerationError(LabelServiceError):
    """Data preparation failed after request validation."""


CHART_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "label",
    "expected_vol",
    "threshold_long",
    "threshold_short",
    "stop_threshold_long",
    "stop_threshold_short",
    "trend_strength",
    "reach_time",
    "scan_len",
    "invalid_reason",
    "tb_anchor_label",
    "tb_anchor_reach_time",
    "trend_source_idx",
    "is_tb_anchor",
    "is_trend_propagated",
)
REQUIRED_CHART_COLUMNS = {"open", "high", "low", "close", "label"}


def label_schema() -> dict[str, Any]:
    """Return the form schema, defaults, and stable label semantics."""

    return {
        "config_schema": LabelConfig.model_json_schema(),
        "defaults": _config_payload(common.BaseDefine()),
        "label_values": {
            "-1": "Invalid",
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive",
        },
        "rules": {
            "tbm_take_profit_price": (
                "Required when label_type is TBM or TBM_TREND."
            ),
            "result_split": "The chart displays the forward dataset.",
        },
    }


def current_labels(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the latest complete forward-label result without generating data."""

    resolved_output_dir = _output_dir(output_dir)
    with LABEL_DATA_LOCK:
        return _read_current_unlocked(resolved_output_dir)


def generate_labels(
    config: LabelConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run preparation synchronously and return its newly persisted result."""

    resolved_output_dir = _output_dir(output_dir)
    parameters = config.to_base_define()
    source_path = Path(common.market_data_path(parameters)).expanduser()
    if not source_path.is_file():
        raise LabelConfigurationError(
            f"Market data source does not exist: {source_path}"
        )

    configured_features = (
        feature.FEATURE_LIST_COMMODITY
        if parameters.market_category == "Forex"
        else []
    )

    with LABEL_DATA_LOCK:
        resolved_output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=".strategy-center-labels-",
                dir=resolved_output_dir.parent,
            )
        )
        try:
            from data_process import preparation

            preparation.main(
                LOGGER,
                feature.FEATURE_GROUP_LIST,
                feature_conf_list=configured_features,
                para=parameters,
                prep_output_dir=str(staging_dir),
            )
            generated_result = _read_current_unlocked(staging_dir)
            _publish_staged_output(staging_dir, resolved_output_dir)
        except LabelServiceError:
            raise
        except Exception as exc:
            raise LabelGenerationError(f"Label generation failed: {exc}") from exc
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        return generated_result


def _publish_staged_output(staging_dir: Path, output_dir: Path) -> None:
    """Publish one complete preparation result and restore the old set on error."""

    staged_files = _result_files(staging_dir)
    missing = [str(path) for path in staged_files if not path.is_file()]
    if missing:
        raise LabelGenerationError(
            "Label generation did not produce every required file: "
            + ", ".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=".strategy-center-label-backup-",
            dir=output_dir.parent,
        )
    )
    target_files = _result_files(output_dir)
    moved_to_backup: list[tuple[Path, Path]] = []
    published: list[Path] = []

    try:
        for target in target_files:
            if target.exists():
                backup = backup_dir / target.name
                os.replace(target, backup)
                moved_to_backup.append((target, backup))
        for staged, target in zip(staged_files, target_files, strict=True):
            os.replace(staged, target)
            published.append(target)
    except Exception as publish_error:
        rollback_errors: list[str] = []
        for target in reversed(published):
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(str(exc))
        for target, backup in reversed(moved_to_backup):
            try:
                os.replace(backup, target)
            except OSError as exc:
                rollback_errors.append(str(exc))

        if rollback_errors:
            raise LabelGenerationError(
                "Unable to publish label output and rollback was incomplete. "
                f"Recovery files remain in {backup_dir}: {rollback_errors}"
            ) from publish_error
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise LabelGenerationError(
            "Unable to publish label output; the previous result was restored"
        ) from publish_error
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _result_files(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        Path(common.get_train_data_path_in_dir(str(base_dir))),
        Path(common.get_test_data_path_in_dir(str(base_dir))),
        Path(common.get_data_config_path_in_dir(str(base_dir))),
        Path(common.get_data_manifest_path_in_dir(str(base_dir))),
    )


def _read_current_unlocked(output_dir: Path) -> dict[str, Any]:
    forward_path = Path(common.get_test_data_path_in_dir(str(output_dir)))
    config_path = Path(common.get_data_config_path_in_dir(str(output_dir)))
    manifest_path = Path(common.get_data_manifest_path_in_dir(str(output_dir)))

    if not forward_path.is_file():
        parameters = _load_config_or_default(output_dir, config_path)
        manifest = _load_optional_manifest(output_dir, manifest_path)
        return _empty_result(parameters, manifest)

    missing_metadata = [
        str(path) for path in (config_path, manifest_path) if not path.is_file()
    ]
    if missing_metadata:
        raise LabelDataError(
            "Forward label data exists but required metadata is missing: "
            + ", ".join(missing_metadata)
        )

    try:
        parameters = common.load_pre_params_from_dir(str(output_dir))
        manifest = common.load_data_manifest_from_dir(str(output_dir))
        frame = common.load_test_df_from_dir(str(output_dir))
    except Exception as exc:
        raise LabelDataError(f"Unable to load persisted label output: {exc}") from exc

    candles, summary = _compact_chart_data(frame, manifest)
    return {
        "available": True,
        "config": _config_payload(parameters),
        "manifest": common.json_safe(manifest),
        "summary": summary,
        "candles": candles,
    }


def _load_config_or_default(
    output_dir: Path,
    config_path: Path,
) -> common.BaseDefine:
    if not config_path.is_file():
        return common.BaseDefine()
    try:
        return common.load_pre_params_from_dir(str(output_dir))
    except Exception as exc:
        raise LabelDataError(f"Unable to load persisted label configuration: {exc}") from exc


def _load_optional_manifest(
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    try:
        return common.load_data_manifest_from_dir(str(output_dir))
    except Exception as exc:
        raise LabelDataError(f"Unable to load persisted label manifest: {exc}") from exc


def _empty_result(
    parameters: common.BaseDefine,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "available": False,
        "config": _config_payload(parameters),
        "manifest": common.json_safe(manifest) if manifest is not None else None,
        "summary": {
            "split": "forward",
            "row_count": 0,
            "start": None,
            "end": None,
            "label_counts": {},
            "label_ratios": {},
            "invalid_reason_counts": {},
            "columns": [],
        },
        "candles": [],
    }


def _compact_chart_data(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing_columns = sorted(REQUIRED_CHART_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise LabelDataError(
            f"Forward label data is missing chart columns: {missing_columns}"
        )

    time_seconds = _time_seconds(frame)
    numeric_ohlc = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    valid_rows = time_seconds.notna() & np.isfinite(numeric_ohlc).all(axis=1)
    if not bool(valid_rows.any()):
        raise LabelDataError("Forward label data contains no valid chart rows")

    available_columns = [column for column in CHART_COLUMNS if column in frame.columns]
    chart = frame.loc[valid_rows, available_columns].copy()
    chart.insert(
        0,
        "time",
        time_seconds.loc[valid_rows].astype(np.int64),
    )
    chart.sort_values("time", kind="stable", inplace=True)
    duplicate_times = int(chart["time"].duplicated().sum())
    if duplicate_times:
        raise LabelDataError(
            f"Forward label data contains {duplicate_times} duplicate timestamps"
        )
    chart.replace([np.inf, -np.inf], np.nan, inplace=True)
    chart = chart.astype(object).where(pd.notna(chart), None)
    candles = chart.to_dict(orient="records")

    labels = pd.to_numeric(chart["label"], errors="coerce").dropna().astype(int)
    label_counts = labels.value_counts().sort_index()
    total_labels = int(label_counts.sum())

    invalid_reason_counts: dict[str, int] = {}
    if "invalid_reason" in chart.columns:
        invalid_reasons = (
            pd.to_numeric(chart["invalid_reason"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        invalid_reason_counts = {
            str(int(key)): int(value) for key, value in invalid_reasons.items()
        }

    first_time = int(chart["time"].iloc[0])
    last_time = int(chart["time"].iloc[-1])
    summary = {
        "split": "forward",
        "row_count": int(len(chart)),
        "start": pd.to_datetime(first_time, unit="s", utc=True).isoformat(),
        "end": pd.to_datetime(last_time, unit="s", utc=True).isoformat(),
        "label_counts": {
            str(int(key)): int(value) for key, value in label_counts.items()
        },
        "label_ratios": {
            str(int(key)): (float(value) / total_labels if total_labels else 0.0)
            for key, value in label_counts.items()
        },
        "invalid_reason_counts": invalid_reason_counts,
        "columns": list(chart.columns),
        "data_id": manifest.get("data_id"),
    }
    return candles, summary


def _time_seconds(frame: pd.DataFrame) -> pd.Series:
    if "open_time_ms_utc" in frame.columns:
        milliseconds = pd.to_numeric(frame["open_time_ms_utc"], errors="coerce")
        return np.floor(milliseconds / 1000.0)
    if "open_time_date_utc" in frame.columns:
        timestamps = pd.to_datetime(
            frame["open_time_date_utc"],
            utc=True,
            errors="coerce",
        )
        seconds = timestamps.astype("int64") // 1_000_000_000
        return seconds.where(timestamps.notna(), np.nan)
    raise LabelDataError(
        "Forward label data requires open_time_ms_utc or open_time_date_utc"
    )


def _config_payload(parameters: common.BaseDefine) -> dict[str, Any]:
    payload = common.json_safe(asdict(parameters))
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            payload[key] = None
    return payload


def _output_dir(output_dir: str | Path | None) -> Path:
    selected = LABEL_OUTPUT_DIR if output_dir is None else Path(output_dir)
    return selected.expanduser()
