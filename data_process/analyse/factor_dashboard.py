"""Build compact factor dashboard pages and on-demand compressed data caches."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from data_process.analyse.report_cache import (
    CACHE_SOURCE_FILES,
    report_source_signature,
)


MAX_ROLLING_POINTS = 240
CACHE_DIR_NAME = "factor_dashboard_data"
ROLLING_COLUMNS = [
    "factor",
    "split",
    "horizon",
    "window_end",
    "n",
    "rank_ic",
    "pearson_ic",
]
REPORT_COLUMNS = {
    "predictive": [
        "factor",
        "split",
        "horizon",
        "n",
        "rank_ic",
        "pearson_ic",
        "icir",
        "rolling_ic_count",
        "rolling_ic_mean",
        "rolling_ic_std",
        "nw_t",
        "nw_fdr_bh",
        "mi_nats",
        "mi_shift_pvalue",
        "quantile_monotonicity",
        "top_minus_bottom",
        "hit_rate",
        "profit_loss_ratio",
        "trade_count",
    ],
    "decay": [
        "factor",
        "split",
        "horizon",
        "rank_ic",
        "pearson_ic",
        "nw_t",
        "nw_fdr_bh",
    ],
    "quantiles": ["factor", "split", "horizon", "quantile", "n", "mean_return"],
    "autocorrelation": ["factor", "split", "lag", "acf", "pacf"],
    "diagnostics": [
        "factor",
        "split",
        "n_total",
        "n_finite",
        "adf_stat",
        "adf_pvalue",
        "kpss_stat",
        "kpss_pvalue",
        "status",
        "stationarity",
    ],
}


def factor_cache_key(factor: str) -> str:
    """Return a filesystem-safe stable key for a factor name."""
    return hashlib.sha256(factor.encode("utf-8")).hexdigest()


def _read_records(directory: Path, name: str) -> list[dict]:
    wanted = set(REPORT_COLUMNS[name])
    frame = pd.read_csv(
        directory / f"{name}.csv", usecols=lambda column: column in wanted
    )
    frame = frame.loc[frame["split"] == "combined"]
    if frame.empty:
        return []
    return json.loads(
        frame.to_json(orient="records", date_format="iso", date_unit="ms")
    )


def _rolling_records(
    directory: Path, diagnostics: list[dict], metadata: dict, max_points: int
) -> list[dict]:
    """Stream the large rolling CSV and retain evenly spaced points per series."""
    counts = {r["factor"]: int(r["n_total"]) for r in diagnostics}
    window, step = int(metadata["config"]["rolling_window"]), int(
        metadata["config"]["rolling_step"]
    )
    selected, closed = [], set()
    current_key, current_count, stride, last = None, 0, 1, None
    total_read = 0

    def finish_group():
        if last is not None and (current_count - 1) % stride:
            selected.append(last)

    for chunk in pd.read_csv(
        directory / "rolling_ic.csv",
        usecols=ROLLING_COLUMNS,
        chunksize=200_000,
        dtype={
            "factor": "string",
            "split": "category",
            "horizon": "int16",
            "n": "int32",
            "rank_ic": "float32",
            "pearson_ic": "float32",
        },
        parse_dates=["window_end"],
    ):
        total_read += len(chunk)
        if total_read % 1_000_000 < len(chunk):
            print(
                f"Read {total_read:,} rolling IC rows; retaining at most {max_points:,} points per series.",
                flush=True,
            )
        for key, group in chunk.groupby(
            ["factor", "split", "horizon"], sort=False, observed=True
        ):
            key = (str(key[0]), str(key[1]), int(key[2]))
            if key[1] != "combined":
                continue
            if key != current_key:
                if current_key is not None:
                    finish_group()
                    closed.add(current_key)
                if key in closed:
                    raise ValueError(
                        "rolling_ic.csv must be ordered by factor, split and horizon"
                    )
                current_key, current_count, last = key, 0, None
                expected = max(0, (counts.get(key[0], 0) - window) // step + 1)
                stride = max(1, math.ceil(expected / max(1, max_points - 1)))
            positions = np.arange(len(group))
            take = np.flatnonzero((current_count + positions) % stride == 0)
            if len(take):
                selected.append(group.iloc[take])
            current_count += len(group)
            last = group.iloc[[-1]]
    if current_key is not None:
        finish_group()
    if not selected:
        return []
    frame = pd.concat(selected, ignore_index=True)
    return json.loads(
        frame.to_json(orient="records", date_format="iso", date_unit="ms")
    )


def read_overview(report_dir: str | Path, horizon: int) -> dict:
    """Read the small overview cache and return only one horizon's factor rows."""
    cache = Path(report_dir) / CACHE_DIR_NAME / "overview.json.gz"
    with gzip.open(cache, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    rows = [row for row in data["predictive"] if int(row["horizon"]) == int(horizon)]
    rows.sort(key=lambda row: abs(float(row.get("rank_ic") or 0)), reverse=True)
    return {"metadata": data["metadata"], "predictive": rows}


def read_factor_detail(report_dir: str | Path, factor: str, horizon: int) -> dict:
    """Read one factor and return-horizon cache."""
    path = (
        Path(report_dir)
        / CACHE_DIR_NAME
        / "details"
        / factor_cache_key(factor)
        / f"{int(horizon)}.json.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    return data


def build_dashboard(
    report_dir: str | Path,
    output: str | Path | None = None,
    max_rolling_points: int = MAX_ROLLING_POINTS,
) -> Path:
    """Build the dashboard shell and factor-scoped compressed JSON caches."""
    directory = Path(report_dir).resolve()
    required = list(CACHE_SOURCE_FILES)
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing analysis reports in {directory}: {', '.join(missing)}"
        )
    if max_rolling_points < 2:
        raise ValueError("max_rolling_points must be at least 2")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    factors = metadata.get("factors", [])
    if not factors:
        raise ValueError("metadata.json contains no analyzed factors")
    cache = directory / CACHE_DIR_NAME
    detail_dir = cache / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    valid_factor_keys = {factor_cache_key(factor) for factor in factors}
    for stale_file in detail_dir.glob("*.json.gz"):
        stale_file.unlink()
    for stale_dir in detail_dir.iterdir():
        if stale_dir.is_dir() and stale_dir.name not in valid_factor_keys:
            shutil.rmtree(stale_dir)

    def write_cache(path: Path, value: object) -> int:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        path.write_bytes(compressed)
        return len(compressed)

    reports = {name: _read_records(directory, name) for name in REPORT_COLUMNS}
    diagnostics = reports["diagnostics"]
    rolling_path = directory / "rolling_ic.csv"
    print(
        f"Streaming rolling IC ({rolling_path.stat().st_size / (1024 ** 3):.2f} GiB) to prepare factor caches...",
        flush=True,
    )
    rolling = _rolling_records(directory, diagnostics, metadata, max_rolling_points)
    metadata["dashboard_rolling_point_cap"] = max_rolling_points
    by_factor = {factor: {} for factor in factors}
    for name, rows in reports.items():
        for row in rows:
            by_factor.setdefault(row["factor"], {}).setdefault(name, []).append(row)
    for row in rolling:
        by_factor.setdefault(row["factor"], {}).setdefault("rolling_ic", []).append(row)

    overview_size = write_cache(
        cache / "overview.json.gz",
        {"metadata": metadata, "predictive": reports["predictive"]},
    )
    detail_size = 0
    horizons = [int(value) for value in metadata["config"]["horizons"]]
    for factor, detail in by_factor.items():
        factor_dir = detail_dir / factor_cache_key(factor)
        factor_dir.mkdir(parents=True, exist_ok=True)
        for horizon in horizons:
            horizon_detail = dict(detail)
            horizon_detail["quantiles"] = [
                row
                for row in detail.get("quantiles", [])
                if int(row["horizon"]) == horizon
            ]
            horizon_detail["rolling_ic"] = [
                row
                for row in detail.get("rolling_ic", [])
                if int(row["horizon"]) == horizon
            ]
            detail_size += write_cache(
                factor_dir / f"{horizon}.json.gz", horizon_detail
            )
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "factor_count": len(factors),
                "horizons": horizons,
                "rolling_point_cap": max_rolling_points,
                "source_signature": report_source_signature(directory),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (cache / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    html_template = Path(__file__).with_name("factor_dashboard_template.html")
    html = html_template.read_text(encoding="utf-8")
    output_path = Path(output) if output else directory / "factor_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(
        f"Dashboard shell: {output_path.stat().st_size / 1024:.1f} KiB; overview cache: {overview_size / 1024:.1f} KiB; factor caches: {detail_size / (1024 ** 2):.1f} MiB.",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "output"
        / "data"
        / "factor_analysis",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rolling-points", type=int, default=MAX_ROLLING_POINTS)
    args = parser.parse_args()
    path = build_dashboard(args.report_dir, args.output, args.max_rolling_points)
    print(f"Dashboard written to {path.resolve()}")


if __name__ == "__main__":
    main()
