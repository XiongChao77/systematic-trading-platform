"""Shared freshness checks for cached factor dashboard data."""

from __future__ import annotations

import hashlib
from pathlib import Path


CACHE_SOURCE_FILES = (
    "metadata.json",
    "predictive.csv",
    "decay.csv",
    "quantiles.csv",
    "autocorrelation.csv",
    "diagnostics.csv",
    "rolling_ic.csv",
)


def report_source_signature(report_dir: str | Path) -> str:
    """Return a stable signature that changes when any dashboard source changes."""
    directory = Path(report_dir)
    entries = []
    for name in CACHE_SOURCE_FILES:
        stat = (directory / name).stat()
        entries.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
