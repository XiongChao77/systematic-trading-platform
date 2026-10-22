"""Background execution and report discovery for factor analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
import logging
import os
from pathlib import Path
import re
from threading import Lock, Thread
from time import monotonic
from typing import Any

from data_process.analyse.report_cache import report_source_signature

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = (
    Path(os.environ.get("FACTOR_DATA_ROOT", PROJECT_ROOT / "output"))
    .expanduser()
    .resolve()
)
REPORT_NAME = "factor_analysis"
FACTOR_LOGGER = logging.getLogger("data_process.analyse.factor_analysis")
_LOCK = Lock()
_JOB: dict[str, Any] = {
    "status": "idle",
    "dataset_id": None,
    "started_at": None,
    "finished_at": None,
    "report_path": None,
    "progress": None,
    "message": "No factor analysis has been run in this session.",
    "error": None,
}


class AnalysisProgressHandler(logging.Handler):
    """Capture factor-level progress from the analysis logger."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        progress = re.search(r"Factor progress (\d+)/(\d+): (.+)", message)
        with _LOCK:
            _JOB["message"] = message
            if progress:
                _JOB["progress"] = {
                    "completed": int(progress.group(1)),
                    "total": int(progress.group(2)),
                    "factor": progress.group(3),
                }


def _dataset_path(dataset_id: str) -> Path:
    candidate = (DATA_ROOT / dataset_id).resolve()
    try:
        candidate.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Dataset must be inside the configured factor data root"
        ) from exc
    if (
        not (candidate / "train_data.feather").is_file()
        or not (candidate / "test_data.feather").is_file()
    ):
        raise FileNotFoundError(
            "Dataset must contain train_data.feather and test_data.feather"
        )
    return candidate


def _report_path(report_path: str) -> Path:
    candidate = Path(report_path).expanduser()
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == PROJECT_ROOT.name:
            candidate = Path(*candidate.parts[1:])
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    allowed_roots = {PROJECT_ROOT, DATA_ROOT}
    if not any(
        candidate == root or root in candidate.parents for root in allowed_roots
    ):
        raise ValueError(
            "Report path must be inside the project or configured factor data root"
        )
    return candidate


def _display_report_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def browse_report_directory(path: str = "") -> dict[str, Any]:
    """List child directories that can be opened as saved factor reports."""
    current = PROJECT_ROOT.resolve() if not path else _report_path(path)
    if not current.is_dir():
        raise ValueError(f"Not a directory: {path}")

    allowed_root = PROJECT_ROOT.resolve()
    if allowed_root not in current.parents and current != allowed_root:
        allowed_root = DATA_ROOT.resolve()
    children = []
    for child in sorted(current.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_dir():
            continue
        try:
            resolved_child = _report_path(str(child))
        except ValueError:
            continue
        children.append(
            {
                "name": child.name,
                "path": _display_report_path(resolved_child),
                "has_report": (resolved_child / "metadata.json").is_file(),
            }
        )

    return {
        "root": _display_report_path(allowed_root),
        "current": _display_report_path(current),
        "parent": (
            None
            if current == allowed_root
            else _display_report_path(current.parent)
        ),
        "direct_report": (current / "metadata.json").is_file(),
        "children": children,
    }


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _report_summary(report_dir: Path) -> dict[str, Any]:
    metadata_path = report_dir / "metadata.json"
    dashboard_path = report_dir / "factor_dashboard.html"
    dashboard_cache = report_dir / "factor_dashboard_data"
    cache_files_exist = (
        dashboard_path.is_file()
        and (dashboard_cache / "overview.json.gz").is_file()
        and (dashboard_cache / "plotly.min.js").is_file()
        and (dashboard_cache / "manifest.json").is_file()
    )
    if not metadata_path.is_file():
        return {
            "available": False,
            "dashboard_available": False,
            "report_path": _display_path(report_dir),
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dashboard_ready = False
    if cache_files_exist:
        try:
            manifest = json.loads(
                (dashboard_cache / "manifest.json").read_text(encoding="utf-8")
            )
            dashboard_ready = manifest.get("source_signature") == report_source_signature(
                report_dir
            )
        except (OSError, ValueError):
            dashboard_ready = False
    return {
        "available": True,
        "dashboard_available": dashboard_ready,
        "report_path": _display_path(report_dir),
        "factor_count": len(metadata.get("factors", [])),
        "row_count": metadata.get("row_count"),
        "start": metadata.get("start"),
        "end": metadata.get("end"),
        "horizon_labels": metadata.get("horizon_labels", {}),
    }


def list_datasets() -> dict[str, Any]:
    """Find prepared train/test pairs under the configured data root."""
    datasets = []
    if DATA_ROOT.is_dir():
        for train_file in DATA_ROOT.rglob("train_data.feather"):
            if any(
                part in {"node_modules", ".git", "dist"} for part in train_file.parts
            ):
                continue
            dataset = train_file.parent.resolve()
            try:
                dataset_id = dataset.relative_to(DATA_ROOT).as_posix()
            except ValueError:
                continue
            if not (dataset / "test_data.feather").is_file():
                continue
            datasets.append(
                {
                    "id": dataset_id,
                    "label": dataset_id or DATA_ROOT.name,
                    **_report_summary(dataset / REPORT_NAME),
                }
            )
    datasets.sort(key=lambda item: (item["id"] != "data", item["id"].lower()))
    return {"data_root": str(DATA_ROOT), "datasets": datasets}


def _run_analysis(dataset_id: str, factors: list[str] | None) -> None:
    dataset = _dataset_path(dataset_id)
    from data_process.analyse.factor_analysis import analyze
    from data_process.analyse.factor_dashboard import build_dashboard

    handler = AnalysisProgressHandler()
    previous_level = FACTOR_LOGGER.level
    FACTOR_LOGGER.setLevel(logging.INFO)
    FACTOR_LOGGER.addHandler(handler)
    started = monotonic()
    try:
        analyze(dataset, dataset / REPORT_NAME, factors=factors)
        build_dashboard(dataset / REPORT_NAME)
        with _LOCK:
            _JOB.update(
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                report_path=_display_path(dataset / REPORT_NAME),
                message=f"Analysis and dashboard complete in {monotonic() - started:.1f}s.",
            )
    except Exception as exc:
        logging.getLogger(__name__).exception("Factor analysis failed")
        with _LOCK:
            _JOB.update(
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
                message="Factor analysis failed.",
            )
    finally:
        FACTOR_LOGGER.removeHandler(handler)
        FACTOR_LOGGER.setLevel(previous_level)


def start_analysis(dataset_id: str, factors: list[str] | None) -> dict[str, Any]:
    """Start one factor analysis job in a background thread."""
    _dataset_path(dataset_id)
    with _LOCK:
        if _JOB["status"] == "running":
            raise RuntimeError("A factor analysis job is already running")
        _JOB.update(
            status="running",
            dataset_id=dataset_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            report_path=_display_path(_dataset_path(dataset_id) / REPORT_NAME),
            progress=None,
            message="Starting factor analysis.",
            error=None,
        )
    Thread(
        target=_run_analysis,
        args=(dataset_id, factors),
        name="factor-analysis",
        daemon=True,
    ).start()
    return get_status()


def get_status() -> dict[str, Any]:
    with _LOCK:
        return dict(_JOB)


def load_report(report_path: str) -> dict[str, Any]:
    """Load report metadata and prepare dashboard caches if they are missing."""
    report_dir = _report_path(report_path)
    summary = _report_summary(report_dir)
    if not summary["available"]:
        raise FileNotFoundError(f"No factor analysis metadata found at {report_dir}")
    if not summary["dashboard_available"]:
        from data_process.analyse.factor_dashboard import build_dashboard

        build_dashboard(report_dir)
        summary = _report_summary(report_dir)
    return summary


def dashboard_path(report_path: str) -> Path:
    """Return a contained dashboard path or raise when it is not ready."""
    report_dir = _report_path(report_path)
    summary = _report_summary(report_dir)
    if not summary["available"]:
        raise FileNotFoundError("No factor analysis metadata found at this report path")
    if not summary["dashboard_available"]:
        from data_process.analyse.factor_dashboard import build_dashboard

        build_dashboard(report_dir)
    path = report_dir / "factor_dashboard.html"
    if not path.is_file():
        raise FileNotFoundError(
            "Factor dashboard is not available for this report path"
        )
    return path


def dashboard_cache_path(report_path: str, cache_name: str) -> Path:
    """Resolve dashboard data or Plotly cache files without allowing path traversal."""
    report_dir = _report_path(report_path)
    if not _report_summary(report_dir)["available"]:
        raise FileNotFoundError("No factor analysis metadata found at this report path")
    cache_dir = report_dir / "factor_dashboard_data"
    if cache_name == "overview":
        path = cache_dir / "overview.json.gz"
    elif cache_name == "plotly":
        path = cache_dir / "plotly.min.js"
    else:
        raise ValueError("Unknown dashboard cache")
    if not path.is_file():
        raise FileNotFoundError(f"Dashboard cache is missing: {cache_name}")
    return path


def dashboard_detail_path(report_path: str, factor: str, horizon: int) -> Path:
    """Resolve a factor detail cache after validating factor and horizon."""
    report_dir = _report_path(report_path)
    metadata_path = report_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("No factor analysis metadata found at this report path")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if factor not in metadata.get("factors", []):
        raise ValueError(f"Unknown factor: {factor}")
    if int(horizon) not in {
        int(value) for value in metadata.get("config", {}).get("horizons", [])
    }:
        raise ValueError(f"Unsupported return horizon: {horizon}")
    from data_process.analyse.factor_dashboard import factor_cache_key

    path = (
        report_dir
        / "factor_dashboard_data"
        / "details"
        / factor_cache_key(factor)
        / f"{int(horizon)}.json.gz"
    )
    if not path.is_file():
        raise FileNotFoundError("Factor detail cache is missing")
    return path


def dashboard_overview_data(
    report_path: str, horizon: int | None = None
) -> dict[str, Any]:
    """Return overview metadata and rows for a single supported horizon."""
    report_dir = _report_path(report_path)
    cache_path = dashboard_cache_path(report_path, "overview")
    if horizon is None:
        import json

        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        horizon = int(data["metadata"]["config"]["horizons"][0])
    import json

    metadata = json.loads((report_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(horizon) not in {
        int(value) for value in metadata.get("config", {}).get("horizons", [])
    }:
        raise ValueError(f"Unsupported return horizon: {horizon}")
    from data_process.analyse.factor_dashboard import read_overview

    return read_overview(report_dir, int(horizon))
