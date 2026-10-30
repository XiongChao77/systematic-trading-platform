"""HTTP routes for standalone and experiment backtest details."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from UI.backend.modules.backtests import service


router = APIRouter(prefix="/api/backtests", tags=["backtests"])


@router.get("/health")
def health() -> dict[str, Any]:
    report_path = service.latest_backtest_path()
    available = report_path.is_file()
    return {
        "status": "ok",
        "report_path": str(report_path),
        "report_available": available,
        "report_size": report_path.stat().st_size if available else None,
    }


@router.get("/latest")
def latest() -> FileResponse:
    report_path = service.latest_backtest_path()
    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No standalone backtest report exists at {report_path}. "
                "Run trade/runner/backtest_runner.py first."
            ),
        )
    return FileResponse(
        report_path,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/experiment/{dataset_id}/{record_id}")
def experiment_detail(
    dataset_id: str,
    record_id: str,
    period: Literal["long", "forward"] = "long",
) -> dict[str, Any]:
    try:
        return service.load_experiment_backtest(dataset_id, record_id, period)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiment/{dataset_id}/{record_id}/complete")
def complete_experiment_detail(
    dataset_id: str,
    record_id: str,
) -> dict[str, Any]:
    """Return one continuous Train/Valid/Test/OOS strategy payload."""

    try:
        return service.load_experiment_backtest(dataset_id, record_id, "all")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
