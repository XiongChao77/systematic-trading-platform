"""HTTP routes for prepared-data factor analysis."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from UI.backend.modules.factors import service


router = APIRouter(prefix="/api/factors", tags=["Factors"])


class AnalysisInput(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=500)
    factors: list[str] | None = None


class ReportInput(BaseModel):
    report_path: str = Field(min_length=1, max_length=1000)


@router.get("/browse")
def browse_reports(path: str = "") -> dict[str, Any]:
    try:
        return service.browse_report_directory(path)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/datasets")
def datasets() -> dict[str, Any]:
    return service.list_datasets()


@router.get("/status")
def status() -> dict[str, Any]:
    return service.get_status()


@router.post("/load")
def load_report(payload: ReportInput) -> dict[str, Any]:
    try:
        return service.load_report(payload.report_path)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyze", status_code=202)
def analyze(payload: AnalysisInput) -> dict[str, Any]:
    try:
        return service.start_analysis(payload.dataset_id, payload.factors)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/dashboard")
def dashboard(report_path: str) -> FileResponse:
    try:
        path = service.dashboard_path(report_path)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/dashboard/overview")
def dashboard_overview(report_path: str, horizon: int | None = None) -> JSONResponse:
    try:
        result = service.dashboard_overview_data(report_path, horizon)
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.get("/dashboard/detail")
def dashboard_detail(report_path: str, factor: str, horizon: int) -> FileResponse:
    try:
        path = service.dashboard_detail_path(report_path, factor, horizon)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/json",
        headers={
            "Content-Encoding": "gzip",
            "Vary": "Accept-Encoding",
            "Cache-Control": "no-store",
        },
    )


@router.get("/dashboard/plotly")
def dashboard_plotly(report_path: str) -> FileResponse:
    try:
        path = service.dashboard_cache_path(report_path, "plotly")
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/javascript")
