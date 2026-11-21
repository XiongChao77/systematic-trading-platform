"""HTTP ingestion and read APIs for live strategy monitoring."""

from __future__ import annotations

from typing import Any
import json
import logging
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from data_process import common

from UI.backend.modules.live.models import RunnerSnapshot
from UI.backend.modules.live.store import (
    LiveSnapshotConflict,
    live_snapshot_store,
)
from trade.recording.strategy_curves import CURVES_REPORT_NAME


logger = logging.getLogger("uvicorn.error")


class TimedLiveRoute(APIRoute):
    """Time request validation, handler execution, and response serialization."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def timed_handler(request: Request):
            started = time.monotonic()
            status = 500
            try:
                response = await handler(request)
                status = response.status_code
                return response
            except HTTPException as exc:
                status = exc.status_code
                raise
            except RequestValidationError:
                status = 422
                raise
            finally:
                logger.info(
                    "Live API timing | method=%s path=%s status=%s handler_ms=%.1f",
                    request.method, request.url.path, status,
                    (time.monotonic() - started) * 1000,
                )

        return timed_handler


router = APIRouter(tags=["live"], route_class=TimedLiveRoute)


@router.get("/api/live/overview")
def portfolio_overview() -> dict[str, Any]:
    root = Path(common.PERSISTENCE_DIR) / "live_runner"
    reports = sorted(
        root.glob("*/bitget_performance/equity_overview.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return {"available": False}
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    return {"available": True, **payload}


@router.post("/internal/live/snapshots")
async def publish_snapshot(payload: RunnerSnapshot) -> dict[str, Any]:
    try:
        result = live_snapshot_store.update(payload)
        return result
    except LiveSnapshotConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/live/strategies")
async def strategies(viewer_id: str = "") -> dict[str, Any]:
    if viewer_id:
        live_snapshot_store.watch(viewer_id)
    return live_snapshot_store.strategies()


@router.get("/api/live/strategies/{instance_id}")
async def strategy_detail(instance_id: str, viewer_id: str = "") -> dict[str, Any]:
    strategy = live_snapshot_store.strategy(instance_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Live strategy not found")
    if viewer_id:
        live_snapshot_store.watch(viewer_id, instance_id)
    return strategy


@router.get("/api/live/strategies/{instance_id}/curve")
def strategy_equity_curve(instance_id: str) -> dict[str, Any]:
    strategy = live_snapshot_store.strategy(instance_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Live strategy not found")
    runner_id = strategy["runner_id"]
    if strategy["venue"].casefold() != "bitget" or runner_id in {".", ".."} or Path(runner_id).name != runner_id:
        return {"available": False}
    path = Path(common.PERSISTENCE_DIR) / "live_runner" / runner_id / "bitget_performance" / CURVES_REPORT_NAME
    try:
        curves = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"available": False}
    curve = curves.get(strategy["account_id"])
    if not curve or curve["strategy_hash"] != strategy["strategy_hash"]:
        return {"available": False}
    return {"available": True, **curve}


@router.delete("/api/live/viewers/{viewer_id}")
async def release_viewer(viewer_id: str) -> dict[str, bool]:
    live_snapshot_store.unwatch(viewer_id)
    return {"released": True}


@router.get("/internal/live/demand/{runner_id}")
def monitoring_demand(runner_id: str, version: int = -1) -> dict[str, Any]:
    return live_snapshot_store.wait_for_demand(runner_id, version)
