"""HTTP ingestion and read APIs for live strategy monitoring."""

from __future__ import annotations

from typing import Any
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from UI.backend.modules.live.models import RunnerSnapshot
from UI.backend.modules.live.store import (
    LiveSnapshotConflict,
    live_snapshot_store,
)


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


@router.delete("/api/live/viewers/{viewer_id}")
async def release_viewer(viewer_id: str) -> dict[str, bool]:
    live_snapshot_store.unwatch(viewer_id)
    return {"released": True}


@router.get("/internal/live/demand/{runner_id}")
def monitoring_demand(runner_id: str, version: int = -1) -> dict[str, Any]:
    return live_snapshot_store.wait_for_demand(runner_id, version)
