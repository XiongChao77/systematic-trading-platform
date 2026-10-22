"""Unified Strategy Center API and frontend deployment entry point.

Run from the repository root with::

    uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000

For development, Vite serves the frontend and proxies ``/api`` to this app.
For deployment, build ``UI/quant-ui`` first and FastAPI serves its ``dist``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from UI.backend.modules.backtests import router as backtests_router
from UI.backend.modules.experiments import router as experiments_router
from UI.backend.modules.factors import router as factors_router
from UI.backend.modules.labels import router as labels_router
from UI.backend.modules.live import router as live_router


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "UI" / "quant-ui" / "dist"


def create_app() -> FastAPI:
    """Build the single FastAPI application used by every UI module."""

    application = FastAPI(title="Strategy Center", version="1.0")
    application.add_middleware(GZipMiddleware, minimum_size=1_024)
    application.include_router(experiments_router)
    application.include_router(factors_router)
    application.include_router(backtests_router)
    application.include_router(labels_router)
    application.include_router(live_router)

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "application": "Strategy Center",
            "modules": ["live", "factors", "experiments", "backtests", "labels"],
            "frontend_built": (FRONTEND_DIST / "index.html").is_file(),
        }

    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="frontend-assets",
        )

    @application.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str) -> FileResponse:
        if frontend_path == "api" or frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")

        index_path = FRONTEND_DIST / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    "The Strategy Center frontend has not been built. "
                    "Run npm run build in UI/quant-ui."
                ),
            )

        requested = (FRONTEND_DIST / frontend_path).resolve()
        try:
            requested.relative_to(FRONTEND_DIST.resolve())
        except ValueError:
            requested = index_path
        if frontend_path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(index_path)

    return application


app = create_app()
