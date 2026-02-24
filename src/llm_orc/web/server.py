"""FastAPI server for llm-orc web UI.

Provides REST API endpoints for ensemble management and execution,
with WebSocket support for streaming execution updates.
"""

import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from llm_orc.web.api import artifacts, ensembles, profiles, scripts

logger = logging.getLogger(__name__)

# Static files directory (built frontend assets)
STATIC_DIR = Path(__file__).parent / "static"


def get_version() -> str:
    """Get the llm-orchestra package version."""
    try:
        return version("llm-orchestra")
    except Exception:
        return "0.0.0"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="llm-orc",
        description="Web UI for llm-orc ensemble management",
        version=get_version(),
    )

    # CORS middleware - localhost only by default
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Return a consistent JSON body for HTTP errors."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch unhandled exceptions and return structured JSON."""
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # Register API routers
    app.include_router(ensembles.router)
    app.include_router(profiles.router)
    app.include_router(scripts.router)
    app.include_router(artifacts.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": get_version()}

    # Serve static files if they exist (production mode)
    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        # Mount static assets
        assets_dir = STATIC_DIR / "assets"
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/")
        async def serve_spa() -> FileResponse:
            """Serve the SPA index.html."""
            return FileResponse(STATIC_DIR / "index.html")

        # Catch-all for SPA routing
        @app.get("/{path:path}", response_class=FileResponse)
        async def spa_fallback(path: str) -> FileResponse:
            """Fallback to index.html for SPA routing."""
            return FileResponse(STATIC_DIR / "index.html")
    else:
        # Development mode - return API info
        @app.get("/")
        async def root() -> dict[str, Any]:
            """Return API information (development mode)."""
            return {
                "name": "llm-orc",
                "version": get_version(),
                "description": "Web UI for llm-orc ensemble management",
                "mode": "development",
                "note": "Run 'npm run build' in frontend/ to build static assets",
                "endpoints": {
                    "health": "/health",
                    "ensembles": "/api/ensembles",
                    "profiles": "/api/profiles",
                    "scripts": "/api/scripts",
                    "artifacts": "/api/artifacts",
                },
            }

    return app
