"""FastAPI application entry: single Assistant gateway + Web UI.

Startup initialises the MySQL metadata schema (prompting for the password
once in the terminal); if MySQL is unavailable the gateway still serves chat
with degraded metadata while the document-management APIs report errors.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.assistant.router import router as assistant_router
from app.docs.router import router as docs_router

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Init MySQL metadata schema at startup (getpass password prompt happens here)."""
    from app.tracing import init_tracing

    if init_tracing():
        logger.info("LangSmith tracing active for this gateway process")
    try:
        from app.db.session import init_schema

        await init_schema()
        logger.info("MySQL metadata schema ready")
    except Exception as exc:
        logger.error(
            "MySQL 初始化失败, 文档管理功能不可用, 聊天将降级为无标签模式: %s", exc
        )
    yield


def create_app() -> FastAPI:
    """Build the gateway application."""
    app = FastAPI(title="MXI Enterprise Multi-Agent Assistant", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(assistant_router)
    app.include_router(docs_router)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/upload", include_in_schema=False)
    async def upload_page() -> FileResponse:
        return FileResponse(WEB_DIR / "upload.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app()
