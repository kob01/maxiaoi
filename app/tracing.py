"""LangSmith tracing bootstrap.

LangChain / LangGraph auto-emit traces when the standard ``LANGSMITH_*``
environment variables are present. This module is the single place that
turns that machinery on/off from :class:`app.config.Settings`, so callers
never touch ``os.environ`` directly.

Compliance note (intranet-only deployment):
    Tracing uploads prompt/completion payloads to ``langsmith_endpoint``.
    It MUST stay disabled in production containers. Enable it only on a
    developer machine by setting ``LANGSMITH_TRACING=true`` and a personal
    ``LANGSMITH_API_KEY`` in the local ``.env`` (git-ignored).
"""

from __future__ import annotations

import logging
import os

from app.config import get_settings

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing() -> bool:
    """Configure LangSmith env vars from settings; return whether active.

    Idempotent: safe to call from both the FastAPI lifespan and the
    LangGraph Studio graph factory.
    """
    global _initialized
    settings = get_settings()

    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        # Explicitly ensure tracing stays off so a stray env var in the
        # container does not silently upload data.
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return False

    os.environ["LANGSMITH_TRACING"] = settings.langsmith_tracing
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    if not _initialized:
        logger.info(
            "LangSmith tracing enabled -> project=%s endpoint=%s",
            settings.langsmith_project,
            settings.langsmith_endpoint,
        )
        _initialized = True
    return True
