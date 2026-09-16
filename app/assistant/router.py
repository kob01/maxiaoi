"""FastAPI router exposing the single Assistant entry point."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.assistant.graph import get_orchestrator
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["assistant"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Unified chat endpoint: every user message enters here."""
    try:
        return await get_orchestrator().handle(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
