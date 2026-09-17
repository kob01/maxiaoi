"""Document management API: upload -> tag suggestion -> confirm ingest."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.docs import service
from app.docs.parsers import modality_of, parse_blocks
from app.security.audit import get_audit_logger, new_trace_id

router = APIRouter(prefix="/api/docs", tags=["docs"])


class IngestRequest(BaseModel):
    """Phase-2 confirmation payload from the upload page."""

    doc_key: str
    filename: str
    tags: list[str] = Field(default_factory=list)
    uploader: str = "anonymous"


@router.post("/upload")
async def upload_doc(file: UploadFile = File(...), uploader: str = Form("anonymous")) -> dict:
    """Phase 1: save + parse + duplicate check + LLM tag suggestion."""
    trace_id = new_trace_id()
    data = await file.read()
    try:
        doc_key, path, ext = service.save_upload(file.filename or "unnamed", data)
        _, blocks = await parse_blocks(path)
    except service.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:  # e.g. vision model unavailable
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    existing = await service.check_existing(doc_key)
    preview = "\n\n".join(b.text for b in blocks)
    tags = await service.suggest_tags(preview)
    get_audit_logger().log(
        trace_id, "docs", "upload_received",
        {"doc_key": doc_key, "filename": path.name, "ext": ext,
         "overwrite": existing is not None, "uploader": uploader},
    )
    return {
        "doc_key": doc_key,
        "filename": path.name,
        "ext": ext,
        "modality": modality_of(path),
        "overwritten": existing is not None,
        "suggested_tags": tags,
        "preview_len": len(preview),
        "preview": preview[:500],
    }


@router.post("/ingest")
async def ingest_doc(req: IngestRequest) -> dict:
    """Phase 2: chunk + embed + Milvus overwrite + MySQL metadata."""
    trace_id = new_trace_id()
    try:
        result = await service.ingest_confirmed(req.doc_key, req.filename, req.tags, req.uploader)
    except service.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    get_audit_logger().log(
        trace_id, "docs", "ingest_completed",
        {"doc_key": req.doc_key, "filename": req.filename, "tags": req.tags,
         "chunk_count": result["chunk_count"], "uploader": req.uploader},
    )
    return result


@router.get("")
async def list_docs() -> list[dict]:
    """Document list for the management page."""
    return await service.list_documents()


@router.get("/tags")
async def list_all_tags() -> list[dict]:
    """All known tags."""
    return await service.list_tags()
