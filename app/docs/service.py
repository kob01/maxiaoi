"""Document upload / ingestion / metadata service layer."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from app.config import get_settings
from app.db.models import Document, DocumentTag, Tag
from app.db.session import get_session_factory
from app.docs.parsers import modality_of, parse_blocks, supported_extensions
from app.rag.embeddings import OllamaEmbedder
from app.rag.ingest import compute_doc_id, ingest_blocks
from app.rag.vectorstore import MilvusStore

logger = logging.getLogger(__name__)

TAG_PROMPT = """你是企业知识库的分类助手。基于文档内容,给出 3~5 个中文分类标签。
优先复用已有标签: {existing}

严格输出 JSON 数组, 不要输出其他内容, 例如: ["财务","报销","制度"]

文档内容节选:
{excerpt}"""


class UploadError(ValueError):
    """Raised for invalid uploads (bad type / oversize / parse failure)."""


def _upload_dir() -> Path:
    settings = get_settings()
    path = Path(settings.upload_dir)
    if not path.is_absolute():
        path = settings.base_dir / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(filename: str) -> str:
    """Strip path components; keep only the bare file name."""
    name = Path(filename).name.strip()
    if not name or name.startswith("."):
        raise UploadError("非法文件名")
    return name


def save_upload(filename: str, data: bytes) -> tuple[str, Path, str]:
    """Validate and persist an uploaded file. Returns (doc_key, path, ext)."""
    name = _safe_filename(filename)
    ext = Path(name).suffix.lower()
    if name.lower().endswith(".transcript.txt"):
        ext = ".transcript.txt"
    if ext not in supported_extensions() and not any(
        name.lower().endswith(e) for e in (".srt", ".vtt")
    ):
        raise UploadError(f"不支持的文件类型: {ext}")
    max_bytes = get_settings().upload_max_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise UploadError(f"文件超过大小限制({get_settings().upload_max_mb}MB)")

    stem = name[: -len(ext)] if name.lower().endswith(ext) else Path(name).stem
    doc_key = compute_doc_id(stem, ext)
    dest_dir = _upload_dir() / doc_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    dest.write_bytes(data)
    return doc_key, dest, ext


async def check_existing(doc_key: str) -> Document | None:
    """Look up an existing document (same name+ext) by doc_key."""
    async with get_session_factory()() as session:
        result = await session.execute(select(Document).where(Document.doc_key == doc_key))
        return result.scalar_one_or_none()


async def _existing_tag_names() -> list[str]:
    async with get_session_factory()() as session:
        result = await session.execute(select(Tag.name).order_by(Tag.name))
        return [r[0] for r in result.all()]


async def suggest_tags(text: str) -> list[str]:
    """Ask the LLM for 3~5 category tags; falls back to [] on any failure."""
    from app.llm import get_chat_model

    settings = get_settings()
    existing = await _existing_tag_names()
    llm = get_chat_model(settings.llm_model, temperature=0.2)
    prompt = TAG_PROMPT.format(
        existing="、".join(existing) if existing else "(暂无)",
        excerpt=text[:3000],
    )
    try:
        resp = await llm.ainvoke(prompt)
        raw = str(resp.content)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        tags = json.loads(match.group(0)) if match else []
        return [str(t).strip() for t in tags if str(t).strip()][:5]
    except Exception as exc:
        logger.warning("tag suggestion failed: %s", exc)
        return []


async def ingest_confirmed(
    doc_key: str, filename: str, tags: list[str], uploader: str
) -> dict[str, Any]:
    """Phase-2 ingest: parse -> chunk -> embed -> Milvus overwrite -> MySQL.

    Returns a summary dict for the API response.
    """
    path = _upload_dir() / doc_key / _safe_filename(filename)
    if not path.exists():
        raise UploadError("上传文件不存在, 请重新上传")

    modality = modality_of(path)
    _, blocks = await parse_blocks(path)
    parsed_text = "\n\n".join(b.text for b in blocks)

    store = MilvusStore()
    embedder = OllamaEmbedder()
    title = path.stem
    chunk_count = await ingest_blocks(
        doc_key, path.name, title, str(path), modality, blocks, store, embedder
    )
    if not chunk_count:
        raise UploadError("文档解析后无有效内容")

    # --- MySQL metadata (transactional) ---
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            existing = await session.execute(
                select(Document).where(Document.doc_key == doc_key)
            )
            doc = existing.scalar_one_or_none()
            if doc is None:
                doc = Document(doc_key=doc_key, name=title, ext=path.suffix.lower(),
                               modality=modality, created_by=uploader)
                session.add(doc)
            doc.file_path = str(path)
            doc.parsed_text = parsed_text
            doc.chunk_count = chunk_count
            doc.size_bytes = path.stat().st_size
            doc.modality = modality

            await session.execute(delete(DocumentTag).where(DocumentTag.doc_key == doc_key))
            for tag_name in dict.fromkeys(t.strip() for t in tags if t.strip()):
                tag_id = await _get_or_create_tag(session, tag_name)
                session.add(DocumentTag(doc_key=doc_key, tag_id=tag_id))

    # --- refresh the BM25 channel of the running assistant, if any ---
    try:
        from app.assistant.graph import get_orchestrator

        await get_orchestrator().refresh_knowledge()
    except Exception as exc:  # BM25 rebuilds on next startup anyway
        logger.warning("bm25 refresh after ingest failed: %s", exc)

    return {"doc_key": doc_key, "chunk_count": chunk_count, "tags": tags, "modality": modality}


async def delete_document(doc_key: str) -> dict[str, Any]:
    """Delete a document: Milvus chunks + MySQL metadata + upload files."""
    factory = get_session_factory()
    async with factory() as session:
        doc = (
            await session.execute(select(Document).where(Document.doc_key == doc_key))
        ).scalar_one_or_none()
        if doc is None:
            raise UploadError("文档不存在或已删除")
        name, file_path = doc.name, doc.file_path

        # Milvus vectors first; MySQL row is the source of truth, so a vector
        # failure aborts before metadata is lost (chunk leftovers can be
        # purged by a re-ingest, but a lost metadata row orphans nothing).
        MilvusStore().delete_by_doc(doc_key)
        await session.execute(delete(DocumentTag).where(DocumentTag.doc_key == doc_key))
        await session.execute(delete(Document).where(Document.doc_key == doc_key))
        await session.commit()

    # remove the uploaded file directory (best effort)
    if file_path:
        shutil.rmtree(Path(file_path).parent, ignore_errors=True)

    # --- refresh the BM25 channel of the running assistant, if any ---
    try:
        from app.assistant.graph import get_orchestrator

        await get_orchestrator().refresh_knowledge()
    except Exception as exc:  # BM25 rebuilds on next startup anyway
        logger.warning("bm25 refresh after delete failed: %s", exc)

    logger.info("document deleted: doc_key=%s name=%s", doc_key, name)
    return {"doc_key": doc_key, "name": name}


async def _get_or_create_tag(session, name: str) -> int:
    result = await session.execute(select(Tag).where(Tag.name == name))
    tag = result.scalar_one_or_none()
    if tag is None:
        tag = Tag(name=name, source="custom")
        session.add(tag)
        await session.flush()
    return int(tag.id)


async def get_meta_map(doc_keys: list[str]) -> dict[str, dict[str, Any]]:
    """Batch-load document metadata (name/tags/modality) for chat citations."""
    if not doc_keys:
        return {}
    factory = get_session_factory()
    async with factory() as session:
        docs = (
            await session.execute(select(Document).where(Document.doc_key.in_(doc_keys)))
        ).scalars().all()
        tag_rows = (
            await session.execute(
                select(DocumentTag.doc_key, Tag.name)
                .join(Tag, Tag.id == DocumentTag.tag_id)
                .where(DocumentTag.doc_key.in_(doc_keys))
            )
        ).all()
    tags_by_doc: dict[str, list[str]] = {}
    for key, tag_name in tag_rows:
        tags_by_doc.setdefault(key, []).append(tag_name)
    return {
        d.doc_key: {"name": d.name, "modality": d.modality, "tags": tags_by_doc.get(d.doc_key, [])}
        for d in docs
    }


async def list_documents() -> list[dict[str, Any]]:
    """Document list with tags for the management page."""
    factory = get_session_factory()
    async with factory() as session:
        docs = (await session.execute(select(Document).order_by(Document.updated_at.desc()))).scalars().all()
        keys = [d.doc_key for d in docs]
        meta = await get_meta_map(keys) if keys else {}
    return [
        {
            "doc_key": d.doc_key,
            "name": d.name,
            "ext": d.ext,
            "modality": d.modality,
            "chunk_count": d.chunk_count,
            "size_bytes": d.size_bytes,
            "tags": meta.get(d.doc_key, {}).get("tags", []),
            "created_by": d.created_by,
            "updated_at": d.updated_at.isoformat(timespec="seconds") if d.updated_at else "",
        }
        for d in docs
    ]


async def list_tags() -> list[dict[str, Any]]:
    """All tags for the management page."""
    factory = get_session_factory()
    async with factory() as session:
        tags = (await session.execute(select(Tag).order_by(Tag.name))).scalars().all()
    return [{"id": t.id, "name": t.name, "source": t.source} for t in tags]
