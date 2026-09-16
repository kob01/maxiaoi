"""Knowledge ingestion pipeline.

Stages: parse (txt/md/pdf/docx + video transcripts) -> chunk -> embed -> upsert
into Milvus, then rebuild the BM25 channel. Re-ingesting the same file is
idempotent: existing chunks of the doc are deleted first (dynamic update).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

from app.rag.embeddings import OllamaEmbedder
from app.rag.vectorstore import MilvusStore
from app.schemas import KnowledgeChunk

SUPPORTED_TEXT_EXT = {".txt", ".md", ".pdf", ".docx"}
SUPPORTED_VIDEO_EXT = {".srt", ".vtt", ".transcript.txt"}

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _doc_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


def _chunk_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}-{idx:05d}"


def parse_text_file(path: Path) -> str:
    """Extract raw text from txt/md/pdf/docx."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported text file: {path}")


def parse_video_transcript(path: Path) -> list[dict[str, str]]:
    """Parse a video subtitle/transcript file into structured agenda items.

    Supports .srt/.vtt style blocks. Returns a list of segments with
    {start, end, text} so that chapter-level chunking stays traceable
    back to the video timeline.
    """
    raw = path.read_text(encoding="utf-8")
    raw = raw.replace("WEBVTT", "")
    segments: list[dict[str, str]] = []
    block_re = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(?P<text>.*?)(?=\n\s*\n|\Z)",
        re.DOTALL,
    )
    for m in block_re.finditer(raw):
        text = re.sub(r"<[^>]+>", "", m.group("text")).strip()
        if text:
            segments.append({"start": m.group("start"), "end": m.group("end"), "text": text})
    if not segments:  # plain transcript fallback
        text = raw.strip()
        if text:
            segments.append({"start": "00:00:00,000", "end": "", "text": text})
    return segments


def split_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware sliding-window chunking."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= size:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            while len(para) > size:
                chunks.append(para[:size])
                para = para[size - overlap :]
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def _is_video_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in SUPPORTED_VIDEO_EXT)


async def ingest_file(path: Path, store: MilvusStore, embedder: OllamaEmbedder) -> int:
    """Ingest a single file; returns number of chunks written."""
    doc_id = _doc_id(path)
    store.delete_by_doc(doc_id)  # idempotent dynamic update

    chunks: list[KnowledgeChunk] = []
    if _is_video_file(path):
        segments = parse_video_transcript(path)
        merged = "\n".join(f"[{s['start']} -> {s['end']}] {s['text']}" for s in segments)
        for idx, piece in enumerate(split_chunks(merged)):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=_chunk_id(doc_id, idx),
                    doc_id=doc_id,
                    title=path.stem,
                    content=piece,
                    source=str(path),
                    modality="video_transcript",
                )
            )
    elif path.suffix.lower() in SUPPORTED_TEXT_EXT:
        text = parse_text_file(path)
        for idx, piece in enumerate(split_chunks(text)):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=_chunk_id(doc_id, idx),
                    doc_id=doc_id,
                    title=path.stem,
                    content=piece,
                    source=str(path),
                    modality="text",
                )
            )
    else:
        return 0

    if not chunks:
        return 0
    vectors = await embedder.embed([c.content for c in chunks])
    return store.upsert(chunks, vectors)


async def ingest_directory(dir_path: Path, store: MilvusStore, embedder: OllamaEmbedder) -> dict[str, int]:
    """Ingest every supported file under a directory (recursive)."""
    report: dict[str, int] = {}
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file():
            continue
        n = await ingest_file(path, store, embedder)
        if n:
            report[str(path)] = n
    return report


def collect_corpus(store: MilvusStore) -> Sequence[KnowledgeChunk]:
    """Fetch the full corpus (for BM25 rebuild)."""
    rows = store.client.query(
        collection_name=store.collection_name,
        filter="chunk_id != ''",
        output_fields=["chunk_id", "doc_id", "title", "content", "source", "modality"],
        limit=16384,
    )
    return [
        KnowledgeChunk(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            title=r["title"],
            content=r["content"],
            source=r["source"],
            modality=r.get("modality", "text"),
        )
        for r in rows
    ]
