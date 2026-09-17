"""Knowledge ingestion pipeline.

Stages: parse (txt/md/pdf/docx/pptx/xlsx + video transcripts + images) into
section-level blocks -> parent-child chunking -> embed -> upsert into Milvus,
then rebuild the BM25 channel.

Document identity: ``doc_id = sha1(file_name + ext)`` — path-independent, so
re-uploading the same name+type overwrites the previous copy (single-version
semantics: ``delete_by_doc`` + upsert).

Parent-child chunking: each parsed block becomes a parent chunk; blocks
larger than ``parent_chunk_max`` are window-split into child chunks.
Retrieval hits children only; parents carry the complete section text for
context assembly.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

from app.config import get_settings
from app.docs.parsers import ParsedBlock, parse_blocks
from app.rag.embeddings import OllamaEmbedder
from app.rag.vectorstore import MilvusStore
from app.schemas import KnowledgeChunk

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def compute_doc_id(name: str, ext: str) -> str:
    """Stable document identity from file name + extension (path-independent)."""
    return hashlib.sha1(f"{name}|{ext.lower()}".encode()).hexdigest()[:16]


def split_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware sliding-window chunking (fallback splitter)."""
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


def build_parent_child_chunks(
    doc_id: str,
    title: str,
    source: str,
    modality: str,
    blocks: Sequence[ParsedBlock],
    parent_max: int | None = None,
) -> list[KnowledgeChunk]:
    """Turn parsed section blocks into parent + child KnowledgeChunks.

    Every block becomes one parent chunk (complete section text). Oversized
    blocks are additionally window-split into child chunks that carry the
    retrieval payload; small blocks get exactly one child identical to the
    parent so retrieval and assembly stay uniform.
    """
    parent_max = parent_max or get_settings().parent_chunk_max
    chunks: list[KnowledgeChunk] = []
    for seq, block in enumerate(blocks):
        text = block.text.strip()
        if not text:
            continue
        parent_id = f"{doc_id}-p{seq:04d}"
        parent = KnowledgeChunk(
            chunk_id=parent_id,
            doc_id=doc_id,
            title=title,
            content=text[:8192],
            source=source,
            modality=modality,  # type: ignore[arg-type]
            is_parent=True,
            page_no=block.page_no,
            section=block.section,
        )
        chunks.append(parent)
        pieces = [text] if len(text) <= parent_max else split_chunks(text)
        for idx, piece in enumerate(pieces):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{parent_id}-c{idx:02d}",
                    doc_id=doc_id,
                    title=title,
                    content=piece,
                    source=source,
                    modality=modality,  # type: ignore[arg-type]
                    parent_id=parent_id,
                    page_no=block.page_no,
                    section=block.section,
                )
            )
    return chunks


async def ingest_file(path: Path, store: MilvusStore, embedder: OllamaEmbedder) -> int:
    """Ingest a single file; returns number of chunks written (parents+children)."""
    doc_id = compute_doc_id(path.stem, path.suffix)
    modality, blocks = await parse_blocks(path)
    return await ingest_blocks(doc_id, path.name, path.stem, str(path), modality, blocks, store, embedder)


async def ingest_blocks(
    doc_id: str,
    filename: str,
    title: str,
    source: str,
    modality: str,
    blocks: Sequence[ParsedBlock],
    store: MilvusStore,
    embedder: OllamaEmbedder,
) -> int:
    """Overwrite-ingest pre-parsed blocks of one document into Milvus."""
    chunks = build_parent_child_chunks(doc_id, title, source, modality, blocks)
    if not chunks:
        return 0
    store.delete_by_doc(doc_id)  # single-version overwrite semantics
    vectors = await embedder.embed([c.content for c in chunks])
    return store.upsert(chunks, vectors)


async def ingest_directory(dir_path: Path, store: MilvusStore, embedder: OllamaEmbedder) -> dict[str, int]:
    """Ingest every supported file under a directory (recursive)."""
    report: dict[str, int] = {}
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file():
            continue
        try:
            n = await ingest_file(path, store, embedder)
        except ValueError:
            continue  # unsupported extension
        if n:
            report[str(path)] = n
    return report


def collect_corpus(store: MilvusStore) -> Sequence[KnowledgeChunk]:
    """Fetch the child-chunk corpus (for BM25 rebuild)."""
    return store.iter_child_chunks()
