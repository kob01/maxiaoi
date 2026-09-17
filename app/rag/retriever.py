"""Hybrid retrieval pipeline: dense (Milvus) + sparse (BM25) -> RRF -> rerank.

Retrieval operates on child chunks only (`is_parent == 0` filter); hit
children are then assembled back into their parent section blocks so the LLM
receives complete section context instead of truncated fragments.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from app.config import get_settings
from app.rag.bm25 import BM25Retriever
from app.rag.embeddings import OllamaEmbedder
from app.rag.reranker import OllamaReranker
from app.rag.vectorstore import MilvusStore
from app.schemas import KnowledgeChunk

logger = logging.getLogger(__name__)

RRF_K = 60  # Reciprocal Rank Fusion constant


def _rrf_fuse(
    dense: Sequence[KnowledgeChunk], sparse: Sequence[KnowledgeChunk], top_k: int
) -> list[KnowledgeChunk]:
    """Merge two ranked lists with Reciprocal Rank Fusion."""
    fused: dict[str, tuple[KnowledgeChunk, float]] = {}
    for rank, chunk in enumerate(dense):
        score = 1.0 / (RRF_K + rank + 1)
        if chunk.chunk_id in fused:
            c, s = fused[chunk.chunk_id]
            fused[chunk.chunk_id] = (c, s + score)
        else:
            fused[chunk.chunk_id] = (chunk, score)
    for rank, chunk in enumerate(sparse):
        score = 1.0 / (RRF_K + rank + 1)
        if chunk.chunk_id in fused:
            c, s = fused[chunk.chunk_id]
            fused[chunk.chunk_id] = (c, s + score)
        else:
            fused[chunk.chunk_id] = (chunk, score)
    ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)[:top_k]
    return [chunk.model_copy(update={"score": score}) for chunk, score in ranked]


class HybridRetriever:
    """Enterprise knowledge retriever used by the Assistant's KB path."""

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self.embedder = OllamaEmbedder()
        self.store = MilvusStore()
        self.bm25 = BM25Retriever()
        self.reranker = OllamaReranker()

    def rebuild_bm25(self) -> None:
        """Rebuild BM25 index from child chunks stored in Milvus."""
        self.bm25.rebuild(self.store.iter_child_chunks())

    async def retrieve(self, query: str, top_k: int | None = None, top_n: int | None = None) -> list[KnowledgeChunk]:
        """Full hybrid pipeline for one query (child-chunk granularity).

        Args:
            query: User query text.
            top_k: Candidates per channel before fusion (default: settings).
            top_n: Final chunks after rerank (default: settings).

        Returns:
            Reranked child chunks.
        """
        top_k = top_k or self._settings.rag_top_k
        top_n = top_n or self._settings.rerank_top_n

        query_vec = await self.embedder.embed_query(query)
        dense_hits = self.store.search(query_vec, top_k)
        sparse_hits = self.bm25.search(query, top_k)
        fused = _rrf_fuse(dense_hits, sparse_hits, top_k)
        if not self._settings.rerank_enabled:
            return fused[:top_n]
        try:
            return await self.reranker.rerank(query, fused, top_n)
        except Exception as exc:  # graceful degradation to RRF order
            logger.warning("rerank failed, fallback to RRF fusion order: %s", exc)
            return fused[:top_n]

    def assemble_parents(self, chunks: Sequence[KnowledgeChunk]) -> list[KnowledgeChunk]:
        """Assemble hit child chunks into complete parent section blocks.

        Keeps the reranked order: each distinct parent appears once, at the
        position of its best-scoring child, with the parent's full text.
        Falls back to the child itself when the parent row is missing.
        """
        best_child: dict[str, KnowledgeChunk] = {}
        order: list[str] = []
        for c in chunks:
            pid = c.parent_id or c.chunk_id
            if pid not in best_child:
                best_child[pid] = c
                order.append(pid)
        parents = self.store.query_parents(order)
        assembled: list[KnowledgeChunk] = []
        for pid in order:
            parent = parents.get(pid)
            child = best_child[pid]
            assembled.append(parent.model_copy(update={"score": child.score}) if parent else child)
        return assembled

    def format_context(
        self,
        chunks: Sequence[KnowledgeChunk],
        meta_map: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Render chunks as grounded context for the LLM prompt.

        Each block cites title / section / page and (when MySQL metadata is
        available) the document tags.
        """
        blocks = []
        for i, c in enumerate(chunks, 1):
            cite = f"《{c.title}》"
            if c.section:
                cite += f" 章节:{c.section}"
            if c.page_no > 0:
                cite += f" 第{c.page_no}页"
            meta = (meta_map or {}).get(c.doc_id) or {}
            if meta.get("tags"):
                cite += f" (标签: {', '.join(meta['tags'])})"
            blocks.append(f"[资料{i}] {cite}\n{c.content}")
        return "\n\n".join(blocks)
