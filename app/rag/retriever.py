"""Hybrid retrieval pipeline: dense (Milvus) + sparse (BM25) -> RRF -> rerank."""

from __future__ import annotations

import logging
from typing import Sequence

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
        """Rebuild BM25 index from everything stored in Milvus."""
        # Milvus Lite: query all entities (fine for SME-scale corpora).
        results = self.store.client.query(
            collection_name=self.store.collection_name,
            filter="chunk_id != ''",
            output_fields=["chunk_id", "doc_id", "title", "content", "source", "modality"],
            limit=16384,
        )
        self.bm25.rebuild(
            [
                KnowledgeChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    title=r["title"],
                    content=r["content"],
                    source=r["source"],
                    modality=r.get("modality", "text"),
                )
                for r in results
            ]
        )

    async def retrieve(self, query: str, top_k: int | None = None, top_n: int | None = None) -> list[KnowledgeChunk]:
        """Full hybrid pipeline for one query.

        Args:
            query: User query text.
            top_k: Candidates per channel before fusion (default: settings).
            top_n: Final chunks after rerank (default: settings).

        Returns:
            Reranked knowledge chunks.
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

    def format_context(self, chunks: Sequence[KnowledgeChunk]) -> str:
        """Render chunks as grounded context for the LLM prompt."""
        blocks = []
        for i, c in enumerate(chunks, 1):
            blocks.append(f"[资料{i}] 《{c.title}》(来源:{c.source})\n{c.content}")
        return "\n\n".join(blocks)
