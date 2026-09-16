"""BM25 sparse retriever (lexical channel of the hybrid search).

Built in-process with rank_bm25 + jieba for Chinese tokenization. The index
is rebuilt from Milvus content on startup / after ingestion; for larger
deployments this can be swapped for Elasticsearch without changing the
retriever interface.
"""

from __future__ import annotations

import re
from typing import Sequence

import jieba
from rank_bm25 import BM25Okapi

from app.schemas import KnowledgeChunk


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25."""
    text = text.lower()
    tokens = jieba.lcut(text)
    return [t for t in tokens if re.search(r"[a-z0-9一-鿿]", t)]


class BM25Retriever:
    """In-memory BM25 index over knowledge chunks."""

    def __init__(self) -> None:
        self._chunks: list[KnowledgeChunk] = []
        self._index: BM25Okapi | None = None

    def rebuild(self, chunks: Sequence[KnowledgeChunk]) -> None:
        """(Re)build the index from a chunk corpus."""
        self._chunks = list(chunks)
        corpus = [tokenize(f"{c.title} {c.content}") for c in self._chunks]
        self._index = BM25Okapi(corpus) if corpus else None

    def add(self, chunks: Sequence[KnowledgeChunk]) -> None:
        """Append chunks and rebuild (simple dynamic-update strategy)."""
        self.rebuild([*self._chunks, *chunks])

    def search(self, query: str, top_k: int) -> list[KnowledgeChunk]:
        """Return top-k chunks ranked by BM25 score."""
        if not self._index or not self._chunks:
            return []
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            chunk.model_copy(update={"score": float(score)}) for chunk, score in ranked if score > 0
        ]
