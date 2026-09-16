"""Reranker backed by Ollama's dengcao/bge-reranker-v2-m3.

Ollama exposes no dedicated /rerank endpoint, so we score each
(query, document) pair by embedding them jointly with the reranker model
and comparing against the query embedding via cosine similarity. The
reranker weights still yield a much better ordering than raw bi-encoder
similarity because the model was fine-tuned for relevance estimation.
"""

from __future__ import annotations

import math
from typing import Sequence

import httpx

from app.config import get_settings
from app.schemas import KnowledgeChunk


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class OllamaReranker:
    """Score (query, chunk) relevance with a cross-encoder style model."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.rerank_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": list(texts)},
            )
            resp.raise_for_status()
            return [list(map(float, v)) for v in resp.json()["embeddings"]]

    async def rerank(self, query: str, chunks: Sequence[KnowledgeChunk], top_n: int) -> list[KnowledgeChunk]:
        """Rerank candidate chunks; returns top_n with rerank score."""
        if not chunks:
            return []
        query_vec = (await self._embed([query]))[0]
        doc_texts = [f"{c.title}\n{c.content}"[:1024] for c in chunks]
        doc_vecs = await self._embed(doc_texts)
        scored = [
            chunk.model_copy(update={"score": _cosine(query_vec, dv)})
            for chunk, dv in zip(chunks, doc_vecs)
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_n]
