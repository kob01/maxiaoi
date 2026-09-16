"""Embedding utilities backed by local Ollama (bge-m3)."""

from typing import Sequence

import httpx

from app.config import get_settings


class OllamaEmbedder:
    """Thin async client for Ollama's /api/embed endpoint.

    bge-m3 produces 1024-dim dense vectors and natively supports
    multilingual + long-context (8k) inputs, which fits enterprise
    Chinese/English mixed documents.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Non-empty list of strings.

        Returns:
            A list of dense vectors, one per input text.
        """
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": list(texts)},
            )
            resp.raise_for_status()
            payload = resp.json()
        return [list(map(float, vec)) for vec in payload["embeddings"]]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return (await self.embed([text]))[0]
