"""Milvus Lite vector store wrapper.

Milvus Lite runs embedded (single file), which keeps local/dev deployment
trivial while the same client code works against Milvus Server in K8s by
simply switching MILVUS_LITE_URI.

Schema carries parent-child chunking fields (parent_id / is_parent /
page_no / section). Retrieval always filters `is_parent == 0` so only child
chunks compete; parents are fetched by id afterwards for context assembly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from pymilvus import DataType, MilvusClient

from app.config import get_settings
from app.schemas import KnowledgeChunk

logger = logging.getLogger(__name__)

DENSE_DIM = 1024  # bge-m3 dense dimension

OUTPUT_FIELDS = [
    "chunk_id", "doc_id", "title", "content", "source", "modality",
    "parent_id", "is_parent", "page_no", "section",
]


class MilvusStore:
    """Vector persistence + ANN search over enterprise knowledge chunks."""

    def __init__(self, uri: str | None = None, collection: str | None = None) -> None:
        settings = get_settings()
        self.uri = self._resolve_uri(uri or settings.milvus_lite_uri)
        self.collection_name = collection or settings.milvus_collection
        self.client = MilvusClient(uri=self.uri)
        self._ensure_collection()

    @staticmethod
    def _resolve_uri(uri: str) -> str:
        """Anchor relative file URIs to the project root and mkdir parents.

        Milvus Lite auto-creates the db file on first use, but the parent
        directory must exist, and a relative path would otherwise resolve
        against the process CWD instead of the project root.
        """
        if uri.startswith(("http://", "https://")):
            return uri
        path = Path(uri)
        if not path.is_absolute():
            path = get_settings().base_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _ensure_collection(self) -> None:
        """Create collection + index on first use; always load it into memory.

        Milvus Lite collections stay 'released' across processes, so every
        process opening the db file must explicitly load() before search.
        A legacy collection without the parent-child fields is dropped and
        recreated (documents must be re-ingested).
        """
        if self.client.has_collection(self.collection_name):
            fields = {f["name"] for f in self.client.describe_collection(self.collection_name)["fields"]}
            if "is_parent" not in fields:
                logger.warning(
                    "collection %s uses the legacy schema, dropping for parent-child schema",
                    self.collection_name,
                )
                self.client.drop_collection(self.collection_name)
            else:
                self.client.load_collection(self.collection_name)
                return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=80)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("modality", DataType.VARCHAR, max_length=32)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=80)
        schema.add_field("is_parent", DataType.INT64)  # 1 = parent block, 0 = child chunk
        schema.add_field("page_no", DataType.INT64)    # -1 = unknown
        schema.add_field("section", DataType.VARCHAR, max_length=256)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=self.collection_name, schema=schema, index_params=index_params
        )
        self.client.load_collection(self.collection_name)

    @staticmethod
    def _row(c: KnowledgeChunk, v: Sequence[float]) -> dict[str, Any]:
        return {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title[:512],
            "content": c.content[:8192],
            "source": c.source[:512],
            "modality": c.modality,
            "parent_id": c.parent_id,
            "is_parent": 1 if c.is_parent else 0,
            "page_no": c.page_no,
            "section": c.section[:256],
            "embedding": list(v),
        }

    @staticmethod
    def _to_chunk(entity: dict[str, Any], score: float = 0.0) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=entity["chunk_id"],
            doc_id=entity["doc_id"],
            title=entity["title"],
            content=entity["content"],
            source=entity["source"],
            modality=entity.get("modality", "text"),
            parent_id=entity.get("parent_id", ""),
            is_parent=bool(entity.get("is_parent", 0)),
            page_no=int(entity.get("page_no", -1)),
            section=entity.get("section", ""),
            score=score,
        )

    def upsert(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> int:
        """Insert or update chunks with their dense vectors."""
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"
        rows = [self._row(c, v) for c, v in zip(chunks, vectors)]
        self.client.upsert(collection_name=self.collection_name, data=rows)
        return len(rows)

    def search(self, query_vector: Sequence[float], top_k: int) -> list[KnowledgeChunk]:
        """ANN cosine search over child chunks; returns chunks with score."""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[list(query_vector)],
            limit=top_k,
            filter="is_parent == 0",
            output_fields=OUTPUT_FIELDS,
        )
        return [self._to_chunk(hit["entity"], float(hit["distance"])) for hit in results[0]]

    def query_parents(self, parent_ids: Sequence[str]) -> dict[str, KnowledgeChunk]:
        """Fetch parent blocks by chunk_id (for context assembly)."""
        if not parent_ids:
            return {}
        quoted = ", ".join(f'"{pid}"' for pid in parent_ids)
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f"chunk_id in [{quoted}]",
            output_fields=OUTPUT_FIELDS,
            limit=len(parent_ids),
        )
        return {r["chunk_id"]: self._to_chunk(r) for r in rows}

    def iter_child_chunks(self, limit: int = 16384) -> list[KnowledgeChunk]:
        """Return all child chunks (BM25 corpus rebuild)."""
        rows = self.client.query(
            collection_name=self.collection_name,
            filter="is_parent == 0",
            output_fields=OUTPUT_FIELDS,
            limit=limit,
        )
        return [self._to_chunk(r) for r in rows]

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove all chunks of a document (used for overwrite re-ingest)."""
        self.client.delete(collection_name=self.collection_name, filter=f'doc_id == "{doc_id}"')

    def count(self) -> int:
        """Return number of stored chunks (parents + children)."""
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))
