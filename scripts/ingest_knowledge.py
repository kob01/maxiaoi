"""RAG knowledge-base build script.

Parses documents under KNOWLEDGE_DIR (txt/md/pdf/docx + video transcripts),
chunks, embeds with bge-m3 and upserts into Milvus Lite, then rebuilds the
BM25 channel.

Usage:
    python -m scripts.ingest_knowledge [--dir ./data/knowledge]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.rag.bm25 import BM25Retriever
from app.rag.embeddings import OllamaEmbedder
from app.rag.ingest import collect_corpus, ingest_directory
from app.rag.vectorstore import MilvusStore


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest enterprise knowledge into Milvus Lite")
    parser.add_argument("--dir", default=get_settings().knowledge_dir, help="knowledge directory")
    args = parser.parse_args()

    knowledge_dir = Path(args.dir)
    if not knowledge_dir.exists():
        raise SystemExit(f"knowledge dir not found: {knowledge_dir}")

    store = MilvusStore()
    embedder = OllamaEmbedder()

    print(f"[ingest] scanning {knowledge_dir} ...")
    report = await ingest_directory(knowledge_dir, store, embedder)
    for path, n in report.items():
        print(f"[ingest] {path}: {n} chunks")
    print(f"[ingest] total chunks in store: {store.count()}")

    bm25 = BM25Retriever()
    bm25.rebuild(collect_corpus(store))
    print("[ingest] BM25 channel rebuilt (in-memory; assistant rebuilds on startup)")


if __name__ == "__main__":
    asyncio.run(main())
