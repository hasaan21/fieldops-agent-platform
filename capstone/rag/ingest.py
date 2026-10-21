"""Ingest knowledge-base documents into an index.

    uv run python -m capstone.rag.ingest --target memory          # smoke test
    uv run python -m capstone.rag.ingest --target pg --strategy parent
"""

from __future__ import annotations

import argparse
from functools import lru_cache

from capstone.data.loader import load_table
from capstone.rag.base import Chunk
from capstone.rag.chunking import Strategy, chunk_document
from capstone.rag.embeddings import Embedder, HashingEmbedder
from capstone.rag.memory_store import InMemoryHybridIndex


def build_chunks(strategy: Strategy = "markdown", size: int = 400,
                 embedder: Embedder | None = None) -> list[Chunk]:
    embed = embedder.embed if embedder else None
    out: list[Chunk] = []
    for doc in load_table("kb_documents"):
        out += chunk_document(doc, strategy, size=size, embed=embed)
    return out


@lru_cache(maxsize=8)
def synthetic_index(strategy: Strategy = "markdown") -> InMemoryHybridIndex:
    embedder = HashingEmbedder()
    idx = InMemoryHybridIndex(embedder)
    idx.add(build_chunks(strategy, embedder=embedder))
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["memory", "pg"], default="memory")
    ap.add_argument("--strategy", default="markdown")
    args = ap.parse_args()
    from capstone.rag.embeddings import default_embedder

    embedder = default_embedder()
    chunks = build_chunks(args.strategy, embedder=embedder)
    if args.target == "pg":
        from capstone.core.config import get_settings
        from capstone.rag.pg_store import PgVectorStore

        n = PgVectorStore(get_settings().database_url, embedder).upsert(chunks)
        print(f"upserted {n} chunks into pgvector ({embedder.name})")
    else:
        idx = InMemoryHybridIndex(embedder)
        idx.add(chunks)
        hits = idx.search("t1", "How much is your call-out fee?", k=3)
        for h in hits:
            print(f"{h.score:.4f} {h.ranks} {h.chunk.chunk_id}")


if __name__ == "__main__":
    main()
