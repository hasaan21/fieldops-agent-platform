"""Retrieval ablation: chunking strategy x search mode x reranker on the rag_qa gold set.

    uv run python -m capstone.rag.ablation                       # offline hashing embedder
    uv run python -m capstone.rag.ablation --embedder openai     # real embeddings
    uv run python -m capstone.rag.ablation --reranker cross-encoder

Writes artifacts/rag/ablation.md and .json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

from capstone.data.loader import load_table
from capstone.rag.base import SearchMode
from capstone.rag.chunking import Strategy
from capstone.rag.embeddings import Embedder, HashingEmbedder
from capstone.rag.ingest import build_chunks
from capstone.rag.memory_store import InMemoryHybridIndex
from capstone.rag.metrics import is_relevant, summarize
from capstone.rag.rerank import NoopReranker, Reranker

STRATEGIES: list[Strategy] = ["fixed", "recursive", "markdown", "parent"]
MODES: list[SearchMode] = ["bm25", "vector", "hybrid"]


def evaluate(index: InMemoryHybridIndex, qa: list[dict[str, Any]], mode: SearchMode,
             reranker: Reranker, k: int = 5, candidates: int = 20) -> dict[str, float]:
    per_query = []
    t0 = time.perf_counter()
    for q in qa:
        hits = index.search(q["tenant_id"], q["question"], k=candidates, mode=mode)
        hits = reranker.rerank(q["question"], hits, k)
        # dedupe parent chunks so small-to-big isn't credited twice for one section
        seen, rels = set(), []
        for h in hits:
            key = h.chunk.parent_id or h.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            rels.append(is_relevant(h.chunk, q))
        per_query.append(rels)
    out = summarize(per_query, k)
    out["ms_per_query"] = (time.perf_counter() - t0) * 1000 / max(1, len(qa))
    return out


def run(embedder: Embedder, reranker: Reranker, strategies: list[Strategy] | None = None,
        modes: list[SearchMode] | None = None) -> list[dict[str, Any]]:
    qa = load_table("rag_qa")
    rows = []
    for strategy in strategies or STRATEGIES:
        chunks = build_chunks(strategy, embedder=embedder)
        index = InMemoryHybridIndex(embedder)
        index.add(chunks)
        for mode in modes or MODES:
            m = evaluate(index, qa, mode, reranker)
            rows.append({"strategy": strategy, "mode": mode, "reranker": reranker.name,
                         "embedder": embedder.name, "n_chunks": len(chunks), **m})
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    head = "| chunking | search | reranker | chunks | recall@5 | MRR | nDCG@5 | ms/query |"
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["mrr"]):
        lines.append(f"| {r['strategy']} | {r['mode']} | {r['reranker']} | {r['n_chunks']} | "
                     f"{r['recall@5']:.3f} | {r['mrr']:.3f} | {r['ndcg@5']:.3f} | "
                     f"{r['ms_per_query']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["hashing", "openai", "bge"], default="hashing")
    ap.add_argument("--reranker", choices=["none", "cross-encoder", "llm"], default="none")
    args = ap.parse_args()
    embedder: Embedder
    if args.embedder == "openai":
        from capstone.rag.embeddings import OpenAIEmbedder

        embedder = OpenAIEmbedder()
    elif args.embedder == "bge":
        from capstone.rag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    else:
        embedder = HashingEmbedder()
    reranker: Reranker = NoopReranker()
    if args.reranker == "cross-encoder":
        from capstone.rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()
    elif args.reranker == "llm":
        from capstone.core.llm import get_chat_model
        from capstone.rag.rerank import LLMReranker

        reranker = LLMReranker(get_chat_model("agent"))
    rows = run(embedder, reranker)
    out = Path("artifacts/rag")
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows)
    (out / f"ablation-{embedder.name}-{reranker.name}.md".replace("/", "_")).write_text(md)
    (out / "ablation.json").write_text(json.dumps(cast(Any, rows), indent=2))
    print(md)


if __name__ == "__main__":
    main()
