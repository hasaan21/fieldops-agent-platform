"""Retrieval metrics over binary relevance.

recall@k: fraction of queries with at least one relevant chunk in the top k (with one gold
          passage per query this is "hit rate").
MRR:      mean of 1/rank of the first relevant chunk.
nDCG@k:   DCG / ideal DCG with gains 1/log2(rank+1); rewards relevant results ranked higher.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from capstone.rag.base import Chunk


def is_relevant(chunk: Chunk, gold: dict[str, Any]) -> bool:
    if chunk.doc_id != gold["doc_id"]:
        return False
    text = (chunk.metadata.get("parent_content") or chunk.content).lower()
    return chunk.section == gold["section"] or str(gold["answer"]).lower() in text


def recall_at_k(rels: Sequence[bool], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def reciprocal_rank(rels: Sequence[bool]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: Sequence[bool], k: int, n_relevant: int = 1) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1) if r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, max(1, n_relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def summarize(per_query: list[list[bool]], k: int = 5) -> dict[str, float]:
    n = len(per_query) or 1
    return {
        f"recall@{k}": sum(recall_at_k(r, k) for r in per_query) / n,
        "mrr": sum(reciprocal_rank(r) for r in per_query) / n,
        f"ndcg@{k}": sum(ndcg_at_k(r, k) for r in per_query) / n,
    }
