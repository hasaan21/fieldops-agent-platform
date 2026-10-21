"""Reciprocal Rank Fusion (Cormack et al., 2009).

RRF(d) = sum over rankers r of  w_r / (k + rank_r(d)),  with k = 60 by convention.

It ignores raw scores (BM25 and cosine are on incompatible scales) and only uses ranks, which
is why it is the default hybrid fusion in Azure AI Search, Elasticsearch and others.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60,
        weights: Sequence[float] | None = None) -> list[tuple[Hashable, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[Hashable, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
