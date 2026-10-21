"""In-process hybrid index (BM25 + dense + RRF). Same interface as PgVectorStore.

Tenant filtering happens BEFORE scoring: a tenant's query never scores another tenant's chunks,
so even IDF statistics are per-tenant (no cross-tenant information leaks through rankings).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from capstone.rag.base import Chunk, Hit, SearchMode
from capstone.rag.bm25 import BM25
from capstone.rag.embeddings import Embedder, HashingEmbedder
from capstone.rag.fusion import rrf


class _TenantIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.bm25 = BM25([c.content for c in chunks])
        self.vecs = embedder.embed([c.content for c in chunks]) if chunks else np.zeros((0, 1))


class InMemoryHybridIndex:
    def __init__(self, embedder: Embedder | None = None, candidate_k: int = 30) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.candidate_k = candidate_k
        self._pending: dict[str, list[Chunk]] = defaultdict(list)
        self._index: dict[str, _TenantIndex] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._pending[c.tenant_id].append(c)
            self._index.pop(c.tenant_id, None)

    def _tenant(self, tenant_id: str) -> _TenantIndex | None:
        if tenant_id not in self._index and self._pending.get(tenant_id):
            self._index[tenant_id] = _TenantIndex(self._pending[tenant_id], self.embedder)
        return self._index.get(tenant_id)

    def search(self, tenant_id: str, query: str, k: int = 5, *, mode: SearchMode = "hybrid",
               filters: dict[str, Any] | None = None) -> list[Hit]:
        idx = self._tenant(tenant_id)
        if idx is None:
            return []
        allowed = [i for i, c in enumerate(idx.chunks)
                   if all(c.metadata.get(fk) == fv or getattr(c, fk, None) == fv
                          for fk, fv in (filters or {}).items())]
        if not allowed:
            return []
        rankings: dict[str, list[int]] = {}
        scores: dict[str, dict[int, float]] = {}
        if mode in {"bm25", "hybrid"}:
            bm = idx.bm25.scores(query)
            order = sorted((i for i in allowed if bm[i] > 0), key=lambda i: -bm[i])
            rankings["bm25"] = order[: self.candidate_k]
            scores["bm25"] = {i: bm[i] for i in order}
        if mode in {"vector", "hybrid"}:
            qv = self.embedder.embed([query])[0]
            sims = idx.vecs @ qv
            order = sorted(allowed, key=lambda i: -sims[i])
            rankings["vector"] = order[: self.candidate_k]
            scores["vector"] = {i: float(sims[i]) for i in order}
        if mode == "hybrid":
            fused = rrf(list(rankings.values()))
        else:
            name = next(iter(rankings))
            fused = [(i, scores[name][i]) for i in rankings[name]]
        hits = []
        for i, score in fused[:k]:
            assert isinstance(i, int)
            ranks = {n: r.index(i) + 1 for n, r in rankings.items() if i in r}
            hits.append(Hit(chunk=idx.chunks[i], score=float(score), ranks=ranks))
        return hits
