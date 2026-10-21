"""pgvector + Postgres full-text hybrid search, fused with RRF inside one SQL statement.

Tenant isolation is enforced twice: an explicit WHERE tenant_id = %s, and row-level security
keyed on `app.tenant_id` (see infra/postgres/init.sql). Set the GUC per transaction with
SET LOCAL so a pooled connection can never carry a stale tenant.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from capstone.rag.base import Chunk, Hit, SearchMode
from capstone.rag.embeddings import Embedder

HYBRID_SQL = """
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(qvec)s::vector) AS rnk
    FROM kb_chunks
    WHERE tenant_id = %(tenant)s {filter_sql}
    ORDER BY embedding <=> %(qvec)s::vector
    LIMIT %(cand)s
),
fts AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rnk
    FROM kb_chunks, websearch_to_tsquery('english', %(qtext)s) q
    WHERE tenant_id = %(tenant)s AND tsv @@ q {filter_sql}
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT %(cand)s
),
fused AS (
    SELECT id,
           COALESCE(%(wv)s / (%(k)s + vec.rnk), 0)
             + COALESCE(%(wf)s / (%(k)s + fts.rnk), 0) AS score,
           vec.rnk AS vec_rank, fts.rnk AS fts_rank
    FROM vec FULL OUTER JOIN fts USING (id)
)
SELECT c.id, c.tenant_id, c.kb_id, c.doc_id, c.parent_id, c.section, c.content, c.metadata,
       f.score, f.vec_rank, f.fts_rank
FROM fused f JOIN kb_chunks c USING (id)
ORDER BY f.score DESC
LIMIT %(k_out)s
"""


class PgVectorStore:
    def __init__(self, dsn: str, embedder: Embedder, candidate_k: int = 40) -> None:
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(dsn, min_size=1, max_size=8, open=True)
        self.embedder = embedder
        self.candidate_k = candidate_k

    @contextmanager
    def _tx(self, tenant_id: str) -> Iterator[Any]:
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
            yield cur

    def upsert(self, chunks: list[Chunk], batch: int = 128) -> int:
        from pgvector.psycopg import register_vector

        n = 0
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            vecs = self.embedder.embed([c.content for c in part])
            by_tenant: dict[str, list[tuple[Chunk, np.ndarray]]] = {}
            for c, v in zip(part, vecs, strict=True):
                by_tenant.setdefault(c.tenant_id, []).append((c, v))
            for tenant_id, rows in by_tenant.items():
                with self._tx(tenant_id) as cur:
                    register_vector(cur.connection)
                    cur.executemany(
                        """INSERT INTO kb_chunks (id, tenant_id, kb_id, doc_id, parent_id, section,
                                                  content, metadata, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,
                               metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                        [(c.chunk_id, c.tenant_id, c.kb_id, c.doc_id, c.parent_id, c.section,
                          c.content, json.dumps(c.metadata), v) for c, v in rows])
                    n += len(rows)
        return n

    def search(self, tenant_id: str, query: str, k: int = 5, *, mode: SearchMode = "hybrid",
               filters: dict[str, Any] | None = None) -> list[Hit]:
        filter_sql, params = "", {}
        for i, (key, val) in enumerate((filters or {}).items()):
            if key not in {"kb_id", "doc_id", "section"}:
                raise ValueError(f"unsupported filter {key}")
            filter_sql += f" AND {key} = %(f{i})s"
            params[f"f{i}"] = val
        wv, wf = {"hybrid": (1.0, 1.0), "vector": (1.0, 0.0), "bm25": (0.0, 1.0)}[mode]
        qvec = self.embedder.embed([query])[0].tolist()
        with self._tx(tenant_id) as cur:
            cur.execute(HYBRID_SQL.format(filter_sql=filter_sql), {
                "qvec": str(qvec), "qtext": query, "tenant": tenant_id, "cand": self.candidate_k,
                "k": 60.0, "wv": wv, "wf": wf, "k_out": k, **params})
            rows = cur.fetchall()
        return [Hit(chunk=Chunk(chunk_id=r[0], tenant_id=r[1], kb_id=r[2], doc_id=r[3],
                                parent_id=r[4], section=r[5], content=r[6], metadata=r[7] or {}),
                    score=float(r[8]),
                    ranks={k: v for k, v in (("vector", r[9]), ("bm25", r[10])) if v})
                for r in rows]
