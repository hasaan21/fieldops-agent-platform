CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Knowledge-base chunks. Every query MUST filter on tenant_id; row-level security enforces it
-- even if application code forgets.
CREATE TABLE IF NOT EXISTS kb_chunks (
    id            TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    kb_id         TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    parent_id     TEXT,
    section       TEXT,
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding     vector(1536),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kb_chunks_tenant_idx ON kb_chunks (tenant_id, kb_id);
CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
    ON kb_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

ALTER TABLE kb_chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS kb_tenant_isolation ON kb_chunks;
CREATE POLICY kb_tenant_isolation ON kb_chunks
    USING (tenant_id = current_setting('app.tenant_id', true));

-- Human-in-the-loop review queue (mirrors LangGraph interrupts for a reviewer UI).
CREATE TABLE IF NOT EXISTS review_items (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    thread_id     TEXT NOT NULL,
    agent         TEXT NOT NULL,
    payload       JSONB NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    decided_by    TEXT,
    decision      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS review_items_pending_idx ON review_items (tenant_id, status);

-- Per-run cost/latency ledger (feeds budgets and dashboards).
CREATE TABLE IF NOT EXISTS run_ledger (
    run_id        TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    agent         TEXT NOT NULL,
    channel       TEXT,
    prompt_ref    TEXT,
    model         TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    outcome       TEXT,
    budget_violations TEXT[] NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS run_ledger_tenant_time ON run_ledger (tenant_id, created_at DESC);
