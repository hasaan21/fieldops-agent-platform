# Week 3: RAG and knowledge bases

**Time:** about 25 hours.
**Outcome:** Engineer Copilot, a hybrid-retrieval, citation-checked, self-verifying RAG agent, plus an ablation table that justifies each design choice with numbers.

## Why it matters

Most useful agents answer from a knowledge base, and most bad answers trace back to retrieval: chunking, embeddings and hybrid/vector search decide whether answers stay grounded and faithful. In a multi-tenant platform the knowledge base is per tenant, so the hardest constraints are isolation and *disambiguation*: many tenants, many near-identical manuals.

## Concepts

1. **The RAG pipeline** is ingest, chunk, embed, index, retrieve, rerank, assemble context, generate, verify. Measure retrieval and generation *separately*. If retrieval recall is 60%, no prompt will fix the answers.
2. **Chunking** (`capstone/rag/chunking.py`):
   - fixed / recursive: structure-blind; good baselines.
   - markdown/section: respects document structure; the section path becomes metadata and a header prefix ("contextual chunk headers").
   - parent (small-to-big): match on small chunks, give the LLM the whole parent section.
   - semantic: break where adjacent-sentence similarity drops.
   - Reading: [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), [Pinecone chunking strategies](https://www.pinecone.io/learn/chunking-strategies/).
3. **Embeddings**: dense models (OpenAI `text-embedding-3-*`, BGE, E5) versus lexical. Know the trade-offs: dimension, cost, multilingual support, domain fit. Check [MTEB](https://huggingface.co/spaces/mteb/leaderboard), but always test on *your* data.
4. **Lexical search, BM25** (`bm25.py`): wins on exact identifiers like error codes ("F28"), model numbers and postcodes. Dense search wins on paraphrase ("won't light" vs "ignition failure").
5. **Hybrid + RRF** (`fusion.py`, `pg_store.py`): run both and fuse by rank. One SQL statement in pgvector + `tsvector`; native in Azure AI Search (`search` + `vectorQueries`, with optional semantic ranker).
6. **Reranking** (`rerank.py`): a cross-encoder scores (query, passage) jointly. It's far more accurate than bi-encoder similarity but O(n) model calls, so rerank only the top 20-50.
7. **Vector index internals**: HNSW (graph; `m`, `ef_construction`, `ef_search`) versus IVFFlat (clusters; `lists`, `probes`); recall vs latency vs memory; filtered-search pitfalls (post-filtering can return fewer than k). [pgvector README](https://github.com/pgvector/pgvector).
8. **Multi-tenancy**: filter *before* scoring, use row-level security (`infra/postgres/init.sql`), and choose a partition strategy: a shared index with a filter, an index per tenant, or a hybrid (big tenants get their own index).
9. **Generation metrics** (`faithfulness.py`): faithfulness (claims supported by context), citation validity, answer relevance, context precision/recall. [RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/).
10. **Self-verification** (`engineer_copilot/graph.py`): check faithfulness, retry once with a stricter prompt, otherwise abstain. Abstaining beats confident nonsense.

## Baseline ablation (offline hashing embedder, no reranker, 111 gold questions)

Produced by `uv run python -m capstone.rag.ablation`:

| chunking | search | recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| markdown | bm25 | 1.000 | 0.869 | 0.904 |
| markdown | hybrid | 1.000 | 0.863 | 0.899 |
| markdown | vector | 0.982 | 0.812 | 0.855 |
| fixed | bm25 | 0.883 | 0.685 | 0.854 |
| recursive | hybrid | 0.658 | 0.348 | 0.424 |

What this shows, and what you should be able to explain:
- **Structure-aware chunking dominates** (MRR 0.87 vs 0.35-0.69). Recursive splitting cuts "### Code F22" headings away from their meaning lines.
- **MRR stalls around 0.87 even with perfect recall.** Several boiler models share fault codes (F22 exists in three manuals), so the right section is often ranked 2nd or 3rd. That's a *disambiguation* problem, and better embeddings won't fix it. Metadata (model to document) filtering and query understanding will. The Copilot's `MODEL_RE` boost is a first step; your lab makes it proper.
- The hashing "embedder" is lexical, so vector search ≈ BM25 here. Re-run with real embeddings and expect vector and hybrid to gain on paraphrased questions (step 3 below).

## Lab

1. Read `chunking.py`, `bm25.py`, `fusion.py` and `memory_store.py`, then reproduce the table above.
2. Start Postgres, then ingest into pgvector and query with RLS:
   ```bash
   docker compose -f infra/docker-compose.yml up -d postgres
   uv run python -m capstone.rag.ingest --target pg --strategy parent
   ```
   Prove RLS works: connect as a non-owner role, `SET app.tenant_id='t1'`, and show that you can't read `t2` rows even with `WHERE tenant_id='t2'`.
3. **Real embeddings (required):** run the ablation with `--embedder openai` and `--embedder bge` (`uv sync --extra deep`). Then write a **paraphrased question set**: 40 questions rewritten the way engineers actually talk ("X200 won't fire up, F28?"). Re-run. This is where hybrid search should earn its keep.
4. **Reranking (required):** add `--reranker cross-encoder`. Report the MRR gain vs the added latency per query.
5. **Disambiguation (required):** add a query-understanding step that extracts `{asset_model, error_code}` with structured output and applies a `doc_id` metadata filter. Target MRR >= 0.97 on the gold set.
6. **Generation eval (required):** build `capstone/rag/eval_generation.py`. For each gold QA, run the Copilot with an LLM, score `LLMFaithfulness`, citation validity, and answer correctness vs the gold `answer`. Calibrate the faithfulness judge against 30 of your own labels (reuse Week 2's kappa).
7. **Azure AI Search comparison (stretch, recommended if you target Azure shops):** index the same chunks in Azure AI Search (free tier) with hybrid + semantic ranker. Write one paragraph comparing quality, latency, cost, ops burden and multi-tenancy options against pgvector.
8. Wire the Copilot behind `POST /v1/agents/engineer-copilot/ask` with tenant auth.

### Acceptance criteria
- [ ] `tests/test_rag.py` passes; new components have tests.
- [ ] `docs/rag-ablation.md`: a table with real embeddings, a paraphrase set and the reranker, plus a recommendation.
- [ ] Faithfulness >= 0.9 and citation validity >= 0.95 on the gold set, with an abstention rate reported.

## Production bar
- Ingestion is idempotent (stable chunk ids, upserts) and re-embeds only changed docs (hash the content).
- Index the embedding model name and version with each chunk; changing the model means a full re-index with a dual-read cutover.
- Retrieved content is spotlighted as untrusted (documents can contain injections too).
- Retrieval quality is in CI (ablation on the gold set, with a threshold on MRR).

## Interview questions
1. *"Users say the bot gives wrong answers. How do you debug RAG?"* Split retrieval from generation. Check whether the right chunk is in the top k (recall); if not, look at chunking, embeddings, filters and query rewriting; if it is, check generation faithfulness, context ordering ("lost in the middle") and prompt instructions.
2. *"Why hybrid search?"* Lexical handles exact tokens (codes, SKUs, names); dense handles semantics. RRF fuses without score calibration. Show the ablation.
3. *"How do you choose chunk size?"* Empirically, on a gold set. It depends on document structure, the embedding model's context window and how answers are distributed. Small-to-big gets precise matching *and* full context.
4. *"How would you serve RAG for 10,000 tenants?"* Shared index with a tenant filter plus RLS for the long tail, dedicated indexes for large tenants, a per-tenant ingestion queue, embedding caching, and per-tenant quotas and metrics. Then cover the filtered-HNSW recall problem (iterative scans in pgvector 0.8+, or partitioning).
5. *"How do you measure faithfulness?"* Claim extraction plus NLI-style verification against the context (RAGAS), calibrated against humans, and tracked online on a sample.
