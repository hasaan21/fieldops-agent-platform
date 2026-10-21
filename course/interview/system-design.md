# System design drills

Three drills, 45 minutes each, in the order an AI-platform interview tends to go. Do each one
twice: once on paper against the clock, then once out loud to someone (or a recording), because
the second half of the interview is follow-ups.

A structure that works for all three:
1. **Clarify** (5 min): users, scale, latency and cost budgets, tenancy, what "good" means.
2. **API and data** (5 min): the main entities and how they're keyed by tenant.
3. **Architecture** (15 min): a diagram, the request path, the async path.
4. **Quality** (10 min): how you'll *know* it works: evals, metrics, alerts.
5. **Trade-offs and failure modes** (10 min): what breaks first at 10x, what you'd cut for v1.

Always say numbers out loud ("10k tenants × 2k docs × 30 chunks = 600M chunks"). Back-of-the-
envelope maths is what separates a design from a list of boxes.

---

## Drill 1: Design a multi-tenant agent evaluation platform

> "Our customers build agents in a no-code builder. Design the evaluation system so every agent
> change is tested before it ships, and quality is tracked after."

**Clarify:** who writes evals (platform engineers, tenant admins, or both)? How many agents
(tens of thousands of tenants, maybe one to five agents each)? Is a tenant's data allowed in
shared datasets (almost certainly not)? Who pays for judge tokens?

**Core entities:** `Dataset(tenant_id | platform, agent_type, version)`, `Example(inputs,
reference, metadata.failure_mode, provenance)`, `Evaluator(kind: heuristic | judge | pairwise,
rubric_version, calibration)`, `Experiment(agent_version, prompt_refs, dataset_version)`,
`Score(example, evaluator, value, comment)`, `Threshold(agent_type, metric, absolute, max_regression)`.

**Architecture to draw:**
- **Two dataset layers.** Platform datasets per agent *type* (built by the platform team from synthetic
  data and consented, redacted traces) and tenant datasets per agent (examples the tenant
  supplies or approves). Platform sets gate template changes; tenant sets gate tenant edits.
- **Offline runner** as a queue of (example × agent version) jobs on workers, rate-limited per
  provider, with caching on (prompt hash, input hash) so re-runs are cheap. Results land in a
  columnar store for slicing.
- **Evaluators:** heuristics first (schema, tool trajectory, policy rules: cheap, deterministic),
  then a rubric judge with *pinned* model and rubric versions, calibrated against human labels
  (kappa ≥ 0.6 before it can gate). Pairwise with position swap for A/B.
- **Gate:** absolute floors (safety = 1.0) and regression limits with confidence intervals; a
  small n can't "pass" a regression check (use Wilson intervals or bootstrap).
- **Online:** sample live runs per tenant (stratified so small tenants are represented), run
  the same evaluators asynchronously, plus user feedback; drift on input mix (PSI) and outcome
  mix; failing runs flow to a review queue and then into regression datasets.
- **Tenant isolation:** every dataset, run and score keyed by tenant with row-level security;
  judges never see another tenant's data; per-tenant judge budgets.

**Numbers:** 30k agents × 1 change/week × 200 examples × 2 judge calls = 12M judge calls/week.
At about $0.001 per small-model judge call that's around $12k/week: too much to judge
everything. Hence heuristics first, judge on a sample or only on changed behaviour, cache
aggressively, and tier (full suites for templates, small suites for tenant edits).

**Follow-ups to prepare:** how do you evaluate a tenant's agent with no labelled data
(synthetic examples from their KB and config, plus golden templates per agent type); how do you
stop the judge drifting when the provider updates the model (pin versions, re-calibrate on a
fixed human-labelled set monthly, alert on agreement drop); how do you handle non-determinism
(multiple samples, report variance, temperature 0 for gates).

**Capstone evidence:** `capstone/evals/` (runner, gate, judge, calibration, online, feedback loop).

## Drill 2: Design RAG for 10k tenants

> "Each tenant uploads manuals, price lists and FAQs. Agents must answer from the tenant's own
> knowledge, grounded, with citations."

**Clarify:** document volume per tenant (long tail: most have 50 docs, a few have 50k), update
frequency (price lists change weekly), languages, latency budget (voice 1.5 s total, so
retrieval gets about 200 ms), whether tenants can share platform content (manufacturer manuals).

**Numbers:** 10k tenants × average 500 docs × 40 chunks = 200M chunks. At 1536 dimensions in
float32 that's about 1.2 TB of vectors; at 512 dimensions with int8 quantisation, around 100 GB.
That decides the storage design.

**Architecture to draw:**
- **Ingestion pipeline (async):** upload, parse (PDF layout-aware, tables kept intact),
  structure-aware chunking (headings, error-code tables as single chunks), metadata (tenant,
  doc type, asset model, effective date), embed in batches, write. Idempotent per document
  version; deletes must propagate (a tenant removing a price list must stop citations to it).
- **Storage and tenancy:** shared index with a mandatory `tenant_id` filter for the long tail
  (pgvector with partitioning by tenant hash, or Azure AI Search with filters); dedicated
  partitions or indexes for the largest tenants. Shared platform content (manufacturer manuals)
  in a separate index, merged at query time. RLS so a missing filter fails closed.
- **Query path:** query rewriting with conversation context, hybrid retrieval (BM25 for error
  codes and model numbers, vectors for paraphrase), reciprocal rank fusion, cross-encoder rerank
  on the top 50, then generation with citations and an abstain path when retrieval is weak.
- **Quality:** retrieval metrics (recall@k, MRR, nDCG) on a gold QA set per document type;
  claim-level faithfulness and citation validity on generations; per-tenant canaries; a
  dashboard of "answers with no citation" and "abstain rate" by tenant.

**Trade-offs to discuss:** one big index vs per-tenant indexes (operational cost vs noisy-
neighbour risk and filter selectivity); embedding model upgrades (re-embedding 200M chunks
needs dual-write and a backfill, so version the embedding model in metadata); freshness vs cost
(re-embed only changed chunks via content hashing).

**Capstone evidence:** the chunking ablation (structure-aware chunking took recall@5 from 0.66 to
1.00; the retriever mattered less), hybrid search with RRF, tenant filters, faithfulness metrics.

## Drill 3: Online drift detection for agents

> "Agents are live for thousands of tenants. How do you know quality is getting worse before
> customers tell you?"

**Clarify:** what can drift: inputs (new kinds of requests, a new channel, seasonality, e.g.
boiler faults in winter), the model (provider updates), prompts and tools (deployments), the
world (prices, KB changes). Which signals exist with no labels?

**Signals, cheapest first:**
- **Operational:** latency by channel, error rate by provider and error message, cost per run,
  tool-call failure rate. (Provider-side breaking changes usually show up first as a spike in one error message.)
- **Behavioural (no labels needed):** outcome mix per tenant (booked / escalated / clarified /
  declined), escalation rate, clarification loops, average turns, reply length, tool-use mix,
  abstain rate in RAG. Track with PSI or chi-squared against a rolling baseline.
- **Input drift:** intent and category distribution from the classifier, embedding-based drift
  (cluster centroids, the share of messages far from any training cluster).
- **Quality (sampled):** online heuristic evaluators on every run, a judge on a stratified
  sample, user feedback, human review agreement rates.

**Architecture to draw:** traces into a stream; a scoring worker computes heuristics and samples
for the judge; aggregates per (tenant, agent, channel, hour) into a time-series store; a drift
job compares windows with PSI and two-proportion z-tests; alerts route by severity; drill-down
links go straight to example traces. Failing examples feed the review queue and regression
datasets.

**Pitfalls to name:** multiple comparisons across thousands of tenants (control false-discovery
rate, require minimum sample sizes, alert on platform-level shifts separately); seasonality
(compare to the same weekday or use rolling baselines); a deploy or prompt tag change is a known
change point, so annotate it and compare before/after explicitly; small tenants (pool by agent
type).

**Capstone evidence:** `capstone/evals/online.py` (PSI and z-tests), the ledger's budget
violations as a metric, `failure_modes.py` clustering, and the feedback loop.
