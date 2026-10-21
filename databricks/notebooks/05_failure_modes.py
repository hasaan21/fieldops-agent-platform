# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Failure-mode analysis over agent traces
# MAGIC
# MAGIC Input: a Delta table of agent runs with eval scores, one row per (run, metric). Feed it
# MAGIC from local eval artifacts (`artifacts/evals/*/results.jsonl`), online-scored LangSmith
# MAGIC runs, or the app's `run_ledger` table.
# MAGIC Same analysis as `capstone/analysis/failure_modes.py`: SQL slices, embedding clusters,
# MAGIC impact ranking. Only the loader changes.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import json

import pandas as pd

from capstone.analysis.failure_modes import SEVERITY, cluster, describe_clusters

TRACES = f"{FQ}.agent_eval_results"
if not spark.catalog.tableExists(TRACES):  # noqa: F821
    # First run: build the table from the labelled transcripts so the notebook is runnable.
    # Each row is one transcript; the real table comes from LangSmith/online scoring.
    tx = spark.table(f"{FQ}.transcripts").toPandas()  # noqa: F821
    rows = []
    for t in tx.itertuples():
        labels = json.loads(t.labels)
        if labels.get("failure_mode"):
            rows.append({"run": "seed", "id": t.transcript_id, "tenant_id": t.tenant_id,
                         "channel": t.channel, "message": t.message,
                         "failure_mode": labels["failure_mode"], "metric": "outcome_correct",
                         "score": 0.0, "comment": "seeded from labelled failure modes"})
    spark.createDataFrame(pd.DataFrame(rows)).write.saveAsTable(TRACES)  # noqa: F821

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Which slices cost the most? Severity weights mirror SEVERITY in the Python module.
# MAGIC SELECT failure_mode, channel, COUNT(*) AS failures,
# MAGIC        SUM(CASE metric WHEN 'safety_escalation' THEN 20 WHEN 'no_sensitive_leak' THEN 15
# MAGIC                        WHEN 'outcome_correct' THEN 5 WHEN 'urgency_score' THEN 3
# MAGIC                        ELSE 2 END * (1 - score)) AS cost
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.agent_eval_results')
# MAGIC GROUP BY failure_mode, channel ORDER BY cost DESC LIMIT 20

# COMMAND ----------

fails = spark.table(TRACES).filter("score < 1").toPandas()  # noqa: F821
fails["cost"] = fails["metric"].map(SEVERITY).fillna(1.0) * (1 - fails["score"])

# Swap in a real embedding model on Databricks, e.g. the Foundation Model API endpoint:
#   from capstone.rag.embeddings import OpenAIEmbedder  (or an ai_query()-based embedder)
msgs = cluster(fails)
clusters = describe_clusters(msgs)
display(clusters)  # noqa: F821

# COMMAND ----------

(spark.createDataFrame(clusters)  # noqa: F821
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{FQ}.failure_mode_backlog"))
print("ranked backlog ->", f"{FQ}.failure_mode_backlog")
