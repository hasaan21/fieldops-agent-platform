# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest: synthetic world to Delta tables
# MAGIC
# MAGIC In production this notebook would be a Lakeflow (DLT) pipeline reading the operational
# MAGIC database via CDC. Here we generate the synthetic world with the same seed as the local
# MAGIC repo and land it as managed Delta tables in Unity Catalog.
# MAGIC
# MAGIC Things to point at in an interview:
# MAGIC - `overwrite` + `overwriteSchema` is fine for a reproducible synthetic load. Real
# MAGIC   ingestion is append/merge (`MERGE INTO ... WHEN MATCHED`), keyed on the business id.
# MAGIC - Table and column comments live in UC, so they show up in Catalog Explorer and Genie.
# MAGIC - `tenant_id` is on every table. Row filters (`ALTER TABLE ... SET ROW FILTER`) enforce
# MAGIC   tenant isolation for analysts, like the Postgres RLS policy in the app.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import json

import pandas as pd

from capstone.data.synthetic.generator import Generator

world = Generator(seed=7).build().tables()
print({k: len(v) for k, v in world.items()})

# COMMAND ----------

COMMENTS = {
    "jobs": "Historical field-service jobs (one row per job). Label columns: duration_hours, "
            "urgency (dispatcher), true_urgency.",
    "asset_snapshots": "Monthly asset snapshots; failed_next_30d is the predictive "
                       "maintenance label.",
    "transcripts": "Inbound customer messages with eval labels.",
}

for name, rows in world.items():
    if not rows:
        continue
    pdf = pd.DataFrame(rows)
    # nested dict/list columns (labels, trades, facts) are stored as JSON strings
    for col in pdf.columns:
        if pdf[col].map(lambda v: isinstance(v, dict | list)).any():
            pdf[col] = pdf[col].map(lambda v: None if v is None else json.dumps(v))
    sdf = spark.createDataFrame(pdf)  # noqa: F821
    (sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{FQ}.{name}"))
    if name in COMMENTS:
        spark.sql(f"COMMENT ON TABLE {FQ}.{name} IS '{COMMENTS[name]}'")  # noqa: F821
    print(f"{FQ}.{name}: {sdf.count()} rows")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- sanity checks a data engineer would add as expectations
# MAGIC SELECT tenant_id, COUNT(*) AS jobs, ROUND(AVG(duration_hours), 2) AS avg_hours,
# MAGIC        ROUND(AVG(CASE WHEN urgency = true_urgency THEN 1 ELSE 0 END), 3) AS label_agreement
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.jobs')
# MAGIC GROUP BY tenant_id ORDER BY tenant_id
