# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Feature table: asset health features in Unity Catalog
# MAGIC
# MAGIC A **time-series feature table** keyed on `(asset_id, snapshot_date)`. Training joins it
# MAGIC with a *point-in-time* lookup: for a label at date T, it takes the latest feature row at
# MAGIC or before T. That stops leakage by construction, not by remembering to filter.
# MAGIC
# MAGIC The same table serves online: publish it to an online store (Lakebase / online tables),
# MAGIC and a model logged with `fe.log_model` looks features up by `asset_id` at request time,
# MAGIC so the caller (our agent tool) sends only the key. That's training/serving parity.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

fe = FeatureEngineeringClient()
snaps = spark.table(f"{FQ}.asset_snapshots")  # noqa: F821

features = (
    snaps.select(
        "asset_id", "tenant_id", F.to_timestamp("snapshot_date").alias("snapshot_date"),
        "asset_type", "age_years", "days_since_service", "service_interval_days",
        "fault_count_90d", "avg_daily_runtime_hours")
    .withColumn("overdue_ratio",
                F.least(F.col("days_since_service") / F.col("service_interval_days"), F.lit(5.0)))
)
labels = snaps.select("asset_id", F.to_timestamp("snapshot_date").alias("snapshot_date"),
                      F.col("failed_next_30d").cast("int").alias("failed_next_30d"))

# COMMAND ----------

TABLE = f"{FQ}.asset_health_features"
if not spark.catalog.tableExists(TABLE):  # noqa: F821
    fe.create_table(
        name=TABLE,
        primary_keys=["asset_id", "snapshot_date"],
        timeseries_columns="snapshot_date",
        df=features,
        description="Monthly asset health features for predictive maintenance. "
                    "Point-in-time lookups on snapshot_date.",
    )
else:
    fe.write_table(name=TABLE, df=features, mode="merge")

labels.write.format("delta").mode("overwrite").saveAsTable(f"{FQ}.asset_failure_labels")
print(TABLE, features.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- base rate by asset type: the imbalance the model has to deal with
# MAGIC SELECT f.asset_type, COUNT(*) AS n, ROUND(AVG(l.failed_next_30d), 3) AS failure_rate
# MAGIC FROM IDENTIFIER(:catalog || '.' || :schema || '.asset_health_features') f
# MAGIC JOIN IDENTIFIER(:catalog || '.' || :schema || '.asset_failure_labels') l
# MAGIC   USING (asset_id, snapshot_date)
# MAGIC GROUP BY f.asset_type ORDER BY failure_rate DESC
