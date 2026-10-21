# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Model Serving: point the endpoint at `@champion`
# MAGIC
# MAGIC Serving endpoints pin a concrete version, so this task resolves the alias and updates the
# MAGIC endpoint only when the champion changed. Traffic shifts with zero downtime (the old
# MAGIC version serves until the new one is ready).
# MAGIC
# MAGIC Production settings worth knowing:
# MAGIC - `scale_to_zero_enabled`: cheap for dev; cold starts are slow, so disable it for agents.
# MAGIC - AI Gateway / inference tables: every request and response lands in a Delta table, which
# MAGIC   feeds drift monitoring (Lakehouse Monitoring) and the failure-mode notebook.
# MAGIC - Canary: two served entities with `traffic_config` routes (90/10 champion/challenger).
# MAGIC
# MAGIC Locally the agent's `asset_failure_risk` tool loads the MLflow model in process
# MAGIC (`capstone/ml/serving.py`). Against Databricks, the tool becomes an HTTP tool calling
# MAGIC `POST /serving-endpoints/fieldops-asset-failure/invocations` with a service principal
# MAGIC token: same tool spec, different implementation (lab in course/05).

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow import MlflowClient

w = WorkspaceClient()
MODEL = f"{FQ}.asset_failure"
ENDPOINT = "fieldops-asset-failure"
champion = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(MODEL, "champion")
entity = ServedEntityInput(entity_name=MODEL, entity_version=str(champion.version),
                           workload_size="Small", scale_to_zero_enabled=True)

existing = {e.name: e for e in w.serving_endpoints.list()}
if ENDPOINT not in existing:
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT, config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[entity]))
    print("created", ENDPOINT, "v", champion.version)
else:
    current = w.serving_endpoints.get(ENDPOINT).config.served_entities[0].entity_version
    if current != str(champion.version):
        w.serving_endpoints.update_config_and_wait(name=ENDPOINT, served_entities=[entity])
        print(f"updated {ENDPOINT}: v{current} -> v{champion.version}")
    else:
        print("endpoint already on champion v", current)

# COMMAND ----------

# Smoke test: the feature-store-packaged model needs only the lookup keys.
sample = spark.table(f"{FQ}.asset_failure_labels").limit(2).toPandas()  # noqa: F821
records = [{"asset_id": r.asset_id, "snapshot_date": r.snapshot_date.isoformat()}
           for r in sample.itertuples()]
resp = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=records)
print(resp.predictions)
