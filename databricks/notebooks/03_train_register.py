# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Train, evaluate, register in Unity Catalog, promote by alias
# MAGIC
# MAGIC - Training set built from the feature table with a **point-in-time** lookup.
# MAGIC - Candidates come from `capstone.ml.train_failure.candidates`, the same code as local.
# MAGIC - Model registered to UC as `<catalog>.<schema>.asset_failure` with `fe.log_model`, so
# MAGIC   the feature lookup is packaged with it (serving needs only `asset_id` + timestamp).
# MAGIC - Promotion: new version gets `@challenger`; it becomes `@champion` only if it beats the
# MAGIC   current champion on PR-AUC on the *same* test window. Serving follows the alias.
# MAGIC
# MAGIC Duration and urgency follow the same pattern (`capstone.ml.train_duration` /
# MAGIC `train_urgency` with `spark.table(...).toPandas()` as input); the asset model is shown in
# MAGIC full because it is the one that uses the feature store. Lab: port the other two.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering scikit-learn>=1.6 --quiet

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from capstone.ml.features import SNAPSHOT_NUMERIC
from capstone.ml.train_failure import candidates, precision_at_k

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]  # noqa: F821
mlflow.set_experiment(f"/Users/{user}/fieldops-asset-failure")
fe = FeatureEngineeringClient()
MODEL = f"{FQ}.asset_failure"
FEATS = SNAPSHOT_NUMERIC + ["asset_type"]

# COMMAND ----------

lookups = [FeatureLookup(table_name=f"{FQ}.asset_health_features", lookup_key="asset_id",
                         timestamp_lookup_key="snapshot_date", feature_names=FEATS)]
training_set = fe.create_training_set(df=spark.table(f"{FQ}.asset_failure_labels"),  # noqa: F821
                                      feature_lookups=lookups, label="failed_next_30d",
                                      exclude_columns=[])
pdf = training_set.load_df().toPandas().sort_values("snapshot_date")
cut = pdf["snapshot_date"].quantile(0.7)
train, test = pdf[pdf["snapshot_date"] <= cut], pdf[pdf["snapshot_date"] > cut]
Xtr, ytr = train[FEATS], train["failed_next_30d"].to_numpy()
Xte, yte = test[FEATS], test["failed_next_30d"].to_numpy()
print(len(train), len(test), "base rate", ytr.mean().round(3))

# COMMAND ----------

results, runs = {}, {}
for name, model in candidates(FEATS).items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        m = {"pr_auc": average_precision_score(yte, p), "roc_auc": roc_auc_score(yte, p),
             "brier": brier_score_loss(yte, p), "precision_at_5pct": precision_at_k(yte, p)}
        mlflow.log_metrics(m)
        mlflow.log_param("model", name)
        fe.log_model(model=model, artifact_path="model", flavor=mlflow.sklearn,
                     training_set=training_set, registered_model_name=None,
                     infer_input_example=True)
        results[name], runs[name] = m, run.info.run_id
best = max(results, key=lambda k: results[k]["pr_auc"])
print(results, "->", best)

# COMMAND ----------

client = MlflowClient(registry_uri="databricks-uc")
mv = mlflow.register_model(f"runs:/{runs[best]}/model", MODEL)
client.set_registered_model_alias(MODEL, "challenger", mv.version)
try:
    champ = client.get_model_version_by_alias(MODEL, "champion")
    champ_pr = client.get_run(champ.run_id).data.metrics["pr_auc"]
except Exception:
    champ_pr = None
new_pr = results[best]["pr_auc"]
if champ_pr is None or new_pr > champ_pr:
    client.set_registered_model_alias(MODEL, "champion", mv.version)
    decision = f"promoted v{mv.version} (pr_auc {new_pr:.4f} vs {champ_pr})"
else:
    decision = f"kept champion (pr_auc {champ_pr:.4f}); v{mv.version} stays challenger"
client.update_model_version(MODEL, mv.version, description=f"{best}: {decision}")
dbutils.jobs.taskValues.set("decision", decision)  # noqa: F821
print(decision)

# COMMAND ----------

# MAGIC %md
# MAGIC Caveat for interviews: comparing PR-AUC of the new model against the *logged* metric of
# MAGIC the old champion is only fair if both were evaluated on the same window. A stricter
# MAGIC version re-scores the current champion on today's test window before deciding.
