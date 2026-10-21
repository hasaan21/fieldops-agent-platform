# Databricks notebook source
# MAGIC %md
# MAGIC Shared setup, included by the other notebooks with `%run ./_bootstrap`.
# MAGIC Puts the synced `capstone` package on `sys.path` and reads the job parameters.

# COMMAND ----------

import os
import sys

# The bundle syncs ../capstone next to ./notebooks, so the package root is two levels up.
_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
for candidate in (_root, os.path.abspath(os.path.join(os.getcwd(), ".."))):
    if os.path.isdir(os.path.join(candidate, "capstone")) and candidate not in sys.path:
        sys.path.insert(0, candidate)

dbutils.widgets.text("catalog", "workspace")  # noqa: F821
dbutils.widgets.text("schema", "fieldops")  # noqa: F821
CATALOG = dbutils.widgets.get("catalog")  # noqa: F821
SCHEMA = dbutils.widgets.get("schema")  # noqa: F821
FQ = f"{CATALOG}.{SCHEMA}"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")  # noqa: F821
print("tables and models go to", FQ)
