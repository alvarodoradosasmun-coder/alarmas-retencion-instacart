# Databricks notebook source
display(dbutils.fs.ls("/Volumes/workspace/bronze/raw_files"))

# COMMAND ----------

from pyspark.sql import functions as F

RAW = "/Volumes/workspace/bronze/raw_files"

archivos = {
    "orders": "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
    "products": "products.csv",
    "aisles": "aisles.csv",
    "departments": "departments.csv",
}

for tabla, archivo in archivos.items():
    df = (spark.read
          .option("header", True)
          .option("quote", '"')
          .option("escape", '"')
          .csv(f"{RAW}/{archivo}")
          .withColumn("_ingest_ts", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path")))
    (df.write.mode("overwrite")
       .option("overwriteSchema", True)
       .saveAsTable(f"workspace.bronze.{tabla}"))
    print(f"{tabla}: {spark.table(f'workspace.bronze.{tabla}').count():,} registros")

# COMMAND ----------

