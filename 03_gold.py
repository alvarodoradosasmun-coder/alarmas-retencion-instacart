# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

orders = spark.table("workspace.silver.orders")
opx = spark.table("workspace.silver.order_products")

canasta = (opx.groupBy("order_id").agg(
    F.count("*").alias("basket_size"),
    F.sum("reordered").alias("n_reordered"),
    F.countDistinct("department").alias("n_departments"),
))

print(f"pedidos con detalle: {canasta.count():,}")

# COMMAND ----------

w_orden = Window.partitionBy("user_id").orderBy("order_number")
w_hist = w_orden.rowsBetween(Window.unboundedPreceding, 0)

base = (orders.join(canasta, "order_id", "left")
    # Etiqueta: días hasta el SIGUIENTE pedido (solo para la etiqueta)
    .withColumn("next_gap", F.lead("days_since_prior_order").over(w_orden))
    .withColumn("label_riesgo", F.when(F.col("next_gap") == 30, 1).otherwise(0))

    # Canasta del primer pedido (define el segmento de la cohorte)
    .withColumn("canasta_inicial", F.first("basket_size").over(w_hist))

    # Historial de días entre pedidos hasta el pedido actual
    .withColumn("gap_promedio", F.avg("days_since_prior_order").over(w_hist))
    .withColumn("gap_std", F.stddev("days_since_prior_order").over(w_hist))
    .withColumn("gap_max", F.max("days_since_prior_order").over(w_hist))
    .withColumn("n_gaps_30", F.sum(
        F.when(F.col("days_since_prior_order") == 30, 1).otherwise(0)).over(w_hist))

    # Historial de canasta y recompra hasta el pedido actual
    .withColumn("canasta_promedio", F.avg("basket_size").over(w_hist))
    .withColumn("reorden_ratio_pedido", F.col("n_reordered") / F.col("basket_size"))
    .withColumn("reorden_ratio_hist",
        F.sum("n_reordered").over(w_hist) / F.sum("basket_size").over(w_hist))

    .withColumn("es_primer_pedido", (F.col("order_number") == 1).cast("int"))
    .withColumn("mes_vida_grupo", F.least(F.col("mes_vida"), F.lit(7)))
)

# Solo pedidos prior con siguiente pedido conocido
features = (base
    .filter((F.col("eval_set") == "prior") & F.col("next_gap").isNotNull())
    .drop("next_gap", "eval_set"))

print(f"renglones: {features.count():,}")

# COMMAND ----------

clientes = features.select("user_id", "canasta_inicial").distinct()
q1, q2 = clientes.approxQuantile("canasta_inicial", [1/3, 2/3], 0.001)
print(f"Cortes canasta inicial -> chica: <= {q1:.0f} | mediana: <= {q2:.0f} | grande: > {q2:.0f}")

features = features.withColumn(
    "segmento_canasta",
    F.when(F.col("canasta_inicial") <= q1, "chica")
     .when(F.col("canasta_inicial") <= q2, "mediana")
     .otherwise("grande"))

# COMMAND ----------

cols_gap = ["days_since_prior_order", "gap_promedio", "gap_std", "gap_max"]

gold = (features
    .fillna(0, subset=cols_gap)
    .select(
        "user_id", "order_id", "order_number",
        "mes_vida", "mes_vida_grupo", "segmento_canasta", "canasta_inicial",
        "order_dow", "order_hour_of_day", "dias_acumulados",
        "days_since_prior_order", "gap_promedio", "gap_std", "gap_max", "n_gaps_30",
        "basket_size", "canasta_promedio", "n_departments",
        "reorden_ratio_pedido", "reorden_ratio_hist", "es_primer_pedido",
        "label_riesgo",
    ))

(gold.write
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.gold.features_pedido"))

print(f"gold.features_pedido: {spark.table('workspace.gold.features_pedido').count():,}")

# COMMAND ----------

g = spark.table("workspace.gold.features_pedido")

print(f"tasa de riesgo global: {g.agg(F.avg('label_riesgo')).first()[0]:.2%}")
print(f"nulos en variables:    {g.filter(F.col('reorden_ratio_hist').isNull() | F.col('basket_size').isNull()).count():,}")

display(g.groupBy("segmento_canasta").agg(
        F.countDistinct("user_id").alias("clientes"),
        F.avg("label_riesgo").alias("tasa_riesgo"))
        .orderBy("segmento_canasta"))

display(g.groupBy("mes_vida_grupo", "segmento_canasta").agg(
        F.count("*").alias("pedidos"),
        F.avg("label_riesgo").alias("tasa_riesgo"))
        .orderBy("mes_vida_grupo", "segmento_canasta"))