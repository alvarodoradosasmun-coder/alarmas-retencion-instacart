# Databricks notebook source
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.window import Window

mlflow.set_registry_uri("databricks-uc")

MODELO = "models:/workspace.gold.modelo_riesgo@champion"
UMBRAL = 0.2038   # umbral F1 del gradient boosting

FEATURES = [
    "order_number", "mes_vida", "order_dow", "order_hour_of_day",
    "dias_acumulados", "days_since_prior_order", "gap_promedio", "gap_std",
    "gap_max", "n_gaps_30", "basket_size", "canasta_promedio",
    "n_departments", "reorden_ratio_pedido", "reorden_ratio_hist",
    "es_primer_pedido", "canasta_inicial",
]

predict = mlflow.pyfunc.spark_udf(spark, MODELO, result_type="array<double>")
print("Modelo cargado")

# COMMAND ----------

ids = ["user_id", "order_id", "mes_vida_grupo", "segmento_canasta", "label_riesgo"]

entrada = spark.table("workspace.gold.features_pedido").select(
    *ids, *[F.col(c).cast("double").alias(c) for c in FEATURES])

pred = (entrada
    .withColumn("prob_riesgo", predict(*[F.col(c) for c in FEATURES])[1])
    .withColumn("riesgo_alto", (F.col("prob_riesgo") >= UMBRAL).cast("int"))
    .select(
        *ids,
        F.col("order_number").cast("int").alias("order_number"),
        F.col("mes_vida").cast("int").alias("mes_vida"),
        "prob_riesgo", "riesgo_alto",
    ))

(pred.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.gold.predicciones_pedido"))

p = spark.table("workspace.gold.predicciones_pedido")
print(f"renglones calificados:   {p.count():,}")
print(f"probabilidad promedio:   {p.agg(F.avg('prob_riesgo')).first()[0]:.2%}")
print(f"tasa real de riesgo:     {p.agg(F.avg('label_riesgo')).first()[0]:.2%}")
print(f"% marcado riesgo alto:   {p.agg(F.avg('riesgo_alto')).first()[0]:.2%}")

# COMMAND ----------

from sklearn.metrics import roc_auc_score

fuera = (p.filter(F.pmod(F.xxhash64("user_id"), F.lit(100)) >= 40)
          .select("label_riesgo", "prob_riesgo").toPandas())

print(f"clientes nunca vistos, renglones: {len(fuera):,}")
print(f"AUC-ROC fuera de muestra:         {roc_auc_score(fuera['label_riesgo'], fuera['prob_riesgo']):.4f}")

# COMMAND ----------

w_ult = Window.partitionBy("user_id").orderBy(F.col("order_number").desc())

estado = (p.withColumn("rn", F.row_number().over(w_ult))
           .filter("rn = 1").drop("rn"))

(estado.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.gold.clientes_estado_actual"))

print(f"clientes: {spark.table('workspace.gold.clientes_estado_actual').count():,}")

# COMMAND ----------

display(p
    .withColumn("es_ultimo_pedido", F.row_number().over(w_ult) == 1)
    .groupBy("es_ultimo_pedido")
    .agg(F.count("*").alias("renglones"),
         F.avg("label_riesgo").alias("tasa_real"),
         F.avg("prob_riesgo").alias("prob_promedio")))

# COMMAND ----------

ROJO, AMARILLO = 1.05, 1.02

w_mes = Window.partitionBy("mes_vida_grupo")

def promedio_mes(col):
    return F.sum(F.col(col) * F.col("clientes")).over(w_mes) / F.sum("clientes").over(w_mes)

cohortes = (spark.table("workspace.gold.clientes_estado_actual")
    .groupBy("mes_vida_grupo", "segmento_canasta")
    .agg(
        F.count("*").alias("clientes"),
        F.avg("prob_riesgo").alias("prob_promedio"),
        F.avg("riesgo_alto").alias("pct_riesgo_alto"),
        F.avg("label_riesgo").alias("tasa_real"),
    )
    .withColumn("prob_mes", promedio_mes("prob_promedio"))
    .withColumn("tasa_real_mes", promedio_mes("tasa_real"))
    .withColumn("indice_vs_mes", F.col("prob_promedio") / F.col("prob_mes"))
    .withColumn("indice_real_vs_mes", F.col("tasa_real") / F.col("tasa_real_mes"))
    .withColumn("semaforo",
        F.when(F.col("indice_vs_mes") >= ROJO, "Alarma")
         .when(F.col("indice_vs_mes") >= AMARILLO, "Vigilar")
         .otherwise("Sana"))
    .withColumn("coincide_con_real",
        ((F.col("indice_vs_mes") >= 1) == (F.col("indice_real_vs_mes") >= 1))))

(cohortes.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("workspace.gold.cohortes"))

display(spark.table("workspace.gold.cohortes")
    .select("mes_vida_grupo", "segmento_canasta", "clientes",
            F.round("prob_promedio", 3).alias("prob"),
            F.round("tasa_real", 3).alias("real"),
            F.round("indice_vs_mes", 3).alias("indice_modelo"),
            F.round("indice_real_vs_mes", 3).alias("indice_real"),
            "semaforo", "coincide_con_real")
    .orderBy("mes_vida_grupo", "segmento_canasta"))

c = spark.table("workspace.gold.cohortes")
print(f"cohortes donde modelo y realidad coinciden: {c.filter('coincide_con_real').count()} de {c.count()}")