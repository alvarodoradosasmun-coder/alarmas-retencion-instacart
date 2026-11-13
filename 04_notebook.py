# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

FEATURES = [
    "order_number", "mes_vida", "order_dow", "order_hour_of_day",
    "dias_acumulados", "days_since_prior_order", "gap_promedio", "gap_std",
    "gap_max", "n_gaps_30", "basket_size", "canasta_promedio",
    "n_departments", "reorden_ratio_pedido", "reorden_ratio_hist",
    "es_primer_pedido", "canasta_inicial",
]
TARGET = "label_riesgo"

g = spark.table("workspace.gold.features_pedido")

# 40% de los clientes, siempre los mismos (hash del user_id)
muestra = g.filter(F.pmod(F.xxhash64("user_id"), F.lit(100)) < 40)

pdf = muestra.select("user_id", *FEATURES, TARGET).toPandas()
pdf[FEATURES] = pdf[FEATURES].astype("float64")

print(f"renglones en la muestra:     {len(pdf):,}")
print(f"clientes en la muestra:      {pdf['user_id'].nunique():,}")
print(f"tasa de riesgo en la muestra: {pdf[TARGET].mean():.2%}")

# COMMAND ----------

from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(gss.split(pdf, groups=pdf["user_id"]))

train, test = pdf.iloc[idx_train], pdf.iloc[idx_test]
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

print(f"entrenamiento: {len(train):,} renglones | {train['user_id'].nunique():,} clientes")
print(f"prueba:        {len(test):,} renglones | {test['user_id'].nunique():,} clientes")
print(f"clientes en ambos: {len(set(train['user_id']) & set(test['user_id']))}")

# COMMAND ----------

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from mlflow.models import infer_signature

usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/instacart_riesgo")
mlflow.set_registry_uri("databricks-uc")

def evaluar(modelo, nombre):
    with mlflow.start_run(run_name=nombre) as run:
        modelo.fit(X_train, y_train)
        prob = modelo.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, prob)
        ap = average_precision_score(y_test, prob)
        prec, rec, thr = precision_recall_curve(y_test, prob)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        i = int(np.argmax(f1[:-1]))

        mlflow.log_param("modelo", nombre)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_metrics({
            "auc_roc": auc,
            "auc_pr": ap,
            "umbral_f1": float(thr[i]),
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        })

        firma = infer_signature(X_test.head(100), modelo.predict_proba(X_test.head(100)))
        mlflow.sklearn.log_model(
            modelo, "model",
            signature=firma,
            input_example=X_test.head(5),
            pyfunc_predict_fn="predict_proba",
        )

        print(f"{nombre}: AUC-ROC {auc:.4f} | AUC-PR {ap:.4f} | "
              f"umbral {thr[i]:.3f} | precisión {prec[i]:.2%} | recall {rec[i]:.2%}")
        return run.info.run_id, auc

# COMMAND ----------

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

run_lr, auc_lr = evaluar(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "regresion_logistica",
)

run_hgb, auc_hgb = evaluar(
    HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
        early_stopping=True, random_state=42,
    ),
    "gradient_boosting",
)

# COMMAND ----------

from mlflow import MlflowClient

NOMBRE = "workspace.gold.modelo_riesgo"
ganador, nombre_ganador = (run_hgb, "gradient_boosting") if auc_hgb >= auc_lr else (run_lr, "regresion_logistica")

version = mlflow.register_model(f"runs:/{ganador}/model", NOMBRE)
MlflowClient().set_registered_model_alias(NOMBRE, "champion", version.version)

print(f"Modelo registrado: {NOMBRE} v{version.version} ({nombre_ganador}) con alias 'champion'")