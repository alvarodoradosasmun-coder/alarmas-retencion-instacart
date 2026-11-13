import os
import streamlit as st
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

# ---------- Configuración ----------
cfg = Config()
w = WorkspaceClient()
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
ENDPOINT = os.getenv("SERVING_ENDPOINT")
UMBRAL = 0.2038

FEATURES = [
    "order_number", "mes_vida", "order_dow", "order_hour_of_day",
    "dias_acumulados", "days_since_prior_order", "gap_promedio", "gap_std",
    "gap_max", "n_gaps_30", "basket_size", "canasta_promedio",
    "n_departments", "reorden_ratio_pedido", "reorden_ratio_hist",
    "es_primer_pedido", "canasta_inicial",
]

ICONO = {"Alarma": "🔴 Alarma", "Vigilar": "🟡 Vigilar", "Sana": "🟢 Sana"}
SEGMENTOS = {"chica": "Chica (≤6 productos)", "mediana": "Mediana (7 a 12)", "grande": "Grande (>12)"}


def consulta(query, params=None):
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: cfg.authenticate,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            return cur.fetchall_arrow().to_pandas()


@st.cache_data(ttl=600)
def cargar_cohortes():
    return consulta("SELECT * FROM workspace.gold.cohortes")


def etiqueta_mes(m):
    return "7 o más" if int(m) == 7 else str(int(m))


# ---------- Pantalla ----------
st.set_page_config(page_title="Alarmas de retención", layout="wide")
st.title("Sistema de alarmas tempranas de retención")
st.caption("Instacart · 3.4 millones de pedidos · gradient boosting (AUC 0.82) · "
           f"endpoint: {ENDPOINT}")

coh = cargar_cohortes()

# ----- Consulta al modelo -----
st.markdown("### Consulta de un cliente")
c1, c2 = st.columns([1, 3])
uid = c1.number_input("ID de cliente (user_id)", min_value=1, max_value=206209, value=1, step=1)
consultar = c1.button("Consultar modelo", type="primary")

if consultar:
    with st.spinner("Consultando datos y modelo publicado..."):
        datos = consulta(
            """SELECT * FROM workspace.gold.features_pedido
               WHERE user_id = :uid
               ORDER BY order_number DESC LIMIT 1""",
            {"uid": int(uid)},
        )

        if datos.empty:
            st.error("No se encontró ese cliente.")
            st.stop()

        entrada = datos[FEATURES].astype(float)
        resp = w.serving_endpoints.query(
            name=ENDPOINT,
            dataframe_records=entrada.to_dict(orient="records"),
        )
        prob = float(resp.predictions[0][1])

    d = datos.iloc[0]
    cohorte = coh[(coh["mes_vida_grupo"] == d["mes_vida_grupo"]) &
                  (coh["segmento_canasta"] == d["segmento_canasta"])].iloc[0]

    clasif = "🔴 Riesgo alto" if prob >= UMBRAL else "🟢 Riesgo bajo"

    with c2:
        st.subheader(f"Cliente {int(uid)}: {clasif}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Prob. de tardar 30+ días", f"{prob:.1%}")
        k2.metric("Mes de vida", etiqueta_mes(d["mes_vida_grupo"]))
        k3.metric("Segmento", d["segmento_canasta"].capitalize())
        k4.metric("Semáforo de su cohorte", ICONO[cohorte["semaforo"]])
        st.caption(f"Riesgo alto si la probabilidad es ≥ {UMBRAL:.1%}. "
                   f"Su cohorte tiene {int(cohorte['clientes']):,} clientes y un índice de "
                   f"{cohorte['indice_vs_mes']:.3f} contra el promedio de su mes de vida.")

# ----- Semáforo de cohortes -----
st.markdown("### Semáforo de cohortes")
st.caption("Índice = riesgo promedio de la cohorte / riesgo promedio de su mismo mes de vida. "
           "🔴 Alarma ≥ 1.05 · 🟡 Vigilar ≥ 1.02 · 🟢 Sana < 1.02")

tabla = (coh.assign(
             mes=coh["mes_vida_grupo"].map(etiqueta_mes),
             semaforo=coh["semaforo"].map(ICONO))
         .pivot(index="mes", columns="segmento_canasta", values="semaforo")
         [["chica", "mediana", "grande"]]
         .rename(columns=SEGMENTOS))
st.dataframe(tabla, use_container_width=True)