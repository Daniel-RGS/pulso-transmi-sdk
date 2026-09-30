from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Pulso TransMi · MLOps", page_icon="📈", layout="wide")


@st.cache_data(ttl=60)
def load_table(table: str, select: str = "*") -> pd.DataFrame:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Configura SUPABASE_URL y SUPABASE_KEY antes de iniciar el dashboard")
    response = httpx.get(
        f"{url.rstrip('/')}/rest/v1/{table}",
        params={"select": select},
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=30,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


st.title("Pulso TransMi")
st.caption("Panel operativo de modelos, predicciones y experimentos")

try:
    metrics = load_table("metrics")
    models = load_table("models")
    runs = load_table("pipeline_runs")
    predictions = load_table("predictions")
except Exception as error:
    st.error(str(error))
    st.stop()

if metrics.empty:
    st.info("Aún no hay métricas registradas.")
    st.stop()

metrics["metric_value"] = pd.to_numeric(metrics["metric_value"])
accuracy = metrics[metrics["metric_name"].str.startswith("accuracy_")].copy()
accuracy["horizon"] = accuracy["metric_name"].str.removeprefix("accuracy_")
wape = metrics[metrics["metric_name"].str.startswith("wape_")].copy()
wape["horizon"] = wape["metric_name"].str.removeprefix("wape_")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy promedio", f"{accuracy['metric_value'].mean():.2f}%")
col2.metric("Modelos registrados", str(len(models)))
col3.metric("Predicciones", str(len(predictions)))
col4.metric("Ejecuciones", str(len(runs)))

left, right = st.columns(2)
with left:
    st.subheader("Accuracy por horizonte")
    chart = accuracy.set_index("horizon")["metric_value"].sort_index()
    st.bar_chart(chart)
with right:
    st.subheader("WAPE por horizonte")
    chart = wape.set_index("horizon")["metric_value"].sort_index()
    st.bar_chart(chart)

st.subheader("Modelos activos y versiones")
st.dataframe(
    models[["model_name", "algorithm", "version", "is_active", "created_at"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Últimas predicciones")
if predictions.empty:
    st.info("No hay predicciones registradas.")
else:
    predictions["predicted_demand"] = pd.to_numeric(predictions["predicted_demand"])
    predictions["target_at"] = pd.to_datetime(predictions["target_at"])
    st.dataframe(
        predictions.sort_values("target_at", ascending=False)[
            ["station_id", "target_at", "horizon_minutes", "predicted_demand", "actual_demand"]
        ].head(48),
        use_container_width=True,
        hide_index=True,
    )