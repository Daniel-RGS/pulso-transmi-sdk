"""
Pipeline Pulso TransMi — Descarga datos, entrena modelo ML y envía predicciones.
Daniel Santiago Rincón Gamba
"""
import os
import sys
import uuid
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
API_KEY = os.getenv("PULSO_API_KEY")
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")
MODEL_VERSION = "hgbr-adaptive-v2"


def get_client():
    """Return an authenticated httpx client."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "pulso-transmi-python/0.1.0",
    }
    return httpx.Client(base_url=API_URL, headers=headers, timeout=60)


# ── 1. DATA COLLECTION ─────────────────────────────────────────────────────
def download_starter_data(client):
    """Download the starter CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("stations.csv", "observations.csv", "context.csv"):
        path = DATA_DIR / fname
        if path.exists():
            continue
        resp = client.get(f"/v1/downloads/{fname}")
        resp.raise_for_status()
        path.write_bytes(resp.content)
        print(f"  Descargado: {fname}")


def download_stream_data(client):
    """Download all stream observations released during competition."""
    all_rows = []
    cursor = None
    while True:
        params = {"limit": 5000}
        if cursor:
            params["cursor"] = cursor
        resp = client.get("/v1/stream/observations", params=params)
        resp.raise_for_status()
        data = resp.json()
        rows = data["data"]
        all_rows.extend(rows)
        cursor = data.get("next_cursor")
        if not cursor:
            break
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(DATA_DIR / "stream_observations.csv", index=False)
        print(f"  Stream: {len(df)} filas")
    return all_rows


def load_all_observations():
    """Merge starter + stream observations into a single DataFrame."""
    obs = pd.read_csv(
        DATA_DIR / "observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)

    stream_path = DATA_DIR / "stream_observations.csv"
    if stream_path.exists():
        stream = pd.read_csv(stream_path, dtype={"station_id": "string"})
        stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
        # drop released_at if present
        if "released_at" in stream.columns:
            stream = stream.drop(columns=["released_at"])
        obs = pd.concat([obs, stream], ignore_index=True)
        obs = obs.drop_duplicates(subset=["observed_at", "station_id"]).sort_values(
            ["station_id", "observed_at"]
        )

    return obs


def load_context():
    """Load context (weather) data."""
    ctx = pd.read_csv(DATA_DIR / "context.csv", parse_dates=["observed_at"])
    ctx["observed_at"] = pd.to_datetime(ctx["observed_at"], utc=True)
    return ctx


# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────
def build_features(obs: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Build features for each (station, timestamp) row."""
    df = obs.copy()

    # Time-based features
    df["hour"] = df["observed_at"].dt.hour
    df["minute"] = df["observed_at"].dt.minute
    df["dow"] = df["observed_at"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15  # 0-95

    # Cyclical encoding of time
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Lag features per station
    df = df.sort_values(["station_id", "observed_at"])
    for lag in [1, 2, 3, 4, 96, 672]:
        # lag 1-4 = last 15-60 min, 96 = same time yesterday, 672 = same time last week
        df[f"lag_{lag}"] = df.groupby("station_id")["demand"].shift(lag)

    # Rolling means per station
    for window in [4, 12, 96]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("station_id")["demand"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Adaptive weekly profile (hl=14 days = 1344 periods of 15 min)
    # Exponentially weighted mean of demand for same (dow, time_slot) per station
    df["profile_key"] = df["station_id"] + "_" + df["dow"].astype(str) + "_" + df["time_slot"].astype(str)
    profile = (
        df.groupby("profile_key")["demand"]
        .transform(lambda x: x.shift(1).ewm(halflife=1344, min_periods=1).mean())
    )
    df["adaptive_profile"] = profile

    # Merge weather context
    ctx_cols = ["observed_at", "temperature_c", "rain_mm", "event_intensity"]
    available_cols = [c for c in ctx_cols if c in ctx.columns]
    if len(available_cols) > 1:
        df = df.merge(ctx[available_cols], on="observed_at", how="left")

    return df


FEATURE_COLS = [
    "hour", "minute", "dow", "is_weekend", "time_slot",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_96", "lag_672",
    "rolling_mean_4", "rolling_mean_12", "rolling_mean_96",
    "adaptive_profile",
    "temperature_c", "rain_mm", "event_intensity",
]


# ── 3. TRAINING ────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame) -> dict:
    """Train one HistGradientBoostingRegressor per station."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}

    available_features = [c for c in FEATURE_COLS if c in df.columns]

    for station_id, group in df.groupby("station_id"):
        train_data = group.dropna(subset=["demand"] + available_features)
        if len(train_data) < 100:
            print(f"  Estación {station_id}: pocos datos ({len(train_data)}), skip")
            continue

        X = train_data[available_features].values
        y = train_data["demand"].values

        model = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X, y)
        models[station_id] = model
        print(f"  Estación {station_id}: entrenado con {len(train_data)} filas")

    # Save models
    model_path = ARTIFACTS_DIR / "models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "features": available_features}, f)
    print(f"  Modelos guardados en {model_path}")

    return models


# ── 4. PREDICTION ──────────────────────────────────────────────────────────
def predict_for_cycle(cycle: dict, obs: pd.DataFrame, ctx: pd.DataFrame, models: dict) -> list:
    """Generate predictions for every target in the current cycle."""
    targets = cycle["targets"]
    data_cutoff = pd.Timestamp(cycle["data_cutoff"]).tz_convert("UTC")
    available_features = [c for c in FEATURE_COLS if c in obs.columns or c in [
        "hour", "minute", "dow", "is_weekend", "time_slot",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]]

    # Build features on recent data
    featured = build_features(obs, ctx)

    predictions = []
    for target in targets:
        station_id = target["station_id"]
        target_at = pd.Timestamp(target["target_at"]).tz_convert("UTC")

        if station_id not in models:
            # Fallback: use last known demand for this station
            station_data = obs[obs["station_id"] == station_id]
            fallback_value = float(station_data["demand"].iloc[-1]) if len(station_data) > 0 else 250.0
            predictions.append({
                "station_id": station_id,
                "target_at": target["target_at"],
                "value": round(max(0, fallback_value), 2),
            })
            continue

        model = models[station_id]

        # Build feature row for target time
        station_hist = featured[featured["station_id"] == station_id].copy()
        if station_hist.empty:
            predictions.append({
                "station_id": station_id,
                "target_at": target["target_at"],
                "value": 250.0,
            })
            continue

        # Use the last row of features as a template and override time features
        last_row = station_hist.iloc[-1].copy()

        hour = target_at.hour
        minute = target_at.minute
        dow = target_at.dayofweek

        last_row["hour"] = hour
        last_row["minute"] = minute
        last_row["dow"] = dow
        last_row["is_weekend"] = 1 if dow >= 5 else 0
        last_row["time_slot"] = hour * 4 + minute // 15
        last_row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        last_row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        last_row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        last_row["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Try to find matching historical demand for lag features
        # Look for same day-of-week and time_slot in history
        same_slot = station_hist[
            (station_hist["dow"] == dow) & (station_hist["time_slot"] == hour * 4 + minute // 15)
        ]
        if not same_slot.empty:
            last_row["adaptive_profile"] = same_slot["adaptive_profile"].iloc[-1]
            last_row["lag_672"] = same_slot["demand"].iloc[-1]

        # Collect features for model
        load_result = ARTIFACTS_DIR / "models.pkl"
        with open(load_result, "rb") as f:
            saved = pickle.load(f)
        feat_cols = saved["features"]

        feat_values = []
        for col in feat_cols:
            val = last_row.get(col, 0)
            if pd.isna(val):
                val = 0
            feat_values.append(float(val))

        X_pred = np.array([feat_values])
        pred_value = model.predict(X_pred)[0]
        pred_value = max(0, pred_value)  # demand can't be negative

        predictions.append({
            "station_id": station_id,
            "target_at": target["target_at"],
            "value": round(pred_value, 2),
        })

    return predictions


# ── 5. SUBMISSION ──────────────────────────────────────────────────────────
def submit_predictions(client, cycle: dict, predictions: list):
    """POST predictions to the API."""
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"run-{uuid.uuid4().hex[:8]}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {"version": MODEL_VERSION},
        "predictions": predictions,
    }

    idem_key = uuid.uuid4().hex
    resp = client.post(
        "/v1/submissions",
        json=payload,
        headers={"Idempotency-Key": idem_key},
    )

    if resp.status_code == 201:
        result = resp.json()
        print(f"  ✅ Enviado: {result['submission_id']} — {result['predictions_received']} predicciones")
        return result
    else:
        print(f"  ❌ Error {resp.status_code}: {resp.text}")
        return None


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    import time

    if not API_KEY:
        print("ERROR: Falta PULSO_API_KEY")
        sys.exit(1)

    print("=" * 60)
    print("PIPELINE PULSO TRANSMI — Daniel Santiago Rincón Gamba")
    print("=" * 60)

    client = get_client()

    # Step 1: Download data
    print("\n[1/5] Descargando datos...")
    download_starter_data(client)
    download_stream_data(client)

    # Step 2: Load and merge data
    print("\n[2/5] Cargando datos...")
    obs = load_all_observations()
    ctx = load_context()
    print(f"  Observaciones: {len(obs)} filas")
    print(f"  Rango: {obs['observed_at'].min()} → {obs['observed_at'].max()}")

    # Step 3: Feature engineering + training
    print("\n[3/5] Construyendo features y entrenando modelos...")
    featured = build_features(obs, ctx)
    models = train_models(featured)
    print(f"  Modelos entrenados: {len(models)} estaciones")

    # Step 4: Get current cycle and predict (with retry)
    print("\n[4/5] Obteniendo ciclo actual...")
    cycle = None
    max_retries = 12
    for attempt in range(1, max_retries + 1):
        resp = client.get("/v1/forecast-cycles/current")
        if resp.status_code == 200:
            cycle = resp.json()
            if cycle.get("state") == "open":
                break
            else:
                print(f"  Intento {attempt}/{max_retries}: ciclo existe pero estado={cycle.get('state')}")
                cycle = None
        else:
            print(f"  Intento {attempt}/{max_retries}: no hay ciclo abierto (HTTP {resp.status_code})")

        if attempt < max_retries:
            print(f"  Esperando 30s antes de reintentar...")
            time.sleep(30)

    if not cycle:
        print("  ⚠️  No se encontró un ciclo abierto tras varios intentos.")
        print("  El pipeline entrenó los modelos correctamente. Se reintentará en la próxima ejecución.")
        client.close()
        return

    print(f"  Ciclo: {cycle['cycle_id']}")
    print(f"  Estado: {cycle['state']}")
    print(f"  Cierra: {cycle['closes_at']}")
    print(f"  Predicciones esperadas: {cycle['expected_predictions']}")

    predictions = predict_for_cycle(cycle, obs, ctx, models)
    print(f"  Predicciones generadas: {len(predictions)}")

    # Step 5: Submit
    print("\n[5/5] Enviando predicciones...")
    result = submit_predictions(client, cycle, predictions)

    client.close()
    print("\n✅ Pipeline completado.")


if __name__ == "__main__":
    main()
