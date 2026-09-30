"""
Script de Predicción - Pulso TransMi
Verifica ciclos, descarga datos recientes, usa el modelo guardado y envía inferencia.
"""
import os
import sys
import uuid
import pickle
from pathlib import Path
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
API_KEY = os.getenv("PULSO_API_KEY")
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")

def get_client():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "pulso-transmi-python/0.1.0",
    }
    return httpx.Client(base_url=API_URL, headers=headers, timeout=60)

# ── DATA & FEATURES (Requeridos para Lags) ───────────────────────────────
def download_stream_data(client):
    all_rows = []
    cursor = None
    while True:
        params = {"limit": 5000}
        if cursor: params["cursor"] = cursor
        resp = client.get("/v1/stream/observations", params=params)
        resp.raise_for_status()
        data = resp.json()
        rows = data["data"]
        all_rows.extend(rows)
        cursor = data.get("next_cursor")
        if not cursor: break
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(DATA_DIR / "stream_observations.csv", index=False)
    return all_rows

def load_all_observations():
    obs = pd.read_csv(DATA_DIR / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"])
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    stream_path = DATA_DIR / "stream_observations.csv"
    if stream_path.exists():
        stream = pd.read_csv(stream_path, dtype={"station_id": "string"})
        stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
        if "released_at" in stream.columns: stream = stream.drop(columns=["released_at"])
        obs = pd.concat([obs, stream], ignore_index=True)
        obs = obs.drop_duplicates(subset=["observed_at", "station_id"]).sort_values(["station_id", "observed_at"])
    return obs

def load_context():
    ctx = pd.read_csv(DATA_DIR / "context.csv", parse_dates=["observed_at"])
    ctx["observed_at"] = pd.to_datetime(ctx["observed_at"], utc=True)
    return ctx

def build_features(obs: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    df = obs.copy()
    df["hour"] = df["observed_at"].dt.hour
    df["minute"] = df["observed_at"].dt.minute
    df["dow"] = df["observed_at"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df = df.sort_values(["station_id", "observed_at"])
    for lag in [1, 2, 3, 4, 96, 672]:
        df[f"lag_{lag}"] = df.groupby("station_id")["demand"].shift(lag)
    for window in [4, 12, 96]:
        df[f"rolling_mean_{window}"] = df.groupby("station_id")["demand"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df["profile_key"] = df["station_id"] + "_" + df["dow"].astype(str) + "_" + df["time_slot"].astype(str)
    df["adaptive_profile"] = df.groupby("profile_key")["demand"].transform(lambda x: x.shift(1).ewm(halflife=1344, min_periods=1).mean())
    ctx_cols = ["observed_at", "temperature_c", "rain_mm", "event_intensity"]
    available_cols = [c for c in ctx_cols if c in ctx.columns]
    if len(available_cols) > 1: df = df.merge(ctx[available_cols], on="observed_at", how="left")
    return df

# ── PREDICTION ─────────────────────────────────────────────────────────────
def predict_for_cycle(cycle: dict, obs: pd.DataFrame, ctx: pd.DataFrame, model_data: dict) -> list:
    targets = cycle["targets"]
    models = model_data["models"]
    feat_cols = model_data["features"]
    
    featured = build_features(obs, ctx)
    predictions = []
    
    for target in targets:
        station_id = target["station_id"]
        target_at = pd.Timestamp(target["target_at"]).tz_convert("UTC")
        if station_id not in models:
            predictions.append({"station_id": station_id, "target_at": target["target_at"], "value": 250.0})
            continue

        model = models[station_id]
        station_hist = featured[featured["station_id"] == station_id].copy()
        if station_hist.empty:
            predictions.append({"station_id": station_id, "target_at": target["target_at"], "value": 250.0})
            continue

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

        same_slot = station_hist[(station_hist["dow"] == dow) & (station_hist["time_slot"] == hour * 4 + minute // 15)]
        if not same_slot.empty:
            last_row["adaptive_profile"] = same_slot["adaptive_profile"].iloc[-1]
            last_row["lag_672"] = same_slot["demand"].iloc[-1]

        feat_values = [float(last_row.get(col, 0) if pd.notna(last_row.get(col, 0)) else 0) for col in feat_cols]
        pred_value = max(0, model.predict(np.array([feat_values]))[0])
        predictions.append({"station_id": station_id, "target_at": target["target_at"], "value": round(pred_value, 2)})

    return predictions

# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        sys.exit("ERROR: Falta PULSO_API_KEY")

    client = get_client()
    resp = client.get("/v1/forecast-cycles/current")
    if resp.status_code == 404:
        print("✅ No hay ciclo abierto (HTTP 404). El run termina en verde.")
        sys.exit(0)
    
    resp.raise_for_status()
    cycle = resp.json()
    if cycle.get("state") != "open":
        print(f"✅ Ciclo {cycle.get('cycle_id')} no está abierto.")
        sys.exit(0)

    model_path = ARTIFACTS_DIR / "models.pkl"
    if not model_path.exists():
        sys.exit("❌ Error: No se encontró artifacts/models.pkl. ¡Falta reentrenar!")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    print(f"Ciclo encontrado: {cycle['cycle_id']}")
    download_stream_data(client)
    obs = load_all_observations()
    ctx = load_context()
    
    predictions = predict_for_cycle(cycle, obs, ctx, model_data)
    
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"run-{uuid.uuid4().hex[:8]}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {"version": model_data.get("version", "unknown")},
        "predictions": predictions,
    }

    submit_resp = client.post("/v1/submissions", json=payload, headers={"Idempotency-Key": uuid.uuid4().hex})
    if submit_resp.status_code == 201:
        print(f"✅ Submission exitosa: {submit_resp.json()['submission_id']}")
    else:
        print(f"❌ Error al enviar: {submit_resp.text}")

if __name__ == "__main__":
    main()
