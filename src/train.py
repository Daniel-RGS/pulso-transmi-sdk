"""
Script de Entrenamiento - Pulso TransMi
Descarga los datos históricos y recientes, calcula features y entrena/re-entrena los modelos.
"""
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
API_KEY = os.getenv("PULSO_API_KEY")
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")
MODEL_VERSION = "hgbr-adaptive-v2"

def get_client():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "pulso-transmi-python/0.1.0",
    }
    return httpx.Client(base_url=API_URL, headers=headers, timeout=60)

# ── DATA COLLECTION ─────────────────────────────────────────────────────
def download_starter_data(client):
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
    obs = pd.read_csv(DATA_DIR / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"])
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    stream_path = DATA_DIR / "stream_observations.csv"
    if stream_path.exists():
        stream = pd.read_csv(stream_path, dtype={"station_id": "string"})
        stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
        if "released_at" in stream.columns:
            stream = stream.drop(columns=["released_at"])
        obs = pd.concat([obs, stream], ignore_index=True)
        obs = obs.drop_duplicates(subset=["observed_at", "station_id"]).sort_values(["station_id", "observed_at"])
    return obs

def load_context():
    ctx = pd.read_csv(DATA_DIR / "context.csv", parse_dates=["observed_at"])
    ctx["observed_at"] = pd.to_datetime(ctx["observed_at"], utc=True)
    return ctx

# ── FEATURE ENGINEERING ─────────────────────────────────────────────────
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
    df["adaptive_profile"] = df.groupby("profile_key")["demand"].transform(lambda x: x.shift(1).ewm(halflife=48, min_periods=1).mean())
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
    "adaptive_profile", "temperature_c", "rain_mm", "event_intensity",
]

# ── TRAINING ────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    for station_id, group in df.groupby("station_id"):
        train_data = group.dropna(subset=["demand"] + available_features)
        if len(train_data) < 50:
            print(f"  Estación {station_id}: pocos datos ({len(train_data)}), skip")
            continue
        # Solo usar el último 40% de datos para priorizar datos post-Drift
        cutoff = max(100, int(len(train_data) * 0.4))
        train_data = train_data.tail(cutoff)
        
        # --- ESTRATEGIA ALLISON (NORMALIZACIÓN GLOBAL) ---
        # En vez de predecir demanda absoluta, predecimos el % respecto al promedio reciente (rolling_mean_96)
        s = train_data["rolling_mean_96"].fillna(1.0)
        s = np.maximum(s, 1.0).values
        
        X = train_data[available_features].values
        y_raw = train_data["demand"].values
        y_norm = y_raw / s
        
        # Para mantener la métrica WAPE, pesamos por la escala 's' y le damos más peso a lo reciente
        time_weights = np.linspace(0.5, 1.0, len(y_norm))
        final_weights = time_weights * s
        
        # --- ESTRATEGIA KEVIN/ALEJANDRO (ENSAMBLE VOTING) ---
        model_hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=7, learning_rate=0.06, random_state=42)
        model_xgb = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.06, random_state=42, tree_method="hist")
        
        model = VotingRegressor(estimators=[
            ("hgb", model_hgb),
            ("xgb", model_xgb)
        ])
        
        model.fit(X, y_norm, sample_weight=final_weights)
        models[station_id] = model
        print(f"  Estación {station_id}: entrenado con {len(train_data)} filas")
    
    model_path = ARTIFACTS_DIR / "models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "features": available_features, "version": MODEL_VERSION, "trained_at": datetime.now(timezone.utc).isoformat()}, f)
    print(f"  ✅ Modelos guardados en {model_path}")
    return models

# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        print("ERROR: Falta PULSO_API_KEY")
        sys.exit(1)
    print("=" * 60)
    print("ENTRENAMIENTO PULSO TRANSMI — FASE DRIFT")
    print("=" * 60)
    client = get_client()
    print("\n[1/3] Descargando datos recientes...")
    download_starter_data(client)
    download_stream_data(client)
    print("\n[2/3] Cargando y preparando dataset...")
    obs = load_all_observations()
    ctx = load_context()
    featured = build_features(obs, ctx)
    print("\n[3/3] Entrenando nuevos modelos campeón...")
    train_models(featured)
    client.close()

if __name__ == "__main__":
    main()
