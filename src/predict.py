"""
Script de Predicción - Pulso TransMi v3
Estrategia: Profile × Drift Ratio con tendencia dampened + Direct Extrapolation blend
"""
import os
import sys
import uuid
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

from httpx import Client, HTTPTransport

def get_client():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "pulso-transmi-python/0.1.0",
    }
    transport = HTTPTransport(retries=3)
    return Client(base_url=API_URL, headers=headers, timeout=60, transport=transport)

# ── DATA ────────────────────────────────────────────────────────────────────
def download_starter_data(client):
    """Download the starter CSV files (observations, context, stations)."""
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    cursor = None
    try:
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
            print(f"  Stream actualizado: {len(df)} filas.")
    except Exception as e:
        print(f"⚠️ Alerta: Falló la descarga de stream observations ({e}).")
        print("Continuaremos usando los datos base para asegurar la entrega del ciclo.")
    return all_rows

def load_all_observations():
    obs = pd.read_csv(DATA_DIR / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"])
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    stream_path = DATA_DIR / "stream_observations.csv"
    if stream_path.exists():
        stream = pd.read_csv(stream_path, dtype={"station_id": "string"})
        stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
        if "released_at" in stream.columns: stream = stream.drop(columns=["released_at"])
        
        # COMPATIBILIDAD CON SCHEMA V2.0
        if "measurement" in stream.columns:
            import ast
            def parse_demand(row):
                if pd.notna(row.get("demand")): return float(row["demand"])
                if pd.isna(row.get("measurement")): return None
                try:
                    m = ast.literal_eval(str(row["measurement"]))
                    return float(m.get("value")) if m.get("value") is not None else None
                except: return None
            stream["demand"] = stream.apply(parse_demand, axis=1)
            
        obs = pd.concat([obs, stream], ignore_index=True)
        obs = obs.drop_duplicates(subset=["observed_at", "station_id"]).sort_values(["station_id", "observed_at"])
    return obs

# ── PROFILE BUILDING ────────────────────────────────────────────────────────
def build_clean_profile():
    """
    Build the baseline demand profile from observations.csv (pre-drift data).
    Uses median demand per (station_id, dow, time_slot) for robustness.
    """
    obs_clean = pd.read_csv(DATA_DIR / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"])
    obs_clean["observed_at"] = pd.to_datetime(obs_clean["observed_at"], utc=True)
    obs_clean["dow"] = obs_clean["observed_at"].dt.dayofweek
    obs_clean["time_slot"] = obs_clean["observed_at"].dt.hour * 4 + obs_clean["observed_at"].dt.minute // 15
    
    profile = obs_clean.groupby(["station_id", "dow", "time_slot"])["demand"].median().reset_index()
    profile.columns = ["station_id", "dow", "time_slot", "profile_demand"]
    return profile

def get_profile_value(profile, station_id, dow, time_slot):
    """Get the profile demand for a specific station/dow/time_slot."""
    p = profile[
        (profile["station_id"] == station_id) & 
        (profile["dow"] == dow) & 
        (profile["time_slot"] == time_slot)
    ]
    return float(p["profile_demand"].values[0]) if not p.empty and p["profile_demand"].values[0] > 0 else None

# ── STATION-SPECIFIC TUNED PARAMETERS (from grid search on historical data) ──
# Format: station_id -> (w_static, dampening, w_direct)
# w_static: weight of static drift vs trend drift (higher = more conservative)
# dampening: how much to slow down the log-trend extrapolation
# w_direct: weight of direct demand extrapolation vs profile-based prediction
STATION_PARAMS = {
    "02300": (0.3, 0.8, 0.5),   # Calle 100 - moderate trend, high direct blend
    "03000": (0.3, 1.0, 0.2),   # Portal Suba - follow full trend, mostly profile
    "05000": (0.7, 0.5, 0.2),   # Portal Américas - stable, conservative
    "05100": (0.7, 0.5, 0.2),   # Banderas - stable, conservative (outlier station)
    "06000": (0.4, 0.5, 0.5),   # Portal El Dorado - aggressive dampening, high direct
    "06111": (0.3, 1.0, 0.2),   # Universidades - follow full trend
    "07105": (0.7, 0.5, 0.2),   # Movistar Arena - stable, conservative
    "07107": (0.7, 0.5, 0.5),   # U. Nacional - stable with direct blend
    "07111": (0.5, 0.5, 0.5),   # Ricaurte NQS - balanced, aggressive dampening
    "09000": (0.3, 1.0, 0.5),   # Portal Usme - full trend + direct blend
    "09122": (0.7, 1.0, 0.5),   # Calle 72 - conservative trend + direct blend
    "10009": (0.7, 0.5, 0.2),   # Museo Nacional - conservative (outlier station)
}
DEFAULT_PARAMS = (0.4, 0.7, 0.3)  # fallback for unknown stations


# ── PREDICTION ─────────────────────────────────────────────────────────────
def predict_for_station_target(station_id, target_at, all_obs, profile, cutoff, n=12):
    """
    Smart hybrid prediction with per-station tuned parameters:
    1. Profile × drift ratio with log-trend extrapolation (station-tuned dampening)
    2. Direct demand extrapolation as safety net
    3. Weighted blend using station-specific weights from grid search
    """
    st_data = all_obs[
        (all_obs["station_id"] == station_id) & 
        (all_obs["observed_at"] <= cutoff)
    ].tail(n).copy()
    
    if st_data.empty:
        return 250.0
    
    # Load per-station parameters
    w_static, dampening, w_direct = STATION_PARAMS.get(station_id, DEFAULT_PARAMS)

    target_dow = target_at.dayofweek
    target_slot = target_at.hour * 4 + target_at.minute // 15
    
    base = get_profile_value(profile, station_id, target_dow, target_slot)
    if base is None:
        base = 250.0
    
    # Compute drift ratios for each recent observation
    st_data_p = st_data.copy()
    st_data_p["dow"] = st_data_p["observed_at"].dt.dayofweek
    st_data_p["time_slot"] = st_data_p["observed_at"].dt.hour * 4 + st_data_p["observed_at"].dt.minute // 15
    merged = st_data_p.merge(profile, on=["station_id", "dow", "time_slot"], how="left")
    merged = merged.dropna(subset=["profile_demand"])
    merged = merged[merged["profile_demand"] > 0].reset_index(drop=True)
    
    # Steps ahead from last observation
    last_time = st_data["observed_at"].iloc[-1]
    steps_ahead = max(1, (target_at - last_time).total_seconds() / 900)
    
    # ── METHOD 1: Profile × drift with per-station tuned dampening ──
    if not merged.empty:
        ratios = merged["demand"].values / merged["profile_demand"].values
        static_drift = float(np.median(ratios[-4:])) if len(ratios) >= 4 else float(np.median(ratios))
        
        if len(ratios) >= 4:
            recent = ratios[-8:]
            log_ratios = np.log(np.maximum(recent, 0.01))
            x = np.arange(len(log_ratios))
            coeffs = np.polyfit(x, log_ratios, 1)
            log_slope = coeffs[0]
            last_log_ratio = log_ratios[-1]
            dampened_slope = log_slope * dampening
            extrapolated_log = last_log_ratio + dampened_slope * steps_ahead
            trend_drift = float(np.exp(np.clip(extrapolated_log, -3, 5)))
        else:
            trend_drift = static_drift
        
        drift = w_static * static_drift + (1 - w_static) * trend_drift
        drift = np.clip(drift, 0.01, 100.0)
    else:
        drift = 1.0
    
    pred_profile = max(1.0, base * drift)
    
    # ── METHOD 2: Direct demand extrapolation ──
    demands = st_data["demand"].values
    last_demand = float(demands[-1])
    
    if len(demands) >= 4:
        recent_d = demands[-8:]
        x = np.arange(len(recent_d))
        d_coeffs = np.polyfit(x, recent_d, 1)
        d_slope = d_coeffs[0]
        pred_direct = last_demand + d_slope * steps_ahead * 0.7
        pred_direct = max(1.0, float(pred_direct))
    else:
        pred_direct = max(1.0, last_demand)
    
    # ── BLEND using station-tuned w_direct ──
    final = (1 - w_direct) * pred_profile + w_direct * pred_direct
    return max(1.0, round(final, 2))



def predict_for_cycle(cycle, all_obs):
    targets = cycle["targets"]
    cutoff = pd.Timestamp(cycle["data_cutoff"]).tz_convert("UTC")
    profile = build_clean_profile()
    
    predictions = []
    drift_info = {}
    
    for target in targets:
        station_id = target["station_id"]
        target_at = pd.Timestamp(target["target_at"]).tz_convert("UTC")
        
        pred_value = predict_for_station_target(
            station_id, target_at, all_obs, profile, cutoff
        )
        
        predictions.append({
            "station_id": station_id,
            "target_at": target["target_at"],
            "value": pred_value
        })
    
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

    print(f"Ciclo encontrado: {cycle['cycle_id']}")
    download_starter_data(client)
    download_stream_data(client)
    obs = load_all_observations()
    
    predictions = predict_for_cycle(cycle, obs)
    
    # Print summary
    for p in predictions[:4]:
        print(f"  {p['station_id']} @ {p['target_at']}: {p['value']}")
    if len(predictions) > 4:
        print(f"  ... ({len(predictions)} predicciones total)")
    
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"run-{uuid.uuid4().hex[:8]}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {"version": "profile-drift-v3"},
        "predictions": predictions,
    }

    submit_resp = client.post("/v1/submissions", json=payload, headers={"Idempotency-Key": uuid.uuid4().hex})
    if submit_resp.status_code == 201:
        print(f"✅ Submission exitosa: {submit_resp.json()['submission_id']}")
        
        # Guardar predicciones localmente
        preds_df = pd.DataFrame(predictions)
        preds_df["cycle_id"] = cycle["cycle_id"]
        preds_df["predicted_at"] = datetime.now(timezone.utc).isoformat()
        
        preds_file = ARTIFACTS_DIR / "my_predictions.csv"
        if not preds_file.exists():
            preds_df.to_csv(preds_file, index=False)
        else:
            preds_df.to_csv(preds_file, mode="a", header=False, index=False)
        print(f"✅ Predicciones guardadas en {preds_file}")
    elif submit_resp.status_code == 409:
        # Ya enviamos para este ciclo (idempotency) - salida limpia
        print(f"✅ Ya existe una submission para este ciclo (409). Nada que hacer.")
    else:
        err_body = submit_resp.json() if submit_resp.text else {}
        err_code = err_body.get("detail", {}).get("code", "") if isinstance(err_body.get("detail"), dict) else ""
        if err_code == "attempt_limit_reached":
            print(f"⚠️ Límite de intentos alcanzado para este ciclo. El profe lo evaluará con la última submission.")
        else:
            print(f"❌ Error al enviar ({submit_resp.status_code}): {submit_resp.text}")

if __name__ == "__main__":
    main()
