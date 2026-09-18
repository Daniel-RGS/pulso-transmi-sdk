from __future__ import annotations

import argparse
import importlib.util
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import joblib
import pandas as pd

_TRAINING_MODULE = importlib.util.spec_from_file_location(
    "training_module", Path(__file__).with_name("06_train_model.py")
)
if _TRAINING_MODULE is None or _TRAINING_MODULE.loader is None:
    raise ImportError("No se pudo cargar el módulo de entrenamiento")
_training = importlib.util.module_from_spec(_TRAINING_MODULE)
_TRAINING_MODULE.loader.exec_module(_training)
HORIZONS = _training.HORIZONS
add_features = _training.add_features


EXPERIMENT_VERSION = "hgb-v1"
FEATURE_SET_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "pulso-transmi:features:hgb-v1"))


def build_rows() -> list[dict[str, Any]]:
    data = Path("data")
    metrics = json.loads((Path("artifacts") / "model_metrics.json").read_text(encoding="utf-8"))
    run_key = f"pulso-transmi:{EXPERIMENT_VERSION}:{metrics['dataset_version']}:{metrics['dataset_hash']}"
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, run_key))
    observations = pd.read_csv(data / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"])
    context = pd.read_csv(data / "context.csv", parse_dates=["observed_at"])
    stations = pd.read_csv(data / "stations.csv", dtype={"station_id": "string"})
    frame = add_features(observations, context, stations)
    categorical = pd.get_dummies(frame[["station_id", "corridor"]].astype("string"), dtype=float)
    numeric_names = [
        "local_hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "latitude", "longitude", "rain_forecast", "temperature_forecast", "event_strong",
        "rolling_4", "rolling_96", "lag_1", "lag_4", "lag_96", "lag_672",
    ]
    features = pd.concat([frame[numeric_names], categorical], axis=1).reindex(columns=metrics["feature_names"], fill_value=0)
    latest = frame.groupby("station_id", sort=False)["observed_at"].idxmax()
    latest_indices = latest.to_numpy()
    valid_latest = features.loc[latest_indices].notna().all(axis=1).to_numpy()
    latest = latest.iloc[valid_latest]
    generated_at = frame.loc[latest, "observed_at"]
    rows: list[dict[str, Any]] = []
    for name, horizon in HORIZONS.items():
        model = joblib.load(Path("artifacts/models") / f"{name}.joblib")
        predictions = model.predict(features.loc[latest]).clip(min=0)
        for index, prediction in zip(latest, predictions):
            generated_timestamp = pd.Timestamp(frame.loc[index, "observed_at"])
            target_at = generated_timestamp + pd.Timedelta(value=int(horizon * 15), unit="m")
            station_id = str(frame.loc[index, "station_id"])
            model_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pulso-transmi:model:{EXPERIMENT_VERSION}:{name}"))
            prediction_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pulso-transmi:prediction:{run_id}:{station_id}:{target_at}:{horizon * 15}"))
            rows.append({
                "prediction_id": prediction_id,
                "run_id": run_id,
                "model_id": model_id,
                "feature_set_id": FEATURE_SET_ID,
                "station_id": station_id,
                "generated_at": generated_at.loc[index].isoformat(),
                "target_at": target_at.isoformat(),
                "horizon_minutes": horizon * 15,
                "predicted_demand": float(prediction),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = build_rows()
    if args.dry_run:
        print(f"[dry-run] predictions: {len(rows)} filas")
        print(rows[0])
        return
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("SUPABASE_URL y SUPABASE_KEY son obligatorias")
    response = httpx.post(
        f"{url.rstrip('/')}/rest/v1/predictions",
        params={"on_conflict": "run_id,station_id,target_at,horizon_minutes"},
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        json=rows,
        timeout=60,
    )
    response.raise_for_status()
    print(f"Predicciones registradas: {len(rows)}")


if __name__ == "__main__":
    main()