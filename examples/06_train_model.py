from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor


HORIZONS = {"h15": 1, "h30": 2, "h45": 3, "h60": 4}
LAGS = (1, 4, 96, 672)


def add_features(observations: pd.DataFrame, context: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    frame = observations.merge(context, on="observed_at", how="left", validate="many_to_one")
    frame = frame.merge(
        stations[["station_id", "station_name", "corridor", "latitude", "longitude"]],
        on="station_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["station_id", "observed_at"])
    grouped = frame.groupby("station_id", sort=False)["demand"]

    frame["local_hour"] = frame["observed_at"].dt.tz_convert("America/Bogota").dt.hour
    frame["day_of_week"] = frame["observed_at"].dt.tz_convert("America/Bogota").dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["local_hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["local_hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    for lag in LAGS:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    frame["rolling_4"] = shifted.groupby(frame["station_id"]).rolling(4).mean().reset_index(level=0, drop=True)
    frame["rolling_96"] = shifted.groupby(frame["station_id"]).rolling(96).mean().reset_index(level=0, drop=True)
    frame["event_strong"] = (frame["event_intensity"].fillna(0) > 0.1).astype(int)
    return frame


def wape(actual: pd.Series, prediction: pd.Series) -> float:
    return float((actual - prediction).abs().sum() / actual.sum())


def main() -> None:
    data = Path("data")
    artifact_dir = Path("artifacts")
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(data / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"])
    context = pd.read_csv(data / "context.csv", parse_dates=["observed_at"])
    stations = pd.read_csv(data / "stations.csv", dtype={"station_id": "string"})
    frame = add_features(observations, context, stations)
    cutoff = frame["observed_at"].max() - pd.Timedelta(days=7)
    categorical = pd.get_dummies(frame[["station_id", "corridor"]].astype("string"), dtype=float)
    numeric_names = [
        "local_hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "latitude", "longitude", "rain_forecast", "temperature_forecast", "event_strong",
        "rolling_4", "rolling_96", *(f"lag_{lag}" for lag in LAGS),
    ]
    features = pd.concat([frame[numeric_names], categorical], axis=1)
    print("Resultados de validación temporal (últimos 7 días):")
    metrics: dict[str, object] = {
        "cutoff": cutoff.isoformat(),
        "feature_names": features.columns.tolist(),
        "models": {},
    }

    for name, horizon in HORIZONS.items():
        target = frame.groupby("station_id")["demand"].shift(-horizon)
        baseline = frame.groupby("station_id")["demand"].shift(672 - horizon)
        valid = features.notna().all(axis=1) & target.notna()
        train = valid & (frame["observed_at"] < cutoff)
        test = valid & (frame["observed_at"] >= cutoff)
        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(features.loc[train], target.loc[train])
        prediction = pd.Series(model.predict(features.loc[test]), index=frame.index[test])
        actual = target.loc[test]
        model_wape = wape(actual, prediction)
        baseline_wape = wape(actual, baseline.loc[test])
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        metrics["models"][name] = {
            "horizon_minutes": horizon * 15,
            "model_wape": model_wape,
            "weekly_baseline_wape": baseline_wape,
            "improvement": baseline_wape - model_wape,
            "artifact": str(model_path),
        }
        print(f"{name}: modelo={model_wape:.4f} baseline_semanal={baseline_wape:.4f} mejora={(baseline_wape - model_wape):.4f}")

    metrics_path = artifact_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Artefactos guardados en {artifact_dir}")


if __name__ == "__main__":
    main()