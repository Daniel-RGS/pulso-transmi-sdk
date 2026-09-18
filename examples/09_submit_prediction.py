from __future__ import annotations

import argparse
import importlib.util
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


_PREDICTION_MODULE = importlib.util.spec_from_file_location(
    "prediction_module", Path(__file__).with_name("08_generate_predictions.py")
)
if _PREDICTION_MODULE is None or _PREDICTION_MODULE.loader is None:
    raise ImportError("No se pudo cargar el generador de predicciones")
_prediction = importlib.util.module_from_spec(_PREDICTION_MODULE)
_PREDICTION_MODULE.loader.exec_module(_prediction)
build_rows = _prediction.build_rows


def build_submission(cycle: dict[str, Any]) -> dict[str, Any]:
    targets = {
        (item["station_id"], pd.Timestamp(item["target_at"]).tz_convert("UTC"))
        for item in cycle["targets"]
    }
    rows = [
        row for row in build_rows()
        if row["horizon_minutes"] == 15
        and (row["station_id"], pd.Timestamp(row["target_at"]).tz_convert("UTC")) in targets
    ]
    rows_by_key = {
        (row["station_id"], pd.Timestamp(row["target_at"]).tz_convert("UTC")): row
        for row in rows
    }
    predictions = [
        {
            "station_id": target["station_id"],
            "target_at": target["target_at"],
            "value": rows_by_key[(target["station_id"], pd.Timestamp(target["target_at"]).tz_convert("UTC"))]["predicted_demand"],
        }
        for target in cycle["targets"]
    ]
    return {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": str(uuid.uuid4()),
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": "hgb-v1",
            "trained_at": "2026-09-18T00:00:00-05:00",
            "training_data_end": "2026-09-08T23:45:00-05:00",
            "git_commit": os.getenv("MODEL_GIT_COMMIT", "1dd5a2c"),
        },
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-url", default=os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io"))
    args = parser.parse_args()

    with httpx.Client(base_url=args.api_url.rstrip("/"), timeout=30) as client:
        cycle_response = client.get("/v1/forecast-cycles/current")
        cycle_response.raise_for_status()
        cycle = cycle_response.json()
        payload = build_submission(cycle)

        if args.dry_run:
            print(f"[dry-run] ciclo: {payload['cycle_id']}")
            print(f"[dry-run] predicciones: {len(payload['predictions'])}")
            print(payload)
            return

        api_key = os.getenv("PULSO_API_KEY")
        if not api_key:
            raise SystemExit("PULSO_API_KEY es obligatoria para enviar la entrega")
        response = client.post(
            "/v1/submissions",
            headers={"Authorization": f"Bearer {api_key}", "Idempotency-Key": payload["client_run_id"]},
            json=payload,
        )
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()