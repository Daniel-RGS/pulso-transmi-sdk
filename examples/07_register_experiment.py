from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx


EXPERIMENT_VERSION = "hgb-v1"
RUN_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pulso-transmi:{EXPERIMENT_VERSION}:2026-09-01"))
FEATURE_SET_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pulso-transmi:features:{EXPERIMENT_VERSION}"))


def upsert(client: httpx.Client, table: str, rows: list[dict[str, Any]], conflict: str) -> list[dict[str, Any]]:
    response = client.post(
        f"/{table}",
        params={"on_conflict": conflict},
        json=rows,
        headers={"Prefer": "resolution=merge-duplicates,return=representation"},
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    metrics_path = Path("artifacts/model_metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("SUPABASE_URL y SUPABASE_KEY son obligatorias")

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(base_url=f"{url.rstrip('/')}/rest/v1", headers=headers, timeout=60) as client:
        upsert(client, "pipeline_runs", [{
            "run_id": RUN_ID,
            "started_at": metrics["cutoff"],
            "finished_at": metrics["cutoff"],
            "data_cutoff": metrics["cutoff"],
            "git_commit": git_commit,
            "status": "success",
            "retrain_reason": "initial_horizon_model",
        }], "run_id")
        upsert(client, "feature_sets", [{
            "feature_set_id": FEATURE_SET_ID,
            "name": "pulso-transmi-features",
            "version": EXPERIMENT_VERSION,
            "definition": {"features": metrics["feature_names"], "validation_days": 7},
        }], "name,version")

        models_by_name: dict[str, str] = {}
        for horizon, details in metrics["models"].items():
            model_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pulso-transmi:model:{EXPERIMENT_VERSION}:{horizon}"))
            models_by_name[horizon] = model_id
            upsert(client, "models", [{
                "model_id": model_id,
                "model_name": f"pulso-transmi-{horizon}",
                "algorithm": "HistGradientBoostingRegressor",
                "version": EXPERIMENT_VERSION,
                "artifact_uri": details["artifact"],
                "hyperparameters": {"random_state": 42, "max_iter": 300, "max_leaf_nodes": 31},
                "is_active": horizon == "h15",
            }], "model_name,version")

        metric_rows = []
        for horizon, details in metrics["models"].items():
            metric_rows.append({
                "metric_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"pulso-transmi:metric:{EXPERIMENT_VERSION}:{horizon}")),
                "run_id": RUN_ID,
                "model_id": models_by_name[horizon],
                "metric_name": f"wape_{horizon}",
                "window_start": metrics["cutoff"],
                "window_end": metrics["cutoff"],
                "metric_value": details["model_wape"],
            })
        upsert(client, "metrics", metric_rows, "metric_id")

    print(f"Experimento registrado: {RUN_ID}")
    print(f"Modelos registrados: {len(models_by_name)}")
    print(f"Métricas registradas: {len(metric_rows)}")


if __name__ == "__main__":
    main()