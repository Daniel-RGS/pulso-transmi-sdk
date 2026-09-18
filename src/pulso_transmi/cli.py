from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Any, Iterable

import httpx

from pulso_transmi import PulsoTransmiClient


def _iso(value: Any) -> str:
    timestamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    return str(timestamp)


class SupabaseWriter:
    def __init__(self, url: str, key: str, *, dry_run: bool = False) -> None:
        self.url = url.rstrip("/")
        self.key = key
        self.dry_run = dry_run
        self.client = httpx.Client(
            base_url=f"{self.url}/rest/v1",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            timeout=60.0,
        )

    def close(self) -> None:
        self.client.close()

    def upsert(self, table: str, rows: list[dict[str, Any]], *, batch_size: int = 1000) -> None:
        if not rows:
            return
        if self.dry_run:
            print(f"[dry-run] {table}: {len(rows)} filas")
            return
        for offset in range(0, len(rows), batch_size):
            response = self.client.post(f"/{table}", json=rows[offset : offset + batch_size])
            response.raise_for_status()


def _time_rows(observations: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp in observations["observed_at"].drop_duplicates().sort_values():
        local = timestamp.tz_convert("America/Bogota")
        rows.append(
            {
                "observed_at": _iso(timestamp),
                "local_date": local.date().isoformat(),
                "local_time": local.time().isoformat(),
                "local_hour": int(local.hour),
                "day_of_week": int(local.dayofweek),
                "is_weekend": bool(local.dayofweek >= 5),
                "interval_number": int(local.hour * 4 + local.minute // 15),
            }
        )
    return rows


def _context_rows(context: Any, dataset_id: str) -> list[dict[str, Any]]:
    rows = context.rename(columns={"observed_at": "observed_at"}).copy()
    rows["dataset_id"] = dataset_id
    rows["observed_at"] = rows["observed_at"].map(_iso)
    return rows[
        [
            "observed_at",
            "dataset_id",
            "rain_mm",
            "rain_forecast",
            "temperature_c",
            "temperature_forecast",
            "event_intensity",
        ]
    ].to_dict("records")


def _demand_rows(observations: Any, dataset_id: str) -> list[dict[str, Any]]:
    rows = observations.copy()
    rows["dataset_id"] = dataset_id
    rows["observed_at"] = rows["observed_at"].map(_iso)
    rows["demand"] = rows["demand"].astype(int)
    return rows[["station_id", "observed_at", "dataset_id", "demand"]].to_dict("records")


def ingest(args: argparse.Namespace) -> None:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not args.dry_run and (not supabase_url or not supabase_key):
        raise SystemExit("SUPABASE_URL y SUPABASE_KEY son obligatorias")

    with PulsoTransmiClient(base_url=args.api_url, api_key=os.getenv("PULSO_API_KEY")) as api:
        metadata = api.meta()["dataset"]
        stations = api.stations()
        observations = api.observations_dataframe(
            start=args.start, end=args.end, page_size=args.page_size
        )
        context = api.context_dataframe(
            start=args.start, end=args.end, page_size=args.page_size
        )

    if observations.duplicated(["station_id", "observed_at"]).any():
        raise SystemExit("La API devolvió observaciones duplicadas")
    if observations[["station_id", "observed_at", "demand"]].isna().any().any():
        raise SystemExit("La API devolvió campos obligatorios nulos")

    dataset_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pulso-transmi:{metadata['dataset']}"))
    dataset_payload = {
        "dataset_id": dataset_id,
        "dataset_key": metadata["dataset"],
        "generated_at": metadata.get("generated_at"),
        "timezone": metadata["timezone"],
        "frequency_minutes": metadata["frequency_minutes"],
        "history_start": metadata.get("history_start"),
        "history_end": metadata.get("history_end"),
        "station_count": metadata["station_count"],
        "observation_rows": metadata["observation_rows"],
        "context_rows": metadata["context_rows"],
        "source_hash": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
    }
    station_rows = stations.rename(columns={"name": "station_name"}).to_dict("records")
    for row in station_rows:
        row["station_id"] = str(row["station_id"])

    writer = SupabaseWriter(supabase_url or "", supabase_key or "", dry_run=args.dry_run)
    try:
        writer.upsert("dataset_catalog", [dataset_payload], batch_size=args.batch_size)
        writer.upsert("stations", station_rows, batch_size=args.batch_size)
        writer.upsert("time_intervals", _time_rows(observations), batch_size=args.batch_size)
        writer.upsert("context_observations", _context_rows(context, dataset_id), batch_size=args.batch_size)
        writer.upsert("station_demand", _demand_rows(observations, dataset_id), batch_size=args.batch_size)
    finally:
        writer.close()
    print(f"Carga completada: {len(observations)} demandas, {len(context)} contextos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pulso-transmi")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest", help="Carga datos en Supabase")
    ingest_parser.add_argument("--api-url", default=None)
    ingest_parser.add_argument("--start")
    ingest_parser.add_argument("--end")
    ingest_parser.add_argument("--page-size", type=int, default=5000)
    ingest_parser.add_argument("--batch-size", type=int, default=1000)
    ingest_parser.add_argument("--dry-run", action="store_true")
    ingest_parser.set_defaults(function=ingest)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()