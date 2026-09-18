from __future__ import annotations

from pathlib import Path

import pandas as pd


BASELINES = {
    "daily_96": 96,
    "weekly_672": 672,
}


def wape_by_station(frame: pd.DataFrame) -> pd.Series:
    errors = (frame["demand"] - frame["prediction"]).abs()
    return errors.groupby(frame["station_id"]).sum() / frame["demand"].groupby(frame["station_id"]).sum()


def evaluate(observations: pd.DataFrame, lag: int) -> pd.Series:
    scored = observations.copy()
    scored["prediction"] = scored.groupby("station_id")["demand"].shift(lag)
    cutoff = scored["observed_at"].max() - pd.Timedelta(days=7)
    validation = scored.loc[scored["observed_at"] > cutoff].dropna(subset=["prediction"])
    return wape_by_station(validation)


def main() -> None:
    observations = (
        pd.read_csv(
            Path("data/observations.csv"),
            dtype={"station_id": "string"},
            parse_dates=["observed_at"],
        )
        .sort_values(["station_id", "observed_at"])
    )

    results = pd.DataFrame(
        {name: evaluate(observations, lag) for name, lag in BASELINES.items()}
    ).sort_index()
    results["winner"] = results.idxmin(axis=1)
    print("WAPE por estación (menor es mejor):")
    print(results.round(4).to_string())
    print("\nWAPE promedio por estación:")
    print(results[list(BASELINES)].mean().round(4).to_string())


if __name__ == "__main__":
    main()