from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pulso_transmi import PulsoTransmiClient


OUTPUT_DIR = Path("artifacts")


def plot_hourly_demand(observations: pd.DataFrame) -> None:
    hourly = (
        observations.groupby(["station_id", "local_hour"])["demand"]
        .mean()
        .unstack()
    )

    figure, axis = plt.subplots(figsize=(14, 7))
    image = axis.imshow(hourly, aspect="auto", cmap="YlOrRd")
    axis.set_title("Demanda promedio por estación y hora")
    axis.set_xlabel("Hora del día (Bogotá)")
    axis.set_ylabel("Estación")
    axis.set_xticks(range(24), range(24))
    axis.set_yticks(range(len(hourly.index)), hourly.index)
    figure.colorbar(image, ax=axis, label="Demanda promedio")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "demanda_por_hora.png", dpi=150)
    plt.close(figure)


def plot_missing_data(observations: pd.DataFrame) -> None:
    dates = pd.date_range(
        observations["local_date"].min(),
        observations["local_date"].max(),
        freq="D",
    ).date
    counts = (
        observations.groupby(["station_id", "local_date"])["observed_at"]
        .nunique()
        .unstack()
        .reindex(columns=dates, fill_value=0)
        .fillna(0)
    )
    missing = (96 - counts).clip(lower=0)

    figure, axis = plt.subplots(figsize=(14, 7))
    image = axis.imshow(
        missing,
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=max(1, int(missing.to_numpy().max())),
    )
    axis.set_title("Datos faltantes por estación y fecha")
    axis.set_xlabel("Fecha")
    axis.set_ylabel("Estación")
    axis.set_yticks(range(len(missing.index)), missing.index)
    tick_positions = range(0, len(dates), max(1, len(dates) // 8))
    axis.set_xticks(list(tick_positions), [str(dates[i]) for i in tick_positions], rotation=45)
    figure.colorbar(image, ax=axis, label="Intervalos faltantes (de 96)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "datos_faltantes.png", dpi=150)
    plt.close(figure)
    print(f"Intervalos faltantes detectados: {int(missing.to_numpy().sum())}")


def main() -> None:
    with PulsoTransmiClient() as client:
        observations = client.observations_dataframe()

    observations["local_time"] = observations["observed_at"].dt.tz_convert("America/Bogota")
    observations["local_hour"] = observations["local_time"].dt.hour
    observations["local_date"] = observations["local_time"].dt.date
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_hourly_demand(observations)
    plot_missing_data(observations)
    print(f"Gráficos guardados en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()