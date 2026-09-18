from pathlib import Path

import matplotlib.pyplot as plt

from pulso_transmi import PulsoTransmiClient


def main() -> None:
    with PulsoTransmiClient() as client:
        observations = client.observations_dataframe()
    observations = (
        observations.set_index("observed_at")
        .groupby("station_id")["demand"]
        .resample("h")
        .mean()
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(14, 7))
    for station_id, station_data in observations.groupby("station_id"):
        axis.plot(
            station_data["observed_at"],
            station_data["demand"],
            linewidth=1,
            label=station_id,
        )

    axis.set_title("Demanda de TransMilenio por estación")
    axis.set_xlabel("Fecha y hora")
    axis.set_ylabel("Demanda promedio por hora")
    axis.legend(title="Estación", ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()

    output = Path("artifacts/demanda_por_estacion.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"Gráfico guardado en: {output}")


if __name__ == "__main__":
    main()