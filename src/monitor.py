"""
Monitor de Drift - Pulso TransMi
Evalúa localmente la precisión de nuestras predicciones pasadas vs las observaciones reales.
Si detecta una caída (Drift), dispara automáticamente el reentrenamiento.
"""
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")

# Umbral de precisión para considerar que el modelo se ha degradado
DRIFT_THRESHOLD_ACCURACY = 60.0  # Si baja del 60%, reentrenamos

def calculate_wape(y_true, y_pred):
    """Calcula el Weighted Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum_abs_error = np.sum(np.abs(y_true - y_pred))
    sum_actuals = np.sum(y_true)
    if sum_actuals == 0:
        return 0.0
    return sum_abs_error / sum_actuals

def calculate_accuracy(wape):
    """Convierte WAPE a Accuracy (0-100)"""
    return max(0.0, 100.0 * (1.0 - wape))

def main():
    print("=" * 60)
    print("MONITOR DE DRIFT MLOPS — Daniel Santiago Rincón Gamba")
    print("=" * 60)

    preds_file = ARTIFACTS_DIR / "my_predictions.csv"
    stream_file = DATA_DIR / "stream_observations.csv"

    if not preds_file.exists():
        print("ℹ️ No hay predicciones locales (artifacts/my_predictions.csv) para evaluar.")
        print("El sistema necesita predecir al menos un par de ciclos antes de monitorear.")
        sys.exit(0)
        
    if not stream_file.exists():
        print("ℹ️ No hay datos de stream reales (data/stream_observations.csv) todavía.")
        sys.exit(0)

    # Cargar datos
    preds = pd.read_csv(preds_file)
    preds["target_at"] = pd.to_datetime(preds["target_at"], utc=True)
    
    stream = pd.read_csv(stream_file, dtype={"station_id": "string"})
    stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)

    # Hacer JOIN (Inner join para cruzar predicciones con la realidad)
    # Renombrar observed_at a target_at en stream para facilitar el join
    stream = stream.rename(columns={"observed_at": "target_at"})
    
    # Merge
    merged = pd.merge(preds, stream, on=["station_id", "target_at"], how="inner")

    if merged.empty:
        print("ℹ️ Aún no han llegado las observaciones reales correspondientes a tus predicciones.")
        print("Espera a que el profesor publique los datos reales (stream) de los ciclos que ya predijiste.")
        sys.exit(0)

    # Evaluar desempeño por ciclo
    cycles = merged["cycle_id"].unique()
    print(f"📊 Evaluando {len(cycles)} ciclos completados...\n")

    recent_accuracies = []
    
    for cycle in cycles:
        cycle_data = merged[merged["cycle_id"] == cycle]
        y_true = cycle_data["demand"]
        y_pred = cycle_data["value"]
        
        wape = calculate_wape(y_true, y_pred)
        acc = calculate_accuracy(wape)
        recent_accuracies.append(acc)
        
        print(f"  Ciclo: {cycle} | Evaluadas: {len(cycle_data)} est. | Accuracy: {acc:.2f}% | WAPE: {wape:.4f}")

    # Calcular accuracy promedio reciente (últimos 6 ciclos o los que haya)
    last_n = min(6, len(recent_accuracies))
    recent_acc_avg = np.mean(recent_accuracies[-last_n:])
    
    print("-" * 60)
    print(f"📈 Accuracy Promedio (Últimos {last_n} ciclos): {recent_acc_avg:.2f}%")
    
    # Decisión de Reentrenamiento (Drift Detection)
    if recent_acc_avg < DRIFT_THRESHOLD_ACCURACY:
        print(f"⚠️ ¡ALERTA DE DRIFT! La precisión cayó por debajo del {DRIFT_THRESHOLD_ACCURACY}%.")
        print("🔄 Disparando reentrenamiento automático del modelo campeón...")
        
        # Llamar a train.py
        result = subprocess.run(["python", "-m", "src.train"], capture_output=False)
        
        if result.returncode == 0:
            print("✅ Reentrenamiento completado con éxito. El nuevo modelo está listo.")
        else:
            print("❌ El reentrenamiento falló.")
            sys.exit(1)
    else:
        print(f"✅ El modelo se mantiene saludable (>{DRIFT_THRESHOLD_ACCURACY}%). No es necesario reentrenar.")

if __name__ == "__main__":
    main()
