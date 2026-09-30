"""
Monitor de Drift - Pulso TransMi
Evalúa localmente la precisión de nuestras predicciones pasadas vs las observaciones reales.
Si detecta una caída (Drift), dispara automáticamente el reentrenamiento.
"""
import os
import sys
import subprocess
import httpx

# Umbral de precisión para considerar que el modelo se ha degradado
DRIFT_THRESHOLD_ACCURACY = 50.0  # Si baja del 50%, reentrenamos

def main():
    print("=" * 60)
    print("MONITOR DE DRIFT MLOPS — Daniel Santiago Rincón Gamba")
    print("=" * 60)

    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        print("❌ Error: falta la variable de entorno PULSO_API_KEY")
        sys.exit(1)
        
    client = httpx.Client(base_url="https://pulso-transmi.72-60-245-2.sslip.io", headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    
    print("Obteniendo ranking de las últimas 24 horas...")
    resp = client.get("/v1/leaderboard?window=rolling_24h")
    
    if resp.status_code != 200:
        print(f"❌ Error al consultar leaderboard: {resp.text}")
        sys.exit(1)
        
    data = resp.json()
    my_stats = None
    for student in data.get("data", []):
        if "Daniel Santiago" in student.get("display_name", ""):
            my_stats = student
            break
            
    if not my_stats:
        print("ℹ️ No se encontraron estadísticas para Daniel en el leaderboard (posiblemente aún no hay ciclos calificados en las últimas 24h).")
        sys.exit(0)
        
    acc = my_stats.get("accuracy", 0.0)
    cov = my_stats.get("coverage", 0.0)
    
    print(f"📊 Estadísticas actuales (Rolling 24h):")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  Coverage: {cov:.2%}")
    
    # Decisión de Reentrenamiento (Drift Detection)
    # Solo reentrenamos si la cobertura es aceptable (ej. ya enviamos algunos ciclos) y la precisión es baja.
    if cov > 0.05 and acc < DRIFT_THRESHOLD_ACCURACY:
        print(f"⚠️ ¡ALERTA DE DRIFT! La precisión cayó por debajo del {DRIFT_THRESHOLD_ACCURACY}%.")
        print("🔄 Disparando reentrenamiento automático del modelo campeón...")
        
        result = subprocess.run(["python", "-m", "src.train"], capture_output=False)
        
        if result.returncode == 0:
            print("✅ Reentrenamiento completado con éxito. El nuevo modelo está listo.")
        else:
            print("❌ El reentrenamiento falló.")
            sys.exit(1)
    else:
        print(f"✅ El modelo se mantiene saludable o aún está acumulando cobertura. No es necesario reentrenar.")

if __name__ == "__main__":
    main()
