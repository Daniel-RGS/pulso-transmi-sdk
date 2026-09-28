import os
import uuid
import httpx

API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
API_KEY = os.getenv("PULSO_API_KEY")

def main():
    if not API_KEY:
        print("Falta PULSO_API_KEY")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "pulso-transmi-python/0.1.0"
    }
    
    # 1. Obtener ciclo actual
    with httpx.Client(base_url=API_URL, headers=headers) as client:
        print("Obteniendo ciclo actual...")
        resp = client.get("/v1/forecast-cycles/current")
        resp.raise_for_status()
        cycle = resp.json()
        
        cycle_id = cycle["cycle_id"]
        data_cutoff = cycle["data_cutoff"]
        targets = cycle["targets"]
        
        print(f"Ciclo: {cycle_id}")
        
        # 2. Generar predicciones tontas (baseline)
        predictions = []
        for t in targets:
            predictions.append({
                "station_id": t["station_id"],
                "target_at": t["target_at"],
                "value": 250.0  # predicción arbitraria para empezar a puntuar
            })
            
        payload = {
            "schema_version": "1.0",
            "cycle_id": cycle_id,
            "client_run_id": f"run-{uuid.uuid4().hex[:8]}",
            "data_cutoff": data_cutoff,
            "model": {
                "version": "baseline-v1"
            },
            "predictions": predictions
        }
        
        # 3. Enviar predicciones
        print(f"Enviando {len(predictions)} predicciones...")
        idem_key = uuid.uuid4().hex
        submit_headers = headers.copy()
        submit_headers["Idempotency-Key"] = idem_key
        
        submit_resp = client.post("/v1/submissions", json=payload, headers=submit_headers)
        
        if submit_resp.status_code == 201:
            print("¡Predicciones enviadas exitosamente!")
            print(submit_resp.json())
        else:
            print(f"Error al enviar: {submit_resp.status_code}")
            print(submit_resp.text)

if __name__ == "__main__":
    main()
