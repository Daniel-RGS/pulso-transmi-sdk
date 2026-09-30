# Politica de experimentos

## Ramas

- `main` contiene solo versiones aprobadas y reproducibles.
- `model-training` es la rama de trabajo para features, modelos y evaluaciones.
- Cada experimento importante debe usar una rama derivada de `model-training` y
  volver mediante Pull Request.

## Versiones que deben viajar juntas

Cada corrida debe identificar:

- `dataset_version`: nombre declarado por la API.
- `dataset_hash`: SHA-256 del corte usado.
- `code_commit`: commit exacto del código.
- `feature_set`: definición y versión de las variables.
- `model_version`: algoritmo y versión del modelo.
- `validation_cutoff`: límite entre entrenamiento y validación.
- `WAPE` y `Accuracy` por horizonte.

Esto evita comparar resultados que usaron datos, código o features diferentes.

## Ciclo de un experimento

```text
crear rama -> cambiar features/modelo -> backtesting temporal
-> guardar artefactos -> registrar métricas en Supabase
-> comparar con baseline -> Pull Request -> promover a main
```

El experimento actual usa `pulso-transmi-starter-v1`, una partición temporal de
38 días de entrenamiento y 7 días de validación, y cuatro horizontes de 15 a
60 minutos. Solo se promueve un modelo si mejora el baseline semanal y deja
trazabilidad suficiente para reproducir el resultado.

## Bono de operación

Supabase funciona como registro operativo: conserva datos, ejecuciones, modelos,
features, métricas y predicciones. Un dashboard puede consultar esas tablas
para mostrar Accuracy por horizonte, última ejecución, modelo activo y evolución
de las predicciones sin mezclar credenciales privadas en el navegador.

El dashboard local se inicia con:

```bash
set -a && source .env && set +a
.venv/bin/streamlit run dashboard.py
```

La clave de Supabase permanece en el proceso servidor de Streamlit; nunca se
envía al navegador.