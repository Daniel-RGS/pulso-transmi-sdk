# Modelado

Esta rama concentra el experimento de pronóstico de demanda de Pulso TransMi.

## Flujo reproducible

Con el entorno virtual activo y los datos en `data/`:

```bash
.venv/bin/python examples/05_compare_baselines.py
.venv/bin/python examples/06_train_model.py
set -a && source .env && set +a
.venv/bin/python examples/07_register_experiment.py
.venv/bin/python examples/08_generate_predictions.py
```

El primer comando compara el baseline diario (`shift(96)`) con el semanal
(`shift(672)`). El segundo entrena cuatro modelos
`HistGradientBoostingRegressor`, uno por horizonte: 15, 30, 45 y 60 minutos.

La validacion usa los primeros 38 dias para entrenamiento y los ultimos 7 dias
para evaluacion temporal. Las features se construyen solo con informacion
disponible hasta cada instante: lags, medias moviles desplazadas, calendario,
estacion, geografia y pronosticos de lluvia y temperatura.

## Artefactos

El entrenamiento escribe modelos locales en `artifacts/models/` y el resumen
de metricas en `artifacts/model_metrics.json`. Esa carpeta esta ignorada por
Git. El registro en Supabase conserva la ejecucion, el conjunto de features,
los modelos y sus metricas, pero no sube los binarios locales.

El ultimo comando genera 12 predicciones por horizonte y las guarda en
`predictions`. `actual_demand` queda vacio hasta que la API libere esos periodos;
en ese momento pueden calcularse metricas de desempeno.

## GitHub Actions

El workflow `.github/workflows/model-training.yml` se ejecuta manualmente desde
la pestana **Actions**. Configura estos secrets en el repositorio:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `PULSO_API_KEY`, si la API lo requiere

Tambien puedes definir `PULSO_API_URL` como variable del repositorio. El workflow
no tiene un horario automatico hasta que se confirme la frecuencia oficial del
reto.

## Criterio de aceptacion

Un modelo se conserva si mejora el WAPE promedio del baseline semanal en la
ventana de validacion, manteniendo una evaluacion separada por horizonte y por
estacion cuando se amplie el reporte.