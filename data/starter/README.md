---
title: Dataset estático inicial de Pulso TransMi
description: Corte de entrenamiento sin futuro para la primera clase del reto MLOps.
updated_at: 2026-09-16
---

# Dataset estático inicial

Este corte permite comenzar exploración, feature engineering y entrenamiento
antes de habilitar la API incremental. Contiene **45 días**, **12 estaciones** y
una observación cada **15 minutos**.

## Tamaño

- 96 periodos diarios por estación;
- 4.320 periodos por estación;
- 51.840 observaciones de demanda;
- 4.320 filas de contexto temporal;
- 6 semanas completas más 3 días de historia.

## Archivos

### `stations.csv`

Catálogo público con ID, nombre, corredor y coordenadas. Los metadatos provienen
del [FeatureServer oficial de TransMilenio](https://gis.transmilenio.gov.co/arcgis/rest/services/Troncal/consulta_estaciones_troncales/FeatureServer/0),
consultado el 16 de septiembre de 2026.

### `observations.csv`

| Campo | Tipo | Descripción |
|---|---|---|
| `observed_at` | timestamp con zona | Inicio del intervalo de 15 minutos |
| `station_id` | texto | ID oficial; conserva ceros iniciales |
| `demand` | entero | Demanda sintética no negativa |

### `context.csv`

Contexto público único por timestamp: lluvia observada y pronosticada,
temperatura observada y pronosticada, e intensidad pública de eventos.

### `metadata.json`

Versión, semilla, rango temporal, conteos y hashes SHA-256 para verificar que
todos los equipos trabajen con el mismo corte.

## Uso rápido

```python
import pandas as pd

observations = pd.read_csv("data/starter/observations.csv", parse_dates=["observed_at"])
context = pd.read_csv("data/starter/context.csv", parse_dates=["observed_at"])
stations = pd.read_csv("data/starter/stations.csv", dtype={"station_id": "string"})

training = observations.merge(context, on="observed_at", validate="many_to_one")
training = training.merge(stations, on="station_id", validate="many_to_one")
```

## Reglas del corte

- La demanda y el contexto son completamente sintéticos.
- Los nombres, corredores y coordenadas son metadatos reales y no implican que
  la demanda represente afluencia histórica real de TransMilenio.
- No se incluye ningún dato de los siete días reservados para competencia.
- No se publican arquetipos, parámetros privados, drift ni media latente.
- La zona horaria es `America/Bogota`.
- No se debe rellenar la serie completa con datos futuros al construir lags o
  validaciones.

## Validación sugerida

Para evaluar localmente, usa backtesting temporal: entrena con el pasado y valida
en bloques posteriores. No uses una partición aleatoria, porque mezcla el futuro
con el pasado y produce métricas artificialmente optimistas.

Un primer ejercicio razonable es reservar los últimos siete días del archivo
como validación y entrenar con los 38 anteriores.
