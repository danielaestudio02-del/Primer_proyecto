# Backtesting temporal de modelos — Pulso TransMi v1

## Protocolo

- Origen de los datos: `local`.
- 3 ventanas semanales consecutivas. Cada fold entrena solo con targets anteriores a su corte y valida en la semana siguiente.
- Las features usan estación, horizonte, calendario, rezagos y promedios calculados hasta el origen. No se incluyen variables futuras de contexto. Prophet se entrena por fold solo con observaciones anteriores al corte.
- Accuracy usa la métrica oficial: WAPE calculado por estación y luego promediado sin ponderar.
- Todos los resultados usan el starter sintético; no son puntajes oficiales de competencia.

| Fold | Inicio validación (Bogotá) | Fin validación (Bogotá)   | Filas train | Filas validación |
| ---- | -------------------------- | ------------------------- | ----------- | ---------------- |
| 1    | 2026-08-19 00:00:00-05:00  | 2026-08-25 23:45:00-05:00 | 19,572      | 8,052            |
| 2    | 2026-08-26 00:00:00-05:00  | 2026-09-01 23:45:00-05:00 | 27,636      | 8,052            |
| 3    | 2026-09-02 00:00:00-05:00  | 2026-09-08 23:45:00-05:00 | 35,700      | 8,052            |

## Estabilidad agregada

La desviación estándar y el mínimo entre folds ayudan a ver si un modelo depende demasiado de una sola semana. Menor desviación y mayor peor-fold suelen ser señales de mayor estabilidad; se deben considerar junto con la media.

| model                       | Accuracy media (%) | Desv. estándar (pp) | Peor fold (%) | Mejor fold (%) | WAPE medio | MAE medio |
| --------------------------- | ------------------ | ------------------- | ------------- | -------------- | ---------- | --------- |
| Ensamble Prophet + LightGBM | 87.884             | 0.285               | 87.581        | 88.147         | 0.121      | 42.973    |
| Random Forest               | 85.165             | 0.384               | 84.934        | 85.608         | 0.148      | 51.898    |
| XGBoost                     | 84.987             | 0.391               | 84.597        | 85.379         | 0.15       | 51.979    |
| Naive estacional semanal    | 83.249             | 0.313               | 83.026        | 83.606         | 0.168      | 59.387    |
| Naive estacional diario     | 77.503             | 0.587               | 76.826        | 77.886         | 0.225      | 76.382    |
| Persistencia                | 74.434             | 0.162               | 74.255        | 74.572         | 0.256      | 90.737    |

## Accuracy por ventana

| Modelo                      | Fold 1 (%) | Fold 2 (%) | Fold 3 (%) |
| --------------------------- | ---------- | ---------- | ---------- |
| Ensamble Prophet + LightGBM | 87.93      | 87.58      | 88.15      |
| Naive estacional diario     | 77.8       | 76.83      | 77.89      |
| Naive estacional semanal    | 83.61      | 83.03      | 83.11      |
| Persistencia                | 74.26      | 74.47      | 74.57      |
| Random Forest               | 84.93      | 84.95      | 85.61      |
| XGBoost                     | 84.99      | 84.6       | 85.38      |

![Accuracy promedio por modelo](figures/backtest_modelos_accuracy.png)

## Métodos comparados

- **Persistencia:** predice que la demanda seguirá igual que en el último intervalo conocido (15 minutos antes del origen).
- **Naive estacional diario:** repite el valor de la misma estación y franja horaria del día anterior.
- **Naive estacional semanal:** repite el valor de la misma estación, hora y día de la semana anterior.
- **Random Forest:** combina rezagos, medias móviles, calendario, estación y horizonte con muchos árboles entrenados sobre muestras/features aleatorias.
- **XGBoost:** combina árboles construidos secuencialmente; cada árbol intenta corregir errores de los anteriores.
- **Ensamble Prophet + LightGBM (champion de producción):** 65% Prophet por estación (perfil diario/semanal promediado de muchas semanas, escalado por la razón demanda real/esperada de las últimas 2 horas) + 35% LightGBM global con rezagos, pendientes, valores de la misma franja ayer y la semana pasada y perfiles promediados de 4 semanas. Es el mismo código (`analysis/ensemble_model.py`) que se entrena y empaqueta en el joblib del pipeline.

Los tres primeros son baselines interpretables. Indican qué tan difícil es la serie y evitan atribuir valor a un modelo complejo que no supere reglas sencillas.
