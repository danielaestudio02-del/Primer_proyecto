# Operación de sincronización, modelo y envíos

## Flujo

- `pulso-hourly.yml` se despierta cada 10 minutos. Sincroniza el stream incremental, consulta `/v1/forecast-cycles/current` y envía solo si hay un ciclo abierto, no existe ya una entrega oficial registrada, el modelo está vigente al cutoff y están disponibles los datos recientes.
- `train-random-forest.yml` (**Train and promote forecast model**) corre todos los días a las 00:07 de Bogotá y también se puede lanzar a mano. Sincroniza, entrena el ensamble Prophet + LightGBM (`analysis/ensemble_model.py`), lo valida en los últimos 7 días contra el naive semanal y contra un Random Forest entrenado con los mismos datos, y solo promueve/sube el artefacto `.joblib` si les gana a ambos.
- El `.joblib` contiene el objeto `PulsoEnsemble` completo: los 12 modelos Prophet (serializados en JSON), el LightGBM, los pesos (65/35), la ventana del ajuste de nivel (2 h), las features y las métricas de validación.
- En cada pronóstico, la parte de Prophet se escala con la razón demanda real / esperada de las últimas 2 horas. Así el modelo se adapta en pocas horas a un cambio de nivel sin reentrenar.
- Después de registrar cada entrega se revisa el drift: si en una estación la razón real/esperada queda fuera de 0,85–1,15, del mismo lado, en las tres últimas ventanas de 4 horas (12 h), se guarda un evento `data_drift` en `monitoring_events` y, si el champion tiene más de 6 horas, se reentrena y se registra la decisión (`retraining_decision`). El monitoreo nunca bloquea ni retrasa la entrega.
- Los champions Random Forest anteriores (artefactos con `pipeline`) siguen funcionando para pronosticar hasta que el primer entrenamiento del ensamble los reemplace.
- El cursor de stream se confirma en Supabase después de guardar cada página. Repetir una página es seguro por el upsert `(station_id, observed_at)`.
- Una entrega conserva su recibo, targets, predicciones, versión de modelo y hash del batch.

## Preparación única en Supabase y GitHub

1. En Supabase, abre **Storage** y crea un bucket privado llamado `model-artifacts`.
2. En el repositorio `Primer_proyecto`, abre **Settings → Secrets and variables → Actions** y crea estos repository secrets:
   - `PULSO_API_KEY`: la llave del portal Pulso TransMi.
   - `SUPABASE_URL`: Project URL.
   - `SUPABASE_SECRET_KEY`: Secret key de backend.
3. Comprueba en GitHub que aparecen los dos workflows en **Actions**.
4. Ejecuta primero **Train and promote forecast model → Run workflow**. Debe registrar la versión champion (`ens-...`) y subir el artefacto al bucket.
5. Ejecuta **Pulso TransMi — sync and forecast → Run workflow** para una primera ejecución. Si no hay ciclo abierto, sincronizará y finalizará sin entregar.

No guardes esos secretos en el repositorio, en la tabla de Supabase ni en archivos de workflow. El workflow no imprime sus valores.

## Prueba local

Con las variables ya guardadas en `.env.local`:

```powershell
python -m pip install -r requirements-pipeline.txt
python analysis/pulso_pipeline.py sync
python analysis/pulso_pipeline.py train
python analysis/pulso_pipeline.py forecast
```

`forecast` es simulación por defecto y no envía. Solo `forecast --send` realiza una submission. El workflow de GitHub Actions pasa esa opción después de recibir los tres secrets.
