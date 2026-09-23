# Operación de sincronización, modelo y envíos

## Flujo

- `pulso-hourly.yml` se despierta cada 10 minutos. Sincroniza el stream incremental, consulta `/v1/forecast-cycles/current` y envía solo si hay un ciclo abierto, no existe ya una entrega oficial registrada, el modelo está vigente al cutoff y están disponibles los datos recientes.
- `train-random-forest.yml` se ejecuta manualmente. Sincroniza, entrena un candidato Random Forest, lo compara con naive semanal en un holdout reciente y solo promueve/sube el artefacto `.joblib` si lo supera.
- El cursor de stream se confirma en Supabase después de guardar cada página. Repetir una página es seguro por el upsert `(station_id, observed_at)`.
- Una entrega conserva su recibo, targets, predicciones, versión de modelo y hash del batch.

## Preparación única en Supabase y GitHub

1. En Supabase, abre **Storage** y crea un bucket privado llamado `model-artifacts`.
2. En el repositorio `Primer_proyecto`, abre **Settings → Secrets and variables → Actions** y crea estos repository secrets:
   - `PULSO_API_KEY`: la llave del portal Pulso TransMi.
   - `SUPABASE_URL`: Project URL.
   - `SUPABASE_SECRET_KEY`: Secret key de backend.
3. Comprueba en GitHub que aparecen los dos workflows en **Actions**.
4. Ejecuta primero **Train and promote Random Forest → Run workflow**. Debe registrar la versión champion y subir el artefacto al bucket.
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
