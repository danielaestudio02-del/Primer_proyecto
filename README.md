# Pulso TransMi — Taller 1, versión 1

Análisis exploratorio y comparación inicial de modelos para la serie temporal de demanda por estación de TransMilenio.

## Contenido

- `analysis/`: scripts de análisis exploratorio y comparación de baselines.
- `data/starter/`: conjunto público de arranque proporcionado para el taller.
- `reports/`: resultados, protocolo de validación y configuración de Supabase.
- `supabase/migrations/`: esquema inicial de PostgreSQL/Supabase.

## Reproducibilidad

Los análisis locales usan Python. Para instalar las dependencias del cargador de Supabase:

```bash
pip install -r requirements-supabase.txt
```

Las credenciales locales van en `.env.local` y no deben subirse al repositorio. Usa `.env.example` como referencia sin agregar secretos reales.

Para repetir la comparación de un solo corte, ejecuta `python analysis/compare_baselines.py --source supabase`. Para comparar persistencia, los naives diario/semanal, Random Forest y XGBoost en tres semanas temporales, instala `python -m pip install -r requirements-modeling.txt` y ejecuta `python analysis/backtest_models.py --source supabase`.

La sincronización continua y los envíos por ciclo están en `analysis/pulso_pipeline.py`. Requieren un bucket privado de Supabase Storage llamado `model-artifacts` y los GitHub Actions Secrets `PULSO_API_KEY`, `SUPABASE_URL` y `SUPABASE_SECRET_KEY`. Consulta [la guía de operación](reports/operacion_envios_v1.md) para la preparación y el primer entrenamiento. El workflow revisa ciclos cada 10 minutos; entrena/promueve aparte y solo entrega si hay ciclo abierto y un champion válido.

## Estado

Los resultados actuales son exploratorios y usan datos sintéticos de arranque. La evaluación de competencia debe seguir el protocolo y los ciclos publicados por el curso.
