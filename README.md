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

## Estado

Los resultados actuales son exploratorios y usan datos sintéticos de arranque. La evaluación de competencia debe seguir el protocolo y los ciclos publicados por el curso.
