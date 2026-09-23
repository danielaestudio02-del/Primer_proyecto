# Montaje de Supabase — taller 1

## Qué conserva la base

La migración crea tablas para estaciones, observaciones, contexto, cursor y ejecuciones del collector, entrenamientos, versiones de modelo, ciclos y targets, recibos y predicciones, métricas y eventos de monitoreo. La clave primaria de observaciones es `(station_id, observed_at)`, así que repetir una importación actualiza las filas y no las duplica.

Todas las tablas tienen Row Level Security habilitado; `anon` y `authenticated` no reciben permisos. El cargador usa una clave secreta de Supabase desde Python, nunca desde el navegador.

## Preparar el proyecto en Supabase

1. Crea un proyecto nuevo en tu cuenta de Supabase. Elige una contraseña de base de datos fuerte y guárdala en tu gestor de contraseñas; no la pongas en el repositorio.
2. En el proyecto, abre **SQL Editor** y ejecuta `supabase/migrations/202609220001_initial_schema.sql`.
3. En **Project Settings → API Keys**, copia la URL del proyecto y crea/copia una **Secret key** para el backend. No uses esa clave en frontend, notebook público o código compartido: tiene acceso administrativo y omite RLS.
4. Añade `SUPABASE_URL` y `SUPABASE_SECRET_KEY` a `.env.local`, conservando la línea local `PULSO_API_KEY`. No compartas el archivo.
5. Instala las dependencias con `python -m pip install -r requirements-supabase.txt`.
6. Ejecuta `python analysis/load_starter_to_supabase.py`. El cargador verifica los hashes, valida los IDs y carga por lotes idempotentes.

## Comprobación en SQL Editor

Después de cargar, las siguientes consultas deben devolver 12, 4.320 y 51.840:

```sql
select
  (select count(*) from public.stations) as stations,
  (select count(*) from public.context_observations) as context_rows,
  (select count(*) from public.observations) as observation_rows;
```

Verifica también que los roles del navegador no puedan leer las tablas:

```sql
select has_table_privilege('anon', 'public.observations', 'select') as anon_can_read,
       has_table_privilege('authenticated', 'public.observations', 'select') as authenticated_can_read;
```

Ambos resultados deben ser `false`. Las consultas futuras desde Python/GitHub Actions se autentican en el backend con la clave secreta.

## Leer los datos desde Supabase para el benchmark

El comparador conserva `local` como origen predeterminado. Para leer la tabla `observations` desde Supabase con paginación y usar el mismo holdout temporal:

```powershell
python analysis/compare_baselines.py --source supabase
```

El programa carga las variables desde `.env.local`, no imprime sus valores y vuelve a escribir `reports/model_baselines_v1.md` y la figura comparativa. Para reproducir el resultado original con los CSV locales, ejecuta `python analysis/compare_baselines.py`.

Para evaluar varios métodos en varios cortes temporales, instala también los modelos y dependencias de visualización con `python -m pip install -r requirements-modeling.txt`, y ejecuta:

```powershell
python analysis/backtest_models.py --source supabase
```

El script compara persistencia, naive diario, naive semanal, Random Forest y XGBoost en tres ventanas semanales consecutivas. Genera `reports/backtest_modelos_v1.md` con métricas medias, variación y peor fold.

## Más adelante

El collector guardará la nueva página de observaciones y solo avanzará `collector_state.confirmed_cursor` después de completar el upsert. Las ejecuciones y recibos se conservan para auditoría; la clave de Pulso TransMi nunca se guarda en tablas ni logs.
