# API de resultados en Supabase y despliegue en Vercel

## Arquitectura

```
Navegador ──► Vercel (dashboard/public/index.html)
                 │
                 └─► /api/pulso?endpoint=...   (función serverless, dashboard/api/pulso.js)
                          │  clave publishable en variables de entorno de Vercel
                          ▼
                 Supabase  /rest/v1/rpc/api_*   (funciones de solo lectura)
                          │  SECURITY DEFINER
                          ▼
                 Tablas privadas (RLS activo, sin acceso para anon/authenticated)
```

Las tablas siguen cerradas para los roles del navegador. La migración `supabase/migrations/202609250001_results_api.sql` añade funciones `api_*` que devuelven solo resultados agregados o curados, sin recibos, secretos ni artefactos del modelo. La **Secret key** nunca va a Vercel.

## Endpoints

| Endpoint del proxy (`/api/pulso?endpoint=`) | Función RPC en Supabase | Parámetros |
|---|---|---|
| `stations` | `api_stations()` | — |
| `champion` | `api_champion_model()` | — |
| `models` | `api_model_history(p_limit)` | `limit` (1–100, def. 20) |
| `submissions` | `api_submissions(p_limit)` | `limit` (1–100, def. 20) |
| `accuracy` | `api_accuracy_summary(p_days)` | `days` (1–90, def. 7) |
| `status` | `api_pipeline_status()` | — |
| `demand` | `api_station_demand(p_station_id, p_hours)` | `station_id`, `hours` (1–336, def. 48) |
| `predictions` | `api_predictions_vs_actuals(p_station_id, p_from, p_to)` | `station_id`, `from`, `to` (ISO 8601, máx. 14 días) |

`accuracy` usa la última entrega oficial (`accepted`/`duplicate`) de cada ciclo, cruza sus predicciones con `observations` y calcula la métrica del curso: WAPE por estación, accuracy = máx(0, 100·(1 − WAPE)) y accuracy oficial = promedio de las estaciones. Además la desglosa por estación y por horizonte.

Ejemplo directo contra Supabase:

```bash
curl -X POST "$SUPABASE_URL/rest/v1/rpc/api_accuracy_summary" \
  -H "apikey: $SUPABASE_PUBLISHABLE_KEY" -H "Content-Type: application/json" \
  -d '{"p_days": 7}'
```

## Pasos

### 1. Supabase

1. Abre **SQL Editor** y ejecuta `supabase/migrations/202609250001_results_api.sql` (después de la migración inicial).
2. Comprueba que el navegador sigue sin poder leer las tablas, pero sí puede usar la API:

   ```sql
   select has_table_privilege('anon', 'public.observations', 'select') as anon_tabla,          -- false
          has_function_privilege('anon', 'public.api_accuracy_summary(integer)', 'execute') as anon_api; -- true
   ```
3. En **Project Settings → API Keys** copia la **Publishable key** (`sb_publishable_...`, o la clave `anon` heredada). No uses la Secret key.

### 2. Vercel

1. En [vercel.com/new](https://vercel.com/new) importa el repositorio `Primer_proyecto`.
2. En **Root Directory** elige `dashboard`. Framework Preset: **Other**. No hace falta comando de build.
3. En **Environment Variables** añade:
   - `SUPABASE_URL`: la Project URL.
   - `SUPABASE_PUBLISHABLE_KEY`: la Publishable key.
4. Pulsa **Deploy**. Cada push a la rama de producción vuelve a desplegar.

La función proxy responde con `Cache-Control: s-maxage=60`, así que Vercel sirve cada consulta desde caché durante un minuto y Supabase no recibe una llamada por cada visita.

### Alternativa con CLI

```bash
npm i -g vercel
cd dashboard
vercel link
vercel env add SUPABASE_URL
vercel env add SUPABASE_PUBLISHABLE_KEY
vercel --prod
```
