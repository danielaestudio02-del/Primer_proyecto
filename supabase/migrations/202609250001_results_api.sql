-- Read-only results API for the Pulso TransMi dashboard (Vercel).
--
-- The base tables stay closed to anon/authenticated (see the initial schema).
-- These SECURITY DEFINER functions expose only curated, aggregated results and
-- are reachable through PostgREST as POST/GET /rest/v1/rpc/<function_name>
-- with the project's publishable (anon) key. No secrets, receipts or model
-- artifacts are returned.

-- Stations catalogue for maps and selectors.
create or replace function public.api_stations()
returns table (
    station_id text,
    station_name text,
    corridor text,
    latitude double precision,
    longitude double precision
)
language sql
stable
security definer
set search_path = ''
as $$
    select s.station_id, s.station_name, s.corridor, s.latitude, s.longitude
    from public.stations s
    order by s.station_id;
$$;

-- Current champion model and its validation against the weekly naive baseline.
create or replace function public.api_champion_model()
returns table (
    model_version text,
    status text,
    promoted_at timestamptz,
    training_data_end timestamptz,
    code_commit text,
    features jsonb,
    validation_metrics jsonb,
    model_family text,
    parameters jsonb,
    validation_start timestamptz,
    validation_end timestamptz
)
language sql
stable
security definer
set search_path = ''
as $$
    select mv.model_version, mv.status, mv.promoted_at, mv.training_data_end,
           mv.code_commit, mv.features, mv.validation_metrics,
           tr.model_family, tr.parameters, tr.validation_start, tr.validation_end
    from public.model_versions mv
    left join public.training_runs tr using (training_run_id)
    where mv.status = 'champion';
$$;

-- Model registry history, newest first.
create or replace function public.api_model_history(p_limit integer default 20)
returns table (
    model_version text,
    status text,
    created_at timestamptz,
    promoted_at timestamptz,
    training_data_end timestamptz,
    validation_metrics jsonb
)
language sql
stable
security definer
set search_path = ''
as $$
    select mv.model_version, mv.status, mv.created_at, mv.promoted_at,
           mv.training_data_end, mv.validation_metrics
    from public.model_versions mv
    order by mv.created_at desc
    limit least(greatest(coalesce(p_limit, 20), 1), 100);
$$;

-- Recent submissions with their cycle and how many predictions were stored.
create or replace function public.api_submissions(p_limit integer default 20)
returns table (
    submission_id text,
    cycle_id text,
    model_version text,
    attempt integer,
    status text,
    is_official boolean,
    accepted_at timestamptz,
    data_cutoff timestamptz,
    cycle_status text,
    expected_predictions integer,
    stored_predictions bigint
)
language sql
stable
security definer
set search_path = ''
as $$
    select s.submission_id, s.cycle_id, s.model_version, s.attempt, s.status,
           s.is_official, s.accepted_at, s.data_cutoff,
           c.status, c.expected_predictions,
           (select count(*) from public.predictions p where p.submission_id = s.submission_id)
    from public.submissions s
    join public.forecast_cycles c using (cycle_id)
    order by s.accepted_at desc
    limit least(greatest(coalesce(p_limit, 20), 1), 100);
$$;

-- Latest official submission per cycle; shared by the result functions below.
create or replace function public.api_scored_predictions(p_from timestamptz, p_to timestamptz)
returns table (
    cycle_id text,
    submission_id text,
    model_version text,
    station_id text,
    target_at timestamptz,
    horizon_minutes integer,
    predicted_value double precision,
    actual_demand integer
)
language sql
stable
security definer
set search_path = ''
as $$
    with latest as (
        select distinct on (s.cycle_id) s.cycle_id, s.submission_id, s.model_version
        from public.submissions s
        where s.is_official and s.status in ('accepted', 'duplicate')
        order by s.cycle_id, s.accepted_at desc, s.attempt desc
    )
    select l.cycle_id, l.submission_id, l.model_version, p.station_id, p.target_at,
           ct.horizon_minutes, p.predicted_value, o.demand
    from latest l
    join public.predictions p on p.submission_id = l.submission_id
    left join public.cycle_targets ct
        on ct.cycle_id = l.cycle_id and ct.station_id = p.station_id and ct.target_at = p.target_at
    left join public.observations o
        on o.station_id = p.station_id and o.observed_at = p.target_at
    where p.target_at >= p_from and p.target_at < p_to;
$$;
-- Internal helper: not callable by browser roles.
revoke all on function public.api_scored_predictions(timestamptz, timestamptz) from public, anon, authenticated;

-- Predictions vs. observed demand for one station (max window 14 days).
create or replace function public.api_predictions_vs_actuals(
    p_station_id text,
    p_from timestamptz default now() - interval '2 days',
    p_to timestamptz default now()
)
returns table (
    target_at timestamptz,
    horizon_minutes integer,
    predicted_value double precision,
    actual_demand integer,
    absolute_error double precision,
    cycle_id text,
    model_version text
)
language sql
stable
security definer
set search_path = ''
as $$
    select sp.target_at, sp.horizon_minutes, sp.predicted_value, sp.actual_demand,
           abs(sp.actual_demand - sp.predicted_value), sp.cycle_id, sp.model_version
    from public.api_scored_predictions(
        greatest(p_from, p_to - interval '14 days'), p_to
    ) sp
    where sp.station_id = p_station_id
    order by sp.target_at, sp.horizon_minutes;
$$;

-- Live accuracy of official submissions over the last p_days (1..90), using the
-- course metric: per-station WAPE, accuracy = max(0, 100 * (1 - WAPE)), and the
-- official accuracy as the mean of station accuracies.
create or replace function public.api_accuracy_summary(p_days integer default 7)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
    with window_rows as (
        select *
        from public.api_scored_predictions(
            now() - make_interval(days => least(greatest(coalesce(p_days, 7), 1), 90)),
            now()
        )
        where actual_demand is not null
    ),
    by_station as (
        select station_id,
               count(*) as evaluated_targets,
               sum(abs(actual_demand - predicted_value)) / greatest(sum(actual_demand), 1) as wape
        from window_rows
        group by station_id
    ),
    by_horizon as (
        select horizon_minutes,
               count(*) as evaluated_targets,
               sum(abs(actual_demand - predicted_value)) / greatest(sum(actual_demand), 1) as wape
        from window_rows
        group by horizon_minutes
    )
    select jsonb_build_object(
        'window_days', least(greatest(coalesce(p_days, 7), 1), 90),
        'calculated_at', now(),
        'evaluated_targets', (select count(*) from window_rows),
        'cycles', (select count(distinct cycle_id) from window_rows),
        'official_accuracy', (select avg(greatest(0, 100 * (1 - wape))) from by_station),
        'mean_station_wape', (select avg(wape) from by_station),
        'mae', (select avg(abs(actual_demand - predicted_value)) from window_rows),
        'by_station', coalesce((
            select jsonb_agg(jsonb_build_object(
                'station_id', station_id,
                'evaluated_targets', evaluated_targets,
                'wape', wape,
                'accuracy', greatest(0, 100 * (1 - wape))
            ) order by station_id)
            from by_station
        ), '[]'::jsonb),
        'by_horizon', coalesce((
            select jsonb_agg(jsonb_build_object(
                'horizon_minutes', horizon_minutes,
                'evaluated_targets', evaluated_targets,
                'wape', wape,
                'accuracy', greatest(0, 100 * (1 - wape))
            ) order by horizon_minutes)
            from by_horizon
        ), '[]'::jsonb)
    );
$$;

-- Observed demand series for one station (max window 14 days).
create or replace function public.api_station_demand(
    p_station_id text,
    p_hours integer default 48
)
returns table (observed_at timestamptz, demand integer)
language sql
stable
security definer
set search_path = ''
as $$
    with bounds as (
        select max(o.observed_at) as last_at
        from public.observations o
        where o.station_id = p_station_id
    )
    select o.observed_at, o.demand
    from public.observations o, bounds b
    where o.station_id = p_station_id
      and o.observed_at > b.last_at
          - make_interval(hours => least(greatest(coalesce(p_hours, 48), 1), 336))
    order by o.observed_at;
$$;

-- Pipeline health: data freshness, last collector run, latest cycle and submission.
create or replace function public.api_pipeline_status()
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
    select jsonb_build_object(
        'last_observed_at', (select last_observed_at from public.collector_state where singleton),
        'observation_rows', (select count(*) from public.observations),
        'stations', (select count(*) from public.stations),
        'last_collector_run', (
            select jsonb_build_object(
                'started_at', r.started_at,
                'finished_at', r.finished_at,
                'status', r.status,
                'rows_upserted', r.rows_upserted
            )
            from public.collector_runs r
            order by r.started_at desc
            limit 1
        ),
        'latest_cycle', (
            select jsonb_build_object(
                'cycle_id', c.cycle_id,
                'data_cutoff', c.data_cutoff,
                'closes_at', c.closes_at,
                'status', c.status
            )
            from public.forecast_cycles c
            order by c.closes_at desc
            limit 1
        ),
        'latest_submission_at', (select max(accepted_at) from public.submissions),
        'champion_model', (select model_version from public.model_versions where status = 'champion')
    );
$$;

-- Supabase grants EXECUTE to PUBLIC by default; expose only the api_* endpoints.
revoke all on function
    public.api_stations(),
    public.api_champion_model(),
    public.api_model_history(integer),
    public.api_submissions(integer),
    public.api_predictions_vs_actuals(text, timestamptz, timestamptz),
    public.api_accuracy_summary(integer),
    public.api_station_demand(text, integer),
    public.api_pipeline_status()
from public;
grant execute on function
    public.api_stations(),
    public.api_champion_model(),
    public.api_model_history(integer),
    public.api_submissions(integer),
    public.api_predictions_vs_actuals(text, timestamptz, timestamptz),
    public.api_accuracy_summary(integer),
    public.api_station_demand(text, integer),
    public.api_pipeline_status()
to anon, authenticated, service_role;
grant execute on function public.api_scored_predictions(timestamptz, timestamptz) to service_role;
