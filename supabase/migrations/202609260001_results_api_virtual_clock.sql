-- The competition runs on a virtual clock: observed_at/target_at are synthetic
-- timestamps that can lag the real date by days. Result windows are therefore
-- anchored to the newest observation in the database instead of now().

drop function if exists public.api_predictions_vs_actuals(text, timestamptz, timestamptz);
drop function if exists public.api_accuracy_summary(integer);

-- Predictions vs. observed demand for one station (max window 14 days).
-- Defaults: the 2 days up to the latest target, so pending targets are included.
create function public.api_predictions_vs_actuals(
    p_station_id text,
    p_from timestamptz default null,
    p_to timestamptz default null
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
    with bounds as (
        select coalesce(
                   p_to,
                   greatest(
                       (select max(o.observed_at) from public.observations o),
                       (select max(p.target_at) from public.predictions p)
                   ) + interval '15 minutes'
               ) as to_at
    ),
    win as (
        select greatest(coalesce(p_from, b.to_at - interval '2 days'), b.to_at - interval '14 days') as from_at,
               b.to_at
        from bounds b
    )
    select sp.target_at, sp.horizon_minutes, sp.predicted_value, sp.actual_demand,
           abs(sp.actual_demand - sp.predicted_value), sp.cycle_id, sp.model_version
    from win w,
         lateral public.api_scored_predictions(w.from_at, w.to_at) sp
    where sp.station_id = p_station_id
    order by sp.target_at, sp.horizon_minutes;
$$;

-- Accuracy of official submissions over the p_days (1..90) of virtual time ending at
-- the newest observation, with the course metric: per-station WAPE,
-- accuracy = max(0, 100 * (1 - WAPE)), official accuracy = mean of station accuracies.
-- Targets whose actual demand is not released yet are reported as pending.
create function public.api_accuracy_summary(p_days integer default 7)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
    with anchor as (
        select max(o.observed_at) + interval '15 minutes' as to_at,
               least(greatest(coalesce(p_days, 7), 1), 90) as days
        from public.observations o
    ),
    scored as (
        select sp.*
        from anchor a,
             lateral public.api_scored_predictions(
                 a.to_at - make_interval(days => a.days),
                 a.to_at + interval '1 day'
             ) sp
    ),
    window_rows as (
        select * from scored where actual_demand is not null
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
        'window_days', (select days from anchor),
        'window_end', (select to_at from anchor),
        'calculated_at', now(),
        'evaluated_targets', (select count(*) from window_rows),
        'pending_targets', (select count(*) from scored where actual_demand is null),
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

revoke all on function
    public.api_predictions_vs_actuals(text, timestamptz, timestamptz),
    public.api_accuracy_summary(integer)
from public;
grant execute on function
    public.api_predictions_vs_actuals(text, timestamptz, timestamptz),
    public.api_accuracy_summary(integer)
to anon, authenticated, service_role;
