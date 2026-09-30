-- Read-only monitoring API for the dashboard: drift/performance events, training
-- decisions, the kind of model behind every version and accuracy per station and
-- cycle. Same pattern as the other api_* functions: SECURITY DEFINER, curated
-- columns only, callable with the publishable key.

-- Monitoring timeline: data drift, performance drift, missed cycles and decisions.
create or replace function public.api_monitoring_events(p_limit integer default 100)
returns table (
    detected_at timestamptz,
    event_type text,
    severity text,
    station_id text,
    model_version text,
    decision text,
    details jsonb
)
language sql
stable
security definer
set search_path = ''
as $$
    select e.detected_at, e.event_type, e.severity, e.station_id, e.model_version, e.decision, e.details
    from public.monitoring_events e
    order by e.detected_at desc
    limit least(greatest(coalesce(p_limit, 100), 1), 500);
$$;

-- Model kind per version: 'adaptive', 'fixed' (65/35 ensemble) or 'random_forest'.
create or replace function public.api_model_versions()
returns table (
    model_version text,
    status text,
    kind text,
    created_at timestamptz,
    promoted_at timestamptz,
    training_data_end timestamptz
)
language sql
stable
security definer
set search_path = ''
as $$
    select mv.model_version, mv.status,
           case
               when coalesce(tr.parameters ->> 'adaptive', 'false') = 'true' then 'adaptive'
               when mv.model_version like 'ens-%' then 'fixed'
               else 'random_forest'
           end,
           mv.created_at, mv.promoted_at, mv.training_data_end
    from public.model_versions mv
    left join public.training_runs tr using (training_run_id)
    order by mv.created_at;
$$;

-- Every training (promoted or not) with the evidence behind the decision.
create or replace function public.api_training_history(p_limit integer default 30)
returns table (
    finished_at timestamptz,
    training_data_end timestamptz,
    promoted boolean,
    model_version text,
    adaptive boolean,
    notes text,
    holdout_ensemble double precision,
    holdout_random_forest double precision,
    holdout_weekly_naive double precision,
    recent_candidate double precision,
    recent_production double precision,
    recent_cycles integer,
    recent_skipped text
)
language sql
stable
security definer
set search_path = ''
as $$
    select tr.finished_at, tr.training_data_end,
           mv.model_version is not null, mv.model_version,
           coalesce(tr.parameters ->> 'adaptive', 'false') = 'true',
           tr.notes,
           (tr.metrics -> 'ensemble' ->> 'official_accuracy')::double precision,
           (tr.metrics -> 'random_forest' ->> 'official_accuracy')::double precision,
           (tr.metrics -> 'weekly_naive' ->> 'official_accuracy')::double precision,
           (tr.metrics -> 'recent_window' ->> 'candidate')::double precision,
           (tr.metrics -> 'recent_window' ->> 'production')::double precision,
           (tr.metrics -> 'recent_window' ->> 'cycles')::integer,
           tr.metrics -> 'recent_window' ->> 'skipped'
    from public.training_runs tr
    left join public.model_versions mv on mv.training_run_id = tr.training_run_id
    order by tr.finished_at desc nulls last
    limit least(greatest(coalesce(p_limit, 30), 1), 200);
$$;

-- Accuracy per station for the most recent fully evaluated cycles (heatmap).
create or replace function public.api_station_cycle_accuracy(p_cycles integer default 24)
returns table (
    cycle_id text,
    data_cutoff timestamptz,
    model_version text,
    station_id text,
    accuracy double precision,
    naive_accuracy double precision
)
language sql
stable
security definer
set search_path = ''
as $$
    with anchor as (
        select max(o.observed_at) + interval '15 minutes' as to_at from public.observations o
    ),
    scored as (
        select sp.*, w.demand as naive_value
        from anchor a,
             lateral public.api_scored_predictions(a.to_at - interval '14 days', a.to_at + interval '1 day') sp
        left join public.observations w
            on w.station_id = sp.station_id and w.observed_at = sp.target_at - interval '7 days'
    ),
    complete as (
        select cycle_id from scored group by cycle_id having count(*) = count(actual_demand)
    ),
    recent as (
        select fc.cycle_id, fc.data_cutoff
        from public.forecast_cycles fc join complete using (cycle_id)
        order by fc.data_cutoff desc
        limit least(greatest(coalesce(p_cycles, 24), 1), 96)
    )
    select s.cycle_id, r.data_cutoff, min(s.model_version), s.station_id,
           greatest(0, 100 * (1 - sum(abs(s.actual_demand - s.predicted_value)) / greatest(sum(s.actual_demand), 1))),
           case when count(s.naive_value) = count(*) then
               greatest(0, 100 * (1 - sum(abs(s.actual_demand - s.naive_value))::double precision / greatest(sum(s.actual_demand), 1)))
           end
    from scored s join recent r using (cycle_id)
    group by s.cycle_id, r.data_cutoff, s.station_id
    order by r.data_cutoff, s.station_id;
$$;

revoke all on function
    public.api_monitoring_events(integer),
    public.api_model_versions(),
    public.api_training_history(integer),
    public.api_station_cycle_accuracy(integer)
from public;
grant execute on function
    public.api_monitoring_events(integer),
    public.api_model_versions(),
    public.api_training_history(integer),
    public.api_station_cycle_accuracy(integer)
to anon, authenticated, service_role;
