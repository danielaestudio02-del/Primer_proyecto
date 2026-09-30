-- Per-cycle accuracy of the official submissions, next to the weekly naive baseline
-- scored on exactly the same targets (the demand 7 days before each target). This
-- separates "the model got worse" from "these hours were hard for everyone".
-- Uses the same virtual-clock anchor as api_accuracy_summary.

create or replace function public.api_cycle_accuracy(p_limit integer default 48)
returns table (
    cycle_id text,
    data_cutoff timestamptz,
    model_version text,
    targets integer,
    evaluated_targets integer,
    accuracy double precision,
    naive_accuracy double precision,
    mae double precision
)
language sql
stable
security definer
set search_path = ''
as $$
    with anchor as (
        select max(o.observed_at) + interval '15 minutes' as to_at
        from public.observations o
    ),
    scored as (
        select sp.*, w.demand as naive_value
        from anchor a,
             lateral public.api_scored_predictions(a.to_at - interval '14 days', a.to_at + interval '1 day') sp
        left join public.observations w
            on w.station_id = sp.station_id and w.observed_at = sp.target_at - interval '7 days'
    ),
    by_station as (
        select cycle_id, station_id,
               sum(abs(actual_demand - predicted_value)) / greatest(sum(actual_demand), 1) as wape,
               (sum(abs(actual_demand - naive_value)) filter (where naive_value is not null))::double precision
                   / nullif(greatest(sum(actual_demand) filter (where naive_value is not null), 1), 0) as naive_wape
        from scored
        where actual_demand is not null
        group by cycle_id, station_id
    ),
    by_cycle as (
        select cycle_id,
               avg(greatest(0, 100 * (1 - wape))) as accuracy,
               avg(case when naive_wape is not null then greatest(0, 100 * (1 - naive_wape)) end) as naive_accuracy
        from by_station
        group by cycle_id
    ),
    counts as (
        select cycle_id,
               min(model_version) as model_version,
               count(*)::integer as targets,
               count(actual_demand)::integer as evaluated_targets,
               avg(abs(actual_demand - predicted_value)) as mae
        from scored
        group by cycle_id
    )
    select c.cycle_id, fc.data_cutoff, c.model_version, c.targets, c.evaluated_targets,
           b.accuracy, b.naive_accuracy, c.mae
    from counts c
    join public.forecast_cycles fc using (cycle_id)
    left join by_cycle b using (cycle_id)
    order by fc.data_cutoff desc
    limit least(greatest(coalesce(p_limit, 48), 1), 200);
$$;

revoke all on function public.api_cycle_accuracy(integer) from public;
grant execute on function public.api_cycle_accuracy(integer) to anon, authenticated, service_role;
