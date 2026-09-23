-- Private student-side MLOps store for Pulso TransMi.
-- Apply with the Supabase SQL Editor or the Supabase CLI migration workflow.

create table if not exists public.stations (
    station_id text primary key,
    station_name text not null,
    corridor text not null,
    latitude double precision not null,
    longitude double precision not null,
    created_at timestamptz not null default now(),
    constraint stations_id_not_empty check (length(station_id) > 0),
    constraint stations_latitude_valid check (latitude between -90 and 90),
    constraint stations_longitude_valid check (longitude between -180 and 180)
);

create table if not exists public.observations (
    station_id text not null references public.stations(station_id),
    observed_at timestamptz not null,
    demand integer not null check (demand >= 0),
    ingested_at timestamptz not null default now(),
    primary key (station_id, observed_at)
);
create index if not exists observations_observed_at_idx
    on public.observations (observed_at desc, station_id);

create table if not exists public.context_observations (
    observed_at timestamptz primary key,
    rain_mm double precision not null check (rain_mm >= 0 and rain_mm < 'Infinity'::double precision),
    rain_forecast double precision not null check (rain_forecast >= 0 and rain_forecast < 'Infinity'::double precision),
    temperature_c double precision not null check (temperature_c > '-Infinity'::double precision and temperature_c < 'Infinity'::double precision),
    temperature_forecast double precision not null check (temperature_forecast > '-Infinity'::double precision and temperature_forecast < 'Infinity'::double precision),
    event_intensity double precision not null check (event_intensity between 0 and 1),
    ingested_at timestamptz not null default now()
);

create table if not exists public.collector_state (
    singleton boolean primary key default true check (singleton),
    confirmed_cursor text,
    last_observed_at timestamptz,
    updated_at timestamptz not null default now()
);
insert into public.collector_state (singleton)
values (true)
on conflict (singleton) do nothing;

create table if not exists public.collector_runs (
    run_id uuid primary key default gen_random_uuid(),
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    status text not null default 'running'
        check (status in ('running', 'succeeded', 'failed')),
    cursor_before text,
    cursor_after text,
    rows_received integer not null default 0 check (rows_received >= 0),
    rows_upserted integer not null default 0 check (rows_upserted >= 0),
    error_summary text,
    metadata jsonb not null default '{}'::jsonb
);
create index if not exists collector_runs_started_at_idx
    on public.collector_runs (started_at desc);

create table if not exists public.training_runs (
    training_run_id uuid primary key default gen_random_uuid(),
    model_family text not null,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    training_data_start timestamptz,
    training_data_end timestamptz not null,
    validation_start timestamptz,
    validation_end timestamptz,
    status text not null default 'completed'
        check (status in ('running', 'completed', 'failed')),
    code_commit text,
    parameters jsonb not null default '{}'::jsonb,
    metrics jsonb not null default '{}'::jsonb,
    notes text
);

create table if not exists public.model_versions (
    model_version text primary key,
    training_run_id uuid references public.training_runs(training_run_id),
    status text not null default 'candidate'
        check (status in ('candidate', 'champion', 'retired')),
    created_at timestamptz not null default now(),
    promoted_at timestamptz,
    training_data_end timestamptz not null,
    code_commit text,
    artifact_uri text,
    features jsonb not null default '[]'::jsonb,
    validation_metrics jsonb not null default '{}'::jsonb,
    notes text
);
create unique index if not exists one_champion_model_idx
    on public.model_versions ((status)) where status = 'champion';

create table if not exists public.forecast_cycles (
    cycle_id text primary key,
    data_cutoff timestamptz not null,
    closes_at timestamptz not null,
    expected_predictions integer not null check (expected_predictions > 0),
    status text not null default 'seen'
        check (status in ('seen', 'predicted', 'submitted', 'evaluated', 'closed')),
    fetched_at timestamptz not null default now(),
    payload jsonb not null default '{}'::jsonb,
    constraint cycle_close_after_cutoff check (closes_at > data_cutoff)
);
create index if not exists forecast_cycles_closes_at_idx
    on public.forecast_cycles (closes_at desc);

create table if not exists public.cycle_targets (
    cycle_id text not null references public.forecast_cycles(cycle_id),
    station_id text not null references public.stations(station_id),
    target_at timestamptz not null,
    horizon_minutes integer not null check (horizon_minutes > 0 and horizon_minutes % 15 = 0),
    primary key (cycle_id, station_id, target_at)
);
create index if not exists cycle_targets_target_at_idx
    on public.cycle_targets (target_at, station_id);

create table if not exists public.submissions (
    submission_id text primary key,
    client_run_id text not null,
    cycle_id text not null references public.forecast_cycles(cycle_id),
    model_version text not null references public.model_versions(model_version),
    attempt integer not null check (attempt between 1 and 3),
    status text not null check (status in ('accepted', 'duplicate', 'superseded')),
    accepted_at timestamptz not null,
    data_cutoff timestamptz not null,
    training_data_end timestamptz not null,
    code_commit text,
    batch_sha256 text,
    is_official boolean not null default true,
    receipt jsonb not null default '{}'::jsonb,
    constraint submission_training_before_cutoff check (training_data_end <= data_cutoff),
    unique (cycle_id, attempt)
);
create index if not exists submissions_cycle_idx
    on public.submissions (cycle_id, accepted_at desc);

create table if not exists public.predictions (
    submission_id text not null references public.submissions(submission_id),
    station_id text not null references public.stations(station_id),
    target_at timestamptz not null,
    predicted_value double precision not null check (
        predicted_value >= 0 and predicted_value < 'Infinity'::double precision
    ),
    issued_at timestamptz not null default now(),
    primary key (submission_id, station_id, target_at)
);
create index if not exists predictions_target_station_idx
    on public.predictions (target_at, station_id);

create table if not exists public.metric_snapshots (
    snapshot_id uuid primary key default gen_random_uuid(),
    model_version text references public.model_versions(model_version),
    window_start timestamptz not null,
    window_end timestamptz not null,
    metric_level text not null check (metric_level in ('overall', 'station', 'horizon')),
    station_id text references public.stations(station_id),
    horizon_minutes integer,
    wape double precision check (wape is null or (wape >= 0 and wape < 'Infinity'::double precision)),
    accuracy double precision check (accuracy is null or accuracy between 0 and 100),
    coverage double precision check (coverage is null or coverage between 0 and 1),
    evaluated_targets integer not null check (evaluated_targets >= 0),
    expected_targets integer not null check (expected_targets >= evaluated_targets),
    calculated_at timestamptz not null default now(),
    constraint metric_window_valid check (window_end > window_start),
    constraint metric_dimensions_valid check (
        (metric_level = 'overall' and station_id is null and horizon_minutes is null)
        or (metric_level = 'station' and station_id is not null and horizon_minutes is null)
        or (metric_level = 'horizon' and station_id is null and horizon_minutes is not null)
    ),
    constraint metric_horizon_valid check (
        horizon_minutes is null or (horizon_minutes > 0 and horizon_minutes % 15 = 0)
    ),
    unique nulls not distinct (model_version, window_start, window_end, metric_level, station_id, horizon_minutes)
);
create index if not exists metric_snapshots_window_idx
    on public.metric_snapshots (window_end desc, metric_level);

create table if not exists public.monitoring_events (
    event_id uuid primary key default gen_random_uuid(),
    detected_at timestamptz not null default now(),
    model_version text references public.model_versions(model_version),
    event_type text not null check (event_type in ('performance_drift', 'data_drift', 'operational_failure', 'retraining_decision')),
    severity text not null default 'info' check (severity in ('info', 'warning', 'critical')),
    station_id text references public.stations(station_id),
    details jsonb not null default '{}'::jsonb,
    decision text
);
create index if not exists monitoring_events_detected_at_idx
    on public.monitoring_events (detected_at desc);

-- The tables are for trusted server-side jobs. No browser/client role gets access.
alter table public.stations enable row level security;
alter table public.observations enable row level security;
alter table public.context_observations enable row level security;
alter table public.collector_state enable row level security;
alter table public.collector_runs enable row level security;
alter table public.training_runs enable row level security;
alter table public.model_versions enable row level security;
alter table public.forecast_cycles enable row level security;
alter table public.cycle_targets enable row level security;
alter table public.submissions enable row level security;
alter table public.predictions enable row level security;
alter table public.metric_snapshots enable row level security;
alter table public.monitoring_events enable row level security;

revoke all on table
    public.stations,
    public.observations,
    public.context_observations,
    public.collector_state,
    public.collector_runs,
    public.training_runs,
    public.model_versions,
    public.forecast_cycles,
    public.cycle_targets,
    public.submissions,
    public.predictions,
    public.metric_snapshots,
    public.monitoring_events
from anon, authenticated;
grant all on table
    public.stations,
    public.observations,
    public.context_observations,
    public.collector_state,
    public.collector_runs,
    public.training_runs,
    public.model_versions,
    public.forecast_cycles,
    public.cycle_targets,
    public.submissions,
    public.predictions,
    public.metric_snapshots,
    public.monitoring_events
to service_role;
alter default privileges for role postgres in schema public
    revoke all on tables from anon, authenticated;
alter default privileges for role postgres in schema public
    grant all on tables to service_role;
