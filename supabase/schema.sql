create extension if not exists pgcrypto;

create table if not exists public.dataset_catalog (
    dataset_id uuid primary key default gen_random_uuid(),
    dataset_key text not null unique,
    generated_at timestamptz,
    timezone text not null default 'America/Bogota',
    frequency_minutes smallint not null check (frequency_minutes > 0),
    history_start timestamptz,
    history_end timestamptz,
    station_count integer,
    observation_rows bigint,
    context_rows bigint,
    source_hash text,
    created_at timestamptz not null default now()
);

create table if not exists public.stations (
    station_id text primary key,
    station_name text not null,
    corridor text,
    latitude double precision,
    longitude double precision,
    active boolean not null default true,
    updated_at timestamptz not null default now()
);

create table if not exists public.time_intervals (
    observed_at timestamptz primary key,
    local_date date not null,
    local_time time not null,
    local_hour smallint not null check (local_hour between 0 and 23),
    day_of_week smallint not null check (day_of_week between 0 and 6),
    is_weekend boolean not null,
    interval_number smallint not null check (interval_number between 0 and 95)
);

create table if not exists public.context_observations (
    observed_at timestamptz primary key references public.time_intervals(observed_at),
    dataset_id uuid not null references public.dataset_catalog(dataset_id),
    rain_mm double precision,
    rain_forecast double precision,
    temperature_c double precision,
    temperature_forecast double precision,
    event_intensity double precision,
    loaded_at timestamptz not null default now()
);

create table if not exists public.station_demand (
    station_id text not null references public.stations(station_id),
    observed_at timestamptz not null references public.time_intervals(observed_at),
    dataset_id uuid not null references public.dataset_catalog(dataset_id),
    demand integer not null check (demand >= 0),
    loaded_at timestamptz not null default now(),
    primary key (station_id, observed_at)
);

create table if not exists public.ingestion_runs (
    ingestion_id uuid primary key default gen_random_uuid(),
    dataset_id uuid references public.dataset_catalog(dataset_id),
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    cursor_start text,
    cursor_end text,
    rows_received bigint not null default 0,
    status text not null check (status in ('running', 'success', 'failed')),
    error_message text
);

create table if not exists public.pipeline_runs (
    run_id uuid primary key default gen_random_uuid(),
    ingestion_id uuid references public.ingestion_runs(ingestion_id),
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    data_cutoff timestamptz,
    git_commit text,
    status text not null check (status in ('running', 'success', 'failed')),
    retrain_reason text
);

create table if not exists public.models (
    model_id uuid primary key default gen_random_uuid(),
    model_name text not null,
    algorithm text not null,
    version text not null,
    artifact_uri text,
    hyperparameters jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    is_active boolean not null default false,
    unique (model_name, version)
);

create table if not exists public.feature_sets (
    feature_set_id uuid primary key default gen_random_uuid(),
    name text not null,
    version text not null,
    definition jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (name, version)
);

create table if not exists public.predictions (
    prediction_id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.pipeline_runs(run_id),
    model_id uuid not null references public.models(model_id),
    feature_set_id uuid not null references public.feature_sets(feature_set_id),
    station_id text not null references public.stations(station_id),
    generated_at timestamptz not null,
    target_at timestamptz not null,
    horizon_minutes smallint not null check (horizon_minutes > 0),
    predicted_demand double precision not null check (predicted_demand >= 0),
    actual_demand double precision,
    unique (run_id, station_id, target_at, horizon_minutes)
);

create table if not exists public.metrics (
    metric_id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.pipeline_runs(run_id),
    model_id uuid references public.models(model_id),
    station_id text references public.stations(station_id),
    metric_name text not null,
    window_start timestamptz not null,
    window_end timestamptz not null,
    metric_value double precision not null,
    check (window_end >= window_start)
);

create table if not exists public.drift_events (
    drift_id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.pipeline_runs(run_id),
    station_id text references public.stations(station_id),
    detected_at timestamptz not null default now(),
    drift_type text not null check (drift_type in ('data', 'concept', 'performance')),
    feature_name text,
    score double precision not null,
    threshold double precision not null,
    severity text not null check (severity in ('low', 'medium', 'high')),
    action_taken text
);

create index if not exists idx_station_demand_time
    on public.station_demand (observed_at);
create index if not exists idx_predictions_target
    on public.predictions (target_at, station_id);
create index if not exists idx_metrics_window
    on public.metrics (window_start, window_end);
create index if not exists idx_drift_detected
    on public.drift_events (detected_at, severity);

alter table public.dataset_catalog enable row level security;
alter table public.stations enable row level security;
alter table public.time_intervals enable row level security;
alter table public.context_observations enable row level security;
alter table public.station_demand enable row level security;
alter table public.predictions enable row level security;
alter table public.metrics enable row level security;

drop policy if exists "public read dataset catalog" on public.dataset_catalog;
drop policy if exists "public read time intervals" on public.time_intervals;
drop policy if exists "public read stations" on public.stations;
drop policy if exists "public read demand" on public.station_demand;
drop policy if exists "public read context" on public.context_observations;
drop policy if exists "public read predictions" on public.predictions;
drop policy if exists "public read metrics" on public.metrics;

create policy "public read dataset catalog" on public.dataset_catalog for select using (true);
create policy "public read time intervals" on public.time_intervals for select using (true);
create policy "public read stations" on public.stations for select using (true);
create policy "public read demand" on public.station_demand for select using (true);
create policy "public read context" on public.context_observations for select using (true);
create policy "public read predictions" on public.predictions for select using (true);
create policy "public read metrics" on public.metrics for select using (true);