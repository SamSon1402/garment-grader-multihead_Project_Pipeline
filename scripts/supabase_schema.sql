-- Minimal event table for the optional Supabase sink.
-- In a real deployment use least-privilege RLS and keep service-role credentials
-- only on the trusted backend/edge service, never in a browser.

create table if not exists public.garment_events (
  event_id text primary key,
  garment_id text not null,
  customer text not null,
  policy_version integer not null,
  timestamp timestamptz not null,
  frame_quality jsonb not null,
  nir jsonb not null,
  vision jsonb not null,
  grade jsonb not null,
  price jsonb not null,
  route jsonb not null,
  projection jsonb not null,
  total_latency_ms double precision not null,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists garment_events_customer_ts_idx
  on public.garment_events(customer, timestamp desc);
