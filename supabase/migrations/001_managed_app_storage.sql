-- Run this once in Supabase: SQL Editor > New query > paste and Run.
-- Every table is private to the signed-in user through row-level security.

create table if not exists public.backtest_runs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
    saved_at timestamptz not null default now(),
    ticker text not null,
    strategy text not null,
    parameters jsonb not null,
    initial_cash numeric not null,
    cost_bps numeric not null,
    metrics jsonb not null
);

create table if not exists public.paper_trades (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
    run_id uuid references public.backtest_runs(id) on delete cascade,
    trade jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists public.watchlists (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
    name text not null,
    source text not null,
    minimum_score numeric not null check (minimum_score between 0 and 100),
    symbols jsonb not null,
    updated_at timestamptz not null default now(),
    unique (user_id, name)
);

create table if not exists public.screen_results (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
    source text not null,
    ranking jsonb not null,
    created_at timestamptz not null default now()
);

-- Encrypted blobs only: encryption/decryption happens in the server app with
-- APP_ENCRYPTION_KEY, which must never be sent to the browser or database.
create table if not exists public.encrypted_credentials (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
    provider text not null,
    encrypted_value text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (user_id, provider)
);

alter table public.backtest_runs enable row level security;
alter table public.paper_trades enable row level security;
alter table public.watchlists enable row level security;
alter table public.screen_results enable row level security;
alter table public.encrypted_credentials enable row level security;

create policy "users manage own backtests" on public.backtest_runs for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "users manage own paper trades" on public.paper_trades for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "users manage own watchlists" on public.watchlists for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "users manage own screen results" on public.screen_results for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "users manage own encrypted credentials" on public.encrypted_credentials for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
