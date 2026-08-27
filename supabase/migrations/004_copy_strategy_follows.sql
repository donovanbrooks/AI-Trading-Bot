-- Run once in Supabase after the earlier migrations.
-- This saves only a user's paper-only strategy-follow preferences.

create table if not exists public.copy_strategy_follows (
    user_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
    strategy_id text not null,
    allocation_amount numeric not null check (allocation_amount between 1 and 1000000),
    risk_cap_pct numeric not null check (risk_cap_pct > 0 and risk_cap_pct <= 1),
    paused boolean not null default false,
    updated_at timestamptz not null default now(),
    primary key (user_id, strategy_id)
);

alter table public.copy_strategy_follows enable row level security;

create policy "users manage own copy strategy follows"
on public.copy_strategy_follows for all
using (auth.uid() = user_id) with check (auth.uid() = user_id);
