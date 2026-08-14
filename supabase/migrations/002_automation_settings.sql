-- Run this in Supabase SQL Editor after 001_managed_app_storage.sql.
create table if not exists public.automation_settings (
    user_id uuid primary key references auth.users(id) on delete cascade default auth.uid(),
    autonomous_paper_orders boolean not null default false,
    max_order_notional numeric not null default 25 check (max_order_notional between 1 and 25),
    updated_at timestamptz not null default now()
);

alter table public.automation_settings enable row level security;
create policy "users manage own automation settings"
on public.automation_settings for all
using (auth.uid() = user_id) with check (auth.uid() = user_id);
