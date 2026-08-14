-- Run in Supabase SQL Editor after 001 and 002.
create table if not exists public.user_research_profiles (
    user_id uuid primary key references auth.users(id) on delete cascade default auth.uid(),
    profile jsonb not null,
    updated_at timestamptz not null default now()
);

alter table public.user_research_profiles enable row level security;
create policy "users manage own research profile"
on public.user_research_profiles for all
using (auth.uid() = user_id) with check (auth.uid() = user_id);
