"""Daily bounded walk-forward strategy validation for opted-in paper users.

This job performs research only. It has no broker-order imports and never
submits, queues, changes, or cancels an order.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from credentials import CredentialError, decrypt_secret
from market_data import load_alpaca_bars
from strategy_search import evaluate_candidates
from validation import BacktestConfig, build_crossover_signals


MAX_SYMBOLS_PER_USER = 2
VARIATIONS = ((10, 50), (20, 50), (20, 100), (50, 200))


def build_daily_candidates(prices: pd.DataFrame) -> list[tuple[dict[str, object], pd.DataFrame]]:
    """Return the four fixed, disclosed candidate variations for one symbol."""
    return [
        ({"Short MA": short, "Long MA": long}, build_crossover_signals(prices, short, long))
        for short, long in VARIATIONS
        if long < len(prices)
    ]


def evaluate_symbol(symbol: str, prices: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, object]]:
    """Rank fixed daily variations on contiguous out-of-sample folds."""
    candidates = build_daily_candidates(prices)
    config = BacktestConfig(
        initial_cash=10_000,
        trading_cost_bps=5,
        position_size_pct=float(profile.get("position_size_pct", 10)) / 100,
        max_daily_loss_pct=float(profile.get("daily_loss_pct", 2)) / 100,
        stop_loss_pct=float(profile.get("stop_loss_pct", 5)) / 100,
    )
    report = evaluate_candidates(candidates, config, periods_per_year=252, folds=3, min_trades=1)
    if report.empty:
        return [{"Symbol": symbol, "Candidates tested": len(candidates), "Status": "No variation met the minimum out-of-sample trade requirement."}]
    return [
        {"Symbol": symbol, "Candidates tested": len(candidates), "Rank": int(index + 1), **row}
        for index, row in report.head(3).iterrows()
    ]


def _symbols_for_user(profile: dict[str, Any], watchlists: list[dict[str, Any]]) -> list[str]:
    symbols = [str(symbol).upper() for symbol in profile.get("starter_symbols", [])]
    for watchlist in watchlists:
        symbols.extend(str(item.get("symbol", "")).upper() for item in watchlist.get("symbols", []))
    return list(dict.fromkeys(symbol for symbol in symbols if symbol))[:MAX_SYMBOLS_PER_USER]


def main() -> None:
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    profiles = client.table("user_research_profiles").select("user_id,profile").execute().data
    active_profiles = [row for row in profiles if row.get("profile", {}).get("scheduled_strategy_testing_enabled")]
    if not active_profiles:
        print("No users have enabled daily strategy validation.")
        return
    credentials = client.table("encrypted_credentials").select("user_id,encrypted_value").eq("provider", "alpaca_paper").execute().data
    credentials_by_user = {row["user_id"]: row["encrypted_value"] for row in credentials}
    for row in active_profiles:
        user_id, profile = row["user_id"], row["profile"]
        encrypted = credentials_by_user.get(user_id)
        if not encrypted:
            print(f"Skipping {user_id}: no paper connection.")
            continue
        try:
            connection = json.loads(decrypt_secret(encrypted))
        except (CredentialError, json.JSONDecodeError):
            print(f"Skipping {user_id}: credentials could not be decrypted.")
            continue
        watchlists = client.table("watchlists").select("symbols").eq("user_id", user_id).execute().data
        results: list[dict[str, object]] = []
        for symbol in _symbols_for_user(profile, watchlists):
            try:
                crypto = "/" in symbol or symbol.endswith("-USD")
                prices = load_alpaca_bars(symbol, "2y", intraday=False, crypto=crypto,
                                          api_key=connection["api_key"], api_secret=connection["api_secret"])
                results.extend(evaluate_symbol(symbol, prices, profile))
            except Exception as error:
                print(f"{user_id} {symbol}: {error}")
        if results:
            client.table("screen_results").insert({"user_id": user_id, "source": "scheduled_strategy_tests", "ranking": results}).execute()
            print(f"Saved {len(results)} daily strategy-test result(s) for {user_id}.")
        else:
            print(f"No daily strategy-test results for {user_id}.")


if __name__ == "__main__":
    main()
