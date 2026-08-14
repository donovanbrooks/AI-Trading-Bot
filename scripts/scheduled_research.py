"""Daily, alert-only research worker for GitHub Actions.

Requires GitHub Actions secrets for Supabase and the server encryption key.
It never imports order-submission functions or creates broker orders.
"""

from __future__ import annotations

import json
import os

from credentials import CredentialError, decrypt_secret
from market_data import load_alpaca_bars
from strategy.crypto_ai_validation import generate_crypto_ai_signals
from strategy.intraday_ai_validation import generate_intraday_ai_signals
from strategy.timing import latest_ai_assessment


def main() -> None:
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    service_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    client = create_client(url, service_key)
    profiles = client.table("user_research_profiles").select("user_id,profile").execute().data
    active_profiles = [row for row in profiles if row.get("profile", {}).get("scheduled_research_enabled")]
    if not active_profiles:
        print("No users have enabled scheduled research.")
        return
    credentials = client.table("encrypted_credentials").select("user_id,encrypted_value").eq("provider", "alpaca_paper").execute().data
    credentials_by_user = {row["user_id"]: row["encrypted_value"] for row in credentials}
    for profile in active_profiles:
        user_id = profile["user_id"]
        encrypted = credentials_by_user.get(user_id)
        if not encrypted:
            print(f"Skipping {user_id}: no paper connection.")
            continue
        try:
            credential_payload = json.loads(decrypt_secret(encrypted))
        except (CredentialError, json.JSONDecodeError):
            print(f"Skipping {user_id}: credentials could not be decrypted.")
            continue
        watchlists = client.table("watchlists").select("symbols").eq("user_id", user_id).execute().data
        symbols = []
        for watchlist in watchlists:
            symbols.extend(item.get("symbol", "") for item in watchlist.get("symbols", []))
        events = []
        for symbol in list(dict.fromkeys(value.upper() for value in symbols if value))[:5]:
            crypto = "/" in symbol or symbol.endswith("-USD")
            try:
                bars = load_alpaca_bars(symbol, "30d", intraday=True, crypto=crypto, api_key=credential_payload["api_key"], api_secret=credential_payload["api_secret"])
                signals = generate_crypto_ai_signals(bars, train_bars=1_008, require_bullish_candle=True, require_trend_filter=True) if crypto else generate_intraday_ai_signals(bars, train_bars=390, require_bullish_candle=True)
                assessment = latest_ai_assessment(signals)
                if assessment["status"] == "Bullish entry setup":
                    events.append({"Severity": "Info", "Type": "Bullish AI research setup", "Symbol": symbol, "Message": assessment["reason"]})
            except Exception as error:
                print(f"{user_id} {symbol}: {error}")
        if events:
            client.table("screen_results").insert({"user_id": user_id, "source": "scheduled_research_alerts", "ranking": events}).execute()
            print(f"Saved {len(events)} alert(s) for {user_id}.")
        else:
            print(f"No bullish research alerts for {user_id}.")


if __name__ == "__main__":
    main()
