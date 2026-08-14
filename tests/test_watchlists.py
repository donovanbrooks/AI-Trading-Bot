from pathlib import Path

import pandas as pd

from onboarding import build_research_preset
from storage import get_automation_settings, get_research_profile, list_watchlist_symbols, list_watchlists, research_alerts, save_automation_settings, save_research_profile, save_watchlist


def test_saved_watchlist_creates_current_score_alert(tmp_path: Path):
    database = tmp_path / "watchlists.db"
    save_watchlist("Quality ideas", "US ETFs", ["SPY", "QQQ"], "ETF", 75, database)
    ranking = pd.DataFrame({
        "Symbol": ["SPY", "QQQ"], "Research score": [80.0, 70.0], "Why it ranked": ["Trend", "Trend"],
    })

    alerts = research_alerts(ranking, database)

    assert list(alerts["Symbol"]) == ["SPY"]
    assert list_watchlists(database).iloc[0]["Symbols"] == "QQQ, SPY"
    assert set(list_watchlist_symbols(database)["Symbol"]) == {"SPY", "QQQ"}


def test_saving_a_watchlist_with_same_name_replaces_symbols(tmp_path: Path):
    database = tmp_path / "watchlists.db"
    save_watchlist("Ideas", "US stocks", ["AAPL"], "Stock", 60, database)
    save_watchlist("Ideas", "US stocks", ["MSFT"], "Stock", 70, database)

    watchlists = list_watchlists(database)

    assert len(watchlists) == 1
    assert watchlists.iloc[0]["Symbols"] == "MSFT"
    assert watchlists.iloc[0]["Alert score"] == 70


def test_automation_permission_defaults_off_and_persists(tmp_path: Path):
    database = tmp_path / "settings.db"
    assert get_automation_settings(database)["autonomous_paper_orders"] is False

    save_automation_settings(True, 12, database)

    assert get_automation_settings(database) == {"autonomous_paper_orders": True, "max_order_notional": 12.0}


def test_onboarding_profile_creates_an_editable_research_preset(tmp_path: Path):
    database = tmp_path / "profile.db"
    profile = build_research_preset(["Crypto"], "Active/day research", "Balanced", "New to investing", "Most days")
    save_research_profile(profile, database)

    saved = get_research_profile(database)
    assert saved is not None
    assert saved["recommended_strategy"] == "Crypto AI direction model"
    assert saved["position_size_pct"] == 10
