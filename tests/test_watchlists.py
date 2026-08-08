from pathlib import Path

import pandas as pd

from storage import list_watchlists, research_alerts, save_watchlist


def test_saved_watchlist_creates_current_score_alert(tmp_path: Path):
    database = tmp_path / "watchlists.db"
    save_watchlist("Quality ideas", "US ETFs", ["SPY", "QQQ"], "ETF", 75, database)
    ranking = pd.DataFrame({
        "Symbol": ["SPY", "QQQ"], "Research score": [80.0, 70.0], "Why it ranked": ["Trend", "Trend"],
    })

    alerts = research_alerts(ranking, database)

    assert list(alerts["Symbol"]) == ["SPY"]
    assert list_watchlists(database).iloc[0]["Symbols"] == "QQQ, SPY"


def test_saving_a_watchlist_with_same_name_replaces_symbols(tmp_path: Path):
    database = tmp_path / "watchlists.db"
    save_watchlist("Ideas", "US stocks", ["AAPL"], "Stock", 60, database)
    save_watchlist("Ideas", "US stocks", ["MSFT"], "Stock", 70, database)

    watchlists = list_watchlists(database)

    assert len(watchlists) == 1
    assert watchlists.iloc[0]["Symbols"] == "MSFT"
    assert watchlists.iloc[0]["Alert score"] == 70
