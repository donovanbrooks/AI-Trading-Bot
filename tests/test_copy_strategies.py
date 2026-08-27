from pathlib import Path

import pandas as pd

from copy_strategies import build_rebalance_proposal, get_strategy_profile, strategy_profiles
from storage import list_copy_strategy_follows, save_copy_strategy_follow


def test_strategy_catalog_has_transparent_profiles():
    profiles = strategy_profiles()

    assert len(profiles) == 4
    assert get_strategy_profile("btc_trend_candles")["holdings"] == {"BTC/USD": 1.0}
    assert all("Paper tracking" in row["Track record"] for row in profiles)


def test_rebalance_proposal_respects_risk_cap_and_never_auto_sells():
    positions = pd.DataFrame({"Symbol": ["VTI"], "Market Value": [700.0]})

    proposal = build_rebalance_proposal("conservative_etf_growth", 2_000, 0.10, 1_000, positions)

    assert proposal["Target value"].sum() == 100.0
    assert proposal.loc[proposal["Symbol"] == "VTI", "Paper action"].iloc[0] == "Review only — do not auto-sell"
    assert (proposal["Paper action"] != "Auto-sell").all()


def test_copy_strategy_follow_persists_locally(tmp_path: Path):
    database = tmp_path / "follows.db"

    save_copy_strategy_follow("momentum_etf_basket", 250, 0.15, False, database)
    follows = list_copy_strategy_follows(database)

    assert follows.iloc[0]["Strategy ID"] == "momentum_etf_basket"
    assert follows.iloc[0]["Paper allocation"] == 250
    assert follows.iloc[0]["Risk cap"] == 0.15
    assert follows.iloc[0]["Paused"] == 0
