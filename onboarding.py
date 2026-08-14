"""Non-advisory research presets derived from onboarding choices."""

from __future__ import annotations

from typing import Any


def build_research_preset(markets: list[str], goal: str, risk: str, experience: str, monitoring: str) -> dict[str, Any]:
    """Return an editable research starting point, never investment advice."""
    chosen_markets = sorted(set(markets))
    if not chosen_markets:
        raise ValueError("Choose at least one market to research.")
    if risk not in {"Conservative", "Balanced", "Growth"}:
        raise ValueError("Choose a supported risk preference.")

    risk_controls = {
        "Conservative": {"position_size_pct": 5, "stop_loss_pct": 3.0, "daily_loss_pct": 1.0},
        "Balanced": {"position_size_pct": 10, "stop_loss_pct": 5.0, "daily_loss_pct": 2.0},
        "Growth": {"position_size_pct": 15, "stop_loss_pct": 8.0, "daily_loss_pct": 3.0},
    }[risk]
    active_goal = goal in {"Active/day research", "Both"}
    crypto_focus = "Crypto" in chosen_markets and (chosen_markets == ["Crypto"] or active_goal)
    if crypto_focus:
        strategy = "Crypto AI direction model"
        symbols = ["BTC-USD", "ETH-USD"]
        rationale = "Start with 24/7 crypto research using confirmation and trend filters."
    elif active_goal:
        strategy = "Day-trading AI direction model"
        symbols = ["SPY", "QQQ"] if "ETFs" in chosen_markets else ["AAPL", "MSFT"]
        rationale = "Start with liquid symbols and intraday research; validate before paper orders."
    else:
        strategy = "Moving-average crossover"
        symbols = ["SPY", "VTI"] if "ETFs" in chosen_markets else ["AAPL", "MSFT"]
        rationale = "Start with a slower long-term research strategy and compare it with buy-and-hold."
    return {
        "markets": chosen_markets,
        "goal": goal,
        "risk": risk,
        "experience": experience,
        "monitoring": monitoring,
        "recommended_strategy": strategy,
        "starter_symbols": symbols,
        "rationale": rationale,
        **risk_controls,
    }
