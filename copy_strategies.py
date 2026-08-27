"""Transparent, paper-only strategy profiles and allocation proposals.

These profiles are product research templates, not investment advice, public
trader profiles, or a mechanism for submitting broker orders.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


STRATEGY_CATALOG: dict[str, dict[str, Any]] = {
    "conservative_etf_growth": {
        "name": "Conservative ETF Growth", "style": "Long-term diversified ETF research",
        "risk_level": "Lower relative volatility", "rebalance": "Quarterly proposal",
        "holdings": {"VTI": 0.45, "BND": 0.35, "VXUS": 0.20},
        "explanation": "A transparent diversified research allocation across US stocks, bonds, and international stocks. It is not a guarantee of lower losses.",
    },
    "btc_trend_candles": {
        "name": "BTC Trend + Candlestick Confirmation", "style": "Crypto research",
        "risk_level": "High volatility", "rebalance": "Signal review on completed five-minute bars",
        "holdings": {"BTC/USD": 1.0},
        "explanation": "Uses the app's BTC trend and bullish-candlestick research conditions. Crypto can move sharply and this profile can remain uninvested when its research conditions are not met.",
    },
    "ai_day_research": {
        "name": "AI Day-Trading Research", "style": "Short-horizon stock and ETF research",
        "risk_level": "High turnover and high risk", "rebalance": "Intraday proposal only",
        "holdings": {"SPY": 0.40, "QQQ": 0.35, "IWM": 0.25},
        "explanation": "A liquid-ETF research basket for evaluating the intraday model. It is not an instruction to day trade and no strategy result is guaranteed.",
    },
    "momentum_etf_basket": {
        "name": "Momentum ETF Basket", "style": "Medium-term ETF research",
        "risk_level": "Moderate to high volatility", "rebalance": "Monthly proposal",
        "holdings": {"SPY": 0.35, "QQQ": 0.30, "IWM": 0.20, "GLD": 0.15},
        "explanation": "A diversified liquid-ETF basket intended for walk-forward testing of momentum and trend research before any paper allocation is considered.",
    },
}


def strategy_profiles() -> list[dict[str, Any]]:
    """Return catalog rows suitable for a profile picker without performance claims."""
    return [
        {"Strategy ID": strategy_id, "Strategy": profile["name"], "Style": profile["style"],
         "Risk": profile["risk_level"], "Rebalance review": profile["rebalance"],
         "Holdings": ", ".join(f"{symbol} {weight:.0%}" for symbol, weight in profile["holdings"].items()),
         "Track record": "Paper tracking begins when followed; validate with walk-forward research."}
        for strategy_id, profile in STRATEGY_CATALOG.items()
    ]


def get_strategy_profile(strategy_id: str) -> dict[str, Any]:
    """Return one known profile or raise a user-facing validation error."""
    try:
        return STRATEGY_CATALOG[strategy_id]
    except KeyError as error:
        raise ValueError("Choose a strategy from the paper-only catalog.") from error


def build_rebalance_proposal(strategy_id: str, allocation_amount: float, risk_cap_pct: float,
                             paper_equity: float, current_positions: pd.DataFrame) -> pd.DataFrame:
    """Build a review-only target allocation; never submits or queues an order."""
    if allocation_amount <= 0 or not 0 < risk_cap_pct <= 1 or paper_equity <= 0:
        raise ValueError("Allocation, risk cap, and paper equity must all be positive.")
    profile = get_strategy_profile(strategy_id)
    effective_budget = min(float(allocation_amount), float(paper_equity) * float(risk_cap_pct))
    current_values: dict[str, float] = {}
    if not current_positions.empty and {"Symbol", "Market Value"}.issubset(current_positions.columns):
        values = current_positions[["Symbol", "Market Value"]].copy()
        values["Market Value"] = pd.to_numeric(values["Market Value"], errors="coerce").fillna(0.0)
        current_values = values.groupby("Symbol")["Market Value"].sum().astype(float).to_dict()
    rows = []
    for symbol, weight in profile["holdings"].items():
        target_value = effective_budget * float(weight)
        current_value = float(current_values.get(symbol, 0.0))
        difference = target_value - current_value
        rows.append({"Symbol": symbol, "Target weight": float(weight), "Target value": target_value,
                     "Current value": current_value, "Difference": difference,
                     "Paper action": "Review paper buy" if difference > 1 else ("No action" if difference >= -1 else "Review only — do not auto-sell")})
    return pd.DataFrame(rows)
