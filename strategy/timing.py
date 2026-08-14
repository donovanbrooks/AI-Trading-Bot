"""Present the latest model output as a cautious research assessment."""

from __future__ import annotations

import pandas as pd


def latest_ai_assessment(signals: pd.DataFrame) -> dict[str, object]:
    """Turn the latest completed model bar into a non-advisory entry assessment."""
    predicted = signals.dropna(subset=["AI Probability"])
    if predicted.empty:
        return {"status": "No model reading", "probability": None, "reason": "Not enough completed model history yet."}
    row = predicted.iloc[-1]
    probability = float(row["AI Probability"])
    if int(row["Signal"]) == 1:
        return {"status": "Bullish entry setup", "probability": probability, "reason": "The model and all enabled entry filters support a bullish research setup."}
    if probability <= 0.5:
        return {"status": "Avoid new buy", "probability": probability, "reason": "The model is not bullish on the latest completed bar."}
    return {"status": "Wait for confirmation", "probability": probability, "reason": "The model is somewhat positive, but an enabled trend or candlestick filter did not confirm an entry."}


def rank_intraday_assessments(assessments: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Return a transparent current ranking; not a recommendation list."""
    rows = [{"Symbol": symbol, "AI timing status": value["status"], "AI bullish probability": value["probability"], "Why": value["reason"]} for symbol, value in assessments.items()]
    return pd.DataFrame(rows).sort_values("AI bullish probability", ascending=False, na_position="last").reset_index(drop=True) if rows else pd.DataFrame()
