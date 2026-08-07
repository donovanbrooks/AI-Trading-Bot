"""Out-of-sample backtest summaries across broad market regimes."""

from __future__ import annotations

import pandas as pd


def regime_performance(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise strategy returns by trend and volatility regime."""
    frame = results.copy()
    frame["Return"] = frame["Equity"].pct_change().fillna(0)
    frame["Trend"] = (frame["Close"] >= frame["Close"].rolling(50, min_periods=10).mean()).map({True: "Uptrend", False: "Downtrend"})
    volatility = frame["Close"].pct_change().rolling(20, min_periods=5).std()
    frame["Volatility"] = (volatility >= volatility.median()).map({True: "High", False: "Low"})
    frame = frame.dropna(subset=["Trend", "Volatility"])
    if frame.empty:
        return pd.DataFrame()
    summary = frame.groupby(["Trend", "Volatility"], observed=True).agg(Bars=("Return", "size"), Strategy_Return=("Return", lambda values: (1 + values).prod() - 1), Average_Return=("Return", "mean"))
    return summary.reset_index()
