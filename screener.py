"""Transparent research ranking for a small, liquid US stock/ETF universe.

The output is a research shortlist, not investment advice or a prediction of
future returns.  Scores are derived only from supplied historical OHLCV data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


US_STOCK_UNIVERSE = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "BRK.B",
    "LLY", "V", "WMT", "XOM", "COST", "UNH", "MA", "HD", "PG", "JNJ",
)
US_ETF_UNIVERSE = (
    "SPY", "IVV", "VOO", "QQQ", "VTI", "IWM", "DIA", "XLK", "XLF", "XLE",
    "XLV", "XLY", "XLP", "GLD", "TLT", "AGG", "VNQ", "EFA", "EEM", "SCHD",
)


def _percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = values.rank(pct=True, method="average")
    return ranked if higher_is_better else 1 - ranked


def rank_research_universe(price_history: dict[str, pd.DataFrame], asset_type: str, top_n: int = 10) -> pd.DataFrame:
    """Rank sufficiently liquid symbols by trend, momentum, risk, and liquidity.

    At least 201 daily rows are required per symbol so the long-term trend and
    drawdown can be measured consistently.  The score is intentionally
    explainable rather than an opaque recommendation.
    """
    rows: list[dict[str, float | str]] = []
    for symbol, data in price_history.items():
        required = {"Close", "Volume"}
        if not required.issubset(data.columns) or len(data) < 201:
            continue
        frame = data.dropna(subset=["Close", "Volume"]).copy()
        if len(frame) < 201 or (frame["Close"] <= 0).any():
            continue
        close = frame["Close"].astype(float)
        daily_returns = close.pct_change().dropna()
        return_3m = float(close.iloc[-1] / close.iloc[-64] - 1)
        return_12m = float(close.iloc[-1] / close.iloc[-201] - 1)
        ma_50 = float(close.tail(50).mean())
        ma_200 = float(close.tail(200).mean())
        volatility = float(daily_returns.tail(63).std() * np.sqrt(252))
        drawdown = close / close.cummax() - 1
        max_drawdown = float(drawdown.tail(252).min())
        dollar_volume = float((close * frame["Volume"].astype(float)).tail(20).median())
        rows.append(
            {
                "Symbol": symbol,
                "Type": asset_type,
                "3-month return": return_3m,
                "12-month return": return_12m,
                "Trend": float(close.iloc[-1] / ma_200 - 1),
                "Volatility": volatility,
                "1-year drawdown": max_drawdown,
                "Median daily dollar volume": dollar_volume,
                "Above 50-day MA": "Yes" if close.iloc[-1] > ma_50 else "No",
            }
        )
    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Higher recent/long-term returns and trend are good; lower volatility and
    # shallower drawdowns are good. Liquidity is a modest eligibility signal.
    result["Research score"] = 100 * (
        0.25 * _percentile(result["3-month return"])
        + 0.30 * _percentile(result["12-month return"])
        + 0.20 * _percentile(result["Trend"])
        + 0.15 * _percentile(result["Volatility"], higher_is_better=False)
        + 0.07 * _percentile(result["1-year drawdown"])
        + 0.03 * _percentile(result["Median daily dollar volume"])
    )
    result["Research score"] = result["Research score"].round(1)
    result["Why it ranked"] = result.apply(
        lambda row: (
            f"{row['3-month return']:.1%} over 3 months, {row['12-month return']:.1%} over 12 months; "
            f"{row['Volatility']:.1%} annualized volatility and {row['1-year drawdown']:.1%} one-year drawdown."
        ),
        axis=1,
    )
    return result.sort_values(["Research score", "Symbol"], ascending=[False, True]).head(top_n).reset_index(drop=True)
