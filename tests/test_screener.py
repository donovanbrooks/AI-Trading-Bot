import numpy as np
import pandas as pd

from screener import rank_research_universe


def _prices(multiplier: float) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=220, freq="B")
    close = 100 * np.cumprod(np.full(len(index), multiplier))
    return pd.DataFrame({"Close": close, "Volume": 1_000_000}, index=index)


def test_research_screener_ranks_stronger_trend_first():
    ranked = rank_research_universe({"SLOW": _prices(1.0001), "FAST": _prices(1.001)}, "Stock")

    assert list(ranked["Symbol"]) == ["FAST", "SLOW"]
    assert {"Research score", "Why it ranked", "Volatility"}.issubset(ranked.columns)


def test_research_screener_ignores_insufficient_history():
    short = _prices(1.001).iloc[:100]
    assert rank_research_universe({"SHORT": short}, "ETF").empty
