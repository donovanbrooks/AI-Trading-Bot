import pandas as pd

from scripts.scheduled_strategy_tests import VARIATIONS, build_daily_candidates, evaluate_symbol


def _prices(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    close = pd.Series(range(100, 100 + rows), index=index, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 100}, index=index)


def test_daily_worker_uses_a_bounded_fixed_candidate_set():
    candidates = build_daily_candidates(_prices())

    assert len(candidates) == len(VARIATIONS)
    assert [candidate[0]["Long MA"] for candidate in candidates] == [50, 50, 100, 200]


def test_daily_worker_returns_research_rows_without_order_data():
    results = evaluate_symbol("SPY", _prices(), {"position_size_pct": 10, "daily_loss_pct": 2, "stop_loss_pct": 5})

    assert results
    assert all(row["Symbol"] == "SPY" for row in results)
    assert all("Order ID" not in row for row in results)
