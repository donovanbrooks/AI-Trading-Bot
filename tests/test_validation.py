import pandas as pd

from ai_validation import generate_ai_signals
from validation import BacktestConfig, calculate_metrics, run_backtest, walk_forward_validate


def price_data(days=320):
    index = pd.date_range("2024-01-01", periods=days, freq="B")
    prices = pd.Series(range(100, 100 + days), index=index, dtype=float)
    return pd.DataFrame({"Open": prices, "High": prices + 1, "Low": prices - 1, "Close": prices, "Volume": 1_000}, index=index)


def test_signal_is_executed_on_following_day():
    data = price_data(10)
    data["Execution Signal"] = [0, 1] + [0] * 8
    results, trades = run_backtest(data, BacktestConfig(initial_cash=1_000, trading_cost_bps=0))

    assert trades.iloc[0]["Entry Date"] == data.index[1]
    assert trades.iloc[0]["Exit Reason"] == "End of test"
    assert results["Equity"].iloc[-1] > 1_000


def test_walk_forward_returns_out_of_sample_windows():
    data = price_data()
    reports = walk_forward_validate(data, 5, 20, BacktestConfig(), train_bars=100, test_bars=50)

    assert not reports.empty
    assert reports["Test Start"].min() > data.index[99]


def test_metrics_has_risk_fields():
    data = price_data(10)
    data["Execution Signal"] = [0, 1] + [0] * 8
    results, trades = run_backtest(data, BacktestConfig())
    metrics = calculate_metrics(results, trades, 10_000)

    assert {"sharpe_ratio", "max_drawdown", "profit_factor"}.issubset(metrics)


def test_ai_signal_generation_does_not_fill_before_training_period():
    data = price_data(360)
    # Alternating movement ensures both target classes are available to train.
    data["Close"] = [100 + index + (2 if index % 2 else 0) for index in range(len(data))]
    data["Open"] = data["Close"]
    signals = generate_ai_signals(data, train_bars=100, retrain_bars=30)

    assert signals["AI Probability"].iloc[:100].isna().all()
    assert signals["AI Probability"].notna().any()
