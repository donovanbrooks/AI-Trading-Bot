import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from storage import list_recent_runs, save_backtest_run
from strategy.ai_validation import generate_ai_signals
from strategy.intraday_ai_validation import bullish_candlestick_patterns, generate_intraday_ai_signals
from strategy.crypto_ai_validation import generate_crypto_ai_signals
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


def test_saved_backtest_persists_run_and_trades():
    data = price_data(10)
    data["Execution Signal"] = [0, 1] + [0] * 8
    results, trades = run_backtest(data, BacktestConfig(initial_cash=1_000))
    metrics = calculate_metrics(results, trades, 1_000)

    with TemporaryDirectory() as directory:
        database = Path(directory) / "trading_bot.db"
        run_id = save_backtest_run("TEST", "Test", {}, 1_000, 5, metrics, trades, database)
        saved_runs = list_recent_runs(database_path=database)

    assert run_id == 1
    assert saved_runs.iloc[0]["Ticker"] == "TEST"
    assert saved_runs.iloc[0]["Trades"] == len(trades)


def test_intraday_ai_uses_regular_session_and_forces_daily_exit():
    sessions = [
        pd.date_range(f"{day.date()} 09:30", f"{day.date()} 16:00", freq="5min", tz="America/New_York")
        for day in pd.date_range("2026-01-05", periods=12, freq="B")
    ]
    index = sessions[0].append(sessions[1:])
    prices = 100 + np.cumsum(np.sin(np.arange(len(index))) / 5)
    data = pd.DataFrame(
        {"Open": prices, "High": prices + 0.1, "Low": prices - 0.1, "Close": prices, "Volume": 1_000 + np.arange(len(index))},
        index=index,
    )

    signals = generate_intraday_ai_signals(data, train_bars=390, retrain_bars=78)

    assert all("09:35:00" <= value.isoformat() <= "15:55:00" for value in signals.index.time)
    assert signals[signals["Session Time"].astype(str) == "15:55:00"]["Execution Signal"].eq(-1).all()
    assert signals["AI Probability"].notna().any()


def test_bullish_candle_filter_identifies_engulfing_pattern():
    bars = pd.DataFrame({"Open": [10.0, 8.0], "High": [10.5, 11.0], "Low": [8.5, 7.5], "Close": [9.0, 10.5]})
    assert bullish_candlestick_patterns(bars).tolist() == [False, True]


def test_crypto_ai_generates_24_hour_predictions():
    index = pd.date_range("2026-01-01", periods=1_000, freq="5min", tz="UTC")
    prices = 100 + np.cumsum(np.random.default_rng(42).normal(0, 0.2, len(index)))
    data = pd.DataFrame(
        {"Open": prices, "High": prices + 0.1, "Low": prices - 0.1, "Close": prices, "Volume": 1_000 + np.arange(len(index))},
        index=index,
    )

    signals = generate_crypto_ai_signals(data, train_bars=500, retrain_bars=100)

    assert signals["AI Probability"].notna().any()
    assert signals.index.hour.nunique() == 24


def test_quality_gate_caps_position_size_and_daily_entries():
    data = price_data(4)
    data.index = pd.date_range("2024-01-01 10:00", periods=4, freq="5min")
    data[["Open", "High", "Low", "Close"]] = 200
    data["Execution Signal"] = [1, -1, 1, -1]
    config = BacktestConfig(initial_cash=1_000, trading_cost_bps=0, position_size_pct=0.1, max_trades_per_day=1)
    results, trades = run_backtest(data, config)

    assert len(trades) == 1
    assert results["Entry Blocked"].eq("Daily trade limit").sum() == 1
    assert trades.iloc[0]["Shares"] < 1


def test_simulated_stop_loss_exits_a_position():
    data = price_data(3)
    data.index = pd.date_range("2024-01-01 10:00", periods=3, freq="5min")
    data["Execution Signal"] = [1, 0, 0]
    data.loc[data.index[1], "Low"] = 90
    _, trades = run_backtest(data, BacktestConfig(initial_cash=1_000, trading_cost_bps=0, stop_loss_pct=0.05))

    assert trades.iloc[0]["Exit Reason"] == "Stop loss"
