import pandas as pd

from portfolio import PortfolioConfig, run_portfolio_backtest


def signal_frame(index, signals, opens, probabilities):
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [value + 1 for value in opens],
            "Low": [value - 1 for value in opens],
            "Close": opens,
            "Volume": [1_000] * len(index),
            "Execution Signal": signals,
            "AI Probability": probabilities,
        },
        index=index,
    )


def test_portfolio_ranks_concurrent_entries_and_caps_positions():
    index = pd.date_range("2025-01-01", periods=2, freq="B")
    signals = {
        "AAA": signal_frame(index, [1, -1], [100, 110], [0.80, 0.5]),
        "BBB": signal_frame(index, [1, -1], [100, 90], [0.60, 0.5]),
    }
    config = PortfolioConfig(initial_cash=1_000, position_size_pct=0.5, total_exposure_cap=0.5, max_positions=1)

    _, trades = run_portfolio_backtest(signals, config)

    assert len(trades) == 1
    assert trades.iloc[0]["Ticker"] == "AAA"


def test_portfolio_daily_loss_lockout_blocks_new_entry():
    index = pd.date_range("2025-01-01 10:00", periods=3, freq="5min")
    signals = {
        "AAA": signal_frame(index, [1, -1, 0], [100, 90, 90], [0.7, 0.5, 0.5]),
        "BBB": signal_frame(index, [0, 1, -1], [100, 100, 100], [0.5, 0.7, 0.5]),
    }
    config = PortfolioConfig(
        initial_cash=1_000,
        trading_cost_bps=0,
        position_size_pct=1.0,
        total_exposure_cap=1.0,
        max_daily_loss_pct=0.01,
    )

    results, trades = run_portfolio_backtest(signals, config)

    assert len(trades) == 1
    assert trades.iloc[0]["Ticker"] == "AAA"
    assert results["Blocked Entries"].sum() >= 1


def test_portfolio_sizes_new_entries_from_current_equity_not_starting_cash():
    index = pd.date_range("2025-01-01", periods=3, freq="B")
    signals = {
        "AAA": signal_frame(index, [1, -1, 0], [100, 50, 50], [0.8, 0.5, 0.5]),
        "BBB": signal_frame(index, [0, 0, 1], [100, 50, 50], [0.5, 0.5, 0.8]),
    }
    config = PortfolioConfig(initial_cash=1_000, trading_cost_bps=0, position_size_pct=0.5, total_exposure_cap=1.0, stop_loss_pct=0.9, max_drawdown_pct=0.9)

    _, trades = run_portfolio_backtest(signals, config)

    assert trades.loc[trades["Ticker"] == "BBB", "Shares"].iloc[0] == 7.5
