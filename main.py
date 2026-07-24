import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from strategy.moving_average import moving_average_strategy
from backtests.backtest import backtest_strategy, buy_and_hold
from analysis.trade_analyzer import analyze_trades
from ai_strategy.ai_trading_strategy import ai_strategy
from strategy.ensemble_strategy import ensemble_strategy
from indicators.atr import calculate_atr


# ==========================
# Download stock data
# ==========================

def get_stock_data(ticker):

    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    return data



# ==========================
# Run AI Trading Bot
# ==========================

def run_bot(ticker):

    print("\n====================")
    print(f"Testing {ticker}")
    print("====================")


    data = get_stock_data(ticker)


    # Indicators
    data = calculate_atr(data)


    # Moving Average Strategy
    data = moving_average_strategy(data)


    # AI compatibility
    data["Signal"] = data["MA_Signal"]


    # Train/Test split
    split = int(len(data) * 0.8)

    train_data = data.iloc[:split].copy()
    test_data = data.iloc[split:].copy()


    # AI prediction
    test_data = ai_strategy(
        train_data,
        test_data
    )


    # Ensemble
    test_data = ensemble_strategy(test_data)


    print(test_data.tail())

    # Performance testing
    performance = backtest_strategy(test_data)

    buy_hold = buy_and_hold(test_data)

    trade_results = analyze_trades()

    return {
        "Ticker": ticker,
        "Profit": performance["Profit"],
        "Return": performance["Return"],
        "Trades": performance["Trades"],
        "Win Rate": performance["Win Rate"],
        "Buy & Hold Return": ((buy_hold - 1000) / 1000) * 100
    }

# ==========================
# Test Multiple Stocks
# ==========================

stocks = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "META",
    "SPY"
]


results = []

for ticker in stocks:
    result = run_bot(ticker)
    results.append(result)

    results_df = pd.DataFrame(results)

    print("\n====================")
    print("FINAL STOCK RANKINGS")
    print("====================")

    print(results_df)

    print("\nSorted by Return:")

    print("Columns:")
    print(results_df.columns.tolist())

    print("\nFirst Result:")
    print(results[0])

    print(
        results_df.sort_values(
            by="Profit",
            ascending=False
        )
    )
