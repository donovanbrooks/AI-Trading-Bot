import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


from strategy.moving_average import moving_average_strategy
from backtests.backtest import backtest_strategy, buy_and_hold
from analysis.trade_analyzer import analyze_trades
from optimization.optimizer import optimize_strategy
from testing.walk_forward import walk_forward_test
from ai_strategy.ai_trading_strategy import ai_strategy

# Download Apple stock data
data = yf.download("AAPL", period="1y", auto_adjust=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Apply trading strategy
data = moving_average_strategy(data)

data["Signal"] = data["MA_Signal"]

# Display the latest data
print(data.tail())


# Plot stock price and moving averages
plt.figure(figsize=(12, 6))

plt.plot(data["Close"], label="AAPL Price")
plt.plot(data["Short_MA"], label="20 Day Moving Average")
plt.plot(data["Long_MA"], label="50 Day Moving Average")

plt.title("Moving Average Trading strategy")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.show()

backtest_strategy(data)

buy_and_hold(data)

analyze_trades()

best_strategy = optimize_strategy(data)

print("\nBest Strategy:")
print(best_strategy)

walk_forward_test(data)



