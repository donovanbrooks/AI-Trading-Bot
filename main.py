import yfinance as yf
import matplotlib.pyplot as plt

from strategy.moving_average import moving_average_strategy


# Download Apple stock data
data = yf.download("AAPL", period="1y")

# Apply trading strategy
data = moving_average_strategy(data)

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