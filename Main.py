import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

print("Trading bot setup successful!")

# Download Apple stock data
stock = yf.download("AAPL", period="1y")

print(stock.head())

# Show stock chart
stock["Close"].plot(title="Apple Stock Price")
plt.show()