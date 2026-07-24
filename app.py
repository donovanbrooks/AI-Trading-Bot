"""Local dashboard for evaluating the trading strategy.

This app is intentionally research and paper-trading only.  It never submits
orders to a broker or stores credentials.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Trading Bot Lab", page_icon="📈", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    """Download daily OHLCV data and return a flat, clean dataframe."""
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna().copy()


def build_signals(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Generate a crossover signal at close, then execute it next session."""
    result = data.copy()
    result["Short MA"] = result["Close"].rolling(short_window).mean()
    result["Long MA"] = result["Close"].rolling(long_window).mean()
    regime = (result["Short MA"] > result["Long MA"]).astype(int)
    result["Signal"] = regime.diff().fillna(0).clip(-1, 1)
    # A signal is only known once the bar closes.  Executing it tomorrow
    # avoids using information that was unavailable at today's open.
    result["Execution Signal"] = result["Signal"].shift(1).fillna(0)
    return result


def run_backtest(data: pd.DataFrame, initial_cash: float, fee_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-only backtest using next-session opens and proportional fees."""
    cash, shares = initial_cash, 0.0
    trades: list[dict[str, object]] = []
    equity: list[float] = []

    for timestamp, row in data.iterrows():
        open_price = float(row["Open"])
        signal = int(row["Execution Signal"])
        if signal == 1 and shares == 0:
            shares = cash / (open_price * (1 + fee_rate))
            cost = shares * open_price * (1 + fee_rate)
            cash -= cost
            trades.append({"Date": timestamp, "Side": "BUY", "Price": open_price, "Shares": shares})
        elif signal == -1 and shares > 0:
            proceeds = shares * open_price * (1 - fee_rate)
            cash += proceeds
            trades.append({"Date": timestamp, "Side": "SELL", "Price": open_price, "Shares": shares})
            shares = 0.0
        equity.append(cash + shares * float(row["Close"]))

    result = data.copy()
    result["Equity"] = equity
    return result, pd.DataFrame(trades)


st.title("Trading Bot Lab")
st.caption("Research dashboard — backtesting only. It does not place real trades.")

with st.sidebar:
    st.header("Backtest settings")
    ticker = st.text_input("Ticker", "SPY").upper().strip()
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
    short_window = st.number_input("Short moving average", min_value=2, max_value=100, value=20)
    long_window = st.number_input("Long moving average", min_value=3, max_value=300, value=50)
    initial_cash = st.number_input("Starting cash ($)", min_value=100.0, value=10_000.0, step=100.0)
    fee_bps = st.number_input("Estimated fee + slippage (basis points)", min_value=0.0, value=5.0, step=1.0)

if short_window >= long_window:
    st.error("The short moving average must be smaller than the long moving average.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker}..."):
        prices = load_prices(ticker, period)
except Exception as error:
    st.error(f"Could not load market data for {ticker}: {error}")
    st.stop()

if prices.empty or len(prices) < long_window + 2:
    st.error("Not enough price history for those settings. Choose a longer history or shorter windows.")
    st.stop()

signals = build_signals(prices, int(short_window), int(long_window))
results, trades = run_backtest(signals, initial_cash, fee_bps / 10_000)
final_value = float(results["Equity"].iloc[-1])
strategy_return = final_value / initial_cash - 1
buy_hold_return = float(results["Close"].iloc[-1] / results["Close"].iloc[0] - 1)
drawdown = results["Equity"] / results["Equity"].cummax() - 1

one, two, three, four = st.columns(4)
one.metric("Portfolio value", f"${final_value:,.2f}")
two.metric("Strategy return", f"{strategy_return:.2%}")
three.metric("Buy & hold", f"{buy_hold_return:.2%}")
four.metric("Maximum drawdown", f"{drawdown.min():.2%}")

price_chart = go.Figure()
price_chart.add_trace(go.Scatter(x=results.index, y=results["Close"], name="Close", line={"color": "#9ec5fe"}))
price_chart.add_trace(go.Scatter(x=results.index, y=results["Short MA"], name=f"MA {short_window}"))
price_chart.add_trace(go.Scatter(x=results.index, y=results["Long MA"], name=f"MA {long_window}"))
if not trades.empty:
    buys = trades[trades["Side"] == "BUY"]
    sells = trades[trades["Side"] == "SELL"]
    price_chart.add_trace(go.Scatter(x=buys["Date"], y=buys["Price"], mode="markers", name="Buy", marker={"color": "#2ecc71", "symbol": "triangle-up", "size": 11}))
    price_chart.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"], mode="markers", name="Sell", marker={"color": "#e74c3c", "symbol": "triangle-down", "size": 11}))
price_chart.update_layout(title=f"{ticker} strategy", height=480, xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(price_chart, use_container_width=True)

equity_chart = go.Figure(go.Scatter(x=results.index, y=results["Equity"], name="Portfolio equity", fill="tozeroy"))
equity_chart.update_layout(title="Equity curve", height=300, xaxis_title="Date", yaxis_title="Value ($)")
st.plotly_chart(equity_chart, use_container_width=True)

st.subheader("Paper-trading activity")
if trades.empty:
    st.info("No crossover trades occurred for these settings.")
else:
    st.dataframe(trades.assign(Date=lambda frame: frame["Date"].dt.date), use_container_width=True, hide_index=True)

with st.expander("Important limits before live trading"):
    st.markdown("""
    - This is a historical simulation, not a promise of future results.
    - It includes a configurable estimated cost, but not every market effect.
    - Validate with walk-forward tests and a broker paper account before considering live orders.
    - Add position limits, stop-loss rules, market-hours checks, and alerting before broker integration.
    """)
