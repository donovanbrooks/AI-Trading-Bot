"""Local dashboard for evaluating the trading strategy.

This app is intentionally research and paper-trading only.  It never submits
orders to a broker or stores credentials.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from broker import BrokerConfigurationError, MAX_PAPER_ORDER_NOTIONAL, get_paper_account_summary, submit_confirmed_paper_buy
from storage import list_recent_runs, save_backtest_run
from strategy.ai_validation import generate_ai_signals
from validation import BacktestConfig, build_crossover_signals, calculate_metrics, run_backtest, walk_forward_validate


st.set_page_config(page_title="Trading Bot Lab", page_icon="📈", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    """Download daily OHLCV data and return a flat, clean dataframe."""
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna().copy()


st.title("Trading Bot Lab")
st.caption("Research dashboard — backtesting only. It does not place real trades.")

with st.sidebar:
    st.header("Backtest settings")
    ticker = st.text_input("Ticker", "SPY").upper().strip()
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
    strategy_type = st.radio("Strategy", ["Moving-average crossover", "AI direction model"])
    short_window = st.number_input("Short moving average", min_value=2, max_value=100, value=20)
    long_window = st.number_input("Long moving average", min_value=3, max_value=300, value=50)
    initial_cash = st.number_input("Starting cash ($)", min_value=100.0, value=10_000.0, step=100.0)
    fee_bps = st.number_input("Estimated fee + slippage (basis points)", min_value=0.0, value=5.0, step=1.0)
    st.divider()
    st.caption("Walk-forward validation")
    train_bars = st.number_input("Training history (trading days)", min_value=60, value=252, step=21)
    test_bars = st.number_input("Out-of-sample window (trading days)", min_value=10, value=63, step=21)
    ai_threshold = st.slider("AI confidence threshold", min_value=0.51, max_value=0.75, value=0.55, step=0.01)

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

config = BacktestConfig(initial_cash=initial_cash, trading_cost_bps=fee_bps)
if strategy_type == "AI direction model":
    with st.spinner("Generating expanding-window AI predictions..."):
        signals = generate_ai_signals(
            prices,
            train_bars=int(train_bars),
            probability_threshold=ai_threshold,
        )
else:
    signals = build_crossover_signals(prices, int(short_window), int(long_window))
results, trades = run_backtest(signals, config)
metrics = calculate_metrics(results, trades, initial_cash)
final_value = float(metrics["final_value"])
strategy_return = float(metrics["total_return"])
buy_hold_return = float(results["Close"].iloc[-1] / results["Close"].iloc[0] - 1)

one, two, three, four = st.columns(4)
one.metric("Portfolio value", f"${final_value:,.2f}")
two.metric("Strategy return", f"{strategy_return:.2%}")
three.metric("Buy & hold", f"{buy_hold_return:.2%}")
four.metric("Maximum drawdown", f"{metrics['max_drawdown']:.2%}")

st.caption(
    f"{metrics['trade_count']} completed trades · {metrics['win_rate']:.0%} win rate · "
    f"{metrics['exposure']:.0%} market exposure · Sharpe {metrics['sharpe_ratio']:.2f}"
)

price_chart = go.Figure()
price_chart.add_trace(go.Scatter(x=results.index, y=results["Close"], name="Close", line={"color": "#9ec5fe"}))
if strategy_type == "Moving-average crossover":
    price_chart.add_trace(go.Scatter(x=results.index, y=results["Short MA"], name=f"MA {short_window}"))
    price_chart.add_trace(go.Scatter(x=results.index, y=results["Long MA"], name=f"MA {long_window}"))
if not trades.empty:
    price_chart.add_trace(
        go.Scatter(
            x=trades["Entry Date"],
            y=trades["Entry Price"],
            mode="markers",
            name="Buy",
            marker={"color": "#2ecc71", "symbol": "triangle-up", "size": 11},
        )
    )
    price_chart.add_trace(
        go.Scatter(
            x=trades["Exit Date"],
            y=trades["Exit Price"],
            mode="markers",
            name="Sell",
            marker={"color": "#e74c3c", "symbol": "triangle-down", "size": 11},
        )
    )
price_chart.update_layout(title=f"{ticker} strategy", height=480, xaxis_title="Date", yaxis_title="Price ($)")
st.plotly_chart(price_chart, use_container_width=True)

equity_chart = go.Figure(go.Scatter(x=results.index, y=results["Equity"], name="Portfolio equity", fill="tozeroy"))
equity_chart.update_layout(title="Equity curve", height=300, xaxis_title="Date", yaxis_title="Value ($)")
st.plotly_chart(equity_chart, use_container_width=True)

st.subheader("Completed paper trades")
if trades.empty:
    st.info("No crossover trades occurred for these settings.")
else:
    st.dataframe(
        trades.assign(**{"Entry Date": lambda frame: frame["Entry Date"].dt.date, "Exit Date": lambda frame: frame["Exit Date"].dt.date}),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Save this backtest")
parameters = {
    "history": period,
    "short_window": int(short_window),
    "long_window": int(long_window),
    "training_bars": int(train_bars),
    "test_bars": int(test_bars),
    "ai_confidence_threshold": ai_threshold if strategy_type == "AI direction model" else None,
}
if st.button("Save current backtest", type="primary"):
    run_id = save_backtest_run(
        ticker=ticker,
        strategy=strategy_type,
        parameters=parameters,
        initial_cash=initial_cash,
        cost_bps=fee_bps,
        metrics=metrics,
        trades=trades,
    )
    st.success(f"Saved backtest #{run_id} and {len(trades)} completed paper trades locally.")

with st.expander("Recent saved backtests"):
    saved_runs = list_recent_runs()
    if saved_runs.empty:
        st.info("No saved backtests yet.")
    else:
        st.dataframe(saved_runs, use_container_width=True, hide_index=True)

st.subheader("Alpaca paper account")
st.caption("Connection check only. This app does not submit orders yet.")
if st.button("Check paper account connection"):
    try:
        account = get_paper_account_summary()
    except BrokerConfigurationError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Alpaca connection failed: {error}")
    else:
        account_columns = st.columns(3)
        account_columns[0].metric("Paper cash", f"${account['cash']:,.2f}")
        account_columns[1].metric("Buying power", f"${account['buying_power']:,.2f}")
        account_columns[2].metric("Paper equity", f"${account['equity']:,.2f}")
        st.success(f"Connected to paper account {account['account_number']} ({account['status']}).")

with st.expander("Manual paper buy"):
    st.warning("This sends a real order to your Alpaca PAPER account. It cannot use real-money credentials.")
    with st.form("manual_paper_buy", clear_on_submit=False):
        paper_symbol = st.text_input("US stock symbol", value=ticker, max_chars=10).upper()
        paper_notional = st.number_input("Dollar amount", min_value=1.0, max_value=MAX_PAPER_ORDER_NOTIONAL, value=5.0, step=1.0)
        confirmation = st.checkbox(f"I confirm a paper-market buy of up to ${paper_notional:.2f} may be submitted.")
        submit_paper_buy = st.form_submit_button("Submit paper buy")
    if submit_paper_buy:
        if not confirmation:
            st.error("Check the confirmation box before submitting a paper order.")
        else:
            try:
                order = submit_confirmed_paper_buy(paper_symbol, paper_notional)
            except BrokerConfigurationError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"Paper order was not submitted: {error}")
            else:
                st.success(f"Paper order submitted: {order['symbol']} · {order['status']} · ID {order['id']}")

st.subheader("Walk-forward validation")
if strategy_type == "AI direction model":
    st.caption("AI predictions are generated in expanding windows: every prediction uses only earlier labeled data. The chart and metrics above are its out-of-sample evaluation.")
elif len(prices) <= int(train_bars) + 1:
    st.info("Choose a longer price history or a smaller training history to see out-of-sample windows.")
else:
    walk_forward = walk_forward_validate(
        prices,
        int(short_window),
        int(long_window),
        config,
        train_bars=int(train_bars),
        test_bars=int(test_bars),
    )
    if walk_forward.empty:
        st.info("No complete out-of-sample windows were available.")
    else:
        st.caption("Each row starts after its training history. Parameters are fixed, not optimized on future test periods.")
        st.dataframe(walk_forward, use_container_width=True, hide_index=True)
        st.metric("Median out-of-sample return", f"{walk_forward['OOS Return'].median():.2%}")

with st.expander("Important limits before live trading"):
    st.markdown("""
    - This is a historical simulation, not a promise of future results.
    - It includes a configurable estimated cost, but not every market effect.
    - Validate with walk-forward tests and a broker paper account before considering live orders.
    - Add position limits, stop-loss rules, market-hours checks, and alerting before broker integration.
    """)
