"""Dashboard for evaluating strategies and managing Alpaca paper trading only."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auth import require_login
from alerts import broker_alerts, bullish_research_alerts
from onboarding import build_research_preset
from broker import (
    BrokerConfigurationError,
    MAX_PAPER_ORDER_NOTIONAL,
    get_paper_account_summary,
    get_paper_portfolio,
    lookup_paper_asset,
    submit_confirmed_paper_buy,
    submit_confirmed_paper_crypto_buy,
    validate_paper_credentials,
)
from storage import (
    list_recent_runs,
    list_watchlists,
    paper_order_ledger,
    recent_paper_order,
    record_paper_order,
    research_alerts,
    get_automation_settings,
    save_automation_settings,
    save_backtest_run,
    save_watchlist,
    list_watchlist_symbols,
    clear_research_profile,
    get_research_profile,
    has_paper_broker_credentials,
    remove_paper_broker_credentials,
    save_paper_broker_credentials,
    save_research_profile,
)
from logging_config import configure_logging
from market_data import load_alpaca_bars, load_twelve_data_bars
from regime_analysis import regime_performance
from portfolio import PortfolioConfig, portfolio_metrics, run_portfolio_backtest
from screener import US_ETF_UNIVERSE, US_STOCK_UNIVERSE, rank_research_universe
from strategy.ai_validation import generate_ai_signals
from strategy.intraday_ai_validation import generate_intraday_ai_signals
from strategy.crypto_ai_validation import generate_crypto_ai_signals
from strategy.timing import latest_ai_assessment, rank_intraday_assessments
from strategy_search import evaluate_candidates, market_regimes, probability_signals, regime_recommendations
from validation import BacktestConfig, build_crossover_signals, calculate_metrics, run_backtest, run_buy_and_hold_backtest, walk_forward_validate


st.set_page_config(page_title="Trading Bot Lab", page_icon="📈", layout="wide")
logger = configure_logging()
require_login()

profile = get_research_profile()
if profile is None:
    st.title("Welcome to Trading Bot Lab")
    st.caption("Answer a few questions to create an editable research starting point. This is educational research support, not personalized investment advice.")
    with st.form("research_onboarding"):
        onboarding_markets = st.multiselect("What do you want to research?", ["Stocks", "ETFs", "Crypto"])
        onboarding_goal = st.radio("What is your main goal?", ["Long-term research", "Active/day research", "Both"])
        onboarding_risk = st.radio("Which risk setting feels closest to your preference?", ["Conservative", "Balanced", "Growth"])
        onboarding_experience = st.radio("Your experience level", ["New to investing", "Some experience", "Experienced"])
        onboarding_monitoring = st.radio("How often can you monitor research and paper positions?", ["A few times per month", "A few times per week", "Most days"])
        onboarding_submit = st.form_submit_button("Create my research preset", type="primary")
    if onboarding_submit:
        try:
            profile = build_research_preset(
                onboarding_markets,
                onboarding_goal,
                onboarding_risk,
                onboarding_experience,
                onboarding_monitoring,
            )
            save_research_profile(profile)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success("Your editable research preset is ready.")
            st.rerun()
    st.stop()


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    """Load daily production-market data from Alpaca."""
    return load_alpaca_bars(ticker, period, intraday=False)


@st.cache_data(ttl=300, show_spinner=False)
def load_intraday_prices(ticker: str, period: str, crypto: bool = False) -> pd.DataFrame:
    """Load five-minute production-market data from Alpaca."""
    return load_alpaca_bars(ticker, period, intraday=True, crypto=crypto)


@st.cache_data(ttl=86_400, show_spinner=False)
def load_global_prices(symbol: str, period: str) -> pd.DataFrame:
    """Load international daily EOD data from the optional Twelve Data source."""
    return load_twelve_data_bars(symbol, period)


st.title("Trading Bot Lab")
st.caption("Private paper-trading research app — strategy signals never submit orders automatically.")

with st.sidebar:
    with st.expander("Your research preset"):
        st.caption(f"Suggested starting strategy: {profile['recommended_strategy']}")
        st.caption(f"Starter symbols: {', '.join(profile['starter_symbols'])}")
        st.caption(profile["rationale"])
        if st.button("Redo onboarding"):
            clear_research_profile()
            st.rerun()
    st.header("Backtest settings")
    research_mode = st.radio(
        "Research mode",
        ["Single ticker backtest", "Multi-ticker portfolio backtest", "Top 10 US research screener"],
    )
    portfolio_enabled = research_mode == "Multi-ticker portfolio backtest"
    screener_enabled = research_mode == "Top 10 US research screener"
    initial_cash = st.number_input("Starting cash ($)", min_value=100.0, value=10_000.0, step=100.0)
    fee_bps = st.number_input("Estimated fee + slippage (basis points)", min_value=0.0, value=5.0, step=1.0)
    st.divider()
    st.caption("Trade-quality gate")
    position_size = st.slider("Maximum position size (% of cash)", min_value=1, max_value=100, value=10, step=1)
    max_trades_per_day = st.number_input("Maximum new entries per day", min_value=1, max_value=20, value=3, step=1)
    max_daily_loss = st.slider("Daily loss lockout (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    stop_loss = st.slider("Simulated stop loss (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    if screener_enabled:
        st.divider()
        st.caption("Top 10 screener settings")
        screener_universe = st.selectbox("Screen", ["US ETFs", "US large-cap stocks", "International stocks (Twelve Data)", "Day-trading US stocks/ETFs", "Crypto timing (5-minute)"])
        intraday_screen = screener_universe in {"Day-trading US stocks/ETFs", "Crypto timing (5-minute)"}
        screener_period = st.selectbox("Screener history", ["30d", "60d"] if intraday_screen else ["1y", "2y"], index=1 if intraday_screen else 1)
        timing_symbols = st.text_input(
            "Symbols to check now (comma separated)",
            "SPY, QQQ, IWM, XLK, GLD" if screener_universe == "Day-trading US stocks/ETFs" else "BTC-USD, ETH-USD, SOL-USD",
        ) if intraday_screen else ""
        global_symbols = st.text_input(
            "International symbols (comma separated)",
            "ASML:EURONEXT, AZN:LSE, NESN:SIX, SBIN:NSE, 7203:TSE, 0700:HKEX",
            help="Use the exact symbol:exchange identifier from Twelve Data's symbol search.",
        ) if screener_universe == "International stocks (Twelve Data)" else ""
        portfolio_tickers = ""
        portfolio_strategy = "Moving-average crossover"
        portfolio_period = "5y"
        max_positions = 3
        portfolio_exposure = 50
        portfolio_drawdown = 10
        short_window, long_window = 20, 50
        ai_threshold = 0.55
        ticker = ""
        strategy_type = ""
        period = ""
        train_bars = 252
        test_bars = 63
    elif portfolio_enabled:
        st.divider()
        st.caption("Portfolio settings")
        saved_watchlists = list_watchlists()
        watchlist_options = ["Manual ticker entry"]
        if not saved_watchlists.empty:
            watchlist_options.extend(saved_watchlists["Name"].astype(str).tolist())
        selected_watchlist = st.selectbox("Load a saved watchlist", watchlist_options)
        if selected_watchlist == "Manual ticker entry":
            default_portfolio_tickers = "SPY, GLD, QQQ"
        else:
            default_portfolio_tickers = str(
                saved_watchlists.loc[saved_watchlists["Name"] == selected_watchlist, "Symbols"].iloc[0]
            )
            st.caption("Loaded from your saved watchlist. You can still edit the symbols below.")
        portfolio_tickers = st.text_input("Portfolio tickers (comma separated)", default_portfolio_tickers)
        portfolio_strategy = st.selectbox("Portfolio strategy", ["Moving-average crossover", "AI direction model"])
        portfolio_period = st.selectbox("Portfolio history", ["1y", "2y", "5y"], index=2)
        max_positions = st.number_input("Maximum simultaneous positions", min_value=1, max_value=20, value=3, step=1)
        portfolio_exposure = st.slider("Maximum total portfolio exposure (%)", min_value=5, max_value=100, value=50, step=5)
        portfolio_drawdown = st.slider("Portfolio drawdown lockout (%)", min_value=1, max_value=50, value=10, step=1)
        short_window = st.number_input("Portfolio short moving average", min_value=2, max_value=100, value=20)
        long_window = st.number_input("Portfolio long moving average", min_value=3, max_value=300, value=50)
        ai_threshold = st.slider("Portfolio AI confidence threshold", min_value=0.51, max_value=0.75, value=0.55, step=0.01)
        ticker = ""
        strategy_type = ""
        period = ""
        train_bars = 252
        test_bars = 63
    else:
        portfolio_tickers = ""
        portfolio_strategy = "Moving-average crossover"
        portfolio_period = "5y"
        max_positions = 3
        portfolio_exposure = 50
        portfolio_drawdown = 10
        st.divider()
        ticker = st.text_input("Ticker", "SPY").upper().strip()
        strategy_type = st.radio("Strategy", ["Moving-average crossover", "AI direction model", "Day-trading AI direction model", "Crypto AI direction model"])
        if strategy_type == "Day-trading AI direction model":
            period = st.selectbox("5-minute history", ["30d", "60d"])
            st.caption("Intraday validation")
            train_bars = st.number_input("Training history (five-minute bars)", min_value=390, value=780, step=78)
            test_bars = 78
            ai_threshold = st.slider("AI confidence threshold", min_value=0.51, max_value=0.75, value=0.58, step=0.01)
            candle_confirmation = st.checkbox("Require bullish candlestick confirmation", value=True)
            short_window, long_window = 20, 50
        elif strategy_type == "Crypto AI direction model":
            period = st.selectbox("Five-minute crypto history", ["30d", "60d"])
            st.caption("24/7 crypto validation")
            st.info("Use a crypto ticker such as BTC-USD. Strategy orders remain disabled.")
            train_bars = st.number_input("Training history (five-minute bars)", min_value=1_008, value=2_016, step=288)
            test_bars = 288
            ai_threshold = st.slider("AI confidence threshold", min_value=0.51, max_value=0.80, value=0.60, step=0.01)
            candle_confirmation = st.checkbox("Require bullish candlestick confirmation", value=True)
            crypto_trend_filter = st.checkbox("Only buy above 50-bar trend average", value=True)
            crypto_position_size = st.slider("Crypto maximum position size (% of cash)", 1, 25, 5)
            crypto_stop_loss = st.slider("Crypto simulated stop loss (%)", 2.0, 30.0, 8.0, 0.5)
            crypto_daily_loss = st.slider("Crypto 24-hour loss lockout (%)", 0.5, 15.0, 3.0, 0.5)
            short_window, long_window = 20, 50
        else:
            period = st.selectbox("History", ["1y", "2y", "5y"], index=2)
            short_window = st.number_input("Short moving average", min_value=2, max_value=100, value=20)
            long_window = st.number_input("Long moving average", min_value=3, max_value=300, value=50)
            st.caption("Walk-forward validation")
            train_bars = st.number_input("Training history (trading days)", min_value=60, value=252, step=21)
            test_bars = st.number_input("Out-of-sample window (trading days)", min_value=10, value=63, step=21)
            ai_threshold = st.slider("AI confidence threshold", min_value=0.51, max_value=0.75, value=0.55, step=0.01)
            candle_confirmation = False
            crypto_trend_filter = False
            crypto_position_size, crypto_stop_loss, crypto_daily_loss = position_size, stop_loss, max_daily_loss

if screener_enabled:
    st.header("Top 10 US research screener")
    st.caption("A transparent, historical-data ranking—not a prediction, personalized advice, or an order signal.")
    intraday_screen = screener_universe in {"Day-trading US stocks/ETFs", "Crypto timing (5-minute)"}
    if intraday_screen:
        symbols = list(dict.fromkeys(symbol.strip().upper() for symbol in timing_symbols.split(",") if symbol.strip()))
        st.warning("These are fast-changing model assessments from the latest completed five-minute bar—not instructions to buy. Re-run before acting; manual paper orders remain your choice.")
        if st.button("Check current AI timing", type="primary"):
            try:
                with st.spinner("Refreshing five-minute model assessments..."):
                    assessments = {}
                    for symbol in symbols:
                        crypto = screener_universe == "Crypto timing (5-minute)"
                        normalized = symbol.replace("-", "/") if crypto else symbol
                        bars = load_intraday_prices(normalized, screener_period, crypto=crypto)
                        if crypto:
                            signals = generate_crypto_ai_signals(bars, train_bars=1_008, require_bullish_candle=True, require_trend_filter=True)
                        else:
                            signals = generate_intraday_ai_signals(bars, train_bars=390, require_bullish_candle=True)
                        assessments[symbol] = latest_ai_assessment(signals)
                st.session_state["timing_ranking"] = rank_intraday_assessments(assessments)
            except Exception as error:
                logger.exception("Intraday timing screen failed")
                st.error(f"Could not refresh AI timing: {error}")
        timing_ranking = st.session_state.get("timing_ranking")
        if isinstance(timing_ranking, pd.DataFrame) and not timing_ranking.empty:
            st.subheader("Current AI timing assessments")
            st.dataframe(timing_ranking.style.format({"AI bullish probability": "{:.1%}"}), use_container_width=True, hide_index=True)
        st.stop()
    international_screen = screener_universe == "International stocks (Twelve Data)"
    universe = (
        [symbol.strip().upper() for symbol in global_symbols.split(",") if symbol.strip()]
        if international_screen
        else (US_ETF_UNIVERSE if screener_universe == "US ETFs" else US_STOCK_UNIVERSE)
    )
    asset_type = "International stock" if international_screen else ("ETF" if screener_universe == "US ETFs" else "Stock")
    if international_screen:
        st.write(f"This screen compares {len(universe)} international symbols you selected. It uses end-of-day data and keeps each listing in its local currency.")
        st.warning("Scores compare percentage returns and risk—not share prices—because listings may use different currencies. Verify each listing and currency before any decision.")
    else:
        st.write(f"This screen compares {len(universe)} liquid US {asset_type.lower()} symbols from a fixed starter universe.")
    st.info("Score weights: 3-month return 25%, 12-month return 30%, long-term trend 20%, lower volatility 15%, shallower drawdown 7%, and liquidity 3%.")
    if st.button("Run Top 10 research screen", type="primary"):
        try:
            with st.spinner(f"Loading daily history for {len(universe)} {asset_type.lower()} symbols..."):
                universe_prices: dict[str, pd.DataFrame] = {}
                unavailable: list[str] = []
                for symbol in universe:
                    try:
                        universe_prices[symbol] = load_global_prices(symbol, screener_period) if international_screen else load_prices(symbol, screener_period)
                    except Exception:
                        unavailable.append(symbol)
                ranking = rank_research_universe(universe_prices, asset_type)
        except Exception as error:
            logger.exception("Research screener failed for %s", screener_universe)
            st.error(f"Could not run the research screen: {error}")
        else:
            if ranking.empty:
                st.warning("No symbols had enough usable price history to rank. Try again later.")
            else:
                st.session_state["screener_ranking"] = ranking
                st.session_state["screener_source"] = screener_universe
                st.session_state["screener_unavailable"] = unavailable

    ranking = st.session_state.get("screener_ranking")
    if isinstance(ranking, pd.DataFrame) and not ranking.empty:
        st.subheader(f"Top 10 {st.session_state.get('screener_source', screener_universe)}")
        st.dataframe(
            ranking.style.format({
                "Research score": "{:.1f}",
                "3-month return": "{:.1%}",
                "12-month return": "{:.1%}",
                "Trend": "{:.1%}",
                "Volatility": "{:.1%}",
                "1-year drawdown": "{:.1%}",
                "Median daily dollar volume": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Use the symbols as ideas for separate paper-trading and walk-forward validation. A high score does not mean a trade will be profitable.")
        if st.session_state.get("screener_unavailable"):
            st.caption(f"Unavailable in this run: {', '.join(st.session_state['screener_unavailable'])}.")

        st.subheader("Save a watchlist")
        with st.form("save_research_watchlist"):
            watchlist_name = st.text_input("Watchlist name", "My research ideas")
            selected_symbols = st.multiselect("Symbols to save", ranking["Symbol"].tolist(), default=ranking["Symbol"].head(3).tolist())
            alert_score = st.slider("Show an in-app alert when research score is at least", 0, 100, 75)
            save_watchlist_clicked = st.form_submit_button("Save watchlist and alert rule")
        if save_watchlist_clicked:
            try:
                watchlist_id = save_watchlist(
                    watchlist_name,
                    st.session_state.get("screener_source", screener_universe),
                    selected_symbols,
                    asset_type,
                    alert_score,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Saved watchlist #{watchlist_id}. Its score alert will be checked whenever you run a screener.")

        alerts = research_alerts(ranking)
        st.subheader("Current in-app alerts")
        if alerts.empty:
            st.info("No saved watchlist symbols meet their score-alert threshold in this screen.")
        else:
            st.warning(f"{len(alerts)} saved watchlist alert(s) triggered in this screen.")
            st.dataframe(alerts, use_container_width=True, hide_index=True)

    with st.expander("Saved watchlists"):
        saved_watchlists = list_watchlists()
        if saved_watchlists.empty:
            st.info("Save a screen result above to create your first watchlist.")
        else:
            st.dataframe(saved_watchlists, use_container_width=True, hide_index=True)
    st.stop()

if portfolio_enabled:
    st.header("Multi-ticker portfolio backtest")
    st.caption("One shared cash balance across daily stock/ETF signals. This is research only and does not submit broker orders.")
    tickers = list(dict.fromkeys(symbol.strip().upper() for symbol in portfolio_tickers.split(",") if symbol.strip()))
    if portfolio_strategy == "Moving-average crossover" and short_window >= long_window:
        st.error("The portfolio short moving average must be smaller than the long moving average.")
    elif not tickers:
        st.warning("Enter at least one stock or ETF ticker.")
    elif any("/" in symbol or "-" in symbol for symbol in tickers):
        st.warning("The first portfolio version supports daily stock and ETF tickers only. Run crypto in its separate research mode.")
    else:
        try:
            with st.spinner("Loading portfolio signals and ranking entries..."):
                portfolio_signals: dict[str, pd.DataFrame] = {}
                for symbol in tickers:
                    portfolio_prices = load_prices(symbol, portfolio_period)
                    if portfolio_strategy == "AI direction model":
                        portfolio_signals[symbol] = generate_ai_signals(portfolio_prices, train_bars=252, probability_threshold=ai_threshold)
                    else:
                        portfolio_signals[symbol] = build_crossover_signals(portfolio_prices, int(short_window), int(long_window))
                portfolio_config = PortfolioConfig(
                    initial_cash=initial_cash,
                    trading_cost_bps=fee_bps,
                    position_size_pct=position_size / 100,
                    total_exposure_cap=portfolio_exposure / 100,
                    max_positions=int(max_positions),
                    max_entries_per_day=int(max_trades_per_day),
                    max_daily_loss_pct=max_daily_loss / 100,
                    max_drawdown_pct=portfolio_drawdown / 100,
                    stop_loss_pct=stop_loss / 100,
                )
                portfolio_results, portfolio_trades = run_portfolio_backtest(portfolio_signals, portfolio_config)
                portfolio_summary = portfolio_metrics(portfolio_results, portfolio_trades, initial_cash)
        except Exception as error:
            logger.exception("Portfolio backtest failed for %s", tickers)
            st.error(f"Could not run the portfolio backtest: {error}")
        else:
            column_one, column_two, column_three, column_four = st.columns(4)
            column_one.metric("Portfolio value", f"${portfolio_summary['final_value']:,.2f}")
            column_two.metric("Portfolio return", f"{portfolio_summary['total_return']:.2%}")
            column_three.metric("Maximum drawdown", f"{portfolio_summary['max_drawdown']:.2%}")
            column_four.metric("Average idle cash", f"${portfolio_summary['average_cash']:,.2f}")
            st.caption(f"{portfolio_summary['trade_count']} completed trades · {portfolio_summary['win_rate']:.0%} win rate · {portfolio_summary['blocked_entries']} ranked entries blocked by portfolio rules")
            portfolio_chart = go.Figure(go.Scatter(x=portfolio_results.index, y=portfolio_results["Portfolio Equity"], name="Shared portfolio equity", fill="tozeroy"))
            portfolio_chart.update_layout(height=320, xaxis_title="Date", yaxis_title="Portfolio value ($)")
            st.plotly_chart(portfolio_chart, use_container_width=True)
            if portfolio_trades.empty:
                st.info("No completed portfolio trades for these settings.")
            else:
                st.dataframe(portfolio_trades, use_container_width=True, hide_index=True)
    st.stop()

if strategy_type == "Moving-average crossover" and short_window >= long_window:
    st.error("The short moving average must be smaller than the long moving average.")
    st.stop()

if strategy_type == "Crypto AI direction model" and "-" not in ticker:
    ticker = "BTC-USD"
    st.info("Crypto mode uses a crypto ticker. Loaded BTC-USD instead of the previous stock ticker.")

try:
    with st.spinner(f"Loading {ticker}..."):
        prices = load_intraday_prices(ticker, period, crypto=strategy_type == "Crypto AI direction model") if strategy_type in {"Day-trading AI direction model", "Crypto AI direction model"} else load_prices(ticker, period)
except Exception as error:
    logger.exception("Market-data load failed for %s", ticker)
    st.error(f"Could not load market data for {ticker}: {error}")
    st.stop()

minimum_bars = int(train_bars) + 50 if strategy_type in {"Day-trading AI direction model", "Crypto AI direction model"} else int(long_window) + 2
if prices.empty or len(prices) < minimum_bars:
    st.error("Not enough price history for those settings. Choose a longer history or shorter windows.")
    st.stop()

config = BacktestConfig(
    initial_cash=initial_cash,
    trading_cost_bps=fee_bps,
    position_size_pct=(crypto_position_size if strategy_type == "Crypto AI direction model" else position_size) / 100,
    max_trades_per_day=int(max_trades_per_day),
    max_daily_loss_pct=(crypto_daily_loss if strategy_type == "Crypto AI direction model" else max_daily_loss) / 100,
    stop_loss_pct=(crypto_stop_loss if strategy_type == "Crypto AI direction model" else stop_loss) / 100,
)
if strategy_type == "Day-trading AI direction model":
    with st.spinner("Generating intraday expanding-window AI predictions..."):
        signals = generate_intraday_ai_signals(
            prices,
            train_bars=int(train_bars),
            probability_threshold=ai_threshold,
            require_bullish_candle=candle_confirmation,
        )
elif strategy_type == "Crypto AI direction model":
    with st.spinner("Generating 24/7 crypto AI predictions..."):
        signals = generate_crypto_ai_signals(
            prices,
            train_bars=int(train_bars),
            probability_threshold=ai_threshold,
            require_bullish_candle=candle_confirmation,
            require_trend_filter=crypto_trend_filter,
        )
elif strategy_type == "AI direction model":
    with st.spinner("Generating expanding-window AI predictions..."):
        signals = generate_ai_signals(
            prices,
            train_bars=int(train_bars),
            probability_threshold=ai_threshold,
        )
else:
    signals = build_crossover_signals(prices, int(short_window), int(long_window))
results, trades = run_backtest(signals, config)
periods_per_year = 365 * 288 if strategy_type == "Crypto AI direction model" else (252 * 78 if strategy_type == "Day-trading AI direction model" else 252)
metrics = calculate_metrics(results, trades, initial_cash, periods_per_year=periods_per_year)
benchmark_results, benchmark_trades = run_buy_and_hold_backtest(prices, config)
benchmark_metrics = calculate_metrics(benchmark_results, benchmark_trades, initial_cash, periods_per_year=periods_per_year)
final_value = float(metrics["final_value"])
strategy_return = float(metrics["total_return"])
buy_hold_return = float(benchmark_metrics["total_return"])
return_difference = strategy_return - buy_hold_return

one, two, three, four = st.columns(4)
one.metric("Portfolio value", f"${final_value:,.2f}")
two.metric("Strategy return", f"{strategy_return:.2%}")
three.metric("Buy & hold", f"{buy_hold_return:.2%}")
four.metric("Maximum drawdown", f"{metrics['max_drawdown']:.2%}")

comparison = pd.DataFrame([
    {
        "Approach": strategy_type,
        "Return": strategy_return,
        "Max drawdown": metrics["max_drawdown"],
        "Sharpe": metrics["sharpe_ratio"],
        "Market exposure": metrics["exposure"],
        "Completed trades": metrics["trade_count"],
    },
    {
        "Approach": "Buy & hold (same estimated costs)",
        "Return": benchmark_metrics["total_return"],
        "Max drawdown": benchmark_metrics["max_drawdown"],
        "Sharpe": benchmark_metrics["sharpe_ratio"],
        "Market exposure": benchmark_metrics["exposure"],
        "Completed trades": benchmark_metrics["trade_count"],
    },
])
st.subheader("Strategy comparison")
st.caption(
    f"Strategy {'outperformed' if return_difference >= 0 else 'underperformed'} buy & hold by "
    f"{abs(return_difference):.2%} over this test window. Both include the same estimated entry and exit costs."
)
st.dataframe(
    comparison.style.format({"Return": "{:.2%}", "Max drawdown": "{:.2%}", "Market exposure": "{:.0%}", "Sharpe": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Automatic strategy explorer"):
    st.caption(
        "Tests a broad, pre-approved grid of strategy settings on three contiguous out-of-sample folds. "
        "It ranks consistency and risk, not the single highest historical return. Your current sizing, stop, loss lockout, and costs remain fixed for every candidate."
    )
    search_depth = st.radio("Search size", ["Standard", "Deep"], horizontal=True, key="strategy_search_depth")
    if st.button("Explore strategy variations", type="primary"):
        candidates = []
        if strategy_type == "Moving-average crossover":
            short_options = [5, 10, 15, 20, 30, 40, 50] if search_depth == "Standard" else [5, 8, 10, 15, 20, 30, 40, 50, 60, 75]
            long_options = [20, 30, 50, 75, 100, 150, 200] if search_depth == "Standard" else [20, 30, 40, 50, 75, 100, 150, 200, 250, 300]
            for candidate_short in short_options:
                for candidate_long in long_options:
                    if candidate_short >= candidate_long or candidate_long >= len(prices):
                        continue
                    candidate_signals = build_crossover_signals(prices, candidate_short, candidate_long)
                    current_status = "Bullish research setup" if int(candidate_signals["Signal"].iloc[-1]) == 1 else "No new long setup"
                    candidates.append(({
                        "Short MA": candidate_short,
                        "Long MA": candidate_long,
                        "Current research status": current_status,
                    }, candidate_signals))
        else:
            thresholds = [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68] if search_depth == "Standard" else [round(value, 2) for value in [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72]]
            candle_options = [False, True] if strategy_type in {"Day-trading AI direction model", "Crypto AI direction model"} else [False]
            trend_options = [False, True] if strategy_type == "Crypto AI direction model" else [False]
            for threshold in thresholds:
                for use_candle in candle_options:
                    for use_trend in trend_options:
                        candidate_signals = probability_signals(
                            signals,
                            threshold,
                            require_bullish_candle=use_candle,
                            require_trend_filter=use_trend,
                        )
                        signal = int(candidate_signals["Signal"].iloc[-1])
                        current_status = "Bullish research setup" if signal == 1 else ("Exit / no new long" if signal == -1 else "No new long setup")
                        candidates.append(({
                            "AI threshold": threshold,
                            "Candle confirmation": use_candle,
                            "Trend filter": use_trend,
                            "Current research status": current_status,
                        }, candidate_signals))
        try:
            with st.spinner(f"Evaluating {len(candidates)} candidate variations across out-of-sample folds..."):
                st.session_state["strategy_explorer_results"] = evaluate_candidates(
                    candidates,
                    config,
                    periods_per_year=periods_per_year,
                    folds=3,
                    min_trades=2,
                )
                st.session_state["strategy_regime_recommendations"] = regime_recommendations(candidates, config)
                st.session_state["current_market_regime"] = market_regimes(prices).iloc[-1]
        except Exception as error:
            logger.exception("Strategy explorer failed")
            st.error(f"Could not complete the strategy exploration: {error}")
    explorer_results = st.session_state.get("strategy_explorer_results", pd.DataFrame())
    if not explorer_results.empty:
        st.success("Top row is the most consistent candidate in this historical test—not a guarantee of future performance.")
        formatters = {
            "Median OOS return": "{:.2%}", "Worst OOS return": "{:.2%}",
            "Worst OOS drawdown": "{:.2%}", "Median OOS Sharpe": "{:.2f}", "Consistency score": "{:.3f}",
        }
        st.dataframe(explorer_results.head(20).style.format(formatters), use_container_width=True, hide_index=True)
        regime_recommendations_frame = st.session_state.get("strategy_regime_recommendations", pd.DataFrame())
        if not regime_recommendations_frame.empty:
            current_regime = st.session_state.get("current_market_regime", "Unknown")
            current_recommendation = regime_recommendations_frame[
                regime_recommendations_frame["Market regime"] == current_regime
            ]
            st.markdown("**Regime-specific research recommendations**")
            st.caption(
                f"Current broad market regime: {current_regime}. These are historical conditional results from the same walk-forward signals, not a prediction or an order instruction."
            )
            if not current_recommendation.empty:
                current_choice = current_recommendation.iloc[0]
                st.success(
                    f"Current-regime research candidate: {current_choice['Current research status']} · "
                    f"conditional return {current_choice['Conditional return']:.2%} · "
                    f"conditional drawdown {current_choice['Conditional drawdown']:.2%}."
                )
            st.dataframe(
                regime_recommendations_frame.style.format({
                    "Conditional return": "{:.2%}", "Conditional drawdown": "{:.2%}",
                    "Conditional Sharpe": "{:.2f}", "Regime score": "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
    elif "strategy_explorer_results" in st.session_state:
        st.warning("No candidate met the minimum two completed out-of-sample trades. Try a longer history or less restrictive settings.")

st.caption(
    f"{metrics['trade_count']} completed trades · {metrics['win_rate']:.0%} win rate · "
    f"{metrics['exposure']:.0%} market exposure · Sharpe {metrics['sharpe_ratio']:.2f}"
)
if metrics["blocked_entries"]:
    st.caption(f"Trade-quality gate blocked {metrics['blocked_entries']} entry signal(s).")

price_chart = go.Figure()
if strategy_type in {"Day-trading AI direction model", "Crypto AI direction model"}:
    price_chart.add_trace(
        go.Candlestick(
            x=results.index,
            open=results["Open"],
            high=results["High"],
            low=results["Low"],
            close=results["Close"],
            name="Five-minute candles",
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
        )
    )
    bullish_candles = results[results.get("Bullish Candle", pd.Series(False, index=results.index)).astype(bool)]
    if not bullish_candles.empty:
        price_chart.add_trace(
            go.Scatter(
                x=bullish_candles.index,
                y=bullish_candles["Low"] * 0.999,
                mode="markers",
                name="Bullish candle pattern",
                marker={"color": "#f1c40f", "symbol": "diamond", "size": 8, "line": {"color": "#ffffff", "width": 1}},
                hovertemplate="Bullish engulfing or hammer<br>%{x}<extra></extra>",
            )
        )
    if strategy_type == "Crypto AI direction model":
        price_chart.add_trace(go.Scatter(x=results.index, y=results["Trend MA 50"], name="50-bar trend", line={"color": "#9ec5fe"}))
else:
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
price_chart.update_layout(title=f"{ticker} strategy", height=480, xaxis_title="Date", yaxis_title="Price ($)", xaxis_rangeslider_visible=False)
st.plotly_chart(price_chart, use_container_width=True)

equity_chart = go.Figure(go.Scatter(x=results.index, y=results["Equity"], name="Portfolio equity", fill="tozeroy"))
equity_chart.update_layout(title="Equity curve", height=300, xaxis_title="Date", yaxis_title="Value ($)")
st.plotly_chart(equity_chart, use_container_width=True)

with st.expander("Market-regime validation"):
    regimes = regime_performance(results)
    if regimes.empty:
        st.info("Not enough observations to classify market regimes.")
    else:
        st.caption("Compare results by broad trend and volatility conditions; do not rely on a single overall return.")
        st.dataframe(regimes, use_container_width=True, hide_index=True)

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
    "ai_confidence_threshold": ai_threshold if "AI direction model" in strategy_type else None,
    "position_size_pct": position_size,
    "max_trades_per_day": int(max_trades_per_day),
    "max_daily_loss_pct": max_daily_loss,
    "stop_loss_pct": stop_loss,
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

st.subheader("Your Alpaca paper connection")
st.caption("Each user connects their own Alpaca paper account. Keys are encrypted on the server and live-trading credentials are not supported.")
paper_connection_ready = has_paper_broker_credentials()
if not paper_connection_ready:
    with st.form("connect_alpaca_paper"):
        paper_api_key = st.text_input("Alpaca paper API key", type="password")
        paper_api_secret = st.text_input("Alpaca paper secret key", type="password")
        connect_paper_account = st.form_submit_button("Verify and connect paper account", type="primary")
    if connect_paper_account:
        try:
            verified_account = validate_paper_credentials(paper_api_key, paper_api_secret)
            save_paper_broker_credentials(paper_api_key, paper_api_secret)
        except (BrokerConfigurationError, ValueError) as error:
            st.error(str(error))
        except Exception as error:
            logger.exception("Could not save paper broker connection")
            st.error(f"Could not securely connect that paper account: {error}")
        else:
            st.success(f"Verified and securely connected paper account {verified_account['account_number']} ({verified_account['status']}).")
            st.rerun()
else:
    st.success("Your Alpaca paper account is securely connected. Live trading is disabled.")
    connection_one, connection_two = st.columns(2)
    if connection_one.button("Check my paper account"):
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
    if connection_two.button("Disconnect my paper account"):
        try:
            remove_paper_broker_credentials()
        except Exception as error:
            st.error(f"Could not remove the saved paper connection: {error}")
        else:
            st.success("Your encrypted paper-broker credentials were removed.")
            st.rerun()

with st.expander("AI paper-order permission"):
    automation = get_automation_settings()
    st.caption("Default: AI creates suggestions and you approve every order. This setting never enables live trading.")
    with st.form("automation_permission"):
        autonomous_orders = st.checkbox(
            "Allow autonomous paper orders within the hard safety limits below",
            value=bool(automation["autonomous_paper_orders"]),
        )
        autonomous_cap = st.number_input(
            "Maximum autonomous order amount ($)",
            min_value=1.0,
            max_value=MAX_PAPER_ORDER_NOTIONAL,
            value=float(automation["max_order_notional"]),
            step=1.0,
        )
        acknowledgement = st.checkbox("I understand this applies only to Alpaca paper orders and I can turn it off at any time.")
        save_automation = st.form_submit_button("Save paper-order permission")
    if save_automation:
        if autonomous_orders and not acknowledgement:
            st.error("Confirm that you understand the paper-only permission before enabling it.")
        else:
            try:
                save_automation_settings(autonomous_orders, autonomous_cap)
            except ValueError as error:
                st.error(str(error))
            else:
                state = "enabled" if autonomous_orders else "disabled"
                st.success(f"Autonomous paper-order eligibility is {state}. Live trading remains disabled.")
    if automation["autonomous_paper_orders"]:
        st.warning("Autonomous paper-order eligibility is on. A scheduled automation worker is not connected yet, so no strategy orders are being placed automatically today.")
    else:
        st.info("AI-generated orders require your approval. You can still submit manual paper buys below.")

st.subheader("Trade ticket")
st.caption("Search a US stock/ETF or supported crypto pair, choose a dollar amount, then either submit a paper buy yourself or hand it to the AI paper-order queue.")
ticket_type = st.radio("What do you want to trade?", ["Stock or ETF", "Crypto"], horizontal=True)
asset_type = "crypto" if ticket_type == "Crypto" else "equity"
ticket_default = "BTC/USD" if asset_type == "crypto" else ticker
with st.form("paper_trade_ticket", clear_on_submit=False):
    ticket_symbol = st.text_input(
        "Search symbol or pair",
        value=ticket_default,
        help="Examples: AAPL, SPY, NVDA, BTC/USD, ETH/USD.",
    )
    ticket_notional = st.number_input("How much do you want to invest? ($)", min_value=1.0, max_value=MAX_PAPER_ORDER_NOTIONAL, value=5.0, step=1.0)
    ticket_mode = st.radio("Who decides when to submit?", ["I approve every paper order", "AI-managed paper order"], horizontal=True)
    review_trade = st.form_submit_button("Review trade")

if review_trade:
    try:
        st.session_state["ticket_asset"] = lookup_paper_asset(ticket_symbol, asset_type)
        st.session_state["ticket_notional"] = ticket_notional
        st.session_state["ticket_mode"] = ticket_mode
        with st.spinner("Checking the latest completed AI signal..."):
            ticket_bars = load_intraday_prices(
                st.session_state["ticket_asset"]["symbol"],
                "30d",
                crypto=asset_type == "crypto",
            )
            ticket_signals = (
                generate_crypto_ai_signals(ticket_bars, train_bars=1_008, require_bullish_candle=True, require_trend_filter=True)
                if asset_type == "crypto"
                else generate_intraday_ai_signals(ticket_bars, train_bars=390, require_bullish_candle=True)
            )
            st.session_state["ticket_assessment"] = latest_ai_assessment(ticket_signals)
    except BrokerConfigurationError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Could not look up that paper-trading asset: {error}")

ticket_asset = st.session_state.get("ticket_asset")
if ticket_asset:
    st.success(f"Selected: {ticket_asset['symbol']} · {ticket_asset['name']}")
    selected_notional = float(st.session_state.get("ticket_notional", 5.0))
    selected_mode = st.session_state.get("ticket_mode", "I approve every paper order")
    assessment = st.session_state.get("ticket_assessment")
    if assessment:
        probability = assessment["probability"]
        probability_text = f" · model confidence {probability:.1%}" if probability is not None else ""
        if assessment["status"] == "Bullish entry setup":
            st.success(f"AI research status: {assessment['status']}{probability_text}. {assessment['reason']}")
        else:
            st.warning(f"AI caution: {assessment['status']}{probability_text}. {assessment['reason']} You can still make your own paper-trading decision.")
    if selected_mode == "AI-managed paper order":
        automation = get_automation_settings()
        if not automation["autonomous_paper_orders"]:
            st.info("AI-managed mode is off for your account. Enable it in AI paper-order permission, or choose manual approval.")
        elif selected_notional > float(automation["max_order_notional"]):
            st.warning(f"Your AI paper-order cap is ${automation['max_order_notional']:.2f}. Lower the amount or update the cap.")
        else:
            st.info("This trade is eligible for AI-managed paper execution. The scheduled AI worker is the next feature to connect; no order is sent yet.")
    else:
        confirmation = st.checkbox(f"I confirm a paper-market buy of up to ${selected_notional:.2f} for {ticket_asset['symbol']} may be submitted.")
        if st.button("Submit my paper buy", type="primary"):
            if not confirmation:
                st.error("Check the confirmation box before submitting a paper order.")
            else:
                try:
                    if recent_paper_order(ticket_asset["symbol"]):
                        raise BrokerConfigurationError(f"A {ticket_asset['symbol']} order was submitted recently. Refresh orders before trying again.")
                    order = submit_confirmed_paper_crypto_buy(ticket_asset["symbol"], selected_notional) if ticket_asset["asset_type"] == "crypto" else submit_confirmed_paper_buy(ticket_asset["symbol"], selected_notional)
                except BrokerConfigurationError as error:
                    st.warning(str(error))
                except Exception as error:
                    st.error(f"Paper order was not submitted: {error}")
                else:
                    record_paper_order(order["id"], order["symbol"], ticket_asset["asset_type"], selected_notional, order["status"])
                    st.success(f"Paper order submitted: {order['symbol']} · {order['status']} · ID {order['id']}")

st.subheader("Portfolio dashboard")
st.caption("Your Alpaca paper account only. Market values and P&L come from Alpaca when you refresh; research labels are not order instructions.")
if st.button("Refresh portfolio dashboard", type="primary"):
    try:
        st.session_state["paper_portfolio"] = get_paper_portfolio()
        st.session_state["paper_account"] = get_paper_account_summary()
    except BrokerConfigurationError as error:
        st.warning(str(error))
    except Exception as error:
        st.error(f"Could not retrieve paper positions and orders: {error}")

portfolio = st.session_state.get("paper_portfolio")
if portfolio:
    position_frame = pd.DataFrame(portfolio["positions"])
    order_frame = pd.DataFrame(portfolio["orders"])
    account = st.session_state.get("paper_account")
    if not position_frame.empty:
        for column in ["Quantity", "Average Entry", "Current Price", "Market Value", "Unrealized P&L", "Unrealized P&L %", "Today's P&L"]:
            position_frame[column] = pd.to_numeric(position_frame[column], errors="coerce")
    invested_value = float(position_frame["Market Value"].sum()) if not position_frame.empty else 0.0
    unrealized_pnl = float(position_frame["Unrealized P&L"].sum()) if not position_frame.empty else 0.0
    dashboard_one, dashboard_two, dashboard_three, dashboard_four = st.columns(4)
    dashboard_one.metric("Paper equity", f"${float(account['equity']):,.2f}" if account else "Refresh to load")
    dashboard_two.metric("Available cash", f"${float(account['cash']):,.2f}" if account else "Refresh to load")
    dashboard_three.metric("Invested value", f"${invested_value:,.2f}")
    dashboard_four.metric("Open P&L", f"${unrealized_pnl:,.2f}")
    st.caption("Snapshot from Alpaca. Refresh after an order fills or when you want the latest market value.")
    st.markdown("**Open positions**")
    if position_frame.empty:
        st.info("No open paper positions.")
    else:
        assessments = st.session_state.get("position_assessments", {})
        if assessments:
            position_frame["AI research status"] = position_frame["Symbol"].map(
                lambda symbol: assessments.get(str(symbol), {}).get("status", "Not assessed")
            )
            position_frame["AI confidence"] = position_frame["Symbol"].map(
                lambda symbol: assessments.get(str(symbol), {}).get("probability")
            )
        st.dataframe(
            position_frame.style.format({
                "Quantity": "{:,.6g}", "Average Entry": "${:,.2f}", "Current Price": "${:,.2f}",
                "Market Value": "${:,.2f}", "Unrealized P&L": "${:,.2f}",
                "Unrealized P&L %": "{:.2%}", "Today's P&L": "{:.2%}", "AI confidence": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Check latest AI research for open positions"):
            assessments = {}
            symbols = position_frame["Symbol"].astype(str).head(10).tolist()
            with st.spinner("Checking the latest completed five-minute bar for each position..."):
                for symbol in symbols:
                    try:
                        crypto = "/" in symbol
                        bars = load_intraday_prices(symbol, "30d", crypto=crypto)
                        signals = (
                            generate_crypto_ai_signals(bars, train_bars=1_008, require_bullish_candle=True, require_trend_filter=True)
                            if crypto else generate_intraday_ai_signals(bars, train_bars=390, require_bullish_candle=True)
                        )
                        assessments[symbol] = latest_ai_assessment(signals)
                    except Exception as error:
                        logger.warning("Could not assess paper position %s: %s", symbol, error)
                        assessments[symbol] = {"status": "Unavailable", "probability": None, "reason": "Could not load enough recent market data."}
            st.session_state["position_assessments"] = assessments
            st.rerun()
    st.markdown("**Recent orders**")
    if order_frame.empty:
        st.info("No recent paper orders.")
    else:
        st.dataframe(order_frame, use_container_width=True, hide_index=True)

    local_orders = paper_order_ledger()
    broker_order_ids = set(order_frame["Order ID"].astype(str)) if not order_frame.empty else set()
    missing_from_broker = local_orders[~local_orders["Order ID"].astype(str).isin(broker_order_ids)] if not local_orders.empty else local_orders
    st.markdown("**Local order ledger**")
    if local_orders.empty:
        st.info("No orders have been submitted through this app yet.")
    else:
        st.dataframe(local_orders, use_container_width=True, hide_index=True)
        if not missing_from_broker.empty:
            st.warning("Some locally recorded orders are not in Alpaca's recent 10-order snapshot. Refresh later or check the Alpaca dashboard.")

    st.subheader("Alert Center")
    st.caption("Alerts are research and paper-account warnings. They never submit, close, or change an order.")
    alert_one, alert_two = st.columns(2)
    position_warning_pct = alert_one.slider("Position-loss warning (%)", 1.0, 20.0, 5.0, 0.5) / 100
    daily_warning_pct = alert_two.slider("Daily paper-risk limit (%)", 0.5, 15.0, 3.0, 0.5) / 100
    if st.button("Run paper-account alert check"):
        account_alerts = broker_alerts(
            account,
            position_frame,
            order_frame,
            local_orders,
            position_loss_limit_pct=position_warning_pct,
            daily_loss_limit_pct=daily_warning_pct,
        )
        st.session_state["paper_account_alerts"] = account_alerts
    account_alerts = st.session_state.get("paper_account_alerts", pd.DataFrame())
    if not account_alerts.empty:
        st.dataframe(account_alerts, use_container_width=True, hide_index=True)
    else:
        st.info("Run the check to evaluate your current paper-account snapshot.")

    with st.expander("Saved-symbol bullish setup alerts"):
        st.caption("Checks the latest completed five-minute bar for up to 10 saved symbols. A bullish label is research only, never a command to trade.")
        with st.form("quick_alert_watchlist"):
            quick_watchlist_name = st.text_input("Watchlist name", "My alert symbols")
            quick_watchlist_symbols = st.text_input("Stocks, ETFs, or crypto pairs (comma separated)", placeholder="SPY, QQQ, BTC-USD")
            save_quick_watchlist = st.form_submit_button("Save symbols for alerts")
        if save_quick_watchlist:
            symbols = [symbol.strip().upper() for symbol in quick_watchlist_symbols.split(",") if symbol.strip()]
            try:
                save_watchlist(quick_watchlist_name, "Alert Center", symbols, "Mixed", 0)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Saved {len(symbols)} symbol(s) for research alerts.")
        if st.button("Check saved symbols for bullish setups"):
            saved_symbols = list_watchlist_symbols()
            assessments = {}
            if saved_symbols.empty:
                st.info("Save at least one symbol above or from a research screen first.")
            else:
                with st.spinner("Checking saved symbols using the latest completed five-minute bars..."):
                    for _, saved in saved_symbols.drop_duplicates("Symbol").head(10).iterrows():
                        symbol = str(saved["Symbol"]).upper()
                        crypto = "/" in symbol or symbol.endswith("-USD") or str(saved["Asset type"]).lower() == "crypto"
                        try:
                            bars = load_intraday_prices(symbol, "30d", crypto=crypto)
                            signals = (
                                generate_crypto_ai_signals(bars, train_bars=1_008, require_bullish_candle=True, require_trend_filter=True)
                                if crypto else generate_intraday_ai_signals(bars, train_bars=390, require_bullish_candle=True)
                            )
                            assessments[symbol] = latest_ai_assessment(signals)
                        except Exception as error:
                            logger.warning("Saved-symbol alert check failed for %s: %s", symbol, error)
                st.session_state["watchlist_bullish_alerts"] = bullish_research_alerts(assessments)
        bullish_alerts = st.session_state.get("watchlist_bullish_alerts", pd.DataFrame())
        if not bullish_alerts.empty:
            st.dataframe(bullish_alerts, use_container_width=True, hide_index=True)
        elif "watchlist_bullish_alerts" in st.session_state:
            st.info("No saved symbols currently meet the bullish research gate.")

    with st.expander("Scheduled watchlist worker"):
        st.markdown(
            "The automatic worker is intentionally unavailable until each user has their own encrypted broker connection. "
            "This prevents a shared paper account from being checked or acted on for the wrong person. "
            "For now, use the on-demand paper-account check and the saved-watchlist research screen."
        )

st.subheader("Walk-forward validation")
if strategy_type == "Day-trading AI direction model":
    st.caption("Five-minute model: expanding-window training, same-session targets, next-bar execution, and a forced end-of-session exit. This is research only; it cannot place strategy orders.")
elif strategy_type == "Crypto AI direction model":
    st.caption("Five-minute crypto model: expanding-window training and next-bar execution across a 24/7 market. This is research only; it cannot place strategy orders.")
elif strategy_type == "AI direction model":
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
