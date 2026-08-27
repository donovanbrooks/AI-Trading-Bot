"""Plain-language default home screen for the paper-trading research app."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from broker import BrokerConfigurationError, get_paper_account_summary, get_paper_portfolio
from copy_strategies import get_strategy_profile
from storage import list_copy_strategy_follows, save_copy_strategy_follow


PROFILE_STRATEGY_IDS = {
    "Moving-average crossover": "conservative_etf_growth",
    "Day-trading AI direction model": "ai_day_research",
    "Crypto AI direction model": "btc_trend_candles",
    "AI direction model": "momentum_etf_basket",
}


def recommended_strategy_id(profile: dict[str, Any]) -> str:
    """Map the onboarding preset to a plain-language paper strategy."""
    return PROFILE_STRATEGY_IDS.get(str(profile.get("recommended_strategy")), "conservative_etf_growth")


def render_simple_home(profile: dict[str, Any]) -> None:
    """Render the beginner-friendly landing screen without exposing trading controls."""
    strategy_id = recommended_strategy_id(profile)
    strategy = get_strategy_profile(strategy_id)

    st.title("Your paper-investing home")
    st.caption("A simple research and paper-trading starting point. Nothing here places a real-money order.")

    summary_one, summary_two, summary_three = st.columns(3)
    summary_one.metric("Your goal", str(profile.get("goal", "Research")))
    summary_two.metric("Comfort level", str(profile.get("risk", "Balanced")))
    summary_three.metric("Markets", ", ".join(profile.get("markets", [])) or "Not selected")

    st.subheader("Your suggested starting plan")
    st.markdown(f"### {strategy['name']}")
    st.write(strategy["explanation"])
    plan_one, plan_two, plan_three = st.columns(3)
    plan_one.metric("Style", strategy["style"])
    plan_two.metric("Risk", strategy["risk_level"])
    plan_three.metric("Review timing", strategy["rebalance"])
    st.caption("Example targets: " + ", ".join(f"{symbol} {weight:.0%}" for symbol, weight in strategy["holdings"].items()))

    with st.form("simple_paper_plan"):
        paper_amount = st.number_input("How much would you like to try in paper trading? ($)", min_value=1.0, max_value=1_000_000.0, value=100.0, step=25.0)
        risk_cap = st.select_slider("How much of your paper account can this plan use?", options=[5, 10, 15, 25, 50], value=10, format_func=lambda value: f"Up to {value}%")
        start_plan = st.form_submit_button("Start my paper plan", type="primary")
    if start_plan:
        try:
            save_copy_strategy_follow(strategy_id, paper_amount, risk_cap / 100, paused=False)
        except Exception as error:
            st.error(f"Could not save your paper plan: {error}")
        else:
            st.success("Your paper plan is ready. You stay in control of every trade.")

    st.subheader("Your paper portfolio")
    st.caption("Connect a paper account in Advanced research to view current paper cash, positions, and order status here.")
    if st.button("Refresh my paper portfolio"):
        try:
            st.session_state["simple_paper_account"] = get_paper_account_summary()
            st.session_state["simple_paper_portfolio"] = get_paper_portfolio()
        except BrokerConfigurationError as error:
            st.info(str(error))
        except Exception as error:
            st.error(f"Could not refresh your paper portfolio: {error}")
    account = st.session_state.get("simple_paper_account")
    portfolio = st.session_state.get("simple_paper_portfolio")
    if account:
        account_one, account_two, account_three = st.columns(3)
        account_one.metric("Paper value", f"${float(account['equity']):,.2f}")
        account_two.metric("Available cash", f"${float(account['cash']):,.2f}")
        account_three.metric("Buying power", f"${float(account['buying_power']):,.2f}")
    if portfolio:
        positions = pd.DataFrame(portfolio.get("positions", []))
        if positions.empty:
            st.info("You do not have any open paper positions yet.")
        else:
            display_columns = [column for column in ["Symbol", "Market Value", "Unrealized P&L", "Unrealized P&L %"] if column in positions.columns]
            st.dataframe(positions[display_columns], use_container_width=True, hide_index=True)

    follows = list_copy_strategy_follows()
    if not follows.empty:
        st.subheader("Your saved plans")
        follows = follows.copy()
        follows["Plan"] = follows["Strategy ID"].map(lambda value: get_strategy_profile(str(value))["name"])
        st.dataframe(
            follows[["Plan", "Paper allocation", "Risk cap", "Paused"]].style.format(
                {"Paper allocation": "${:,.2f}", "Risk cap": "{:.0%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Want more control?")
    st.write("Advanced research lets you review charts, test strategies, explore screeners, connect a paper account, and submit a manual paper order.")
    if st.button("Open advanced research"):
        st.session_state["app_view"] = "Advanced research"
        st.rerun()
