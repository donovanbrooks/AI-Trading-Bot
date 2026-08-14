"""Pure alert rules for the paper-trading dashboard.

The rules never create, modify, or submit orders.  They turn broker snapshots
and model research assessments into clear in-app warnings.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def broker_alerts(
    account: dict[str, Any] | None,
    positions: pd.DataFrame,
    broker_orders: pd.DataFrame,
    local_orders: pd.DataFrame,
    position_loss_limit_pct: float = 0.05,
    daily_loss_limit_pct: float = 0.03,
) -> pd.DataFrame:
    """Return paper-account loss, order-status, and reconciliation alerts."""
    events: list[dict[str, str]] = []
    if not 0 < position_loss_limit_pct < 1 or not 0 < daily_loss_limit_pct < 1:
        raise ValueError("Alert loss limits must be between 0 and 1.")

    if account:
        equity = float(account.get("equity", 0) or 0)
        last_equity = float(account.get("last_equity", 0) or 0)
        if last_equity > 0:
            daily_change = equity / last_equity - 1
            if daily_change <= -daily_loss_limit_pct:
                events.append({
                    "Severity": "Critical",
                    "Type": "Daily paper-risk limit",
                    "Symbol": "Account",
                    "Message": f"Paper equity is {daily_change:.2%} versus the prior close, beyond the {daily_loss_limit_pct:.2%} daily-loss limit.",
                })

    if not positions.empty and {"Symbol", "Unrealized P&L %"}.issubset(positions.columns):
        for _, position in positions.iterrows():
            pnl_pct = float(position["Unrealized P&L %"])
            if pnl_pct <= -position_loss_limit_pct:
                events.append({
                    "Severity": "Warning",
                    "Type": "Position loss warning",
                    "Symbol": str(position["Symbol"]),
                    "Message": f"Unrealized P&L is {pnl_pct:.2%}, beyond the {position_loss_limit_pct:.2%} warning limit.",
                })

    if not broker_orders.empty and {"Symbol", "Status"}.issubset(broker_orders.columns):
        for _, order in broker_orders.iterrows():
            status = str(order["Status"]).lower()
            if any(word in status for word in ("rejected", "canceled", "expired", "suspended")):
                events.append({
                    "Severity": "Warning",
                    "Type": "Paper order needs attention",
                    "Symbol": str(order["Symbol"]),
                    "Message": f"Broker reports order status: {order['Status']}.",
                })
            elif "filled" in status:
                events.append({
                    "Severity": "Info",
                    "Type": "Paper order filled",
                    "Symbol": str(order["Symbol"]),
                    "Message": "Broker reports this recent paper order as filled.",
                })

    if not local_orders.empty and "Order ID" in local_orders.columns:
        broker_ids = set(broker_orders["Order ID"].astype(str)) if "Order ID" in broker_orders.columns else set()
        for _, order in local_orders[~local_orders["Order ID"].astype(str).isin(broker_ids)].iterrows():
            events.append({
                "Severity": "Warning",
                "Type": "Broker reconciliation",
                "Symbol": str(order["Symbol"]),
                "Message": "This locally recorded order is absent from Alpaca's recent-order snapshot. Refresh later or check Alpaca directly.",
            })

    return pd.DataFrame(events, columns=["Severity", "Type", "Symbol", "Message"])


def bullish_research_alerts(assessments: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Return only saved-symbol research assessments that meet the bullish gate."""
    events = [
        {
            "Severity": "Info",
            "Type": "Bullish AI research setup",
            "Symbol": symbol,
            "Message": f"Latest completed bar: {assessment.get('reason', 'Bullish setup detected.')}",
        }
        for symbol, assessment in assessments.items()
        if assessment.get("status") == "Bullish entry setup"
    ]
    return pd.DataFrame(events, columns=["Severity", "Type", "Symbol", "Message"])
