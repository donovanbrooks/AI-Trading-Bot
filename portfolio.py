"""Shared-cash, multi-ticker portfolio backtesting for research only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 10_000.0
    trading_cost_bps: float = 5.0
    position_size_pct: float = 0.10
    total_exposure_cap: float = 0.50
    max_positions: int = 3
    max_entries_per_day: int = 3
    max_daily_loss_pct: float = 0.01
    max_drawdown_pct: float = 0.10
    stop_loss_pct: float = 0.02

    @property
    def cost_rate(self) -> float:
        return self.trading_cost_bps / 10_000


def _day_key(timestamp: object) -> object:
    return timestamp.date() if hasattr(timestamp, "date") else timestamp


def _signal_strength(row: pd.Series) -> float:
    probability = pd.to_numeric(pd.Series([row.get("AI Probability")]), errors="coerce").iloc[0]
    if pd.notna(probability):
        return float(abs(probability - 0.5))
    if {"Short MA", "Long MA", "Close"}.issubset(row.index) and row["Close"]:
        return float(abs(row["Short MA"] - row["Long MA"]) / row["Close"])
    return 0.0


def run_portfolio_backtest(signals_by_ticker: dict[str, pd.DataFrame], config: PortfolioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest multiple signal frames against one cash balance.

    Exit signals are handled before ranked entries at each timestamp. New
    entries must satisfy every shared portfolio risk rule.
    """
    if not signals_by_ticker:
        raise ValueError("At least one ticker is required")
    if not 0 < config.position_size_pct <= 1 or not 0 < config.total_exposure_cap <= 1:
        raise ValueError("Position size and exposure cap must be between 0 and 1")
    if config.max_positions < 1 or config.max_entries_per_day < 1:
        raise ValueError("Position and daily-entry limits must be at least 1")

    required = {"Open", "High", "Low", "Close", "Execution Signal"}
    frames: dict[str, pd.DataFrame] = {}
    for ticker, frame in signals_by_ticker.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{ticker} is missing columns: {sorted(missing)}")
        frames[ticker] = frame.sort_index()

    timestamps = sorted(set().union(*(frame.index for frame in frames.values())))
    cash = config.initial_cash
    positions: dict[str, dict[str, object]] = {}
    last_prices: dict[str, float] = {}
    trades: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    peak_equity = config.initial_cash
    current_day: object | None = None
    day_start_equity = config.initial_cash
    daily_entries = 0

    def equity() -> float:
        return cash + sum(float(position["shares"]) * last_prices[ticker] for ticker, position in positions.items())

    def close_position(ticker: str, timestamp: object, price: float, reason: str) -> None:
        nonlocal cash
        position = positions.pop(ticker)
        shares, entry_price = float(position["shares"]), float(position["entry_price"])
        proceeds = shares * price * (1 - config.cost_rate)
        cash += proceeds
        trades.append({
            "Ticker": ticker,
            "Entry Date": position["entry_date"],
            "Exit Date": timestamp,
            "Entry Price": entry_price,
            "Exit Price": price,
            "Shares": shares,
            "Exit Reason": reason,
            "Return": (price * (1 - config.cost_rate)) / (entry_price * (1 + config.cost_rate)) - 1,
            "Net P&L": (price * (1 - config.cost_rate) - entry_price * (1 + config.cost_rate)) * shares,
        })

    for timestamp in timestamps:
        rows = {ticker: frame.loc[timestamp] for ticker, frame in frames.items() if timestamp in frame.index}
        for ticker, row in rows.items():
            last_prices[ticker] = float(row["Close"])

        day = _day_key(timestamp)
        if day != current_day:
            current_day = day
            day_start_equity = equity()
            daily_entries = 0

        # Existing positions leave before any new entries are considered.
        for ticker, row in rows.items():
            if ticker not in positions:
                continue
            stop_price = float(positions[ticker]["entry_price"]) * (1 - config.stop_loss_pct)
            if float(row["Low"]) <= stop_price:
                close_position(ticker, timestamp, stop_price, "Stop loss")
            elif int(row["Execution Signal"]) == -1:
                close_position(ticker, timestamp, float(row["Open"]), "Signal")

        current_equity = equity()
        peak_equity = max(peak_equity, current_equity)
        daily_locked = current_equity <= day_start_equity * (1 - config.max_daily_loss_pct)
        drawdown_locked = current_equity <= peak_equity * (1 - config.max_drawdown_pct)
        candidates = sorted(
            ((ticker, row, _signal_strength(row)) for ticker, row in rows.items() if int(row["Execution Signal"]) == 1 and ticker not in positions),
            key=lambda candidate: (-candidate[2], candidate[0]),
        )
        blocked = 0
        for ticker, row, _ in candidates:
            if daily_locked or drawdown_locked:
                blocked += 1
                continue
            if len(positions) >= config.max_positions or daily_entries >= config.max_entries_per_day:
                blocked += 1
                continue
            invested = sum(float(position["shares"]) * last_prices[symbol] for symbol, position in positions.items())
            allocation = min(config.initial_cash * config.position_size_pct, cash, max(0.0, current_equity * config.total_exposure_cap - invested))
            if allocation <= 0:
                blocked += 1
                continue
            open_price = float(row["Open"])
            shares = allocation / (open_price * (1 + config.cost_rate))
            cash -= shares * open_price * (1 + config.cost_rate)
            positions[ticker] = {"shares": shares, "entry_price": open_price, "entry_date": timestamp}
            daily_entries += 1

        current_equity = equity()
        peak_equity = max(peak_equity, current_equity)
        history.append({
            "Date": timestamp,
            "Portfolio Equity": current_equity,
            "Cash": cash,
            "Open Positions": len(positions),
            "Drawdown": current_equity / peak_equity - 1,
            "Blocked Entries": blocked,
        })

    final_timestamp = timestamps[-1]
    for ticker in list(positions):
        close_position(ticker, final_timestamp, last_prices[ticker], "End of test")
    history[-1]["Portfolio Equity"] = cash
    history[-1]["Cash"] = cash
    history[-1]["Open Positions"] = 0
    return pd.DataFrame(history).set_index("Date"), pd.DataFrame(trades)


def portfolio_metrics(results: pd.DataFrame, trades: pd.DataFrame, initial_cash: float) -> dict[str, float | int]:
    """Return comparable portfolio-level performance and risk metrics."""
    returns = results["Portfolio Equity"].pct_change().dropna()
    final_value = float(results["Portfolio Equity"].iloc[-1])
    wins = int((trades["Net P&L"] > 0).sum()) if not trades.empty else 0
    return {
        "final_value": final_value,
        "total_return": final_value / initial_cash - 1,
        "max_drawdown": float(results["Drawdown"].min()),
        "trade_count": int(len(trades)),
        "win_rate": wins / len(trades) if len(trades) else 0.0,
        "blocked_entries": int(results["Blocked Entries"].sum()),
        "average_cash": float(results["Cash"].mean()),
        "volatility": float(returns.std(ddof=0)) if not returns.empty else 0.0,
    }
