"""Backtesting and walk-forward validation utilities.

These functions are deliberately broker-agnostic.  They model a long-only
daily strategy: decisions are made at a day's close and filled at the next
session's open, with an estimated proportional trading cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    trading_cost_bps: float = 5.0
    position_size_pct: float = 0.10
    max_trades_per_day: int | None = 3
    max_daily_loss_pct: float | None = 0.01
    stop_loss_pct: float | None = 0.02

    @property
    def cost_rate(self) -> float:
        return self.trading_cost_bps / 10_000


def build_crossover_signals(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Build crossover signals and delay execution until the next open."""
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    result = data.copy()
    result["Short MA"] = result["Close"].rolling(short_window).mean()
    result["Long MA"] = result["Close"].rolling(long_window).mean()
    regime = (result["Short MA"] > result["Long MA"]).astype(int)
    result["Signal"] = regime.diff().fillna(0).clip(-1, 1).astype(int)
    result["Execution Signal"] = result["Signal"].shift(1).fillna(0).astype(int)
    return result


def run_backtest(data: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a cost-aware long-only backtest against already-generated signals."""
    required_columns = {"Open", "Close", "Execution Signal"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required backtest columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Cannot backtest an empty dataframe")

    cash, shares = config.initial_cash, 0.0
    if not 0 < config.position_size_pct <= 1:
        raise ValueError("position_size_pct must be greater than 0 and no more than 1")
    if config.max_trades_per_day is not None and config.max_trades_per_day < 1:
        raise ValueError("max_trades_per_day must be at least 1")
    if config.max_daily_loss_pct is not None and not 0 < config.max_daily_loss_pct < 1:
        raise ValueError("max_daily_loss_pct must be between 0 and 1")
    if config.stop_loss_pct is not None and not 0 < config.stop_loss_pct < 1:
        raise ValueError("stop_loss_pct must be between 0 and 1")

    entry_price: float | None = None
    entry_date: object | None = None
    trades: list[dict[str, object]] = []
    equity: list[float] = []
    in_market: list[bool] = []
    blocked_entries: list[str] = []
    current_day: object | None = None
    day_start_equity = config.initial_cash
    day_trade_count = 0

    def sell(timestamp: object, price: float, reason: str) -> None:
        nonlocal cash, shares, entry_price, entry_date
        gross_proceeds = shares * price
        fee = gross_proceeds * config.cost_rate
        cash += gross_proceeds - fee
        trade_return = ((price * (1 - config.cost_rate)) / (entry_price * (1 + config.cost_rate))) - 1
        trades.append({
            "Entry Date": entry_date,
            "Exit Date": timestamp,
            "Entry Price": entry_price,
            "Exit Price": price,
            "Shares": shares,
            "Exit Reason": reason,
            "Return": trade_return,
            "Net P&L": (price * (1 - config.cost_rate) - entry_price * (1 + config.cost_rate)) * shares,
        })
        shares, entry_price, entry_date = 0.0, None, None

    for timestamp, row in data.iterrows():
        open_price = float(row["Open"])
        signal = int(row["Execution Signal"])
        trade_day = timestamp.date() if hasattr(timestamp, "date") else timestamp
        if trade_day != current_day:
            current_day = trade_day
            day_start_equity = cash + shares * open_price
            day_trade_count = 0

        equity_at_open = cash + shares * open_price
        daily_loss_limit_hit = (
            config.max_daily_loss_pct is not None
            and equity_at_open <= day_start_equity * (1 - config.max_daily_loss_pct)
        )
        blocked_reason = ""
        stop_triggered = (
            shares > 0
            and config.stop_loss_pct is not None
            and float(row["Low"]) <= entry_price * (1 - config.stop_loss_pct)
        )
        if stop_triggered:
            sell(timestamp, entry_price * (1 - config.stop_loss_pct), "Stop loss")
        elif signal == 1 and shares == 0:
            if config.max_trades_per_day is not None and day_trade_count >= config.max_trades_per_day:
                blocked_reason = "Daily trade limit"
            elif daily_loss_limit_hit:
                blocked_reason = "Daily loss limit"
            else:
                allocation = cash * config.position_size_pct
                shares = allocation / (open_price * (1 + config.cost_rate))
                cash -= shares * open_price * (1 + config.cost_rate)
                entry_price, entry_date = open_price, timestamp
                day_trade_count += 1
        elif signal == -1 and shares > 0:
            sell(timestamp, open_price, "Signal")
        equity.append(cash + shares * float(row["Close"]))
        in_market.append(shares > 0)
        blocked_entries.append(blocked_reason)

    if shares > 0:
        final_timestamp = data.index[-1]
        final_price = float(data.iloc[-1]["Close"])
        sell(final_timestamp, final_price, "End of test")
        equity[-1] = cash

    result = data.copy()
    result["Equity"] = equity
    result["In Market"] = in_market
    result["Entry Blocked"] = blocked_entries
    result["Drawdown"] = result["Equity"] / result["Equity"].cummax() - 1
    return result, pd.DataFrame(trades)


def run_buy_and_hold_backtest(data: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a fully invested benchmark using the same costs as a strategy."""
    required_columns = {"Open", "Close"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required buy-and-hold columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Cannot backtest an empty dataframe")

    first_open = float(data["Open"].iloc[0])
    final_close = float(data["Close"].iloc[-1])
    shares = config.initial_cash / (first_open * (1 + config.cost_rate))
    equity = shares * data["Close"].astype(float)
    final_value = float(equity.iloc[-1]) * (1 - config.cost_rate)
    equity.iloc[-1] = final_value

    result = data.copy()
    result["Equity"] = equity
    result["In Market"] = True
    result["Entry Blocked"] = ""
    result["Drawdown"] = result["Equity"] / result["Equity"].cummax() - 1
    trades = pd.DataFrame([{
        "Entry Date": data.index[0],
        "Exit Date": data.index[-1],
        "Entry Price": first_open,
        "Exit Price": final_close,
        "Shares": shares,
        "Exit Reason": "End of test",
        "Return": final_value / config.initial_cash - 1,
        "Net P&L": final_value - config.initial_cash,
    }])
    return result, trades


def calculate_metrics(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float | int]:
    """Calculate risk and trade metrics from a completed backtest."""
    final_value = float(results["Equity"].iloc[-1])
    daily_returns = results["Equity"].pct_change().dropna()
    periods = max(len(results) - 1, 1)
    years = periods / periods_per_year
    annualized_return = (final_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0.0
    annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(periods_per_year)) if not daily_returns.empty else 0.0
    sharpe_ratio = (float(daily_returns.mean()) / float(daily_returns.std(ddof=0)) * np.sqrt(periods_per_year)
                    if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0 else 0.0)
    win_rate = float((trades["Return"] > 0).mean()) if not trades.empty else 0.0
    gross_wins = float(trades.loc[trades["Net P&L"] > 0, "Net P&L"].sum()) if not trades.empty else 0.0
    gross_losses = float(-trades.loc[trades["Net P&L"] < 0, "Net P&L"].sum()) if not trades.empty else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses else np.nan
    exposure = float(results["In Market"].mean())

    return {
        "final_value": final_value,
        "total_return": final_value / initial_cash - 1,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": float(results["Drawdown"].min()),
        "trade_count": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "exposure": exposure,
        "blocked_entries": int((results["Entry Blocked"] != "").sum()),
    }


def walk_forward_validate(
    data: pd.DataFrame,
    short_window: int,
    long_window: int,
    config: BacktestConfig,
    train_bars: int = 252,
    test_bars: int = 63,
) -> pd.DataFrame:
    """Evaluate successive out-of-sample windows using only prior history.

    The strategy parameters stay fixed; this measures their stability rather
    than selecting the best settings after seeing future test periods.
    """
    if train_bars < long_window or test_bars < 2:
        raise ValueError("train_bars must cover the long window and test_bars must be at least 2")

    signals = build_crossover_signals(data, short_window, long_window)
    reports: list[dict[str, object]] = []
    for start in range(train_bars, len(signals) - 1, test_bars):
        end = min(start + test_bars, len(signals))
        test_data = signals.iloc[start:end].copy()
        if len(test_data) < 2:
            continue
        results, trades = run_backtest(test_data, config)
        metrics = calculate_metrics(results, trades, config.initial_cash)
        reports.append({
            "Train End": signals.index[start - 1],
            "Test Start": test_data.index[0],
            "Test End": test_data.index[-1],
            "OOS Return": metrics["total_return"],
            "Max Drawdown": metrics["max_drawdown"],
            "Sharpe": metrics["sharpe_ratio"],
            "Trades": metrics["trade_count"],
        })
    return pd.DataFrame(reports)
