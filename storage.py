"""Local SQLite persistence for saved backtests and paper trades."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATABASE_PATH = Path("data/trading_bot.db")


def _connect(database_path: Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Create the local database tables if they do not exist yet."""
    with _connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                strategy TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                initial_cash REAL NOT NULL,
                cost_bps REAL NOT NULL,
                final_value REAL NOT NULL,
                total_return REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                trade_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
                entry_date TEXT NOT NULL,
                exit_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                shares REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                return_pct REAL NOT NULL,
                net_pnl REAL NOT NULL
            );
            """
        )


def save_backtest_run(
    ticker: str,
    strategy: str,
    parameters: dict[str, Any],
    initial_cash: float,
    cost_bps: float,
    metrics: dict[str, float | int],
    trades: pd.DataFrame,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Persist one backtest and its completed paper trades, returning its id."""
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO backtest_runs (
                ticker, strategy, parameters_json, initial_cash, cost_bps,
                final_value, total_return, sharpe_ratio, max_drawdown, trade_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                strategy,
                json.dumps(parameters, sort_keys=True),
                float(initial_cash),
                float(cost_bps),
                float(metrics["final_value"]),
                float(metrics["total_return"]),
                float(metrics["sharpe_ratio"]),
                float(metrics["max_drawdown"]),
                int(metrics["trade_count"]),
            ),
        )
        run_id = int(cursor.lastrowid)
        if not trades.empty:
            rows = [
                (
                    run_id,
                    pd.Timestamp(row["Entry Date"]).isoformat(),
                    pd.Timestamp(row["Exit Date"]).isoformat(),
                    float(row["Entry Price"]),
                    float(row["Exit Price"]),
                    float(row["Shares"]),
                    str(row["Exit Reason"]),
                    float(row["Return"]),
                    float(row["Net P&L"]),
                )
                for _, row in trades.iterrows()
            ]
            connection.executemany(
                """
                INSERT INTO paper_trades (
                    run_id, entry_date, exit_date, entry_price, exit_price, shares,
                    exit_reason, return_pct, net_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    return run_id


def list_recent_runs(limit: int = 10, database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    """Return a compact history of the latest saved backtests."""
    initialize_database(database_path)
    with _connect(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT id AS "Run", saved_at AS "Saved", ticker AS "Ticker", strategy AS "Strategy",
                   total_return AS "Return", sharpe_ratio AS "Sharpe",
                   max_drawdown AS "Max Drawdown", trade_count AS "Trades"
            FROM backtest_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=(limit,),
        )
