"""Local SQLite persistence for saved backtests and paper trades."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


DEFAULT_DATABASE_PATH = Path("data/trading_bot.db")


def _cloud_context():
    """Return a server-side Supabase client scoped to the signed-in user."""
    try:
        import streamlit as st
        from supabase import create_client

        load_dotenv(override=True)
        user_id = st.session_state.get("supabase_user_id")
        url = os.getenv("SUPABASE_URL")
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if user_id and url and service_key:
            return create_client(url, service_key), str(user_id)
    except Exception:
        pass
    return None, None


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

            CREATE TABLE IF NOT EXISTS paper_order_ledger (
                broker_order_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                notional REAL NOT NULL,
                broker_status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                minimum_score REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS watchlist_symbols (
                watchlist_id INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                PRIMARY KEY (watchlist_id, symbol)
            );
            """
        )


def save_watchlist(
    name: str,
    source: str,
    symbols: list[str],
    asset_type: str,
    minimum_score: float,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Create or replace a named research watchlist and its in-app alert rule."""
    cleaned_name = name.strip()
    cleaned_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not cleaned_name:
        raise ValueError("A watchlist name is required.")
    if not cleaned_symbols:
        raise ValueError("Select at least one symbol for the watchlist.")
    if not 0 <= minimum_score <= 100:
        raise ValueError("The alert score must be between 0 and 100.")
    client, user_id = _cloud_context()
    if client and user_id:
        response = client.table("watchlists").upsert(
            {
                "user_id": user_id,
                "name": cleaned_name,
                "source": source,
                "minimum_score": float(minimum_score),
                "symbols": [{"symbol": symbol, "asset_type": asset_type} for symbol in cleaned_symbols],
            },
            on_conflict="user_id,name",
        ).execute()
        return int(response.data[0]["id"].replace("-", "")[:8], 16)
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO watchlists (name, source, minimum_score)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                source = excluded.source,
                minimum_score = excluded.minimum_score,
                updated_at = CURRENT_TIMESTAMP
            """,
            (cleaned_name, source, float(minimum_score)),
        )
        row = connection.execute("SELECT id FROM watchlists WHERE name = ?", (cleaned_name,)).fetchone()
        watchlist_id = int(row[0])
        connection.execute("DELETE FROM watchlist_symbols WHERE watchlist_id = ?", (watchlist_id,))
        connection.executemany(
            "INSERT INTO watchlist_symbols (watchlist_id, symbol, asset_type) VALUES (?, ?, ?)",
            [(watchlist_id, symbol, asset_type) for symbol in cleaned_symbols],
        )
    return watchlist_id


def list_watchlists(database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    """Return saved watchlists with their symbols and score-alert threshold."""
    client, user_id = _cloud_context()
    if client and user_id:
        rows = client.table("watchlists").select("id,name,source,minimum_score,updated_at,symbols").eq("user_id", user_id).order("updated_at", desc=True).execute().data
        return pd.DataFrame([
            {"Watchlist": row["id"], "Name": row["name"], "Source": row["source"], "Alert score": row["minimum_score"], "Updated": row["updated_at"], "Symbols": ", ".join(item["symbol"] for item in row["symbols"])}
            for row in rows
        ])
    initialize_database(database_path)
    with _connect(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT w.id AS "Watchlist", w.name AS "Name", w.source AS "Source",
                   w.minimum_score AS "Alert score", w.updated_at AS "Updated",
                   GROUP_CONCAT(s.symbol, ', ') AS "Symbols"
            FROM watchlists w
            LEFT JOIN watchlist_symbols s ON s.watchlist_id = w.id
            GROUP BY w.id
            ORDER BY w.updated_at DESC, w.id DESC
            """,
            connection,
        )


def research_alerts(ranking: pd.DataFrame, database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    """Return saved-symbol score alerts triggered by the current screener results."""
    if ranking.empty or not {"Symbol", "Research score"}.issubset(ranking.columns):
        return pd.DataFrame()
    client, user_id = _cloud_context()
    if client and user_id:
        rows = client.table("watchlists").select("name,minimum_score,symbols").eq("user_id", user_id).execute().data
        saved = pd.DataFrame([
            {"Watchlist": row["name"], "Symbol": item["symbol"], "Alert score": row["minimum_score"]}
            for row in rows for item in row["symbols"]
        ])
    else:
        initialize_database(database_path)
        with _connect(database_path) as connection:
            saved = pd.read_sql_query(
                """
                SELECT w.name AS "Watchlist", s.symbol AS "Symbol", w.minimum_score AS "Alert score"
                FROM watchlists w JOIN watchlist_symbols s ON s.watchlist_id = w.id
                """,
                connection,
            )
    if saved.empty:
        return saved
    alerts = saved.merge(ranking[["Symbol", "Research score", "Why it ranked"]], on="Symbol", how="inner")
    return alerts[alerts["Research score"] >= alerts["Alert score"]].sort_values("Research score", ascending=False).reset_index(drop=True)


def record_paper_order(
    broker_order_id: str,
    symbol: str,
    asset_type: str,
    notional: float,
    broker_status: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Persist a submitted paper order for duplicate protection and reconciliation."""
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO paper_order_ledger (
                broker_order_id, symbol, asset_type, notional, broker_status
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (broker_order_id, symbol, asset_type, float(notional), broker_status),
        )


def recent_paper_order(symbol: str, minutes: int = 5, database_path: Path = DEFAULT_DATABASE_PATH) -> bool:
    """Return true if the app recently submitted an order for this symbol."""
    initialize_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT 1 FROM paper_order_ledger
            WHERE symbol = ? AND created_at >= datetime('now', ?)
            LIMIT 1
            """,
            (symbol, f"-{minutes} minutes"),
        ).fetchone()
    return row is not None


def paper_order_ledger(database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    """Return the most recent locally recorded paper-order submissions."""
    initialize_database(database_path)
    with _connect(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT broker_order_id AS "Order ID", created_at AS "Recorded", symbol AS "Symbol",
                   asset_type AS "Asset", notional AS "Notional", broker_status AS "Status"
            FROM paper_order_ledger
            ORDER BY created_at DESC
            LIMIT 50
            """,
            connection,
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
