"""Local SQLite persistence for saved backtests and paper trades."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from credentials import CredentialError, decrypt_secret, encrypt_secret


DEFAULT_DATABASE_PATH = Path("data/trading_bot.db")


class StorageUnavailableError(RuntimeError):
    """Raised when configured cloud storage cannot safely serve a user request."""


def _cloud_context():
    """Return an RLS-enforced client for the signed-in user.

    The Streamlit app must never use SUPABASE_SERVICE_ROLE_KEY for a user
    request. The scheduled worker is the only process that needs that key.
    """
    import streamlit as st
    from supabase import create_client

    load_dotenv(override=True)
    user_id = st.session_state.get("supabase_user_id")
    url = os.getenv("SUPABASE_URL")
    publishable_key = os.getenv("SUPABASE_ANON_KEY")
    access_token = st.session_state.get("supabase_access_token")
    refresh_token = st.session_state.get("supabase_refresh_token")
    if not user_id:
        return None, None
    if not all((url, publishable_key, user_id, access_token, refresh_token)):
        raise StorageUnavailableError("Cloud storage is not ready. Sign in again or check the Supabase server configuration.")
    try:
        client = create_client(url, publishable_key)
        client.auth.set_session(access_token, refresh_token)
        return client, str(user_id)
    except Exception as error:
        raise StorageUnavailableError("Cloud storage is unavailable. Your data was not saved locally; try again later.") from error


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

            CREATE TABLE IF NOT EXISTS automation_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                autonomous_paper_orders INTEGER NOT NULL DEFAULT 0,
                max_order_notional REAL NOT NULL DEFAULT 25,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_research_profiles (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS encrypted_credentials (
                provider TEXT PRIMARY KEY,
                encrypted_value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS copy_strategy_follows (
                strategy_id TEXT PRIMARY KEY,
                allocation_amount REAL NOT NULL,
                risk_cap_pct REAL NOT NULL,
                paused INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def save_paper_broker_credentials(api_key: str, api_secret: str, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Encrypt and persist the current user's Alpaca paper credentials only."""
    if not api_key.strip() or not api_secret.strip():
        raise ValueError("Both the Alpaca paper API key and secret are required.")
    encrypted_value = encrypt_secret(json.dumps({"api_key": api_key.strip(), "api_secret": api_secret.strip()}))
    client, user_id = _cloud_context()
    if client and user_id:
        client.table("encrypted_credentials").upsert(
            {"user_id": user_id, "provider": "alpaca_paper", "encrypted_value": encrypted_value},
            on_conflict="user_id,provider",
        ).execute()
        return
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """INSERT INTO encrypted_credentials (provider, encrypted_value) VALUES ('alpaca_paper', ?)
            ON CONFLICT(provider) DO UPDATE SET encrypted_value=excluded.encrypted_value, updated_at=CURRENT_TIMESTAMP""",
            (encrypted_value,),
        )


def get_paper_broker_credentials(database_path: Path = DEFAULT_DATABASE_PATH) -> tuple[str, str]:
    """Decrypt the signed-in user's paper credentials for a server-side SDK call."""
    encrypted_value = None
    client, user_id = _cloud_context()
    if client and user_id:
        rows = client.table("encrypted_credentials").select("encrypted_value").eq("user_id", user_id).eq("provider", "alpaca_paper").limit(1).execute().data
        if rows:
            encrypted_value = rows[0]["encrypted_value"]
    else:
        initialize_database(database_path)
        with _connect(database_path) as connection:
            row = connection.execute("SELECT encrypted_value FROM encrypted_credentials WHERE provider = 'alpaca_paper'").fetchone()
        encrypted_value = row[0] if row else None
    if not encrypted_value:
        raise CredentialError("Connect your own Alpaca paper account before using broker features.")
    try:
        payload = json.loads(decrypt_secret(encrypted_value))
        return str(payload["api_key"]), str(payload["api_secret"])
    except (CredentialError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise CredentialError("Stored paper credentials are invalid. Reconnect your paper account.") from error


def has_paper_broker_credentials(database_path: Path = DEFAULT_DATABASE_PATH) -> bool:
    try:
        get_paper_broker_credentials(database_path)
        return True
    except CredentialError:
        return False


def remove_paper_broker_credentials(database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Remove only the signed-in user's encrypted Alpaca paper credentials."""
    client, user_id = _cloud_context()
    if client and user_id:
        client.table("encrypted_credentials").delete().eq("user_id", user_id).eq("provider", "alpaca_paper").execute()
        return
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute("DELETE FROM encrypted_credentials WHERE provider = 'alpaca_paper'")


def get_research_profile(database_path: Path = DEFAULT_DATABASE_PATH) -> dict[str, Any] | None:
    """Load the signed-in user's onboarding research profile, if completed."""
    client, user_id = _cloud_context()
    if client and user_id:
        try:
            rows = client.table("user_research_profiles").select("profile").eq("user_id", user_id).limit(1).execute().data
            if rows:
                return dict(rows[0]["profile"])
        except Exception:
            # The local fallback lets the app work until the optional migration
            # is run in Supabase.
            pass
    initialize_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute("SELECT profile_json FROM user_research_profiles WHERE id = 1").fetchone()
    return json.loads(row[0]) if row else None


def save_research_profile(profile: dict[str, Any], database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Persist a non-advisory onboarding profile."""
    if not profile.get("recommended_strategy") or not profile.get("starter_symbols"):
        raise ValueError("The research profile is incomplete.")
    client, user_id = _cloud_context()
    if client and user_id:
        try:
            client.table("user_research_profiles").upsert(
                {"user_id": user_id, "profile": profile}, on_conflict="user_id"
            ).execute()
            return
        except Exception:
            pass
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """INSERT INTO user_research_profiles (id, profile_json) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET profile_json=excluded.profile_json, updated_at=CURRENT_TIMESTAMP""",
            (json.dumps(profile, sort_keys=True),),
        )


def clear_research_profile(database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Remove the current user's onboarding profile so they can start again."""
    client, user_id = _cloud_context()
    if client and user_id:
        try:
            client.table("user_research_profiles").delete().eq("user_id", user_id).execute()
            return
        except Exception:
            pass
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute("DELETE FROM user_research_profiles WHERE id = 1")


def list_scheduled_research_alerts(limit: int = 20) -> pd.DataFrame:
    """Return recent alert-only scheduled research results for this user."""
    client, user_id = _cloud_context()
    if not client or not user_id:
        return pd.DataFrame()
    try:
        rows = client.table("screen_results").select("ranking,created_at").eq("user_id", user_id).eq("source", "scheduled_research_alerts").order("created_at", desc=True).limit(limit).execute().data
        return pd.DataFrame([
            {"Checked": row["created_at"], **event}
            for row in rows for event in row.get("ranking", [])
        ])
    except Exception:
        return pd.DataFrame()


def list_scheduled_strategy_tests(limit: int = 50) -> pd.DataFrame:
    """Return the signed-in user's latest bounded daily strategy-test reports."""
    client, user_id = _cloud_context()
    if not client or not user_id:
        return pd.DataFrame()
    try:
        rows = client.table("screen_results").select("ranking,created_at").eq(
            "user_id", user_id
        ).eq("source", "scheduled_strategy_tests").order("created_at", desc=True).limit(limit).execute().data
        return pd.DataFrame([
            {"Tested": row["created_at"], **result}
            for row in rows for result in row.get("ranking", [])
        ])
    except Exception:
        return pd.DataFrame()


def get_automation_settings(database_path: Path = DEFAULT_DATABASE_PATH) -> dict[str, float | bool]:
    """Return the signed-in user's paper automation preference and hard cap."""
    client, user_id = _cloud_context()
    if client and user_id:
        rows = client.table("automation_settings").select("autonomous_paper_orders,max_order_notional").eq("user_id", user_id).limit(1).execute().data
        if rows:
            return {"autonomous_paper_orders": bool(rows[0]["autonomous_paper_orders"]), "max_order_notional": float(rows[0]["max_order_notional"])}
        return {"autonomous_paper_orders": False, "max_order_notional": 25.0}
    initialize_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute("SELECT autonomous_paper_orders, max_order_notional FROM automation_settings WHERE id = 1").fetchone()
    return {"autonomous_paper_orders": bool(row[0]) if row else False, "max_order_notional": float(row[1]) if row else 25.0}


def save_automation_settings(enabled: bool, max_order_notional: float, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Persist explicit consent for autonomous paper-only orders; live orders are never enabled."""
    if not 1 <= max_order_notional <= 25:
        raise ValueError("Autonomous paper orders are capped between $1 and $25 per order.")
    client, user_id = _cloud_context()
    if client and user_id:
        client.table("automation_settings").upsert(
            {"user_id": user_id, "autonomous_paper_orders": bool(enabled), "max_order_notional": float(max_order_notional)},
            on_conflict="user_id",
        ).execute()
        return
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """INSERT INTO automation_settings (id, autonomous_paper_orders, max_order_notional)
            VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET
            autonomous_paper_orders=excluded.autonomous_paper_orders,
            max_order_notional=excluded.max_order_notional, updated_at=CURRENT_TIMESTAMP""",
            (int(enabled), float(max_order_notional)),
        )


def save_copy_strategy_follow(strategy_id: str, allocation_amount: float, risk_cap_pct: float, paused: bool,
                              database_path: Path = DEFAULT_DATABASE_PATH) -> None:
    """Save a user's paper-only strategy-follow plan; this never places orders."""
    if not strategy_id.strip():
        raise ValueError("A strategy is required.")
    if not 1 <= float(allocation_amount) <= 1_000_000:
        raise ValueError("Paper allocation must be between $1 and $1,000,000.")
    if not 0 < float(risk_cap_pct) <= 1:
        raise ValueError("Risk cap must be greater than 0% and no more than 100%.")
    client, user_id = _cloud_context()
    payload = {"strategy_id": strategy_id.strip(), "allocation_amount": float(allocation_amount),
               "risk_cap_pct": float(risk_cap_pct), "paused": bool(paused)}
    if client and user_id:
        client.table("copy_strategy_follows").upsert({"user_id": user_id, **payload}, on_conflict="user_id,strategy_id").execute()
        return
    initialize_database(database_path)
    with _connect(database_path) as connection:
        connection.execute(
            """INSERT INTO copy_strategy_follows (strategy_id, allocation_amount, risk_cap_pct, paused)
            VALUES (?, ?, ?, ?) ON CONFLICT(strategy_id) DO UPDATE SET allocation_amount=excluded.allocation_amount,
            risk_cap_pct=excluded.risk_cap_pct, paused=excluded.paused, updated_at=CURRENT_TIMESTAMP""",
            (payload["strategy_id"], payload["allocation_amount"], payload["risk_cap_pct"], int(payload["paused"])),
        )


def list_copy_strategy_follows(database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    """Return the current user's saved paper-only strategy follows."""
    client, user_id = _cloud_context()
    if client and user_id:
        try:
            rows = client.table("copy_strategy_follows").select("strategy_id,allocation_amount,risk_cap_pct,paused,updated_at").eq("user_id", user_id).order("updated_at", desc=True).execute().data
            return pd.DataFrame([{"Strategy ID": row["strategy_id"], "Paper allocation": float(row["allocation_amount"]),
                                  "Risk cap": float(row["risk_cap_pct"]), "Paused": bool(row["paused"]), "Updated": row["updated_at"]} for row in rows])
        except Exception:
            # The app remains usable until the optional Supabase migration is run.
            return pd.DataFrame()
    initialize_database(database_path)
    with _connect(database_path) as connection:
        return pd.read_sql_query(
            """SELECT strategy_id AS "Strategy ID", allocation_amount AS "Paper allocation", risk_cap_pct AS "Risk cap",
            paused AS "Paused", updated_at AS "Updated" FROM copy_strategy_follows ORDER BY updated_at DESC""", connection)


def save_watchlist(
    name: str,
    source: str,
    symbols: list[str],
    asset_type: str,
    minimum_score: float,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> int | str:
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
        return str(response.data[0]["id"])
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


def list_watchlist_symbols(database_path: Path = DEFAULT_DATABASE_PATH) -> pd.DataFrame:
    """Return every saved symbol with its watchlist and declared asset type."""
    client, user_id = _cloud_context()
    if client and user_id:
        rows = client.table("watchlists").select("name,symbols").eq("user_id", user_id).execute().data
        return pd.DataFrame([
            {"Watchlist": row["name"], "Symbol": item["symbol"], "Asset type": item.get("asset_type", "")}
            for row in rows for item in row.get("symbols", [])
        ])
    initialize_database(database_path)
    with _connect(database_path) as connection:
        return pd.read_sql_query(
            """
            SELECT w.name AS "Watchlist", s.symbol AS "Symbol", s.asset_type AS "Asset type"
            FROM watchlists w JOIN watchlist_symbols s ON s.watchlist_id = w.id
            ORDER BY w.updated_at DESC, s.symbol
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
