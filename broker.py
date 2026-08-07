"""Alpaca paper-account connection helpers.

The only order pathway is a deliberately small, manually confirmed paper buy.
"""

from __future__ import annotations

import os
import re
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv


class BrokerConfigurationError(RuntimeError):
    """Raised when local paper-trading credentials are unavailable or unsafe."""


MAX_PAPER_ORDER_NOTIONAL = 25.0


def _paper_client():
    """Create the Alpaca SDK client with the paper environment forced on."""
    load_dotenv()
    if os.getenv("ALPACA_PAPER", "true").lower() != "true":
        raise BrokerConfigurationError("This app only permits ALPACA_PAPER=true.")

    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise BrokerConfigurationError(
            "Add APCA_API_KEY_ID and APCA_API_SECRET_KEY to your local .env file."
        )

    try:
        from alpaca.trading.client import TradingClient
    except ImportError as error:
        raise BrokerConfigurationError("Install dependencies with: pip install -r requirements.txt") from error
    return TradingClient(api_key, api_secret, paper=True)


def get_paper_account_summary() -> dict[str, Any]:
    """Return a small account summary using Alpaca paper credentials only."""
    client = _paper_client()
    account = client.get_account()
    return {
        "status": str(account.status),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
        "account_number": str(account.account_number),
    }


def get_paper_portfolio() -> dict[str, list[dict[str, str]]]:
    """Fetch current paper positions and the ten most recent paper orders."""
    client = _paper_client()
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    positions = [
        {
            "Symbol": str(position.symbol),
            "Quantity": str(position.qty),
            "Market Value": str(position.market_value),
            "Unrealized P&L": str(position.unrealized_pl),
        }
        for position in client.get_all_positions()
    ]
    orders = client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10, nested=False)
    )
    recent_orders = [
        {
            "Submitted": str(order.submitted_at),
            "Symbol": str(order.symbol),
            "Side": str(order.side),
            "Amount": str(order.notional or order.qty),
            "Status": str(order.status),
            "Order ID": str(order.id),
        }
        for order in orders
    ]
    return {"positions": positions, "orders": recent_orders}


def submit_confirmed_paper_buy(symbol: str, notional: float) -> dict[str, str]:
    """Submit one small market buy after broker-side and local safety checks."""
    normalized_symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z.]{1,10}", normalized_symbol):
        raise BrokerConfigurationError("Enter a valid US stock symbol, such as SPY or AAPL.")
    if not 1 <= notional <= MAX_PAPER_ORDER_NOTIONAL:
        raise BrokerConfigurationError(f"Paper orders must be between $1 and ${MAX_PAPER_ORDER_NOTIONAL:.0f}.")

    client = _paper_client()
    clock = client.get_clock()
    if not clock.is_open:
        raise BrokerConfigurationError(f"US market is closed. Next open: {clock.next_open}.")

    asset = client.get_asset(normalized_symbol)
    if not asset.tradable or str(asset.asset_class) not in {"us_equity", "AssetClass.US_EQUITY"}:
        raise BrokerConfigurationError(f"{normalized_symbol} is not a tradable US equity in this account.")

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    order = client.submit_order(
        order_data=MarketOrderRequest(
            symbol=normalized_symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=f"bot-paper-{uuid4().hex[:20]}",
        )
    )
    return {"id": str(order.id), "symbol": str(order.symbol), "status": str(order.status)}


def submit_confirmed_paper_crypto_buy(symbol: str, notional: float) -> dict[str, str]:
    """Submit one small, manually confirmed 24/7 crypto paper market buy."""
    normalized_symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z]{2,10}/[A-Z]{2,10}", normalized_symbol):
        raise BrokerConfigurationError("Enter a crypto pair such as BTC/USD.")
    if not 1 <= notional <= MAX_PAPER_ORDER_NOTIONAL:
        raise BrokerConfigurationError(f"Paper orders must be between $1 and ${MAX_PAPER_ORDER_NOTIONAL:.0f}.")

    client = _paper_client()
    asset = client.get_asset(normalized_symbol)
    if not asset.tradable or str(asset.asset_class) not in {"crypto", "AssetClass.CRYPTO"}:
        raise BrokerConfigurationError(f"{normalized_symbol} is not a tradable crypto asset in this account.")

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    order = client.submit_order(
        order_data=MarketOrderRequest(
            symbol=normalized_symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            client_order_id=f"bot-crypto-paper-{uuid4().hex[:16]}",
        )
    )
    return {"id": str(order.id), "symbol": str(order.symbol), "status": str(order.status)}
