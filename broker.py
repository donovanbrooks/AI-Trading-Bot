"""Alpaca paper-account connection helpers.

The only order pathway is a deliberately small, manually confirmed paper buy.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from credentials import CredentialError
from logging_config import configure_logging
from storage import get_paper_broker_credentials


class BrokerConfigurationError(RuntimeError):
    """Raised when local paper-trading credentials are unavailable or unsafe."""


MAX_PAPER_ORDER_NOTIONAL = 25.0
logger = configure_logging()


def _paper_client():
    """Create the Alpaca SDK client with the paper environment forced on."""
    try:
        api_key, api_secret = get_paper_broker_credentials()
    except CredentialError as error:
        raise BrokerConfigurationError(str(error)) from error

    try:
        from alpaca.trading.client import TradingClient
    except ImportError as error:
        raise BrokerConfigurationError("Install dependencies with: pip install -r requirements.txt") from error
    return TradingClient(api_key, api_secret, paper=True)


def validate_paper_credentials(api_key: str, api_secret: str) -> dict[str, Any]:
    """Verify credentials against Alpaca's paper endpoint without storing them."""
    if not api_key.strip() or not api_secret.strip():
        raise BrokerConfigurationError("Both the Alpaca paper API key and secret are required.")
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as error:
        raise BrokerConfigurationError("Install dependencies with: pip install -r requirements.txt") from error
    try:
        account = TradingClient(api_key.strip(), api_secret.strip(), paper=True).get_account()
    except Exception as error:
        raise BrokerConfigurationError("Alpaca could not validate those paper credentials. Check that they are paper keys.") from error
    return {"account_number": str(account.account_number), "status": str(account.status)}


def get_paper_account_summary() -> dict[str, Any]:
    """Return a small account summary using Alpaca paper credentials only."""
    client = _paper_client()
    account = client.get_account()
    return {
        "status": str(account.status),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
        "last_equity": float(account.last_equity),
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
            "Average Entry": str(position.avg_entry_price),
            "Current Price": str(position.current_price),
            "Market Value": str(position.market_value),
            "Unrealized P&L": str(position.unrealized_pl),
            "Unrealized P&L %": str(position.unrealized_plpc),
            "Today's P&L": str(position.change_today),
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


def lookup_paper_asset(symbol: str, asset_type: str) -> dict[str, str]:
    """Validate a user-selected stock or crypto pair before an order review."""
    normalized_symbol = symbol.strip().upper()
    if asset_type == "crypto":
        normalized_symbol = normalized_symbol.replace("-", "/")
        valid = re.fullmatch(r"[A-Z]{2,10}/[A-Z]{2,10}", normalized_symbol)
    else:
        valid = re.fullmatch(r"[A-Z.]{1,10}", normalized_symbol)
    if not valid:
        example = "BTC/USD" if asset_type == "crypto" else "AAPL"
        raise BrokerConfigurationError(f"Enter a valid {asset_type} symbol, such as {example}.")
    asset = _paper_client().get_asset(normalized_symbol)
    actual_type = str(asset.asset_class).lower()
    if not asset.tradable or (asset_type == "crypto" and "crypto" not in actual_type) or (asset_type == "equity" and "equity" not in actual_type):
        raise BrokerConfigurationError(f"{normalized_symbol} is not a tradable Alpaca paper {asset_type} asset.")
    return {"symbol": normalized_symbol, "name": str(getattr(asset, "name", normalized_symbol)), "asset_type": asset_type}


def _assert_safe_to_open(client: Any, symbol: str) -> None:
    """Block duplicate orders and additional exposure in an existing position."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    if any(str(position.symbol) == symbol for position in client.get_all_positions()):
        raise BrokerConfigurationError(f"A paper position in {symbol} already exists. Close or reconcile it before opening another.")
    open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100, nested=False))
    if any(str(order.symbol) == symbol and str(order.side).lower().endswith("buy") for order in open_orders):
        raise BrokerConfigurationError(f"An open paper buy order for {symbol} already exists.")


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
    _assert_safe_to_open(client, normalized_symbol)

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
    logger.info("Submitted paper equity buy: symbol=%s notional=%.2f order_id=%s", normalized_symbol, notional, order.id)
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
    _assert_safe_to_open(client, normalized_symbol)

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
    logger.info("Submitted paper crypto buy: symbol=%s notional=%.2f order_id=%s", normalized_symbol, notional, order.id)
    return {"id": str(order.id), "symbol": str(order.symbol), "status": str(order.status)}
