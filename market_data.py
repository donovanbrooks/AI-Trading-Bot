"""Alpaca-backed market-data access for the private app."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv


class MarketDataError(RuntimeError):
    """Raised when market data cannot be loaded safely."""


def _credentials() -> tuple[str, str]:
    load_dotenv()
    key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise MarketDataError("Alpaca API credentials are required for production market data.")
    return key, secret


def _days_for_period(period: str) -> int:
    mapping = {"5d": 5, "30d": 30, "60d": 60, "1y": 365, "2y": 730, "5y": 1825}
    try:
        return mapping[period]
    except KeyError as error:
        raise MarketDataError(f"Unsupported history period: {period}") from error


def _normalise_bars(dataframe: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if dataframe.empty:
        raise MarketDataError(f"No market data was returned for {symbol}.")
    frame = dataframe.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.xs(symbol, level="symbol")
    frame.index.name = "timestamp"
    frame = frame.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise MarketDataError(f"Market data is missing columns: {sorted(missing)}")
    return frame[required].dropna().sort_index()


def load_alpaca_bars(ticker: str, period: str, intraday: bool, crypto: bool = False) -> pd.DataFrame:
    """Fetch daily or five-minute bars from Alpaca's data API."""
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=_days_for_period(period))
    timeframe = TimeFrame(5, TimeFrameUnit.Minute) if intraday else TimeFrame.Day
    if crypto:
        symbol = ticker.upper().replace("-", "/")
        key, secret = _credentials()
        client = CryptoHistoricalDataClient(key, secret)
        bars = client.get_crypto_bars(CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end))
    else:
        symbol = ticker.upper()
        key, secret = _credentials()
        client = StockHistoricalDataClient(key, secret)
        bars = client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end, feed=DataFeed.IEX)
        )
    return _normalise_bars(bars.df, symbol)
