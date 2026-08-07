from types import SimpleNamespace

import pandas as pd

import broker
from auth import create_password_hash, verify_password
from market_data import _normalise_bars
from regime_analysis import regime_performance
from storage import paper_order_ledger, recent_paper_order, record_paper_order


def test_password_hash_verifies_only_the_matching_password():
    stored_hash = create_password_hash("this is a strong test password")
    assert verify_password("this is a strong test password", stored_hash)
    assert not verify_password("wrong password", stored_hash)


def test_normalise_alpaca_bar_dataframe():
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    index = pd.MultiIndex.from_tuples([("SPY", timestamp)], names=["symbol", "timestamp"])
    raw = pd.DataFrame({"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [10]}, index=index)

    bars = _normalise_bars(raw, "SPY")

    assert list(bars.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert float(bars.iloc[0]["Close"]) == 100.5


def test_regime_report_has_rows():
    index = pd.date_range("2025-01-01", periods=60, freq="B")
    results = pd.DataFrame({"Close": range(100, 160), "Equity": range(1_000, 1_060)}, index=index)

    report = regime_performance(results)

    assert not report.empty
    assert {"Trend", "Volatility", "Strategy_Return"}.issubset(report.columns)


def test_paper_buy_rejects_duplicate_position():
    fake = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(is_open=True, next_open="later"),
        get_asset=lambda symbol: SimpleNamespace(tradable=True, asset_class="us_equity"),
        get_all_positions=lambda: [SimpleNamespace(symbol="SPY")],
        get_orders=lambda filter: [],
    )
    original_client = broker._paper_client
    broker._paper_client = lambda: fake
    try:
        try:
            broker.submit_confirmed_paper_buy("SPY", 5)
        except broker.BrokerConfigurationError as error:
            assert "position" in str(error).lower()
        else:
            raise AssertionError("Expected existing-position check to reject the order")
    finally:
        broker._paper_client = original_client


def test_paper_order_ledger_blocks_recent_duplicate(tmp_path):
    database = tmp_path / "trading_bot.db"
    record_paper_order("order-1", "SPY", "equity", 5, "accepted", database)

    assert recent_paper_order("SPY", database_path=database)
    assert len(paper_order_ledger(database)) == 1
