import pandas as pd

from market_data import _normalise_twelve_data_response


def test_normalise_twelve_data_response_returns_standard_ohlcv_columns():
    response = {
        "status": "ok",
        "values": [{
            "datetime": "2025-01-02", "open": "10", "high": "12", "low": "9", "close": "11", "volume": "1000",
        }],
    }

    bars = _normalise_twelve_data_response(response, "TEST:EXCHANGE")

    assert list(bars.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert bars.iloc[0]["Close"] == 11
