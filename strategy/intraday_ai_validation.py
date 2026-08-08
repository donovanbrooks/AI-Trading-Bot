"""Intraday, expanding-window AI signals for research and paper trading.

The model uses five-minute bars only during the regular US equity session. It
never carries a simulated position overnight and delays every signal by one
bar, so it cannot trade using information from the execution bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


INTRADAY_FEATURES = [
    "return_1",
    "return_3",
    "rsi_14",
    "volatility_12",
    "range_pct",
    "vwap_gap",
    "volume_zscore",
]


def bullish_candlestick_patterns(data: pd.DataFrame) -> pd.Series:
    """Identify conservative bullish engulfing and hammer confirmation bars."""
    previous_open, previous_close = data["Open"].shift(1), data["Close"].shift(1)
    body = (data["Close"] - data["Open"]).abs().clip(lower=1e-12)
    lower_wick = data[["Open", "Close"]].min(axis=1) - data["Low"]
    upper_wick = data["High"] - data[["Open", "Close"]].max(axis=1)
    engulfing = (
        (previous_close < previous_open)
        & (data["Close"] > data["Open"])
        & (data["Open"] <= previous_close)
        & (data["Close"] >= previous_open)
    )
    hammer = (data["Close"] > data["Open"]) & (lower_wick >= 2 * body) & (upper_wick <= body)
    return (engulfing | hammer).fillna(False)


def _regular_session(data: pd.DataFrame) -> pd.DataFrame:
    """Keep 09:35–15:55 New York bars, leaving room for a next-bar exit."""
    frame = data.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Intraday data requires a DatetimeIndex")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index = frame.index.tz_convert("America/New_York")
    frame["Session Date"] = frame.index.date
    frame["Session Time"] = frame.index.time
    return frame.between_time("09:35", "15:55").copy()


def create_intraday_features(data: pd.DataFrame, horizon_bars: int = 3, minimum_move_bps: float = 8) -> pd.DataFrame:
    """Create features and a cost-aware, same-session forward-return target."""
    frame = _regular_session(data)
    frame["return_1"] = frame["Close"].pct_change()
    frame["return_3"] = frame["Close"].pct_change(3)

    change = frame["Close"].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    frame["rsi_14"] = 100 - 100 / (1 + gain / loss)
    frame["volatility_12"] = frame["return_1"].rolling(12).std()
    frame["range_pct"] = (frame["High"] - frame["Low"]) / frame["Close"]

    typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    cumulative_volume = frame.groupby("Session Date")["Volume"].cumsum()
    frame["vwap"] = (typical_price * frame["Volume"]).groupby(frame["Session Date"]).cumsum() / cumulative_volume
    frame["vwap_gap"] = frame["Close"] / frame["vwap"] - 1
    volume_mean = frame.groupby("Session Date")["Volume"].transform(lambda values: values.rolling(20).mean())
    volume_std = frame.groupby("Session Date")["Volume"].transform(lambda values: values.rolling(20).std())
    frame["volume_zscore"] = (frame["Volume"] - volume_mean) / volume_std

    future_close = frame["Close"].shift(-horizon_bars)
    same_session = frame["Session Date"] == frame["Session Date"].shift(-horizon_bars)
    threshold = minimum_move_bps / 10_000
    frame["target"] = (future_close / frame["Close"] - 1 > threshold).where(same_session)
    return frame.dropna(subset=INTRADAY_FEATURES).copy()


def generate_intraday_ai_signals(
    data: pd.DataFrame,
    train_bars: int = 780,
    retrain_bars: int = 78,
    probability_threshold: float = 0.58,
    horizon_bars: int = 3,
    minimum_move_bps: float = 8,
    require_bullish_candle: bool = False,
) -> pd.DataFrame:
    """Return five-minute signals fitted only on earlier intraday observations."""
    if not 0.5 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0.5 and 1")
    features = create_intraday_features(data, horizon_bars, minimum_move_bps)
    if len(features) <= train_bars:
        raise ValueError("Not enough intraday bars for the selected training window")

    result = features[["Open", "High", "Low", "Close", "Volume", "Session Date", "Session Time"]].copy()
    result["AI Probability"] = np.nan
    result["Signal"] = 0
    result["Bullish Candle"] = bullish_candlestick_patterns(result)
    for start in range(train_bars, len(features), retrain_bars):
        training = features.iloc[:start].dropna(subset=["target"])
        prediction = features.iloc[start:start + retrain_bars]
        if len(training) < train_bars or prediction.empty or training["target"].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(training[INTRADAY_FEATURES], training["target"].astype(int))
        result.loc[prediction.index, "AI Probability"] = model.predict_proba(prediction[INTRADAY_FEATURES])[:, 1]

    result.loc[result["AI Probability"] >= probability_threshold, "Signal"] = 1
    result.loc[result["AI Probability"] <= 1 - probability_threshold, "Signal"] = -1
    if require_bullish_candle:
        result.loc[(result["Signal"] == 1) & ~result["Bullish Candle"], "Signal"] = 0
    # Flatten by 15:55 ET: the prior 15:50 signal is filled at 15:55.
    result.loc[result["Session Time"].astype(str) == "15:50:00", "Signal"] = -1
    result["Execution Signal"] = result["Signal"].shift(1).fillna(0).astype(int)
    return result
