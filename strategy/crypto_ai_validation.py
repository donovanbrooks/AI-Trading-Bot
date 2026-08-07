"""24/7 intraday AI signals for crypto research and paper trading."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


CRYPTO_FEATURES = ["return_1", "return_6", "rsi_14", "volatility_24", "range_pct", "volume_zscore"]


def create_crypto_features(data: pd.DataFrame, horizon_bars: int = 6, minimum_move_bps: float = 10) -> pd.DataFrame:
    """Build five-minute crypto features without session-based assumptions."""
    frame = data.copy()
    frame["return_1"] = frame["Close"].pct_change()
    frame["return_6"] = frame["Close"].pct_change(6)
    change = frame["Close"].diff()
    gains = change.clip(lower=0).rolling(14).mean()
    losses = (-change.clip(upper=0)).rolling(14).mean()
    frame["rsi_14"] = 100 - 100 / (1 + gains / losses)
    frame["volatility_24"] = frame["return_1"].rolling(24).std()
    frame["range_pct"] = (frame["High"] - frame["Low"]) / frame["Close"]
    volume_mean = frame["Volume"].rolling(48).mean()
    volume_std = frame["Volume"].rolling(48).std()
    frame["volume_zscore"] = (frame["Volume"] - volume_mean) / volume_std
    future_close = frame["Close"].shift(-horizon_bars)
    frame["target"] = (future_close / frame["Close"] - 1 > minimum_move_bps / 10_000).where(future_close.notna())
    return frame.dropna(subset=CRYPTO_FEATURES).copy()


def generate_crypto_ai_signals(
    data: pd.DataFrame,
    train_bars: int = 2_016,
    retrain_bars: int = 288,
    probability_threshold: float = 0.60,
    horizon_bars: int = 6,
    minimum_move_bps: float = 10,
) -> pd.DataFrame:
    """Generate 24/7 signals from expanding-window, five-minute crypto data."""
    if not 0.5 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0.5 and 1")
    features = create_crypto_features(data, horizon_bars, minimum_move_bps)
    if len(features) <= train_bars:
        raise ValueError("Not enough crypto bars for the selected training window")

    result = features[["Open", "High", "Low", "Close", "Volume"]].copy()
    result["AI Probability"] = np.nan
    result["Signal"] = 0
    for start in range(train_bars, len(features), retrain_bars):
        training = features.iloc[:start].dropna(subset=["target"])
        prediction = features.iloc[start:start + retrain_bars]
        if len(training) < train_bars or prediction.empty or training["target"].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(training[CRYPTO_FEATURES], training["target"].astype(int))
        result.loc[prediction.index, "AI Probability"] = model.predict_proba(prediction[CRYPTO_FEATURES])[:, 1]

    result.loc[result["AI Probability"] >= probability_threshold, "Signal"] = 1
    result.loc[result["AI Probability"] <= 1 - probability_threshold, "Signal"] = -1
    result["Execution Signal"] = result["Signal"].shift(1).fillna(0).astype(int)
    return result
