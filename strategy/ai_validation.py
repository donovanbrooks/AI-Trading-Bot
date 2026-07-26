"""Time-series-safe AI signal generation for research dashboards."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "ma_gap",
    "rsi_14",
    "volatility_10",
    "volume_change",
]


def create_ai_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features that use only current and earlier market data."""
    frame = data.copy()
    frame["return_1"] = frame["Close"].pct_change()
    frame["return_5"] = frame["Close"].pct_change(5)
    short_ma = frame["Close"].rolling(10).mean()
    long_ma = frame["Close"].rolling(30).mean()
    frame["ma_gap"] = short_ma / long_ma - 1

    change = frame["Close"].diff()
    gains = change.clip(lower=0).rolling(14).mean()
    losses = (-change.clip(upper=0)).rolling(14).mean()
    frame["rsi_14"] = 100 - 100 / (1 + gains / losses)
    frame["volatility_10"] = frame["return_1"].rolling(10).std()
    frame["volume_change"] = frame["Volume"].pct_change()
    # Keep target as nullable until after the final row is removed. Casting a
    # final missing target to zero silently creates a false training example.
    frame["target"] = (frame["Close"].shift(-1) > frame["Close"]).where(frame["Close"].shift(-1).notna())
    return frame.dropna(subset=FEATURE_COLUMNS).copy()


def generate_ai_signals(
    data: pd.DataFrame,
    train_bars: int = 252,
    retrain_bars: int = 21,
    probability_threshold: float = 0.55,
) -> pd.DataFrame:
    """Generate expanding-window predictions without training on future rows.

    Each block is predicted by a model fit only on observations before that
    block. The resulting signal is filled at the next session's open.
    """
    if not 0.5 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0.5 and 1")

    features = create_ai_features(data)
    result = data.copy()
    result["AI Probability"] = pd.NA
    result["Signal"] = 0

    # Feature rows can begin later than raw price rows, so work on their index
    # and map outputs back to the full price dataframe.
    for start in range(train_bars, len(features), retrain_bars):
        training = features.iloc[:start].dropna(subset=["target"])
        prediction = features.iloc[start:start + retrain_bars]
        if len(training) < train_bars or prediction.empty or training["target"].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(training[FEATURE_COLUMNS], training["target"].astype(int))
        probabilities = model.predict_proba(prediction[FEATURE_COLUMNS])[:, 1]
        result.loc[prediction.index, "AI Probability"] = probabilities

    probability = pd.to_numeric(result["AI Probability"], errors="coerce")
    result.loc[probability >= probability_threshold, "Signal"] = 1
    result.loc[probability <= 1 - probability_threshold, "Signal"] = -1
    result["Execution Signal"] = result["Signal"].shift(1).fillna(0).astype(int)
    return result
