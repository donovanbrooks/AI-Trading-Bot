import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)


def create_features(data):

    df = data.copy()

    # Returns
    df["Return"] = df["Close"].pct_change()
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_10"] = df["Close"].pct_change(10)
    df["Return_20"] = df["Close"].pct_change(20)

    # Momentum
    df["Momentum"] = df["Close"].pct_change(5)

    # Moving averages
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    # Exponential moving averages
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # ---------------- MACD ----------------
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ---------------- Bollinger Bands ----------------
    rolling_std = df["Close"].rolling(20).std()

    df["BB_Upper"] = df["MA_20"] + 2 * rolling_std
    df["BB_Lower"] = df["MA_20"] - 2 * rolling_std
    df["BB_Width"] = (
        df["BB_Upper"] - df["BB_Lower"]
    ) / df["MA_20"]

    # ---------------- RSI ----------------
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # ---------------- Stochastic RSI ----------------
    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()

    df["Stoch_RSI"] = (
        (df["RSI"] - rsi_min) /
        (rsi_max - rsi_min)
    )

    # ---------------- ATR ----------------
    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift()
    ).abs()

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # ---------------- Other ----------------
    df["Volume_Change"] = df["Volume"].pct_change()

    df["Volatility"] = (
        df["Return"]
        .rolling(10)
        .std()
    )

    return df

def train_price_model(data):

    df = create_features(data)


    # Create target
    # 1 = tomorrow goes up
    # 0 = tomorrow goes down

    df["Target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)


    # Remove missing values

    df = df.dropna()

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_10",
        "MA_20",
        "EMA_10",
        "EMA_20",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "Stoch_RSI",
        "Return_5",
        "Return_10",
        "Return_20",
        "Momentum",
        "Volatility",
        "ATR"
    ]


    X = df[features]
    y = df["Target"]


    # Split data

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )


    # Train model

    from config import RANDOM_FOREST_TREES, RANDOM_SEED

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    rf_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_params,
        n_iter=20,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    rf_search.fit(X_train, y_train)

    print("\nBest Random Forest Parameters:")
    print(rf_search.best_params_)

    rf = rf_search.best_estimator_

    model = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("et", et)
        ],
        voting="soft"
    )

    model.fit(
        X_train,
        y_train
    )


    # Test accuracy

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n--------------------")
    print("AI Price Prediction Model")
    print("--------------------")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1:.2%}")


    return model

def predict_price_movement(model, data):

    df = create_features(data)

    df = df.dropna()

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_10",
        "MA_20",
        "EMA_10",
        "EMA_20",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "Stoch_RSI",
        "Return_5",
        "Return_10",
        "Return_20",
        "Momentum",
        "Volatility",
        "ATR"
    ]

    print(df.columns.tolist())

    missing = [col for col in features if col not in df.columns]

    if missing:
        print("Missing columns:", missing)
        raise ValueError("Feature engineering is incomplete.")
    
    probabilities = model.predict_proba(df[features])

    df["Probability_Up"] = probabilities[:, 1]
    df["Confidence"] = probabilities.max(axis=1)
    df["Prediction"] = model.predict(df[features])

    return df