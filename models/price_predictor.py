import pandas as pd

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

    df["Return"] = df["Close"].pct_change()

    df["Momentum"] = df["Close"].pct_change(periods=5)

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    df["MA_20"] = df["Close"].rolling(window=20).mean()

    # Exponential Moving Averages
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    

    df["Volume_Change"] = df["Volume"].pct_change()

    # Volatility (10-day rolling standard deviation)
    df["Volatility"] = df["Return"].rolling(window=10).std()

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

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
        "Return",
        "Momentum",
        "MA_10",
        "MA_20",
        "EMA_10",
        "EMA_20",
        "Volume_Change",
        "Volatility",
        "RSI"
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
        "Return",
        "Momentum",
        "MA_10",
        "MA_20",
        "EMA_10",
        "EMA_20",
        "Volume_Change",
        "Volatility",
        "RSI"
    ]

    probabilities = model.predict_proba(df[features])

    df["Probability_Up"] = probabilities[:, 1]
    df["Confidence"] = probabilities.max(axis=1)
    df["Prediction"] = model.predict(df[features])

    return df