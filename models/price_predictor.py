import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_price_model(data):

    df = data.copy()

    # Create features
    df["Return"] = df["Close"].pct_change()

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    df["MA_20"] = df["Close"].rolling(window=20).mean()

    df["Volume_Change"] = df["Volume"].pct_change()


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
        "MA_10",
        "MA_20",
        "Volume_Change"
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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
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


    print("\n--------------------")
    print("AI Price Prediction Model")
    print("--------------------")
    print(f"Accuracy: {accuracy:.2%}")


    return model