import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_features(data):

    df = data.copy()

    df["Return"] = df["Close"].pct_change()

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    df["MA_20"] = df["Close"].rolling(window=20).mean()

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
        "MA_10",
        "MA_20",
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
def predict_price_movement(model, data):

    df = create_features(data)

    df = df.dropna()

    features = [
        "Return",
        "MA_10",
        "MA_20",
        "Volume_Change",
        "Volatility",
        "RSI"
    ]

    df["Prediction"] = model.predict(
        df[features]
    )

    return df