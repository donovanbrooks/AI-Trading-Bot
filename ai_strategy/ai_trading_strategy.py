from models.price_predictor import train_price_model, predict_price_movement
from config import BUY_THRESHOLD, SELL_THRESHOLD

def ai_strategy(train_data, test_data):

    model = train_price_model(train_data)

    df = predict_price_movement(
        model,
        test_data
    )

    df["Signal"] = 0

    # AI + Moving Average Confirmation BUY
    df.loc[
        (df["Prediction"] == 1) &
        (df["Probability_Up"] >= 0.52) &
        (df["MA_10"] > df["MA_20"]) &
        (df["Close"] > df["MA_10"]) &
        (df["RSI"] < 70) &
        (df["Momentum"] > 0),
        "Signal"
    ] = 1

    # AI + Moving Average Confirmation SELL
    df.loc[
        (df["Prediction"] == 0) &
        (df["Probability_Up"] <= 0.40) &
        (df["MA_10"] < df["MA_20"]),
        "Signal"
    ] = -1

    print("\nAverage AI Confidence:")
    print(f"{df['Probability_Up'].mean():.2%}")
    print("\nAI Signals:")
    print(df[df["Signal"] != 0][[
        "Close",
        "Prediction",
        "Probability_Up",
        "MA_10",
        "MA_20",
        "RSI",
        "Signal"
    ]])

    return df