from models.price_predictor import train_price_model, predict_price_movement
from config import BUY_THRESHOLD, SELL_THRESHOLD

def ai_strategy(train_data, test_data):

    model = train_price_model(train_data)

    df = predict_price_movement(
        model,
        test_data
    )

    df["Signal"] = 0

    df.loc[df["Probability_Up"] >= BUY_THRESHOLD, "Signal"] = 1

    df.loc[df["Probability_Up"] <= SELL_THRESHOLD, "Signal"] = -1

    print("\nAverage AI Confidence:")
    print(f"{df['Probability_Up'].mean():.2%}")

    return df