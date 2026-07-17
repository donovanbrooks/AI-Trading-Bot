from models.price_predictor import train_price_model, predict_price_movement


def ai_strategy(train_data, test_data):

    model = train_price_model(train_data)

    df = predict_price_movement(
        model,
        test_data
    )

    df["Signal"] = 0

    df.loc[df["Prediction"] == 1, "Signal"] = 1

    df.loc[df["Prediction"] == 0, "Signal"] = -1

    return df