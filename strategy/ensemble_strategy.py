def ensemble_strategy(df):

    df = df.copy()

    df["Ensemble_Score"] = 0


    # Moving Average
    df.loc[df["MA_Signal"] == 1,"Ensemble_Score"] += 1
    df.loc[df["MA_Signal"] == -1,"Ensemble_Score"] -= 1


    # AI Prediction

    # Strong AI confidence

    df.loc[
        df["Probability_Up"] >= 0.60,
        "Ensemble_Score"
    ] += 4

    # Moderate AI confidence

    df.loc[
        (df["Probability_Up"] >= 0.52) &
        (df["Probability_Up"] < 0.60),
        "Ensemble_Score"
    ] += 2

    # Bearish AI

    df.loc[
        df["Probability_Up"] <= 0.40,
        "Ensemble_Score"
    ] -= 3


    df.loc[
        (df["AI_Signal"] == -1) &
        (df["Probability_Up"] <= 0.45),
        "Ensemble_Score"
    ] -= 3


    # Momentum
    df.loc[df["Momentum"] > 0,"Ensemble_Score"] += 1
    df.loc[df["Momentum"] < 0,"Ensemble_Score"] -= 1


    # RSI filter
    df.loc[df["RSI"] < 70,"Ensemble_Score"] += 1
    df.loc[df["RSI"] > 75,"Ensemble_Score"] -= 1


    df["Signal"] = 0


    df.loc[
        df["Ensemble_Score"] >= 3,
        "Signal"
    ] = 1


    df.loc[
        df["Ensemble_Score"] <= -3,
        "Signal"
    ] = -1

    trailing_stop = 0.02

    return df