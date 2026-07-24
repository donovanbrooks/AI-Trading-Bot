def ensemble_strategy(df):

    print("Inside ensemble:")
    print(df.columns.tolist())

    df = df.copy()

    df["Ensemble_Score"] = 0.0

    # ==========================
    # MOVING AVERAGE
    # ==========================

    df.loc[df["MA_Signal"] == 1, "Ensemble_Score"] += 2
    df.loc[df["MA_Signal"] == -1, "Ensemble_Score"] -= 2

    # ==========================
    # AI PREDICTION
    # ==========================

    # Strong bullish AI
    df.loc[
        df["Probability_Up"] >= 0.55,
        "Ensemble_Score"
    ] += 2

    # Weak bullish AI
    df.loc[
        (df["Probability_Up"] >= 0.50) &
        (df["Probability_Up"] < 0.55),
        "Ensemble_Score"
    ] += 1

    # Bearish AI
    df.loc[
        df["Probability_Up"] <= 0.40,
        "Ensemble_Score"
    ] -= 2

    df.loc[
        df["Prediction"] == 1,
        "Ensemble_Score"
    ] += 1


    # ==========================
    # MOMENTUM
    # ==========================

    df.loc[
        df["Momentum"] > 0,
        "Ensemble_Score"
    ] += 1

    df.loc[
        df["Momentum"] < 0,
        "Ensemble_Score"
    ] -= 1

    # ==========================
    # RSI
    # ==========================

    # Avoid buying extreme overbought
    df.loc[
        df["RSI"] > 80,
        "Ensemble_Score"
    ] -= 1

    # Reward healthy RSI
    df.loc[
        (df["RSI"] > 40) &
        (df["RSI"] < 70),
        "Ensemble_Score"
    ] += 1

    # ==========================
    # FINAL SIGNAL
    # ==========================

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

    print(df[[
        "Probability_Up",
        "Prediction",
        "AI_Signal"
    ]].tail(20))

    return df