def ensemble_strategy(df):

    df = df.copy()

    # Reset score
    df["Ensemble_Score"] = 0


    # Moving Average contribution
    df.loc[
        df["MA_Signal"] == 1,
        "Ensemble_Score"
    ] += 1

    df.loc[
        df["MA_Signal"] == -1,
        "Ensemble_Score"
    ] -= 1

    # AI contribution based on confidence

    df.loc[
        (df["AI_Signal"] == 1) &
        (df["Probability_Up"] >= 0.60),
        "Ensemble_Score"
    ] += 3

    df.loc[
        (df["AI_Signal"] == 1) &
        (df["Probability_Up"] < 0.60),
        "Ensemble_Score"
    ] += 1

    df.loc[
        (df["AI_Signal"] == -1) &
        (df["Probability_Up"] <= 0.40),
        "Ensemble_Score"
    ] -= 3

    # Momentum confirmation
    df.loc[
        df["Momentum"] > 0,
        "Ensemble_Score"
    ] += 1

    df.loc[
        df["Momentum"] < 0,
        "Ensemble_Score"
    ] -= 1

    # Final signal
    df["Signal"] = 0


    # Require stronger agreement
    df.loc[
        df["Ensemble_Score"] >= 3,
        "Signal"
    ] = 1

    df.loc[
        df["Ensemble_Score"] <= -3,
        "Signal"
    ] = -1


    return df