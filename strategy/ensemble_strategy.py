def ensemble_strategy(data):

    df = data.copy()

    df["Ensemble_Score"] = 0

    # Moving average vote
    df.loc[df["MA_Signal"] == 1, "Ensemble_Score"] += 1
    df.loc[df["MA_Signal"] == -1, "Ensemble_Score"] -= 1

    # AI vote
    df.loc[df["AI_Signal"] == 1, "Ensemble_Score"] += 1
    df.loc[df["AI_Signal"] == -1, "Ensemble_Score"] -= 1


    # Final decision

    df["Signal"] = 0

    df.loc[
        df["Ensemble_Score"] >= 2,
        "Signal"
    ] = 1


    df.loc[
        df["Ensemble_Score"] <= -2,
        "Signal"
    ] = -1


    return df