def moving_average_strategy(data, short_window=20, long_window=50):

    data = data.copy()

    data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
    data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

    # Momentum feature
    data["Momentum"] = data["Close"].pct_change(periods=5)

    data.loc[data["Short_MA"] > data["Long_MA"], "MA_Signal"] = 1
    data.loc[data["Short_MA"] < data["Long_MA"], "MA_Signal"] = -1

    return data