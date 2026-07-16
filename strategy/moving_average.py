def moving_average_strategy(data):
    data["Short_MA"] = data["Close"].rolling(window=20).mean()
    data["Long_MA"] = data["Close"].rolling(window=50).mean()

    data["Signal"] = 0

    data.loc[data["Short_MA"] > data["Long_MA"], "Signal"] = 1
    data.loc[data["Short_MA"] < data["Long_MA"], "Signal"] = -1

    return data