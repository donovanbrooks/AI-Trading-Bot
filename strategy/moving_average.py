def moving_average_strategy(data, short_window=20, long_window=50):

    data = data.copy()

    # Moving averages
    data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
    data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

    # Momentum feature
    data["Momentum"] = data["Close"].pct_change(periods=5)

    # Current trend signal
    data["MA_Signal"] = 0

    data.loc[
        data["Short_MA"] > data["Long_MA"],
        "MA_Signal"
    ] = 1

    data.loc[
        data["Short_MA"] < data["Long_MA"],
        "MA_Signal"
    ] = -1


    # Generate actual buy/sell crossover signals
    data["Signal"] = data["MA_Signal"].diff()


    # Convert crossover values into trading signals
    data.loc[
        data["Signal"] > 0,
        "Signal"
    ] = 1

    data.loc[
        data["Signal"] < 0,
        "Signal"
    ] = -1


    # Fill initial NaN values
    data["Signal"] = data["Signal"].fillna(0)


    return data