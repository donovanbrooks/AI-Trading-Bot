import pandas as pd


def calculate_atr(df, period=14):

    df = df.copy()

    high_low = df["High"] - df["Low"]

    high_close = abs(
        df["High"] - df["Close"].shift()
    )

    low_close = abs(
        df["Low"] - df["Close"].shift()
    )

    ranges = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    )

    true_range = ranges.max(axis=1)

    df["ATR"] = (
        true_range
        .rolling(period)
        .mean()
    )

    return df