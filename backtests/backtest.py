def backtest_strategy(data, initial_balance=1000):
    balance = initial_balance
    shares = 0

    for index, row in data.iterrows():

        # Buy signal
        if row["Signal"] == 1 and shares == 0:
            shares = balance / row["Close"]
            balance = 0
            print(f"BUY at {row['Close']}")

        # Sell signal
        elif row["Signal"] == -1 and shares > 0:
            balance = shares * row["Close"]
            shares = 0
            print(f"SELL at {row['Close']}")

    # If still holding stock, sell at last price
    if shares > 0:
        balance = shares * data.iloc[-1]["Close"]

    profit = balance - initial_balance
    percent_return = (profit / initial_balance) * 100

    print("--------------------")
    print(f"Starting Balance: ${initial_balance:.2f}")
    print(f"Ending Balance: ${balance:.2f}")
    print(f"Profit/Loss: ${profit:.2f}")
    print(f"Return: {percent_return:.2f}%")

    return balance