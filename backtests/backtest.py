def backtest_strategy(data, initial_balance=1000):
    balance = initial_balance
    shares = 0

    trades = []
    wins = 0
    losses = 0

    buy_price = 0

    for index, row in data.iterrows():

        price = row["Close"]

        # Buy signal
        if row["Signal"] == 1 and shares == 0:
            shares = balance / price
            balance = 0
            buy_price = price

            trades.append({
                "type": "BUY",
                "price": price,
                "date": index
            })

            print(f"BUY at {price:.2f}")

        # Sell signal
        elif row["Signal"] == -1 and shares > 0:
            balance = shares * price
            shares = 0

            profit = price - buy_price

            if profit > 0:
                wins += 1
            else:
                losses += 1

            trades.append({
                "type": "SELL",
                "price": price,
                "date": index
            })

            print(f"SELL at {price:.2f}")

    # Sell remaining shares
    if shares > 0:
        balance = shares * data.iloc[-1]["Close"]

    total_profit = balance - initial_balance
    percent_return = (total_profit / initial_balance) * 100

    total_trades = wins + losses

    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
    else:
        win_rate = 0


    print("\n--------------------")
    print("Strategy Performance Report")
    print("--------------------")
    print(f"Starting Balance: ${initial_balance:.2f}")
    print(f"Ending Balance: ${balance:.2f}")
    print(f"Profit/Loss: ${total_profit:.2f}")
    print(f"Return: {percent_return:.2f}%")
    print(f"Trades: {total_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")

    return balance