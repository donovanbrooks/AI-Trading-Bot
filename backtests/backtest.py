def backtest_strategy(data, initial_balance=1000, stop_loss=0.05, take_profit=0.10):
    balance = initial_balance
    shares = 0

    wins = 0
    losses = 0
    buy_price = 0

    for index, row in data.iterrows():

        price = row["Close"]

        # Buy
        if row["Signal"] == 1 and shares == 0:
            shares = balance / price
            balance = 0
            buy_price = price

            print(f"BUY at {price:.2f}")


        # Risk management while holding
        elif shares > 0:

            change = (price - buy_price) / buy_price

            # Stop loss
            if change <= -stop_loss:
                balance = shares * price
                shares = 0
                losses += 1

                print(f"STOP LOSS at {price:.2f}")


            # Take profit
            elif change >= take_profit:
                balance = shares * price
                shares = 0
                wins += 1

                print(f"TAKE PROFIT at {price:.2f}")


            # Normal sell signal
            elif row["Signal"] == -1:
                balance = shares * price
                shares = 0

                if price > buy_price:
                    wins += 1
                else:
                    losses += 1

                print(f"SELL at {price:.2f}")


    # Close remaining position
    if shares > 0:
        balance = shares * data.iloc[-1]["Close"]


    profit = balance - initial_balance
    percent_return = (profit / initial_balance) * 100

    total_trades = wins + losses

    win_rate = 0
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100


    print("\n--------------------")
    print("Risk Managed Performance Report")
    print("--------------------")
    print(f"Starting Balance: ${initial_balance:.2f}")
    print(f"Ending Balance: ${balance:.2f}")
    print(f"Profit/Loss: ${profit:.2f}")
    print(f"Return: {percent_return:.2f}%")
    print(f"Trades: {total_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")