import csv
from datetime import datetime
from config import (
    INITIAL_BALANCE,
    ATR_STOP_MULTIPLIER,
    ATR_TARGET_MULTIPLIER
)

def risk_managed_backtest(
    data,
    initial_balance=INITIAL_BALANCE,
    verbose=True
):
    balance = initial_balance
    shares = 0

    wins = 0
    losses = 0
    buy_price = 0
    stop_price = None
    target_price = None
    trade_history = []

    for index, row in data.iterrows():

        price = row["Close"]

        # Buy
        if row["Signal"] == 1 and shares == 0:

            shares = balance / price
            balance = 0

            buy_price = price
            buy_atr = row["ATR"]

            stop_price = buy_price - (buy_atr * ATR_STOP_MULTIPLIER)
            target_price = buy_price + (buy_atr * ATR_TARGET_MULTIPLIER)

            trade_history.append([
                index,
                "BUY",
                price
            ])

            if verbose:
                print(f"BUY at {price:.2f}")


        # Risk management while holding
        elif shares > 0:

            # ATR Stop Loss
            if stop_price is not None and price <= stop_price:

                balance = shares * price
                shares = 0
                stop_price = None
                target_price = None
                losses += 1

                trade_history.append([
                    index,
                    "STOP LOSS",
                    price
                ])

                if verbose:
                    print(f"STOP LOSS at {price:.2f}")


            # ATR Take Profit
            elif target_price is not None and price >= target_price:

                balance = shares * price
                shares = 0
                stop_price = None
                target_price = None
                wins += 1

                trade_history.append([
                    index,
                    "TAKE PROFIT",
                    price
                ])

                if verbose:
                    print(f"TAKE PROFIT at {price:.2f}")


            # Normal sell signal
            elif row["Signal"] == -1:
                balance = shares * price
                shares = 0
                stop_price = None
                target_price = None

                if price > buy_price:
                    wins += 1
                else:
                    losses += 1

                trade_history.append([
                    index,
                    "SELL",
                    price
                ])

                if verbose:
                    print(f"SELL at {price:.2f}")


    # Close remaining position
    # Close remaining position at end of test
    if shares > 0:
        final_price = data.iloc[-1]["Close"]

        balance = shares * final_price
        shares = 0
        stop_price = None
        target_price = None

        if final_price > buy_price:
            wins += 1
        else:
            losses += 1

        trade_history.append([
            data.index[-1],
            "FINAL SELL",
            final_price
        ])

        if verbose:
            print(f"FINAL SELL at {final_price:.2f}")

    # Save trade history to CSV
    with open("logs/trade_log.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Action",
            "Price"
        ])

        writer.writerows(trade_history)

    profit = balance - initial_balance
    percent_return = (profit / initial_balance) * 100

    total_trades = wins + losses

    win_rate = 0
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100

    if verbose:
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

    return {
        "Ending Balance": balance,
        "Profit": profit,
        "Return": percent_return,
        "Trades": total_trades,
        "Win Rate": win_rate
    }


def buy_and_hold(data, initial_balance=1000):

    start_price = data.iloc[0]["Close"]
    end_price = data.iloc[-1]["Close"]

    shares = initial_balance / start_price

    final_balance = shares * end_price

    profit = final_balance - initial_balance
    percent_return = (profit / initial_balance) * 100

    print("\n--------------------")
    print("Buy & Hold Comparison")
    print("--------------------")
    print(f"Starting Balance: ${initial_balance:.2f}")
    print(f"Ending Balance: ${final_balance:.2f}")
    print(f"Return: {percent_return:.2f}%")

    return final_balance

backtest_strategy = risk_managed_backtest
