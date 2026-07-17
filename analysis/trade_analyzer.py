import pandas as pd


def analyze_trades(file_path="logs/trade_log.csv"):

    trades = pd.read_csv(file_path)

    completed_trades = []

    buy_price = None

    for _, trade in trades.iterrows():

        if trade["Action"] == "BUY":
            buy_price = trade["Price"]

        elif buy_price is not None and trade["Action"] != "FINAL SELL":
            sell_price = trade["Price"]

            percent_change = (
                (sell_price - buy_price) / buy_price
            ) * 100

            completed_trades.append(percent_change)

            buy_price = None


    if len(completed_trades) == 0:
        print("No completed trades found.")
        return


    wins = [x for x in completed_trades if x > 0]
    losses = [x for x in completed_trades if x < 0]


    print("\n--------------------")
    print("Trade Analysis Report")
    print("--------------------")

    print(f"Total Trades: {len(completed_trades)}")

    print(f"Average Trade: {sum(completed_trades)/len(completed_trades):.2f}%")

    print(f"Best Trade: {max(completed_trades):.2f}%")

    print(f"Worst Trade: {min(completed_trades):.2f}%")

    if wins:
        print(f"Average Win: {sum(wins)/len(wins):.2f}%")

    if losses:
        print(f"Average Loss: {sum(losses)/len(losses):.2f}%")