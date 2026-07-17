from strategy.moving_average import moving_average_strategy
from backtests.backtest import backtest_strategy


def optimize_strategy(data):

    short_windows = [5, 10, 20, 30]
    long_windows = [20, 50, 100]

    results = []

    for short in short_windows:
        for long in long_windows:

            if short >= long:
                continue

            test_data = data.copy()

            test_data = moving_average_strategy(
                test_data,
                short_window=short,
                long_window=long
            )

            final_balance = backtest_strategy(
                test_data,
                initial_balance=1000
            )

            profit = final_balance - 1000

            results.append({
                "Short MA": short,
                "Long MA": long,
                "Profit": float(round(profit, 2))
            })


    results = sorted(
        results,
        key=lambda x: x["Profit"],
        reverse=True
    )


    print("\n--------------------")
    print("Optimization Results")
    print("--------------------")

    for result in results:
        print(result)

    return results[0]