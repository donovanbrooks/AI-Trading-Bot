from optimization.optimizer import optimize_strategy
from strategy.moving_average import moving_average_strategy
from backtests.backtest import backtest_strategy


def walk_forward_test(data):

    split = int(len(data) * 0.8)

    train_data = data.iloc[:split].copy()
    test_data = data.iloc[split:].copy()


    print("Training Period")
    print("----------------")

    best = optimize_strategy(train_data)

    short = best["Short MA"]
    long = best["Long MA"]


    print("\nBest Parameters:")
    print(best)


    print("\nTesting Period")
    print("----------------")


    test_data = moving_average_strategy(
        test_data,
        short_window=short,
        long_window=long
    )


    backtest_strategy(test_data)