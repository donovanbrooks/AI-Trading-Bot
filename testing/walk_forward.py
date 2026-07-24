from optimization.optimizer import optimize_strategy
from strategy.moving_average import moving_average_strategy
from backtests.backtest import backtest_strategy
from ai_strategy.ai_trading_strategy import ai_strategy
from strategy.ensemble_strategy import ensemble_strategy
from models.price_predictor import create_features


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

    ai_test_data = ai_strategy(
        train_data,
        test_data
    )

    print(test_data.columns)

    test_data["AI_Signal"] = ai_test_data["AI_Signal"]

    test_data["Probability_Up"] = ai_test_data["Probability_Up"]

    test_data["Prediction"] = ai_test_data["Prediction"]

    print(test_data[["MA_Signal", "AI_Signal"]].tail())

    print(test_data.columns)

    test_data = create_features(test_data)

    ensemble_data = ensemble_strategy(test_data)

    print("\nEnsemble Signals:")
    print(
        ensemble_data[
            [
                "Close",
                "MA_Signal",
                "AI_Signal",
                "Ensemble_Score",
                "Signal"
            ]
        ].tail(20)
    )

    print("\n====================")
    print("AI Strategy Testing")
    print("====================")

    backtest_strategy(ensemble_data)

    print(test_data.columns.tolist())

    print("Before create_features:", test_data.columns.tolist())

    test_data = create_features(test_data)

    print("After create_features:", test_data.columns.tolist())

    print("Before ensemble:")
    print(test_data.columns.tolist())

    ensemble_data = ensemble_strategy(test_data)