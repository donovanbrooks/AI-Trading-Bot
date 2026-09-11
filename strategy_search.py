"""Bounded, walk-forward strategy exploration utilities.

The explorer searches pre-approved candidate settings. It is deliberately not
an order generator and it ranks out-of-sample consistency above one lucky
historical return.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import tanh

import pandas as pd

from validation import BacktestConfig, calculate_metrics, run_backtest


def probability_signals(
    base_signals: pd.DataFrame,
    probability_threshold: float,
    require_bullish_candle: bool = False,
    require_trend_filter: bool = False,
) -> pd.DataFrame:
    """Reuse already walk-forward-generated probabilities for a new threshold."""
    if not 0.5 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0.5 and 1")
    result = base_signals.copy()
    probability = pd.to_numeric(result["AI Probability"], errors="coerce")
    result["Signal"] = 0
    result.loc[probability >= probability_threshold, "Signal"] = 1
    result.loc[probability <= 1 - probability_threshold, "Signal"] = -1
    if require_bullish_candle and "Bullish Candle" in result:
        result.loc[(result["Signal"] == 1) & ~result["Bullish Candle"].astype(bool), "Signal"] = 0
    if require_trend_filter and {"Trend MA 50", "Close"}.issubset(result.columns):
        result.loc[(result["Signal"] == 1) & (result["Close"] <= result["Trend MA 50"]), "Signal"] = 0
    if "Session Time" in result:
        result.loc[result["Session Time"].astype(str) == "15:50:00", "Signal"] = -1
    result["Execution Signal"] = result["Signal"].shift(1).fillna(0).astype(int)
    return result


def evaluate_candidates(
    candidates: Iterable[tuple[dict[str, object], pd.DataFrame]],
    config: BacktestConfig,
    periods_per_year: int,
    folds: int = 3,
    min_trades: int = 2,
) -> pd.DataFrame:
    """Score candidates on independent contiguous folds, with risk penalties."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    reports: list[dict[str, object]] = []
    for parameters, signals in candidates:
        if len(signals) < folds * 3:
            continue
        fold_size = len(signals) // folds
        fold_metrics = []
        for fold in range(folds):
            start = fold * fold_size
            end = len(signals) if fold == folds - 1 else (fold + 1) * fold_size
            segment = signals.iloc[start:end]
            if len(segment) < 2:
                continue
            results, trades = run_backtest(segment, config)
            fold_metrics.append(calculate_metrics(results, trades, config.initial_cash, periods_per_year))
        if len(fold_metrics) != folds:
            continue
        total_trades = sum(int(item["trade_count"]) for item in fold_metrics)
        if total_trades < min_trades:
            continue
        median_return = float(pd.Series([item["total_return"] for item in fold_metrics]).median())
        worst_return = float(min(item["total_return"] for item in fold_metrics))
        median_sharpe = float(pd.Series([item["sharpe_ratio"] for item in fold_metrics]).median())
        worst_drawdown = float(min(item["max_drawdown"] for item in fold_metrics))
        # Return/drawdown components are fractions. Sharpe is bounded so it
        # cannot dominate a result merely because it is unitless.
        consistency_score = ((median_return + worst_return) / 2) + worst_drawdown + 0.02 * tanh(median_sharpe)
        reports.append({
            **parameters,
            "Median OOS return": median_return,
            "Worst OOS return": worst_return,
            "Worst OOS drawdown": worst_drawdown,
            "Median OOS Sharpe": median_sharpe,
            "OOS trades": total_trades,
            "Consistency score": consistency_score,
        })
    if not reports:
        return pd.DataFrame()
    return pd.DataFrame(reports).sort_values(
        ["Consistency score", "Median OOS return"], ascending=False
    ).reset_index(drop=True)


def market_regimes(data: pd.DataFrame) -> pd.Series:
    """Classify bars into broad trend/volatility regimes using prior prices."""
    close = pd.to_numeric(data["Close"], errors="coerce")
    trend = pd.Series("Downtrend", index=data.index)
    trend.loc[close >= close.rolling(50, min_periods=10).mean()] = "Uptrend"
    volatility = close.pct_change().rolling(20, min_periods=5).std()
    median_volatility = volatility.median()
    volatility_label = pd.Series("Low volatility", index=data.index)
    if pd.notna(median_volatility):
        volatility_label.loc[volatility >= median_volatility] = "High volatility"
    return trend + " · " + volatility_label


def regime_recommendations(
    candidates: Iterable[tuple[dict[str, object], pd.DataFrame]],
    config: BacktestConfig,
    min_bars: int = 20,
) -> pd.DataFrame:
    """Select the most risk-adjusted candidate for each broad market regime."""
    rows: list[dict[str, object]] = []
    for parameters, signals in candidates:
        results, trades = run_backtest(signals, config)
        regimes = market_regimes(results)
        returns = results["Equity"].pct_change().fillna(0)
        for regime in sorted(regimes.dropna().unique()):
            regime_returns = returns[regimes == regime]
            if len(regime_returns) < min_bars:
                continue
            equity = (1 + regime_returns).cumprod()
            conditional_return = float(equity.iloc[-1] - 1)
            conditional_drawdown = float((equity / equity.cummax() - 1).min())
            conditional_sharpe = (
                float(regime_returns.mean() / regime_returns.std(ddof=0))
                if len(regime_returns) > 1 and regime_returns.std(ddof=0) > 0
                else 0.0
            )
            rows.append({
                **parameters,
                "Market regime": regime,
                "Regime bars": len(regime_returns),
                "Conditional return": conditional_return,
                "Conditional drawdown": conditional_drawdown,
                "Conditional Sharpe": conditional_sharpe,
                "Regime score": conditional_return + conditional_drawdown + 0.01 * conditional_sharpe,
                "All-period trades": len(trades),
            })
    if not rows:
        return pd.DataFrame()
    report = pd.DataFrame(rows).sort_values(
        ["Market regime", "Regime score", "Conditional return"], ascending=[True, False, False]
    )
    return report.groupby("Market regime", as_index=False, sort=False).head(1).reset_index(drop=True)
