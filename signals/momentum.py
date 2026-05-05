"""
Momentum Signal — Short-Term Breakout + Medium-Term Trend

Two sub-signals:
1. Breakout: stock hit a 20-day high in the last 5 days (fresh breakout)
2. Acceleration: 2-week return in the top 20% of universe (recent acceleration)

These catch SWING TRADE entries — not "has this been going up for a year"
but "is this starting to move NOW."

Also retains the 12m quintile as a secondary filter (only trade breakouts
in stocks that have underlying trend support).
"""

import numpy as np
import pandas as pd

from data.provider import get_price_history


def compute_momentum_signal(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum signal combining:
    - 12m trend quintile (background trend)
    - 20-day breakout (timing signal)
    - 2-week acceleration (recent momentum)
    """
    if "ticker" not in universe.columns:
        return pd.DataFrame(columns=["ticker", "momentum_score", "momentum_quintile", "signal_strength", "signal_active"])

    results = universe[["ticker"]].copy()

    # 12m trend quintile (from universe screener data)
    if "return_12m" in universe.columns:
        results["trend_12m"] = universe["return_12m"].values
    else:
        results["trend_12m"] = 0.0

    valid = results["trend_12m"].notna()
    results["momentum_quintile"] = np.nan
    if valid.sum() >= 5:
        results.loc[valid, "momentum_quintile"] = pd.qcut(
            results.loc[valid, "trend_12m"], q=5, labels=[1, 2, 3, 4, 5]
        ).astype(float)

    # Short-term signals: breakout + acceleration
    results["breakout"] = False
    results["acceleration"] = False
    results["short_term_return"] = np.nan

    for idx, row in results.iterrows():
        ticker = row["ticker"]
        try:
            prices = get_price_history(ticker, period_days=30)
            if prices.empty or len(prices) < 20:
                continue

            close_col = "close" if "close" in prices.columns else "Close"
            close = prices[close_col].dropna()
            if len(close) < 20:
                continue

            # Breakout: current price within 2% of 20-day high
            high_20d = close.iloc[-20:].max()
            current = close.iloc[-1]
            if current >= high_20d * 0.98:
                results.at[idx, "breakout"] = True

            # 2-week return
            if len(close) >= 10:
                ret_2w = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
                results.at[idx, "short_term_return"] = ret_2w

        except Exception:
            continue

    # Acceleration: top 20% of 2-week returns
    valid_st = results["short_term_return"].notna()
    if valid_st.sum() >= 5:
        threshold = results.loc[valid_st, "short_term_return"].quantile(0.80)
        results.loc[valid_st, "acceleration"] = results.loc[valid_st, "short_term_return"] >= threshold

    # Signal fires when: (breakout OR acceleration) AND trend quintile >= 3 (not fighting the trend)
    results["signal_active"] = (
        (results["breakout"] | results["acceleration"]) &
        (results["momentum_quintile"] >= 3.0)
    )

    # Composite score: weighted blend
    results["momentum_score"] = (
        results["short_term_return"].fillna(0) * 0.6 +
        results["trend_12m"].fillna(0) * 0.4
    )

    # Signal strength: normalize to 0-1
    valid_score = results["momentum_score"].notna() & results["signal_active"]
    results["signal_strength"] = 0.0
    if valid_score.sum() > 1:
        active_scores = results.loc[valid_score, "momentum_score"]
        min_s = active_scores.min()
        max_s = active_scores.max()
        if max_s > min_s:
            results.loc[valid_score, "signal_strength"] = (
                (results.loc[valid_score, "momentum_score"] - min_s) / (max_s - min_s)
            )
        else:
            results.loc[valid_score, "signal_strength"] = 0.5
    elif valid_score.sum() == 1:
        results.loc[valid_score, "signal_strength"] = 1.0

    return results[["ticker", "momentum_score", "momentum_quintile", "signal_strength", "signal_active"]]


def likelihood_ratio() -> float:
    return 1.5
