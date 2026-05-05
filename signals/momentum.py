"""
Momentum Signal — 12-1 Month Factor

Uses pre-computed 12m returns from the universe screener.
No additional API calls needed — momentum is computed during universe scan.
"""

import numpy as np
import pandas as pd


def compute_momentum_signal(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum quintiles from universe's return_12m column.
    The universe screener already computed 12-month returns.
    """
    if "ticker" not in universe.columns or "return_12m" not in universe.columns:
        return pd.DataFrame(columns=["ticker", "momentum_score", "momentum_quintile", "signal_strength", "signal_active"])

    results = universe[["ticker"]].copy()
    results["momentum_score"] = universe["return_12m"].values

    valid = results["momentum_score"].notna()
    results["momentum_quintile"] = np.nan

    if valid.sum() >= 5:
        results.loc[valid, "momentum_quintile"] = pd.qcut(
            results.loc[valid, "momentum_score"], q=5, labels=[1, 2, 3, 4, 5]
        ).astype(float)

    results["signal_strength"] = np.nan
    if valid.sum() > 1:
        min_score = results.loc[valid, "momentum_score"].min()
        max_score = results.loc[valid, "momentum_score"].max()
        denom = max_score - min_score
        if denom > 0:
            results.loc[valid, "signal_strength"] = (
                (results.loc[valid, "momentum_score"] - min_score) / denom
            )
        else:
            results.loc[valid, "signal_strength"] = 0.5

    results["signal_active"] = results["momentum_quintile"] == 5.0

    return results[["ticker", "momentum_score", "momentum_quintile", "signal_strength", "signal_active"]]


def likelihood_ratio() -> float:
    return 1.5  # updated from research: momentum stronger on ASX small caps
