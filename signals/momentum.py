from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _fetch_price_history(ticker: str, start: str, end: str) -> tuple[str, Optional[pd.Series]]:
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty or len(data) < 20:
            return ticker, None
        return ticker, data["Close"].squeeze()
    except Exception:
        return ticker, None


def _compute_momentum_12_1(prices: pd.Series) -> Optional[float]:
    if prices is None or len(prices) < 252:
        return None
    price_12m_ago = prices.iloc[0]
    price_1m_ago = prices.iloc[-21]
    if price_12m_ago == 0:
        return None
    ret_12m = (price_1m_ago / price_12m_ago) - 1.0
    return float(ret_12m)


def compute_momentum_signal(
    universe: pd.DataFrame,
    max_workers: int = 10,
    as_of_date: Optional[datetime] = None,
) -> pd.DataFrame:
    if "ticker" not in universe.columns:
        raise ValueError("universe DataFrame must contain a 'ticker' column")

    tickers = universe["ticker"].tolist()

    if as_of_date is None:
        as_of_date = datetime.now()

    end_date = as_of_date.strftime("%Y-%m-%d")
    start_date = (as_of_date - timedelta(days=365)).strftime("%Y-%m-%d")

    momentum_scores: dict[str, Optional[float]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_price_history, ticker, start_date, end_date): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker, prices = future.result()
            momentum_scores[ticker] = _compute_momentum_12_1(prices)

    results = pd.DataFrame({
        "ticker": tickers,
        "momentum_score": [momentum_scores.get(t) for t in tickers],
    })

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
    return 1.4
