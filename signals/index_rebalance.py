"""
Index Rebalance Signal — ASX Small Cap Quant System.

Identifies stocks approaching index inclusion thresholds for the S&P/ASX 300
and Small Ordinaries indices. When stocks get promoted into these indices,
ETFs and index funds (STW, IOZ, VAS, VSO) are forced to buy, creating
predictable buying pressure and alpha.

Rebalance dates: 3rd Friday of March, June, September, December.
Signal is structural — index funds MUST buy when rebalance occurs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ASX index thresholds (approximate market cap boundaries)
ASX_300_THRESHOLD_LOW = 500_000_000  # $500M lower bound for ASX 300 inclusion
ASX_300_THRESHOLD_HIGH = 800_000_000  # $800M more likely inclusion zone
SMALL_ORDS_RANK_LOW = 101  # Small Ordinaries = ranks 101-300 by market cap

# Signal parameters
LIKELIHOOD_RATIO = 1.3  # Modest but structural edge
REBALANCE_PROXIMITY_DAYS = 30  # Signal strengthens within 30 days of rebalance
MARKET_CAP_TOP_PERCENTILE = 0.80  # Top 20% of universe ($800M-$1B range)
RAPID_GROWTH_THRESHOLD = 0.50  # 3-month market cap growth > 50%

# Rebalance months (March, June, September, December)
REBALANCE_MONTHS = [3, 6, 9, 12]


@dataclass
class RebalanceSignal:
    """Signal output for a stock approaching index inclusion."""

    ticker: str
    market_cap: float
    market_cap_rank: int
    near_rebalance_date: bool
    growth_rate_3m: float
    signal_active: bool
    signal_strength: float


def _get_next_rebalance_date(reference_date: date) -> date:
    """
    Compute the next ASX index rebalance date (3rd Friday of a rebalance month).

    Rebalance occurs on the 3rd Friday of March, June, September, December.
    """
    year = reference_date.year
    candidates: list[date] = []

    for month in REBALANCE_MONTHS:
        rebalance_date = _third_friday(year, month)
        candidates.append(rebalance_date)
        # Also check next year's first rebalance month
        if month == 3:
            candidates.append(_third_friday(year + 1, month))

    # Find the next upcoming rebalance date
    future_dates = [d for d in candidates if d >= reference_date]
    if future_dates:
        return min(future_dates)

    # If none found this year, use March of next year
    return _third_friday(year + 1, 3)


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of a given month/year."""
    # Find the first day of the month
    first_day = date(year, month, 1)
    # Find the first Friday (weekday 4 = Friday)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    # Third Friday is 14 days after the first Friday
    third_friday = first_friday + timedelta(days=14)
    return third_friday


def _is_near_rebalance(reference_date: date, proximity_days: int = REBALANCE_PROXIMITY_DAYS) -> bool:
    """Check if the reference date is within proximity_days of the next rebalance."""
    next_rebalance = _get_next_rebalance_date(reference_date)
    days_until = (next_rebalance - reference_date).days
    return 0 <= days_until <= proximity_days


def _compute_signal_strength(
    market_cap_percentile: float,
    growth_rate_3m: float,
    near_rebalance: bool,
) -> float:
    """
    Compute signal strength from 0.0 to 1.0 based on multiple factors.

    Components:
    - Market cap percentile within universe (higher = closer to inclusion)
    - 3-month growth rate (faster growth = more likely candidate)
    - Proximity to rebalance date (amplifies signal)
    """
    # Base strength from market cap positioning (0 to 0.4)
    cap_component = min(market_cap_percentile, 1.0) * 0.4

    # Growth component (0 to 0.3) — capped at 100% growth for scoring
    growth_component = min(growth_rate_3m / 1.0, 1.0) * 0.3

    # Rebalance proximity bonus (0 or 0.3)
    rebalance_component = 0.3 if near_rebalance else 0.0

    strength = cap_component + growth_component + rebalance_component
    return round(min(max(strength, 0.0), 1.0), 4)


def compute_rebalance_signal(universe: pd.DataFrame) -> dict[str, RebalanceSignal]:
    """
    Compute index rebalance signals for all stocks in the universe.

    Expects a DataFrame with columns:
    - ticker: str — ASX ticker symbol
    - market_cap: float — current market capitalisation in AUD
    - market_cap_3m_ago: float (optional) — market cap 3 months prior

    A signal fires when:
    1. Stock is in the top 20% of the universe by market cap (approaching index zone)
    2. Stock has positive momentum (market cap growing)
    3. Optionally: 3-month growth > 50% (rapid growth = strong candidate)

    Signal is amplified when within 30 days of a quarterly rebalance date.

    Args:
        universe: DataFrame of stocks in our $100M-$1B small cap universe.

    Returns:
        Dictionary mapping ticker to RebalanceSignal.
    """
    required_cols = {"ticker", "market_cap"}
    if not required_cols.issubset(set(universe.columns)):
        logger.warning(
            "Universe DataFrame missing required columns: %s",
            required_cols - set(universe.columns),
        )
        return {}

    if universe.empty:
        return {}

    today = date.today()
    near_rebalance = _is_near_rebalance(today)

    # Rank stocks by market cap (1 = largest)
    df = universe.copy()
    df = df.dropna(subset=["market_cap"])
    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    df["market_cap_rank"] = range(1, len(df) + 1)

    # Compute market cap percentile (1.0 = largest in universe)
    df["market_cap_percentile"] = 1.0 - (df["market_cap_rank"] - 1) / max(len(df) - 1, 1)

    # Compute 3-month growth rate if historical data available
    has_historical = "market_cap_3m_ago" in df.columns
    if has_historical:
        df["growth_rate_3m"] = (
            (df["market_cap"] - df["market_cap_3m_ago"]) / df["market_cap_3m_ago"]
        ).fillna(0.0)
    else:
        df["growth_rate_3m"] = 0.0
        logger.info("No market_cap_3m_ago column — growth rate defaulting to 0.")

    signals: dict[str, RebalanceSignal] = {}

    for _, row in df.iterrows():
        ticker = row["ticker"]
        market_cap = row["market_cap"]
        market_cap_rank = int(row["market_cap_rank"])
        market_cap_percentile = row["market_cap_percentile"]
        growth_rate_3m = row["growth_rate_3m"]

        # Signal fires when stock is in top 20% of universe AND has positive momentum
        in_top_percentile = market_cap_percentile >= MARKET_CAP_TOP_PERCENTILE
        has_positive_momentum = growth_rate_3m > 0.0
        has_rapid_growth = growth_rate_3m > RAPID_GROWTH_THRESHOLD

        # Signal activates if:
        # - In top 20% by market cap AND positive momentum, OR
        # - Has rapid 3-month growth (>50%) regardless of current percentile
        #   (these are fast-movers that may leapfrog into index territory)
        signal_active = (in_top_percentile and has_positive_momentum) or has_rapid_growth

        # Compute signal strength
        signal_strength = _compute_signal_strength(
            market_cap_percentile=market_cap_percentile,
            growth_rate_3m=growth_rate_3m,
            near_rebalance=near_rebalance if signal_active else False,
        )

        # Only assign full strength if signal is active
        if not signal_active:
            signal_strength = 0.0

        signals[ticker] = RebalanceSignal(
            ticker=ticker,
            market_cap=market_cap,
            market_cap_rank=market_cap_rank,
            near_rebalance_date=near_rebalance,
            growth_rate_3m=round(growth_rate_3m, 4),
            signal_active=signal_active,
            signal_strength=signal_strength,
        )

    active_count = sum(1 for s in signals.values() if s.signal_active)
    logger.info(
        "Rebalance signal computed: %d/%d stocks active | near_rebalance=%s",
        active_count,
        len(signals),
        near_rebalance,
    )

    return signals


def likelihood_ratio() -> float:
    """
    Return the likelihood ratio for the index rebalance signal.

    Index rebalance buying pressure is structural (ETFs must buy) but
    the edge is modest because the market partially anticipates inclusions.
    Historical back-tests on ASX show ~1.3x likelihood of outperformance
    in the month following inclusion.
    """
    return LIKELIHOOD_RATIO


def get_upcoming_rebalance_info(reference_date: date | None = None) -> dict:
    """
    Return information about the next upcoming ASX index rebalance.

    Useful for dashboards and monitoring.

    Returns:
        Dict with next_rebalance_date, days_until, and is_within_window.
    """
    if reference_date is None:
        reference_date = date.today()

    next_rebalance = _get_next_rebalance_date(reference_date)
    days_until = (next_rebalance - reference_date).days
    within_window = days_until <= REBALANCE_PROXIMITY_DAYS

    return {
        "next_rebalance_date": next_rebalance,
        "days_until": days_until,
        "is_within_window": within_window,
        "reference_date": reference_date,
    }
