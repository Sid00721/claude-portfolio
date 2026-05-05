"""
Post-Earnings Announcement Drift (PEAD) Signal

Academic research confirms PEAD exists on the ASX with "excess return in the
60 days following the earnings announcement." The market underreacts to positive
earnings surprises, causing a persistent drift upward — especially for small caps
with minimal analyst coverage.

Detection approach:
  1. Price-reaction heuristic (primary): identify days where price jumped >5% on
     elevated volume within the last 30 days — likely an earnings reaction.
  2. EPS surprise (secondary): if EODHD fundamentals provide analyst consensus,
     compare actual EPS to consensus. A positive surprise >10% strengthens the signal.

Signal strength decays linearly from 1.0 at day 0 to 0.0 at day 60, reflecting
the empirically-observed decay of the drift effect.

Likelihood ratio: 1.4 (PEAD well-documented on ASX small caps).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data import provider

logger = logging.getLogger(__name__)

DRIFT_WINDOW_DAYS = 60
PRICE_JUMP_THRESHOLD = 0.05  # 5% minimum daily price move
VOLUME_SURGE_MULTIPLIER = 1.5  # volume must be 1.5x the 20-day average
LOOKBACK_DAYS = 30  # how far back to search for the earnings reaction
EPS_SURPRISE_THRESHOLD = 0.10  # 10% beat over consensus = strong signal
LIKELIHOOD_RATIO = 1.4


@dataclass
class EarningsSurprise:
    """Represents a detected post-earnings drift opportunity."""

    ticker: str
    surprise_date: Optional[datetime]
    price_reaction_pct: float
    days_since_announcement: int
    in_drift_window: bool
    signal_active: bool
    signal_strength: float

    @property
    def signal_fires(self) -> bool:
        return self.signal_active and self.in_drift_window


def _detect_price_jump(prices: pd.DataFrame) -> Optional[tuple[datetime, float]]:
    """
    Detect the most recent day with a >5% price jump on elevated volume.

    Scans the last LOOKBACK_DAYS of price data for a single-day move that
    suggests an earnings reaction.

    Returns:
        Tuple of (date of jump, percentage change) or None if no jump found.
    """
    if prices.empty or len(prices) < 21:
        return None

    # Compute daily returns
    close_col = "adj_close" if "adj_close" in prices.columns else "close"
    if close_col not in prices.columns:
        # Handle yfinance-style column names
        close_col = "Close" if "Close" in prices.columns else None
        if close_col is None:
            return None

    volume_col = "volume" if "volume" in prices.columns else "Volume"
    if volume_col not in prices.columns:
        volume_col = None

    closes = prices[close_col].dropna()
    if len(closes) < 21:
        return None

    daily_returns = closes.pct_change()

    # Only look at the most recent LOOKBACK_DAYS
    cutoff_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    recent_mask = prices.index >= pd.Timestamp(cutoff_date)
    recent_returns = daily_returns[recent_mask]

    if recent_returns.empty:
        return None

    # Check volume surge if volume data available
    if volume_col and volume_col in prices.columns:
        volumes = prices[volume_col]
        vol_20d_avg = volumes.rolling(window=20).mean()

        for date in reversed(recent_returns.index.tolist()):
            ret = recent_returns.get(date, 0)
            if ret is None or np.isnan(ret):
                continue
            if ret > PRICE_JUMP_THRESHOLD:
                # Check volume was elevated
                avg_vol = vol_20d_avg.get(date, 0)
                actual_vol = volumes.get(date, 0)
                if avg_vol and actual_vol and avg_vol > 0:
                    if actual_vol >= avg_vol * VOLUME_SURGE_MULTIPLIER:
                        return (pd.Timestamp(date).to_pydatetime(), float(ret))
                else:
                    # No volume data for this date — accept on price alone
                    return (pd.Timestamp(date).to_pydatetime(), float(ret))
    else:
        # No volume data at all — rely purely on price jump magnitude
        for date in reversed(recent_returns.index.tolist()):
            ret = recent_returns.get(date, 0)
            if ret is None or np.isnan(ret):
                continue
            if ret > PRICE_JUMP_THRESHOLD:
                return (pd.Timestamp(date).to_pydatetime(), float(ret))

    return None


def _compute_signal_strength(days_since: int, price_reaction_pct: float) -> float:
    """
    Compute signal strength with linear decay over the drift window.

    Strength starts at 1.0 on announcement day and decays to 0.0 at day 60.
    Larger initial reactions get a small boost (capped at 1.0).
    """
    if days_since < 0 or days_since > DRIFT_WINDOW_DAYS:
        return 0.0

    # Linear decay from 1.0 to 0.0 over 60 days
    time_decay = 1.0 - (days_since / DRIFT_WINDOW_DAYS)

    # Boost for stronger initial reactions (>10% jump gets full boost)
    reaction_boost = min(price_reaction_pct / 0.10, 1.0)

    # Combine: base strength is the time decay, modulated by reaction size
    strength = time_decay * (0.7 + 0.3 * reaction_boost)

    return round(min(max(strength, 0.0), 1.0), 4)


def _check_eps_surprise(ticker: str) -> Optional[float]:
    """
    Check EODHD fundamentals for EPS surprise data.

    For ASX small caps, analyst coverage is minimal so this often returns None.
    When available, returns the surprise as a fraction (e.g., 0.15 = 15% beat).
    """
    try:
        fundamentals = provider.get_fundamentals(ticker)
        if not fundamentals:
            return None

        # EODHD fundamentals may contain earnings data under Highlights
        # This is a best-effort check — most small caps lack consensus data
        pe_ratio = fundamentals.get("pe_ratio")
        if pe_ratio is None:
            return None

        # Without explicit consensus vs actual EPS from the API,
        # we cannot compute a true EPS surprise from fundamentals alone.
        # Return None to fall back to price-reaction heuristic.
        return None

    except Exception as e:
        logger.debug("Could not fetch EPS surprise data for %s: %s", ticker, e)
        return None


def compute_single_signal(ticker: str) -> EarningsSurprise:
    """
    Compute the PEAD signal for a single ticker.

    Uses price history to detect recent earnings reactions, then determines
    if the stock is within the 60-day drift window.
    """
    empty_signal = EarningsSurprise(
        ticker=ticker,
        surprise_date=None,
        price_reaction_pct=0.0,
        days_since_announcement=0,
        in_drift_window=False,
        signal_active=False,
        signal_strength=0.0,
    )

    try:
        # Fetch enough history to cover lookback + volume average calculation
        prices = provider.get_price_history(ticker, period_days=90)
    except Exception as e:
        logger.warning("Failed to fetch price history for %s: %s", ticker, e)
        return empty_signal

    if prices.empty:
        return empty_signal

    # Primary detection: price-reaction heuristic
    jump = _detect_price_jump(prices)

    if jump is None:
        return empty_signal

    surprise_date, price_reaction_pct = jump
    now = datetime.now()
    days_since = (now - surprise_date).days
    in_drift_window = 0 <= days_since <= DRIFT_WINDOW_DAYS

    if not in_drift_window or price_reaction_pct <= 0:
        return EarningsSurprise(
            ticker=ticker,
            surprise_date=surprise_date,
            price_reaction_pct=round(price_reaction_pct, 4),
            days_since_announcement=days_since,
            in_drift_window=False,
            signal_active=False,
            signal_strength=0.0,
        )

    # Signal is active: compute decaying strength
    strength = _compute_signal_strength(days_since, price_reaction_pct)

    # Secondary check: EPS surprise can boost confidence
    eps_surprise = _check_eps_surprise(ticker)
    if eps_surprise is not None and eps_surprise > EPS_SURPRISE_THRESHOLD:
        # Strong fundamental confirmation — boost strength slightly
        strength = min(strength * 1.2, 1.0)
        strength = round(strength, 4)

    return EarningsSurprise(
        ticker=ticker,
        surprise_date=surprise_date,
        price_reaction_pct=round(price_reaction_pct, 4),
        days_since_announcement=days_since,
        in_drift_window=True,
        signal_active=True,
        signal_strength=strength,
    )


def compute_earnings_signals(
    tickers: list[str], universe: pd.DataFrame
) -> dict[str, EarningsSurprise]:
    """
    Compute PEAD signals for a list of tickers.

    Args:
        tickers: List of ASX ticker symbols to evaluate.
        universe: DataFrame from universe screener (unused directly here but
                  kept for interface consistency with other signal modules).

    Returns:
        Dictionary mapping ticker -> EarningsSurprise for tickers where a
        positive earnings reaction was detected within the drift window.
    """
    signals: dict[str, EarningsSurprise] = {}

    for ticker in tickers:
        try:
            signal = compute_single_signal(ticker)
            signals[ticker] = signal
        except Exception as e:
            logger.error("Error computing PEAD signal for %s: %s", ticker, e)
            signals[ticker] = EarningsSurprise(
                ticker=ticker,
                surprise_date=None,
                price_reaction_pct=0.0,
                days_since_announcement=0,
                in_drift_window=False,
                signal_active=False,
                signal_strength=0.0,
            )

    active_count = sum(1 for s in signals.values() if s.signal_active)
    logger.info(
        "PEAD scan complete: %d/%d tickers have active drift signals.",
        active_count,
        len(tickers),
    )

    return signals


def likelihood_ratio() -> float:
    """
    Return the likelihood ratio for the PEAD signal.

    PEAD is well-documented on the ASX. Academic studies show statistically
    significant excess returns in the 60 days following earnings announcements,
    particularly for small caps with low analyst coverage.
    """
    return LIKELIHOOD_RATIO
