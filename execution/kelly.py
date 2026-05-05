"""
Kelly Criterion position sizer for an ASX small cap quant system.

Implements fractional Kelly sizing with portfolio-level constraints,
ATR-based risk calculation, and beta exposure limits.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data.provider import get_price_history


@dataclass
class PositionSize:
    ticker: str
    full_kelly_pct: float
    fractional_kelly_pct: float
    position_dollars: float
    shares: int
    risk_per_share: float  # 2x ATR


MAX_SINGLE_POSITION_PCT = 0.15
MIN_POSITION_DOLLARS = 500.0
MAX_PORTFOLIO_BETA = 1.5


def kelly_fraction_calc(
    posterior_probability: float,
    expected_win: float,
    expected_loss: float,
) -> float:
    """
    Raw Kelly formula: f = (bp - q) / b

    Where:
        b = win/loss ratio (expected_win / expected_loss)
        p = probability of winning (posterior_probability)
        q = 1 - p
    """
    if expected_loss <= 0 or expected_win <= 0:
        return 0.0

    b = expected_win / expected_loss
    p = posterior_probability
    q = 1.0 - p

    f = (b * p - q) / b
    return max(f, 0.0)


def calculate_atr(ticker: str, period: int = 14) -> float:
    """
    Compute 14-day Average True Range for a given ticker using EODHD data.

    Returns ATR in dollar terms.
    """
    df = get_price_history(ticker, period_days=60)

    if df.empty or len(df) < period + 1:
        return 0.0

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    true_ranges = []
    for i in range(1, len(df)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        true_ranges.append(tr)

    # Simple moving average of true range over the period
    atr = float(np.mean(true_ranges[-period:]))
    return atr


def size_position(
    ticker: str,
    posterior_probability: float,
    portfolio_value: float,
    price: float,
    expected_win: float = 0.08,
    expected_loss: float = 0.04,
    kelly_fraction: float = 0.5,
) -> Optional[PositionSize]:
    """
    Main position sizing function.

    Args:
        ticker: ASX ticker (e.g. "BHP.AX")
        posterior_probability: Bayesian posterior probability of a winning trade
        portfolio_value: Total portfolio value in AUD
        price: Current share price
        expected_win: Average winner size as a decimal (default 8%)
        expected_loss: Average loser size as a decimal (default 4%)
        kelly_fraction: Fraction of Kelly to use (default 0.5 = half Kelly)

    Returns:
        PositionSize dataclass or None if position is below minimum threshold.
    """
    full_kelly = kelly_fraction_calc(posterior_probability, expected_win, expected_loss)
    fractional_kelly = full_kelly * kelly_fraction

    # Cap at max single position size
    capped_pct = min(fractional_kelly, MAX_SINGLE_POSITION_PCT)

    position_dollars = capped_pct * portfolio_value

    # Skip if below minimum (transaction costs eat the edge)
    if position_dollars < MIN_POSITION_DOLLARS:
        return None

    shares = int(position_dollars // price)
    if shares == 0:
        return None

    # Recalculate actual dollar position based on whole shares
    position_dollars = shares * price

    atr = calculate_atr(ticker)
    risk_per_share = 2.0 * atr

    return PositionSize(
        ticker=ticker,
        full_kelly_pct=full_kelly,
        fractional_kelly_pct=capped_pct,
        position_dollars=position_dollars,
        shares=shares,
        risk_per_share=risk_per_share,
    )


def estimate_beta(ticker: str) -> float:
    """
    Estimate stock beta relative to ASX200 (^AXJO) using 90 days of returns.
    """
    stock_data = get_price_history(ticker, period_days=90)
    market_data = get_price_history("^AXJO", period_days=90)

    if stock_data.empty or market_data.empty:
        return 1.0  # Default to market beta if data unavailable

    stock_returns = stock_data["close"].pct_change().dropna().values
    market_returns = market_data["close"].pct_change().dropna().values

    # Align lengths
    min_len = min(len(stock_returns), len(market_returns))
    stock_returns = stock_returns[-min_len:]
    market_returns = market_returns[-min_len:]

    if len(market_returns) < 5:
        return 1.0

    covariance = np.cov(stock_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)

    if market_variance == 0:
        return 1.0

    return float(covariance / market_variance)


def batch_size(
    positions: list[dict],
    portfolio_value: float,
    current_positions_beta: float = 0.0,
    kelly_fraction: float = 0.5,
) -> list[Optional[PositionSize]]:
    """
    Size multiple positions with portfolio-level constraints.

    Args:
        positions: List of dicts with keys:
            - ticker: str
            - posterior_probability: float
            - price: float
            - expected_win: float (optional, default 0.08)
            - expected_loss: float (optional, default 0.04)
        portfolio_value: Total portfolio value in AUD
        current_positions_beta: Weighted beta of existing portfolio positions
        kelly_fraction: Fraction of Kelly to use

    Returns:
        List of PositionSize (or None for skipped positions).

    Constraints:
        - Sum of all new positions <= 100% of portfolio value
        - If adding a position would push net portfolio beta > 1.5, reduce its size
    """
    sized_positions: list[Optional[PositionSize]] = []
    total_allocated = 0.0
    running_beta_exposure = current_positions_beta * portfolio_value

    for pos in positions:
        ticker = pos["ticker"]
        posterior = pos["posterior_probability"]
        price = pos["price"]
        exp_win = pos.get("expected_win", 0.08)
        exp_loss = pos.get("expected_loss", 0.04)

        result = size_position(
            ticker=ticker,
            posterior_probability=posterior,
            portfolio_value=portfolio_value,
            price=price,
            expected_win=exp_win,
            expected_loss=exp_loss,
            kelly_fraction=kelly_fraction,
        )

        if result is None:
            sized_positions.append(None)
            continue

        # Check total allocation constraint
        if total_allocated + result.position_dollars > portfolio_value:
            remaining = portfolio_value - total_allocated
            if remaining < MIN_POSITION_DOLLARS:
                sized_positions.append(None)
                continue
            # Scale down to fit
            scale_factor = remaining / result.position_dollars
            result.position_dollars = remaining
            result.shares = int(result.position_dollars // price)
            result.fractional_kelly_pct *= scale_factor
            if result.shares == 0:
                sized_positions.append(None)
                continue
            result.position_dollars = result.shares * price

        # Beta risk check
        stock_beta = estimate_beta(ticker)
        position_beta_contribution = stock_beta * result.position_dollars
        new_net_beta = (running_beta_exposure + position_beta_contribution) / portfolio_value

        if new_net_beta > MAX_PORTFOLIO_BETA:
            # Reduce size so portfolio beta stays at limit
            allowable_beta_dollars = (
                (MAX_PORTFOLIO_BETA * portfolio_value - running_beta_exposure) / stock_beta
            )
            if allowable_beta_dollars < MIN_POSITION_DOLLARS:
                sized_positions.append(None)
                continue

            result.shares = int(allowable_beta_dollars // price)
            if result.shares == 0:
                sized_positions.append(None)
                continue
            result.position_dollars = result.shares * price
            result.fractional_kelly_pct = result.position_dollars / portfolio_value

        total_allocated += result.position_dollars
        running_beta_exposure += stock_beta * result.position_dollars
        sized_positions.append(result)

    return sized_positions
