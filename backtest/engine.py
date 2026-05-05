"""
ASX Small Cap Backtest Engine

Event-driven backtester that simulates the full pipeline historically:
Universe Selection -> Regime Detection -> Signal Generation ->
Bayesian Aggregation -> Kelly Sizing -> Execution -> Attribution

Processes one day at a time with next-day-open execution for realism.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
import math

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """All tunable parameters for the backtest."""

    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    kelly_fraction: float = 0.5  # half-Kelly per CLAUDE.md rules
    min_posterior: float = 0.75
    max_position_pct: float = 0.15
    stop_loss_atr_multiple: float = 2.0
    risk_free_rate: float = 0.04  # current AU cash rate
    atr_period: int = 14
    momentum_lookback: int = 252  # ~12 months trading days
    momentum_skip: int = 21  # skip most recent month


# ---------------------------------------------------------------------------
# Trade & Result Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A completed round-trip trade."""

    ticker: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    signals_fired: list[str]
    posterior_at_entry: float


@dataclass
class Position:
    """An open position in the portfolio."""

    ticker: str
    shares: int
    entry_price: float
    entry_date: date
    stop_price: float
    signals_fired: list[str] = field(default_factory=list)
    posterior_at_entry: float = 0.0


@dataclass
class BacktestResult:
    """Aggregate performance metrics from a completed backtest."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_winner: float
    avg_loser: float
    total_trades: int
    profit_factor: float
    trades: list[Trade]
    equity_curve: pd.Series  # daily portfolio values indexed by date


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

def detect_regime(vix_value: float) -> str:
    """
    Classify market regime from VIX level.

    Returns: 'risk_on', 'selective', or 'risk_off'
    """
    if vix_value < 18:
        return "risk_on"
    elif vix_value <= 28:
        return "selective"
    else:
        return "risk_off"


def fetch_vix_series(start: date, end: date) -> pd.Series:
    """Fetch VIX daily close from Yahoo Finance."""
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    if vix.empty:
        return pd.Series(dtype=float)
    close = vix["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = close.index.date if hasattr(close.index, 'date') else close.index
    return close


# ---------------------------------------------------------------------------
# Signal Computation
# ---------------------------------------------------------------------------

def compute_momentum(prices: pd.Series, lookback: int = 252, skip: int = 21) -> Optional[float]:
    """
    12-1 month momentum signal.
    Returns the return from (lookback) days ago to (skip) days ago.
    None if insufficient data.
    """
    if len(prices) < lookback:
        return None
    price_end = prices.iloc[-(skip + 1)]
    price_start = prices.iloc[-lookback]
    if price_start <= 0:
        return None
    return (price_end - price_start) / price_start


def compute_insider_signal(ticker: str, as_of: date, insider_cache: dict) -> Optional[float]:
    """
    Insider cluster buying score from cached Appendix 3Y data.

    insider_cache: dict of ticker -> list of {date, net_shares, value_aud}
    Returns a score between 0 and 1, or None if no data.
    """
    trades = insider_cache.get(ticker, [])
    if not trades:
        return None

    # Look at insider activity in the last 90 days
    lookback_start = as_of - timedelta(days=90)
    recent = [
        t for t in trades
        if lookback_start <= t["date"] <= as_of
    ]

    if not recent:
        return None

    # Score: number of distinct buy transactions, capped at 5 for normalization
    buy_count = sum(1 for t in recent if t["net_shares"] > 0)
    return min(buy_count / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Bayesian Aggregation
# ---------------------------------------------------------------------------

def bayesian_aggregate(
    signals: dict[str, float],
    base_rate: float = 0.5,
) -> float:
    """
    Combine multiple signal scores into a posterior probability using
    naive Bayesian update.

    Each signal value is treated as P(signal | stock goes up).
    We assume P(signal | stock goes down) = 1 - signal_value.

    Returns posterior probability of the stock going up.
    """
    if not signals:
        return base_rate

    log_odds = math.log(base_rate / (1 - base_rate))

    for signal_name, signal_value in signals.items():
        # Clamp to avoid log(0)
        p = max(min(signal_value, 0.99), 0.01)
        # Likelihood ratio contribution
        log_odds += math.log(p / (1 - p))

    posterior = 1.0 / (1.0 + math.exp(-log_odds))
    return posterior


# ---------------------------------------------------------------------------
# Kelly Criterion Sizing
# ---------------------------------------------------------------------------

def kelly_size(
    posterior: float,
    avg_win: float = 0.15,
    avg_loss: float = 0.08,
    kelly_fraction: float = 0.5,
) -> float:
    """
    Half-Kelly position size as fraction of portfolio.

    f* = (p * b - q) / b
    where p = probability of win, q = 1-p, b = win/loss ratio

    Returns fraction of portfolio to allocate (0 if negative edge).
    """
    p = posterior
    q = 1.0 - p
    b = avg_win / avg_loss if avg_loss > 0 else 1.0

    kelly_full = (p * b - q) / b
    if kelly_full <= 0:
        return 0.0

    return kelly_full * kelly_fraction


# ---------------------------------------------------------------------------
# ATR Calculation
# ---------------------------------------------------------------------------

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Compute Average True Range over the given period."""
    if len(close) < period + 1:
        return 0.0

    tr_values = []
    for i in range(1, len(close)):
        tr = max(
            high.iloc[i] - low.iloc[i],
            abs(high.iloc[i] - close.iloc[i - 1]),
            abs(low.iloc[i] - close.iloc[i - 1]),
        )
        tr_values.append(tr)

    if len(tr_values) < period:
        return np.mean(tr_values) if tr_values else 0.0

    return np.mean(tr_values[-period:])


# ---------------------------------------------------------------------------
# Price Data Fetching
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers in the universe.
    Returns dict of ticker -> DataFrame with columns [Open, High, Low, Close, Volume].
    """
    # Add buffer for lookback calculations
    buffer_start = start - timedelta(days=400)
    price_data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=buffer_start, end=end, progress=False)
            if df.empty:
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.date if hasattr(df.index, 'date') else df.index
            price_data[ticker] = df
        except Exception:
            continue

    return price_data


# ---------------------------------------------------------------------------
# Portfolio Engine
# ---------------------------------------------------------------------------

class Portfolio:
    """Tracks portfolio state through the backtest."""

    def __init__(self, initial_capital: float):
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_log: list[Trade] = []
        self.equity_history: list[tuple[date, float]] = []

    def total_value(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value: cash + mark-to-market positions."""
        position_value = sum(
            pos.shares * current_prices.get(pos.ticker, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def record_equity(self, dt: date, current_prices: dict[str, float]) -> None:
        """Snapshot the portfolio value for equity curve."""
        self.equity_history.append((dt, self.total_value(current_prices)))

    def open_position(
        self,
        ticker: str,
        shares: int,
        price: float,
        dt: date,
        stop_price: float,
        signals_fired: list[str],
        posterior: float,
    ) -> None:
        """Enter a new position."""
        cost = shares * price
        if cost > self.cash:
            # Reduce shares to fit available cash
            shares = int(self.cash / price)
            cost = shares * price
        if shares <= 0:
            return

        self.cash -= cost
        self.positions[ticker] = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            entry_date=dt,
            stop_price=stop_price,
            signals_fired=signals_fired,
            posterior_at_entry=posterior,
        )

    def close_position(self, ticker: str, price: float, dt: date) -> Optional[Trade]:
        """Exit a position and log the trade."""
        if ticker not in self.positions:
            return None

        pos = self.positions.pop(ticker)
        proceeds = pos.shares * price
        self.cash += proceeds

        pnl = (price - pos.entry_price) * pos.shares
        return_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0

        trade = Trade(
            ticker=pos.ticker,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=dt,
            exit_price=price,
            shares=pos.shares,
            pnl=pnl,
            return_pct=return_pct,
            signals_fired=pos.signals_fired,
            posterior_at_entry=pos.posterior_at_entry,
        )
        self.trade_log.append(trade)
        return trade


# ---------------------------------------------------------------------------
# Main Backtest Loop
# ---------------------------------------------------------------------------

def run_backtest(
    config: BacktestConfig,
    universe: list[str],
    insider_cache: Optional[dict] = None,
) -> BacktestResult:
    """
    Run the full event-driven backtest.

    Args:
        config: BacktestConfig with all parameters.
        universe: List of ASX tickers to trade (e.g. ['CBA.AX', ...]).
        insider_cache: Optional dict of ticker -> list of insider trade dicts.

    Returns:
        BacktestResult with all metrics and trade list.
    """
    if insider_cache is None:
        insider_cache = {}

    # Fetch all price data upfront
    print(f"Fetching price data for {len(universe)} tickers...")
    price_data = fetch_price_data(universe, config.start_date, config.end_date)
    available_tickers = list(price_data.keys())
    print(f"Got data for {len(available_tickers)} tickers.")

    # Fetch VIX for regime detection
    print("Fetching VIX data...")
    vix_series = fetch_vix_series(
        config.start_date - timedelta(days=5),
        config.end_date,
    )

    # Build trading calendar from available data
    all_dates: set[date] = set()
    for df in price_data.values():
        all_dates.update(df.index)
    trading_days = sorted(d for d in all_dates if config.start_date <= d <= config.end_date)

    if not trading_days:
        raise ValueError("No trading days found in the specified date range.")

    print(f"Simulating {len(trading_days)} trading days from {trading_days[0]} to {trading_days[-1]}...")

    portfolio = Portfolio(config.initial_capital)
    pending_entries: list[dict] = []  # orders to execute at next day's open

    for day_idx, today in enumerate(trading_days):
        # Get current prices for all tickers
        current_prices: dict[str, float] = {}
        for ticker, df in price_data.items():
            if today in df.index:
                current_prices[ticker] = float(df.loc[today, "Close"])

        # --- Step 0: Execute pending entries from yesterday at today's open ---
        for order in pending_entries:
            ticker = order["ticker"]
            if ticker in price_data and today in price_data[ticker].index:
                open_price = float(price_data[ticker].loc[today, "Open"])
                portfolio.open_position(
                    ticker=ticker,
                    shares=order["shares"],
                    price=open_price,
                    dt=today,
                    stop_price=order["stop_price"],
                    signals_fired=order["signals_fired"],
                    posterior=order["posterior"],
                )
        pending_entries = []

        # --- Step 1: Check stops ---
        tickers_to_close = []
        for ticker, pos in portfolio.positions.items():
            price = current_prices.get(ticker)
            if price is not None and price < pos.stop_price:
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            price = current_prices[ticker]
            portfolio.close_position(ticker, price, today)

        # --- Step 2: Check regime ---
        vix_today = vix_series.get(today)
        if vix_today is None:
            # Use most recent available VIX value
            prior_dates = [d for d in vix_series.index if d <= today]
            vix_today = vix_series[prior_dates[-1]] if prior_dates else 20.0

        regime = detect_regime(float(vix_today))
        skip_new_entries = regime == "risk_off"

        # --- Step 3-5: Signal generation, aggregation, sizing ---
        if not skip_new_entries:
            for ticker in available_tickers:
                # Skip if already in portfolio
                if ticker in portfolio.positions:
                    continue

                df = price_data[ticker]
                if today not in df.index:
                    continue

                # Need enough historical data
                hist_mask = df.index <= today
                hist = df.loc[hist_mask]
                if len(hist) < config.momentum_lookback:
                    continue

                # Compute signals
                signals: dict[str, float] = {}

                # Momentum signal (convert raw return to probability-like score)
                mom = compute_momentum(
                    hist["Close"],
                    lookback=config.momentum_lookback,
                    skip=config.momentum_skip,
                )
                if mom is not None:
                    # Convert momentum to a probability estimate via sigmoid
                    mom_signal = 1.0 / (1.0 + math.exp(-10 * mom))
                    signals["momentum"] = mom_signal

                # Insider signal
                insider_score = compute_insider_signal(ticker, today, insider_cache)
                if insider_score is not None:
                    signals["insider"] = 0.5 + 0.5 * insider_score  # map [0,1] to [0.5, 1.0]

                if not signals:
                    continue

                # Bayesian aggregation
                posterior = bayesian_aggregate(signals)

                # Apply minimum posterior threshold
                if posterior < config.min_posterior:
                    continue

                # In selective regime, require higher conviction
                if regime == "selective" and posterior < 0.85:
                    continue

                # Kelly sizing
                position_fraction = kelly_size(
                    posterior=posterior,
                    kelly_fraction=config.kelly_fraction,
                )
                if position_fraction <= 0:
                    continue

                # Cap at max position size
                position_fraction = min(position_fraction, config.max_position_pct)

                # Calculate shares
                portfolio_value = portfolio.total_value(current_prices)
                position_value = portfolio_value * position_fraction
                current_price = current_prices[ticker]
                shares = int(position_value / current_price)

                if shares <= 0:
                    continue

                # Calculate stop price using ATR
                atr = compute_atr(
                    hist["High"],
                    hist["Low"],
                    hist["Close"],
                    period=config.atr_period,
                )
                stop_price = current_price - (config.stop_loss_atr_multiple * atr)

                # Queue for next-day-open execution
                pending_entries.append({
                    "ticker": ticker,
                    "shares": shares,
                    "stop_price": stop_price,
                    "signals_fired": list(signals.keys()),
                    "posterior": posterior,
                })

        # --- Step 6: Record daily equity ---
        portfolio.record_equity(today, current_prices)

    # Close all remaining positions at final prices on last day
    last_day = trading_days[-1]
    for ticker in list(portfolio.positions.keys()):
        if ticker in current_prices:
            portfolio.close_position(ticker, current_prices[ticker], last_day)

    # --- Compute result metrics ---
    return _compute_result(portfolio, config)


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def _compute_result(portfolio: Portfolio, config: BacktestConfig) -> BacktestResult:
    """Compute all performance metrics from the completed backtest."""
    trades = portfolio.trade_log
    equity_series = pd.Series(
        {dt: val for dt, val in portfolio.equity_history},
        dtype=float,
    )

    # Total and annualized return
    if equity_series.empty:
        total_return = 0.0
        annualized_return = 0.0
    else:
        total_return = (equity_series.iloc[-1] / config.initial_capital) - 1.0
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.25 if days > 0 else 1.0
        annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0

    # Sharpe ratio (annualized, daily returns)
    if len(equity_series) > 1:
        daily_returns = equity_series.pct_change().dropna()
        excess_returns = daily_returns - (config.risk_free_rate / 252)
        if excess_returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * math.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Max drawdown (peak-to-trough)
    if len(equity_series) > 0:
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = abs(drawdown.min())
    else:
        max_drawdown = 0.0

    # Trade statistics
    total_trades = len(trades)
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    win_rate = len(winners) / total_trades if total_trades > 0 else 0.0
    avg_winner = np.mean([t.return_pct for t in winners]) if winners else 0.0
    avg_loser = np.mean([t.return_pct for t in losers]) if losers else 0.0

    # Profit factor
    gross_profit = sum(t.pnl for t in winners) if winners else 0.0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return BacktestResult(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        avg_winner=float(avg_winner),
        avg_loser=float(avg_loser),
        total_trades=total_trades,
        profit_factor=profit_factor,
        trades=trades,
        equity_curve=equity_series,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(result: BacktestResult) -> None:
    """Display key backtest metrics in a readable format."""
    print("\n" + "=" * 60)
    print("           BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total Return:        {result.total_return:>10.2%}")
    print(f"  Annualized Return:   {result.annualized_return:>10.2%}")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown:>10.2%}")
    print("-" * 60)
    print(f"  Total Trades:        {result.total_trades:>10d}")
    print(f"  Win Rate:            {result.win_rate:>10.2%}")
    print(f"  Avg Winner:          {result.avg_winner:>10.2%}")
    print(f"  Avg Loser:           {result.avg_loser:>10.2%}")
    print(f"  Profit Factor:       {result.profit_factor:>10.2f}")
    print("=" * 60)

    if not result.equity_curve.empty:
        print(f"\n  Start Equity:  ${result.equity_curve.iloc[0]:>12,.2f}")
        print(f"  End Equity:    ${result.equity_curve.iloc[-1]:>12,.2f}")
        print(f"  Period:        {result.equity_curve.index[0]} to {result.equity_curve.index[-1]}")
    print()


def signal_attribution_report(result: BacktestResult) -> pd.DataFrame:
    """
    Analyse which signals contributed most to P&L.

    Returns a DataFrame with columns:
    - signal: name of the signal
    - trades: number of trades where this signal fired
    - total_pnl: sum of P&L for trades where this signal fired
    - avg_return: average return for trades with this signal
    - win_rate: percentage of winning trades with this signal
    """
    if not result.trades:
        print("No trades to attribute.")
        return pd.DataFrame()

    # Build attribution data
    signal_stats: dict[str, dict] = {}

    for trade in result.trades:
        for signal in trade.signals_fired:
            if signal not in signal_stats:
                signal_stats[signal] = {
                    "trades": 0,
                    "total_pnl": 0.0,
                    "returns": [],
                    "wins": 0,
                }
            stats = signal_stats[signal]
            stats["trades"] += 1
            stats["total_pnl"] += trade.pnl
            stats["returns"].append(trade.return_pct)
            if trade.pnl > 0:
                stats["wins"] += 1

    rows = []
    for signal, stats in signal_stats.items():
        rows.append({
            "signal": signal,
            "trades": stats["trades"],
            "total_pnl": round(stats["total_pnl"], 2),
            "avg_return": round(np.mean(stats["returns"]), 4),
            "win_rate": round(stats["wins"] / stats["trades"], 4) if stats["trades"] > 0 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)

    # Pretty print
    print("\n" + "=" * 60)
    print("           SIGNAL ATTRIBUTION REPORT")
    print("=" * 60)
    for _, row in df.iterrows():
        print(
            f"  {row['signal']:<20s}  "
            f"Trades: {row['trades']:>4d}  "
            f"P&L: ${row['total_pnl']:>10,.2f}  "
            f"Avg Ret: {row['avg_return']:>7.2%}  "
            f"Win: {row['win_rate']:>6.1%}"
        )
    print("=" * 60 + "\n")

    return df


# ---------------------------------------------------------------------------
# Convenience Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage with a small set of ASX tickers
    example_universe = [
        "CBA.AX", "BHP.AX", "CSL.AX", "WBC.AX", "NAB.AX",
        "ANZ.AX", "MQG.AX", "WES.AX", "TLS.AX", "RIO.AX",
    ]

    config = BacktestConfig(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        initial_capital=100_000,
    )

    result = run_backtest(config, universe=example_universe)
    print_summary(result)
    signal_attribution_report(result)
