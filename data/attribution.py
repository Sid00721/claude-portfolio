"""
Attribution Engine

Logs every trade decision and its outcome, then computes which signals
actually generate alpha. Uses local SQLite for storage.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from data.db import get_db


@dataclass
class TradeRecord:
    ticker: str
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl: Optional[float]
    return_pct: Optional[float]
    signals_at_entry: dict
    posterior_at_entry: float
    regime_at_entry: str
    kelly_size_pct: float


def log_trade(record: TradeRecord) -> None:
    """Save a TradeRecord to the local trades table."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO trades
               (ticker, entry_date, exit_date, entry_price, exit_price,
                position_size, pnl, return_pct, signals_at_entry,
                posterior_at_entry, regime_at_entry, kelly_size_pct, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.ticker,
                record.entry_date,
                record.exit_date,
                record.entry_price,
                record.exit_price,
                record.position_size,
                record.pnl,
                record.return_pct,
                json.dumps(record.signals_at_entry),
                record.posterior_at_entry,
                record.regime_at_entry,
                record.kelly_size_pct,
                "open" if record.exit_date is None else "closed",
            ),
        )


def log_signal_state(ticker: str, date_str: str, signals: dict) -> None:
    """Save a daily signal snapshot."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_states (ticker, date, raw_signals)
               VALUES (?, ?, ?)""",
            (ticker, date_str, json.dumps(signals)),
        )


def get_signal_attribution(lookback_days: int = 90) -> pd.DataFrame:
    """
    Compute attribution for each signal over the lookback period.
    Returns DataFrame sorted by contribution to total PnL.
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE entry_date >= ? AND status = 'closed'",
            (cutoff,),
        ).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=["signal", "times_fired", "win_rate_when_fired",
                     "avg_return_when_fired", "contribution_to_total_pnl"]
        )

    trades = [dict(r) for r in rows]
    total_pnl = sum(t["pnl"] or 0 for t in trades)

    signal_stats: dict = {}
    for trade in trades:
        signals_at_entry = json.loads(trade.get("signals_at_entry", "{}"))
        for signal_name, info in signals_at_entry.items():
            if not info.get("active", False):
                continue
            if signal_name not in signal_stats:
                signal_stats[signal_name] = {
                    "times_fired": 0, "wins": 0, "returns": [], "pnl_sum": 0.0
                }
            stats = signal_stats[signal_name]
            stats["times_fired"] += 1
            if (trade["pnl"] or 0) > 0:
                stats["wins"] += 1
            stats["returns"].append(trade["return_pct"] or 0)
            stats["pnl_sum"] += trade["pnl"] or 0

    rows_out = []
    for signal_name, stats in signal_stats.items():
        n = stats["times_fired"]
        rows_out.append({
            "signal": signal_name,
            "times_fired": n,
            "win_rate_when_fired": round(stats["wins"] / n, 4) if n > 0 else 0.0,
            "avg_return_when_fired": round(sum(stats["returns"]) / len(stats["returns"]), 4) if stats["returns"] else 0.0,
            "contribution_to_total_pnl": round(stats["pnl_sum"] / total_pnl, 4) if total_pnl != 0 else 0.0,
        })

    df = pd.DataFrame(rows_out)
    return df.sort_values("contribution_to_total_pnl", ascending=False).reset_index(drop=True)


def get_regime_attribution(lookback_days: int = 90) -> pd.DataFrame:
    """Performance broken down by market regime."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE entry_date >= ? AND status = 'closed'",
            (cutoff,),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["regime", "num_trades", "win_rate", "avg_return", "total_pnl"])

    trades = [dict(r) for r in rows]
    regime_stats: dict = {}
    for trade in trades:
        regime = trade.get("regime_at_entry", "unknown")
        if regime not in regime_stats:
            regime_stats[regime] = {"num_trades": 0, "wins": 0, "returns": [], "total_pnl": 0.0}
        stats = regime_stats[regime]
        stats["num_trades"] += 1
        if (trade["pnl"] or 0) > 0:
            stats["wins"] += 1
        stats["returns"].append(trade["return_pct"] or 0)
        stats["total_pnl"] += trade["pnl"] or 0

    rows_out = []
    for regime, stats in regime_stats.items():
        n = stats["num_trades"]
        rows_out.append({
            "regime": regime,
            "num_trades": n,
            "win_rate": round(stats["wins"] / n, 4) if n > 0 else 0.0,
            "avg_return": round(sum(stats["returns"]) / len(stats["returns"]), 4) if stats["returns"] else 0.0,
            "total_pnl": round(stats["total_pnl"], 2),
        })

    return pd.DataFrame(rows_out).sort_values("total_pnl", ascending=False).reset_index(drop=True)


def get_portfolio_stats(lookback_days: int = 90) -> dict:
    """Aggregate portfolio statistics over the lookback period."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE entry_date >= ? AND status = 'closed' ORDER BY entry_date",
            (cutoff,),
        ).fetchall()

    if not rows:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "avg_trade": 0.0}

    trades = [dict(r) for r in rows]
    returns = [t["return_pct"] or 0 for t in trades]
    pnls = [t["pnl"] or 0 for t in trades]

    total_return = sum(returns)
    avg_trade = sum(pnls) / len(pnls)
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

    returns_series = pd.Series(returns)
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0.0

    cumulative = pd.Series(pnls).cumsum()
    running_max = cumulative.cummax()
    max_drawdown = (cumulative - running_max).min()

    return {
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": round(win_rate, 4),
        "avg_trade": round(avg_trade, 2),
    }


def generate_report(lookback_days: int = 90) -> str:
    """Formatted summary of all attribution metrics."""
    stats = get_portfolio_stats(lookback_days)
    signal_attr = get_signal_attribution(lookback_days)
    regime_attr = get_regime_attribution(lookback_days)

    lines = [
        "=" * 60,
        f"  ATTRIBUTION REPORT  (last {lookback_days} days)",
        "=" * 60,
        "",
        "--- Portfolio Stats ---",
        f"  Total Return:   {stats['total_return']:.2%}",
        f"  Sharpe Ratio:   {stats['sharpe']:.2f}",
        f"  Max Drawdown:   ${stats['max_drawdown']:.2f}",
        f"  Win Rate:       {stats['win_rate']:.2%}",
        f"  Avg Trade PnL:  ${stats['avg_trade']:.2f}",
        "",
        "--- Signal Attribution ---",
    ]

    if signal_attr.empty:
        lines.append("  No signal data available.")
    else:
        lines.append(f"  {'Signal':<20} {'Fired':<8} {'Win%':<8} {'AvgRet':<10} {'Contribution':<12}")
        lines.append("  " + "-" * 58)
        for _, row in signal_attr.iterrows():
            lines.append(
                f"  {row['signal']:<20} {row['times_fired']:<8} "
                f"{row['win_rate_when_fired']:.1%}   "
                f"{row['avg_return_when_fired']:.2%}     "
                f"{row['contribution_to_total_pnl']:.1%}"
            )

    lines.append("")
    lines.append("--- Regime Attribution ---")

    if regime_attr.empty:
        lines.append("  No regime data available.")
    else:
        lines.append(f"  {'Regime':<15} {'Trades':<8} {'Win%':<8} {'AvgRet':<10} {'TotalPnL':<10}")
        lines.append("  " + "-" * 51)
        for _, row in regime_attr.iterrows():
            lines.append(
                f"  {row['regime']:<15} {row['num_trades']:<8} "
                f"{row['win_rate']:.1%}   "
                f"{row['avg_return']:.2%}     "
                f"${row['total_pnl']:.2f}"
            )

    lines.extend(["", "=" * 60])
    return "\n".join(lines)
