"""
Position Monitor — Exit Management for Open Trades

Runs after every pipeline cycle. Checks all open positions against exit triggers:
1. Stop loss hit (hard stop)
2. Trailing stop (dynamic stop adjustment)
3. Signal fade (posterior drops below threshold)
4. Time decay (held too long with no fresh signals)

Each position is checked independently with fault isolation so one failure
does not prevent the rest from being evaluated.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from data.activity import log_activity
from data.db import get_db
from data.provider import get_price_history
from execution.broker import get_broker, Order, Side, OrderType
from signals.momentum import compute_momentum_signal

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

SIGNAL_FADE_THRESHOLD = 0.45
TIME_DECAY_DAYS = 60
TRAILING_BREAKEVEN_PCT = 0.15  # Move stop to breakeven when up 15%
TRAILING_LOCK_IN_PCT = 0.30    # Lock in gains when up 30%
TRAILING_LOCK_IN_STOP = 0.15   # Lock in 15% of gains


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def monitor_positions() -> dict:
    """
    Main monitor function. Checks all open positions and applies exit logic.

    Returns:
        Dict with positions_checked, exits_made, total_pnl_realized.
    """
    results = {
        "positions_checked": 0,
        "exits_made": 0,
        "total_pnl_realized": 0.0,
    }

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'open'"
        ).fetchall()

    open_trades = [dict(row) for row in rows]

    if not open_trades:
        logger.info("No open positions to monitor.")
        return results

    logger.info(f"Monitoring {len(open_trades)} open position(s).")

    for trade in open_trades:
        results["positions_checked"] += 1
        try:
            exit_result = _evaluate_position(trade)
            if exit_result is not None:
                results["exits_made"] += 1
                results["total_pnl_realized"] += exit_result
        except Exception as e:
            ticker = trade.get("ticker", "UNKNOWN")
            logger.error(f"Error monitoring position {ticker} (id={trade.get('id')}): {e}")
            log_activity(
                category="error",
                title=f"Monitor error: {ticker}",
                detail=str(e),
                ticker=ticker,
                severity="warning",
            )

    if results["exits_made"] > 0:
        log_activity(
            category="trade",
            title=f"Monitor cycle complete: {results['exits_made']} exit(s)",
            detail=f"Checked {results['positions_checked']} positions. "
                   f"Realized P&L: ${results['total_pnl_realized']:.2f}",
            severity="info",
        )

    return results


# ─── Position Evaluation ──────────────────────────────────────────────────────

def _evaluate_position(trade: dict) -> Optional[float]:
    """
    Evaluate a single open position against all exit triggers.

    Returns realized PnL if an exit was triggered, None otherwise.
    """
    ticker = trade["ticker"]
    current_price = _get_current_price(ticker)

    if current_price is None:
        logger.warning(f"Could not fetch price for {ticker}, skipping.")
        return None

    # 1. Check hard stop loss
    if _check_stop_loss(trade, current_price):
        return _execute_exit(trade, current_price, reason="stop_loss", status="stopped_out")

    # 2. Check trailing stop (may update stop price in DB)
    new_stop = _check_trailing_stop(trade, current_price)
    if new_stop is not None:
        # Update stop price in the database
        with get_db() as conn:
            conn.execute(
                "UPDATE trades SET stop_price = ? WHERE id = ?",
                (new_stop, trade["id"]),
            )
        trade["stop_price"] = new_stop
        logger.info(f"{ticker}: Trailing stop updated to ${new_stop:.4f}")
        log_activity(
            category="trade",
            title=f"Trailing stop raised: {ticker}",
            detail=f"New stop: ${new_stop:.4f} (current: ${current_price:.4f})",
            ticker=ticker,
            severity="info",
        )

        # Re-check if the new trailing stop is already breached
        # (shouldn't happen in normal flow, but be defensive)
        if current_price <= new_stop:
            return _execute_exit(trade, current_price, reason="trailing_stop", status="stopped_out")

    # 3. Check signal fade
    if _check_signal_fade(trade):
        return _execute_exit(trade, current_price, reason="signal_fade", status="closed")

    # 4. Check time decay
    if _check_time_decay(trade):
        return _execute_exit(trade, current_price, reason="time_decay", status="closed")

    return None


# ─── Price Fetching ───────────────────────────────────────────────────────────

def _get_current_price(ticker: str) -> Optional[float]:
    """
    Fetch the latest closing price for a ticker from EODHD provider.

    Returns None if price cannot be retrieved.
    """
    try:
        df = get_price_history(ticker, period_days=5)
        if df.empty:
            return None
        # Get the most recent close price
        return float(df["close"].iloc[-1])
    except Exception as e:
        logger.error(f"Failed to fetch price for {ticker}: {e}")
        return None


# ─── Exit Trigger Checks ──────────────────────────────────────────────────────

def _check_stop_loss(trade: dict, current_price: float) -> bool:
    """
    Check if the hard stop loss has been hit.

    Returns True if current_price <= stop_price.
    """
    stop_price = trade.get("stop_price")
    if stop_price is None:
        return False
    return current_price <= stop_price


def _check_trailing_stop(trade: dict, current_price: float) -> Optional[float]:
    """
    Check if the trailing stop should be moved up.

    Rules:
    - If position is up >15% from entry, move stop to entry_price (breakeven).
    - If position is up >30% from entry, move stop to entry + 15%.

    Returns the new stop price if it should be raised, None otherwise.
    The stop is only ever moved UP, never down.
    """
    entry_price = trade["entry_price"]
    current_stop = trade.get("stop_price")

    if entry_price is None or entry_price <= 0:
        return None

    gain_pct = (current_price - entry_price) / entry_price

    new_stop: Optional[float] = None

    if gain_pct > TRAILING_LOCK_IN_PCT:
        # Up >30%: lock in 15% gain
        new_stop = entry_price * (1.0 + TRAILING_LOCK_IN_STOP)
    elif gain_pct > TRAILING_BREAKEVEN_PCT:
        # Up >15%: move stop to breakeven
        new_stop = entry_price

    if new_stop is None:
        return None

    # Only move stop UP, never down
    if current_stop is not None and new_stop <= current_stop:
        return None

    return round(new_stop, 4)


def _check_signal_fade(trade: dict) -> bool:
    """
    Check if the momentum signal has faded for this position.

    Re-evaluates whether the stock is still in momentum quintile 4 or 5.
    If it has dropped below quintile 4, the signal has faded.

    This is a lightweight check — only momentum is re-evaluated, not the
    full insider/value/sentiment stack.
    """
    ticker = trade["ticker"]

    try:
        # Fetch the current universe from the database to compute momentum quintiles
        with get_db() as conn:
            rows = conn.execute(
                "SELECT ticker, return_12m FROM universe WHERE return_12m IS NOT NULL"
            ).fetchall()

        if not rows:
            # Cannot evaluate without universe data — do not trigger exit
            return False

        universe_df = pd.DataFrame([dict(r) for r in rows])
        momentum_df = compute_momentum_signal(universe_df)

        if momentum_df.empty:
            return False

        # Find this ticker's current momentum quintile
        ticker_row = momentum_df[momentum_df["ticker"] == ticker]

        if ticker_row.empty:
            # Ticker no longer in universe — signal has faded
            logger.info(f"{ticker}: No longer in universe, signal faded.")
            return True

        quintile = ticker_row["momentum_quintile"].iloc[0]

        if pd.isna(quintile):
            return False

        # Signal fade: dropped below quintile 4
        if quintile < 4.0:
            logger.info(f"{ticker}: Momentum quintile dropped to {quintile}, signal faded.")
            return True

        return False

    except Exception as e:
        logger.error(f"Signal fade check failed for {ticker}: {e}")
        # On error, do not trigger exit
        return False


def _check_time_decay(trade: dict) -> bool:
    """
    Check if the position has been held too long without new signal activity.

    Triggers if position held > 60 days with no new signal firing for this ticker.
    """
    entry_date_str = trade.get("entry_date")
    if not entry_date_str:
        return False

    try:
        entry_date = datetime.fromisoformat(entry_date_str)
    except (ValueError, TypeError):
        return False

    days_held = (datetime.now() - entry_date).days

    if days_held <= TIME_DECAY_DAYS:
        return False

    # Check if there has been any recent signal activity for this ticker
    ticker = trade["ticker"]
    cutoff_date = (datetime.now() - timedelta(days=TIME_DECAY_DAYS)).isoformat()

    try:
        with get_db() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM signal_states
                   WHERE ticker = ? AND date > ? AND (
                       momentum_active = 1 OR insider_active = 1 OR
                       sentiment_active = 1 OR alt_data_active = 1
                   )""",
                (ticker, cutoff_date),
            ).fetchone()

        if row and row["cnt"] > 0:
            # Fresh signals exist — do not trigger time decay
            return False

        logger.info(f"{ticker}: Held {days_held} days with no fresh signals, time decay triggered.")
        return True

    except Exception as e:
        logger.error(f"Time decay check failed for {ticker}: {e}")
        return False


# ─── Exit Execution ───────────────────────────────────────────────────────────

def _execute_exit(trade: dict, current_price: float, reason: str, status: str) -> float:
    """
    Execute a position exit: place sell order, update DB, log activity.

    Args:
        trade: The trade dict from the database.
        current_price: Current market price.
        reason: Exit reason (stop_loss, trailing_stop, signal_fade, time_decay).
        status: New trade status (stopped_out, closed).

    Returns:
        Realized PnL for this trade.
    """
    ticker = trade["ticker"]
    shares = trade["shares"]
    entry_price = trade["entry_price"]
    trade_id = trade["id"]

    # Calculate PnL
    pnl = (current_price - entry_price) * shares
    return_pct = ((current_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0

    # Place sell order via broker
    broker = get_broker()
    try:
        broker.connect()
        order = Order(
            ticker=ticker,
            side=Side.SELL,
            quantity=shares,
            order_type=OrderType.MARKET,
        )
        order_id = broker.place_order(order)
        logger.info(f"{ticker}: Sell order placed (id={order_id}), reason={reason}")
    except Exception as e:
        logger.error(f"{ticker}: Failed to place sell order: {e}")
        log_activity(
            category="error",
            title=f"Sell order failed: {ticker}",
            detail=f"Reason: {reason}. Error: {e}",
            ticker=ticker,
            severity="alert",
        )
        raise
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass

    # Update trade record in database
    exit_date = datetime.now().isoformat()

    with get_db() as conn:
        conn.execute(
            """UPDATE trades
               SET exit_date = ?, exit_price = ?, pnl = ?, return_pct = ?, status = ?
               WHERE id = ?""",
            (exit_date, current_price, pnl, return_pct, status, trade_id),
        )

    # Log to activity feed
    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    severity = "success" if pnl >= 0 else "warning"

    reason_labels = {
        "stop_loss": "Stop loss hit",
        "trailing_stop": "Trailing stop hit",
        "signal_fade": "Signal faded (momentum dropped)",
        "time_decay": f"Time decay (>{TIME_DECAY_DAYS} days, no fresh signals)",
    }

    log_activity(
        category="trade",
        title=f"EXIT {ticker} — {reason_labels.get(reason, reason)}",
        detail=(
            f"Sold {shares} shares @ ${current_price:.4f} | "
            f"Entry: ${entry_price:.4f} | P&L: {pnl_str} ({return_pct:+.1f}%) | "
            f"Status: {status}"
        ),
        ticker=ticker,
        severity=severity,
    )

    logger.info(
        f"{ticker}: Exited — {reason} | {shares} shares @ ${current_price:.4f} | "
        f"P&L: {pnl_str} ({return_pct:+.1f}%)"
    )

    return pnl
