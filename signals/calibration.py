"""
Self-Learning Calibration Engine

After every closed trade, recalculates likelihood ratios from actual data.
This is what makes the system improve over time — not static assumptions,
but empirically measured edge from YOUR track record.

The loop:
1. Trade closes → log outcome
2. Recalculate win rate per signal from all historical trades
3. Convert win rates to likelihood ratios
4. Next trade uses updated LRs
5. Repeat forever
"""

import json
from data.db import get_db
from data.activity import log_activity

# Minimum trades before we trust empirical data over priors
MIN_TRADES_FOR_CALIBRATION = 10

# Default priors (used until we have enough data)
DEFAULT_LRS = {
    "momentum": 1.4,
    "insider": 1.8,
    "sentiment": 1.6,
    "risk_factors": 0.5,
    "alt_data": 1.3,
}


def get_calibrated_likelihood_ratios() -> dict[str, float]:
    """
    Returns likelihood ratios based on actual trading history.
    Falls back to defaults if insufficient data.
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT signals_at_entry, pnl FROM trades WHERE status='closed' AND pnl IS NOT NULL"
        ).fetchall()

    if len(rows) < MIN_TRADES_FOR_CALIBRATION:
        return DEFAULT_LRS.copy()

    # Count wins/losses per signal
    signal_stats: dict[str, dict] = {}
    for row in rows:
        signals = json.loads(row["signals_at_entry"])
        won = (row["pnl"] or 0) > 0

        for signal_name, info in signals.items():
            if not info.get("active", False):
                continue
            if signal_name not in signal_stats:
                signal_stats[signal_name] = {"wins": 0, "losses": 0, "total": 0}
            signal_stats[signal_name]["total"] += 1
            if won:
                signal_stats[signal_name]["wins"] += 1
            else:
                signal_stats[signal_name]["losses"] += 1

    # Convert win rates to likelihood ratios
    # LR = P(signal fires | stock wins) / P(signal fires | stock loses)
    # Simplified: LR = win_rate / (1 - win_rate) * (1 - base_rate) / base_rate
    # With base_rate = 0.5, this simplifies to: LR = win_rate / (1 - win_rate)
    calibrated = {}
    for signal_name, stats in signal_stats.items():
        if stats["total"] < 5:
            calibrated[signal_name] = DEFAULT_LRS.get(signal_name, 1.0)
            continue

        win_rate = stats["wins"] / stats["total"]
        # Clamp to avoid extreme values
        win_rate = max(0.2, min(0.9, win_rate))
        lr = win_rate / (1 - win_rate)
        # Blend with prior (70% empirical, 30% prior) for stability
        prior_lr = DEFAULT_LRS.get(signal_name, 1.0)
        blended = 0.7 * lr + 0.3 * prior_lr
        calibrated[signal_name] = round(blended, 3)

    # Fill in any missing signals with defaults
    for signal_name, default_lr in DEFAULT_LRS.items():
        if signal_name not in calibrated:
            calibrated[signal_name] = default_lr

    return calibrated


def recalibrate() -> dict:
    """
    Run recalibration and log results.
    Called after each trade closes.
    Returns the new LRs and any signals that were killed/boosted.
    """
    old_lrs = DEFAULT_LRS.copy()
    new_lrs = get_calibrated_likelihood_ratios()

    changes = []
    for signal, new_lr in new_lrs.items():
        old_lr = old_lrs.get(signal, 1.0)
        if abs(new_lr - old_lr) > 0.1:
            direction = "BOOSTED" if new_lr > old_lr else "WEAKENED"
            changes.append(f"{signal}: {old_lr:.2f} → {new_lr:.2f} ({direction})")

    # Kill signals that have LR < 0.8 (consistently wrong)
    killed = [s for s, lr in new_lrs.items() if lr < 0.8 and s != "risk_factors"]
    # Boost signals that have LR > 2.0 (consistently right)
    boosted = [s for s, lr in new_lrs.items() if lr > 2.0]

    if changes:
        log_activity("signal",
                     "Self-calibration complete",
                     f"Updated LRs: {'; '.join(changes)}",
                     severity="info")

    if killed:
        log_activity("signal",
                     f"Signal weakening: {', '.join(killed)}",
                     "LR dropped below 0.8 — signal may be noise. Will reduce weight.",
                     severity="warning")

    if boosted:
        log_activity("signal",
                     f"Signal strengthening: {', '.join(boosted)}",
                     "LR above 2.0 — strong empirical edge confirmed.",
                     severity="success")

    # Save calibrated LRs to DB for persistence
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fund_state (key, value) VALUES ('calibrated_lrs', ?)",
            (json.dumps(new_lrs),),
        )

    return {
        "likelihood_ratios": new_lrs,
        "changes": changes,
        "killed_signals": killed,
        "boosted_signals": boosted,
    }


def get_active_lrs() -> dict[str, float]:
    """
    Get the currently active LRs (calibrated if available, else defaults).
    This is what the pipeline should call instead of hardcoded values.
    """
    with get_db() as conn:
        row = conn.execute("SELECT value FROM fund_state WHERE key='calibrated_lrs'").fetchone()

    if row:
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            pass

    return DEFAULT_LRS.copy()
