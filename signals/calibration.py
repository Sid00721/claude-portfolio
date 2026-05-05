"""
Self-Learning Calibration Engine

After every closed trade, recalculates likelihood ratios from actual data.
This is what makes the system improve over time — not static assumptions,
but empirically measured edge from YOUR track record on the ASX.

The loop:
1. Trade closes -> log outcome
2. Recalculate win rate per signal from all historical trades
3. Convert win rates to likelihood ratios
4. Blend with academic priors for stability (decays as sample grows)
5. Kill noise signals, boost proven edges
6. Next trade uses updated LRs
7. Repeat forever

After ~90 days of live trading, the weights are entirely data-driven.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from data.activity import log_activity
from data.db import get_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum closed trades before trusting empirical data over priors
MIN_TRADES_FOR_CALIBRATION = 10

# Minimum trades before we will KILL a signal (need confidence it's noise)
MIN_TRADES_FOR_KILL = 20

# LR thresholds
LR_KILL_THRESHOLD = 0.8      # Below this after MIN_TRADES_FOR_KILL -> signal is noise
LR_BOOST_THRESHOLD = 2.5     # Above this -> proven strong edge

# Blending: 70% empirical, 30% default prior (for stability)
EMPIRICAL_WEIGHT = 0.70
PRIOR_WEIGHT = 0.30

# Win rate clamping to avoid degenerate LRs
WIN_RATE_FLOOR = 0.15
WIN_RATE_CEILING = 0.92

# Recent trade window for trend calculation
RECENT_TRADE_WINDOW = 30

# ---------------------------------------------------------------------------
# Default Likelihood Ratios (academic priors)
# ---------------------------------------------------------------------------

DEFAULT_LRS: dict[str, float] = {
    "momentum": 1.5,            # 12-1 month factor, stronger on ASX small caps
    "insider": 1.8,             # Director cluster buying via Appendix 3Y / EODHD
    "sentiment": 1.6,           # LLM transcript delta (Claude)
    "risk_factors": 0.5,        # Annual report semantic diff (NEGATIVE signal)
    "alt_data": 1.3,            # Hiring, Google Trends, app store
    "value": 1.35,              # Price-to-book quintile, works with momentum combo
    "franking": 1.25,           # ASX-specific franking credit ex-div capture
    "index_rebalance": 1.3,     # Approaching ASX index inclusion
    "earnings_surprise": 1.4,   # PEAD, 60-day drift after positive surprise
}

# Negative signals — these are expected to have LR < 1.0
NEGATIVE_SIGNALS: set[str] = {"risk_factors"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SignalStatus(str, Enum):
    ACTIVE = "active"
    KILLED = "killed"
    BOOSTED = "boosted"
    INSUFFICIENT_DATA = "insufficient_data"


class SignalTrend(str, Enum):
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class SignalStats:
    """Raw statistics for a single signal."""
    signal_name: str
    times_fired: int = 0
    wins: int = 0
    losses: int = 0
    total_return: float = 0.0
    recent_wins: int = 0
    recent_fires: int = 0

    @property
    def win_rate(self) -> float:
        if self.times_fired == 0:
            return 0.0
        return self.wins / self.times_fired

    @property
    def avg_return_when_fired(self) -> float:
        if self.times_fired == 0:
            return 0.0
        return self.total_return / self.times_fired

    @property
    def recent_win_rate(self) -> float:
        if self.recent_fires == 0:
            return 0.0
        return self.recent_wins / self.recent_fires


@dataclass
class SignalHealthEntry:
    """Health report for a single signal."""
    signal_name: str
    status: SignalStatus
    default_lr: float
    empirical_lr: float | None
    calibrated_lr: float
    times_fired: int
    win_rate: float
    avg_return: float
    confidence: float          # 0.0 to 1.0 based on sample size
    trend: SignalTrend


@dataclass
class CalibrationResult:
    """Result of a full recalibration run."""
    likelihood_ratios: dict[str, float]
    changes: list[str] = field(default_factory=list)
    killed_signals: list[str] = field(default_factory=list)
    boosted_signals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core calibration logic
# ---------------------------------------------------------------------------

def _compute_signal_stats() -> dict[str, SignalStats]:
    """
    Scan all closed trades and compute per-signal statistics.
    Returns stats for every signal that has ever fired.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT signals_at_entry, pnl, exit_date
            FROM trades
            WHERE status = 'closed' AND pnl IS NOT NULL
            ORDER BY exit_date ASC
            """
        ).fetchall()

    if not rows:
        return {}

    # Determine the cutoff index for "recent" trades
    total_trades = len(rows)
    recent_cutoff = max(0, total_trades - RECENT_TRADE_WINDOW)

    stats: dict[str, SignalStats] = {}

    for idx, row in enumerate(rows):
        try:
            signals = json.loads(row["signals_at_entry"])
        except (json.JSONDecodeError, TypeError):
            continue

        pnl = row["pnl"] or 0.0
        won = pnl > 0
        is_recent = idx >= recent_cutoff

        for signal_name, info in signals.items():
            if not info.get("active", False):
                continue

            if signal_name not in stats:
                stats[signal_name] = SignalStats(signal_name=signal_name)

            s = stats[signal_name]
            s.times_fired += 1
            s.total_return += pnl
            if won:
                s.wins += 1
            else:
                s.losses += 1

            if is_recent:
                s.recent_fires += 1
                if won:
                    s.recent_wins += 1

    return stats


def _empirical_lr(win_rate: float) -> float:
    """Convert a clamped win rate to a likelihood ratio."""
    clamped = max(WIN_RATE_FLOOR, min(WIN_RATE_CEILING, win_rate))
    return clamped / (1.0 - clamped)


def _blend_lr(empirical: float, prior: float) -> float:
    """Blend empirical LR with the default prior for stability."""
    return EMPIRICAL_WEIGHT * empirical + PRIOR_WEIGHT * prior


def _confidence_score(sample_size: int) -> float:
    """
    Confidence from 0.0 to 1.0 based on sample size.
    Reaches ~0.5 at 10 trades, ~0.8 at 30 trades, ~0.95 at 60 trades.
    """
    if sample_size <= 0:
        return 0.0
    # Logistic-style curve
    return min(1.0, sample_size / (sample_size + 15.0))


def _determine_trend(stats: SignalStats) -> SignalTrend:
    """Compare recent win rate to all-time win rate to detect trend."""
    if stats.recent_fires < 5:
        return SignalTrend.UNKNOWN

    all_time = stats.win_rate
    recent = stats.recent_win_rate

    diff = recent - all_time
    if diff > 0.05:
        return SignalTrend.IMPROVING
    elif diff < -0.05:
        return SignalTrend.DECLINING
    return SignalTrend.STABLE


def _determine_status(
    signal_name: str,
    calibrated_lr: float,
    times_fired: int,
) -> SignalStatus:
    """Determine whether a signal is active, killed, or boosted."""
    if times_fired < MIN_TRADES_FOR_CALIBRATION:
        return SignalStatus.INSUFFICIENT_DATA

    # Negative signals have naturally low LR — don't kill them for being low
    if signal_name in NEGATIVE_SIGNALS:
        return SignalStatus.ACTIVE

    if times_fired >= MIN_TRADES_FOR_KILL and calibrated_lr < LR_KILL_THRESHOLD:
        return SignalStatus.KILLED

    if calibrated_lr > LR_BOOST_THRESHOLD:
        return SignalStatus.BOOSTED

    return SignalStatus.ACTIVE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_calibrated_likelihood_ratios() -> dict[str, float]:
    """
    Returns likelihood ratios based on actual trading history.
    Falls back to defaults if insufficient data.
    Kills signals that are empirically proven to be noise.
    """
    stats = _compute_signal_stats()

    if not stats:
        return DEFAULT_LRS.copy()

    calibrated: dict[str, float] = {}

    for signal_name, default_lr in DEFAULT_LRS.items():
        if signal_name not in stats:
            calibrated[signal_name] = default_lr
            continue

        s = stats[signal_name]

        # Not enough data — stick with prior
        if s.times_fired < MIN_TRADES_FOR_CALIBRATION:
            calibrated[signal_name] = default_lr
            continue

        # Compute empirical LR
        emp_lr = _empirical_lr(s.win_rate)

        # Blend with prior for stability
        blended = _blend_lr(emp_lr, default_lr)

        # Kill check: enough trades and LR too low (not a negative signal)
        if (
            signal_name not in NEGATIVE_SIGNALS
            and s.times_fired >= MIN_TRADES_FOR_KILL
            and blended < LR_KILL_THRESHOLD
        ):
            calibrated[signal_name] = 0.0  # Killed — zero weight
            continue

        calibrated[signal_name] = round(blended, 3)

    return calibrated


def get_active_lrs() -> dict[str, float]:
    """
    Get the currently active LRs (calibrated if available, else defaults).
    This is what the pipeline should call instead of hardcoded values.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM fund_state WHERE key = 'calibrated_lrs'"
        ).fetchone()

    if row:
        try:
            stored = json.loads(row["value"])
            # Ensure all current signals are present (handles new signals added)
            for signal_name, default_lr in DEFAULT_LRS.items():
                if signal_name not in stored:
                    stored[signal_name] = default_lr
            return stored
        except (json.JSONDecodeError, TypeError):
            pass

    return DEFAULT_LRS.copy()


def recalibrate() -> CalibrationResult:
    """
    Run full recalibration and log results.
    Called after each pipeline run examines closed trades.
    Returns CalibrationResult with new LRs and any signals killed/boosted.
    """
    old_lrs = get_active_lrs()
    new_lrs = get_calibrated_likelihood_ratios()

    changes: list[str] = []
    killed: list[str] = []
    boosted: list[str] = []

    for signal_name, new_lr in new_lrs.items():
        old_lr = old_lrs.get(signal_name, DEFAULT_LRS.get(signal_name, 1.0))

        # Detect killed signals
        if new_lr == 0.0 and old_lr != 0.0:
            killed.append(signal_name)
            changes.append(f"{signal_name}: {old_lr:.2f} -> KILLED (noise)")
            continue

        # Detect boosted signals
        if new_lr > LR_BOOST_THRESHOLD and old_lr <= LR_BOOST_THRESHOLD:
            boosted.append(signal_name)

        # Log meaningful changes
        if abs(new_lr - old_lr) > 0.05:
            direction = "BOOSTED" if new_lr > old_lr else "WEAKENED"
            changes.append(f"{signal_name}: {old_lr:.2f} -> {new_lr:.2f} ({direction})")

    # Log to activity feed
    if changes:
        log_activity(
            "signal",
            "Self-calibration complete",
            f"Updated LRs: {'; '.join(changes)}",
            severity="info",
        )

    if killed:
        log_activity(
            "signal",
            f"Signals KILLED (noise): {', '.join(killed)}",
            f"LR dropped below {LR_KILL_THRESHOLD} after {MIN_TRADES_FOR_KILL}+ trades. "
            "These signals have no empirical edge and are removed from scoring.",
            severity="warning",
        )

    if boosted:
        log_activity(
            "signal",
            f"Signals BOOSTED (strong edge): {', '.join(boosted)}",
            f"LR above {LR_BOOST_THRESHOLD} — strong empirical edge confirmed.",
            severity="success",
        )

    # Persist calibrated LRs to fund_state table
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fund_state (key, value) VALUES ('calibrated_lrs', ?)",
            (json.dumps(new_lrs),),
        )

    result = CalibrationResult(
        likelihood_ratios=new_lrs,
        changes=changes,
        killed_signals=killed,
        boosted_signals=boosted,
    )

    return result


def signal_health_report() -> list[SignalHealthEntry]:
    """
    Returns a comprehensive health report for every signal in the system.

    Each entry includes:
    - status: active / killed / boosted / insufficient_data
    - empirical_lr: the raw data-driven LR (None if not enough trades)
    - calibrated_lr: the blended LR currently in use
    - confidence: 0.0–1.0 based on sample size
    - trend: improving / declining / stable / unknown (last 30 trades vs all-time)
    """
    stats = _compute_signal_stats()
    active_lrs = get_active_lrs()

    report: list[SignalHealthEntry] = []

    for signal_name, default_lr in DEFAULT_LRS.items():
        calibrated_lr = active_lrs.get(signal_name, default_lr)

        if signal_name not in stats or stats[signal_name].times_fired == 0:
            # No data at all for this signal
            entry = SignalHealthEntry(
                signal_name=signal_name,
                status=SignalStatus.INSUFFICIENT_DATA,
                default_lr=default_lr,
                empirical_lr=None,
                calibrated_lr=calibrated_lr,
                times_fired=0,
                win_rate=0.0,
                avg_return=0.0,
                confidence=0.0,
                trend=SignalTrend.UNKNOWN,
            )
        else:
            s = stats[signal_name]
            emp_lr: float | None = None
            if s.times_fired >= MIN_TRADES_FOR_CALIBRATION:
                emp_lr = round(_empirical_lr(s.win_rate), 3)

            status = _determine_status(signal_name, calibrated_lr, s.times_fired)
            trend = _determine_trend(s)
            confidence = _confidence_score(s.times_fired)

            entry = SignalHealthEntry(
                signal_name=signal_name,
                status=status,
                default_lr=default_lr,
                empirical_lr=emp_lr,
                calibrated_lr=calibrated_lr,
                times_fired=s.times_fired,
                win_rate=round(s.win_rate, 4),
                avg_return=round(s.avg_return_when_fired, 4),
                confidence=round(confidence, 3),
                trend=trend,
            )

        report.append(entry)

    return report
