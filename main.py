"""
ASX Small-Cap Quant System — Main Pipeline Orchestrator

Runs the full pipeline:
Universe Load → Regime Detection → Signal Stack (5 signals) →
Bayesian Aggregator → Kelly Sizer → Execution → Recalibration → Attribution

All stages are logged to the activity feed for live dashboard visibility.
"""

import time
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from data.db import get_db
from data.universe import build_universe, save_to_db
from data.activity import log_activity
from data.attribution import log_trade, log_signal_state, TradeRecord

from signals.regime import get_regime
from signals.momentum import compute_momentum_signal
from signals.insider import get_insider_signals
from signals.value import compute_value_signal, compute_franking_signal
from signals.index_rebalance import compute_rebalance_signal
from signals.calibration import get_active_lrs, recalibrate
from signals.bayesian import aggregate, SignalInput

from execution.kelly import size_position
from execution.broker import get_broker, Order, Side, OrderType


# ---------------------------------------------------------------------------
# Default likelihood ratios (fallback if calibration has no data yet)
# ---------------------------------------------------------------------------
DEFAULT_LRS = {
    "momentum": 1.5,
    "insider": 1.8,
    "value": 1.35,
    "franking": 1.25,
    "index_rebalance": 1.3,
}


def run_daily_pipeline():
    """Top-level entry point. Wraps everything so crashes are always logged."""
    try:
        _run_pipeline_inner()
    except Exception as e:
        log_activity(
            "error",
            "Pipeline crashed",
            f"{type(e).__name__}: {str(e)[:300]}",
            ticker=None,
            severity="alert",
        )


# ---------------------------------------------------------------------------
# Layer 1: Universe Loading
# ---------------------------------------------------------------------------

def _load_universe() -> pd.DataFrame | None:
    """Load universe from DB cache, seed file, or (last resort) live scan."""

    # Attempt 1: DB cache (instant)
    try:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM universe ORDER BY return_12m DESC").fetchall()
        if rows:
            universe = pd.DataFrame([dict(r) for r in rows])
            log_activity(
                "scan",
                f"Universe loaded: {len(universe)} stocks from DB cache",
                f"Top movers: {', '.join(universe['ticker'].head(3).tolist())}",
                ticker=None,
                severity="success",
            )
            return universe
    except Exception as e:
        log_activity("error", "DB cache read failed", str(e)[:200], ticker=None, severity="warning")

    # Attempt 2: Seed file
    seed_path = os.path.join(os.path.dirname(__file__), "data", "universe_seed.json")
    if os.path.exists(seed_path):
        try:
            with open(seed_path) as f:
                seed_data = json.load(f)
            universe = pd.DataFrame(seed_data)
            save_to_db(universe)
            log_activity(
                "scan",
                f"Universe seeded: {len(universe)} stocks from bundled data",
                f"Loaded from {seed_path}",
                ticker=None,
                severity="success",
            )
            return universe
        except Exception as e:
            log_activity("error", "Seed file load failed", str(e)[:200], ticker=None, severity="warning")

    # Attempt 3: Live scan (last resort)
    log_activity("scan", "No cache or seed available, running live scan", "", ticker=None, severity="warning")
    try:
        universe = build_universe()
        if universe is not None and not universe.empty:
            save_to_db(universe)
            log_activity("scan", f"Universe built: {len(universe)} stocks via live scan", "", ticker=None, severity="success")
            return universe
    except Exception as e:
        log_activity("error", "Live universe scan failed", str(e)[:200], ticker=None, severity="alert")

    return None


# ---------------------------------------------------------------------------
# Layer 3: Signal Runners (each isolated so one failure doesn't kill others)
# ---------------------------------------------------------------------------

def _run_momentum_signal(universe: pd.DataFrame) -> pd.DataFrame | None:
    """Compute momentum signal. Returns DataFrame or None on failure."""
    try:
        df = compute_momentum_signal(universe)
        if df is not None and not df.empty:
            top_quintile = len(df[df["signal_active"] == True])
            top_tickers = df[df["signal_active"] == True]["ticker"].head(5).tolist()
            log_activity(
                "signal",
                f"Momentum: {top_quintile} stocks in top quintile",
                f"Top momentum: {', '.join(top_tickers) if top_tickers else 'none'}",
                ticker=None,
                severity="info",
            )
            return df
        else:
            log_activity("signal", "Momentum signal returned empty", "", ticker=None, severity="warning")
            return None
    except Exception as e:
        log_activity("error", "Momentum signal failed", f"{type(e).__name__}: {str(e)[:200]}", ticker=None, severity="alert")
        return None


def _run_insider_signal(tickers: list[str]) -> dict:
    """Compute insider signals. Returns dict[str, InsiderSignal] or empty on failure."""
    try:
        signals = get_insider_signals(tickers)
        active_count = sum(1 for s in signals.values() if s.signal_active)
        if active_count > 0:
            insider_tickers = [t for t, s in signals.items() if s.signal_active]
            log_activity(
                "signal",
                f"Insider buying detected: {active_count} stocks",
                f"Activity in: {', '.join(insider_tickers[:8])}",
                ticker=None,
                severity="success",
            )
        else:
            log_activity("signal", "Insider: no active insider buying detected", "", ticker=None, severity="info")
        return signals
    except Exception as e:
        log_activity("error", "Insider signal failed", f"{type(e).__name__}: {str(e)[:200]}", ticker=None, severity="alert")
        return {}


def _run_value_signal(universe: pd.DataFrame) -> pd.DataFrame | None:
    """Compute value signal. Returns DataFrame or None on failure."""
    try:
        df = compute_value_signal(universe)
        if df is not None and not df.empty:
            top_value = len(df[df["signal_active"] == True])
            log_activity(
                "signal",
                f"Value: {top_value} stocks with active value signal",
                f"Top value tickers: {', '.join(df[df['signal_active'] == True]['ticker'].head(5).tolist())}",
                ticker=None,
                severity="info",
            )
            return df
        else:
            log_activity("signal", "Value signal returned empty", "", ticker=None, severity="warning")
            return None
    except Exception as e:
        log_activity("error", "Value signal failed", f"{type(e).__name__}: {str(e)[:200]}", ticker=None, severity="alert")
        return None


def _run_franking_signal(tickers: list[str]) -> dict:
    """Compute franking credit signal. Returns dict or empty on failure."""
    try:
        signals = compute_franking_signal(tickers)
        active_count = sum(1 for s in signals.values() if s.signal_active)
        if active_count > 0:
            franking_tickers = [t for t, s in signals.items() if s.signal_active]
            log_activity(
                "signal",
                f"Franking: {active_count} stocks with high franking credits",
                f"Tickers: {', '.join(franking_tickers[:8])}",
                ticker=None,
                severity="info",
            )
        else:
            log_activity("signal", "Franking: no active franking signals", "", ticker=None, severity="info")
        return signals
    except Exception as e:
        log_activity("error", "Franking signal failed", f"{type(e).__name__}: {str(e)[:200]}", ticker=None, severity="alert")
        return {}


def _run_rebalance_signal(universe: pd.DataFrame) -> dict:
    """Compute index rebalance signal. Returns dict or empty on failure."""
    try:
        signals = compute_rebalance_signal(universe)
        active_count = sum(1 for s in signals.values() if s.signal_active)
        if active_count > 0:
            rebal_tickers = [t for t, s in signals.items() if s.signal_active]
            log_activity(
                "signal",
                f"Index rebalance: {active_count} potential additions",
                f"Candidates: {', '.join(rebal_tickers[:8])}",
                ticker=None,
                severity="info",
            )
        else:
            log_activity("signal", "Index rebalance: no active signals", "", ticker=None, severity="info")
        return signals
    except Exception as e:
        log_activity("error", "Index rebalance signal failed", f"{type(e).__name__}: {str(e)[:200]}", ticker=None, severity="alert")
        return {}


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline_inner():
    start_time = time.time()
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_activity(
        "scan",
        "Pipeline started",
        f"Beginning daily scan cycle at {run_date}",
        ticker=None,
        severity="info",
    )

    # ------------------------------------------------------------------
    # Layer 1: Universe
    # ------------------------------------------------------------------
    universe = _load_universe()
    if universe is None or universe.empty:
        log_activity("scan", "No universe data available — aborting", "", ticker=None, severity="alert")
        return

    tickers = universe["ticker"].tolist()

    # ------------------------------------------------------------------
    # Layer 2: Regime Detection
    # ------------------------------------------------------------------
    log_activity("regime", "Detecting market regime", "Checking VIX, yield curve, sector flows", ticker=None, severity="info")

    try:
        regime = get_regime()
    except Exception as e:
        log_activity("error", "Regime detection failed — aborting", str(e)[:200], ticker=None, severity="alert")
        return

    regime_severity = "success" if regime.overall_regime == "risk_on" else "warning"
    log_activity(
        "regime",
        f"Regime: {regime.vix_regime.upper()} (VIX {regime.vix_level:.1f})",
        f"Yield spread: {regime.yield_spread:.2f}% ({regime.yield_signal}). "
        f"Overall: {regime.overall_regime}. Position scalar: {regime.position_scalar:.0%}",
        ticker=None,
        severity=regime_severity,
    )

    if regime.overall_regime == "risk_off":
        log_activity(
            "regime",
            "RISK OFF — halting trades",
            "VIX elevated or yield curve inverted. Sitting in cash until conditions improve.",
            ticker=None,
            severity="alert",
        )
        return

    # ------------------------------------------------------------------
    # Layer 3: Signal Stack (run all 5 signals in parallel)
    # ------------------------------------------------------------------
    log_activity("signal", "Running full signal stack", "5 signals: momentum, insider, value, franking, index_rebalance", ticker=None, severity="info")

    # Load calibrated likelihood ratios
    try:
        lrs = get_active_lrs()
    except Exception as e:
        log_activity("error", "Calibration load failed, using defaults", str(e)[:100], ticker=None, severity="warning")
        lrs = {}

    # Merge with defaults (calibrated values override)
    active_lrs = {**DEFAULT_LRS, **lrs}

    log_activity(
        "signal",
        "Likelihood ratios loaded",
        f"mom={active_lrs['momentum']:.2f}, ins={active_lrs['insider']:.2f}, "
        f"val={active_lrs['value']:.2f}, frank={active_lrs['franking']:.2f}, "
        f"rebal={active_lrs['index_rebalance']:.2f}",
        ticker=None,
        severity="info",
    )

    # Run signals in parallel
    momentum_df = None
    insider_signals = {}
    value_df = None
    franking_signals = {}
    rebalance_signals = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_run_momentum_signal, universe): "momentum",
            executor.submit(_run_insider_signal, tickers): "insider",
            executor.submit(_run_value_signal, universe): "value",
            executor.submit(_run_franking_signal, tickers): "franking",
            executor.submit(_run_rebalance_signal, universe): "index_rebalance",
        }

        for future in as_completed(futures):
            signal_name = futures[future]
            try:
                result = future.result()
                if signal_name == "momentum":
                    momentum_df = result
                elif signal_name == "insider":
                    insider_signals = result or {}
                elif signal_name == "value":
                    value_df = result
                elif signal_name == "franking":
                    franking_signals = result or {}
                elif signal_name == "index_rebalance":
                    rebalance_signals = result or {}
            except Exception as e:
                log_activity("error", f"Signal thread failed: {signal_name}", str(e)[:200], ticker=None, severity="alert")

    signals_active = sum([
        momentum_df is not None,
        len(insider_signals) > 0,
        value_df is not None,
        len(franking_signals) > 0,
        len(rebalance_signals) > 0,
    ])

    log_activity(
        "signal",
        f"Signal stack complete: {signals_active}/5 signals produced data",
        "",
        ticker=None,
        severity="success" if signals_active >= 3 else "warning",
    )

    if signals_active == 0:
        log_activity("signal", "All signals failed — aborting", "", ticker=None, severity="alert")
        return

    # ------------------------------------------------------------------
    # Layer 4: Bayesian Aggregation
    # ------------------------------------------------------------------
    log_activity("signal", "Bayesian aggregation", "Combining signals, computing posteriors for each ticker", ticker=None, severity="info")

    candidates = []

    for ticker in tickers:
        signals = []

        # Momentum
        if momentum_df is not None and not momentum_df.empty:
            mom_row = momentum_df[momentum_df["ticker"] == ticker]
            if not mom_row.empty:
                row = mom_row.iloc[0]
                is_active = bool(row["signal_active"])
                strength = float(row["signal_strength"])
                signals.append(SignalInput(
                    signal_name="momentum",
                    signal_active=is_active,
                    signal_strength=strength,
                    likelihood_ratio=active_lrs["momentum"] if is_active else 1.0,
                ))

        # Insider
        if ticker in insider_signals:
            ins = insider_signals[ticker]
            is_active = bool(ins.signal_active)
            strength = float(ins.signal_strength)
            signals.append(SignalInput(
                signal_name="insider",
                signal_active=is_active,
                signal_strength=strength,
                likelihood_ratio=active_lrs["insider"] if is_active else 1.0,
            ))

        # Value
        if value_df is not None and not value_df.empty:
            val_row = value_df[value_df["ticker"] == ticker]
            if not val_row.empty:
                row = val_row.iloc[0]
                is_active = bool(row["signal_active"])
                strength = float(row["signal_strength"])
                signals.append(SignalInput(
                    signal_name="value",
                    signal_active=is_active,
                    signal_strength=strength,
                    likelihood_ratio=active_lrs["value"] if is_active else 1.0,
                ))

        # Franking
        if ticker in franking_signals:
            frank = franking_signals[ticker]
            is_active = bool(frank.signal_active)
            strength = float(frank.signal_strength)
            signals.append(SignalInput(
                signal_name="franking",
                signal_active=is_active,
                signal_strength=strength,
                likelihood_ratio=active_lrs["franking"] if is_active else 1.0,
            ))

        # Index rebalance
        if ticker in rebalance_signals:
            rebal = rebalance_signals[ticker]
            is_active = bool(rebal.signal_active)
            strength = float(rebal.signal_strength)
            signals.append(SignalInput(
                signal_name="index_rebalance",
                signal_active=is_active,
                signal_strength=strength,
                likelihood_ratio=active_lrs["index_rebalance"] if is_active else 1.0,
            ))

        # Skip tickers with no signal data at all
        if not signals:
            continue

        # Aggregate
        try:
            result = aggregate(ticker, signals)
        except Exception as e:
            log_activity("error", f"Bayesian aggregation failed for {ticker}", str(e)[:150], ticker=ticker, severity="warning")
            continue

        # Log signal state for attribution tracking
        try:
            log_signal_state(ticker, datetime.now().date().isoformat(), {
                s.signal_name: {
                    "active": bool(s.signal_active),
                    "strength": float(s.signal_strength),
                    "lr": float(s.likelihood_ratio),
                }
                for s in signals
            })
        except Exception:
            pass  # Non-critical

        if result.should_trade:
            candidates.append((result, signals))
            active_signal_names = [s.signal_name for s in signals if s.signal_active]
            log_activity(
                "signal",
                f"HIGH CONVICTION: {ticker} ({result.posterior:.0%})",
                f"Active signals: {', '.join(active_signal_names)}. Conviction: {result.conviction_level}",
                ticker=ticker,
                severity="success",
            )

    if not candidates:
        log_activity(
            "signal",
            "No high-conviction setups today",
            f"Scanned {len(tickers)} stocks across 5 signals, none passed threshold",
            ticker=None,
            severity="info",
        )
        _run_recalibration()
        elapsed = time.time() - start_time
        log_activity("scan", "Pipeline complete (no trades)", f"Elapsed: {elapsed:.1f}s", ticker=None, severity="info")
        return

    # Sort by posterior probability (best first)
    candidates.sort(key=lambda x: x[0].posterior, reverse=True)
    log_activity(
        "trade",
        f"{len(candidates)} candidates passed Bayesian filter",
        f"Best: {candidates[0][0].ticker} at {candidates[0][0].posterior:.0%}",
        ticker=None,
        severity="success",
    )

    # ------------------------------------------------------------------
    # Layer 5: Kelly Sizing
    # ------------------------------------------------------------------
    log_activity("trade", "Sizing positions", f"{len(candidates)} candidates, applying Kelly criterion", ticker=None, severity="info")

    broker = None
    try:
        broker = get_broker()
        broker.connect()
        portfolio_value = broker.get_portfolio_value()
    except Exception as e:
        log_activity("error", "Broker connection failed — aborting execution", str(e)[:200], ticker=None, severity="alert")
        _run_recalibration()
        return

    adjusted_portfolio = portfolio_value * regime.position_scalar

    positions_to_take = []
    for result, signals in candidates[:10]:  # Cap at 10 positions
        try:
            price_row = universe[universe["ticker"] == result.ticker]
            if price_row.empty:
                continue
            price = float(price_row.iloc[0]["price"])

            position = size_position(
                ticker=result.ticker,
                posterior_probability=result.posterior,
                portfolio_value=adjusted_portfolio,
                price=price,
            )

            if position is not None:
                positions_to_take.append((result, signals, position))
                log_activity(
                    "trade",
                    f"Sized: {result.ticker} — ${position.position_dollars:,.0f} ({position.fractional_kelly_pct:.1%} Kelly)",
                    f"Shares: {position.shares}, Price: ${price:.4f}",
                    ticker=result.ticker,
                    severity="info",
                )
        except Exception as e:
            log_activity("error", f"Kelly sizing failed for {result.ticker}", str(e)[:150], ticker=result.ticker, severity="warning")

    if not positions_to_take:
        log_activity("trade", "No positions passed Kelly sizing", "All candidates too small or rejected by sizer", ticker=None, severity="info")
        if broker:
            broker.disconnect()
        _run_recalibration()
        elapsed = time.time() - start_time
        log_activity("scan", "Pipeline complete (no trades)", f"Elapsed: {elapsed:.1f}s", ticker=None, severity="info")
        return

    # ------------------------------------------------------------------
    # Layer 6: Execution
    # ------------------------------------------------------------------
    log_activity("trade", f"Executing {len(positions_to_take)} orders", "Sending to broker", ticker=None, severity="info")

    trades_executed = 0
    for result, signals, position in positions_to_take:
        try:
            order = Order(
                ticker=result.ticker,
                side=Side.BUY,
                quantity=position.shares,
                order_type=OrderType.MARKET,
            )
            broker.place_order(order)
            trades_executed += 1

            entry_price = float(universe[universe["ticker"] == result.ticker].iloc[0]["price"])

            log_activity(
                "trade",
                f"BOUGHT {result.ticker} x{position.shares}",
                f"${position.position_dollars:,.2f} at ${entry_price:.4f}/share. "
                f"Posterior: {result.posterior:.0%}, Kelly: {position.fractional_kelly_pct:.1%}",
                ticker=result.ticker,
                severity="success",
            )

            # Calculate stop price: entry minus 2x ATR
            stop_price = entry_price - position.risk_per_share if position.risk_per_share > 0 else None

            # Attribution record
            record = TradeRecord(
                ticker=result.ticker,
                entry_date=datetime.now().isoformat(),
                exit_date=None,
                entry_price=entry_price,
                exit_price=None,
                position_size=float(position.position_dollars),
                shares=position.shares,
                pnl=None,
                return_pct=None,
                signals_at_entry={
                    s.signal_name: {
                        "active": bool(s.signal_active),
                        "strength": float(s.signal_strength),
                        "likelihood_ratio": float(s.likelihood_ratio),
                    }
                    for s in signals
                },
                posterior_at_entry=float(result.posterior),
                regime_at_entry=regime.overall_regime,
                kelly_size_pct=float(position.fractional_kelly_pct),
                stop_price=stop_price,
            )
            log_trade(record)

        except Exception as e:
            log_activity(
                "error",
                f"Trade execution failed: {result.ticker}",
                f"{type(e).__name__}: {str(e)[:200]}",
                ticker=result.ticker,
                severity="alert",
            )

    try:
        broker.disconnect()
    except Exception:
        pass

    log_activity(
        "trade",
        f"Execution complete: {trades_executed}/{len(positions_to_take)} orders filled",
        "",
        ticker=None,
        severity="success" if trades_executed == len(positions_to_take) else "warning",
    )

    # ------------------------------------------------------------------
    # Layer 7: Recalibration (Self-Learning)
    # ------------------------------------------------------------------
    _run_recalibration()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    log_activity(
        "scan",
        "Pipeline complete",
        f"{trades_executed} new positions opened across {signals_active} active signals. "
        f"Elapsed: {elapsed:.1f}s. Next run in 24h.",
        ticker=None,
        severity="success",
    )


def _run_recalibration():
    """Run signal weight recalibration from trade history."""
    try:
        calibration = recalibrate()
        if calibration.changes:
            changes_summary = "; ".join(calibration.changes[:5])
            log_activity(
                "scan",
                "Self-learning update",
                f"Recalibrated signal weights: {changes_summary}",
                ticker=None,
                severity="info",
            )
        else:
            log_activity("scan", "Recalibration: no weight changes needed", "", ticker=None, severity="info")
    except Exception as e:
        log_activity("error", "Recalibration failed", f"{type(e).__name__}: {str(e)[:150]}", ticker=None, severity="warning")


if __name__ == "__main__":
    run_daily_pipeline()
