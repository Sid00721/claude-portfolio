"""
ASX Quant System — Main Orchestrator

Runs the full pipeline:
Universe Filter → Regime Detection → Signal Stack →
Bayesian Aggregator → Kelly Sizer → Execution → Attribution
"""

import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from data.universe import build_universe, save_to_db
from data.activity import log_activity
from signals.regime import get_regime
from signals.momentum import compute_momentum_signal
from signals.insider import get_insider_signal
from signals.bayesian import aggregate, SignalInput
from signals.calibration import get_active_lrs, recalibrate
from execution.kelly import size_position
from execution.broker import get_broker, Order, Side, OrderType
from data.attribution import log_trade, log_signal_state, TradeRecord


def run_daily_pipeline():
    try:
        _run_pipeline_inner()
    except Exception as e:
        log_activity("error", "Pipeline crashed", str(e)[:200], severity="alert")


def _run_pipeline_inner():
    log_activity("scan", "Pipeline started", "Beginning daily scan cycle", severity="info")

    # Layer 1: Universe — try fresh scan, fall back to cached
    log_activity("scan", "Scanning ASX universe", "Filtering $100M-$1B market cap, >50k volume")
    try:
        universe = build_universe()
        if not universe.empty:
            save_to_db(universe)
            log_activity("scan", f"Universe built: {len(universe)} stocks",
                         f"Top movers: {', '.join(universe['ticker'].head(5).tolist())}",
                         severity="success")
    except Exception as e:
        log_activity("scan", "Universe scan failed, using cache", str(e)[:100], severity="warning")
        universe = None

    # Fall back to cached universe from DB
    if universe is None or universe.empty:
        import pandas as pd
        from data.db import get_db as _get_db
        with _get_db() as conn:
            rows = conn.execute("SELECT * FROM universe ORDER BY return_12m DESC").fetchall()
        if rows:
            universe = pd.DataFrame([dict(r) for r in rows])
            log_activity("scan", f"Using cached universe: {len(universe)} stocks", severity="info")
        else:
            log_activity("scan", "No universe data available", "Need at least one successful scan first", severity="alert")
            return

    time.sleep(5)

    # Layer 2: Regime
    log_activity("regime", "Detecting market regime", "Checking VIX, yield curve, sector flows")
    regime = get_regime()
    log_activity("regime",
                 f"Regime: {regime.vix_regime.upper()} (VIX {regime.vix_level:.1f})",
                 f"Yield spread: {regime.yield_spread:.2f}% ({regime.yield_signal}). Position scalar: {regime.position_scalar:.0%}",
                 severity="success" if regime.overall_regime == "risk_on" else "warning")

    if regime.overall_regime == "risk_off":
        log_activity("regime", "RISK OFF — halting trades",
                     "VIX > 28 or yield curve inverted. Sitting in cash.",
                     severity="alert")
        return

    # Layer 3: Signals
    lrs = get_active_lrs()
    log_activity("signal", "Running signal stack",
                 f"Using calibrated LRs: mom={lrs.get('momentum', 1.4):.2f}, ins={lrs.get('insider', 1.8):.2f}")
    tickers = universe["ticker"].tolist()

    momentum_df = compute_momentum_signal(universe)

    insider_signals = {}
    for ticker in tickers:
        try:
            insider_signals[ticker] = get_insider_signal(ticker)
        except Exception:
            pass

    mom_top = len(momentum_df[momentum_df["momentum_quintile"] == 5]) if not momentum_df.empty else 0
    ins_active = sum(1 for s in insider_signals.values() if s.signal_fires)

    log_activity("signal", f"Momentum: {mom_top} stocks in top quintile",
                 f"Top momentum: {', '.join(momentum_df[momentum_df['momentum_quintile'] == 5]['ticker'].head(5).tolist()) if mom_top > 0 else 'none'}")

    if ins_active > 0:
        insider_tickers = [t for t, s in insider_signals.items() if s.signal_fires]
        log_activity("signal", f"Insider buying detected: {ins_active} stocks",
                     f"Insider activity in: {', '.join(insider_tickers[:5])}",
                     severity="success")

    # Layer 4: Bayesian Aggregation
    log_activity("signal", "Bayesian aggregation", "Combining signals, computing posteriors")
    candidates = []

    for ticker in tickers:
        signals = []

        if not momentum_df.empty:
            mom_row = momentum_df[momentum_df["ticker"] == ticker]
            if not mom_row.empty:
                mom_active = mom_row.iloc[0]["momentum_quintile"] == 5
                signals.append(SignalInput(
                    signal_name="momentum",
                    signal_active=mom_active,
                    signal_strength=float(mom_row.iloc[0]["signal_strength"]),
                    likelihood_ratio=lrs.get("momentum", 1.4) if mom_active else 1.0,
                ))

        if ticker in insider_signals:
            ins = insider_signals[ticker]
            signals.append(SignalInput(
                signal_name="insider",
                signal_active=ins.signal_fires,
                signal_strength=ins.signal_strength,
                likelihood_ratio=lrs.get("insider", 1.8) if ins.signal_fires else 1.0,
            ))

        if not signals:
            continue

        result = aggregate(ticker, signals)

        log_signal_state(ticker, datetime.now().date().isoformat(), {
            s.signal_name: {"active": s.signal_active, "strength": s.signal_strength}
            for s in signals
        })

        if result.should_trade:
            candidates.append((result, signals))
            log_activity("signal",
                         f"HIGH CONVICTION: {ticker} ({result.posterior:.0%})",
                         f"Signals: {', '.join(s.signal_name for s in signals if s.signal_active)}. Conviction: {result.conviction_level}",
                         ticker=ticker,
                         severity="success")

    if not candidates:
        log_activity("signal", "No high-conviction setups today",
                     f"Scanned {len(tickers)} stocks, none passed 75% threshold",
                     severity="info")
        return

    candidates.sort(key=lambda x: x[0].posterior, reverse=True)

    # Layer 5: Kelly Sizing
    log_activity("trade", "Sizing positions", f"{len(candidates)} candidates, applying half-Kelly")
    broker = get_broker()
    broker.connect()
    portfolio_value = broker.get_portfolio_value()

    positions_to_take = []
    for result, signals in candidates[:10]:
        price_row = universe[universe["ticker"] == result.ticker]
        if price_row.empty:
            continue
        price = float(price_row.iloc[0]["price"])

        adjusted_portfolio = portfolio_value * regime.position_scalar

        position = size_position(
            ticker=result.ticker,
            posterior_probability=result.posterior,
            portfolio_value=adjusted_portfolio,
            price=price,
        )

        if position is not None:
            positions_to_take.append((result, signals, position))

    # Layer 6: Execution
    for result, signals, position in positions_to_take:
        try:
            order = Order(
                ticker=result.ticker,
                side=Side.BUY,
                quantity=position.shares,
                order_type=OrderType.MARKET,
            )
            order_id = broker.place_order(order)

            log_activity("trade",
                         f"BOUGHT {result.ticker} x{position.shares}",
                         f"${position.position_dollars:.2f} at ${position.position_dollars/position.shares:.4f}/share. "
                         f"Posterior: {result.posterior:.0%}, Kelly: {position.fractional_kelly_pct:.1%}",
                         ticker=result.ticker,
                         severity="success")

            price_row = universe[universe["ticker"] == result.ticker]
            entry_price = float(price_row.iloc[0]["price"])

            record = TradeRecord(
                ticker=result.ticker,
                entry_date=datetime.now().isoformat(),
                exit_date=None,
                entry_price=entry_price,
                exit_price=None,
                position_size=position.position_dollars,
                pnl=None,
                return_pct=None,
                signals_at_entry={
                    s.signal_name: {
                        "active": s.signal_active,
                        "strength": s.signal_strength,
                        "likelihood_ratio": s.likelihood_ratio,
                    }
                    for s in signals
                },
                posterior_at_entry=result.posterior,
                regime_at_entry=regime.overall_regime,
                kelly_size_pct=position.fractional_kelly_pct,
            )
            log_trade(record)
        except Exception as e:
            log_activity("error", f"Trade failed: {result.ticker}", str(e),
                         ticker=result.ticker, severity="alert")

    broker.disconnect()

    # Layer 7: Attribution + Self-Learning
    calibration = recalibrate()
    if calibration["changes"]:
        log_activity("scan", "Self-learning update",
                     f"Recalibrated {len(calibration['changes'])} signal weights from trade history",
                     severity="info")

    log_activity("scan", "Pipeline complete",
                 f"{len(positions_to_take)} new positions opened. Next run in 24h.",
                 severity="success")


if __name__ == "__main__":
    run_daily_pipeline()
