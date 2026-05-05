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
from signals.regime import get_regime
from signals.momentum import compute_momentum_signal
from signals.insider import get_insider_signal
from signals.bayesian import aggregate, SignalInput
from execution.kelly import size_position
from execution.broker import get_broker, Order, Side, OrderType
from data.attribution import log_trade, log_signal_state, TradeRecord


def run_daily_pipeline():
    print(f"\n{'='*60}")
    print(f"ASX QUANT SYSTEM — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # Layer 1: Universe
    print("[1/7] Building universe...")
    universe = build_universe()
    print(f"      {len(universe)} stocks in universe")

    if universe.empty:
        print("      No stocks passed filters. Exiting.")
        return

    save_to_db(universe)

    time.sleep(5)  # avoid yfinance rate limit after universe scan

    # Layer 2: Regime
    print("[2/7] Detecting regime...")
    regime = get_regime()
    print(f"      VIX: {regime.vix_level:.1f} → {regime.vix_regime}")
    print(f"      Yield spread: {regime.yield_spread:.2f}% → {regime.yield_signal}")
    print(f"      Position scalar: {regime.position_scalar:.2f}")

    if regime.overall_regime == "risk_off":
        print("      RISK OFF — no new positions. Exiting.")
        return

    # Layer 3: Signals
    print("[3/7] Running signal stack...")
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
    print(f"      Momentum: {mom_top} in top quintile")
    print(f"      Insider: {ins_active} active signals")

    # Layer 4: Bayesian Aggregation
    print("[4/7] Bayesian aggregation...")
    candidates = []

    for ticker in tickers:
        signals = []

        # Momentum signal
        if not momentum_df.empty:
            mom_row = momentum_df[momentum_df["ticker"] == ticker]
            if not mom_row.empty:
                mom_active = mom_row.iloc[0]["momentum_quintile"] == 5
                signals.append(SignalInput(
                    signal_name="momentum",
                    signal_active=mom_active,
                    signal_strength=float(mom_row.iloc[0]["signal_strength"]),
                    likelihood_ratio=1.4 if mom_active else 1.0,
                ))

        # Insider signal
        if ticker in insider_signals:
            ins = insider_signals[ticker]
            signals.append(SignalInput(
                signal_name="insider",
                signal_active=ins.signal_fires,
                signal_strength=ins.signal_strength,
                likelihood_ratio=1.8 if ins.signal_fires else 1.0,
            ))

        if not signals:
            continue

        result = aggregate(ticker, signals)

        log_signal_state(ticker, datetime.utcnow().date().isoformat(), {
            s.signal_name: {"active": s.signal_active, "strength": s.signal_strength}
            for s in signals
        })

        if result.should_trade:
            candidates.append((result, signals))

    print(f"      {len(candidates)} candidates above 75% posterior")

    if not candidates:
        print("      No high-conviction setups today.")
        return

    candidates.sort(key=lambda x: x[0].posterior, reverse=True)

    # Layer 5: Kelly Sizing
    print("[5/7] Position sizing...")
    broker = get_broker()
    broker.connect()
    portfolio_value = broker.get_portfolio_value()

    positions_to_take = []
    for result, signals in candidates[:10]:
        price_row = universe[universe["ticker"] == result.ticker]
        if price_row.empty:
            continue
        price = float(price_row.iloc[0]["price"])

        # Apply regime scalar to portfolio value for sizing
        adjusted_portfolio = portfolio_value * regime.position_scalar

        position = size_position(
            ticker=result.ticker,
            posterior_probability=result.posterior,
            portfolio_value=adjusted_portfolio,
            price=price,
        )

        if position is not None:
            positions_to_take.append((result, signals, position))
            print(f"      {result.ticker}: {result.posterior:.1%} posterior → "
                  f"${position.position_dollars:,.0f} ({position.fractional_kelly_pct:.1%})")

    # Layer 6: Execution
    print("[6/7] Executing trades...")
    for result, signals, position in positions_to_take:
        try:
            order = Order(
                ticker=result.ticker,
                side=Side.BUY,
                quantity=position.shares,
                order_type=OrderType.MARKET,
            )
            order_id = broker.place_order(order)
            print(f"      ORDER PLACED: {result.ticker} x{position.shares} (id: {order_id})")

            price_row = universe[universe["ticker"] == result.ticker]
            entry_price = float(price_row.iloc[0]["price"])

            record = TradeRecord(
                ticker=result.ticker,
                entry_date=datetime.utcnow().isoformat(),
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
            print(f"      FAILED: {result.ticker} — {e}")

    broker.disconnect()

    # Layer 7: Attribution
    print("[7/7] Attribution logged.")
    print(f"\n{'='*60}")
    print(f"Pipeline complete. {len(positions_to_take)} new positions.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_daily_pipeline()
