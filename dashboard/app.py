"""
ASX Quant Dashboard

Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from data.db import get_db
from signals.bayesian import aggregate, SignalInput, signal_attribution
from execution.kelly import kelly_fraction_calc

st.set_page_config(page_title="ASX Quant System", layout="wide")


def load_universe():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM universe ORDER BY return_12m DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_trades():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM trades ORDER BY entry_date DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_signal_states():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM signal_states ORDER BY date DESC LIMIT 500").fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def load_portfolio_snapshots():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM portfolio_snapshots ORDER BY date").fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ASX Quant System")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Universe",
    "Signals",
    "Trades",
    "Backtest",
    "Attribution",
])

# ─── Overview Page ─────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("System Overview")

    universe = load_universe()
    trades = load_trades()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Universe Size", len(universe))
    col2.metric("Open Trades", len(trades[trades["status"] == "open"]) if not trades.empty else 0)
    col3.metric("Closed Trades", len(trades[trades["status"] == "closed"]) if not trades.empty else 0)

    if not trades.empty and "pnl" in trades.columns:
        total_pnl = trades["pnl"].dropna().sum()
        col4.metric("Total P&L", f"${total_pnl:,.2f}")
    else:
        col4.metric("Total P&L", "$0.00")

    st.divider()

    # Portfolio equity curve
    snapshots = load_portfolio_snapshots()
    if not snapshots.empty:
        st.subheader("Equity Curve")
        fig = px.line(snapshots, x="date", y="nav", title="Portfolio NAV")
        fig.update_layout(yaxis_title="NAV ($)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No portfolio snapshots yet. Run the pipeline to generate data.")

    # Universe sector breakdown
    if not universe.empty:
        st.subheader("Universe by Sector")
        sector_counts = universe["sector"].value_counts().reset_index()
        sector_counts.columns = ["sector", "count"]
        fig = px.pie(sector_counts, values="count", names="sector", hole=0.4)
        st.plotly_chart(fig, width="stretch")


# ─── Universe Page ─────────────────────────────────────────────────────────────
elif page == "Universe":
    st.title("Stock Universe")

    universe = load_universe()
    if universe.empty:
        st.warning("Universe is empty. Run `python main.py` to populate.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        sectors = ["All"] + sorted(universe["sector"].unique().tolist())
        selected_sector = col1.selectbox("Sector", sectors)

        min_ret = col2.number_input("Min 12m Return %", value=-100.0)
        max_ret = col3.number_input("Max 12m Return %", value=10000.0)

        filtered = universe.copy()
        if selected_sector != "All":
            filtered = filtered[filtered["sector"] == selected_sector]
        filtered = filtered[
            (filtered["return_12m"] >= min_ret / 100) &
            (filtered["return_12m"] <= max_ret / 100)
        ]

        st.metric("Stocks Displayed", len(filtered))

        # Return distribution
        fig = px.histogram(filtered, x="return_12m", nbins=30, title="12-Month Return Distribution")
        fig.update_layout(xaxis_title="12m Return", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

        # Table
        display_df = filtered[["ticker", "market_cap", "sector", "return_12m", "price", "avg_volume"]].copy()
        display_df["market_cap"] = display_df["market_cap"].apply(lambda x: f"${x/1e6:.0f}M")
        display_df["return_12m"] = display_df["return_12m"].apply(lambda x: f"{x:.1%}")
        display_df["avg_volume"] = display_df["avg_volume"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, width="stretch", hide_index=True)


# ─── Signals Page ──────────────────────────────────────────────────────────────
elif page == "Signals":
    st.title("Signal Scanner")

    universe = load_universe()
    if universe.empty:
        st.warning("No universe data. Run the pipeline first.")
    else:
        st.subheader("Bayesian Signal Simulator")
        st.caption("Adjust signals to see how posterior probability changes")

        col1, col2 = st.columns([1, 2])

        with col1:
            momentum_on = st.checkbox("Momentum (Top Quintile)", value=True)
            insider_on = st.checkbox("Insider Cluster Buying", value=False)
            insider_strength = st.slider("Insider Strength", 0.0, 1.0, 0.8, key="ins_str") if insider_on else 0.0
            sentiment_on = st.checkbox("Sentiment Delta Positive", value=False)
            sentiment_strength = st.slider("Sentiment Strength", 0.0, 1.0, 0.7, key="sent_str") if sentiment_on else 0.0
            risk_negative = st.checkbox("Risk Factor NEGATIVE", value=False)
            alt_data_on = st.checkbox("Alt Data (Hiring Surge)", value=False)

        signals = []
        if momentum_on:
            signals.append(SignalInput("momentum", True, 1.0, 1.4))
        if insider_on:
            signals.append(SignalInput("insider", True, insider_strength, 1.8))
        if sentiment_on:
            signals.append(SignalInput("sentiment", True, sentiment_strength, 1.6))
        if risk_negative:
            signals.append(SignalInput("risk_factors", True, 1.0, 0.5))
        if alt_data_on:
            signals.append(SignalInput("alt_data", True, 0.6, 1.3))

        with col2:
            if signals:
                result = aggregate("SIMULATED", signals)

                # Gauge chart for posterior
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result.posterior * 100,
                    title={"text": "Posterior Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 60], "color": "lightcoral"},
                            {"range": [60, 75], "color": "lightyellow"},
                            {"range": [75, 85], "color": "lightgreen"},
                            {"range": [85, 100], "color": "darkgreen"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 75,
                        },
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, width="stretch")

                st.write(f"**Conviction:** {result.conviction_level}")
                st.write(f"**Should Trade:** {'YES' if result.should_trade else 'NO (below 75%)'}")

                # Attribution
                if len(signals) > 1:
                    attr = signal_attribution(result, signals)
                    attr_df = pd.DataFrame(attr)
                    fig2 = px.bar(attr_df, x="signal_name", y="marginal_contribution",
                                  title="Signal Contribution", color="marginal_contribution",
                                  color_continuous_scale="RdYlGn")
                    fig2.update_layout(yaxis_title="Marginal Contribution", xaxis_title="")
                    st.plotly_chart(fig2, width="stretch")
            else:
                st.info("Toggle signals on the left to see Bayesian updating in action.")

        # Position sizing preview
        st.divider()
        st.subheader("Position Sizing Preview")
        col1, col2, col3 = st.columns(3)
        portfolio_val = col1.number_input("Portfolio Value ($)", value=100000, step=10000)
        stock_price = col2.number_input("Stock Price ($)", value=1.50, step=0.10)
        kelly_frac = col3.slider("Kelly Fraction", 0.1, 1.0, 0.5)

        if signals:
            result = aggregate("PREVIEW", signals)
            if result.posterior > 0.5:
                full_kelly = kelly_fraction_calc(result.posterior, 0.08, 0.04)
                half_kelly = full_kelly * kelly_frac
                capped = min(half_kelly, 0.15)
                position_dollars = capped * portfolio_val
                shares = int(position_dollars // stock_price) if stock_price > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Full Kelly", f"{full_kelly:.1%}")
                c2.metric("Fractional Kelly", f"{capped:.1%}")
                c3.metric("Position Size", f"${position_dollars:,.0f}")
                c4.metric("Shares", f"{shares:,}")


# ─── Trades Page ───────────────────────────────────────────────────────────────
elif page == "Trades":
    st.title("Trade Log")

    trades = load_trades()
    if trades.empty:
        st.info("No trades yet. Run the pipeline to generate trades.")
    else:
        # Summary metrics
        open_trades = trades[trades["status"] == "open"]
        closed_trades = trades[trades["status"] == "closed"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Open", len(open_trades))
        col2.metric("Closed", len(closed_trades))

        if not closed_trades.empty:
            win_rate = (closed_trades["pnl"].dropna() > 0).mean()
            avg_return = closed_trades["return_pct"].dropna().mean()
            col3.metric("Win Rate", f"{win_rate:.0%}")
            col4.metric("Avg Return", f"{avg_return:.1%}")

        # Trade table
        st.subheader("All Trades")
        display_cols = ["ticker", "entry_date", "exit_date", "entry_price", "exit_price",
                        "position_size", "pnl", "return_pct", "posterior_at_entry", "regime_at_entry", "status"]
        available_cols = [c for c in display_cols if c in trades.columns]
        st.dataframe(trades[available_cols], width="stretch", hide_index=True)

        # P&L chart
        if not closed_trades.empty and "pnl" in closed_trades.columns:
            closed_trades_sorted = closed_trades.sort_values("exit_date")
            closed_trades_sorted["cumulative_pnl"] = closed_trades_sorted["pnl"].cumsum()
            fig = px.line(closed_trades_sorted, x="exit_date", y="cumulative_pnl",
                          title="Cumulative P&L")
            st.plotly_chart(fig, width="stretch")


# ─── Backtest Page ─────────────────────────────────────────────────────────────
elif page == "Backtest":
    st.title("Backtest Engine")
    st.info("Configure and run backtests from here. The backtest engine is at `backtest/engine.py`.")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
        kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.5, key="bt_kelly")

    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 12, 31))
        min_posterior = st.slider("Min Posterior", 0.5, 0.95, 0.75)
        max_position = st.slider("Max Position %", 0.05, 0.30, 0.15)

    st.code(f"""
from backtest.engine import run_backtest, BacktestConfig, print_summary

config = BacktestConfig(
    start_date="{start_date}",
    end_date="{end_date}",
    initial_capital={initial_capital},
    kelly_fraction={kelly_fraction},
    min_posterior={min_posterior},
    max_position_pct={max_position},
)

result = run_backtest(config)
print_summary(result)
""", language="python")

    if st.button("Run Backtest"):
        st.warning("Backtest requires price data from yfinance. Run from terminal: `python -c 'from backtest.engine import ...'`")


# ─── Attribution Page ──────────────────────────────────────────────────────────
elif page == "Attribution":
    st.title("Signal Attribution")

    trades = load_trades()
    closed = trades[trades["status"] == "closed"] if not trades.empty else pd.DataFrame()

    if closed.empty:
        st.info("No closed trades yet. Attribution requires completed trades with P&L data.")
        st.divider()
        st.subheader("How Attribution Works")
        st.markdown("""
        After 90 days of live trading, this page will show:

        - **Which signals generate alpha** — ranked by contribution to total P&L
        - **Win rate per signal** — does the signal predict winners?
        - **Regime performance** — which market conditions favor our system?
        - **Signal decay** — are signals weakening over time?

        This is the feedback loop that makes the system self-improving.
        Kill weak signals. Weight strong ones higher.
        """)
    else:
        from data.attribution import get_signal_attribution, get_regime_attribution, get_portfolio_stats

        lookback = st.slider("Lookback (days)", 30, 365, 90)
        stats = get_portfolio_stats(lookback)
        signal_attr = get_signal_attribution(lookback)
        regime_attr = get_regime_attribution(lookback)

        # Portfolio stats
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{stats['total_return']:.1%}")
        col2.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}")
        col3.metric("Max Drawdown", f"${stats['max_drawdown']:.0f}")
        col4.metric("Win Rate", f"{stats['win_rate']:.0%}")
        col5.metric("Avg Trade", f"${stats['avg_trade']:.0f}")

        # Signal attribution chart
        if not signal_attr.empty:
            st.subheader("Signal Performance")
            fig = px.bar(signal_attr, x="signal", y="contribution_to_total_pnl",
                         color="win_rate_when_fired", title="Signal Contribution to P&L",
                         color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(signal_attr, width="stretch", hide_index=True)

        # Regime attribution
        if not regime_attr.empty:
            st.subheader("Performance by Regime")
            fig = px.bar(regime_attr, x="regime", y="total_pnl", color="win_rate",
                         title="P&L by Market Regime", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, width="stretch")
