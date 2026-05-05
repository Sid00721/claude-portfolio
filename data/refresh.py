"""
Weekly Universe Refresh & Daily Price Update

- refresh_universe(): Full rescan of ASX for stocks in $100M-$1B range (weekly, Sunday night AEST)
- update_prices(): Lightweight daily price/return update for cached universe
"""

import json
import os
from datetime import datetime

import pandas as pd

from data.activity import log_activity
from data.db import get_db
from data.provider import get_price_history
from data.universe import build_universe, save_to_db

_SEED_PATH = os.path.join(os.path.dirname(__file__), "universe_seed.json")


def _load_existing_universe() -> set[str]:
    """Load current universe tickers from DB."""
    with get_db() as conn:
        rows = conn.execute("SELECT ticker FROM universe").fetchall()
    return {row["ticker"] for row in rows}


def _format_market_cap(mc: int) -> str:
    """Format market cap for human-readable logs."""
    if mc >= 1_000_000_000:
        return f"${mc / 1_000_000_000:.2f}B"
    return f"${mc / 1_000_000:.0f}M"


def refresh_universe() -> pd.DataFrame | None:
    """
    Weekly universe refresh. Rescans all ASX stocks via EODHD to find
    new entrants and remove graduates from the $100M-$1B range.

    Returns the new universe DataFrame on success, None on failure.
    """
    log_activity("scan", "Weekly universe refresh started", severity="info")

    existing_tickers = _load_existing_universe()

    try:
        new_df = build_universe()
    except Exception as e:
        log_activity(
            "error",
            "Universe refresh failed — keeping existing universe",
            detail=str(e),
            severity="alert",
        )
        return None

    if new_df.empty:
        log_activity(
            "error",
            "Universe refresh returned empty — keeping existing universe",
            severity="alert",
        )
        return None

    new_tickers = set(new_df["ticker"].tolist())

    # Determine additions and removals
    additions = new_tickers - existing_tickers
    removals = existing_tickers - new_tickers

    # Log each significant change
    for ticker in sorted(additions):
        row = new_df[new_df["ticker"] == ticker].iloc[0]
        mc_str = _format_market_cap(int(row["market_cap"]))
        log_activity(
            "scan",
            f"NEW: {ticker} entered universe (market cap {mc_str})",
            ticker=ticker,
            severity="success",
        )

    for ticker in sorted(removals):
        # Try to determine why it left (we don't have its new data, just note the removal)
        log_activity(
            "scan",
            f"REMOVED: {ticker} left universe (no longer meets $100M-$1B criteria)",
            ticker=ticker,
            severity="warning",
        )

    # Save new universe to DB
    save_to_db(new_df)

    # Update universe_seed.json for deployment seeding
    seed_data = new_df.to_dict(orient="records")
    with open(_SEED_PATH, "w") as f:
        json.dump(seed_data, f)

    total = len(new_tickers)
    log_activity(
        "scan",
        f"Universe refresh complete: {len(additions)} added, {len(removals)} removed, {total} total",
        severity="success",
    )

    return new_df


def update_prices() -> None:
    """
    Daily price update. For each stock in the cached universe, fetches the
    latest price from EODHD and updates price + return_12m in the DB.
    Faster than a full rescan since it skips fundamentals.
    """
    log_activity("scan", "Daily price update started", severity="info")

    with get_db() as conn:
        rows = conn.execute("SELECT ticker FROM universe").fetchall()

    tickers = [row["ticker"] for row in rows]

    if not tickers:
        log_activity("scan", "No stocks in universe — skipping price update", severity="warning")
        return

    updated = 0
    errors = 0

    for i, ticker in enumerate(tickers):
        try:
            prices = get_price_history(ticker, period_days=365)
            if prices.empty or len(prices) < 2:
                errors += 1
                continue

            close_col = "adj_close" if "adj_close" in prices.columns else "close"
            price_now = float(prices[close_col].iloc[-1])
            price_1y = float(prices[close_col].iloc[0])
            return_12m = round((price_now - price_1y) / price_1y, 4)

            with get_db() as conn:
                conn.execute(
                    "UPDATE universe SET price = ?, return_12m = ? WHERE ticker = ?",
                    (round(price_now, 4), return_12m, ticker),
                )
            updated += 1

        except Exception:
            errors += 1
            continue

        # Log progress every 25 stocks
        if (i + 1) % 25 == 0:
            log_activity(
                "scan",
                f"Price update progress: {i + 1}/{len(tickers)} processed",
                severity="info",
            )

    log_activity(
        "scan",
        f"Daily price update complete: {updated} updated, {errors} errors, {len(tickers)} total",
        severity="success",
    )
