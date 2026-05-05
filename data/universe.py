"""
ASX Small Cap Universe Screener

Uses EODHD for reliable data (falls back to yfinance in dev).
Filters: $100M-$1B AUD market cap, >50k avg volume.
"""

import pandas as pd
from datetime import datetime
import os
import json
import time

from data.db import get_db
from data.provider import get_asx_listings, get_fundamentals, get_price_history, _has_eodhd
from data.activity import log_activity

MIN_MARKET_CAP = 100_000_000
MAX_MARKET_CAP = 1_000_000_000
MIN_AVG_VOLUME = 50_000


def build_universe() -> pd.DataFrame:
    """Screen ASX, return filtered universe."""
    listings = get_asx_listings()
    if listings.empty:
        log_activity("error", "Could not fetch ASX listings", severity="alert")
        return pd.DataFrame()

    log_activity("scan", f"Checking {len(listings)} ASX tickers", severity="info")

    results = []
    checked = 0

    for _, row in listings.iterrows():
        ticker = row["ticker"]
        try:
            fundies = get_fundamentals(ticker)
            mc = fundies.get("market_cap", 0)
            if mc is None or not (MIN_MARKET_CAP <= mc <= MAX_MARKET_CAP):
                continue

            vol = fundies.get("avg_volume", 0) or 0
            if vol < MIN_AVG_VOLUME and _has_eodhd():
                continue

            # Get 12-month return
            prices = get_price_history(ticker, 365)
            if prices.empty or len(prices) < 200:
                continue

            close_col = "adj_close" if "adj_close" in prices.columns else "close"
            price_now = float(prices[close_col].iloc[-1])
            price_1y = float(prices[close_col].iloc[0])
            return_12m = (price_now - price_1y) / price_1y

            results.append({
                "ticker": ticker,
                "market_cap": int(mc),
                "avg_volume": int(vol),
                "sector": fundies.get("sector", "Unknown"),
                "return_12m": round(return_12m, 4),
                "price": round(price_now, 4),
                "price_to_book": fundies.get("price_to_book"),
            })
        except Exception:
            continue

        checked += 1
        if checked % 50 == 0:
            log_activity("scan", f"Screened {checked} tickers, {len(results)} passed so far", severity="info")

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("return_12m", ascending=False).reset_index(drop=True)
    df["screened_at"] = datetime.now().isoformat()
    return df


def save_to_db(df: pd.DataFrame) -> None:
    """Save universe to local SQLite."""
    with get_db() as conn:
        conn.execute("DELETE FROM universe")
        for _, row in df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO universe
                   (ticker, market_cap, avg_volume, sector, return_12m, price, screened_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (row["ticker"], int(row["market_cap"]), int(row["avg_volume"]),
                 row["sector"], row["return_12m"], row["price"], row.get("screened_at")),
            )


def run():
    """Main entry point."""
    print(f"[{datetime.now().isoformat()}] Building ASX universe...")
    df = build_universe()
    print(f"Universe: {len(df)} stocks passed filters")
    if not df.empty:
        print(df[["ticker", "market_cap", "sector", "return_12m"]].head(20).to_string())
    save_to_db(df)
    print("Saved to local database.")
    return df


if __name__ == "__main__":
    run()
