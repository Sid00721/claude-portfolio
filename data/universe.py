"""
ASX Small Cap Universe Screener

Filters ASX-listed stocks to the target universe:
- Market cap $100M-$1B AUD
- Average daily volume > 50k shares (liquidity floor)
- Returns clean dataframe with ticker, market cap, sector, 12-month return
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import time

from data.db import get_db


ASX_TICKERS_URL = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"

MIN_MARKET_CAP = 100_000_000
MAX_MARKET_CAP = 1_000_000_000
MIN_AVG_VOLUME = 50_000

BATCH_SIZE = 50
BATCH_DELAY = 2


def get_asx_tickers() -> list[str]:
    """Fetch all ASX-listed company tickers."""
    try:
        df = pd.read_csv(ASX_TICKERS_URL, skiprows=1)
        df.columns = [c.strip() for c in df.columns]
        tickers = df["ASX code"].dropna().str.strip().tolist()
        return [f"{t}.AX" for t in tickers]
    except Exception:
        cache_path = os.path.join(os.path.dirname(__file__), "asx_tickers_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)
        raise


def _fetch_info_batch(tickers: list[str]) -> list[dict]:
    """Fetch info for a batch of tickers with rate limiting."""
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or info.get("quoteType") == "NONE":
                continue

            market_cap = info.get("marketCap")
            if market_cap is None:
                continue
            if not (MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP):
                continue

            avg_volume = info.get("averageVolume", 0)
            if avg_volume < MIN_AVG_VOLUME:
                continue

            results.append({
                "ticker": ticker,
                "market_cap": market_cap,
                "avg_volume": avg_volume,
                "sector": info.get("sector", "Unknown"),
            })
        except Exception:
            continue
        time.sleep(0.3)
    return results


def build_universe(max_workers: int = 5) -> pd.DataFrame:
    """
    Screen entire ASX in batches, then bulk-fetch price history.
    """
    all_tickers = get_asx_tickers()
    print(f"      Screening {len(all_tickers)} tickers in batches of {BATCH_SIZE}...")

    # Phase 1: Filter by market cap and volume using .info (batched)
    candidates = []
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i + BATCH_SIZE]
        batch_results = _fetch_info_batch(batch)
        candidates.extend(batch_results)
        if i + BATCH_SIZE < len(all_tickers):
            time.sleep(BATCH_DELAY)
        done = min(i + BATCH_SIZE, len(all_tickers))
        print(f"      [{done}/{len(all_tickers)}] {len(candidates)} candidates so far", end="\r")

    print(f"\n      {len(candidates)} passed market cap + volume filters")

    if not candidates:
        return pd.DataFrame()

    # Phase 2: Bulk download 1-year price history for candidates
    candidate_tickers = [c["ticker"] for c in candidates]
    print(f"      Downloading price history for {len(candidate_tickers)} stocks...")

    price_data = yf.download(
        candidate_tickers,
        period="1y",
        group_by="ticker",
        progress=False,
        threads=True,
    )

    # Phase 3: Compute 12-month returns and filter
    results = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        try:
            if len(candidate_tickers) == 1:
                hist = price_data
            else:
                hist = price_data[ticker]

            close = hist["Close"].dropna()
            if len(close) < 200:
                continue

            price_now = float(close.iloc[-1])
            price_1y = float(close.iloc[0])
            return_12m = (price_now - price_1y) / price_1y

            candidate["return_12m"] = round(return_12m, 4)
            candidate["price"] = round(price_now, 2)
            results.append(candidate)
        except (KeyError, IndexError, TypeError):
            continue

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
    """Main entry point: build universe and persist."""
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
