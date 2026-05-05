"""
Market Data Provider — EODHD

Single source for all market data:
- Price history (daily OHLCV)
- Fundamentals (market cap, P/B, sector, dividends)
- Insider transactions
- Exchange listings

EODHD API: https://eodhd.com/financial-apis
Tickers use format: SYMBOL.AU (not .AX like yfinance)

Falls back to yfinance if EODHD key not set (dev mode).
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

EODHD_KEY = os.environ.get("EODHD_API_KEY", "")
BASE_URL = "https://eodhd.com/api"


def _has_eodhd() -> bool:
    return bool(EODHD_KEY) and EODHD_KEY != "demo"


def _get(endpoint: str, params: dict = None) -> dict | list:
    """Make authenticated EODHD API request."""
    if params is None:
        params = {}
    params["api_token"] = EODHD_KEY
    params["fmt"] = "json"
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─── Exchange Listings ─────────────────────────────────────────────────────────

def get_asx_listings() -> pd.DataFrame:
    """Get all ASX-listed stocks with basic info."""
    if not _has_eodhd():
        return _fallback_listings()

    data = _get("exchange-symbol-list/AU")
    df = pd.DataFrame(data)
    df = df[df["Type"] == "Common Stock"]
    df = df.rename(columns={"Code": "ticker", "Name": "name", "Exchange": "exchange"})
    df["ticker"] = df["ticker"] + ".AU"
    return df[["ticker", "name", "exchange"]].reset_index(drop=True)


# ─── Price Data ────────────────────────────────────────────────────────────────

def get_price_history(ticker: str, period_days: int = 365) -> pd.DataFrame:
    """Get daily OHLCV for a single ticker."""
    if not _has_eodhd():
        return _fallback_price(ticker, period_days)

    # EODHD uses .AU not .AX
    eodhd_ticker = ticker.replace(".AX", ".AU")
    start = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")

    data = _get(f"eod/{eodhd_ticker}", {"from": start, "period": "d"})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.rename(columns={"adjusted_close": "adj_close"})
    return df


def get_bulk_prices(tickers: list[str], period_days: int = 365) -> dict[str, pd.DataFrame]:
    """Get price history for multiple tickers."""
    results = {}
    for ticker in tickers:
        try:
            df = get_price_history(ticker, period_days)
            if not df.empty:
                results[ticker] = df
        except Exception:
            continue
        if _has_eodhd():
            time.sleep(0.1)  # rate limit: 10 req/sec on paid plan
    return results


# ─── Fundamentals ──────────────────────────────────────────────────────────────

def get_fundamentals(ticker: str) -> dict:
    """Get company fundamentals (market cap, P/B, sector, dividends, etc.)"""
    if not _has_eodhd():
        return _fallback_fundamentals(ticker)

    eodhd_ticker = ticker.replace(".AX", ".AU")
    data = _get(f"fundamentals/{eodhd_ticker}")

    general = data.get("General", {})
    highlights = data.get("Highlights", {})
    valuation = data.get("Valuation", {})

    return {
        "ticker": ticker,
        "name": general.get("Name", ""),
        "sector": general.get("Sector", "Unknown"),
        "industry": general.get("Industry", ""),
        "market_cap": highlights.get("MarketCapitalization", 0),
        "price_to_book": valuation.get("PriceBookMRQ", None),
        "pe_ratio": highlights.get("PERatio", None),
        "dividend_yield": highlights.get("DividendYield", 0),
        "avg_volume": highlights.get("50DayMA", 0),  # use as proxy
    }


def get_bulk_fundamentals(tickers: list[str]) -> list[dict]:
    """Get fundamentals for multiple tickers."""
    results = []
    for ticker in tickers:
        try:
            data = get_fundamentals(ticker)
            if data.get("market_cap", 0) > 0:
                results.append(data)
        except Exception:
            continue
        if _has_eodhd():
            time.sleep(0.1)
    return results


# ─── Insider Transactions ──────────────────────────────────────────────────────

def get_insider_transactions(ticker: str, limit: int = 50) -> list[dict]:
    """Get insider transactions for a ticker from EODHD."""
    if not _has_eodhd():
        return []

    eodhd_ticker = ticker.replace(".AX", ".AU")
    try:
        data = _get(f"insider-transactions", {"code": eodhd_ticker, "limit": limit})
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ─── Dividends (for franking credit signal) ────────────────────────────────────

def get_dividends(ticker: str) -> list[dict]:
    """Get dividend history including ex-dates."""
    if not _has_eodhd():
        return _fallback_dividends(ticker)

    eodhd_ticker = ticker.replace(".AX", ".AU")
    try:
        data = _get(f"div/{eodhd_ticker}", {"from": "2024-01-01"})
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ─── Regime Data ───────────────────────────────────────────────────────────────

def get_asx_vix() -> float:
    """Get ASX VIX (XVI index)."""
    if not _has_eodhd():
        return _fallback_vix()

    try:
        data = _get("eod/XVI.AU", {"period": "d", "order": "d", "limit": 1})
        if data:
            return float(data[0]["close"])
    except Exception:
        pass
    return _fallback_vix()


def get_au_yield_spread() -> tuple[float, str]:
    """Get Australian 10Y - 2Y yield spread."""
    if not _has_eodhd():
        return _fallback_yield()

    try:
        # AU 10Y bond
        au10y = _get("eod/AU10YB.BOND", {"period": "d", "order": "d", "limit": 1})
        # AU 2Y bond
        au2y = _get("eod/AU2YB.BOND", {"period": "d", "order": "d", "limit": 1})

        if au10y and au2y:
            spread = float(au10y[0]["close"]) - float(au2y[0]["close"])
            signal = "bullish" if spread > 0 else "bearish"
            return spread, signal
    except Exception:
        pass
    return _fallback_yield()


# ─── Fallbacks (yfinance) ──────────────────────────────────────────────────────

def _fallback_listings() -> pd.DataFrame:
    """Fallback: get ASX tickers from CSV."""
    try:
        df = pd.read_csv("https://www.asx.com.au/asx/research/ASXListedCompanies.csv", skiprows=1)
        df.columns = [c.strip() for c in df.columns]
        tickers = df["ASX code"].dropna().str.strip().tolist()
        return pd.DataFrame({"ticker": [f"{t}.AX" for t in tickers], "name": df["Company name"].tolist()})
    except Exception:
        return pd.DataFrame()


def _fallback_price(ticker: str, period_days: int) -> pd.DataFrame:
    """Fallback: use yfinance for price data."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{period_days}d")
        return df
    except Exception:
        return pd.DataFrame()


def _fallback_fundamentals(ticker: str) -> dict:
    """Fallback: use yfinance for fundamentals."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "name": info.get("shortName", ""),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "price_to_book": info.get("priceToBook", None),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", 0),
            "avg_volume": info.get("averageVolume", 0),
        }
    except Exception:
        return {"ticker": ticker, "market_cap": 0}


def _fallback_dividends(ticker: str) -> list:
    return []


def _fallback_vix() -> float:
    """Fallback: use yfinance for VIX."""
    try:
        import yfinance as yf
        hist = yf.Ticker("^VIX").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 20.0  # default to selective


def _fallback_yield() -> tuple[float, str]:
    """Fallback: use yfinance for US yields as proxy."""
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX").history(period="5d")
        irx = yf.Ticker("^IRX").history(period="5d")
        if not tnx.empty and not irx.empty:
            spread = float(tnx["Close"].iloc[-1]) - float(irx["Close"].iloc[-1])
            return spread, "bullish" if spread > 0 else "bearish"
    except Exception:
        pass
    return 0.5, "bullish"
