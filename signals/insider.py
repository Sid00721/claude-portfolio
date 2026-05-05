"""
Insider Trading Signal for ASX Small Cap Quant System.

Uses EODHD API (via data.provider) to detect high-conviction insider buying activity.
Falls back to local JSON cache if API returns nothing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path

from data.provider import get_insider_transactions

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "insider_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_WINDOW_DAYS = 7
CLUSTER_MULTIPLIER = 1.5
SIGNAL_THRESHOLD = 7
LIKELIHOOD_RATIO = 1.8

SENIOR_TITLES = re.compile(
    r"\b(CEO|Managing Director|MD|Chief Executive|Executive Director|Chairman)\b",
    re.IGNORECASE,
)


@dataclass
class InsiderSignal:
    """Aggregated insider signal output for a given ticker."""

    ticker: str
    insider_score: float
    cluster_detected: bool
    signal_strength: float
    signal_active: bool
    transactions: list[dict] = field(default_factory=list)


def _is_senior(owner_name: str) -> bool:
    """Determine if an insider holds a CEO/MD/Chairman or equivalent role."""
    return bool(SENIOR_TITLES.search(owner_name))


def _classify_transaction_type(txn: dict) -> str:
    """Classify an EODHD transaction into buy/sell/grant."""
    txn_type = (txn.get("transactionType") or "").lower()
    if "buy" in txn_type or "purchase" in txn_type:
        return "buy"
    if "sell" in txn_type or "sale" in txn_type or "disposal" in txn_type:
        return "sell"
    if "grant" in txn_type or "exercise" in txn_type or "option" in txn_type:
        return "grant"
    # Default: treat unknown as grant (low score, non-zero)
    return "grant"


def _compute_value(txn: dict) -> float:
    """Compute AUD value of a transaction from EODHD fields."""
    amount = txn.get("transactionAmount")
    if amount is not None:
        try:
            return abs(float(amount))
        except (ValueError, TypeError):
            pass
    # Fallback: shares * price
    shares = txn.get("transactionShares")
    price = txn.get("transactionPrice")
    if shares is not None and price is not None:
        try:
            return abs(float(shares) * float(price))
        except (ValueError, TypeError):
            pass
    return 0.0


def _score_transaction(txn: dict) -> float:
    """
    Score an individual insider transaction by quality (0-10).

    Scoring:
        CEO/MD/Chairman open market buy > $100k AUD = 10
        Director open market buy > $100k = 8
        CEO/MD buy > $50k = 7
        Director buy > $50k = 5
        Any buy < $50k = 3
        Options exercise / grant = 1
        Selling = 0 (ignored entirely)
    """
    txn_class = _classify_transaction_type(txn)

    if txn_class == "sell":
        return 0.0
    if txn_class == "grant":
        return 1.0

    # It's a buy
    value = _compute_value(txn)
    owner = txn.get("ownerName") or ""
    senior = _is_senior(owner)

    if senior and value > 100_000:
        return 10.0
    if not senior and value > 100_000:
        return 8.0
    if senior and value > 50_000:
        return 7.0
    if not senior and value > 50_000:
        return 5.0

    return 3.0


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse a date string from EODHD into a datetime object."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _detect_cluster(transactions: list[dict]) -> bool:
    """
    Detect if multiple distinct insiders bought within the same 7-day window.

    Returns True if 2+ distinct insiders have purchases within CLUSTER_WINDOW_DAYS.
    """
    buys = [
        t for t in transactions if _classify_transaction_type(t) == "buy"
    ]
    if len(buys) < 2:
        return False

    # Parse dates and pair with owner
    dated_buys: list[tuple[datetime, str]] = []
    for txn in buys:
        dt = _parse_date(txn.get("reportDate"))
        if dt is None:
            continue
        owner = txn.get("ownerName") or "Unknown"
        dated_buys.append((dt, owner))

    if len(dated_buys) < 2:
        return False

    dated_buys.sort(key=lambda x: x[0])

    for i, (date_a, owner_a) in enumerate(dated_buys):
        distinct_owners = {owner_a}
        for date_b, owner_b in dated_buys[i + 1:]:
            if (date_b - date_a) <= timedelta(days=CLUSTER_WINDOW_DAYS):
                distinct_owners.add(owner_b)
            else:
                break
        if len(distinct_owners) >= 2:
            return True

    return False


def _cache_path(ticker: str) -> Path:
    """Get the cache file path for a ticker."""
    return CACHE_DIR / f"{ticker.upper()}_insider.json"


def _is_cache_fresh(ticker: str) -> bool:
    """Check if the cache was written today (avoid repeated API calls same day)."""
    path = _cache_path(ticker)
    if not path.exists():
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        cached_date = data.get("cached_date")
        if cached_date == date.today().isoformat():
            return True
    except (json.JSONDecodeError, IOError, KeyError):
        pass
    return False


def _load_cache(ticker: str) -> list[dict]:
    """Load cached insider transactions from local JSON file."""
    path = _cache_path(ticker)
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("transactions", [])
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to read insider cache for %s: %s", ticker, e)
        return []


def _save_cache(ticker: str, transactions: list[dict]) -> None:
    """Persist insider transactions to local JSON cache with today's date."""
    path = _cache_path(ticker)
    payload = {
        "cached_date": date.today().isoformat(),
        "ticker": ticker.upper(),
        "transactions": transactions,
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except IOError as e:
        logger.warning("Failed to write insider cache for %s: %s", ticker, e)


def get_insider_signal(ticker: str) -> InsiderSignal:
    """
    Generate the insider trading signal for a given ASX ticker.

    Fetches insider transactions from EODHD API via data.provider, scores them,
    detects clusters, and returns a normalized signal.

    Uses same-day cache to avoid repeated API calls. Falls back to local JSON
    cache if EODHD returns no data.

    Args:
        ticker: ASX ticker symbol (e.g. "ABC" or "ABC.AU").

    Returns:
        InsiderSignal with score, cluster detection, and transaction details.
    """
    # Check if we already have fresh cache for today
    if _is_cache_fresh(ticker):
        transactions = _load_cache(ticker)
    else:
        # Fetch from EODHD API
        transactions = get_insider_transactions(ticker, limit=50)

        if transactions:
            _save_cache(ticker, transactions)
        else:
            # Fall back to local cache if API returned nothing
            logger.info(
                "EODHD returned no insider data for %s, falling back to cache.", ticker
            )
            transactions = _load_cache(ticker)

    # Filter out sells entirely
    actionable = [
        t for t in transactions if _classify_transaction_type(t) != "sell"
    ]

    if not actionable:
        return InsiderSignal(
            ticker=ticker.upper(),
            insider_score=0.0,
            cluster_detected=False,
            signal_strength=0.0,
            signal_active=False,
            transactions=[],
        )

    # Score each transaction and take the maximum
    scores = [_score_transaction(t) for t in actionable]
    raw_score = max(scores) if scores else 0.0

    # Cluster detection (uses all transactions including sells for date context)
    cluster_detected = _detect_cluster(transactions)
    if cluster_detected:
        raw_score = min(raw_score * CLUSTER_MULTIPLIER, 10.0)

    # Normalize signal strength to 0-1
    signal_strength = min(raw_score / 10.0, 1.0)

    # Signal fires when score >= threshold
    signal_active = raw_score >= SIGNAL_THRESHOLD

    return InsiderSignal(
        ticker=ticker.upper(),
        insider_score=round(raw_score, 2),
        cluster_detected=cluster_detected,
        signal_strength=round(signal_strength, 4),
        signal_active=signal_active,
        transactions=actionable,
    )


def get_insider_signals(tickers: list[str]) -> dict[str, InsiderSignal]:
    """
    Generate insider signals for multiple tickers (batch processing).

    Args:
        tickers: List of ASX ticker symbols.

    Returns:
        Dictionary mapping ticker -> InsiderSignal.
    """
    results: dict[str, InsiderSignal] = {}
    for ticker in tickers:
        try:
            results[ticker] = get_insider_signal(ticker)
        except Exception as e:
            logger.error("Error generating insider signal for %s: %s", ticker, e)
            results[ticker] = InsiderSignal(
                ticker=ticker.upper(),
                insider_score=0.0,
                cluster_detected=False,
                signal_strength=0.0,
                signal_active=False,
                transactions=[],
            )
    return results


def likelihood_ratio() -> float:
    """
    Return the likelihood ratio for insider cluster buying.

    Insider cluster buying historically predicts outperformance approximately
    64% of the time, yielding a likelihood ratio of ~1.8.
    """
    return LIKELIHOOD_RATIO
