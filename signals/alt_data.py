"""
Alternative Data Signal Module for ASX Small Cap Quant System.

Uses free-tier alternative data sources as leading indicators:
- Job postings (Indeed AU) — 2-3 month lead on revenue changes
- Google Trends search interest — consumer demand proxy
- App Store ratings — churn/quality leading indicator
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from pytrends.request import TrendReq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _MODULE_DIR / ".alt_data_cache"
_CACHE_DIR.mkdir(exist_ok=True)

_HISTORY_FILE = _CACHE_DIR / "job_history.json"
_CACHE_FILE = _CACHE_DIR / "alt_data_cache.json"

_RATE_LIMIT_INTERVAL = 1.0  # seconds between requests per source
_last_request_time: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HiringSignal:
    """Signal derived from job posting changes."""

    company_name: str
    current_count: int
    previous_count: int
    change_pct: float
    signal: str  # "bullish", "bearish", or "neutral"
    score: float  # -5 to +5
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TrendSignal:
    """Signal derived from Google Trends search interest."""

    company_name: str
    ticker: str
    current_interest: float
    three_month_avg: float
    change_pct: float
    signal: str  # "bullish", "bearish", or "neutral"
    score: float  # -5 to +5
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AppSignal:
    """Signal derived from App Store rating changes."""

    app_id: str
    app_name: str
    current_rating: float
    three_month_avg_rating: float
    rating_change: float
    signal: str  # "bullish", "bearish", or "neutral"
    score: float  # -5 to +5
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AltDataResult:
    """Combined alternative data signal result."""

    company_name: str
    ticker: str
    hiring_signal: Optional[HiringSignal]
    trend_signal: Optional[TrendSignal]
    app_signal: Optional[AppSignal]
    composite_score: float  # -5 to +5
    signal_strength: float  # 0 to 1
    likelihood_ratio: float  # 1.3 bullish, 0.7 bearish, 1.0 neutral
    signal: str  # "bullish", "bearish", or "neutral"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


def _rate_limit(source: str) -> None:
    """Enforce minimum interval between requests for a given source."""
    now = time.time()
    last = _last_request_time.get(source, 0.0)
    elapsed = now - last
    if elapsed < _RATE_LIMIT_INTERVAL:
        time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
    _last_request_time[source] = time.time()


# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    """Load the results cache from disk."""
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    """Persist the results cache to disk."""
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def _get_cached(key: str, max_age_hours: int = 12) -> Optional[dict]:
    """Return cached result if fresh enough, else None."""
    cache = _load_cache()
    entry = cache.get(key)
    if entry is None:
        return None
    cached_time = datetime.fromisoformat(entry.get("_cached_at", "2000-01-01"))
    if datetime.utcnow() - cached_time > timedelta(hours=max_age_hours):
        return None
    return entry


def _set_cached(key: str, data: dict) -> None:
    """Store a result in the cache."""
    cache = _load_cache()
    data["_cached_at"] = datetime.utcnow().isoformat()
    cache[key] = data
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Job History Helpers
# ---------------------------------------------------------------------------


def _load_job_history() -> dict:
    """Load historical job posting counts."""
    if _HISTORY_FILE.exists():
        with open(_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_job_history(history: dict) -> None:
    """Persist historical job posting counts."""
    with open(_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _record_job_count(company_name: str, count: int) -> None:
    """Record today's job count for a company."""
    history = _load_job_history()
    key = company_name.lower().replace(" ", "_")
    if key not in history:
        history[key] = []
    history[key].append({
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "count": count,
    })
    # Keep only last 90 days of data
    history[key] = history[key][-90:]
    _save_job_history(history)


def _get_previous_job_count(company_name: str, days_ago: int = 30) -> Optional[int]:
    """Get the job count from approximately `days_ago` days in the past."""
    history = _load_job_history()
    key = company_name.lower().replace(" ", "_")
    entries = history.get(key, [])
    if not entries:
        return None
    target_date = datetime.utcnow() - timedelta(days=days_ago)
    # Find the entry closest to the target date
    best_entry = None
    best_diff = timedelta(days=999)
    for entry in entries:
        entry_date = datetime.strptime(entry["date"], "%Y-%m-%d")
        diff = abs(entry_date - target_date)
        if diff < best_diff:
            best_diff = diff
            best_entry = entry
    if best_entry and best_diff <= timedelta(days=7):
        return best_entry["count"]
    return None


# ---------------------------------------------------------------------------
# Sub-Signal 1: Job Postings (Indeed AU)
# ---------------------------------------------------------------------------


def _scrape_indeed_job_count(company_name: str) -> int:
    """
    Scrape approximate job count from Indeed AU for a company.

    Uses Indeed's search page and parses the result count from the page.
    """
    _rate_limit("indeed")
    url = "https://au.indeed.com/jobs"
    params = {
        "q": f'"{company_name}"',
        "l": "Australia",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        # Parse job count from the search results page
        # Indeed typically shows "Page 1 of X jobs" or "X jobs"
        text = resp.text
        import re

        # Try various patterns Indeed uses
        patterns = [
            r'"jobCount":\s*(\d+)',
            r'of\s+([\d,]+)\s+jobs',
            r'([\d,]+)\s+jobs',
            r'"totalResults":\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                count_str = match.group(1).replace(",", "")
                return int(count_str)
        return 0
    except (requests.RequestException, ValueError):
        return 0


def get_hiring_signal(company_name: str) -> HiringSignal:
    """
    Analyse job posting trends for a company as a leading indicator.

    A hiring surge (>30% increase over 30 days) is a bullish signal with
    approximately 2-3 month lead time on revenue growth.
    Mass reduction (>30% decrease) is bearish.

    Args:
        company_name: The company name to search for on Indeed AU.

    Returns:
        HiringSignal dataclass with signal assessment.
    """
    # Check cache first
    cache_key = f"hiring_{company_name.lower().replace(' ', '_')}"
    cached = _get_cached(cache_key, max_age_hours=24)
    if cached and "_cached_at" in cached:
        cached_copy = {k: v for k, v in cached.items() if k != "_cached_at"}
        return HiringSignal(**cached_copy)

    current_count = _scrape_indeed_job_count(company_name)
    _record_job_count(company_name, current_count)

    previous_count = _get_previous_job_count(company_name, days_ago=30)

    if previous_count is None or previous_count == 0:
        # Not enough history — neutral signal
        signal = HiringSignal(
            company_name=company_name,
            current_count=current_count,
            previous_count=previous_count or 0,
            change_pct=0.0,
            signal="neutral",
            score=0.0,
        )
    else:
        change_pct = (current_count - previous_count) / previous_count * 100.0

        if change_pct > 30.0:
            direction = "bullish"
            # Scale score: 30% → +2, 100%+ → +5
            score = min(5.0, 2.0 + (change_pct - 30.0) / 70.0 * 3.0)
        elif change_pct < -30.0:
            direction = "bearish"
            # Scale score: -30% → -2, -100% → -5
            score = max(-5.0, -2.0 + (change_pct + 30.0) / 70.0 * 3.0)
        else:
            direction = "neutral"
            # Mild score within ±2 range
            score = change_pct / 30.0 * 2.0

        signal = HiringSignal(
            company_name=company_name,
            current_count=current_count,
            previous_count=previous_count,
            change_pct=round(change_pct, 2),
            signal=direction,
            score=round(score, 2),
        )

    # Cache result
    _set_cached(cache_key, asdict(signal))
    return signal


# ---------------------------------------------------------------------------
# Sub-Signal 2: Google Trends
# ---------------------------------------------------------------------------


def get_search_trend(company_name: str, ticker: str) -> TrendSignal:
    """
    Analyse Google Trends search interest as a consumer demand proxy.

    For consumer-facing companies, rising search interest (>20% above
    3-month average) indicates growing mindshare and is a bullish signal.

    Args:
        company_name: The company name to search in Google Trends.
        ticker: The ASX ticker symbol.

    Returns:
        TrendSignal dataclass with signal assessment.
    """
    cache_key = f"trend_{ticker.lower()}"
    cached = _get_cached(cache_key, max_age_hours=12)
    if cached and "_cached_at" in cached:
        cached_copy = {k: v for k, v in cached.items() if k != "_cached_at"}
        return TrendSignal(**cached_copy)

    _rate_limit("google_trends")

    try:
        pytrends = TrendReq(hl="en-AU", tz=600)  # AEST = UTC+10 = 600 min
        # Search for company name over last 90 days
        pytrends.build_payload(
            [company_name],
            cat=0,
            timeframe="today 3-m",
            geo="AU",
        )
        interest_df = pytrends.interest_over_time()

        if interest_df.empty:
            signal = TrendSignal(
                company_name=company_name,
                ticker=ticker,
                current_interest=0.0,
                three_month_avg=0.0,
                change_pct=0.0,
                signal="neutral",
                score=0.0,
            )
        else:
            values = interest_df[company_name].tolist()
            # Current month: last 4 weeks of data
            current_interest = sum(values[-4:]) / min(4, len(values[-4:]))
            # 3-month average
            three_month_avg = sum(values) / len(values) if values else 0.0

            if three_month_avg == 0:
                change_pct = 0.0
            else:
                change_pct = (
                    (current_interest - three_month_avg) / three_month_avg * 100.0
                )

            if change_pct > 20.0:
                direction = "bullish"
                score = min(5.0, 2.0 + (change_pct - 20.0) / 80.0 * 3.0)
            elif change_pct < -20.0:
                direction = "bearish"
                score = max(-5.0, -2.0 + (change_pct + 20.0) / 80.0 * 3.0)
            else:
                direction = "neutral"
                score = change_pct / 20.0 * 2.0

            signal = TrendSignal(
                company_name=company_name,
                ticker=ticker,
                current_interest=round(current_interest, 2),
                three_month_avg=round(three_month_avg, 2),
                change_pct=round(change_pct, 2),
                signal=direction,
                score=round(score, 2),
            )

    except Exception:
        # Gracefully handle API failures
        signal = TrendSignal(
            company_name=company_name,
            ticker=ticker,
            current_interest=0.0,
            three_month_avg=0.0,
            change_pct=0.0,
            signal="neutral",
            score=0.0,
        )

    _set_cached(cache_key, asdict(signal))
    return signal


# ---------------------------------------------------------------------------
# Sub-Signal 3: App Store Rating
# ---------------------------------------------------------------------------


def _scrape_app_store_rating(app_id: str) -> Optional[dict]:
    """
    Fetch app rating from Apple App Store AU using the iTunes Lookup API.

    Args:
        app_id: The numeric Apple App Store app ID.

    Returns:
        Dict with 'rating' and 'name' keys, or None if not found.
    """
    _rate_limit("app_store")
    url = f"https://itunes.apple.com/au/lookup?id={app_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        app_info = results[0]
        return {
            "rating": app_info.get("averageUserRating", 0.0),
            "name": app_info.get("trackName", "Unknown"),
        }
    except (requests.RequestException, ValueError, KeyError):
        return None


def get_app_signal(app_id: Optional[str]) -> Optional[AppSignal]:
    """
    Analyse App Store rating trends as a quality/churn indicator.

    A rating drop > 0.3 stars vs 3-month average precedes customer churn
    and is a bearish signal.

    Args:
        app_id: The Apple App Store numeric app ID, or None if company
                has no app.

    Returns:
        AppSignal dataclass, or None if no app_id provided or app not found.
    """
    if app_id is None:
        return None

    cache_key = f"app_{app_id}"
    cached = _get_cached(cache_key, max_age_hours=24)
    if cached and "_cached_at" in cached:
        cached_copy = {k: v for k, v in cached.items() if k != "_cached_at"}
        return AppSignal(**cached_copy)

    app_data = _scrape_app_store_rating(app_id)
    if app_data is None:
        return None

    current_rating = app_data["rating"]
    app_name = app_data["name"]

    # Load rating history
    history = _load_job_history()  # Reuse history file for simplicity
    history_key = f"app_rating_{app_id}"
    entries = history.get(history_key, [])

    # Record current rating
    entries.append({
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "rating": current_rating,
    })
    entries = entries[-90:]  # Keep 90 days
    history[history_key] = entries
    _save_job_history(history)

    # Calculate 3-month average
    if len(entries) >= 7:
        # Use all historical ratings for the average
        ratings = [e["rating"] for e in entries[:-1]]  # Exclude today
        three_month_avg = sum(ratings) / len(ratings)
    else:
        # Not enough history
        three_month_avg = current_rating

    rating_change = current_rating - three_month_avg

    if rating_change < -0.3:
        direction = "bearish"
        # Scale: -0.3 → -2, -1.0+ → -5
        score = max(-5.0, -2.0 + (rating_change + 0.3) / 0.7 * 3.0)
    elif rating_change > 0.3:
        direction = "bullish"
        score = min(5.0, 2.0 + (rating_change - 0.3) / 0.7 * 3.0)
    else:
        direction = "neutral"
        score = rating_change / 0.3 * 2.0

    signal = AppSignal(
        app_id=app_id,
        app_name=app_name,
        current_rating=round(current_rating, 2),
        three_month_avg_rating=round(three_month_avg, 2),
        rating_change=round(rating_change, 2),
        signal=direction,
        score=round(score, 2),
    )

    _set_cached(cache_key, asdict(signal))
    return signal


# ---------------------------------------------------------------------------
# Composite Signal
# ---------------------------------------------------------------------------


def _compute_composite(
    hiring: Optional[HiringSignal],
    trend: Optional[TrendSignal],
    app: Optional[AppSignal],
) -> tuple[float, float, float, str]:
    """
    Compute composite score, signal strength, likelihood ratio, and direction.

    Returns:
        Tuple of (composite_score, signal_strength, likelihood_ratio, signal).
    """
    scores: list[float] = []
    weights: list[float] = []

    if hiring is not None:
        scores.append(hiring.score)
        weights.append(1.0)

    if trend is not None:
        scores.append(trend.score)
        weights.append(0.8)

    if app is not None:
        scores.append(app.score)
        weights.append(0.7)

    if not scores:
        return 0.0, 0.0, 1.0, "neutral"

    # Weighted average
    total_weight = sum(weights)
    composite_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    composite_score = max(-5.0, min(5.0, composite_score))

    # Signal strength: how far from zero on 0-1 scale
    signal_strength = min(1.0, abs(composite_score) / 5.0)

    # Direction and likelihood ratio
    if composite_score > 2.0:
        signal = "bullish"
        likelihood_ratio = 1.3
    elif composite_score < -2.0:
        signal = "bearish"
        likelihood_ratio = 0.7
    else:
        signal = "neutral"
        likelihood_ratio = 1.0

    return (
        round(composite_score, 2),
        round(signal_strength, 2),
        likelihood_ratio,
        signal,
    )


def get_alt_data(
    company_name: str,
    ticker: str,
    app_id: Optional[str] = None,
) -> AltDataResult:
    """
    Get combined alternative data signal for a single stock.

    Aggregates hiring trends, Google search interest, and optionally
    App Store ratings into a composite signal.

    Args:
        company_name: Full company name for search queries.
        ticker: ASX ticker symbol (e.g. "APX").
        app_id: Optional Apple App Store numeric ID if the company has an app.

    Returns:
        AltDataResult with composite score and individual sub-signals.
    """
    hiring = get_hiring_signal(company_name)
    trend = get_search_trend(company_name, ticker)
    app = get_app_signal(app_id)

    composite_score, signal_strength, likelihood_ratio, signal = _compute_composite(
        hiring, trend, app
    )

    return AltDataResult(
        company_name=company_name,
        ticker=ticker,
        hiring_signal=hiring,
        trend_signal=trend,
        app_signal=app,
        composite_score=composite_score,
        signal_strength=signal_strength,
        likelihood_ratio=likelihood_ratio,
        signal=signal,
    )


def batch_get_alt_data(
    stocks: list[dict[str, Optional[str]]],
) -> list[AltDataResult]:
    """
    Get alternative data signals for multiple stocks.

    Args:
        stocks: List of dicts, each with keys:
            - "company_name" (str): Full company name.
            - "ticker" (str): ASX ticker symbol.
            - "app_id" (Optional[str]): Apple App Store ID or None.

    Returns:
        List of AltDataResult for each stock.

    Example:
        results = batch_get_alt_data([
            {"company_name": "Appen Limited", "ticker": "APX", "app_id": None},
            {"company_name": "Afterpay", "ticker": "APT", "app_id": "1033379464"},
        ])
    """
    results: list[AltDataResult] = []
    for stock in stocks:
        result = get_alt_data(
            company_name=stock["company_name"],
            ticker=stock["ticker"],
            app_id=stock.get("app_id"),
        )
        results.append(result)
    return results
