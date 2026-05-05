"""
Insider Trading Signal for ASX Small Cap Quant System.

Scrapes ASX Appendix 3Y (Change of Director's Interest Notice) filings
to detect high-conviction insider buying activity.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "insider_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASX_ANNOUNCEMENTS_BASE = "https://www.asx.com.au/asx/statistics/announcements.do"

CLUSTER_WINDOW_DAYS = 7
CLUSTER_MULTIPLIER = 1.5
SIGNAL_THRESHOLD = 7
LIKELIHOOD_RATIO = 1.8

SENIOR_TITLES = re.compile(
    r"\b(CEO|Managing Director|MD|Chief Executive|Executive Director)\b",
    re.IGNORECASE,
)


@dataclass
class InsiderTransaction:
    """A single parsed insider transaction from an Appendix 3Y filing."""

    ticker: str
    director_name: str
    director_title: str
    transaction_type: str  # "buy", "sell", "options_exercise"
    value_aud: float
    shares: int
    date: datetime
    on_market: bool
    filing_url: str

    @property
    def is_senior(self) -> bool:
        """Whether the director holds a CEO/MD or equivalent role."""
        return bool(SENIOR_TITLES.search(self.director_title))


@dataclass
class InsiderSignal:
    """Aggregated insider signal output for a given ticker."""

    ticker: str
    insider_score: float
    cluster_detected: bool
    signal_strength: float
    transactions: list[InsiderTransaction] = field(default_factory=list)

    @property
    def signal_fires(self) -> bool:
        return self.insider_score >= SIGNAL_THRESHOLD


def _score_transaction(txn: InsiderTransaction) -> float:
    """Score an individual insider transaction by quality (0-10)."""
    if txn.transaction_type == "sell":
        return 0.0
    if txn.transaction_type == "options_exercise":
        return 1.0

    # On-market purchase scoring
    if txn.is_senior and txn.value_aud > 100_000:
        return 10.0
    if not txn.is_senior and txn.value_aud > 100_000:
        return 8.0
    if txn.is_senior and txn.value_aud > 50_000:
        return 7.0
    if not txn.is_senior and txn.value_aud > 50_000:
        return 5.0
    if txn.on_market and txn.value_aud < 50_000:
        return 3.0

    return 2.0


def _detect_cluster(transactions: list[InsiderTransaction]) -> bool:
    """
    Detect if multiple directors bought within the same 7-day window.

    Returns True if 2+ distinct directors have purchases within CLUSTER_WINDOW_DAYS.
    """
    buys = [t for t in transactions if t.transaction_type == "buy"]
    if len(buys) < 2:
        return False

    buys_sorted = sorted(buys, key=lambda t: t.date)
    for i, txn_a in enumerate(buys_sorted):
        directors_in_window = {txn_a.director_name}
        for txn_b in buys_sorted[i + 1 :]:
            if (txn_b.date - txn_a.date) <= timedelta(days=CLUSTER_WINDOW_DAYS):
                directors_in_window.add(txn_b.director_name)
            else:
                break
        if len(directors_in_window) >= 2:
            return True

    return False


def _build_announcements_url(ticker: str, num_days: int = 90) -> str:
    """Build the ASX announcements URL for Appendix 3Y filings."""
    params = {
        "asxCode": ticker.upper(),
        "timeframe": "D",
        "period": str(num_days),
        "announceType": "03Y",  # Appendix 3Y
    }
    return f"{ASX_ANNOUNCEMENTS_BASE}?{urlencode(params)}"


def _parse_filing_page(html: str, ticker: str, filing_url: str) -> Optional[InsiderTransaction]:
    """
    Parse an individual Appendix 3Y filing page to extract transaction details.

    Returns None if parsing fails or the filing contains no actionable transaction.
    """
    soup = BeautifulSoup(html, "html.parser")

    director_name = ""
    director_title = ""
    transaction_type = "buy"
    value_aud = 0.0
    shares = 0
    date = datetime.now()
    on_market = True

    # Extract director name (Part 1 of Appendix 3Y)
    name_pattern = re.compile(r"Name of Director", re.IGNORECASE)
    name_element = soup.find(string=name_pattern)
    if name_element:
        parent = name_element.find_parent("td") or name_element.find_parent("div")
        if parent:
            next_cell = parent.find_next_sibling("td") or parent.find_next("td")
            if next_cell:
                director_name = next_cell.get_text(strip=True)

    # Extract director title
    title_pattern = re.compile(r"(Position|Title|Office)", re.IGNORECASE)
    title_element = soup.find(string=title_pattern)
    if title_element:
        parent = title_element.find_parent("td") or title_element.find_parent("div")
        if parent:
            next_cell = parent.find_next_sibling("td") or parent.find_next("td")
            if next_cell:
                director_title = next_cell.get_text(strip=True)

    # Extract consideration/value (Part 3 - Nature of change)
    consideration_pattern = re.compile(r"Consideration", re.IGNORECASE)
    consideration_element = soup.find(string=consideration_pattern)
    if consideration_element:
        parent = consideration_element.find_parent("td") or consideration_element.find_parent("div")
        if parent:
            next_cell = parent.find_next_sibling("td") or parent.find_next("td")
            if next_cell:
                text = next_cell.get_text(strip=True)
                money_match = re.search(r"\$?([\d,]+\.?\d*)", text)
                if money_match:
                    value_aud = float(money_match.group(1).replace(",", ""))

    # Extract number of shares
    shares_pattern = re.compile(r"Number of securities", re.IGNORECASE)
    shares_element = soup.find(string=shares_pattern)
    if shares_element:
        parent = shares_element.find_parent("td") or shares_element.find_parent("div")
        if parent:
            next_cell = parent.find_next_sibling("td") or parent.find_next("td")
            if next_cell:
                text = next_cell.get_text(strip=True)
                num_match = re.search(r"([\d,]+)", text)
                if num_match:
                    shares = int(num_match.group(1).replace(",", ""))

    # Determine transaction type
    nature_pattern = re.compile(r"Nature of change", re.IGNORECASE)
    nature_element = soup.find(string=nature_pattern)
    if nature_element:
        parent = nature_element.find_parent("td") or nature_element.find_parent("div")
        if parent:
            next_cell = parent.find_next_sibling("td") or parent.find_next("td")
            if next_cell:
                nature_text = next_cell.get_text(strip=True).lower()
                if "exercise" in nature_text or "option" in nature_text or "right" in nature_text:
                    transaction_type = "options_exercise"
                elif "sale" in nature_text or "sold" in nature_text or "sell" in nature_text:
                    transaction_type = "sell"
                else:
                    transaction_type = "buy"

    # Determine if on-market
    market_pattern = re.compile(r"(on-market|off-market)", re.IGNORECASE)
    full_text = soup.get_text()
    market_match = market_pattern.search(full_text)
    if market_match:
        on_market = "on-market" in market_match.group(0).lower()

    # Extract date
    date_pattern = re.compile(r"Date of change", re.IGNORECASE)
    date_element = soup.find(string=date_pattern)
    if date_element:
        parent = date_element.find_parent("td") or date_element.find_parent("div")
        if parent:
            next_cell = parent.find_next_sibling("td") or parent.find_next("td")
            if next_cell:
                date_text = next_cell.get_text(strip=True)
                for fmt in ("%d/%m/%Y", "%d %B %Y", "%d-%m-%Y", "%Y-%m-%d"):
                    try:
                        date = datetime.strptime(date_text, fmt)
                        break
                    except ValueError:
                        continue

    if not director_name:
        return None

    return InsiderTransaction(
        ticker=ticker.upper(),
        director_name=director_name,
        director_title=director_title,
        transaction_type=transaction_type,
        value_aud=value_aud,
        shares=shares,
        date=date,
        on_market=on_market,
        filing_url=filing_url,
    )


def _scrape_announcements(ticker: str, num_days: int = 90) -> list[InsiderTransaction]:
    """
    Scrape the ASX announcements page for Appendix 3Y filings for a given ticker.

    Returns a list of parsed insider transactions.
    """
    url = _build_announcements_url(ticker, num_days)
    transactions: list[InsiderTransaction] = []

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch announcements for %s: %s", ticker, e)
        return transactions

    soup = BeautifulSoup(response.text, "html.parser")

    # Parse announcement links from the results table
    rows = soup.select("table tr") or soup.find_all("tr")
    for row in rows:
        links = row.find_all("a", href=True)
        for link in links:
            href = link["href"]
            if "3Y" in link.get_text() or "director" in link.get_text().lower():
                filing_url = href if href.startswith("http") else f"https://www.asx.com.au{href}"
                try:
                    time.sleep(0.5)  # Rate limiting
                    filing_response = requests.get(filing_url, headers=headers, timeout=30)
                    filing_response.raise_for_status()
                    txn = _parse_filing_page(filing_response.text, ticker, filing_url)
                    if txn:
                        transactions.append(txn)
                except requests.RequestException as e:
                    logger.warning("Failed to fetch filing %s: %s", filing_url, e)
                    continue

    return transactions


def _load_cache(ticker: str) -> list[InsiderTransaction]:
    """Load cached insider transactions from local JSON file."""
    cache_file = CACHE_DIR / f"{ticker.upper()}_insider.json"
    if not cache_file.exists():
        return []

    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to read cache for %s: %s", ticker, e)
        return []

    transactions: list[InsiderTransaction] = []
    for entry in data:
        try:
            txn = InsiderTransaction(
                ticker=entry["ticker"],
                director_name=entry["director_name"],
                director_title=entry["director_title"],
                transaction_type=entry["transaction_type"],
                value_aud=float(entry["value_aud"]),
                shares=int(entry["shares"]),
                date=datetime.fromisoformat(entry["date"]),
                on_market=entry["on_market"],
                filing_url=entry.get("filing_url", ""),
            )
            transactions.append(txn)
        except (KeyError, ValueError) as e:
            logger.warning("Skipping malformed cache entry for %s: %s", ticker, e)
            continue

    return transactions


def _save_cache(ticker: str, transactions: list[InsiderTransaction]) -> None:
    """Persist insider transactions to local JSON cache."""
    cache_file = CACHE_DIR / f"{ticker.upper()}_insider.json"
    data = [
        {
            "ticker": txn.ticker,
            "director_name": txn.director_name,
            "director_title": txn.director_title,
            "transaction_type": txn.transaction_type,
            "value_aud": txn.value_aud,
            "shares": txn.shares,
            "date": txn.date.isoformat(),
            "on_market": txn.on_market,
            "filing_url": txn.filing_url,
        }
        for txn in transactions
    ]

    try:
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        logger.warning("Failed to write cache for %s: %s", ticker, e)


def get_insider_signal(ticker: str, num_days: int = 90, use_cache_fallback: bool = True) -> InsiderSignal:
    """
    Generate the insider trading signal for a given ASX ticker.

    Scrapes ASX Appendix 3Y filings, scores transactions, detects clusters,
    and returns a normalized signal.

    Args:
        ticker: ASX ticker symbol (e.g., "ABC").
        num_days: Lookback period in days for announcement scraping.
        use_cache_fallback: If True, falls back to local cache on scrape failure.

    Returns:
        InsiderSignal with score, cluster detection, and transaction details.
    """
    transactions = _scrape_announcements(ticker, num_days)

    if not transactions and use_cache_fallback:
        logger.info("Scraping returned no results for %s, falling back to cache.", ticker)
        transactions = _load_cache(ticker)
    elif transactions:
        _save_cache(ticker, transactions)

    # Filter to buys only for scoring (selling is ignored entirely)
    buys = [t for t in transactions if t.transaction_type != "sell"]

    if not buys:
        return InsiderSignal(
            ticker=ticker.upper(),
            insider_score=0.0,
            cluster_detected=False,
            signal_strength=0.0,
            transactions=[],
        )

    # Score each transaction and take the maximum
    scores = [_score_transaction(txn) for txn in buys]
    raw_score = max(scores) if scores else 0.0

    # Cluster detection
    cluster_detected = _detect_cluster(transactions)
    if cluster_detected:
        raw_score = min(raw_score * CLUSTER_MULTIPLIER, 10.0)

    # Normalize signal strength to 0-1
    signal_strength = min(raw_score / 10.0, 1.0)

    return InsiderSignal(
        ticker=ticker.upper(),
        insider_score=round(raw_score, 2),
        cluster_detected=cluster_detected,
        signal_strength=round(signal_strength, 4),
        transactions=buys,
    )


def signal_fires(ticker: str) -> bool:
    """Check whether the insider signal fires for a given ticker."""
    signal = get_insider_signal(ticker)
    return signal.signal_fires


def likelihood_ratio() -> float:
    """
    Return the likelihood ratio for insider cluster buying.

    Insider cluster buying historically predicts outperformance approximately
    64% of the time, yielding a likelihood ratio of ~1.8.
    """
    return LIKELIHOOD_RATIO


def monitor(
    tickers: list[str],
    poll_interval_seconds: int = 300,
    score_threshold: float = 7.0,
    callback: Optional[callable] = None,
) -> None:
    """
    Continuously monitor a list of tickers for high-conviction insider buys.

    Runs in an infinite loop, checking each ticker on the specified interval.
    When a signal fires above the threshold, triggers the callback or logs an alert.

    Args:
        tickers: List of ASX ticker symbols to monitor.
        poll_interval_seconds: Seconds between each polling cycle (default: 300s / 5min).
        score_threshold: Minimum insider_score to trigger an alert.
        callback: Optional callable(InsiderSignal) invoked on alert. Defaults to logging.
    """
    logger.info(
        "Starting insider monitor for %d tickers, poll interval %ds, threshold %.1f",
        len(tickers),
        poll_interval_seconds,
        score_threshold,
    )

    while True:
        for ticker in tickers:
            try:
                signal = get_insider_signal(ticker)

                if signal.insider_score >= score_threshold:
                    alert_msg = (
                        f"INSIDER ALERT: {ticker} | score={signal.insider_score} | "
                        f"cluster={signal.cluster_detected} | "
                        f"strength={signal.signal_strength} | "
                        f"txns={len(signal.transactions)}"
                    )
                    logger.warning(alert_msg)

                    if callback:
                        callback(signal)
                else:
                    logger.debug(
                        "No signal for %s (score=%.2f)", ticker, signal.insider_score
                    )

            except Exception as e:
                logger.error("Error monitoring %s: %s", ticker, e)

            # Small delay between tickers to avoid hammering ASX
            time.sleep(1)

        logger.info("Monitor cycle complete. Sleeping %ds.", poll_interval_seconds)
        time.sleep(poll_interval_seconds)
