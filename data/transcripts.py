"""
Earnings Transcript Scraper for ASX Companies

Fetches earnings call transcripts from ASX announcements and caches
them locally for subsequent analysis and comparison.
"""

import os
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


# Base directory for cached transcripts
CACHE_DIR = Path(__file__).parent / "transcripts"


def _ensure_cache_dir(ticker: str) -> Path:
    """Ensure the cache directory for a given ticker exists and return its path."""
    ticker_dir = CACHE_DIR / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir


def _cache_path(ticker: str, quarter: str) -> Path:
    """Return the file path for a cached transcript."""
    ticker_dir = _ensure_cache_dir(ticker)
    safe_quarter = quarter.replace("/", "_").replace(" ", "_")
    return ticker_dir / f"{safe_quarter}.txt"


def _fetch_from_asx(ticker: str, quarter: str) -> Optional[str]:
    """
    Attempt to fetch an earnings transcript from the ASX announcements page.

    ASX announcements are listed at:
        https://www.asx.com.au/asx/statistics/announcements.do?issuerId={ticker}

    This searches for earnings-related PDFs/documents in the announcements list.
    Returns the transcript text if found, None otherwise.
    """
    ticker_upper = ticker.upper()
    url = (
        f"https://www.asx.com.au/asx/statistics/announcements.do"
        f"?issuerId={ticker_upper}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Look for announcement links containing earnings-related keywords
    earnings_keywords = [
        "earnings",
        "results",
        "half year",
        "full year",
        "quarterly",
        "investor presentation",
        "earnings call",
        "transcript",
    ]

    # Parse the quarter string to help match relevant announcements
    # Expected formats: "FY24", "H1FY24", "Q1FY24", "2024Q1", etc.
    quarter_lower = quarter.lower()

    links = soup.find_all("a", href=True)
    candidate_urls: list[str] = []

    for link in links:
        link_text = link.get_text(strip=True).lower()
        href = link["href"]

        # Check if link text matches earnings keywords and quarter
        is_earnings = any(kw in link_text for kw in earnings_keywords)
        matches_quarter = quarter_lower in link_text or any(
            part in link_text for part in quarter_lower.split()
        )

        if is_earnings and matches_quarter and href.endswith(".pdf"):
            if href.startswith("/"):
                href = f"https://www.asx.com.au{href}"
            candidate_urls.append(href)

    if not candidate_urls:
        return None

    # Attempt to download the first matching PDF and extract text
    for pdf_url in candidate_urls[:3]:
        transcript_text = _extract_text_from_pdf_url(pdf_url, headers)
        if transcript_text:
            return transcript_text

    return None


def _extract_text_from_pdf_url(url: str, headers: dict) -> Optional[str]:
    """
    Download a PDF from a URL and attempt to extract its text content.

    Uses pdfplumber if available, otherwise returns None.
    """
    try:
        import pdfplumber
        import io

        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        pdf_bytes = io.BytesIO(response.content)
        text_parts: list[str] = []

        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        if text_parts:
            return "\n\n".join(text_parts)
    except ImportError:
        # pdfplumber not available
        pass
    except Exception:
        pass

    return None


def fetch_transcript(ticker: str, quarter: str) -> str:
    """
    Fetch an earnings transcript for an ASX company.

    Strategy:
        1. Check local cache first
        2. Try fetching from ASX announcements
        3. Store fetched transcript in cache for future use

    Args:
        ticker: ASX ticker symbol (e.g., "CBA", "BHP")
        quarter: Quarter identifier (e.g., "FY24", "H1FY24", "Q1FY25")

    Returns:
        The transcript text.

    Raises:
        FileNotFoundError: If no transcript could be found or fetched.
    """
    cache_file = _cache_path(ticker, quarter)

    # Check local cache
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    # Try ASX announcements
    transcript = _fetch_from_asx(ticker, quarter)

    if transcript:
        # Cache for future use
        cache_file.write_text(transcript, encoding="utf-8")
        return transcript

    raise FileNotFoundError(
        f"No transcript found for {ticker.upper()} {quarter}. "
        f"You can manually place a transcript file at: {cache_file}"
    )


def fetch_previous_transcript(ticker: str, quarter: str) -> str:
    """
    Fetch the previous quarter's transcript for comparison.

    Derives the previous quarter from the given quarter string and fetches it.

    Args:
        ticker: ASX ticker symbol
        quarter: Current quarter identifier (e.g., "H1FY24" -> previous is "H2FY23")

    Returns:
        The previous quarter's transcript text.

    Raises:
        FileNotFoundError: If no previous transcript could be found.
    """
    prev_quarter = _derive_previous_quarter(quarter)
    return fetch_transcript(ticker, prev_quarter)


def _derive_previous_quarter(quarter: str) -> str:
    """
    Derive the previous quarter identifier from the given one.

    Handles common ASX reporting formats:
        - "FY24" -> "FY23"
        - "H1FY24" -> "H2FY23"
        - "H2FY24" -> "H1FY24"
        - "Q1FY24" -> "Q4FY23"
        - "Q2FY24" -> "Q1FY24"
        - "Q3FY24" -> "Q2FY24"
        - "Q4FY24" -> "Q3FY24"
    """
    q = quarter.upper()

    if q.startswith("H1FY"):
        year = int(q[4:])
        return f"H2FY{year - 1:02d}" if year < 100 else f"H2FY{year - 1}"
    elif q.startswith("H2FY"):
        year = q[4:]
        return f"H1FY{year}"
    elif q.startswith("Q1FY"):
        year = int(q[4:])
        return f"Q4FY{year - 1:02d}" if year < 100 else f"Q4FY{year - 1}"
    elif q.startswith("Q") and q[1].isdigit() and "FY" in q:
        q_num = int(q[1])
        year = q[q.index("FY") + 2:]
        return f"Q{q_num - 1}FY{year}"
    elif q.startswith("FY"):
        year = int(q[2:])
        return f"FY{year - 1:02d}" if year < 100 else f"FY{year - 1}"
    else:
        # Unknown format — just return as-is with _prev suffix
        return f"{quarter}_prev"


def list_available_transcripts(ticker: str) -> list[tuple[str, str]]:
    """
    List all locally cached transcripts for a given ticker.

    Returns:
        A list of (quarter, filepath) tuples sorted alphabetically by quarter.
    """
    ticker_dir = CACHE_DIR / ticker.upper()

    if not ticker_dir.exists():
        return []

    results: list[tuple[str, str]] = []
    for file in sorted(ticker_dir.glob("*.txt")):
        # Reconstruct quarter from filename
        quarter = file.stem.replace("_", "/")
        results.append((quarter, str(file)))

    return results
