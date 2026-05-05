"""
Annual Report Risk Factor Language Change Detector

Detects changes in risk factor language between consecutive annual reports
for ASX-listed companies. Uses Claude to perform semantic diffing and scoring
of risk section changes.

Signal is NEGATIVE: fires when risk is INCREASING (composite > 2.0).
When signal fires, likelihood_ratio = 0.5 (evidence AGAINST the stock).
When risk is DECREASING (composite < -2.0), likelihood_ratio = 1.3 (mild positive).
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class RiskItem:
    """A single risk factor with its severity score."""

    description: str
    severity: int  # 1-5


@dataclass
class RiskFactorDelta:
    """Complete risk factor change analysis between two annual reports."""

    ticker: str
    new_risks_added: list[RiskItem] = field(default_factory=list)
    removed_risks: list[str] = field(default_factory=list)
    language_hedging_change: int = 0  # -5 to +5, more hedged = more concerning
    revenue_recognition_changes: bool = False  # any change = red flag
    litigation_language_change: int = 0  # -5 to +5, more litigation = concerning
    overall_risk_delta: int = 0  # -5 to +5, net change in risk profile
    composite_score: float = 0.0
    likelihood_ratio: float = 1.0
    signal_fired: bool = False
    signal_direction: Optional[str] = None  # "negative", "positive", or None

    def compute_composite(self) -> None:
        """
        Compute the weighted composite score.

        Weights:
          - new_risks (avg severity): 0.30
          - language_hedging_change:   0.25
          - revenue_recognition:       0.20 (binary but high weight)
          - litigation_language_change: 0.15
          - overall_risk_delta:        0.10
        """
        # Average severity of new risks (0 if none added)
        if self.new_risks_added:
            avg_severity = sum(r.severity for r in self.new_risks_added) / len(
                self.new_risks_added
            )
        else:
            avg_severity = 0.0

        # Revenue recognition is binary: 5.0 if changed, 0.0 if not
        revenue_score = 5.0 if self.revenue_recognition_changes else 0.0

        self.composite_score = (
            0.30 * avg_severity
            + 0.25 * self.language_hedging_change
            + 0.20 * revenue_score
            + 0.15 * self.litigation_language_change
            + 0.10 * self.overall_risk_delta
        )

        # Determine signal
        if self.composite_score > 2.0:
            self.signal_fired = True
            self.signal_direction = "negative"
            self.likelihood_ratio = 0.5  # Evidence AGAINST the stock
        elif self.composite_score < -2.0:
            self.signal_fired = True
            self.signal_direction = "positive"
            self.likelihood_ratio = 1.3  # Mild positive
        else:
            self.signal_fired = False
            self.signal_direction = None
            self.likelihood_ratio = 1.0  # Neutral


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst specialising in ASX-listed small cap companies.
You are comparing risk factor sections from two consecutive annual reports for the same company.
Your task is to perform a semantic diff and produce structured scores.

Respond ONLY with valid JSON matching this schema:
{
  "new_risks_added": [
    {"description": "string describing the new risk", "severity": <int 1-5>}
  ],
  "removed_risks": ["string describing each risk that was removed"],
  "language_hedging_change": <int -5 to +5>,
  "revenue_recognition_changes": <bool>,
  "litigation_language_change": <int -5 to +5>,
  "overall_risk_delta": <int -5 to +5>
}

Scoring definitions:
- new_risks_added: List of risk factors present in the CURRENT year but NOT in the PREVIOUS year. Severity 1 = minor/boilerplate, 5 = existential threat.
- removed_risks: Risk factors present in the PREVIOUS year but absent from the CURRENT year (potentially resolved issues).
- language_hedging_change: Score from -5 to +5. Positive means MORE hedging language (e.g., "may", "could", "potentially", "uncertain") was added — this is MORE CONCERNING. Negative means language became more definitive/confident.
- revenue_recognition_changes: true if there are ANY changes to how the company describes its revenue recognition policies or practices in the risk section. This is a red flag regardless of direction.
- litigation_language_change: Score from -5 to +5. Positive means MORE litigation-related language (lawsuits, claims, disputes, regulatory actions). Negative means less.
- overall_risk_delta: Net change in overall risk profile from -5 to +5. Positive = riskier, negative = less risky.

Be precise and conservative. Only flag genuine semantic changes, not minor rewording."""


def _build_user_prompt(
    current_year_risk_section: str, previous_year_risk_section: str
) -> str:
    return f"""Compare the following two risk factor sections from consecutive annual reports of the same ASX-listed company.

## PREVIOUS YEAR RISK SECTION:
{previous_year_risk_section}

## CURRENT YEAR RISK SECTION:
{current_year_risk_section}

Analyse the changes and provide your structured JSON assessment."""


# ---------------------------------------------------------------------------
# API Interaction
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0


def _call_claude_with_retries(
    client: anthropic.Anthropic,
    current_year_risk_section: str,
    previous_year_risk_section: str,
) -> dict:
    """
    Call the Anthropic API with exponential backoff retries.

    Returns the parsed JSON response dict.
    Raises the last exception if all retries are exhausted.
    """
    user_prompt = _build_user_prompt(current_year_risk_section, previous_year_risk_section)
    last_exception: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract text content from response
            response_text = message.content[0].text

            # Parse JSON from response
            parsed = json.loads(response_text)
            return parsed

        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                backoff = BASE_BACKOFF_SECONDS * (2**attempt)
                time.sleep(backoff)
        except json.JSONDecodeError as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                backoff = BASE_BACKOFF_SECONDS * (2**attempt)
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to get valid response after {MAX_RETRIES} attempts: {last_exception}"
    )


def _parse_response(ticker: str, data: dict) -> RiskFactorDelta:
    """Parse the Claude JSON response into a RiskFactorDelta dataclass."""
    new_risks = [
        RiskItem(description=r["description"], severity=r["severity"])
        for r in data.get("new_risks_added", [])
    ]

    delta = RiskFactorDelta(
        ticker=ticker,
        new_risks_added=new_risks,
        removed_risks=data.get("removed_risks", []),
        language_hedging_change=int(data.get("language_hedging_change", 0)),
        revenue_recognition_changes=bool(data.get("revenue_recognition_changes", False)),
        litigation_language_change=int(data.get("litigation_language_change", 0)),
        overall_risk_delta=int(data.get("overall_risk_delta", 0)),
    )

    delta.compute_composite()
    return delta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze(
    ticker: str,
    current_year_risk_section: str,
    previous_year_risk_section: str,
    client: Optional[anthropic.Anthropic] = None,
) -> RiskFactorDelta:
    """
    Analyze risk factor language changes for a single ASX stock.

    Args:
        ticker: ASX ticker symbol (e.g., "APX.AX")
        current_year_risk_section: Full text of the risk section from the current annual report
        previous_year_risk_section: Full text of the risk section from the previous annual report
        client: Optional pre-configured Anthropic client. If None, creates a new one.

    Returns:
        RiskFactorDelta with composite score and signal information.
    """
    if client is None:
        client = anthropic.Anthropic()

    data = _call_claude_with_retries(client, current_year_risk_section, previous_year_risk_section)
    return _parse_response(ticker, data)


def batch_analyze(
    stocks: list[dict],
    client: Optional[anthropic.Anthropic] = None,
) -> list[RiskFactorDelta]:
    """
    Analyze risk factor changes for multiple ASX stocks.

    Args:
        stocks: List of dicts, each containing:
            - "ticker": str (ASX ticker symbol)
            - "current_year_risk_section": str
            - "previous_year_risk_section": str
        client: Optional pre-configured Anthropic client. If None, creates a new one.

    Returns:
        List of RiskFactorDelta results for each stock.
    """
    if client is None:
        client = anthropic.Anthropic()

    results: list[RiskFactorDelta] = []

    for stock in stocks:
        ticker = stock["ticker"]
        current_section = stock["current_year_risk_section"]
        previous_section = stock["previous_year_risk_section"]

        try:
            delta = analyze(
                ticker=ticker,
                current_year_risk_section=current_section,
                previous_year_risk_section=previous_section,
                client=client,
            )
            results.append(delta)
        except RuntimeError:
            # If all retries fail for a stock, append a neutral delta
            neutral = RiskFactorDelta(ticker=ticker)
            neutral.compute_composite()
            results.append(neutral)

    return results
