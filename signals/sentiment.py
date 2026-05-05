"""
LLM-powered earnings transcript sentiment delta signal.

Compares current quarter's earnings transcript to previous quarter's to detect
shifts in management confidence, guidance specificity, hedging language, and
forward-looking density. This is the primary LLM edge for the ASX small cap
quant system.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

import anthropic


SENTIMENT_PROMPT = """\
You are an expert financial analyst specialising in ASX small-cap earnings calls.

You will be given two earnings call transcripts from the same company:
- PREVIOUS QUARTER transcript
- CURRENT QUARTER transcript

Your task is to compare the two and score the CHANGE in sentiment across five dimensions.
Each score ranges from -5 to +5 where:
  -5 = dramatically worse / more bearish than last quarter
   0 = no meaningful change
  +5 = dramatically better / more bullish than last quarter

Dimensions to score:

1. management_confidence_shift (-5 to +5)
   How has management's confidence changed? Look for:
   - Tone of voice in prepared remarks (definitive vs tentative language)
   - Willingness to make bold statements about the business
   - Use of words like "confident", "strong", "pleased" vs "challenging", "uncertain", "cautious"
   Examples: +5 = went from "we're navigating headwinds" to "we're firing on all cylinders"
             -5 = went from "excellent momentum" to "significant challenges ahead"

2. guidance_specificity_change (-5 to +5)
   Has guidance become more or less specific? More specific = bullish (management has visibility).
   Look for:
   - Concrete numbers vs ranges vs qualitative statements
   - Timeframes becoming more specific or more vague
   - Withdrawal or narrowing of guidance
   Examples: +5 = went from "we expect growth" to "we expect 18-22% revenue growth in H2"
             -5 = went from specific targets to "we're not providing guidance at this time"

3. hedging_language_change (-5 to +5)
   Has hedging language increased or decreased? More hedging = bearish.
   Look for:
   - Qualifiers: "subject to", "depending on", "if market conditions allow"
   - Disclaimers and caveats increasing or decreasing
   - Legal/risk language density
   Examples: +5 = removed most caveats, speaking with conviction
             -5 = added many qualifiers, heavy use of "potential", "possible", "subject to"

4. forward_looking_density (-5 to +5)
   Are they making more or fewer forward-looking statements? More = bullish (they see opportunity).
   Look for:
   - Plans, initiatives, pipeline commentary
   - Discussion of future projects, contracts, expansion
   - Balance of retrospective vs prospective commentary
   Examples: +5 = shifted from reviewing past results to outlining ambitious future plans
             -5 = mostly backward-looking, few mentions of future plans

5. overall_sentiment_delta (-5 to +5)
   Net change in overall tone, capturing anything not covered above.
   Consider Q&A tone, analyst interactions, and general energy of the call.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "management_confidence_shift": <int>,
  "guidance_specificity_change": <int>,
  "hedging_language_change": <int>,
  "forward_looking_density": <int>,
  "overall_sentiment_delta": <int>
}

--- PREVIOUS QUARTER TRANSCRIPT ---
{previous_transcript}

--- CURRENT QUARTER TRANSCRIPT ---
{current_transcript}
"""

# Weights for composite score
WEIGHTS = {
    "management_confidence_shift": 0.30,
    "guidance_specificity_change": 0.25,
    "hedging_language_change": 0.20,
    "forward_looking_density": 0.15,
    "overall_sentiment_delta": 0.10,
}

# Signal thresholds
COMPOSITE_THRESHOLD = 2.0
POSITIVE_LIKELIHOOD_RATIO = 1.6
NEGATIVE_LIKELIHOOD_RATIO = 0.6

# Retry config
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0


@dataclass
class SentimentDelta:
    """Sentiment delta scores from comparing two earnings transcripts."""

    management_confidence_shift: int
    guidance_specificity_change: int
    hedging_language_change: int
    forward_looking_density: int
    overall_sentiment_delta: int
    composite_score: float
    signal_strength: float
    ticker: Optional[str] = None

    @property
    def signal_fires(self) -> bool:
        """Signal fires when composite is meaningfully more bullish than last quarter."""
        return self.composite_score > COMPOSITE_THRESHOLD

    def likelihood_ratio(self) -> float:
        """
        Returns likelihood ratio for Bayesian updating.
        1.6 for positive delta (bullish shift), 0.6 for negative delta.
        """
        if self.composite_score > 0:
            return POSITIVE_LIKELIHOOD_RATIO
        return NEGATIVE_LIKELIHOOD_RATIO


def _compute_composite(scores: dict[str, int]) -> float:
    """Compute weighted average composite score from individual dimensions."""
    composite = 0.0
    for dimension, weight in WEIGHTS.items():
        composite += scores[dimension] * weight
    return composite


def _normalize_signal_strength(composite: float) -> float:
    """Normalize composite score from [-5, +5] range to [0, 1] range."""
    return (composite + 5.0) / 10.0


def _call_claude_with_retries(
    client: anthropic.Anthropic,
    current_transcript: str,
    previous_transcript: str,
) -> dict[str, int]:
    """
    Call Claude to analyze transcript sentiment delta with exponential backoff retries.

    Returns parsed JSON scores dict on success.
    Raises the last exception after all retries are exhausted.
    """
    prompt = SENTIMENT_PROMPT.format(
        current_transcript=current_transcript,
        previous_transcript=previous_transcript,
    )

    last_exception: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = message.content[0].text
            scores = json.loads(response_text)

            # Validate all expected keys are present and values are in range
            expected_keys = list(WEIGHTS.keys())
            for key in expected_keys:
                if key not in scores:
                    raise ValueError(f"Missing key in response: {key}")
                val = scores[key]
                if not isinstance(val, int) or val < -5 or val > 5:
                    raise ValueError(
                        f"Invalid value for {key}: {val} (must be int in [-5, +5])"
                    )

            return scores

        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                backoff = BASE_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(backoff)
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                backoff = BASE_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to get valid sentiment scores after {MAX_RETRIES} attempts. "
        f"Last error: {last_exception}"
    )


def analyze_sentiment_delta(
    current_transcript: str,
    previous_transcript: str,
    ticker: Optional[str] = None,
    client: Optional[anthropic.Anthropic] = None,
) -> SentimentDelta:
    """
    Analyze the sentiment delta between two earnings transcripts.

    Args:
        current_transcript: The current quarter's earnings call transcript.
        previous_transcript: The previous quarter's earnings call transcript.
        ticker: Optional ASX ticker symbol for reference.
        client: Optional Anthropic client instance. Creates one if not provided.

    Returns:
        SentimentDelta dataclass with all scores, composite, and signal strength.
    """
    if client is None:
        client = anthropic.Anthropic()

    scores = _call_claude_with_retries(client, current_transcript, previous_transcript)

    composite = _compute_composite(scores)
    signal_strength = _normalize_signal_strength(composite)

    return SentimentDelta(
        management_confidence_shift=scores["management_confidence_shift"],
        guidance_specificity_change=scores["guidance_specificity_change"],
        hedging_language_change=scores["hedging_language_change"],
        forward_looking_density=scores["forward_looking_density"],
        overall_sentiment_delta=scores["overall_sentiment_delta"],
        composite_score=composite,
        signal_strength=signal_strength,
        ticker=ticker,
    )


def batch_analyze(
    transcripts: list[tuple[str, str, str]],
    client: Optional[anthropic.Anthropic] = None,
) -> list[SentimentDelta]:
    """
    Analyze sentiment deltas for multiple companies.

    Args:
        transcripts: List of (ticker, current_transcript, previous_transcript) tuples.
        client: Optional Anthropic client instance. Creates one if not provided.

    Returns:
        List of SentimentDelta results for each company.
    """
    if client is None:
        client = anthropic.Anthropic()

    results: list[SentimentDelta] = []

    for ticker, current, previous in transcripts:
        try:
            delta = analyze_sentiment_delta(
                current_transcript=current,
                previous_transcript=previous,
                ticker=ticker,
                client=client,
            )
            results.append(delta)
        except RuntimeError:
            # Log failure but continue processing remaining tickers
            results.append(
                SentimentDelta(
                    management_confidence_shift=0,
                    guidance_specificity_change=0,
                    hedging_language_change=0,
                    forward_looking_density=0,
                    overall_sentiment_delta=0,
                    composite_score=0.0,
                    signal_strength=0.5,
                    ticker=ticker,
                )
            )

    return results
