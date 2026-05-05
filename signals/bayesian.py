"""
Bayesian Belief Aggregator

Core intelligence layer for the ASX small cap quant system.
Takes signals from multiple sources and produces a posterior probability
of outperformance for each stock via iterative Bayesian updating.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# Default likelihood ratios for each signal type
DEFAULT_LIKELIHOOD_RATIOS: Dict[str, float] = {
    "momentum_q5": 1.4,
    "insider_cluster_buying": 1.8,
    "sentiment_delta_positive": 1.6,
    "risk_factor_change_negative": 0.5,
    "alternative_data_hiring_surge": 1.3,
}

BASE_PRIOR = 0.5


@dataclass
class SignalInput:
    """Represents a single signal feeding into the Bayesian updater."""

    signal_name: str
    signal_active: bool
    signal_strength: float  # 0-1, used to scale the likelihood ratio
    likelihood_ratio: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.signal_strength <= 1.0:
            raise ValueError(
                f"signal_strength must be between 0 and 1, got {self.signal_strength}"
            )
        if self.likelihood_ratio <= 0.0:
            raise ValueError(
                f"likelihood_ratio must be positive, got {self.likelihood_ratio}"
            )

    @property
    def effective_lr(self) -> float:
        """
        Compute the effective likelihood ratio scaled by signal strength.

        When strength is 1.0, the full LR applies.
        When strength is 0.0, the LR is neutral (1.0).
        Interpolates log-linearly between 1.0 and the raw LR.
        """
        if not self.signal_active:
            return 1.0
        # Log-linear interpolation: LR_effective = LR ^ strength
        return self.likelihood_ratio ** self.signal_strength


@dataclass
class BayesianResult:
    """Result of Bayesian aggregation for a single ticker."""

    ticker: str
    prior: float
    posterior: float
    signals_fired: List[str] = field(default_factory=list)
    conviction_level: str = "low"
    should_trade: bool = False

    def __post_init__(self) -> None:
        self.conviction_level = _compute_conviction(self.posterior)
        self.should_trade = self.posterior > 0.75


def _compute_conviction(posterior: float) -> str:
    """Map posterior probability to conviction level."""
    if posterior > 0.85:
        return "very_high"
    elif posterior > 0.75:
        return "high"
    elif posterior >= 0.6:
        return "medium"
    else:
        return "low"


def _probability_to_odds(p: float) -> float:
    """Convert a probability to odds."""
    if p >= 1.0:
        return float("inf")
    if p <= 0.0:
        return 0.0
    return p / (1.0 - p)


def _odds_to_probability(odds: float) -> float:
    """Convert odds back to a probability."""
    if odds == float("inf"):
        return 1.0
    if odds <= 0.0:
        return 0.0
    return odds / (1.0 + odds)


def aggregate(ticker: str, signals: List[SignalInput], prior: float = BASE_PRIOR) -> BayesianResult:
    """
    Perform iterative Bayesian updating for a single ticker.

    Starting from the prior probability, applies each active signal's
    effective likelihood ratio to update the posterior odds.

    Args:
        ticker: Stock ticker symbol.
        signals: List of SignalInput objects representing available signals.
        prior: Prior probability of outperformance (default 0.5).

    Returns:
        BayesianResult with posterior probability and metadata.
    """
    prior_odds = _probability_to_odds(prior)
    cumulative_odds = prior_odds
    signals_fired: List[str] = []

    for signal in signals:
        if signal.signal_active:
            effective_lr = signal.effective_lr
            cumulative_odds *= effective_lr
            signals_fired.append(signal.signal_name)

    posterior = _odds_to_probability(cumulative_odds)

    return BayesianResult(
        ticker=ticker,
        prior=prior,
        posterior=posterior,
        signals_fired=signals_fired,
    )


def batch_aggregate(
    tickers_signals: Dict[str, List[SignalInput]], prior: float = BASE_PRIOR
) -> List[BayesianResult]:
    """
    Process multiple tickers through Bayesian aggregation.

    Args:
        tickers_signals: Mapping of ticker -> list of SignalInput.
        prior: Prior probability (default 0.5).

    Returns:
        List of BayesianResult sorted by posterior descending.
    """
    results: List[BayesianResult] = []

    for ticker, signals in tickers_signals.items():
        result = aggregate(ticker, signals, prior=prior)
        results.append(result)

    results.sort(key=lambda r: r.posterior, reverse=True)
    return results


def signal_attribution(result: BayesianResult, signals: List[SignalInput]) -> List[Dict[str, float]]:
    """
    Compute the marginal contribution of each fired signal to the posterior.

    For each active signal, calculates what the posterior would be without it
    and reports the marginal lift that signal provides.

    Args:
        result: The BayesianResult from aggregate().
        signals: The same list of SignalInput used in aggregate().

    Returns:
        List of dicts with keys: signal_name, marginal_contribution,
        posterior_without, posterior_with_all.
    """
    active_signals = [s for s in signals if s.signal_active]
    attributions: List[Dict[str, float]] = []

    for target_signal in active_signals:
        # Compute posterior without this signal
        remaining_signals = [s for s in active_signals if s.signal_name != target_signal.signal_name]
        prior_odds = _probability_to_odds(result.prior)
        odds_without = prior_odds

        for s in remaining_signals:
            odds_without *= s.effective_lr

        posterior_without = _odds_to_probability(odds_without)
        marginal_contribution = result.posterior - posterior_without

        attributions.append({
            "signal_name": target_signal.signal_name,
            "likelihood_ratio": target_signal.likelihood_ratio,
            "effective_lr": target_signal.effective_lr,
            "marginal_contribution": round(marginal_contribution, 6),
            "posterior_without": round(posterior_without, 6),
            "posterior_with_all": round(result.posterior, 6),
        })

    # Sort by absolute marginal contribution descending
    attributions.sort(key=lambda a: abs(a["marginal_contribution"]), reverse=True)
    return attributions


def build_default_signals(
    momentum_active: bool = False,
    momentum_strength: float = 1.0,
    insider_active: bool = False,
    insider_strength: float = 1.0,
    sentiment_active: bool = False,
    sentiment_strength: float = 1.0,
    risk_factor_active: bool = False,
    risk_factor_strength: float = 1.0,
    hiring_surge_active: bool = False,
    hiring_surge_strength: float = 1.0,
) -> List[SignalInput]:
    """
    Convenience factory to build a standard set of signals with default LRs.

    Args:
        *_active: Whether each signal is firing.
        *_strength: Strength of each signal (0-1).

    Returns:
        List of SignalInput with the system's default likelihood ratios.
    """
    return [
        SignalInput(
            signal_name="momentum_q5",
            signal_active=momentum_active,
            signal_strength=momentum_strength,
            likelihood_ratio=DEFAULT_LIKELIHOOD_RATIOS["momentum_q5"],
        ),
        SignalInput(
            signal_name="insider_cluster_buying",
            signal_active=insider_active,
            signal_strength=insider_strength,
            likelihood_ratio=DEFAULT_LIKELIHOOD_RATIOS["insider_cluster_buying"],
        ),
        SignalInput(
            signal_name="sentiment_delta_positive",
            signal_active=sentiment_active,
            signal_strength=sentiment_strength,
            likelihood_ratio=DEFAULT_LIKELIHOOD_RATIOS["sentiment_delta_positive"],
        ),
        SignalInput(
            signal_name="risk_factor_change_negative",
            signal_active=risk_factor_active,
            signal_strength=risk_factor_strength,
            likelihood_ratio=DEFAULT_LIKELIHOOD_RATIOS["risk_factor_change_negative"],
        ),
        SignalInput(
            signal_name="alternative_data_hiring_surge",
            signal_active=hiring_surge_active,
            signal_strength=hiring_surge_strength,
            likelihood_ratio=DEFAULT_LIKELIHOOD_RATIOS["alternative_data_hiring_surge"],
        ),
    ]
