"""
Regime Detection — Australian Market Indicators

Uses ASX VIX (XVI), Australian yield curve, and ASX sector ETFs.
Falls back to US indicators if Australian data unavailable.
"""

from dataclasses import dataclass, field
from typing import Dict

from data.provider import get_asx_vix, get_au_yield_spread


@dataclass
class RegimeState:
    vix_level: float = 20.0
    vix_regime: str = "selective"
    yield_spread: float = 0.0
    yield_signal: str = "bullish"
    sector_flows: Dict[str, float] = field(default_factory=dict)
    overall_regime: str = "selective"
    position_scalar: float = 0.5


def _classify_vix_regime(vix_level: float) -> tuple[str, float]:
    if vix_level < 18:
        return "risk_on", 1.0
    elif vix_level <= 28:
        return "selective", 0.5
    else:
        return "risk_off", 0.0


def get_regime() -> RegimeState:
    vix_level = get_asx_vix()
    vix_regime, base_scalar = _classify_vix_regime(vix_level)

    yield_spread, yield_signal = get_au_yield_spread()

    # Halve position scalar if yield curve inverted
    position_scalar = base_scalar
    if yield_signal == "bearish":
        position_scalar *= 0.5

    overall_regime = vix_regime
    if yield_signal == "bearish" and vix_regime == "risk_on":
        overall_regime = "selective"

    return RegimeState(
        vix_level=vix_level,
        vix_regime=vix_regime,
        yield_spread=yield_spread,
        yield_signal=yield_signal,
        sector_flows={},
        overall_regime=overall_regime,
        position_scalar=position_scalar,
    )
