from dataclasses import dataclass, field
from typing import Dict
import time

import yfinance as yf
import pandas as pd


def _yf_fetch_with_retry(ticker_str: str, period: str = "1d", max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker_str)
            hist = t.history(period=period)
            if not hist.empty:
                return hist
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return pd.DataFrame()


SECTOR_ETFS = {
    "technology": "XLK",
    "financials": "XLF",
    "energy": "XLE",
    "healthcare": "XLV",
    "industrials": "XLI",
    "communication": "XLC",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "materials": "XLB",
}


@dataclass
class RegimeState:
    vix_level: float
    vix_regime: str
    yield_spread: float
    yield_signal: str
    sector_flows: Dict[str, float] = field(default_factory=dict)
    overall_regime: str = ""
    position_scalar: float = 1.0


def _fetch_vix() -> float:
    hist = _yf_fetch_with_retry("^VIX", period="5d")
    if hist.empty:
        return 20.0  # default to selective regime if data unavailable
    return float(hist["Close"].iloc[-1])


def _classify_vix_regime(vix_level: float) -> tuple[str, float]:
    if vix_level < 18:
        return "risk_on", 1.0
    elif vix_level <= 28:
        return "selective", 0.5
    else:
        return "risk_off", 0.0


def _fetch_yield_spread() -> tuple[float, str]:
    tnx = _yf_fetch_with_retry("^TNX", period="5d")
    irx = _yf_fetch_with_retry("^IRX", period="5d")

    if tnx.empty or irx.empty:
        return 0.5, "bullish"  # default to mildly bullish if unavailable

    yield_10y = float(tnx["Close"].iloc[-1])
    yield_short = float(irx["Close"].iloc[-1])

    spread = yield_10y - yield_short

    signal = "bullish" if spread > 0 else "bearish"
    return spread, signal


def _compute_sector_flows() -> Dict[str, float]:
    tickers = list(SECTOR_ETFS.values())
    data = yf.download(tickers, period="30d", group_by="ticker", progress=False)

    flows: Dict[str, float] = {}

    for sector, ticker in SECTOR_ETFS.items():
        try:
            volume = data[ticker]["Volume"]
            recent_20d = volume.iloc[-20:]
            prior_10d = volume.iloc[-30:-20]

            avg_recent = recent_20d.mean()
            avg_prior = prior_10d.mean()

            if avg_prior > 0:
                pct_change = ((avg_recent - avg_prior) / avg_prior) * 100
            else:
                pct_change = 0.0

            flows[sector] = round(pct_change, 2)
        except (KeyError, IndexError):
            flows[sector] = 0.0

    return flows


def get_regime() -> RegimeState:
    vix_level = _fetch_vix()
    vix_regime, base_scalar = _classify_vix_regime(vix_level)

    yield_spread, yield_signal = _fetch_yield_spread()

    sector_flows = _compute_sector_flows()

    position_scalar = base_scalar
    if yield_signal == "bearish":
        position_scalar *= 0.5

    overall_regime = vix_regime
    if yield_signal == "bearish" and vix_regime != "risk_off":
        overall_regime = "selective"

    return RegimeState(
        vix_level=round(vix_level, 2),
        vix_regime=vix_regime,
        yield_spread=round(yield_spread, 4),
        yield_signal=yield_signal,
        sector_flows=sector_flows,
        overall_regime=overall_regime,
        position_scalar=position_scalar,
    )


if __name__ == "__main__":
    regime = get_regime()
    print(f"VIX: {regime.vix_level} -> {regime.vix_regime}")
    print(f"Yield Spread: {regime.yield_spread} -> {regime.yield_signal}")
    print(f"Overall: {regime.overall_regime} | Position Scalar: {regime.position_scalar}")
    print(f"Sector Flows:")
    for sector, flow in sorted(regime.sector_flows.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sector}: {flow:+.2f}%")
