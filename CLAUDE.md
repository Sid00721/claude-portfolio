# ASX Quant System

## Goal
Build a systematic trading system targeting ASX small caps ($100M-$1B AUD market cap).
15-20% annual return target. Sharpe > 1.

## Stack
- Python 3.11+
- yfinance for price data
- ASX Appendix 3Y filings for insider trades (scraped from ASX announcements)
- Anthropic API (claude-sonnet-4-20250514) for LLM signals (transcript sentiment, 10-K diffs)
- IBKR for ASX execution (paper then live)
- Supabase for logging all trades, signal states, and attribution data
- Next.js dashboard for public track record

## Architecture
Universe Filter → Regime Detection → Signal Stack →
Bayesian Aggregator → Kelly Sizer → Execution → Attribution

## Signal Stack
1. Momentum: 12-1 month return (skip recent month)
2. Insider Cluster Buying: ASX Appendix 3Y scoring
3. Earnings Transcript Sentiment Delta: LLM quarter-over-quarter comparison
4. Annual Report Risk Factor Changes: Semantic diff on risk sections
5. Alternative Data: Job postings, Google Trends (free tier)

## Regime Detection
- VIX < 18: Momentum regime, trend following, risk on
- VIX 18-28: Selective, only highest conviction signals
- VIX > 28: Cash or inverse, no new longs
- Yield curve steepening = risk on, inversion = reduce exposure

## Position Rules
- Only trade when posterior probability > 75% after Bayesian update
- Never size more than 15% of portfolio in one position
- Use half-Kelly (50% of full Kelly) for position sizing
- Stop loss: 2x ATR below entry
- No holding through earnings unless earnings sentiment is specifically the thesis

## Logging
Every signal state, trade entry, exit, and outcome goes to Supabase.
This data feeds signal attribution — which signals actually generate alpha.

## Build Order
1. Universe screener — filter ASX by market cap + low analyst coverage
2. Regime detection module
3. ASX Appendix 3Y monitor — insider buy scoring agent
4. Earnings transcript sentiment delta pipeline
5. Bayesian signal aggregator
6. Kelly criterion position sizer
7. Backtest engine
8. Paper trading live
9. Public dashboard
