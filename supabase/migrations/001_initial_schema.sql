-- ASX Quant System — Database Schema

-- Universe: screened stocks that pass our filters
CREATE TABLE IF NOT EXISTS universe (
    ticker TEXT PRIMARY KEY,
    market_cap BIGINT NOT NULL,
    avg_volume BIGINT,
    sector TEXT,
    return_12m REAL,
    price REAL,
    screened_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_universe_market_cap ON universe(market_cap);
CREATE INDEX idx_universe_sector ON universe(sector);

-- Trades: every entry and exit with full context
CREATE TABLE IF NOT EXISTS trades (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker TEXT NOT NULL,
    entry_date TIMESTAMPTZ NOT NULL,
    exit_date TIMESTAMPTZ,
    entry_price REAL NOT NULL,
    exit_price REAL,
    position_size REAL NOT NULL,
    shares INTEGER,
    pnl REAL,
    return_pct REAL,
    signals_at_entry JSONB NOT NULL DEFAULT '{}',
    posterior_at_entry REAL NOT NULL,
    regime_at_entry TEXT,
    kelly_size_pct REAL,
    stop_price REAL,
    status TEXT DEFAULT 'open' CHECK (status IN ('open', 'closed', 'stopped_out')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trades_ticker ON trades(ticker);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_entry_date ON trades(entry_date);

-- Signal states: daily snapshot of all signals for each stock
CREATE TABLE IF NOT EXISTS signal_states (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    momentum_active BOOLEAN,
    momentum_strength REAL,
    insider_active BOOLEAN,
    insider_strength REAL,
    sentiment_active BOOLEAN,
    sentiment_strength REAL,
    risk_factor_active BOOLEAN,
    risk_factor_strength REAL,
    alt_data_active BOOLEAN,
    alt_data_strength REAL,
    posterior REAL,
    regime TEXT,
    raw_signals JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE INDEX idx_signal_states_date ON signal_states(date);
CREATE INDEX idx_signal_states_ticker ON signal_states(ticker);

-- Regime history: track regime changes over time
CREATE TABLE IF NOT EXISTS regime_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    vix_level REAL,
    vix_regime TEXT,
    yield_spread REAL,
    yield_signal TEXT,
    overall_regime TEXT,
    position_scalar REAL,
    sector_flows JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Signal attribution: aggregated performance by signal
CREATE TABLE IF NOT EXISTS signal_attribution (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    signal_name TEXT NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    times_fired INTEGER DEFAULT 0,
    win_rate REAL,
    avg_return_when_fired REAL,
    contribution_to_pnl REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(signal_name, period_start, period_end)
);

-- Portfolio snapshots: daily NAV tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    nav REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    num_positions INTEGER,
    daily_return REAL,
    cumulative_return REAL,
    drawdown REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_portfolio_snapshots_date ON portfolio_snapshots(date);
