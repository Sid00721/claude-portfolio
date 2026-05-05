"""
Local SQLite database for the quant system.
Single file, zero config, no server needed.
"""

import sqlite3
import os
import json
from contextlib import contextmanager

_data_dir = os.environ.get("DATA_DIR", os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(_data_dir, "quant.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS universe (
                ticker TEXT PRIMARY KEY,
                market_cap INTEGER NOT NULL,
                avg_volume INTEGER,
                sector TEXT,
                return_12m REAL,
                price REAL,
                screened_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                exit_date TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                position_size REAL NOT NULL,
                shares INTEGER,
                pnl REAL,
                return_pct REAL,
                signals_at_entry TEXT NOT NULL DEFAULT '{}',
                posterior_at_entry REAL NOT NULL,
                regime_at_entry TEXT,
                kelly_size_pct REAL,
                stop_price REAL,
                status TEXT DEFAULT 'open' CHECK (status IN ('open', 'closed', 'stopped_out')),
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS signal_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                momentum_active INTEGER,
                momentum_strength REAL,
                insider_active INTEGER,
                insider_strength REAL,
                sentiment_active INTEGER,
                sentiment_strength REAL,
                risk_factor_active INTEGER,
                risk_factor_strength REAL,
                alt_data_active INTEGER,
                alt_data_strength REAL,
                posterior REAL,
                regime TEXT,
                raw_signals TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, date)
            );

            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                vix_level REAL,
                vix_regime TEXT,
                yield_spread REAL,
                yield_signal TEXT,
                overall_regime TEXT,
                position_scalar REAL,
                sector_flows TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                nav REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                num_positions INTEGER,
                daily_return REAL,
                cumulative_return REAL,
                drawdown REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS investors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS fund_shares (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                investor_id INTEGER NOT NULL REFERENCES investors(id),
                shares REAL NOT NULL DEFAULT 0,
                UNIQUE(investor_id)
            );

            CREATE TABLE IF NOT EXISTS fund_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                investor_id INTEGER NOT NULL REFERENCES investors(id),
                type TEXT NOT NULL CHECK (type IN ('deposit', 'withdrawal')),
                amount REAL NOT NULL,
                shares_issued REAL NOT NULL,
                nav_per_share REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS fund_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS activity_feed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                detail TEXT DEFAULT '',
                ticker TEXT DEFAULT '',
                severity TEXT DEFAULT 'info',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_feed(created_at);
            CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_signal_states_date ON signal_states(date);
            CREATE INDEX IF NOT EXISTS idx_signal_states_ticker ON signal_states(ticker);
            CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio_snapshots(date);
            CREATE INDEX IF NOT EXISTS idx_fund_transactions_investor ON fund_transactions(investor_id);
        """)


def ensure_fund_state():
    """Ensure fund seed state exists."""
    with get_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS paper_state (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS paper_positions (ticker TEXT PRIMARY KEY, quantity INTEGER, avg_cost REAL)")
        row = conn.execute("SELECT value FROM paper_state WHERE key='cash'").fetchone()
        if row is None:
            conn.execute("INSERT INTO paper_state (key, value) VALUES ('cash', '500.0')")
        seed = conn.execute("SELECT value FROM fund_state WHERE key='seed_shares'").fetchone()
        if seed is None:
            conn.execute("INSERT INTO fund_state (key, value) VALUES ('seed_shares', '500.0')")


init_db()
ensure_fund_state()
