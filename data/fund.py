"""
Virtual Fund — Unit Trust Model

Everyone owns "shares" in the fund. When you deposit, you buy shares at the
current NAV/share. When you withdraw, you sell shares at current NAV/share.
Your return tracks the fund's performance proportionally.

This is exactly how real managed funds work (unit pricing).
"""

from datetime import datetime
import hashlib
import secrets

from data.db import get_db


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${h}"


def _verify_password(password: str, stored: str) -> bool:
    salt, h = stored.split("$", 1)
    return hashlib.sha256((salt + password).encode()).hexdigest() == h


INITIAL_NAV_PER_SHARE = 1.00


def get_fund_nav() -> float:
    """Get current total fund NAV (portfolio value)."""
    with get_db() as conn:
        row = conn.execute("SELECT value FROM paper_state WHERE key='cash'").fetchone()
        cash = float(row["value"]) if row else 0.0

        positions = conn.execute("SELECT * FROM paper_positions").fetchall()
        positions_value = 0.0
        for pos in positions:
            price_row = conn.execute(
                "SELECT price FROM universe WHERE ticker=?", (pos["ticker"],)
            ).fetchone()
            price = price_row["price"] if price_row else pos["avg_cost"]
            positions_value += pos["quantity"] * price

    return cash + positions_value


def get_total_shares() -> float:
    """Total shares outstanding (seed + investor shares)."""
    with get_db() as conn:
        investor_shares = conn.execute("SELECT COALESCE(SUM(shares), 0) as total FROM fund_shares").fetchone()
        seed = conn.execute("SELECT value FROM fund_state WHERE key='seed_shares'").fetchone()
    seed_shares = float(seed["value"]) if seed else 500.0
    return seed_shares + float(investor_shares["total"])


def get_nav_per_share() -> float:
    """Current price per share."""
    total_shares = get_total_shares()
    if total_shares <= 0:
        return INITIAL_NAV_PER_SHARE
    return get_fund_nav() / total_shares


def register_investor(email: str, password: str, name: str = "") -> int:
    """Create a new investor account. Returns investor ID."""
    password_hash = _hash_password(password)
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO investors (email, password_hash, name) VALUES (?, ?, ?)",
            (email, password_hash, name),
        )
        investor_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO fund_shares (investor_id, shares) VALUES (?, 0)",
            (investor_id,),
        )
    return investor_id


def authenticate_investor(email: str, password: str) -> int | None:
    """Verify credentials. Returns investor ID or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM investors WHERE email=?", (email,)
        ).fetchone()
    if row and _verify_password(password, row["password_hash"]):
        return row["id"]
    return None


MAX_DEPOSIT = 500.0


def deposit(investor_id: int, amount: float) -> dict:
    """
    Deposit virtual money into the fund.
    Issues shares at current NAV/share.
    """
    if amount <= 0:
        raise ValueError("Deposit must be positive")
    if amount > MAX_DEPOSIT:
        raise ValueError(f"Maximum deposit is ${MAX_DEPOSIT:.0f}")

    # Check total deposited by this investor
    with get_db() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) as total FROM fund_transactions WHERE investor_id=? AND type='deposit'",
            (investor_id,),
        ).fetchone()
        withs = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) as total FROM fund_transactions WHERE investor_id=? AND type='withdrawal'",
            (investor_id,),
        ).fetchone()
    net_deposited = float(row["total"]) - float(withs["total"])
    if net_deposited + amount > MAX_DEPOSIT:
        remaining = MAX_DEPOSIT - net_deposited
        raise ValueError(f"Max $500 per investor. You can deposit up to ${remaining:.2f} more.")

    nav_per_share = get_nav_per_share()
    shares_issued = amount / nav_per_share

    with get_db() as conn:
        # Issue shares to investor
        conn.execute(
            "UPDATE fund_shares SET shares = shares + ? WHERE investor_id = ?",
            (shares_issued, investor_id),
        )
        # Add cash to the fund's paper account
        conn.execute(
            "UPDATE paper_state SET value = CAST(CAST(value AS REAL) + ? AS TEXT) WHERE key='cash'",
            (amount,),
        )
        # Log transaction
        conn.execute(
            """INSERT INTO fund_transactions (investor_id, type, amount, shares_issued, nav_per_share)
               VALUES (?, 'deposit', ?, ?, ?)""",
            (investor_id, amount, shares_issued, nav_per_share),
        )

    return {
        "amount": amount,
        "shares_issued": round(shares_issued, 6),
        "nav_per_share": round(nav_per_share, 6),
        "message": f"Deposited ${amount:.2f}, received {shares_issued:.4f} shares @ ${nav_per_share:.4f}/share",
    }


def withdraw(investor_id: int, amount: float) -> dict:
    """
    Withdraw virtual money from the fund.
    Redeems shares at current NAV/share.
    """
    if amount <= 0:
        raise ValueError("Withdrawal must be positive")

    nav_per_share = get_nav_per_share()
    shares_to_redeem = amount / nav_per_share

    with get_db() as conn:
        # Check investor has enough shares
        row = conn.execute(
            "SELECT shares FROM fund_shares WHERE investor_id=?", (investor_id,)
        ).fetchone()
        current_shares = float(row["shares"]) if row else 0.0

        if shares_to_redeem > current_shares:
            max_withdrawal = current_shares * nav_per_share
            raise ValueError(f"Insufficient shares. Max withdrawal: ${max_withdrawal:.2f}")

        # Redeem shares
        conn.execute(
            "UPDATE fund_shares SET shares = shares - ? WHERE investor_id = ?",
            (shares_to_redeem, investor_id),
        )
        # Remove cash from the fund
        conn.execute(
            "UPDATE paper_state SET value = CAST(CAST(value AS REAL) - ? AS TEXT) WHERE key='cash'",
            (amount,),
        )
        # Log transaction
        conn.execute(
            """INSERT INTO fund_transactions (investor_id, type, amount, shares_issued, nav_per_share)
               VALUES (?, 'withdrawal', ?, ?, ?)""",
            (investor_id, amount, shares_to_redeem, nav_per_share),
        )

    return {
        "amount": amount,
        "shares_redeemed": round(shares_to_redeem, 6),
        "nav_per_share": round(nav_per_share, 6),
        "message": f"Withdrew ${amount:.2f}, redeemed {shares_to_redeem:.4f} shares @ ${nav_per_share:.4f}/share",
    }


def get_investor_portfolio(investor_id: int) -> dict:
    """Get an investor's current position in the fund."""
    nav_per_share = get_nav_per_share()

    with get_db() as conn:
        row = conn.execute(
            "SELECT shares FROM fund_shares WHERE investor_id=?", (investor_id,)
        ).fetchone()
        shares = float(row["shares"]) if row else 0.0

        # Get total deposited (sum of deposits minus withdrawals)
        deps = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) as total FROM fund_transactions WHERE investor_id=? AND type='deposit'",
            (investor_id,),
        ).fetchone()
        withs = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) as total FROM fund_transactions WHERE investor_id=? AND type='withdrawal'",
            (investor_id,),
        ).fetchone()
        net_deposited = float(deps["total"]) - float(withs["total"])

        # Recent transactions
        txns = conn.execute(
            "SELECT * FROM fund_transactions WHERE investor_id=? ORDER BY created_at DESC LIMIT 10",
            (investor_id,),
        ).fetchall()

    current_value = shares * nav_per_share
    pnl = current_value - net_deposited
    return_pct = (pnl / net_deposited * 100) if net_deposited > 0 else 0.0

    return {
        "shares": round(shares, 6),
        "nav_per_share": round(nav_per_share, 6),
        "current_value": round(current_value, 2),
        "net_deposited": round(net_deposited, 2),
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
        "transactions": [dict(t) for t in txns],
    }


def get_fund_overview() -> dict:
    """Public fund stats."""
    nav = get_fund_nav()
    total_shares = get_total_shares()
    nav_per_share = get_nav_per_share()

    with get_db() as conn:
        num_investors = conn.execute("SELECT COUNT(*) as c FROM investors").fetchone()["c"]
        # Fund inception return
        first_nav = conn.execute(
            "SELECT nav FROM portfolio_snapshots ORDER BY date ASC LIMIT 1"
        ).fetchone()

    inception_nav = float(first_nav["nav"]) if first_nav else nav
    fund_return = ((nav - inception_nav) / inception_nav * 100) if inception_nav > 0 else 0.0

    return {
        "total_nav": round(nav, 2),
        "nav_per_share": round(nav_per_share, 6),
        "total_shares": round(total_shares, 4),
        "num_investors": num_investors,
        "fund_return_pct": round(fund_return, 2),
    }
