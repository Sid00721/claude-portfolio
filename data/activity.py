"""
Activity Feed — logs every action the system takes so users can watch it think.
"""

import json
from datetime import datetime
from data.db import get_db


def log_activity(category: str, title: str, detail: str = "", ticker: str = "", severity: str = "info"):
    """
    Log a system activity.

    Categories: scan, signal, regime, trade, news, error
    Severity: info, success, warning, alert
    """
    with get_db() as conn:
        conn.execute(
            """INSERT INTO activity_feed (category, title, detail, ticker, severity, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (category, title, detail, ticker, severity, datetime.now().isoformat()),
        )


def get_recent_activity(limit: int = 50) -> list[dict]:
    """Get recent activity feed entries."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_feed ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_activity_since(since_iso: str, limit: int = 20) -> list[dict]:
    """Get activity since a specific timestamp (for polling)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_feed WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
            (since_iso, limit),
        ).fetchall()
    return [dict(r) for r in rows]
