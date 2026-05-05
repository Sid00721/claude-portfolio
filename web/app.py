"""
The Claude Portfolio — Public Web App

Fully autonomous ASX small cap paper trading system.
Zero manual intervention. Public performance tracking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
import json
from datetime import datetime

from data.db import get_db
from data.fund import (
    register_investor, authenticate_investor, deposit, withdraw,
    get_investor_portfolio, get_fund_overview, get_nav_per_share,
)


def run_pipeline_job():
    """Run the daily pipeline. Called by scheduler."""
    try:
        from main import run_daily_pipeline
        run_daily_pipeline()
        _snapshot_portfolio()
    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")


def _snapshot_portfolio():
    """Record daily portfolio value."""
    with get_db() as conn:
        # Get cash
        row = conn.execute("SELECT value FROM paper_state WHERE key='cash'").fetchone()
        cash = float(row["value"]) if row else 500.0

        # Get positions value (use last known price from universe table)
        positions = conn.execute("SELECT * FROM paper_positions").fetchall()
        positions_value = 0.0
        for pos in positions:
            price_row = conn.execute(
                "SELECT price FROM universe WHERE ticker=?", (pos["ticker"],)
            ).fetchone()
            if price_row:
                positions_value += pos["quantity"] * price_row["price"]
            else:
                positions_value += pos["quantity"] * pos["avg_cost"]

        nav = cash + positions_value
        today = datetime.now().strftime("%Y-%m-%d")

        # Get previous snapshot for return calc
        prev = conn.execute(
            "SELECT nav FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
        ).fetchone()
        prev_nav = float(prev["nav"]) if prev else 500.0
        daily_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0

        # Cumulative return from initial
        cumulative_return = (nav - 500.0) / 500.0

        # Drawdown
        peak_row = conn.execute("SELECT MAX(nav) as peak FROM portfolio_snapshots").fetchone()
        peak = float(peak_row["peak"]) if peak_row and peak_row["peak"] else nav
        peak = max(peak, nav)
        drawdown = (nav - peak) / peak if peak > 0 else 0.0

        num_positions = len(positions)

        conn.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
            (date, nav, cash, positions_value, num_positions, daily_return, cumulative_return, drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (today, nav, cash, positions_value, num_positions, daily_return, cumulative_return, drawdown))


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    # Run daily at 00:00 UTC (10:00 AEST)
    scheduler.add_job(run_pipeline_job, "cron", hour=0, minute=0)
    scheduler.start()
    print("[SCHEDULER] Daily pipeline scheduled at 00:00 UTC")
    yield
    scheduler.shutdown()


app = FastAPI(title="The Claude Portfolio", lifespan=lifespan)

templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")

templates = Jinja2Templates(directory=templates_dir)
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/portfolio")
async def api_portfolio():
    with get_db() as conn:
        snapshots = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY date"
        ).fetchall()

        row = conn.execute("SELECT value FROM paper_state WHERE key='cash'").fetchone()
        cash = float(row["value"]) if row else 500.0

        positions = conn.execute("SELECT * FROM paper_positions").fetchall()
        positions_value = 0.0
        pos_list = []
        for pos in positions:
            price_row = conn.execute(
                "SELECT price FROM universe WHERE ticker=?", (pos["ticker"],)
            ).fetchone()
            price = price_row["price"] if price_row else pos["avg_cost"]
            market_val = pos["quantity"] * price
            pnl = (price - pos["avg_cost"]) * pos["quantity"]
            positions_value += market_val
            pos_list.append({
                "ticker": pos["ticker"],
                "quantity": pos["quantity"],
                "avg_cost": round(pos["avg_cost"], 4),
                "current_price": round(price, 4),
                "market_value": round(market_val, 2),
                "pnl": round(pnl, 2),
                "return_pct": round((price - pos["avg_cost"]) / pos["avg_cost"] * 100, 1) if pos["avg_cost"] > 0 else 0,
            })

    nav = cash + positions_value
    total_return = (nav - 500.0) / 500.0

    return {
        "nav": round(nav, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_return_pct": round(total_return * 100, 2),
        "initial_capital": 500.0,
        "positions": pos_list,
        "equity_curve": [{"date": dict(s)["date"], "nav": dict(s)["nav"]} for s in snapshots],
    }


@app.get("/api/trades")
async def api_trades():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY entry_date DESC LIMIT 50"
        ).fetchall()

    trades = []
    for r in rows:
        t = dict(r)
        t["signals_at_entry"] = json.loads(t.get("signals_at_entry", "{}"))
        trades.append(t)
    return {"trades": trades}


@app.get("/api/signals")
async def api_signals():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM signal_states ORDER BY date DESC, ticker LIMIT 100"
        ).fetchall()
    return {"signals": [dict(r) for r in rows]}


@app.get("/api/universe")
async def api_universe():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM universe ORDER BY return_12m DESC"
        ).fetchall()
    return {"stocks": [dict(r) for r in rows], "count": len(rows)}


# ─── Fund API ──────────────────────────────────────────────────────────────────

@app.get("/api/fund")
async def api_fund():
    """Public fund overview."""
    return get_fund_overview()


@app.post("/api/fund/register")
async def api_register(request: Request):
    body = await request.json()
    email = body.get("email", "").strip()
    password = body.get("password", "")
    name = body.get("name", "")

    if not email or not password:
        return {"error": "Email and password required"}, 400

    try:
        investor_id = register_investor(email, password, name)
        return {"investor_id": investor_id, "message": "Account created"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/fund/login")
async def api_login(request: Request):
    body = await request.json()
    email = body.get("email", "")
    password = body.get("password", "")

    investor_id = authenticate_investor(email, password)
    if investor_id is None:
        return {"error": "Invalid credentials"}

    import jwt
    token = jwt.encode(
        {"investor_id": investor_id, "email": email},
        os.environ.get("JWT_SECRET", "claude-portfolio-dev-secret"),
        algorithm="HS256",
    )
    return {"token": token, "investor_id": investor_id}


def _get_investor_from_token(request: Request) -> int | None:
    import jwt
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        payload = jwt.decode(
            auth[7:],
            os.environ.get("JWT_SECRET", "claude-portfolio-dev-secret"),
            algorithms=["HS256"],
        )
        return payload["investor_id"]
    except Exception:
        return None


@app.post("/api/fund/deposit")
async def api_deposit(request: Request):
    investor_id = _get_investor_from_token(request)
    if not investor_id:
        return {"error": "Unauthorized"}

    body = await request.json()
    amount = float(body.get("amount", 0))

    try:
        result = deposit(investor_id, amount)
        return result
    except ValueError as e:
        return {"error": str(e)}


@app.post("/api/fund/withdraw")
async def api_withdraw(request: Request):
    investor_id = _get_investor_from_token(request)
    if not investor_id:
        return {"error": "Unauthorized"}

    body = await request.json()
    amount = float(body.get("amount", 0))

    try:
        result = withdraw(investor_id, amount)
        return result
    except ValueError as e:
        return {"error": str(e)}


@app.get("/api/fund/me")
async def api_my_portfolio(request: Request):
    investor_id = _get_investor_from_token(request)
    if not investor_id:
        return {"error": "Unauthorized"}
    return get_investor_portfolio(investor_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
