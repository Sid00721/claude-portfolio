"""
Scheduler — runs the daily pipeline at market-relevant times.

ASX market hours: 10:00 AM - 4:00 PM AEST
Run pipeline at 9:30 AM AEST (pre-market) for signal generation
Run stop-loss check every 15 minutes during market hours
"""

import schedule
import time
from datetime import datetime
import pytz

from main import run_daily_pipeline
from execution.broker import get_broker


AEST = pytz.timezone("Australia/Sydney")


def is_trading_day() -> bool:
    now = datetime.now(AEST)
    return now.weekday() < 5  # Mon-Fri


def check_stops():
    if not is_trading_day():
        return

    broker = get_broker()
    positions = broker.get_positions()

    for pos in positions:
        if pos.get("stop_price") and pos.get("current_price"):
            if pos["current_price"] <= pos["stop_price"]:
                print(f"STOP HIT: {pos['ticker']} @ ${pos['current_price']:.2f} "
                      f"(stop: ${pos['stop_price']:.2f})")
                broker.place_order(
                    ticker=pos["ticker"],
                    side="sell",
                    quantity=pos["shares"],
                )


def run():
    print("ASX Quant Scheduler started.")
    print("Pipeline runs daily at 09:30 AEST.")
    print("Stop checks every 15 minutes during market hours.\n")

    schedule.every().day.at("23:30").do(run_daily_pipeline)  # 09:30 AEST = 23:30 UTC (approx)
    schedule.every(15).minutes.do(check_stops)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    run()
