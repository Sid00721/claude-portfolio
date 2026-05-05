"""
Broker connector for IBKR (Interactive Brokers).

Provides a clean interface that can switch between paper and live trading
via the BROKER_MODE environment variable.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import yfinance as yf


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    ticker: str
    side: Side
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None


@dataclass
class Fill:
    ticker: str
    side: Side
    quantity: int
    fill_price: float
    timestamp: datetime
    commission: float


class BrokerInterface(ABC):
    """Abstract broker interface for order management."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the broker."""
        ...

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """
        Submit an order to the broker.

        Returns:
            Order ID as a string.
        """
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, int]:
        """
        Get current positions.

        Returns:
            Dict mapping ticker to quantity held.
        """
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value (cash + positions at market value).

        Returns:
            Portfolio value in AUD.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Returns:
            True if cancellation was successful.
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Check the status of an order.

        Returns:
            Current OrderStatus.
        """
        ...


class IBKRBroker(BrokerInterface):
    """
    Interactive Brokers implementation using ib_insync.

    Supports both paper and live trading depending on the port used.
    Paper trading default port: 7497
    Live trading default port: 7496
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None, client_id: int = 1):
        self._host = host
        self._port = port or (7496 if os.environ.get("BROKER_MODE") == "live" else 7497)
        self._client_id = client_id
        self._ib = None
        self._orders: dict[str, object] = {}

    def connect(self) -> None:
        from ib_insync import IB

        self._ib = IB()
        self._ib.connect(self._host, self._port, clientId=self._client_id)

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    def place_order(self, order: Order) -> str:
        from ib_insync import LimitOrder, MarketOrder, Stock

        if self._ib is None or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

        contract = Stock(order.ticker.replace(".AX", ""), "ASX", "AUD")
        self._ib.qualifyContracts(contract)

        action = "BUY" if order.side == Side.BUY else "SELL"

        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("Limit price required for limit orders")
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        trade = self._ib.placeOrder(contract, ib_order)
        order_id = str(trade.order.orderId)
        self._orders[order_id] = trade
        return order_id

    def get_positions(self) -> dict[str, int]:
        if self._ib is None or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

        positions = {}
        for pos in self._ib.positions():
            ticker = f"{pos.contract.symbol}.AX"
            positions[ticker] = int(pos.position)
        return positions

    def get_portfolio_value(self) -> float:
        if self._ib is None or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

        account_values = self._ib.accountSummary()
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "AUD":
                return float(av.value)
        return 0.0

    def cancel_order(self, order_id: str) -> bool:
        if self._ib is None or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

        trade = self._orders.get(order_id)
        if trade is None:
            return False

        self._ib.cancelOrder(trade.order)
        return True

    def get_order_status(self, order_id: str) -> OrderStatus:
        if self._ib is None or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

        trade = self._orders.get(order_id)
        if trade is None:
            return OrderStatus.REJECTED

        status = trade.orderStatus.status.lower()
        if status in ("filled",):
            return OrderStatus.FILLED
        elif status in ("cancelled", "inactive"):
            return OrderStatus.CANCELLED
        elif status in ("submitted", "presubmitted", "pendingsubmit"):
            return OrderStatus.PENDING
        else:
            return OrderStatus.PENDING


@dataclass
class PaperPosition:
    ticker: str
    quantity: int
    avg_cost: float


class PaperBroker(BrokerInterface):
    """
    Paper trading broker that simulates fills for testing without an API connection.

    Persists state to SQLite so positions survive between runs.
    """

    def __init__(self, initial_cash: float = 100_000.0, commission_per_trade: float = 0.0):
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._order_statuses: dict[str, OrderStatus] = {}
        self._next_order_id: int = 1
        self._commission: float = commission_per_trade
        self._initial_cash = initial_cash
        self._connected: bool = False
        self._init_db()

    def _init_db(self):
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "quant.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS paper_state (
            key TEXT PRIMARY KEY, value TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS paper_positions (
            ticker TEXT PRIMARY KEY, quantity INTEGER, avg_cost REAL
        )""")
        # Initialize cash if first run
        row = conn.execute("SELECT value FROM paper_state WHERE key='cash'").fetchone()
        if row is None:
            conn.execute("INSERT INTO paper_state (key, value) VALUES ('cash', ?)", (str(self._initial_cash),))
            conn.commit()
        conn.close()

    def _get_conn(self):
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "quant.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def _cash(self) -> float:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM paper_state WHERE key='cash'").fetchone()
        conn.close()
        return float(row["value"]) if row else self._initial_cash

    @_cash.setter
    def _cash(self, val: float):
        conn = self._get_conn()
        conn.execute("INSERT OR REPLACE INTO paper_state (key, value) VALUES ('cash', ?)", (str(val),))
        conn.commit()
        conn.close()

    @property
    def _positions(self) -> dict[str, "PaperPosition"]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM paper_positions").fetchall()
        conn.close()
        return {r["ticker"]: PaperPosition(r["ticker"], r["quantity"], r["avg_cost"]) for r in rows}

    def _save_position(self, pos: "PaperPosition"):
        conn = self._get_conn()
        if pos.quantity <= 0:
            conn.execute("DELETE FROM paper_positions WHERE ticker=?", (pos.ticker,))
        else:
            conn.execute("INSERT OR REPLACE INTO paper_positions (ticker, quantity, avg_cost) VALUES (?, ?, ?)",
                         (pos.ticker, pos.quantity, pos.avg_cost))
        conn.commit()
        conn.close()

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _get_current_price(self, ticker: str) -> float:
        """Fetch current market price via yfinance."""
        data = yf.download(ticker, period="1d", progress=False)
        if data.empty:
            raise ValueError(f"Could not fetch price for {ticker}")
        return float(data["Close"].iloc[-1].item())

    def place_order(self, order: Order) -> str:
        if not self._connected:
            raise ConnectionError("Paper broker not connected. Call connect() first.")

        order_id = str(self._next_order_id)
        self._next_order_id += 1
        self._orders[order_id] = order

        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            price = self._get_current_price(order.ticker)
            self._execute_fill(order_id, order, price)
        elif order.order_type == OrderType.LIMIT:
            # For limit orders, check if current price satisfies limit
            price = self._get_current_price(order.ticker)
            if order.side == Side.BUY and price <= (order.limit_price or float("inf")):
                self._execute_fill(order_id, order, order.limit_price or price)
            elif order.side == Side.SELL and price >= (order.limit_price or 0):
                self._execute_fill(order_id, order, order.limit_price or price)
            else:
                self._order_statuses[order_id] = OrderStatus.PENDING

        return order_id

    def _execute_fill(self, order_id: str, order: Order, fill_price: float) -> None:
        """Execute a simulated fill and persist state."""
        fill = Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            timestamp=datetime.now(),
            commission=self._commission,
        )
        self._fills.append(fill)
        self._order_statuses[order_id] = OrderStatus.FILLED

        positions = self._positions

        if order.side == Side.BUY:
            cost = fill_price * order.quantity + self._commission
            self._cash = self._cash - cost

            if order.ticker in positions:
                pos = positions[order.ticker]
                total_cost = pos.avg_cost * pos.quantity + fill_price * order.quantity
                new_qty = pos.quantity + order.quantity
                new_avg = total_cost / new_qty
                updated = PaperPosition(order.ticker, new_qty, new_avg)
            else:
                updated = PaperPosition(order.ticker, order.quantity, fill_price)
            self._save_position(updated)

        elif order.side == Side.SELL:
            proceeds = fill_price * order.quantity - self._commission
            self._cash = self._cash + proceeds

            if order.ticker in positions:
                pos = positions[order.ticker]
                new_qty = pos.quantity - order.quantity
                updated = PaperPosition(order.ticker, max(new_qty, 0), pos.avg_cost)
                self._save_position(updated)

    def get_positions(self) -> dict[str, int]:
        if not self._connected:
            raise ConnectionError("Paper broker not connected. Call connect() first.")
        return {ticker: pos.quantity for ticker, pos in self._positions.items()}

    def get_portfolio_value(self) -> float:
        if not self._connected:
            raise ConnectionError("Paper broker not connected. Call connect() first.")

        total = self._cash
        for ticker, pos in self._positions.items():
            try:
                price = self._get_current_price(ticker)
                total += price * pos.quantity
            except ValueError:
                # If price fetch fails, use avg cost as fallback
                total += pos.avg_cost * pos.quantity
        return total

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            raise ConnectionError("Paper broker not connected. Call connect() first.")

        if order_id in self._order_statuses:
            if self._order_statuses[order_id] == OrderStatus.PENDING:
                self._order_statuses[order_id] = OrderStatus.CANCELLED
                return True
        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        if not self._connected:
            raise ConnectionError("Paper broker not connected. Call connect() first.")

        return self._order_statuses.get(order_id, OrderStatus.REJECTED)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)


def get_broker() -> BrokerInterface:
    """
    Factory function that returns the appropriate broker based on environment.

    Set BROKER_MODE=live for live trading via IBKR.
    Defaults to paper trading.
    """
    mode = os.environ.get("BROKER_MODE", "paper").lower()

    if mode == "live":
        return IBKRBroker()
    else:
        initial = float(os.environ.get("INITIAL_CAPITAL", "100000"))
        return PaperBroker(initial_cash=initial, commission_per_trade=0.0)
