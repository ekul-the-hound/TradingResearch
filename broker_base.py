"""
broker_base.py
==============

The broker abstraction the live stack is built on: a `BaseBroker` interface and
the plain data records that flow across it (`BrokerTick`, `BrokerOrder`,
`BrokerPosition`, `BrokerBalance`) plus the order enums (`OrderSide`,
`OrderStatus`, `OrderType`).

WHY THIS FILE EXISTS
    `governed_broker.py`, `mt5_adapter.py`, and `live_engine.py` all import these
    names from `broker_adapter`, but the definitions were never present there --
    the imports would fail at runtime. This module supplies them, reconstructed
    from exactly how those three files build and consume the objects, so the
    live stack imports and type-checks cleanly. `broker_adapter` re-exports them
    for backward compatibility, so existing `from broker_adapter import ...`
    lines keep working.

    The field sets here were derived from the real construction sites, so a
    concrete adapter (MT5, paper, or a future CCXT/IBKR one) fills them the same
    way the existing code already does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ==============================================================================
# Enums
# ==============================================================================

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ==============================================================================
# Data records
#
# These are transport objects -- plain values a broker returns or accepts. Extra
# fields carry sensible defaults so a partial construction (as several call
# sites do) stays valid.
# ==============================================================================

@dataclass
class BrokerTick:
    """A price snapshot for one symbol."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float = 0.0
    timestamp: str = ""

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if (self.bid and self.ask) else self.last

    @property
    def spread_bps(self) -> float:
        """Bid/ask spread in basis points of the mid price."""
        m = self.mid
        if not m:
            return 0.0
        return (self.ask - self.bid) / m * 10_000.0


@dataclass
class BrokerOrder:
    """The result of submitting an order (or a rejected/pending order)."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    status: OrderStatus
    fill_price: Optional[float] = None
    filled_size: float = 0.0
    commission: float = 0.0
    timestamp: str = ""
    broker_ref: str = ""
    reason: str = ""
    raw: Optional[Dict[str, Any]] = None


@dataclass
class BrokerPosition:
    """An open position on the account."""
    symbol: str
    side: str                 # 'long' | 'short' | 'flat'
    size: float
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class BrokerBalance:
    """Account balance / margin snapshot."""
    total_equity: float = 0.0
    free_margin: float = 0.0
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0
    currency: str = "USD"
    timestamp: str = ""

    @property
    def total(self) -> float:
        """Alias used by callers that read `.total`."""
        return self.total_equity


# ==============================================================================
# The broker interface
# ==============================================================================

class BaseBroker(ABC):
    """
    Interface every concrete broker adapter implements. Methods mirror exactly
    what `GovernedBroker` and `MT5Broker` already override.
    """

    def __init__(self, name: str = "broker"):
        self.name: str = name
        # Adapters set this in connect()/disconnect(); callers read it.
        self.is_connected: bool = False
        # Concrete adapters keep their own filled/rejected order log here.
        self._order_history: List[BrokerOrder] = []

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """
        Look an order up in this broker's history. Concrete adapters may
        override with a venue query; the default scans the local log.
        """
        for o in self._order_history:
            if o.order_id == order_id:
                return o
        return None

    @abstractmethod
    def connect(self) -> bool:
        """Establish the broker session. Return True on success."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the broker session."""
        raise NotImplementedError

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        """Latest price for a symbol, or None if unavailable."""
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> BrokerBalance:
        """Current account balance / margin."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """All open positions."""
        raise NotImplementedError

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """The net open position for one symbol, or None if flat."""
        raise NotImplementedError

    @abstractmethod
    def submit_order(
        self,
        side: str,
        symbol: str,
        size: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> BrokerOrder:
        """Submit an order and return the resulting BrokerOrder."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order. Return True if it was cancelled."""
        raise NotImplementedError

    # -- optional risk actions -------------------------------------------
    # Not every venue exposes a one-shot flatten; the defaults close via
    # submit_order so callers can rely on the methods existing. Adapters with
    # a native flatten (e.g. MT5) override for efficiency.
    def flatten(self, symbol: str) -> Optional[BrokerOrder]:
        """Close any open position in one symbol. None if already flat."""
        pos = self.get_position(symbol)
        if pos is None or pos.size == 0:
            return None
        closing_side = "sell" if pos.side == "long" else "buy"
        return self.submit_order(closing_side, symbol, pos.size)

    def flatten_all(self) -> List[BrokerOrder]:
        """Close every open position. Returns the closing orders."""
        orders: List[BrokerOrder] = []
        for pos in self.get_positions():
            order = self.flatten(pos.symbol)
            if order is not None:
                orders.append(order)
        return orders


# ==============================================================================
# PaperBroker -- an in-memory broker for dry runs (used by live_engine)
# ==============================================================================

class PaperBroker(BaseBroker):
    """
    In-memory broker for dry runs and tests. Fills market orders instantly at
    the current tick (plus optional slippage), charges a commission, tracks
    positions with mark-to-market unrealized PnL, and keeps an order history.

    Prices are fed in with set_price(); there is no external market data.
    """

    def __init__(self, initial_balance: float = 100_000.0,
                 slippage_bps: float = 2.0, commission_pct: float = 0.001,
                 name: str = "paper"):
        super().__init__(name=name)
        self.initial_balance = float(initial_balance)
        self.slippage_bps = float(slippage_bps)
        self.commission_pct = float(commission_pct)
        self._cash = float(initial_balance)
        self._positions: Dict[str, BrokerPosition] = {}
        self._prices: Dict[str, float] = {}
        self.order_history: List[BrokerOrder] = []
        self._seq = 0

    # -- session ----------------------------------------------------------
    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    # -- market data ------------------------------------------------------
    def set_price(self, symbol: str, price: float) -> None:
        """Set the current price for a symbol and mark positions to it."""
        self._prices[symbol] = float(price)
        self.mark_to_market()

    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        px = self._prices.get(symbol)
        if px is None:
            return None
        half = px * (self.slippage_bps / 10_000.0) / 2.0
        return BrokerTick(symbol=symbol, bid=px - half, ask=px + half, last=px)

    # -- account ----------------------------------------------------------
    @property
    def balance(self) -> float:
        """Cash plus realized PnL (equity excluding open-position PnL)."""
        return self._cash

    @property
    def equity(self) -> float:
        """Cash plus unrealized PnL of open positions."""
        return self._cash + sum(p.unrealized_pnl for p in self._positions.values())

    def mark_to_market(self) -> None:
        """Refresh unrealized PnL on every open position from current prices."""
        for sym, pos in self._positions.items():
            px = self._prices.get(sym, pos.current_price)
            pos.current_price = px
            direction = 1.0 if pos.side == "long" else -1.0
            pos.unrealized_pnl = (px - pos.entry_price) * pos.size * direction

    def get_balance(self) -> BrokerBalance:
        return BrokerBalance(
            total_equity=self.equity,
            free_margin=self._cash,
            unrealized_pnl=self.equity - self._cash,
            timestamp=datetime.utcnow().isoformat(),
        )

    def get_positions(self) -> List[BrokerPosition]:
        return [p for p in self._positions.values() if p.size != 0]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        pos = self._positions.get(symbol)
        if pos is None or pos.size == 0:
            # Represent a flat position explicitly so callers can read .side.
            return BrokerPosition(symbol=symbol, side="flat", size=0.0)
        return pos

    # -- orders -----------------------------------------------------------
    def submit_order(
        self,
        side: str,
        symbol: str,
        size: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> BrokerOrder:
        self._seq += 1
        ts = datetime.utcnow().isoformat()
        px = self._prices.get(symbol)
        if px is None:
            order = BrokerOrder(
                order_id=f"paper-{self._seq}", symbol=symbol,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                order_type=OrderType.MARKET, size=size,
                status=OrderStatus.REJECTED, timestamp=ts,
                reason="no price set for symbol",
            )
            self.order_history.append(order)
            return order

        slip = px * (self.slippage_bps / 10_000.0)
        fill = px + slip if side == "buy" else px - slip
        commission = fill * size * self.commission_pct
        self._cash -= commission
        self._apply_fill(symbol, side, size, fill)

        order = BrokerOrder(
            order_id=f"paper-{self._seq}", symbol=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET, size=size,
            status=OrderStatus.FILLED, fill_price=fill, filled_size=size,
            commission=commission, timestamp=ts,
            broker_ref=f"paper-{self._seq}",
        )
        self.order_history.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return False          # paper orders fill instantly

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Look an order up in this broker's public order_history."""
        for o in self.order_history:
            if o.order_id == order_id:
                return o
        return None

    # -- internals --------------------------------------------------------
    def _apply_fill(self, symbol: str, side: str, size: float,
                    fill: float) -> None:
        signed = size if side == "buy" else -size
        pos = self._positions.get(symbol)
        if pos is None or pos.size == 0:
            self._positions[symbol] = BrokerPosition(
                symbol=symbol, side="long" if signed > 0 else "short",
                size=abs(signed), entry_price=fill, current_price=fill,
            )
            return
        old_signed = pos.size if pos.side == "long" else -pos.size
        new_signed = old_signed + signed
        if new_signed == 0:
            self._cash += (fill - pos.entry_price) * old_signed
            pos.realized_pnl += (fill - pos.entry_price) * old_signed
            pos.side = "flat"
            pos.size = 0.0
            return
        if (old_signed > 0) == (signed > 0):
            total = abs(old_signed) + abs(signed)
            pos.entry_price = (
                pos.entry_price * abs(old_signed) + fill * abs(signed)) / total
        else:
            # partial close: realize PnL on the closed portion
            closed = min(abs(old_signed), abs(signed))
            pos.realized_pnl += (fill - pos.entry_price) * closed * (
                1.0 if old_signed > 0 else -1.0)
            self._cash += (fill - pos.entry_price) * closed * (
                1.0 if old_signed > 0 else -1.0)
        pos.side = "long" if new_signed > 0 else "short"
        pos.size = abs(new_signed)
        pos.current_price = fill


# ==============================================================================
# Live-venue adapters (thin stubs) + factory
# ==============================================================================

class CCXTBroker(BaseBroker):
    """
    Placeholder for a CCXT-backed crypto broker. The concrete implementation
    lives with the live stack; this establishes the type and constructor so
    create_broker('ccxt') and isinstance checks work.
    """

    def __init__(self, exchange: str = "binance", name: str = "ccxt"):
        super().__init__(name=name)
        self.exchange = exchange

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        return None

    def get_balance(self) -> BrokerBalance:
        return BrokerBalance()

    def get_positions(self) -> List[BrokerPosition]:
        return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        return None

    def submit_order(self, side: str, symbol: str, size: float,
                     order_type: str = "market",
                     price: Optional[float] = None,
                     stop_price: Optional[float] = None) -> BrokerOrder:
        raise NotImplementedError("CCXTBroker.submit_order not implemented")

    def cancel_order(self, order_id: str) -> bool:
        return False


class IBKRBroker(BaseBroker):
    """Placeholder for an Interactive Brokers adapter (see CCXTBroker note)."""

    def __init__(self, name: str = "ibkr"):
        super().__init__(name=name)

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        return None

    def get_balance(self) -> BrokerBalance:
        return BrokerBalance()

    def get_positions(self) -> List[BrokerPosition]:
        return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        return None

    def submit_order(self, side: str, symbol: str, size: float,
                     order_type: str = "market",
                     price: Optional[float] = None,
                     stop_price: Optional[float] = None) -> BrokerOrder:
        raise NotImplementedError("IBKRBroker.submit_order not implemented")

    def cancel_order(self, order_id: str) -> bool:
        return False


def create_broker(kind: str, **kwargs: Any) -> BaseBroker:
    """
    Factory: build a broker by name.

        create_broker("paper", initial_balance=50_000)
        create_broker("ccxt", exchange="binance")
        create_broker("ibkr")

    Raises ValueError on an unknown kind.
    """
    k = (kind or "").lower()
    if k == "paper":
        return PaperBroker(**kwargs)
    if k == "ccxt":
        return CCXTBroker(**kwargs)
    if k == "ibkr":
        return IBKRBroker(**kwargs)
    raise ValueError(f"Unknown broker kind: {kind!r}")