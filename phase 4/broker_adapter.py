# ==============================================================================
# broker_adapter.py
# ==============================================================================
# Unified broker connection layer for live trading.
#
# Wraps CCXT (crypto) and ib_insync (futures/indices) into a single
# interface that the rest of TradingLab consumes. No other module needs
# to know which broker is being used.
#
# Architecture:
#   broker_adapter.py (this file)
#       ├── CCXTBroker    — Binance, Bybit, Hyperliquid, etc.
#       ├── IBKRBroker    — Interactive Brokers (TWS / IB Gateway)
#       └── PaperBroker   — Simulated broker for testing (no real orders)
#
# Consumed by:
#   - live_engine.py       — runs strategy signals through broker
#   - shadow_trader.py     — paper trading (uses PaperBroker)
#   - live_monitor.py      — polls positions and PnL
#   - kill_switch.py       — flatten positions on trigger
#   - strategy_lifecycle.py — promotion gate checks
#
# Usage:
#     from broker_adapter import create_broker
#
#     # Crypto
#     broker = create_broker("ccxt", exchange="binance",
#                            api_key="...", api_secret="...")
#
#     # Futures/Indices
#     broker = create_broker("ibkr", host="127.0.0.1", port=7497)
#
#     # Paper (testing)
#     broker = create_broker("paper")
#
#     # Unified interface
#     broker.submit_order("BUY", "BTC/USDT", size=0.01, order_type="market")
#     positions = broker.get_positions()
#     broker.flatten("BTC/USDT")
#     broker.flatten_all()  # kill switch calls this
# ==============================================================================

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ==============================================================================
# SHARED TYPES
# ==============================================================================

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class BrokerOrder:
    """Standardized order representation across all brokers."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None          # Limit/stop price
    stop_price: Optional[float] = None     # Stop trigger price
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    filled_size: float = 0.0
    commission: float = 0.0
    timestamp: str = ""
    broker_ref: str = ""                   # Native broker order ID
    raw: Dict = field(default_factory=dict) # Raw broker response

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                               OrderStatus.REJECTED, OrderStatus.EXPIRED)


@dataclass
class BrokerPosition:
    """Standardized position representation."""
    symbol: str
    side: str              # "long", "short", "flat"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float = 0.0
    liquidation_price: Optional[float] = None

    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl


@dataclass
class BrokerBalance:
    """Account balance snapshot."""
    total_equity: float
    free_margin: float
    used_margin: float
    unrealized_pnl: float
    currency: str = "USD"
    timestamp: str = ""


@dataclass
class BrokerTick:
    """Real-time price tick."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float = 0.0
    timestamp: str = ""

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return self.spread / max(self.mid, 1e-10) * 10000


# ==============================================================================
# ABSTRACT BROKER
# ==============================================================================

class BaseBroker(ABC):
    """Interface that all broker adapters implement."""

    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self._order_history: List[BrokerOrder] = []

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self):
        """Clean shutdown."""
        ...

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        """Get current price for a symbol."""
        ...

    @abstractmethod
    def get_balance(self) -> BrokerBalance:
        """Get account balance."""
        ...

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Get all open positions."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a specific symbol."""
        ...

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
        """Submit an order. Returns BrokerOrder with status."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order status by ID."""
        ...

    def flatten(self, symbol: str) -> Optional[BrokerOrder]:
        """Close position for a symbol."""
        pos = self.get_position(symbol)
        if pos is None or pos.side == "flat" or pos.size == 0:
            return None
        close_side = "sell" if pos.side == "long" else "buy"
        return self.submit_order(close_side, symbol, pos.size, "market")

    def flatten_all(self) -> List[BrokerOrder]:
        """Close ALL positions. Called by kill switch."""
        orders = []
        for pos in self.get_positions():
            if pos.side != "flat" and pos.size > 0:
                order = self.flatten(pos.symbol)
                if order:
                    orders.append(order)
        return orders

    def get_open_orders(self) -> List[BrokerOrder]:
        """Get all orders that aren't complete."""
        return [o for o in self._order_history if not o.is_complete]

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        count = 0
        for o in self.get_open_orders():
            if self.cancel_order(o.order_id):
                count += 1
        return count

    @property
    def order_history(self) -> List[BrokerOrder]:
        return list(self._order_history)


# ==============================================================================
# CCXT BROKER (Crypto)
# ==============================================================================

class CCXTBroker(BaseBroker):
    """
    Crypto broker via CCXT. Supports Binance, Bybit, Hyperliquid, etc.

    Requirements:
        pip install ccxt
    """

    def __init__(
        self,
        exchange: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = False,
        default_type: str = "spot",   # "spot", "future", "swap"
    ):
        super().__init__(f"ccxt_{exchange}")
        self.exchange_name = exchange
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.default_type = default_type
        self._exchange = None
        self._order_counter = 0

    def connect(self) -> bool:
        try:
            import ccxt
        except ImportError:
            logger.error("CCXT not installed. Run: pip install ccxt")
            return False

        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            config = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": self.default_type},
            }
            self._exchange = exchange_class(config)

            if self.sandbox:
                self._exchange.set_sandbox_mode(True)

            # Test connection
            self._exchange.load_markets()
            self.is_connected = True
            logger.info(f"Connected to {self.exchange_name} "
                        f"({'sandbox' if self.sandbox else 'live'}, {self.default_type})")
            return True

        except Exception as e:
            logger.error(f"CCXT connection failed: {e}")
            return False

    def disconnect(self):
        self._exchange = None
        self.is_connected = False

    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        if not self._exchange:
            return None
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return BrokerTick(
                symbol=symbol,
                bid=ticker.get("bid", 0) or 0,
                ask=ticker.get("ask", 0) or 0,
                last=ticker.get("last", 0) or 0,
                volume_24h=ticker.get("quoteVolume", 0) or 0,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"get_tick({symbol}) failed: {e}")
            return None

    def get_balance(self) -> BrokerBalance:
        if not self._exchange:
            return BrokerBalance(0, 0, 0, 0)
        try:
            bal = self._exchange.fetch_balance()
            total = bal.get("total", {})
            free = bal.get("free", {})
            used = bal.get("used", {})
            # Sum USDT values
            equity = float(total.get("USDT", 0) or 0)
            free_m = float(free.get("USDT", 0) or 0)
            used_m = float(used.get("USDT", 0) or 0)
            return BrokerBalance(
                total_equity=equity, free_margin=free_m,
                used_margin=used_m, unrealized_pnl=0,
                currency="USDT", timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return BrokerBalance(0, 0, 0, 0)

    def get_positions(self) -> List[BrokerPosition]:
        if not self._exchange:
            return []
        try:
            positions = self._exchange.fetch_positions()
            result = []
            for p in positions:
                size = abs(float(p.get("contracts", 0) or 0))
                if size == 0:
                    continue
                side = p.get("side", "long")
                result.append(BrokerPosition(
                    symbol=p.get("symbol", ""),
                    side=side,
                    size=size,
                    entry_price=float(p.get("entryPrice", 0) or 0),
                    current_price=float(p.get("markPrice", 0) or 0),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0) or 0),
                    realized_pnl=0,
                    margin_used=float(p.get("initialMargin", 0) or 0),
                    liquidation_price=float(p.get("liquidationPrice", 0) or 0)
                    if p.get("liquidationPrice") else None,
                ))
            return result
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    def submit_order(
        self, side: str, symbol: str, size: float,
        order_type: str = "market", price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> BrokerOrder:
        self._order_counter += 1
        order = BrokerOrder(
            order_id=f"ccxt_{self._order_counter}",
            symbol=symbol, side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            size=size, price=price, stop_price=stop_price,
            timestamp=datetime.now().isoformat(),
        )

        if not self._exchange:
            order.status = OrderStatus.REJECTED
            self._order_history.append(order)
            return order

        try:
            params = {}
            if stop_price:
                params["stopPrice"] = stop_price

            result = self._exchange.create_order(
                symbol=symbol,
                type=order_type.lower(),
                side=side.lower(),
                amount=size,
                price=price,
                params=params,
            )

            order.broker_ref = str(result.get("id", ""))
            order.status = self._map_ccxt_status(result.get("status", "open"))
            order.fill_price = float(result.get("average", 0) or 0)
            order.filled_size = float(result.get("filled", 0) or 0)
            fee = result.get("fee", {})
            order.commission = float(fee.get("cost", 0) or 0) if fee else 0
            order.raw = result

            logger.info(f"Order {order.order_id}: {side} {size} {symbol} "
                        f"@ {order.fill_price} [{order.status.value}]")

        except Exception as e:
            logger.error(f"submit_order failed: {e}")
            order.status = OrderStatus.REJECTED
            order.raw = {"error": str(e)}

        self._order_history.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self.get_order(order_id)
        if not order or not self._exchange:
            return False
        try:
            self._exchange.cancel_order(order.broker_ref, order.symbol)
            order.status = OrderStatus.CANCELLED
            return True
        except Exception as e:
            logger.error(f"cancel_order failed: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        for o in self._order_history:
            if o.order_id == order_id:
                return o
        return None

    def _map_ccxt_status(self, status: str) -> OrderStatus:
        return {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }.get(status, OrderStatus.PENDING)


# ==============================================================================
# IBKR BROKER (Futures / Indices / Equities)
# ==============================================================================

class IBKRBroker(BaseBroker):
    """
    Interactive Brokers via ib_insync.

    Requirements:
        pip install ib_insync
        TWS or IB Gateway running with API enabled

    Setup:
        1. Install TWS: https://www.interactivebrokers.com/en/trading/tws.php
        2. Enable API: TWS → Edit → Global Configuration → API → Settings
           - Enable ActiveX and Socket Clients
           - Socket port: 7497 (paper) or 7496 (live)
        3. Or use IB Gateway (lighter): same config
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,      # 7497=paper, 7496=live
        client_id: int = 1,
    ):
        super().__init__("ibkr")
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None
        self._order_counter = 0

    def connect(self) -> bool:
        try:
            from ib_insync import IB
        except ImportError:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False

        try:
            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self.is_connected = True
            account = self._ib.managedAccounts()
            logger.info(f"Connected to IBKR at {self.host}:{self.port} "
                        f"(account: {account[0] if account else 'unknown'})")
            return True
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            return False

    def disconnect(self):
        if self._ib:
            self._ib.disconnect()
        self.is_connected = False

    def _parse_symbol(self, symbol: str) -> Any:
        """Convert string symbol to IB contract."""
        from ib_insync import Stock, Future, Forex, Contract

        # Forex: "EUR/USD" or "EUR-USD"
        if "/" in symbol or (len(symbol) == 7 and "-" in symbol):
            pair = symbol.replace("-", "").replace("/", "")
            base, quote = pair[:3], pair[3:]
            return Forex(base + quote)

        # Futures: "ES", "NQ", "CL", "GC"
        futures_map = {
            "ES": ("ES", "CME"), "NQ": ("NQ", "CME"),
            "YM": ("YM", "CBOT"), "RTY": ("RTY", "CME"),
            "CL": ("CL", "NYMEX"), "GC": ("GC", "COMEX"),
            "SI": ("SI", "COMEX"), "NG": ("NG", "NYMEX"),
            "ZB": ("ZB", "CBOT"), "ZN": ("ZN", "CBOT"),
        }
        if symbol.upper() in futures_map:
            sym, exch = futures_map[symbol.upper()]
            return Future(sym, exchange=exch)

        # Default: stock
        return Stock(symbol, "SMART", "USD")

    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        if not self._ib:
            return None
        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)
            ticker = self._ib.reqMktData(contract)
            self._ib.sleep(1)  # Wait for data
            return BrokerTick(
                symbol=symbol,
                bid=ticker.bid if ticker.bid > 0 else 0,
                ask=ticker.ask if ticker.ask > 0 else 0,
                last=ticker.last if ticker.last > 0 else 0,
                volume_24h=ticker.volume if ticker.volume else 0,
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"get_tick({symbol}) failed: {e}")
            return None

    def get_balance(self) -> BrokerBalance:
        if not self._ib:
            return BrokerBalance(0, 0, 0, 0)
        try:
            account_values = self._ib.accountSummary()
            vals = {v.tag: float(v.value) for v in account_values
                    if v.currency == "USD"}
            return BrokerBalance(
                total_equity=vals.get("NetLiquidation", 0),
                free_margin=vals.get("AvailableFunds", 0),
                used_margin=vals.get("InitMarginReq", 0),
                unrealized_pnl=vals.get("UnrealizedPnL", 0),
                currency="USD",
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return BrokerBalance(0, 0, 0, 0)

    def get_positions(self) -> List[BrokerPosition]:
        if not self._ib:
            return []
        try:
            positions = self._ib.positions()
            result = []
            for p in positions:
                size = abs(p.position)
                if size == 0:
                    continue
                side = "long" if p.position > 0 else "short"
                result.append(BrokerPosition(
                    symbol=p.contract.localSymbol or p.contract.symbol,
                    side=side,
                    size=size,
                    entry_price=p.avgCost / (p.contract.multiplier or 1),
                    current_price=0,  # Need market data for this
                    unrealized_pnl=0,
                    realized_pnl=0,
                    margin_used=0,
                ))
            return result
        except Exception as e:
            logger.error(f"get_positions failed: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        for p in self.get_positions():
            if p.symbol == symbol or symbol.upper() in p.symbol.upper():
                return p
        return None

    def submit_order(
        self, side: str, symbol: str, size: float,
        order_type: str = "market", price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> BrokerOrder:
        from ib_insync import MarketOrder, LimitOrder, StopOrder, StopLimitOrder

        self._order_counter += 1
        order = BrokerOrder(
            order_id=f"ibkr_{self._order_counter}",
            symbol=symbol, side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            size=size, price=price, stop_price=stop_price,
            timestamp=datetime.now().isoformat(),
        )

        if not self._ib:
            order.status = OrderStatus.REJECTED
            self._order_history.append(order)
            return order

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            action = "BUY" if side.lower() == "buy" else "SELL"

            if order_type.lower() == "market":
                ib_order = MarketOrder(action, size)
            elif order_type.lower() == "limit":
                ib_order = LimitOrder(action, size, price)
            elif order_type.lower() == "stop":
                ib_order = StopOrder(action, size, stop_price or price)
            elif order_type.lower() == "stop_limit":
                ib_order = StopLimitOrder(action, size, price, stop_price)
            else:
                ib_order = MarketOrder(action, size)

            trade = self._ib.placeOrder(contract, ib_order)
            self._ib.sleep(1)  # Wait for fill

            order.broker_ref = str(trade.order.orderId)
            if trade.orderStatus.status == "Filled":
                order.status = OrderStatus.FILLED
                order.fill_price = trade.orderStatus.avgFillPrice
                order.filled_size = trade.orderStatus.filled
            elif trade.orderStatus.status == "Submitted":
                order.status = OrderStatus.OPEN
            else:
                order.status = OrderStatus.PENDING

            order.commission = sum(f.commission for f in trade.fills) if trade.fills else 0
            logger.info(f"IBKR Order {order.order_id}: {action} {size} {symbol} "
                        f"[{order.status.value}]")

        except Exception as e:
            logger.error(f"IBKR submit_order failed: {e}")
            order.status = OrderStatus.REJECTED
            order.raw = {"error": str(e)}

        self._order_history.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self.get_order(order_id)
        if not order or not self._ib:
            return False
        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order.broker_ref:
                    self._ib.cancelOrder(trade.order)
                    order.status = OrderStatus.CANCELLED
                    return True
            return False
        except Exception as e:
            logger.error(f"IBKR cancel_order failed: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        for o in self._order_history:
            if o.order_id == order_id:
                return o
        return None


# ==============================================================================
# PAPER BROKER (Testing / Shadow Trading)
# ==============================================================================

class PaperBroker(BaseBroker):
    """
    Simulated broker for testing. No real orders, no external connections.
    Used by shadow_trader.py and for pipeline integration testing.
    """

    def __init__(self, initial_balance: float = 100_000, slippage_bps: float = 1.0):
        super().__init__("paper")
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.slippage_bps = slippage_bps
        self._positions: Dict[str, BrokerPosition] = {}
        self._prices: Dict[str, float] = {}
        self._order_counter = 0

    def connect(self) -> bool:
        self.is_connected = True
        logger.info("PaperBroker connected")
        return True

    def disconnect(self):
        self.is_connected = False

    def set_price(self, symbol: str, price: float):
        """Set simulated market price for a symbol."""
        self._prices[symbol] = price

    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        price = self._prices.get(symbol, 0)
        if price == 0:
            return None
        spread = price * self.slippage_bps / 10000
        return BrokerTick(
            symbol=symbol, bid=price - spread / 2, ask=price + spread / 2,
            last=price, timestamp=datetime.now().isoformat(),
        )

    def get_balance(self) -> BrokerBalance:
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return BrokerBalance(
            total_equity=self.balance + unrealized,
            free_margin=self.balance,
            used_margin=sum(p.margin_used for p in self._positions.values()),
            unrealized_pnl=unrealized,
            currency="USD",
            timestamp=datetime.now().isoformat(),
        )

    def get_positions(self) -> List[BrokerPosition]:
        return [p for p in self._positions.values() if p.size > 0]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        return self._positions.get(symbol)

    def submit_order(
        self, side: str, symbol: str, size: float,
        order_type: str = "market", price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> BrokerOrder:
        self._order_counter += 1
        mkt_price = self._prices.get(symbol, price or 0)
        slip = mkt_price * self.slippage_bps / 10000

        if side.lower() == "buy":
            fill_price = mkt_price + slip
        else:
            fill_price = mkt_price - slip

        order = BrokerOrder(
            order_id=f"paper_{self._order_counter}",
            symbol=symbol, side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            size=size, price=price, fill_price=fill_price,
            filled_size=size, status=OrderStatus.FILLED,
            commission=fill_price * size * 0.001,
            timestamp=datetime.now().isoformat(),
        )

        # Update position
        self._update_position(symbol, side.lower(), size, fill_price, order.commission)
        self._order_history.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return False  # Paper orders fill instantly

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        for o in self._order_history:
            if o.order_id == order_id:
                return o
        return None

    def mark_to_market(self):
        """Update all positions with current prices."""
        for sym, pos in self._positions.items():
            if sym in self._prices:
                pos.current_price = self._prices[sym]
                if pos.side == "long":
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.size
                elif pos.side == "short":
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.size

    def _update_position(self, symbol: str, side: str, size: float,
                          fill_price: float, commission: float):
        pos = self._positions.get(symbol)
        if pos is None:
            self._positions[symbol] = BrokerPosition(
                symbol=symbol, side="long" if side == "buy" else "short",
                size=size, entry_price=fill_price,
                current_price=fill_price, unrealized_pnl=0, realized_pnl=0,
            )
            self.balance -= commission
            return

        # Closing or reversing
        if (pos.side == "long" and side == "sell") or (pos.side == "short" and side == "buy"):
            close_size = min(size, pos.size)
            if pos.side == "long":
                pnl = (fill_price - pos.entry_price) * close_size
            else:
                pnl = (pos.entry_price - fill_price) * close_size
            pnl -= commission
            self.balance += pnl
            pos.realized_pnl += pnl

            remaining = size - close_size
            pos.size -= close_size
            if pos.size <= 0:
                if remaining > 0:
                    pos.side = "long" if side == "buy" else "short"
                    pos.size = remaining
                    pos.entry_price = fill_price
                else:
                    pos.side = "flat"
                    pos.size = 0
        else:
            # Adding to position
            total = pos.size + size
            pos.entry_price = (pos.entry_price * pos.size + fill_price * size) / total
            pos.size = total
            self.balance -= commission


# ==============================================================================
# FACTORY
# ==============================================================================

def create_broker(
    broker_type: str,
    **kwargs,
) -> BaseBroker:
    """
    Factory function to create the right broker.

    Args:
        broker_type: "ccxt", "ibkr", or "paper"
        **kwargs: Broker-specific configuration

    Usage:
        broker = create_broker("ccxt", exchange="binance",
                               api_key="...", api_secret="...")
        broker = create_broker("ibkr", host="127.0.0.1", port=7497)
        broker = create_broker("paper", initial_balance=100000)
    """
    if broker_type.lower() == "ccxt":
        return CCXTBroker(**kwargs)
    elif broker_type.lower() in ("ibkr", "ib", "interactive_brokers"):
        return IBKRBroker(**kwargs)
    elif broker_type.lower() == "paper":
        return PaperBroker(**kwargs)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}. Use 'ccxt', 'ibkr', or 'paper'")
