# ==============================================================================
# shadow_trader.py
# ==============================================================================
# Phase 4, Module 2 (Week 16): Shadow (Paper) Trading System
#
# Runs strategies in shadow mode — tracks virtual positions and PnL
# without committing real capital. Validates that live behavior matches
# backtest expectations before promotion to real trading.
#
# Features:
#   - Virtual order book with position tracking
#   - Simulated fills with configurable slippage
#   - Real-time PnL, Sharpe, drawdown tracking
#   - Comparison to backtest benchmark
#   - Integration with drift_detector for divergence alerts
#
# Consumed by:
#   - strategy_lifecycle.py (paper → live promotion gate)
#   - live_monitor.py (dashboard display)
#
# Usage:
#     from shadow_trader import ShadowTrader
#     trader = ShadowTrader(strategy_id="S_001", initial_capital=100000)
#     trader.submit_order("BUY", price=1.10, size=10000)
#     trader.mark_to_market(current_price=1.11)
#     print(trader.get_status())
# ==============================================================================

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ShadowOrder:
    """Virtual order."""
    order_id: str
    strategy_id: str
    side: OrderSide
    price: float
    size: float
    timestamp: str
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    slippage: float = 0.0


@dataclass
class ShadowPosition:
    """Current virtual position."""
    strategy_id: str
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class ShadowMetrics:
    """Running performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    start_time: str = ""

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(self.total_trades, 1)

    @property
    def days_active(self) -> int:
        return len(self.daily_returns)


@dataclass
class ShadowStatus:
    """Complete snapshot of shadow trader state."""
    strategy_id: str
    position: ShadowPosition
    metrics: ShadowMetrics
    capital: float
    is_active: bool
    orders_pending: int
    last_update: str

    def __str__(self):
        pos = self.position
        m = self.metrics
        return (
            f"\n  Shadow Trader: {self.strategy_id}\n"
            f"  Capital: ${self.capital:,.0f} | Position: {pos.side.value} {pos.size}\n"
            f"  PnL: ${m.total_pnl:,.2f} | DD: {m.max_drawdown:.2%}\n"
            f"  Trades: {m.total_trades} (WR: {m.win_rate:.0%})\n"
            f"  Sharpe: {m.sharpe_ratio:.3f} | Days: {m.days_active}"
        )


# ==============================================================================
# SHADOW TRADER
# ==============================================================================

class ShadowTrader:
    """Paper trading engine for strategy validation."""

    def __init__(
        self,
        strategy_id: str,
        initial_capital: float = 100_000,
        slippage_bps: float = 1.0,
        commission_pct: float = 0.001,
    ):
        self.strategy_id = strategy_id
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_pct = commission_pct

        self.position = ShadowPosition(strategy_id=strategy_id)
        self.metrics = ShadowMetrics(start_time=datetime.now().isoformat())
        self.metrics.equity_curve.append(initial_capital)

        self.orders: List[ShadowOrder] = []
        self.trades: List[Dict] = []
        self.is_active = True
        self._order_counter = 0
        self._last_equity = initial_capital
        self._peak_equity = initial_capital

    # ------------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------------
    def submit_order(
        self,
        side: str,
        price: float,
        size: float,
        timestamp: Optional[str] = None,
    ) -> ShadowOrder:
        """Submit a virtual order. Fills immediately with slippage."""
        if not self.is_active:
            order = ShadowOrder(
                f"O_{self._order_counter}", self.strategy_id,
                OrderSide(side), price, size,
                timestamp or datetime.now().isoformat(),
                status=OrderStatus.REJECTED,
            )
            self.orders.append(order)
            return order

        self._order_counter += 1
        slip = price * self.slippage_bps / 10000
        if side == "BUY":
            fill_price = price + slip
        else:
            fill_price = price - slip

        order = ShadowOrder(
            order_id=f"O_{self._order_counter}",
            strategy_id=self.strategy_id,
            side=OrderSide(side),
            price=price,
            size=size,
            timestamp=timestamp or datetime.now().isoformat(),
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            slippage=slip,
        )
        self.orders.append(order)
        self._process_fill(order)
        return order

    def _process_fill(self, order: ShadowOrder):
        """Update position and PnL from a filled order."""
        commission = order.fill_price * order.size * self.commission_pct

        if order.side == OrderSide.BUY:
            if self.position.side == PositionSide.SHORT:
                # Closing short
                pnl = (self.position.entry_price - order.fill_price) * min(order.size, self.position.size)
                pnl -= commission
                self._record_trade(pnl, order)
                remaining = order.size - self.position.size
                if remaining > 0:
                    self.position.side = PositionSide.LONG
                    self.position.size = remaining
                    self.position.entry_price = order.fill_price
                elif remaining < 0:
                    self.position.size = -remaining
                else:
                    self.position.side = PositionSide.FLAT
                    self.position.size = 0
            else:
                # Opening/adding long
                if self.position.size > 0:
                    total = self.position.size + order.size
                    self.position.entry_price = (
                        self.position.entry_price * self.position.size +
                        order.fill_price * order.size
                    ) / total
                    self.position.size = total
                else:
                    self.position.side = PositionSide.LONG
                    self.position.size = order.size
                    self.position.entry_price = order.fill_price
                self.capital -= commission

        elif order.side == OrderSide.SELL:
            if self.position.side == PositionSide.LONG:
                # Closing long
                pnl = (order.fill_price - self.position.entry_price) * min(order.size, self.position.size)
                pnl -= commission
                self._record_trade(pnl, order)
                remaining = order.size - self.position.size
                if remaining > 0:
                    self.position.side = PositionSide.SHORT
                    self.position.size = remaining
                    self.position.entry_price = order.fill_price
                elif remaining < 0:
                    self.position.size = -remaining
                else:
                    self.position.side = PositionSide.FLAT
                    self.position.size = 0
            else:
                # Opening/adding short
                if self.position.size > 0:
                    total = self.position.size + order.size
                    self.position.entry_price = (
                        self.position.entry_price * self.position.size +
                        order.fill_price * order.size
                    ) / total
                    self.position.size = total
                else:
                    self.position.side = PositionSide.SHORT
                    self.position.size = order.size
                    self.position.entry_price = order.fill_price
                self.capital -= commission

    def _record_trade(self, pnl: float, order: ShadowOrder):
        self.position.realized_pnl += pnl
        self.capital += pnl
        self.metrics.total_trades += 1
        if pnl > 0:
            self.metrics.winning_trades += 1
        else:
            self.metrics.losing_trades += 1
        self.trades.append({
            "order_id": order.order_id,
            "side": order.side.value,
            "fill_price": order.fill_price,
            "size": order.size,
            "pnl": pnl,
            "timestamp": order.timestamp,
        })

    # ------------------------------------------------------------------
    # MARK TO MARKET
    # ------------------------------------------------------------------
    def mark_to_market(self, current_price: float):
        """Update unrealized PnL and metrics."""
        self.position.current_price = current_price

        if self.position.side == PositionSide.LONG:
            self.position.unrealized_pnl = (
                (current_price - self.position.entry_price) * self.position.size
            )
        elif self.position.side == PositionSide.SHORT:
            self.position.unrealized_pnl = (
                (self.position.entry_price - current_price) * self.position.size
            )
        else:
            self.position.unrealized_pnl = 0.0

        equity = self.capital + self.position.unrealized_pnl
        self.metrics.equity_curve.append(equity)
        self.metrics.total_pnl = equity - self.initial_capital
        self.metrics.max_pnl = max(self.metrics.max_pnl, self.metrics.total_pnl)

        # Drawdown
        self._peak_equity = max(self._peak_equity, equity)
        dd = (self._peak_equity - equity) / max(self._peak_equity, 1)
        self.metrics.current_drawdown = dd
        self.metrics.max_drawdown = max(self.metrics.max_drawdown, dd)

    def end_of_day(self, current_price: float):
        """Record daily return and update Sharpe."""
        self.mark_to_market(current_price)
        equity = self.capital + self.position.unrealized_pnl
        daily_ret = (equity - self._last_equity) / max(self._last_equity, 1)
        self.metrics.daily_returns.append(daily_ret)
        self._last_equity = equity

        # Rolling Sharpe
        if len(self.metrics.daily_returns) >= 5:
            rets = np.array(self.metrics.daily_returns)
            self.metrics.sharpe_ratio = float(
                np.mean(rets) / max(np.std(rets, ddof=1), 1e-10) * np.sqrt(252)
            )

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------
    def get_status(self) -> ShadowStatus:
        return ShadowStatus(
            strategy_id=self.strategy_id,
            position=self.position,
            metrics=self.metrics,
            capital=self.capital,
            is_active=self.is_active,
            orders_pending=sum(1 for o in self.orders if o.status == OrderStatus.PENDING),
            last_update=datetime.now().isoformat(),
        )

    def stop(self):
        """Deactivate shadow trader."""
        self.is_active = False

    def get_comparison(self, backtest_sharpe: float) -> Dict[str, Any]:
        """Compare shadow performance to backtest benchmark."""
        live_sharpe = self.metrics.sharpe_ratio
        degradation = (1 - live_sharpe / max(backtest_sharpe, 1e-6)) * 100 if backtest_sharpe > 0 else 0
        return {
            "strategy_id": self.strategy_id,
            "backtest_sharpe": backtest_sharpe,
            "live_sharpe": live_sharpe,
            "degradation_pct": degradation,
            "days_active": self.metrics.days_active,
            "total_trades": self.metrics.total_trades,
            "max_drawdown": self.metrics.max_drawdown,
            "meets_promotion_criteria": (
                degradation < 20 and
                self.metrics.days_active >= 30 and
                self.metrics.max_drawdown < 0.20
            ),
        }
