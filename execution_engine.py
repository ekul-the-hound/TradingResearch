# ==============================================================================
# execution_engine.py
# ==============================================================================
# Paper & Live Trading Execution Engine
#
# Executes trading signals from validated strategies in paper or live mode.
# Includes realistic simulation of execution conditions.
#
# Modes:
# 1. Paper Trading - Simulated execution with realistic fills
# 2. Live Trading - Real broker integration (future)
#
# Features:
# - Slippage simulation
# - Latency modeling
# - Position sizing
# - Risk management (daily loss limits)
# - Order management
# - Performance tracking
#
# Usage:
#     from execution_engine import PaperTrader, ExecutionEngine
#     
#     # Paper trading
#     trader = PaperTrader(initial_capital=100000)
#     trader.submit_order('EUR-USD', 'BUY', size=10000)
#     trader.update(current_price=1.1050)
#     
#     # Full execution engine
#     engine = ExecutionEngine(mode='paper')
#     engine.connect()
#     engine.run_strategy(strategy_signals)
#
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import time
import warnings


# ==============================================================================
# ENUMS & DATA CLASSES
# ==============================================================================

class OrderType(Enum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'
    STOP_LIMIT = 'STOP_LIMIT'


class OrderSide(Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderStatus(Enum):
    PENDING = 'PENDING'
    SUBMITTED = 'SUBMITTED'
    FILLED = 'FILLED'
    PARTIAL = 'PARTIAL'
    CANCELLED = 'CANCELLED'
    REJECTED = 'REJECTED'


class PositionSide(Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'
    FLAT = 'FLAT'


@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0
    filled_price: float = 0
    commission: float = 0
    slippage: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


@dataclass
class Position:
    """Open position"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    commission_paid: float = 0
    
    def update_price(self, price: float):
        self.current_price = price
        multiplier = 1 if self.side == PositionSide.LONG else -1
        self.unrealized_pnl = (price - self.entry_price) * self.size * multiplier


@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    commission: float
    slippage: float
    entry_time: datetime
    exit_time: datetime
    duration: timedelta
    

@dataclass
class ExecutionConfig:
    """Execution configuration"""
    # Slippage model
    slippage_pct: float = 0.01  # 1 pip
    slippage_per_lot: float = 0.5  # Additional per lot
    
    # Latency model
    min_latency_ms: int = 10
    max_latency_ms: int = 100
    
    # Commission
    commission_per_lot: float = 5.0
    commission_pct: float = 0.0
    
    # Risk limits
    max_position_size: float = 100000
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    
    # Order settings
    max_orders_per_second: int = 10
    order_timeout_seconds: int = 30


# ==============================================================================
# PAPER TRADER
# ==============================================================================

class PaperTrader:
    """
    Paper trading simulator with realistic execution modeling.
    
    Simulates:
    - Market/limit/stop orders
    - Slippage based on order size
    - Latency delays
    - Commission
    - Position tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        config: ExecutionConfig = None
    ):
        self.initial_capital = initial_capital
        self.config = config or ExecutionConfig()
        
        # Account state
        self.cash = initial_capital
        self.equity = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Tracking
        self.order_counter = 0
        self.trade_counter = 0
        self.daily_pnl = 0
        self.peak_equity = initial_capital
        self.current_drawdown = 0
        
        # Price feeds (symbol -> last price)
        self.prices: Dict[str, float] = {}
        
        # Timestamps
        self.current_time = datetime.now()
        self.day_start_equity = initial_capital
    
    def update_price(self, symbol: str, price: float, timestamp: datetime = None):
        """Update price for a symbol"""
        self.prices[symbol] = price
        
        if timestamp:
            self.current_time = timestamp
        
        # Update position PnL
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
        
        # Update equity
        self._update_equity()
        
        # Check pending orders
        self._process_pending_orders(symbol, price)
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = 'MARKET',
        limit_price: float = None,
        stop_price: float = None
    ) -> Order:
        """
        Submit a trading order.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            size: Order size (positive)
            order_type: 'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
        
        Returns:
            Order object
        """
        
        # Check risk limits
        if not self._check_risk_limits(symbol, side, size):
            order = self._create_order(symbol, side, size, order_type, limit_price, stop_price)
            order.status = OrderStatus.REJECTED
            self.orders[order.order_id] = order
            return order
        
        # Create order
        order = self._create_order(symbol, side, size, order_type, limit_price, stop_price)
        self.orders[order.order_id] = order
        
        # Simulate latency
        latency = np.random.uniform(
            self.config.min_latency_ms,
            self.config.max_latency_ms
        ) / 1000
        
        # Process market orders immediately
        if order_type == 'MARKET':
            current_price = self.prices.get(symbol)
            if current_price:
                self._fill_order(order, current_price)
        else:
            order.status = OrderStatus.SUBMITTED
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close an open position"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        side = 'SELL' if pos.side == PositionSide.LONG else 'BUY'
        
        return self.submit_order(symbol, side, pos.size, 'MARKET')
    
    def close_all_positions(self) -> List[Order]:
        """Close all open positions"""
        orders = []
        for symbol in list(self.positions.keys()):
            order = self.close_position(symbol)
            if order:
                orders.append(order)
        return orders
    
    def _create_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str,
        limit_price: float,
        stop_price: float
    ) -> Order:
        """Create order object"""
        self.order_counter += 1
        
        return Order(
            order_id=f"ORD_{self.order_counter:06d}",
            symbol=symbol,
            side=OrderSide[side],
            order_type=OrderType[order_type],
            size=abs(size),
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            created_at=self.current_time
        )
    
    def _fill_order(self, order: Order, price: float):
        """Fill an order at given price"""
        
        # Calculate slippage
        slippage = self._calculate_slippage(order, price)
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        # Calculate commission
        commission = self._calculate_commission(order, fill_price)
        
        # Update order
        order.filled_size = order.size
        order.filled_price = fill_price
        order.commission = commission
        order.slippage = slippage
        order.status = OrderStatus.FILLED
        order.filled_at = self.current_time
        
        # Update position
        self._update_position(order)
        
        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= order.size * fill_price + commission
        else:
            self.cash += order.size * fill_price - commission
    
    def _calculate_slippage(self, order: Order, price: float) -> float:
        """Calculate slippage for order"""
        # Base slippage
        slippage = price * self.config.slippage_pct / 100
        
        # Size-based slippage (larger orders get more slippage)
        lots = order.size / 100000  # Standard lot
        slippage += lots * self.config.slippage_per_lot * 0.0001 * price
        
        # Add randomness
        slippage *= np.random.uniform(0.5, 1.5)
        
        return slippage
    
    def _calculate_commission(self, order: Order, price: float) -> float:
        """Calculate commission for order"""
        lots = order.size / 100000
        
        commission = lots * self.config.commission_per_lot
        commission += order.size * price * self.config.commission_pct / 100
        
        return commission
    
    def _update_position(self, order: Order):
        """Update position after order fill"""
        symbol = order.symbol
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            
            # Check if closing or reversing
            if (order.side == OrderSide.SELL and pos.side == PositionSide.LONG) or \
               (order.side == OrderSide.BUY and pos.side == PositionSide.SHORT):
                
                # Calculate realized PnL
                if pos.side == PositionSide.LONG:
                    pnl = (order.filled_price - pos.entry_price) * min(order.size, pos.size)
                else:
                    pnl = (pos.entry_price - order.filled_price) * min(order.size, pos.size)
                
                pnl -= order.commission
                
                # Record trade
                self._record_trade(pos, order, pnl)
                
                # Update or close position
                if order.size >= pos.size:
                    # Close position
                    del self.positions[symbol]
                    
                    # Check for reversal
                    remaining = order.size - pos.size
                    if remaining > 0:
                        self._open_position(order, remaining)
                else:
                    # Reduce position
                    pos.size -= order.size
                    pos.realized_pnl += pnl
            else:
                # Adding to position (average up/down)
                total_size = pos.size + order.size
                pos.entry_price = (pos.entry_price * pos.size + order.filled_price * order.size) / total_size
                pos.size = total_size
                pos.commission_paid += order.commission
        else:
            # Open new position
            self._open_position(order, order.size)
    
    def _open_position(self, order: Order, size: float):
        """Open a new position"""
        side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
        
        self.positions[order.symbol] = Position(
            symbol=order.symbol,
            side=side,
            size=size,
            entry_price=order.filled_price,
            current_price=order.filled_price,
            entry_time=self.current_time,
            commission_paid=order.commission
        )
    
    def _record_trade(self, position: Position, order: Order, pnl: float):
        """Record a completed trade"""
        self.trade_counter += 1
        
        entry_value = position.entry_price * min(order.size, position.size)
        return_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
        
        trade = Trade(
            trade_id=f"TRD_{self.trade_counter:06d}",
            symbol=order.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            exit_price=order.filled_price,
            size=min(order.size, position.size),
            pnl=pnl,
            return_pct=return_pct,
            commission=order.commission + position.commission_paid,
            slippage=order.slippage,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            duration=self.current_time - position.entry_time
        )
        
        self.trades.append(trade)
        self.daily_pnl += pnl
    
    def _process_pending_orders(self, symbol: str, price: float):
        """Process pending limit/stop orders"""
        for order in self.orders.values():
            if order.symbol != symbol or order.is_complete:
                continue
            
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    self._fill_order(order, order.limit_price)
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    self._fill_order(order, order.limit_price)
            
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    self._fill_order(order, price)
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    self._fill_order(order, price)
    
    def _update_equity(self):
        """Update account equity"""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.equity = self.cash + unrealized
        
        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
    
    def _check_risk_limits(self, symbol: str, side: str, size: float) -> bool:
        """Check if order passes risk limits"""
        
        # Max position size
        if size > self.config.max_position_size:
            warnings.warn(f"Order size {size} exceeds max {self.config.max_position_size}")
            return False
        
        # Daily loss limit
        if self.day_start_equity > 0:
            daily_loss_pct = -self.daily_pnl / self.day_start_equity * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                warnings.warn(f"Daily loss limit reached: {daily_loss_pct:.2f}%")
                return False
        
        # Max drawdown
        if self.current_drawdown >= self.config.max_drawdown_pct:
            warnings.warn(f"Max drawdown reached: {self.current_drawdown:.2f}%")
            return False
        
        # Cash sufficiency check for BUY orders
        if side == 'BUY':
            current_price = self.prices.get(symbol, 0)
            if current_price > 0:
                required_cash = size * current_price * 0.1  # 10% margin requirement
                if self.cash < required_cash:
                    warnings.warn(f"Insufficient cash: have ${self.cash:.2f}, need ${required_cash:.2f}")
                    return False
        
        # Minimum equity check
        min_equity = self.initial_capital * 0.05  # 5% floor
        if self.equity < min_equity:
            warnings.warn(f"Equity too low: ${self.equity:.2f} < ${min_equity:.2f} minimum")
            return False
        
        return True
    
    def reset_daily(self):
        """Reset daily tracking (call at day boundary)"""
        self.daily_pnl = 0
        self.day_start_equity = self.equity
    
    def get_account_summary(self) -> Dict:
        """Get account summary"""
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'equity': self.equity,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'realized_pnl': self.equity - self.initial_capital,
            'return_pct': (self.equity - self.initial_capital) / self.initial_capital * 100,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'open_positions': len(self.positions),
            'total_trades': len(self.trades),
            'total_orders': len(self.orders)
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'trade_id': t.trade_id,
                'symbol': t.symbol,
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'commission': t.commission,
                'slippage': t.slippage,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'duration_minutes': t.duration.total_seconds() / 60
            }
            for t in self.trades
        ])
    
    def print_summary(self):
        """Print account summary"""
        summary = self.get_account_summary()
        
        print("\n" + "="*60)
        print("PAPER TRADING SUMMARY")
        print("="*60)
        print(f"  Initial Capital:   ${summary['initial_capital']:,.2f}")
        print(f"  Current Equity:    ${summary['equity']:,.2f}")
        print(f"  Total Return:      {summary['return_pct']:+.2f}%")
        print(f"  Current Drawdown:  {summary['current_drawdown']:.2f}%")
        print(f"  Daily PnL:         ${summary['daily_pnl']:+,.2f}")
        print(f"  Open Positions:    {summary['open_positions']}")
        print(f"  Total Trades:      {summary['total_trades']}")
        print("="*60)


# ==============================================================================
# EXECUTION ENGINE (Full System)
# ==============================================================================

class ExecutionEngine:
    """
    Full execution engine for running strategies.
    
    Connects paper trading with strategy signals.
    """
    
    def __init__(
        self,
        mode: str = 'paper',
        initial_capital: float = 100000,
        config: ExecutionConfig = None
    ):
        self.mode = mode
        self.initial_capital = initial_capital
        self.config = config or ExecutionConfig()
        
        if mode == 'paper':
            self.trader = PaperTrader(initial_capital, self.config)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented yet")
        
        self.is_running = False
        self.signal_queue = queue.Queue()
    
    def process_signal(
        self,
        symbol: str,
        signal: int,  # 1 = long, -1 = short, 0 = flat
        price: float,
        size: float = None,
        timestamp: datetime = None
    ):
        """
        Process a trading signal.
        
        Args:
            symbol: Trading symbol
            signal: 1 (long), -1 (short), 0 (flat/close)
            price: Current price
            size: Position size (optional, uses default sizing)
            timestamp: Signal timestamp
        """
        
        # Update price
        self.trader.update_price(symbol, price, timestamp)
        
        # Get current position
        current_pos = self.trader.positions.get(symbol)
        current_side = current_pos.side if current_pos else PositionSide.FLAT
        
        # Determine action
        if signal == 0:
            # Close position
            if current_pos:
                self.trader.close_position(symbol)
        
        elif signal == 1:  # Long
            if current_side == PositionSide.SHORT:
                # Close short first
                self.trader.close_position(symbol)
            
            if current_side != PositionSide.LONG:
                # Open long
                order_size = size or self._calculate_size(symbol, price)
                self.trader.submit_order(symbol, 'BUY', order_size)
        
        elif signal == -1:  # Short
            if current_side == PositionSide.LONG:
                # Close long first
                self.trader.close_position(symbol)
            
            if current_side != PositionSide.SHORT:
                # Open short
                order_size = size or self._calculate_size(symbol, price)
                self.trader.submit_order(symbol, 'SELL', order_size)
    
    def _calculate_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk parameters"""
        # Simple fixed fractional sizing
        risk_pct = 0.02  # 2% risk per trade
        risk_amount = self.trader.equity * risk_pct
        
        # Assume 2% stop loss
        stop_distance = price * 0.02
        
        size = risk_amount / stop_distance
        
        # Apply limits
        size = min(size, self.config.max_position_size)
        
        # Ensure position value doesn't exceed 20% of equity (leverage limit)
        max_position_value = self.trader.equity * 0.20
        max_size_by_value = max_position_value / price if price > 0 else 0
        size = min(size, max_size_by_value)
        
        # Ensure we have cash for at least 10% margin
        margin_requirement = 0.10
        max_size_by_cash = self.trader.cash / (price * margin_requirement) if price > 0 else 0
        size = min(size, max_size_by_cash)
        
        return max(0, size)  # Never return negative
    
    def run_backtest_signals(
        self,
        signals: pd.DataFrame,
        price_col: str = 'close',
        signal_col: str = 'signal'
    ) -> Dict:
        """
        Run paper trading on historical signals.
        
        Args:
            signals: DataFrame with index as datetime, columns for price and signal
            price_col: Column name for price
            signal_col: Column name for signal
        
        Returns:
            Performance summary
        """
        
        print(f"\n{'='*60}")
        print("RUNNING PAPER TRADE SIMULATION")
        print(f"{'='*60}")
        print(f"Signals: {len(signals)}")
        print(f"Period: {signals.index[0]} to {signals.index[-1]}")
        
        # Get symbol from data if available
        symbol = signals.get('symbol', pd.Series(['UNKNOWN'])).iloc[0]
        if symbol == 'UNKNOWN':
            symbol = 'EUR-USD'  # Default
        
        equity_curve = []
        
        for timestamp, row in signals.iterrows():
            price = row[price_col]
            signal = row[signal_col]
            
            # Process signal
            self.process_signal(
                symbol=symbol,
                signal=int(signal),
                price=price,
                timestamp=timestamp
            )
            
            # Track equity
            equity_curve.append({
                'timestamp': timestamp,
                'equity': self.trader.equity,
                'drawdown': self.trader.current_drawdown
            })
        
        # Close any remaining positions
        self.trader.close_all_positions()
        
        # Get results
        equity_df = pd.DataFrame(equity_curve)
        trades_df = self.trader.get_trades_df()
        summary = self.trader.get_account_summary()
        
        # Print results
        self.trader.print_summary()
        
        if len(trades_df) > 0:
            print(f"\nTrade Statistics:")
            print(f"  Win Rate:     {(trades_df['pnl'] > 0).mean() * 100:.1f}%")
            print(f"  Avg Trade:    ${trades_df['pnl'].mean():.2f}")
            print(f"  Best Trade:   ${trades_df['pnl'].max():.2f}")
            print(f"  Worst Trade:  ${trades_df['pnl'].min():.2f}")
            print(f"  Avg Slippage: ${trades_df['slippage'].mean():.4f}")
        
        return {
            'summary': summary,
            'trades': trades_df,
            'equity_curve': equity_df
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def paper_trade_signals(
    signals: pd.DataFrame,
    initial_capital: float = 100000
) -> Dict:
    """Quick paper trading simulation"""
    engine = ExecutionEngine(mode='paper', initial_capital=initial_capital)
    return engine.run_backtest_signals(signals)


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXECUTION ENGINE TEST")
    print("="*70)
    
    # Test Paper Trader
    print("\n--- Paper Trader Test ---")
    
    trader = PaperTrader(initial_capital=100000)
    
    # Simulate price updates and orders
    trader.update_price('EUR-USD', 1.1000)
    
    # Buy order
    order = trader.submit_order('EUR-USD', 'BUY', 10000, 'MARKET')
    print(f"Order: {order.order_id}, Status: {order.status}, Fill: {order.filled_price}")
    
    # Price moves up
    trader.update_price('EUR-USD', 1.1050)
    print(f"Unrealized PnL: ${trader.positions['EUR-USD'].unrealized_pnl:.2f}")
    
    # Close position
    close_order = trader.close_position('EUR-USD')
    print(f"Close Order: {close_order.order_id}, Fill: {close_order.filled_price}")
    
    # Summary
    trader.print_summary()
    
    # Test Execution Engine with signals
    print("\n--- Execution Engine Test ---")
    
    # Generate sample signals
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.001)
    
    # Simple signal: buy when price rising, sell when falling
    returns = np.diff(prices, prepend=prices[0])
    signals = np.where(returns > 0.0005, 1, np.where(returns < -0.0005, -1, 0))
    
    signal_df = pd.DataFrame({
        'close': prices,
        'signal': signals,
        'symbol': 'EUR-USD'
    }, index=dates)
    
    # Run simulation
    engine = ExecutionEngine(mode='paper', initial_capital=100000)
    results = engine.run_backtest_signals(signal_df)
    
    print("\n" + "="*70)
    print("Execution engine working!")
    print("="*70)