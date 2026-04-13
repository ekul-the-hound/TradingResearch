# ==============================================================================
# robustness_tests.py
# ==============================================================================
# Week 3: Robustness Testing Suite
#
# Includes:
# 1. Latency Sensitivity Test - Test with +1, +2, +3 bar execution delay
# 2. Slippage Stress Sweep - Test with 1x, 2x, 3x transaction costs
# 3. Combined Stress Test - Both latency AND slippage together
#
# Usage:
#     from robustness_tests import RobustnessTests
#     
#     tester = RobustnessTests()
#     
#     # Test latency sensitivity
#     latency_results = tester.latency_sensitivity_test(
#         strategy_class=MyStrategy,
#         symbol='EUR-USD',
#         timeframe='1hour'
#     )
#     
#     # Test slippage sensitivity
#     slippage_results = tester.slippage_stress_test(
#         strategy_class=MyStrategy,
#         symbol='EUR-USD',
#         timeframe='1hour'
#     )
#
# ==============================================================================

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass
from datetime import datetime

import config
from data_manager import DataManager


@dataclass
class LatencyTestResult:
    """Results from latency sensitivity test"""
    symbol: str
    timeframe: str
    base_return: float
    base_sharpe: Optional[float]
    base_trades: int
    delay_results: Dict[int, Dict[str, Any]]
    degradation_per_bar: float
    max_tolerable_delay: int
    is_latency_sensitive: bool


@dataclass
class SlippageTestResult:
    """Results from slippage stress test"""
    symbol: str
    timeframe: str
    base_commission: float
    base_return: float
    base_sharpe: Optional[float]
    cost_results: Dict[float, Dict[str, Any]]
    breakeven_multiplier: Optional[float]
    is_cost_sensitive: bool


@dataclass
class CombinedStressResult:
    """Results from combined latency + slippage test"""
    symbol: str
    timeframe: str
    stress_matrix: pd.DataFrame
    worst_case_return: float
    best_case_return: float
    survival_rate: float  # % of scenarios still profitable


# ==============================================================================
# DELAYED EXECUTION WRAPPER
# ==============================================================================

class DelayedExecutionStrategy(bt.Strategy):
    """
    Wrapper strategy that delays order execution by N bars.
    
    This simulates real-world latency where your signal fires
    but execution happens 1-3 bars later due to:
    - Network latency
    - Broker processing time
    - Slippage during fast markets
    """
    
    params = (
        ('base_strategy', None),  # The actual strategy class
        ('delay_bars', 1),        # How many bars to delay execution
        ('base_params', {}),      # Parameters for the base strategy
    )
    
    def __init__(self):
        # Instantiate the base strategy's indicators
        self.base = self.p.base_strategy
        self.delay = self.p.delay_bars
        
        # Queue to hold pending orders
        self.pending_orders = []
        
        # Track position intent from base strategy
        self.position_intent = 0  # 1 = want long, -1 = want short, 0 = flat
        
        # Create base strategy instance for indicator calculation
        # We'll manually call its logic
        self.base_instance = None
        
        # Simple moving averages for demonstration
        # In practice, this would mirror the base strategy's indicators
        if hasattr(self.p, 'base_params') and self.p.base_params:
            fast = self.p.base_params.get('fast_period', 10)
            slow = self.p.base_params.get('slow_period', 30)
        else:
            fast, slow = 10, 30
            
        self.fast_ma = bt.indicators.SMA(self.data.close, period=fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Bar counter
        self.bar_count = 0
    
    def next(self):
        self.bar_count += 1
        
        # Determine what the base strategy WOULD do right now
        current_signal = 0
        if self.crossover > 0:
            current_signal = 1  # Buy signal
        elif self.crossover < 0:
            current_signal = -1  # Sell signal
        
        # Add to pending queue with execution time
        if current_signal != 0:
            execute_at = self.bar_count + self.delay
            self.pending_orders.append({
                'signal': current_signal,
                'execute_at': execute_at,
                'signal_price': self.data.close[0]
            })
        
        # Execute any orders that are due
        orders_to_remove = []
        for i, order in enumerate(self.pending_orders):
            if self.bar_count >= order['execute_at']:
                if order['signal'] == 1 and not self.position:
                    self.buy()
                elif order['signal'] == -1 and self.position:
                    self.sell()
                orders_to_remove.append(i)
        
        # Remove executed orders (in reverse to maintain indices)
        for i in reversed(orders_to_remove):
            self.pending_orders.pop(i)


# ==============================================================================
# ROBUSTNESS TESTS CLASS
# ==============================================================================

class RobustnessTests:
    """
    Robustness testing suite for trading strategies.
    
    Tests how strategies perform under adverse conditions:
    - Execution delays (latency)
    - Higher transaction costs (slippage)
    - Combined stress scenarios
    """
    
    def __init__(self):
        self.data_manager = DataManager()
    
    # =========================================================================
    # LATENCY SENSITIVITY TEST
    # =========================================================================
    
    def latency_sensitivity_test(
        self,
        strategy_class: Type[bt.Strategy],
        symbol: str,
        timeframe: str,
        delay_bars: List[int] = [0, 1, 2, 3],
        initial_cash: float = None,
        commission: float = None,
        strategy_params: Dict = None,
        max_bars: int = None
    ) -> LatencyTestResult:
        """
        Test how strategy performance degrades with execution delay.
        
        Args:
            strategy_class: The strategy to test
            symbol: Trading symbol
            timeframe: Timeframe string
            delay_bars: List of delay values to test (0 = no delay)
            initial_cash: Starting capital
            commission: Commission rate
            strategy_params: Strategy parameters
            max_bars: Maximum data bars to use
        
        Returns:
            LatencyTestResult with degradation analysis
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if commission is None:
            commission = config.DEFAULT_COMMISSION
        
        print(f"\n{'='*60}")
        print(f"LATENCY SENSITIVITY TEST")
        print(f"{'='*60}")
        print(f"Symbol: {symbol} | Timeframe: {timeframe}")
        print(f"Testing delays: {delay_bars} bars")
        print(f"{'='*60}")
        
        # Get data
        data = self.data_manager.get_data(symbol, timeframe, max_bars)
        if data is None or len(data) < 100:
            print(f"[FAIL] Insufficient data for {symbol}")
            return None
        
        results = {}
        
        for delay in delay_bars:
            print(f"\n  Testing {delay}-bar delay...", end=" ")
            
            try:
                result = self._run_single_backtest(
                    strategy_class=strategy_class,
                    data=data,
                    initial_cash=initial_cash,
                    commission=commission,
                    strategy_params=strategy_params,
                    delay_bars=delay
                )
                
                results[delay] = result
                ret = result.get('total_return_pct', 0)
                trades = result.get('total_trades', 0)
                print(f"Return: {ret:+.2f}% | Trades: {trades}")
                
            except Exception as e:
                print(f"[FAIL] Error: {e}")
                results[delay] = {'total_return_pct': 0, 'sharpe_ratio': None, 'total_trades': 0}
        
        # Analyze degradation
        base_return = results.get(0, {}).get('total_return_pct', 0)
        base_sharpe = results.get(0, {}).get('sharpe_ratio')
        base_trades = results.get(0, {}).get('total_trades', 0)
        
        # Calculate degradation per bar of delay
        returns = [results.get(d, {}).get('total_return_pct', 0) for d in delay_bars]
        if len(returns) > 1 and delay_bars[-1] > 0:
            degradation_per_bar = (returns[0] - returns[-1]) / delay_bars[-1]
        else:
            degradation_per_bar = 0
        
        # Find max tolerable delay (last delay where return is still positive)
        max_tolerable = 0
        for delay in delay_bars:
            if results.get(delay, {}).get('total_return_pct', 0) > 0:
                max_tolerable = delay
        
        # Determine if latency sensitive (>20% degradation at 1-bar delay)
        one_bar_return = results.get(1, {}).get('total_return_pct', 0)
        is_sensitive = (base_return - one_bar_return) / abs(base_return) > 0.2 if base_return != 0 else False
        
        # Print summary
        print(f"\n{'-'*60}")
        print(f"LATENCY ANALYSIS:")
        print(f"{'-'*60}")
        print(f"  Base return (0 delay):    {base_return:+.2f}%")
        print(f"  1-bar delay return:       {one_bar_return:+.2f}%")
        print(f"  Degradation per bar:      {degradation_per_bar:.2f}%")
        print(f"  Max tolerable delay:      {max_tolerable} bars")
        print(f"  Latency sensitive:        {'[WARN]  YES' if is_sensitive else '[OK] NO'}")
        print(f"{'='*60}")
        
        return LatencyTestResult(
            symbol=symbol,
            timeframe=timeframe,
            base_return=base_return,
            base_sharpe=base_sharpe,
            base_trades=base_trades,
            delay_results=results,
            degradation_per_bar=degradation_per_bar,
            max_tolerable_delay=max_tolerable,
            is_latency_sensitive=is_sensitive
        )
    
    # =========================================================================
    # SLIPPAGE STRESS TEST
    # =========================================================================
    
    def slippage_stress_test(
        self,
        strategy_class: Type[bt.Strategy],
        symbol: str,
        timeframe: str,
        cost_multipliers: List[float] = [1.0, 1.5, 2.0, 3.0, 5.0],
        initial_cash: float = None,
        base_commission: float = None,
        strategy_params: Dict = None,
        max_bars: int = None
    ) -> SlippageTestResult:
        """
        Test how strategy performance degrades with higher transaction costs.
        
        Args:
            strategy_class: The strategy to test
            symbol: Trading symbol
            timeframe: Timeframe string
            cost_multipliers: Multipliers for base commission (1.0 = normal)
            initial_cash: Starting capital
            base_commission: Base commission rate to multiply
            strategy_params: Strategy parameters
            max_bars: Maximum data bars
        
        Returns:
            SlippageTestResult with cost sensitivity analysis
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if base_commission is None:
            base_commission = config.DEFAULT_COMMISSION
        
        print(f"\n{'='*60}")
        print(f"SLIPPAGE STRESS TEST")
        print(f"{'='*60}")
        print(f"Symbol: {symbol} | Timeframe: {timeframe}")
        print(f"Base commission: {base_commission*100:.2f}%")
        print(f"Testing multipliers: {cost_multipliers}x")
        print(f"{'='*60}")
        
        # Get data
        data = self.data_manager.get_data(symbol, timeframe, max_bars)
        if data is None or len(data) < 100:
            print(f"[FAIL] Insufficient data for {symbol}")
            return None
        
        results = {}
        
        for multiplier in cost_multipliers:
            commission = base_commission * multiplier
            print(f"\n  Testing {multiplier}x costs ({commission*100:.3f}%)...", end=" ")
            
            try:
                result = self._run_single_backtest(
                    strategy_class=strategy_class,
                    data=data,
                    initial_cash=initial_cash,
                    commission=commission,
                    strategy_params=strategy_params,
                    delay_bars=0
                )
                
                results[multiplier] = result
                ret = result.get('total_return_pct', 0)
                trades = result.get('total_trades', 0)
                print(f"Return: {ret:+.2f}% | Trades: {trades}")
                
            except Exception as e:
                print(f"[FAIL] Error: {e}")
                results[multiplier] = {'total_return_pct': 0, 'sharpe_ratio': None, 'total_trades': 0}
        
        # Analyze cost sensitivity
        base_return = results.get(1.0, {}).get('total_return_pct', 0)
        base_sharpe = results.get(1.0, {}).get('sharpe_ratio')
        
        # Find breakeven multiplier (where return crosses zero)
        breakeven = None
        prev_mult, prev_ret = None, None
        for mult in sorted(results.keys()):
            ret = results[mult].get('total_return_pct', 0)
            if prev_ret is not None and prev_ret > 0 and ret <= 0:
                # Linear interpolation to find breakeven
                breakeven = prev_mult + (prev_ret / (prev_ret - ret)) * (mult - prev_mult)
                break
            prev_mult, prev_ret = mult, ret
        
        # Determine if cost sensitive (>30% return drop at 2x costs)
        two_x_return = results.get(2.0, {}).get('total_return_pct', 0)
        is_sensitive = (base_return - two_x_return) / abs(base_return) > 0.3 if base_return != 0 else False
        
        # Print summary
        print(f"\n{'-'*60}")
        print(f"COST SENSITIVITY ANALYSIS:")
        print(f"{'-'*60}")
        print(f"  Base return (1x cost):    {base_return:+.2f}%")
        print(f"  2x cost return:           {two_x_return:+.2f}%")
        print(f"  Breakeven multiplier:     {breakeven:.2f}x" if breakeven else "  Breakeven multiplier:     N/A (always profitable or unprofitable)")
        print(f"  Cost sensitive:           {'[WARN]  YES' if is_sensitive else '[OK] NO'}")
        print(f"{'='*60}")
        
        return SlippageTestResult(
            symbol=symbol,
            timeframe=timeframe,
            base_commission=base_commission,
            base_return=base_return,
            base_sharpe=base_sharpe,
            cost_results=results,
            breakeven_multiplier=breakeven,
            is_cost_sensitive=is_sensitive
        )
    
    # =========================================================================
    # COMBINED STRESS TEST
    # =========================================================================
    
    def combined_stress_test(
        self,
        strategy_class: Type[bt.Strategy],
        symbol: str,
        timeframe: str,
        delay_bars: List[int] = [0, 1, 2],
        cost_multipliers: List[float] = [1.0, 2.0, 3.0],
        initial_cash: float = None,
        base_commission: float = None,
        strategy_params: Dict = None,
        max_bars: int = None
    ) -> CombinedStressResult:
        """
        Test all combinations of latency AND slippage.
        
        Creates a stress matrix showing performance under various
        adverse conditions simultaneously.
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if base_commission is None:
            base_commission = config.DEFAULT_COMMISSION
        
        print(f"\n{'='*60}")
        print(f"COMBINED STRESS TEST")
        print(f"{'='*60}")
        print(f"Symbol: {symbol} | Timeframe: {timeframe}")
        print(f"Delays: {delay_bars} | Cost multipliers: {cost_multipliers}")
        print(f"Total scenarios: {len(delay_bars) * len(cost_multipliers)}")
        print(f"{'='*60}")
        
        # Get data
        data = self.data_manager.get_data(symbol, timeframe, max_bars)
        if data is None or len(data) < 100:
            print(f"[FAIL] Insufficient data for {symbol}")
            return None
        
        # Build stress matrix
        matrix_data = []
        
        for delay in delay_bars:
            for mult in cost_multipliers:
                commission = base_commission * mult
                print(f"\n  Delay={delay}, Cost={mult}x...", end=" ")
                
                try:
                    result = self._run_single_backtest(
                        strategy_class=strategy_class,
                        data=data,
                        initial_cash=initial_cash,
                        commission=commission,
                        strategy_params=strategy_params,
                        delay_bars=delay
                    )
                    
                    ret = result.get('total_return_pct', 0)
                    print(f"Return: {ret:+.2f}%")
                    
                    matrix_data.append({
                        'delay_bars': delay,
                        'cost_multiplier': mult,
                        'return_pct': ret,
                        'sharpe': result.get('sharpe_ratio'),
                        'trades': result.get('total_trades', 0)
                    })
                    
                except Exception as e:
                    print(f"[FAIL] Error: {e}")
                    matrix_data.append({
                        'delay_bars': delay,
                        'cost_multiplier': mult,
                        'return_pct': 0,
                        'sharpe': None,
                        'trades': 0
                    })
        
        # Create DataFrame
        df = pd.DataFrame(matrix_data)
        
        # Pivot for matrix view
        stress_matrix = df.pivot(
            index='delay_bars',
            columns='cost_multiplier',
            values='return_pct'
        )
        
        # Calculate summary stats
        returns = df['return_pct'].values
        worst_case = returns.min()
        best_case = returns.max()
        survival_rate = (returns > 0).mean() * 100
        
        # Print matrix
        print(f"\n{'-'*60}")
        print(f"STRESS MATRIX (Returns %):")
        print(f"{'-'*60}")
        print(stress_matrix.to_string())
        print(f"\n{'-'*60}")
        print(f"SUMMARY:")
        print(f"{'-'*60}")
        print(f"  Best case:      {best_case:+.2f}%")
        print(f"  Worst case:     {worst_case:+.2f}%")
        print(f"  Survival rate:  {survival_rate:.1f}% scenarios profitable")
        
        if survival_rate >= 80:
            print(f"  Assessment:     [OK] ROBUST - Strategy survives most stress scenarios")
        elif survival_rate >= 50:
            print(f"  Assessment:     [WARN]  MODERATE - Strategy vulnerable to adverse conditions")
        else:
            print(f"  Assessment:     [FAIL] FRAGILE - Strategy fails under stress")
        
        print(f"{'='*60}")
        
        return CombinedStressResult(
            symbol=symbol,
            timeframe=timeframe,
            stress_matrix=stress_matrix,
            worst_case_return=worst_case,
            best_case_return=best_case,
            survival_rate=survival_rate
        )
    
    # =========================================================================
    # HELPER: RUN SINGLE BACKTEST
    # =========================================================================
    
    def _run_single_backtest(
        self,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        initial_cash: float,
        commission: float,
        strategy_params: Dict = None,
        delay_bars: int = 0
    ) -> Dict:
        """Run a single backtest with optional delay"""
        
        # Create cerebro
        cerebro = bt.Cerebro()
        
        # Add data
        data_copy = data.copy()
        data_copy.columns = [c.lower() for c in data_copy.columns]
        
        if data_copy.index.tz is not None:
            data_copy.index = data_copy.index.tz_convert("UTC").tz_localize(None)
        
        data_feed = bt.feeds.PandasData(
            dataname=data_copy,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume' if 'volume' in data_copy.columns else None,
            openinterest=-1
        )
        cerebro.adddata(data_feed)
        
        # Add strategy (with or without delay wrapper)
        if delay_bars > 0:
            cerebro.addstrategy(
                DelayedExecutionStrategy,
                base_strategy=strategy_class,
                delay_bars=delay_bars,
                base_params=strategy_params or {}
            )
        else:
            if strategy_params:
                cerebro.addstrategy(strategy_class, **strategy_params)
            else:
                cerebro.addstrategy(strategy_class)
        
        # Configure broker
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run
        starting_value = cerebro.broker.getvalue()
        results = cerebro.run()
        ending_value = cerebro.broker.getvalue()
        
        # Extract results
        strat = results[0]
        sharpe = strat.analyzers.sharpe.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        
        total_return = ((ending_value - starting_value) / starting_value) * 100
        total_trades = trades.total.closed if hasattr(trades, 'total') and hasattr(trades.total, 'closed') else 0
        
        return {
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe.get('sharperatio'),
            'total_trades': total_trades,
            'ending_value': ending_value
        }
    
    # =========================================================================
    # PRINT FULL REPORT
    # =========================================================================
    
    def print_robustness_report(
        self,
        latency_result: LatencyTestResult = None,
        slippage_result: SlippageTestResult = None,
        combined_result: CombinedStressResult = None
    ):
        """Print a comprehensive robustness report"""
        
        print(f"\n{'='*70}")
        print(f"ROBUSTNESS REPORT")
        print(f"{'='*70}")
        
        issues = []
        
        if latency_result:
            print(f"\n[STATS] LATENCY SENSITIVITY:")
            print(f"   Base return:        {latency_result.base_return:+.2f}%")
            print(f"   Degradation/bar:    {latency_result.degradation_per_bar:.2f}%")
            print(f"   Max safe delay:     {latency_result.max_tolerable_delay} bars")
            if latency_result.is_latency_sensitive:
                issues.append("Latency sensitive - needs fast execution")
                print(f"   Status:             [WARN]  SENSITIVE")
            else:
                print(f"   Status:             [OK] ROBUST")
        
        if slippage_result:
            print(f"\n[COST] COST SENSITIVITY:")
            print(f"   Base return:        {slippage_result.base_return:+.2f}%")
            if slippage_result.breakeven_multiplier:
                print(f"   Breakeven at:       {slippage_result.breakeven_multiplier:.2f}x costs")
            else:
                print(f"   Breakeven at:       N/A")
            if slippage_result.is_cost_sensitive:
                issues.append("Cost sensitive - margin erosion risk")
                print(f"   Status:             [WARN]  SENSITIVE")
            else:
                print(f"   Status:             [OK] ROBUST")
        
        if combined_result:
            print(f"\n[FIRE] STRESS TEST:")
            print(f"   Best scenario:      {combined_result.best_case_return:+.2f}%")
            print(f"   Worst scenario:     {combined_result.worst_case_return:+.2f}%")
            print(f"   Survival rate:      {combined_result.survival_rate:.1f}%")
            if combined_result.survival_rate < 50:
                issues.append("Fails majority of stress scenarios")
                print(f"   Status:             [FAIL] FRAGILE")
            elif combined_result.survival_rate < 80:
                issues.append("Vulnerable to adverse conditions")
                print(f"   Status:             [WARN]  MODERATE")
            else:
                print(f"   Status:             [OK] ROBUST")
        
        # Overall assessment
        print(f"\n{'-'*70}")
        print(f"OVERALL ASSESSMENT:")
        print(f"{'-'*70}")
        
        if not issues:
            print(f"   [OK] Strategy appears ROBUST under adverse conditions")
        else:
            print(f"   [WARN]  Issues found:")
            for issue in issues:
                print(f"      - {issue}")
        
        print(f"{'='*70}\n")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_robustness_check(
    strategy_class: Type[bt.Strategy],
    symbol: str = 'EUR-USD',
    timeframe: str = '1hour'
) -> Dict:
    """
    Quick robustness check with default parameters.
    
    Returns dict with all three test results.
    """
    tester = RobustnessTests()
    
    latency = tester.latency_sensitivity_test(strategy_class, symbol, timeframe)
    slippage = tester.slippage_stress_test(strategy_class, symbol, timeframe)
    combined = tester.combined_stress_test(strategy_class, symbol, timeframe)
    
    tester.print_robustness_report(latency, slippage, combined)
    
    return {
        'latency': latency,
        'slippage': slippage,
        'combined': combined
    }


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ROBUSTNESS TESTS MODULE")
    print("="*70)
    
    # Try to import test strategy
    try:
        from strategies.simple_strategy import SimpleMovingAverageCrossover
        strategy_class = SimpleMovingAverageCrossover
        print("[OK] Loaded SimpleMovingAverageCrossover strategy")
    except ImportError:
        print("[WARN]  Could not import strategy, using built-in test")
        strategy_class = None
    
    if strategy_class:
        # Run quick robustness check
        results = quick_robustness_check(
            strategy_class=strategy_class,
            symbol='EUR-USD',
            timeframe='1hour'
        )
        
        print("\n[OK] Robustness tests complete!")
    else:
        print("\n[WARN]  Add strategies/simple_strategy.py to run tests")
    
    print("="*70)