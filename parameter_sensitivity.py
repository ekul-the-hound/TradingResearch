# ==============================================================================
# parameter_sensitivity.py
# ==============================================================================
# Week 4: Parameter Sensitivity Analysis & Heatmaps
#
# Visual confirmation of strategy robustness. Tests how performance changes
# when you vary strategy parameters. A robust strategy should have a smooth
# "performance plateau" - not a sharp spike at specific parameter values.
#
# Usage:
#     from parameter_sensitivity import ParameterSensitivity
#     
#     analyzer = ParameterSensitivity()
#     
#     # Single parameter sweep
#     result = analyzer.single_param_sweep(
#         strategy_class=MyStrategy,
#         param_name='fast_period',
#         param_range=range(5, 30, 2),
#         symbol='EUR-USD'
#     )
#     
#     # Two parameter heatmap
#     result = analyzer.two_param_heatmap(
#         strategy_class=MyStrategy,
#         param1=('fast_period', range(5, 20, 2)),
#         param2=('slow_period', range(20, 50, 5)),
#         symbol='EUR-USD'
#     )
#     
#     # Save heatmap image
#     analyzer.save_heatmap(result, 'heatmap.png')
#
# ==============================================================================

import numpy as np
import pandas as pd
import backtrader as bt
from typing import Type, Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config
from data_manager import DataManager

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN]  matplotlib not installed - heatmap visualization disabled")


@dataclass
class SingleParamResult:
    """Results from single parameter sweep"""
    strategy_name: str
    symbol: str
    timeframe: str
    param_name: str
    param_values: List[Any]
    returns: List[float]
    sharpes: List[float]
    drawdowns: List[float]
    trades: List[int]
    best_param: Any
    best_return: float
    best_sharpe: float
    stability_score: float  # How smooth is the curve? (lower = more stable)


@dataclass 
class TwoParamResult:
    """Results from two parameter heatmap"""
    strategy_name: str
    symbol: str
    timeframe: str
    param1_name: str
    param1_values: List[Any]
    param2_name: str
    param2_values: List[Any]
    return_matrix: np.ndarray
    sharpe_matrix: np.ndarray
    best_params: Tuple[Any, Any]
    best_return: float
    best_sharpe: float
    plateau_score: float  # How flat is the top? (higher = more robust)
    cliff_score: float    # How steep are the edges? (lower = more robust)


class ParameterSensitivity:
    """
    Parameter sensitivity analysis for trading strategies.
    
    Helps identify:
    - Optimal parameter values
    - Whether strategy is overfit to specific parameters
    - Robustness of parameter choices
    """
    
    def __init__(self):
        self.data_manager = DataManager()
        self.results_cache = {}
    
    def single_param_sweep(
        self,
        strategy_class: Type[bt.Strategy],
        param_name: str,
        param_range: List[Any],
        symbol: str,
        timeframe: str = '1hour',
        fixed_params: Dict = None,
        initial_cash: float = None,
        commission: float = None,
        max_bars: int = None
    ) -> SingleParamResult:
        """
        Sweep a single parameter and measure performance.
        
        Args:
            strategy_class: The strategy to test
            param_name: Name of parameter to vary
            param_range: List/range of values to test
            symbol: Trading symbol
            timeframe: Timeframe string
            fixed_params: Other parameters to keep fixed
            initial_cash: Starting capital
            commission: Commission rate
            max_bars: Maximum data bars
        
        Returns:
            SingleParamResult with performance across parameter range
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if commission is None:
            commission = config.DEFAULT_COMMISSION
        if fixed_params is None:
            fixed_params = {}
        
        param_values = list(param_range)
        
        print(f"\n{'='*60}")
        print(f"SINGLE PARAMETER SWEEP")
        print(f"{'='*60}")
        print(f"Strategy: {strategy_class.__name__}")
        print(f"Parameter: {param_name}")
        print(f"Range: {param_values[0]} to {param_values[-1]} ({len(param_values)} values)")
        print(f"Symbol: {symbol} | Timeframe: {timeframe}")
        print(f"{'='*60}")
        
        # Get data once
        data = self.data_manager.get_data(symbol, timeframe, max_bars)
        if data is None or len(data) < 100:
            print(f"[FAIL] Insufficient data")
            return None
        
        returns = []
        sharpes = []
        drawdowns = []
        trades = []
        
        for i, value in enumerate(param_values):
            print(f"  [{i+1}/{len(param_values)}] {param_name}={value}...", end=" ")
            
            params = {**fixed_params, param_name: value}
            
            try:
                result = self._run_backtest(
                    strategy_class, data, initial_cash, commission, params
                )
                
                ret = result.get('total_return_pct', 0)
                sharpe = result.get('sharpe_ratio') or 0
                dd = result.get('max_drawdown_pct', 0)
                n_trades = result.get('total_trades', 0)
                
                returns.append(ret)
                sharpes.append(sharpe)
                drawdowns.append(dd)
                trades.append(n_trades)
                
                print(f"Return: {ret:+.2f}%")
                
            except Exception as e:
                print(f"Error: {e}")
                returns.append(0)
                sharpes.append(0)
                drawdowns.append(0)
                trades.append(0)
        
        # Find best
        best_idx = np.argmax(returns)
        best_param = param_values[best_idx]
        best_return = returns[best_idx]
        best_sharpe = sharpes[best_idx]
        
        # Calculate stability score (lower = more stable)
        # Based on how much the curve changes between adjacent values
        if len(returns) > 1:
            diffs = np.abs(np.diff(returns))
            stability_score = np.mean(diffs)
        else:
            stability_score = 0
        
        result = SingleParamResult(
            strategy_name=strategy_class.__name__,
            symbol=symbol,
            timeframe=timeframe,
            param_name=param_name,
            param_values=param_values,
            returns=returns,
            sharpes=sharpes,
            drawdowns=drawdowns,
            trades=trades,
            best_param=best_param,
            best_return=best_return,
            best_sharpe=best_sharpe,
            stability_score=stability_score
        )
        
        self._print_single_result(result)
        
        return result
    
    def two_param_heatmap(
        self,
        strategy_class: Type[bt.Strategy],
        param1: Tuple[str, List[Any]],
        param2: Tuple[str, List[Any]],
        symbol: str,
        timeframe: str = '1hour',
        fixed_params: Dict = None,
        initial_cash: float = None,
        commission: float = None,
        max_bars: int = None,
        metric: str = 'return'
    ) -> TwoParamResult:
        """
        Create a heatmap of performance across two parameters.
        
        Args:
            strategy_class: The strategy to test
            param1: Tuple of (param_name, values_list)
            param2: Tuple of (param_name, values_list)
            symbol: Trading symbol
            timeframe: Timeframe string
            fixed_params: Other parameters to keep fixed
            metric: Which metric to optimize ('return' or 'sharpe')
        
        Returns:
            TwoParamResult with performance matrices
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if commission is None:
            commission = config.DEFAULT_COMMISSION
        if fixed_params is None:
            fixed_params = {}
        
        param1_name, param1_values = param1
        param2_name, param2_values = param2
        param1_values = list(param1_values)
        param2_values = list(param2_values)
        
        total_tests = len(param1_values) * len(param2_values)
        
        print(f"\n{'='*60}")
        print(f"TWO PARAMETER HEATMAP")
        print(f"{'='*60}")
        print(f"Strategy: {strategy_class.__name__}")
        print(f"Param 1: {param1_name} ({len(param1_values)} values)")
        print(f"Param 2: {param2_name} ({len(param2_values)} values)")
        print(f"Total tests: {total_tests}")
        print(f"Symbol: {symbol} | Timeframe: {timeframe}")
        print(f"{'='*60}")
        
        # Get data once
        data = self.data_manager.get_data(symbol, timeframe, max_bars)
        if data is None or len(data) < 100:
            print(f"[FAIL] Insufficient data")
            return None
        
        # Initialize matrices
        return_matrix = np.zeros((len(param1_values), len(param2_values)))
        sharpe_matrix = np.zeros((len(param1_values), len(param2_values)))
        
        test_num = 0
        for i, val1 in enumerate(param1_values):
            for j, val2 in enumerate(param2_values):
                test_num += 1
                
                if test_num % 10 == 0 or test_num == 1:
                    print(f"  [{test_num}/{total_tests}] {param1_name}={val1}, {param2_name}={val2}")
                
                params = {**fixed_params, param1_name: val1, param2_name: val2}
                
                try:
                    result = self._run_backtest(
                        strategy_class, data, initial_cash, commission, params
                    )
                    
                    return_matrix[i, j] = result.get('total_return_pct', 0)
                    sharpe_matrix[i, j] = result.get('sharpe_ratio') or 0
                    
                except Exception as e:
                    return_matrix[i, j] = 0
                    sharpe_matrix[i, j] = 0
        
        # Find best
        if metric == 'return':
            best_idx = np.unravel_index(np.argmax(return_matrix), return_matrix.shape)
        else:
            best_idx = np.unravel_index(np.argmax(sharpe_matrix), sharpe_matrix.shape)
        
        best_params = (param1_values[best_idx[0]], param2_values[best_idx[1]])
        best_return = return_matrix[best_idx]
        best_sharpe = sharpe_matrix[best_idx]
        
        # Calculate robustness scores
        plateau_score = self._calculate_plateau_score(return_matrix)
        cliff_score = self._calculate_cliff_score(return_matrix)
        
        result = TwoParamResult(
            strategy_name=strategy_class.__name__,
            symbol=symbol,
            timeframe=timeframe,
            param1_name=param1_name,
            param1_values=param1_values,
            param2_name=param2_name,
            param2_values=param2_values,
            return_matrix=return_matrix,
            sharpe_matrix=sharpe_matrix,
            best_params=best_params,
            best_return=best_return,
            best_sharpe=best_sharpe,
            plateau_score=plateau_score,
            cliff_score=cliff_score
        )
        
        self._print_two_param_result(result)
        
        return result
    
    def _run_backtest(
        self,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        initial_cash: float,
        commission: float,
        strategy_params: Dict = None
    ) -> Dict:
        """Run a single backtest"""
        
        cerebro = bt.Cerebro()
        
        # Prepare data
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
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Configure
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run
        starting = cerebro.broker.getvalue()
        results = cerebro.run()
        ending = cerebro.broker.getvalue()
        
        strat = results[0]
        
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        trade_analysis = strat.analyzers.trades.get_analysis()
        
        total_trades = 0
        if hasattr(trade_analysis, 'total') and hasattr(trade_analysis.total, 'closed'):
            total_trades = trade_analysis.total.closed
        
        return {
            'total_return_pct': ((ending - starting) / starting) * 100,
            'sharpe_ratio': sharpe_analysis.get('sharperatio'),
            'max_drawdown_pct': drawdown_analysis.get('max', {}).get('drawdown', 0),
            'total_trades': total_trades,
        }
    
    def _calculate_plateau_score(self, matrix: np.ndarray) -> float:
        """
        Calculate how flat the top of the heatmap is.
        Higher score = more robust (performance plateau exists)
        """
        
        # Find values within 90% of the best
        best = np.max(matrix)
        if best <= 0:
            return 0
        
        threshold = best * 0.9
        plateau_count = np.sum(matrix >= threshold)
        total = matrix.size
        
        return (plateau_count / total) * 100
    
    def _calculate_cliff_score(self, matrix: np.ndarray) -> float:
        """
        Calculate how steep the edges are.
        Lower score = more robust (gradual transitions)
        """
        
        # Calculate gradient magnitude
        grad_x = np.abs(np.diff(matrix, axis=0))
        grad_y = np.abs(np.diff(matrix, axis=1))
        
        max_grad = max(np.max(grad_x) if grad_x.size > 0 else 0,
                       np.max(grad_y) if grad_y.size > 0 else 0)
        
        return max_grad
    
    def _print_single_result(self, result: SingleParamResult):
        """Print single parameter result"""
        
        print(f"\n{'-'*60}")
        print(f"SINGLE PARAMETER ANALYSIS: {result.param_name}")
        print(f"{'-'*60}")
        
        print(f"\n[STATS] Results:")
        print(f"   Best {result.param_name}: {result.best_param}")
        print(f"   Best return: {result.best_return:+.2f}%")
        print(f"   Best Sharpe: {result.best_sharpe:.2f}")
        print(f"   Stability score: {result.stability_score:.2f} (lower = more stable)")
        
        # Simple ASCII chart
        print(f"\n[UP] Return by {result.param_name}:")
        max_ret = max(result.returns) if result.returns else 0
        min_ret = min(result.returns) if result.returns else 0
        range_ret = max_ret - min_ret if max_ret != min_ret else 1
        
        for i, (val, ret) in enumerate(zip(result.param_values, result.returns)):
            # Scale to 0-20 characters
            if range_ret > 0:
                bar_len = int(((ret - min_ret) / range_ret) * 20)
            else:
                bar_len = 10
            bar = '█' * bar_len
            marker = ' <-BEST' if val == result.best_param else ''
            print(f"   {val:>6}: {bar} {ret:+.2f}%{marker}")
        
        # Robustness assessment
        print(f"\n[TARGET] Robustness Assessment:")
        if result.stability_score < 5:
            print(f"   [OK] ROBUST - Performance changes gradually with parameter")
        elif result.stability_score < 15:
            print(f"   [WARN]  MODERATE - Some sensitivity to parameter choice")
        else:
            print(f"   [FAIL] FRAGILE - Performance highly dependent on exact parameter")
        
        print(f"\n{'='*60}")
    
    def _print_two_param_result(self, result: TwoParamResult):
        """Print two parameter result"""
        
        print(f"\n{'-'*60}")
        print(f"TWO PARAMETER HEATMAP ANALYSIS")
        print(f"{'-'*60}")
        
        print(f"\n[STATS] Results:")
        print(f"   Best {result.param1_name}: {result.best_params[0]}")
        print(f"   Best {result.param2_name}: {result.best_params[1]}")
        print(f"   Best return: {result.best_return:+.2f}%")
        print(f"   Best Sharpe: {result.best_sharpe:.2f}")
        
        print(f"\n[UP] Robustness Metrics:")
        print(f"   Plateau score: {result.plateau_score:.1f}% (higher = more robust)")
        print(f"   Cliff score: {result.cliff_score:.2f} (lower = more robust)")
        
        # Print mini heatmap with ASCII
        print(f"\n🗺️  Return Heatmap (ASCII):")
        print(f"      {result.param2_name} ->")
        print(f"   {result.param1_name}")
        print(f"   v")
        
        matrix = result.return_matrix
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        range_val = max_val - min_val if max_val != min_val else 1
        
        symbols = [' ', '░', '▒', '▓', '█']
        
        for i, val1 in enumerate(result.param1_values):
            row = f"   {val1:>4}|"
            for j, val2 in enumerate(result.param2_values):
                # Normalize to 0-4
                normalized = (matrix[i, j] - min_val) / range_val
                idx = min(4, int(normalized * 5))
                row += symbols[idx]
            print(row)
        
        # Legend
        print(f"\n   Legend: ' '=low  ░  ▒  ▓  █=high")
        print(f"   Range: {min_val:.1f}% to {max_val:.1f}%")
        
        # Robustness assessment
        print(f"\n[TARGET] Robustness Assessment:")
        if result.plateau_score > 20 and result.cliff_score < 10:
            print(f"   [OK] ROBUST - Wide plateau of good performance")
        elif result.plateau_score > 10:
            print(f"   [WARN]  MODERATE - Some plateau exists, but watch edges")
        else:
            print(f"   [FAIL] FRAGILE - Performance highly dependent on exact parameters")
        
        print(f"\n{'='*60}")
    
    def save_heatmap(
        self,
        result: TwoParamResult,
        filepath: str = None,
        metric: str = 'return'
    ):
        """
        Save heatmap visualization to file.
        
        Requires matplotlib to be installed.
        """
        
        if not HAS_MATPLOTLIB:
            print("[FAIL] matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if filepath is None:
            filepath = f"heatmap_{result.param1_name}_{result.param2_name}.png"
        
        # Select matrix
        if metric == 'return':
            matrix = result.return_matrix
            title = f'Return (%) - {result.strategy_name}'
        else:
            matrix = result.sharpe_matrix
            title = f'Sharpe Ratio - {result.strategy_name}'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Return (%)' if metric == 'return' else 'Sharpe', rotation=-90, va="bottom")
        
        # Set ticks
        ax.set_xticks(np.arange(len(result.param2_values)))
        ax.set_yticks(np.arange(len(result.param1_values)))
        ax.set_xticklabels(result.param2_values)
        ax.set_yticklabels(result.param1_values)
        
        # Labels
        ax.set_xlabel(result.param2_name)
        ax.set_ylabel(result.param1_name)
        ax.set_title(title)
        
        # Add values in cells
        for i in range(len(result.param1_values)):
            for j in range(len(result.param2_values)):
                val = matrix[i, j]
                text = ax.text(j, i, f'{val:.1f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        print(f"[STATS] Saved heatmap to {filepath}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_param_sweep(
    strategy_class: Type[bt.Strategy],
    param_name: str,
    param_range: List[Any],
    symbol: str = 'EUR-USD'
) -> SingleParamResult:
    """Quick single parameter sweep"""
    
    analyzer = ParameterSensitivity()
    return analyzer.single_param_sweep(
        strategy_class=strategy_class,
        param_name=param_name,
        param_range=param_range,
        symbol=symbol
    )


def quick_heatmap(
    strategy_class: Type[bt.Strategy],
    param1: Tuple[str, List],
    param2: Tuple[str, List],
    symbol: str = 'EUR-USD'
) -> TwoParamResult:
    """Quick two parameter heatmap"""
    
    analyzer = ParameterSensitivity()
    return analyzer.two_param_heatmap(
        strategy_class=strategy_class,
        param1=param1,
        param2=param2,
        symbol=symbol
    )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    try:
        from strategies.simple_strategy import SimpleMovingAverageCrossover
        
        print("\nChoose analysis type:")
        print("  1. Single parameter sweep (fast_period)")
        print("  2. Two parameter heatmap (fast_period x slow_period)")
        
        choice = input("\nChoice (1/2): ").strip()
        
        if choice == '1':
            result = quick_param_sweep(
                strategy_class=SimpleMovingAverageCrossover,
                param_name='fast_period',
                param_range=range(5, 25, 2),
                symbol='EUR-USD'
            )
        elif choice == '2':
            print("\n[WARN]  This will run ~50 backtests")
            confirm = input("Continue? (Y/N): ").strip().upper()
            
            if confirm == 'Y':
                result = quick_heatmap(
                    strategy_class=SimpleMovingAverageCrossover,
                    param1=('fast_period', range(5, 20, 3)),
                    param2=('slow_period', range(20, 50, 5)),
                    symbol='EUR-USD'
                )
            else:
                print("Cancelled.")
        else:
            print("Invalid choice")
            
    except ImportError as e:
        print(f"[FAIL] Could not import strategy: {e}")
    
    print("="*70)