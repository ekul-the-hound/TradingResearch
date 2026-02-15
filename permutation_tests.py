# ==============================================================================
# permutation_tests.py
# ==============================================================================
# Week 4: Permutation Testing
#
# Gold standard for overfitting detection. Tests whether your strategy's
# performance is statistically significant or just luck/overfitting.
#
# How it works:
# 1. Run your strategy on real data, get performance metric (e.g., Sharpe)
# 2. Shuffle the data labels (prices) many times, run strategy on each
# 3. If real performance is better than 95% of shuffled runs, it's significant
#
# Usage:
#     from permutation_tests import PermutationTester
#     
#     tester = PermutationTester()
#     result = tester.test_strategy(
#         strategy_class=MyStrategy,
#         symbol='EUR-USD',
#         timeframe='1hour',
#         n_permutations=100
#     )
#     
#     print(f"p-value: {result.p_value}")
#     print(f"Significant: {result.is_significant}")
#
# ==============================================================================

import numpy as np
import pandas as pd
import backtrader as bt
from typing import Type, Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

import config
from data_manager import DataManager


@dataclass
class PermutationResult:
    """Results from permutation test"""
    strategy_name: str
    symbol: str
    timeframe: str
    metric_name: str
    real_value: float
    permutation_values: np.ndarray
    permutation_mean: float
    permutation_std: float
    p_value: float
    is_significant: bool
    significance_level: float
    percentile_rank: float
    n_permutations: int


class PermutationTester:
    """
    Permutation testing for strategy validation.
    
    This is the gold standard for detecting overfitting. If your strategy
    can't beat randomly shuffled data, it's probably overfit.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.data_manager = DataManager()
        np.random.seed(random_seed)
    
    def test_strategy(
        self,
        strategy_class: Type[bt.Strategy],
        symbol: str,
        timeframe: str,
        metric: str = 'sharpe',
        n_permutations: int = 100,
        significance_level: float = 0.05,
        initial_cash: float = None,
        commission: float = None,
        strategy_params: Dict = None,
        max_bars: int = None,
        permutation_method: str = 'returns'
    ) -> PermutationResult:
        """
        Run permutation test on a strategy.
        
        Args:
            strategy_class: The strategy to test
            symbol: Trading symbol
            timeframe: Timeframe string
            metric: Which metric to test ('sharpe', 'return', 'sortino')
            n_permutations: Number of permutations to run
            significance_level: Alpha for significance (default 0.05)
            initial_cash: Starting capital
            commission: Commission rate
            strategy_params: Strategy parameters
            max_bars: Maximum data bars
            permutation_method: How to shuffle ('returns', 'blocks', 'circular')
        
        Returns:
            PermutationResult with p-value and significance
        """
        
        if initial_cash is None:
            initial_cash = config.DEFAULT_INITIAL_CASH
        if commission is None:
            commission = config.DEFAULT_COMMISSION
        
        print(f"\n{'='*60}")
        print(f"PERMUTATION TEST")
        print(f"{'='*60}")
        print(f"Strategy: {strategy_class.__name__}")
        print(f"Symbol: {symbol} | Timeframe: {timeframe}")
        print(f"Metric: {metric} | Permutations: {n_permutations}")
        print(f"Method: {permutation_method}")
        print(f"{'='*60}")
        
        # Get real data
        data = self.data_manager.get_data(symbol, timeframe, max_bars)
        if data is None or len(data) < 100:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        print(f"\n📊 Data: {len(data)} bars")
        
        # Run on real data
        print(f"\n🎯 Running on REAL data...")
        real_result = self._run_backtest(
            strategy_class, data, initial_cash, commission, strategy_params
        )
        real_value = self._extract_metric(real_result, metric)
        print(f"   Real {metric}: {real_value:.4f}")
        
        # Run permutations
        print(f"\n🔀 Running {n_permutations} permutations...")
        permutation_values = []
        
        for i in range(n_permutations):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"   Permutation {i+1}/{n_permutations}...")
            
            # Shuffle data
            shuffled_data = self._permute_data(data, method=permutation_method)
            
            # Run backtest on shuffled data
            try:
                perm_result = self._run_backtest(
                    strategy_class, shuffled_data, initial_cash, commission, strategy_params
                )
                perm_value = self._extract_metric(perm_result, metric)
                permutation_values.append(perm_value)
            except Exception as e:
                # Some permutations may fail, that's okay
                permutation_values.append(0)
        
        permutation_values = np.array(permutation_values)
        
        # Calculate statistics
        perm_mean = np.mean(permutation_values)
        perm_std = np.std(permutation_values)
        
        # Calculate p-value (proportion of permutations >= real value)
        # For metrics where higher is better (Sharpe, return)
        p_value = np.mean(permutation_values >= real_value)
        
        # Percentile rank of real value
        percentile_rank = np.mean(permutation_values < real_value) * 100
        
        # Is it significant?
        is_significant = p_value < significance_level
        
        result = PermutationResult(
            strategy_name=strategy_class.__name__,
            symbol=symbol,
            timeframe=timeframe,
            metric_name=metric,
            real_value=real_value,
            permutation_values=permutation_values,
            permutation_mean=perm_mean,
            permutation_std=perm_std,
            p_value=p_value,
            is_significant=is_significant,
            significance_level=significance_level,
            percentile_rank=percentile_rank,
            n_permutations=n_permutations
        )
        
        self._print_result(result)
        
        return result
    
    def _permute_data(
        self,
        data: pd.DataFrame,
        method: str = 'returns'
    ) -> pd.DataFrame:
        """
        Create a permuted version of the data.
        
        Methods:
        - 'returns': Shuffle daily returns, reconstruct prices
        - 'blocks': Shuffle blocks of data (preserves some structure)
        - 'circular': Circular shift (preserves all structure except start point)
        """
        
        df = data.copy()
        
        if method == 'returns':
            # Calculate returns
            returns = df['close'].pct_change().dropna().values
            
            # Shuffle returns
            np.random.shuffle(returns)
            
            # Reconstruct prices from shuffled returns
            initial_price = df['close'].iloc[0]
            new_prices = [initial_price]
            for r in returns:
                new_prices.append(new_prices[-1] * (1 + r))
            
            # Scale OHLC proportionally
            scale = np.array(new_prices) / df['close'].values
            
            df['open'] = df['open'] * scale
            df['high'] = df['high'] * scale
            df['low'] = df['low'] * scale
            df['close'] = new_prices
            
        elif method == 'blocks':
            # Shuffle in blocks of ~20 bars
            block_size = 20
            n_blocks = len(df) // block_size
            
            indices = list(range(n_blocks))
            np.random.shuffle(indices)
            
            new_df_parts = []
            for idx in indices:
                start = idx * block_size
                end = start + block_size
                new_df_parts.append(df.iloc[start:end].copy())
            
            # Add remaining bars
            remaining = len(df) % block_size
            if remaining > 0:
                new_df_parts.append(df.iloc[-remaining:].copy())
            
            df = pd.concat(new_df_parts, ignore_index=True)
            df.index = data.index[:len(df)]
            
        elif method == 'circular':
            # Random circular shift
            shift = np.random.randint(0, len(df))
            df = pd.concat([df.iloc[shift:], df.iloc[:shift]])
            df.index = data.index
        
        return df
    
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
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SortinoRatio, _name='sortino')
        
        # Run
        starting = cerebro.broker.getvalue()
        results = cerebro.run()
        ending = cerebro.broker.getvalue()
        
        strat = results[0]
        
        return {
            'total_return_pct': ((ending - starting) / starting) * 100,
            'sharpe': strat.analyzers.sharpe.get_analysis().get('sharperatio'),
            'sortino': strat.analyzers.sortino.get_analysis().get('sortinoratio'),
        }
    
    def _extract_metric(self, result: Dict, metric: str) -> float:
        """Extract the specified metric from results"""
        
        if metric == 'sharpe':
            val = result.get('sharpe')
            return val if val is not None else 0
        elif metric == 'return':
            return result.get('total_return_pct', 0)
        elif metric == 'sortino':
            val = result.get('sortino')
            return val if val is not None else 0
        else:
            return result.get(metric, 0)
    
    def _print_result(self, result: PermutationResult):
        """Print formatted result"""
        
        print(f"\n{'─'*60}")
        print(f"PERMUTATION TEST RESULTS")
        print(f"{'─'*60}")
        
        print(f"\n📊 {result.metric_name.upper()} Analysis:")
        print(f"   Real value:        {result.real_value:.4f}")
        print(f"   Permutation mean:  {result.permutation_mean:.4f}")
        print(f"   Permutation std:   {result.permutation_std:.4f}")
        print(f"   Percentile rank:   {result.percentile_rank:.1f}%")
        
        print(f"\n📈 Statistical Significance:")
        print(f"   p-value:           {result.p_value:.4f}")
        print(f"   Alpha:             {result.significance_level}")
        
        if result.is_significant:
            print(f"\n   ✅ SIGNIFICANT - Strategy beats random at {(1-result.significance_level)*100:.0f}% confidence")
            print(f"      Your strategy's {result.metric_name} is better than {result.percentile_rank:.1f}% of random shuffles")
        else:
            print(f"\n   ❌ NOT SIGNIFICANT - Strategy may be overfit")
            print(f"      Your strategy's {result.metric_name} is only better than {result.percentile_rank:.1f}% of random shuffles")
            print(f"      This suggests the results could be due to chance/overfitting")
        
        print(f"\n{'='*60}")


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def quick_permutation_test(
    strategy_class: Type[bt.Strategy],
    symbol: str = 'EUR-USD',
    timeframe: str = '1hour',
    n_permutations: int = 100
) -> PermutationResult:
    """Quick permutation test with defaults"""
    
    tester = PermutationTester()
    return tester.test_strategy(
        strategy_class=strategy_class,
        symbol=symbol,
        timeframe=timeframe,
        n_permutations=n_permutations
    )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PERMUTATION TESTING MODULE")
    print("="*70)
    
    try:
        from strategies.simple_strategy import SimpleMovingAverageCrossover
        
        print("\n⚠️  This will run 100 backtests (may take a few minutes)")
        confirm = input("Run permutation test? (Y/N): ").strip().upper()
        
        if confirm == 'Y':
            result = quick_permutation_test(
                strategy_class=SimpleMovingAverageCrossover,
                symbol='EUR-USD',
                timeframe='1hour',
                n_permutations=100
            )
            
            print("\n✅ Permutation test complete!")
        else:
            print("Cancelled.")
            
    except ImportError as e:
        print(f"❌ Could not import strategy: {e}")
    
    print("="*70)