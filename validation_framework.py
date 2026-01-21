# ==============================================================================
# validation_framework.py
# ==============================================================================
# Comprehensive Validation Framework for Backtesting
#
# Includes:
# 1. Bootstrap Validation - Resample trades for confidence intervals
# 2. Monte Carlo Simulation - Equity curve distributions, probability of ruin
# 3. Walk-Forward Testing - Rolling train/test, in-sample vs out-of-sample
# ==============================================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class BootstrapResult:
    """Results from bootstrap validation"""
    metric_name: str
    original_value: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    p_value: float
    distribution: np.ndarray


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    initial_capital: float
    n_simulations: int
    mean_final_equity: float
    median_final_equity: float
    std_final_equity: float
    ci_lower: float
    ci_upper: float
    probability_of_profit: float
    probability_of_ruin: float
    max_drawdown_mean: float
    max_drawdown_95th: float
    mean_return: float
    median_return: float
    sharpe_ratio_mean: float
    final_equities: np.ndarray
    max_drawdowns: np.ndarray


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    n_windows: int
    is_returns: List[float]
    oos_returns: List[float]
    is_sharpe: List[float]
    oos_sharpe: List[float]
    return_degradation: float
    sharpe_degradation: float
    oos_positive_pct: float
    correlation_is_oos: float
    total_oos_return: float
    total_oos_sharpe: float


class ValidationFramework:
    """Comprehensive validation framework for trading strategies."""
    
    def __init__(
        self,
        random_seed: int = 42,
        n_bootstrap: int = 1000,
        n_monte_carlo: int = 1000,
        confidence_level: float = 0.95,
        ruin_threshold: float = 0.5,
    ):
        self.random_seed = random_seed
        self.n_bootstrap = n_bootstrap
        self.n_monte_carlo = n_monte_carlo
        self.confidence_level = confidence_level
        self.ruin_threshold = ruin_threshold
        np.random.seed(random_seed)
    
    # =========================================================================
    # BOOTSTRAP VALIDATION
    # =========================================================================
    
    def bootstrap_trades(
        self,
        trades: pd.DataFrame,
        metric: str = 'return_pct',
        n_samples: Optional[int] = None
    ) -> BootstrapResult:
        """Bootstrap resample trades to get confidence intervals."""
        
        if n_samples is None:
            n_samples = self.n_bootstrap
        
        if metric not in trades.columns:
            raise ValueError(f"Metric '{metric}' not found in trades DataFrame")
        
        values = trades[metric].dropna().values
        n_trades = len(values)
        
        if n_trades < 10:
            warnings.warn(f"Only {n_trades} trades - bootstrap may be unreliable")
        
        original_mean = np.mean(values)
        bootstrap_means = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample = np.random.choice(values, size=n_trades, replace=True)
            bootstrap_means[i] = np.mean(sample)
        
        mean = np.mean(bootstrap_means)
        std = np.std(bootstrap_means)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        p_value = np.mean(bootstrap_means <= 0)
        
        return BootstrapResult(
            metric_name=metric,
            original_value=original_mean,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            distribution=bootstrap_means
        )
    
    def bootstrap_sharpe(
        self,
        trades: pd.DataFrame,
        return_col: str = 'return_pct',
        n_samples: Optional[int] = None
    ) -> BootstrapResult:
        """Bootstrap the Sharpe ratio of trades."""
        
        if n_samples is None:
            n_samples = self.n_bootstrap
        
        values = trades[return_col].dropna().values
        n_trades = len(values)
        
        if n_trades < 10:
            warnings.warn(f"Only {n_trades} trades - bootstrap may be unreliable")
        
        original_sharpe = np.mean(values) / np.std(values) if np.std(values) > 0 else 0
        bootstrap_sharpes = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample = np.random.choice(values, size=n_trades, replace=True)
            std = np.std(sample)
            bootstrap_sharpes[i] = np.mean(sample) / std if std > 0 else 0
        
        mean = np.mean(bootstrap_sharpes)
        std = np.std(bootstrap_sharpes)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_sharpes, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha/2) * 100)
        p_value = np.mean(bootstrap_sharpes <= 0)
        
        return BootstrapResult(
            metric_name='sharpe_ratio',
            original_value=original_sharpe,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            distribution=bootstrap_sharpes
        )
    
    def bootstrap_multiple_metrics(
        self,
        trades: pd.DataFrame,
        metrics: List[str] = ['return_pct', 'win_rate', 'profit_factor']
    ) -> Dict[str, BootstrapResult]:
        """Bootstrap multiple metrics at once."""
        
        results = {}
        
        for metric in metrics:
            if metric in trades.columns:
                results[metric] = self.bootstrap_trades(trades, metric)
            elif metric == 'win_rate' and 'return_pct' in trades.columns:
                trades_copy = trades.copy()
                trades_copy['win'] = (trades_copy['return_pct'] > 0).astype(float)
                results['win_rate'] = self.bootstrap_trades(trades_copy, 'win')
            elif metric == 'profit_factor' and 'return_pct' in trades.columns:
                results['profit_factor'] = self._bootstrap_profit_factor(trades)
        
        return results
    
    def _bootstrap_profit_factor(
        self,
        trades: pd.DataFrame,
        return_col: str = 'return_pct'
    ) -> BootstrapResult:
        """Bootstrap profit factor specifically"""
        
        values = trades[return_col].dropna().values
        n_trades = len(values)
        
        gains = values[values > 0].sum()
        losses = abs(values[values < 0].sum())
        original_pf = gains / losses if losses > 0 else float('inf')
        
        bootstrap_pfs = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            sample = np.random.choice(values, size=n_trades, replace=True)
            gains = sample[sample > 0].sum()
            losses = abs(sample[sample < 0].sum())
            bootstrap_pfs[i] = gains / losses if losses > 0 else 10
        
        mean = np.mean(bootstrap_pfs)
        std = np.std(bootstrap_pfs)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_pfs, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_pfs, (1 - alpha/2) * 100)
        p_value = np.mean(bootstrap_pfs <= 1)
        
        return BootstrapResult(
            metric_name='profit_factor',
            original_value=original_pf if original_pf != float('inf') else 10,
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            distribution=bootstrap_pfs
        )
    
    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================
    
    def monte_carlo_equity(
        self,
        trades: pd.DataFrame,
        initial_capital: float = 10000,
        return_col: str = 'return_pct',
        n_simulations: Optional[int] = None
    ) -> MonteCarloResult:
        """Monte Carlo simulation of equity curves."""
        
        if n_simulations is None:
            n_simulations = self.n_monte_carlo
        
        returns = trades[return_col].dropna().values
        n_trades = len(returns)
        
        if n_trades < 5:
            raise ValueError("Need at least 5 trades for Monte Carlo simulation")
        
        final_equities = np.zeros(n_simulations)
        max_drawdowns = np.zeros(n_simulations)
        
        for sim in range(n_simulations):
            shuffled_returns = np.random.permutation(returns)
            equity = initial_capital
            peak = initial_capital
            max_dd = 0
            
            for ret in shuffled_returns:
                equity *= (1 + ret / 100)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
            
            final_equities[sim] = equity
            max_drawdowns[sim] = max_dd
        
        returns_pct = (final_equities / initial_capital - 1) * 100
        prob_ruin = np.mean(max_drawdowns >= self.ruin_threshold)
        prob_profit = np.mean(final_equities > initial_capital)
        
        mean_return = np.mean(returns_pct)
        std_return = np.std(returns_pct)
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(final_equities, alpha/2 * 100)
        ci_upper = np.percentile(final_equities, (1 - alpha/2) * 100)
        
        return MonteCarloResult(
            initial_capital=initial_capital,
            n_simulations=n_simulations,
            mean_final_equity=np.mean(final_equities),
            median_final_equity=np.median(final_equities),
            std_final_equity=np.std(final_equities),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            probability_of_profit=prob_profit,
            probability_of_ruin=prob_ruin,
            max_drawdown_mean=np.mean(max_drawdowns) * 100,
            max_drawdown_95th=np.percentile(max_drawdowns, 95) * 100,
            mean_return=mean_return,
            median_return=np.median(returns_pct),
            sharpe_ratio_mean=sharpe,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns
        )
    
    def monte_carlo_with_costs(
        self,
        trades: pd.DataFrame,
        initial_capital: float = 10000,
        return_col: str = 'return_pct',
        slippage_pct: float = 0.1,
        commission_pct: float = 0.1,
        n_simulations: Optional[int] = None
    ) -> MonteCarloResult:
        """Monte Carlo with additional costs to stress test."""
        
        trades_adjusted = trades.copy()
        total_cost = slippage_pct + commission_pct
        trades_adjusted[return_col] = trades_adjusted[return_col] - total_cost
        
        return self.monte_carlo_equity(
            trades_adjusted,
            initial_capital,
            return_col,
            n_simulations
        )
    
    # =========================================================================
    # WALK-FORWARD TESTING
    # =========================================================================
    
    def walk_forward_test(
        self,
        run_backtest_func: Callable,
        data: pd.DataFrame,
        n_windows: int = 5,
        train_pct: float = 0.7,
        min_train_bars: int = 200,
        min_test_bars: int = 50,
        **backtest_kwargs
    ) -> WalkForwardResult:
        """Perform walk-forward analysis."""
        
        total_bars = len(data)
        window_size = total_bars // n_windows
        
        if window_size < min_train_bars + min_test_bars:
            raise ValueError(
                f"Window size ({window_size}) too small for min_train ({min_train_bars}) + min_test ({min_test_bars})"
            )
        
        is_returns = []
        oos_returns = []
        is_sharpes = []
        oos_sharpes = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size if i < n_windows - 1 else total_bars
            
            window_data = data.iloc[start_idx:end_idx]
            window_len = len(window_data)
            
            train_size = int(window_len * train_pct)
            train_data = window_data.iloc[:train_size]
            test_data = window_data.iloc[train_size:]
            
            if len(train_data) < min_train_bars or len(test_data) < min_test_bars:
                warnings.warn(f"Window {i+1} skipped: insufficient data")
                continue
            
            try:
                is_result = run_backtest_func(train_data, **backtest_kwargs)
                is_return = is_result.get('total_return_pct', 0)
                is_sharpe = is_result.get('sharpe_ratio', 0) or 0
            except Exception as e:
                warnings.warn(f"Window {i+1} IS backtest failed: {e}")
                is_return = 0
                is_sharpe = 0
            
            try:
                oos_result = run_backtest_func(test_data, **backtest_kwargs)
                oos_return = oos_result.get('total_return_pct', 0)
                oos_sharpe = oos_result.get('sharpe_ratio', 0) or 0
            except Exception as e:
                warnings.warn(f"Window {i+1} OOS backtest failed: {e}")
                oos_return = 0
                oos_sharpe = 0
            
            is_returns.append(is_return)
            oos_returns.append(oos_return)
            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)
        
        if not is_returns:
            raise ValueError("No valid walk-forward windows completed")
        
        return_degradation = np.mean(is_returns) - np.mean(oos_returns)
        sharpe_degradation = np.mean(is_sharpes) - np.mean(oos_sharpes)
        oos_positive_pct = np.mean([r > 0 for r in oos_returns]) * 100
        
        if len(is_returns) >= 3:
            correlation = np.corrcoef(is_returns, oos_returns)[0, 1]
        else:
            correlation = 0
        
        total_oos_return = np.prod([1 + r/100 for r in oos_returns]) * 100 - 100
        total_oos_sharpe = np.mean(oos_sharpes)
        
        return WalkForwardResult(
            n_windows=len(is_returns),
            is_returns=is_returns,
            oos_returns=oos_returns,
            is_sharpe=is_sharpes,
            oos_sharpe=oos_sharpes,
            return_degradation=return_degradation,
            sharpe_degradation=sharpe_degradation,
            oos_positive_pct=oos_positive_pct,
            correlation_is_oos=correlation,
            total_oos_return=total_oos_return,
            total_oos_sharpe=total_oos_sharpe
        )
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_bootstrap_report(self, results: Dict[str, BootstrapResult]):
        """Print formatted bootstrap validation report"""
        
        print("\n" + "="*70)
        print("BOOTSTRAP VALIDATION REPORT")
        print("="*70)
        print(f"Samples: {self.n_bootstrap} | Confidence: {self.confidence_level*100:.0f}%")
        print("-"*70)
        print(f"{'Metric':<20} {'Original':>10} {'Mean':>10} {'95% CI':>20} {'P-value':>10}")
        print("-"*70)
        
        for metric, result in results.items():
            ci_str = f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
            sig = "***" if result.p_value < 0.01 else "**" if result.p_value < 0.05 else "*" if result.p_value < 0.1 else ""
            print(f"{metric:<20} {result.original_value:>10.3f} {result.mean:>10.3f} {ci_str:>20} {result.p_value:>8.3f} {sig}")
        
        print("-"*70)
        print("Significance: *** p<0.01, ** p<0.05, * p<0.1")
        print("="*70 + "\n")
    
    def print_monte_carlo_report(self, result: MonteCarloResult):
        """Print formatted Monte Carlo report"""
        
        print("\n" + "="*70)
        print("MONTE CARLO SIMULATION REPORT")
        print("="*70)
        print(f"Simulations: {result.n_simulations} | Initial Capital: ${result.initial_capital:,.0f}")
        print("-"*70)
        
        print("\nFINAL EQUITY DISTRIBUTION:")
        print(f"  Mean:     ${result.mean_final_equity:,.2f}")
        print(f"  Median:   ${result.median_final_equity:,.2f}")
        print(f"  Std Dev:  ${result.std_final_equity:,.2f}")
        print(f"  95% CI:   [${result.ci_lower:,.2f}, ${result.ci_upper:,.2f}]")
        
        print("\nRETURN STATISTICS:")
        print(f"  Mean Return:    {result.mean_return:+.2f}%")
        print(f"  Median Return:  {result.median_return:+.2f}%")
        print(f"  Sharpe Ratio:   {result.sharpe_ratio_mean:.2f}")
        
        print("\nRISK METRICS:")
        print(f"  Prob. of Profit:    {result.probability_of_profit*100:.1f}%")
        print(f"  Prob. of Ruin:      {result.probability_of_ruin*100:.1f}% (>{self.ruin_threshold*100:.0f}% DD)")
        print(f"  Mean Max Drawdown:  {result.max_drawdown_mean:.1f}%")
        print(f"  95th %ile Max DD:   {result.max_drawdown_95th:.1f}%")
        
        print("="*70 + "\n")
    
    def print_walk_forward_report(self, result: WalkForwardResult):
        """Print formatted walk-forward report"""
        
        print("\n" + "="*70)
        print("WALK-FORWARD ANALYSIS REPORT")
        print("="*70)
        print(f"Windows: {result.n_windows}")
        print("-"*70)
        
        print("\nWINDOW RESULTS:")
        print(f"{'Window':<10} {'IS Return':>12} {'OOS Return':>12} {'IS Sharpe':>12} {'OOS Sharpe':>12}")
        print("-"*70)
        
        for i in range(result.n_windows):
            print(f"  {i+1:<8} {result.is_returns[i]:>11.2f}% {result.oos_returns[i]:>11.2f}% "
                  f"{result.is_sharpe[i]:>12.2f} {result.oos_sharpe[i]:>12.2f}")
        
        print("-"*70)
        print(f"  {'Average':<8} {np.mean(result.is_returns):>11.2f}% {np.mean(result.oos_returns):>11.2f}% "
              f"{np.mean(result.is_sharpe):>12.2f} {np.mean(result.oos_sharpe):>12.2f}")
        
        print("\nDEGRADATION ANALYSIS:")
        print(f"  Return Degradation:  {result.return_degradation:+.2f}% (IS - OOS)")
        print(f"  Sharpe Degradation:  {result.sharpe_degradation:+.2f} (IS - OOS)")
        
        if result.return_degradation > 5:
            print("  WARNING: HIGH DEGRADATION - Possible overfitting!")
        elif result.return_degradation > 2:
            print("  CAUTION: MODERATE DEGRADATION - Some overfitting likely")
        else:
            print("  OK: LOW DEGRADATION - Strategy appears robust")
        
        print("\nOUT-OF-SAMPLE PERFORMANCE:")
        print(f"  Total OOS Return:    {result.total_oos_return:+.2f}%")
        print(f"  Average OOS Sharpe:  {result.total_oos_sharpe:.2f}")
        print(f"  OOS Win Rate:        {result.oos_positive_pct:.1f}% of windows profitable")
        print(f"  IS/OOS Correlation:  {result.correlation_is_oos:.2f}")
        
        print("="*70 + "\n")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_bootstrap(trades: pd.DataFrame, return_col: str = 'return_pct') -> BootstrapResult:
    """Quick bootstrap validation of trade returns"""
    validator = ValidationFramework()
    return validator.bootstrap_trades(trades, return_col)


def quick_monte_carlo(trades: pd.DataFrame, initial_capital: float = 10000) -> MonteCarloResult:
    """Quick Monte Carlo simulation"""
    validator = ValidationFramework()
    return validator.monte_carlo_equity(trades, initial_capital)


if __name__ == "__main__":
    print("="*70)
    print("VALIDATION FRAMEWORK TEST")
    print("="*70)
    
    np.random.seed(42)
    n_trades = 100
    returns = np.random.normal(0.5, 3.0, n_trades)
    
    trades = pd.DataFrame({
        'trade_id': range(n_trades),
        'return_pct': returns,
        'duration_bars': np.random.randint(1, 20, n_trades)
    })
    
    validator = ValidationFramework(n_bootstrap=1000, n_monte_carlo=1000)
    
    print("\nTesting Bootstrap Validation...")
    bootstrap_results = validator.bootstrap_multiple_metrics(trades, metrics=['return_pct'])
    bootstrap_results['sharpe'] = validator.bootstrap_sharpe(trades)
    validator.print_bootstrap_report(bootstrap_results)
    
    print("\nTesting Monte Carlo Simulation...")
    mc_result = validator.monte_carlo_equity(trades, initial_capital=10000)
    validator.print_monte_carlo_report(mc_result)
    
    print("\n" + "="*70)
    print("Validation framework working!")
    print("="*70)