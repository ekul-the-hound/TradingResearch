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
from typing import List, Dict, Tuple, Optional, Callable, Any
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
# STATISTICAL ANALYSIS - Serial Dependence, Distribution Tests, GARCH, VaR
# ==============================================================================

@dataclass
class SerialDependenceResult:
    """Results from serial dependence tests"""
    autocorr_lag1: float
    autocorr_lag5: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    has_serial_dependence: bool  # True if p < 0.05
    interpretation: str


@dataclass
class DistributionResult:
    """Results from distribution analysis"""
    mean: float
    std: float
    skewness: float
    kurtosis: float  # Excess kurtosis (normal = 0)
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal: bool  # True if p > 0.05
    interpretation: str


@dataclass
class GARCHResult:
    """Results from GARCH volatility modeling"""
    omega: float  # Constant
    alpha: float  # ARCH term (reaction to shocks)
    beta: float   # GARCH term (persistence)
    persistence: float  # alpha + beta
    unconditional_vol: float
    forecast_vol_1day: float
    aic: float
    model_fit: bool
    interpretation: str


@dataclass
class VaRResult:
    """Value at Risk results"""
    confidence_level: float
    historical_var: float
    parametric_var: float
    cornish_fisher_var: float  # Adjusted for skew/kurtosis
    cvar_expected_shortfall: float
    interpretation: str


class StatisticalAnalysis:
    """
    Statistical analysis suite for validating trading strategy returns.
    
    Tests:
    1. Serial Dependence - Are returns autocorrelated? (violates bootstrap/MC assumptions)
    2. Distribution Analysis - Skewness, kurtosis, normality tests
    3. GARCH - Volatility clustering and forecasting
    4. VaR - Value at Risk with multiple methods
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    # =========================================================================
    # SERIAL DEPENDENCE TESTS
    # =========================================================================
    
    def test_serial_dependence(
        self,
        returns: np.ndarray,
        max_lags: int = 10
    ) -> SerialDependenceResult:
        """
        Test for serial dependence (autocorrelation) in returns.
        
        If significant, bootstrap and Monte Carlo assumptions are violated.
        Consider using block bootstrap or adjusting confidence intervals.
        
        Args:
            returns: Array of returns
            max_lags: Number of lags for Ljung-Box test
        
        Returns:
            SerialDependenceResult with test statistics
        """
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < max_lags + 10:
            return SerialDependenceResult(
                autocorr_lag1=0, autocorr_lag5=0,
                ljung_box_stat=0, ljung_box_pvalue=1.0,
                has_serial_dependence=False,
                interpretation="Insufficient data for serial dependence test"
            )
        
        # Calculate autocorrelations
        n = len(returns)
        mean_r = np.mean(returns)
        var_r = np.var(returns)
        
        def autocorr(lag):
            if var_r == 0:
                return 0
            return np.sum((returns[lag:] - mean_r) * (returns[:-lag] - mean_r)) / (n * var_r)
        
        acf_1 = autocorr(1) if len(returns) > 1 else 0
        acf_5 = autocorr(5) if len(returns) > 5 else 0
        
        # Ljung-Box test statistic
        # Q = n(n+2) * sum(acf_k^2 / (n-k)) for k=1 to max_lags
        lb_stat = 0
        for k in range(1, min(max_lags + 1, n)):
            acf_k = autocorr(k)
            lb_stat += (acf_k ** 2) / (n - k)
        lb_stat *= n * (n + 2)
        
        # p-value from chi-squared distribution
        from scipy import stats
        lb_pvalue = 1 - stats.chi2.cdf(lb_stat, df=max_lags)
        
        has_dependence = lb_pvalue < 0.05
        
        if has_dependence:
            interpretation = (
                f"SIGNIFICANT serial dependence detected (p={lb_pvalue:.4f}). "
                f"Returns are autocorrelated - bootstrap CI may be too narrow. "
                f"Consider block bootstrap or regime-aware analysis."
            )
        else:
            interpretation = (
                f"No significant serial dependence (p={lb_pvalue:.4f}). "
                f"Returns appear independent - bootstrap assumptions valid."
            )
        
        return SerialDependenceResult(
            autocorr_lag1=acf_1,
            autocorr_lag5=acf_5,
            ljung_box_stat=lb_stat,
            ljung_box_pvalue=lb_pvalue,
            has_serial_dependence=has_dependence,
            interpretation=interpretation
        )
    
    # =========================================================================
    # DISTRIBUTION ANALYSIS (Skew, Kurtosis, Normality)
    # =========================================================================
    
    def analyze_distribution(
        self,
        returns: np.ndarray
    ) -> DistributionResult:
        """
        Analyze return distribution for skewness, kurtosis, and normality.
        
        - Skewness: Asymmetry. Negative = left tail (crash risk). Normal = 0.
        - Kurtosis: Tail thickness. Positive = fat tails. Normal = 0 (excess).
        - Jarque-Bera: Tests if distribution is normal.
        
        Args:
            returns: Array of returns
        
        Returns:
            DistributionResult with statistics and interpretation
        """
        from scipy import stats
        
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 10:
            return DistributionResult(
                mean=0, std=0, skewness=0, kurtosis=0,
                jarque_bera_stat=0, jarque_bera_pvalue=1.0,
                is_normal=True,
                interpretation="Insufficient data for distribution analysis"
            )
        
        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis (Fisher's)
        
        # Jarque-Bera test
        n = len(returns)
        jb_stat = (n / 6) * (skew**2 + (kurt**2) / 4)
        jb_pvalue = 1 - stats.chi2.cdf(jb_stat, df=2)
        
        is_normal = jb_pvalue > 0.05
        
        # Build interpretation
        interp_parts = []
        
        if skew < -0.5:
            interp_parts.append(f"Negative skew ({skew:.2f}) indicates LEFT TAIL risk (crashes)")
        elif skew > 0.5:
            interp_parts.append(f"Positive skew ({skew:.2f}) indicates RIGHT TAIL (large gains possible)")
        else:
            interp_parts.append(f"Skewness ({skew:.2f}) near symmetric")
        
        if kurt > 1:
            interp_parts.append(f"High kurtosis ({kurt:.2f}) = FAT TAILS (extreme events more likely)")
        elif kurt < -0.5:
            interp_parts.append(f"Low kurtosis ({kurt:.2f}) = thin tails (fewer extremes)")
        else:
            interp_parts.append(f"Kurtosis ({kurt:.2f}) near normal")
        
        if not is_normal:
            interp_parts.append(f"Distribution is NON-NORMAL (JB p={jb_pvalue:.4f})")
        else:
            interp_parts.append(f"Distribution consistent with normal (JB p={jb_pvalue:.4f})")
        
        return DistributionResult(
            mean=mean,
            std=std,
            skewness=skew,
            kurtosis=kurt,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            is_normal=is_normal,
            interpretation=". ".join(interp_parts)
        )
    
    # =========================================================================
    # GARCH VOLATILITY MODELING
    # =========================================================================
    
    def fit_garch(
        self,
        returns: np.ndarray,
        p: int = 1,
        q: int = 1
    ) -> GARCHResult:
        """
        Fit GARCH(p,q) model to returns for volatility analysis.
        
        GARCH captures volatility clustering - periods of high/low volatility
        tend to persist. This improves Monte Carlo realism.
        
        Model: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        
        Args:
            returns: Array of returns (preferably percentage returns)
            p: GARCH lag order
            q: ARCH lag order
        
        Returns:
            GARCHResult with model parameters and forecast
        """
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 50:
            return GARCHResult(
                omega=0, alpha=0, beta=0, persistence=0,
                unconditional_vol=np.std(returns) if len(returns) > 0 else 0,
                forecast_vol_1day=np.std(returns) if len(returns) > 0 else 0,
                aic=0, model_fit=False,
                interpretation="Insufficient data for GARCH (need 50+ observations)"
            )
        
        try:
            from arch import arch_model
            
            # Scale returns if they look like percentages
            if np.std(returns) > 0.5:
                # Likely percentage returns, use as-is
                scaled_returns = returns
            else:
                # Likely decimal returns, convert to percentage
                scaled_returns = returns * 100
            
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            omega = result.params.get('omega', 0)
            alpha = result.params.get('alpha[1]', 0)
            beta = result.params.get('beta[1]', 0)
            persistence = alpha + beta
            
            # Unconditional variance
            if persistence < 1:
                uncond_var = omega / (1 - persistence)
                uncond_vol = np.sqrt(uncond_var)
            else:
                uncond_vol = np.std(scaled_returns)
            
            # 1-day ahead forecast
            forecast = result.forecast(horizon=1)
            forecast_var = forecast.variance.values[-1, 0]
            forecast_vol = np.sqrt(forecast_var)
            
            # Interpretation
            if persistence > 0.95:
                interp = f"HIGH persistence ({persistence:.3f}) - volatility shocks are very persistent"
            elif persistence > 0.8:
                interp = f"MODERATE persistence ({persistence:.3f}) - typical for financial returns"
            else:
                interp = f"LOW persistence ({persistence:.3f}) - volatility mean-reverts quickly"
            
            if alpha > 0.15:
                interp += f". High alpha ({alpha:.3f}) = strong reaction to shocks"
            
            return GARCHResult(
                omega=omega,
                alpha=alpha,
                beta=beta,
                persistence=persistence,
                unconditional_vol=uncond_vol,
                forecast_vol_1day=forecast_vol,
                aic=result.aic,
                model_fit=True,
                interpretation=interp
            )
            
        except ImportError:
            # arch library not installed
            simple_vol = np.std(returns)
            return GARCHResult(
                omega=0, alpha=0, beta=0, persistence=0,
                unconditional_vol=simple_vol,
                forecast_vol_1day=simple_vol,
                aic=0, model_fit=False,
                interpretation="GARCH requires 'arch' library. Install with: pip install arch"
            )
        except Exception as e:
            simple_vol = np.std(returns)
            return GARCHResult(
                omega=0, alpha=0, beta=0, persistence=0,
                unconditional_vol=simple_vol,
                forecast_vol_1day=simple_vol,
                aic=0, model_fit=False,
                interpretation=f"GARCH fitting failed: {str(e)}"
            )
    
    # =========================================================================
    # VALUE AT RISK (VaR) SUITE
    # =========================================================================
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = None
    ) -> VaRResult:
        """
        Calculate Value at Risk using multiple methods.
        
        Methods:
        1. Historical VaR - Percentile of actual returns
        2. Parametric VaR - Assumes normal distribution
        3. Cornish-Fisher VaR - Adjusts for skew and kurtosis
        4. CVaR (Expected Shortfall) - Average loss beyond VaR
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (default: 0.95)
        
        Returns:
            VaRResult with all VaR measures
        """
        from scipy import stats
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 10:
            return VaRResult(
                confidence_level=confidence_level,
                historical_var=0, parametric_var=0,
                cornish_fisher_var=0, cvar_expected_shortfall=0,
                interpretation="Insufficient data for VaR calculation"
            )
        
        alpha = 1 - confidence_level  # e.g., 0.05 for 95% VaR
        
        # Historical VaR (percentile)
        hist_var = np.percentile(returns, alpha * 100)
        
        # Parametric VaR (assumes normal)
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(alpha)
        param_var = mean + z_score * std
        
        # Cornish-Fisher VaR (adjusted for skew/kurtosis)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Cornish-Fisher expansion
        z = z_score
        cf_z = (z + 
                (z**2 - 1) * skew / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * skew**2 / 36)
        cf_var = mean + cf_z * std
        
        # CVaR / Expected Shortfall (average of losses beyond VaR)
        losses_beyond_var = returns[returns <= hist_var]
        if len(losses_beyond_var) > 0:
            cvar = np.mean(losses_beyond_var)
        else:
            cvar = hist_var
        
        # Interpretation
        interp_parts = []
        interp_parts.append(f"At {confidence_level*100:.0f}% confidence:")
        interp_parts.append(f"Historical VaR: {hist_var:.2f}% (worst {alpha*100:.0f}% of days)")
        
        if abs(cf_var - param_var) > 0.5:
            interp_parts.append(
                f"Cornish-Fisher adjustment significant ({cf_var:.2f}% vs parametric {param_var:.2f}%) - "
                f"fat tails matter"
            )
        
        interp_parts.append(f"CVaR/ES: {cvar:.2f}% (expected loss on bad days)")
        
        return VaRResult(
            confidence_level=confidence_level,
            historical_var=hist_var,
            parametric_var=param_var,
            cornish_fisher_var=cf_var,
            cvar_expected_shortfall=cvar,
            interpretation=". ".join(interp_parts)
        )
    
    # =========================================================================
    # FULL STATISTICAL REPORT
    # =========================================================================
    
    def full_analysis(
        self,
        returns: np.ndarray,
        print_report: bool = True
    ) -> Dict:
        """
        Run all statistical analyses and optionally print report.
        
        Args:
            returns: Array of returns
            print_report: Whether to print formatted report
        
        Returns:
            Dict with all analysis results
        """
        serial = self.test_serial_dependence(returns)
        distribution = self.analyze_distribution(returns)
        garch = self.fit_garch(returns)
        var = self.calculate_var(returns)
        
        results = {
            'serial_dependence': serial,
            'distribution': distribution,
            'garch': garch,
            'var': var
        }
        
        if print_report:
            self.print_statistical_report(results)
        
        return results
    
    def print_statistical_report(self, results: Dict):
        """Print formatted statistical analysis report"""
        
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS REPORT")
        print("="*70)
        
        # Serial Dependence
        serial = results['serial_dependence']
        print("\n📊 SERIAL DEPENDENCE (Autocorrelation)")
        print("-"*70)
        print(f"  Lag-1 Autocorr:    {serial.autocorr_lag1:+.4f}")
        print(f"  Lag-5 Autocorr:    {serial.autocorr_lag5:+.4f}")
        print(f"  Ljung-Box Stat:    {serial.ljung_box_stat:.2f}")
        print(f"  Ljung-Box p-value: {serial.ljung_box_pvalue:.4f}")
        status = "⚠️  DEPENDENT" if serial.has_serial_dependence else "✅ INDEPENDENT"
        print(f"  Status:            {status}")
        print(f"  → {serial.interpretation}")
        
        # Distribution
        dist = results['distribution']
        print("\n📊 DISTRIBUTION ANALYSIS")
        print("-"*70)
        print(f"  Mean:              {dist.mean:+.4f}")
        print(f"  Std Dev:           {dist.std:.4f}")
        print(f"  Skewness:          {dist.skewness:+.4f}")
        print(f"  Excess Kurtosis:   {dist.kurtosis:+.4f}")
        print(f"  Jarque-Bera Stat:  {dist.jarque_bera_stat:.2f}")
        print(f"  JB p-value:        {dist.jarque_bera_pvalue:.4f}")
        status = "✅ NORMAL" if dist.is_normal else "⚠️  NON-NORMAL"
        print(f"  Status:            {status}")
        print(f"  → {dist.interpretation}")
        
        # GARCH
        garch = results['garch']
        print("\n📊 GARCH VOLATILITY MODEL")
        print("-"*70)
        if garch.model_fit:
            print(f"  Omega (const):     {garch.omega:.6f}")
            print(f"  Alpha (ARCH):      {garch.alpha:.4f}")
            print(f"  Beta (GARCH):      {garch.beta:.4f}")
            print(f"  Persistence:       {garch.persistence:.4f}")
            print(f"  Unconditional Vol: {garch.unconditional_vol:.4f}")
            print(f"  1-Day Forecast:    {garch.forecast_vol_1day:.4f}")
            print(f"  AIC:               {garch.aic:.2f}")
        print(f"  → {garch.interpretation}")
        
        # VaR
        var = results['var']
        print("\n📊 VALUE AT RISK")
        print("-"*70)
        print(f"  Confidence Level:  {var.confidence_level*100:.0f}%")
        print(f"  Historical VaR:    {var.historical_var:+.4f}")
        print(f"  Parametric VaR:    {var.parametric_var:+.4f}")
        print(f"  Cornish-Fisher:    {var.cornish_fisher_var:+.4f}")
        print(f"  CVaR (ES):         {var.cvar_expected_shortfall:+.4f}")
        print(f"  → {var.interpretation}")
        
        print("\n" + "="*70)


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


def quick_statistical_analysis(returns: np.ndarray, print_report: bool = True) -> Dict:
    """
    Quick statistical analysis of returns.
    
    Includes: serial dependence, skew/kurtosis, GARCH, VaR.
    
    Args:
        returns: Array of returns
        print_report: Whether to print formatted report
    
    Returns:
        Dict with all analysis results
    """
    analyzer = StatisticalAnalysis()
    return analyzer.full_analysis(returns, print_report=print_report)


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
    
    print("\nTesting Statistical Analysis...")
    stat_results = quick_statistical_analysis(returns, print_report=True)
    
    print("\n" + "="*70)
    print("Validation framework working!")
    print("="*70)