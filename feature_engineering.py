# ==============================================================================
# feature_engineering.py
# ==============================================================================
# Feature Engineering & Aggregation Module
#
# Collects metrics from ALL analysis modules into a unified feature table.
# Each row represents one strategy, with columns for every metric.
#
# This enables:
# 1. Cross-strategy comparison on all dimensions
# 2. Input for ML meta-model (survival prediction)
# 3. Portfolio construction based on feature profiles
# 4. Automated strategy selection and ranking
#
# Usage:
#     from feature_engineering import FeatureEngineer
#     
#     engineer = FeatureEngineer()
#     
#     # Build features for a single strategy
#     features = engineer.build_features(
#         backtest_result=result,
#         trades_df=trades,
#         price_data=data
#     )
#     
#     # Build feature table for multiple strategies
#     feature_table = engineer.build_feature_table(strategy_results)
#
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings


@dataclass
class StrategyFeatures:
    """Complete feature set for a single strategy"""
    
    # =========================================================================
    # IDENTIFIERS
    # =========================================================================
    strategy_name: str
    symbol: str
    timeframe: str
    
    # =========================================================================
    # BASIC PERFORMANCE (from backtester)
    # =========================================================================
    total_return_pct: float
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float
    total_trades: int
    win_rate: Optional[float]
    profit_factor: Optional[float]
    
    # =========================================================================
    # TRADE ANALYTICS (from backtester)
    # =========================================================================
    trades_per_day: float
    avg_trade_duration_bars: float
    avg_trade_return_pct: float
    time_in_market_pct: float
    
    # =========================================================================
    # STATISTICAL ANALYSIS (from validation_framework)
    # =========================================================================
    # Serial dependence
    autocorr_lag1: float
    has_serial_dependence: bool
    
    # Distribution
    skewness: float
    kurtosis: float
    is_normal_distribution: bool
    
    # GARCH
    garch_persistence: float
    garch_forecast_vol: float
    
    # VaR
    var_95_historical: float
    var_95_cornish_fisher: float
    cvar_95: float
    
    # =========================================================================
    # ROBUSTNESS (from robustness_tests)
    # =========================================================================
    latency_sensitivity: float  # Return degradation per bar delay
    slippage_breakeven: Optional[float]  # Cost multiplier at breakeven
    combined_stress_survival: float  # % of stress scenarios profitable
    
    # =========================================================================
    # STATISTICAL SIGNIFICANCE (from permutation_tests)
    # =========================================================================
    permutation_pvalue: Optional[float]
    is_significant: bool
    
    # =========================================================================
    # REGIME PERFORMANCE (from regime_classifier)
    # =========================================================================
    bull_return_pct: Optional[float]
    bear_return_pct: Optional[float]
    ranging_return_pct: Optional[float]
    high_vol_return_pct: Optional[float]
    
    # =========================================================================
    # PROP FIRM COMPLIANCE (from ftmo_compliance)
    # =========================================================================
    ftmo_pass_rate: Optional[float]
    ftmo_primary_fail_reason: Optional[str]
    
    # =========================================================================
    # COST ANALYSIS (from cost_adjusted_scoring)
    # =========================================================================
    net_return_pct: float
    total_cost_pct: float
    cost_ratio: float  # Costs as % of gross profit
    is_cost_viable: bool
    
    # =========================================================================
    # METADATA
    # =========================================================================
    feature_timestamp: str
    bars_tested: int
    start_date: str
    end_date: str


class FeatureEngineer:
    """
    Aggregates metrics from all analysis modules into a unified feature set.
    
    This is the central hub that pulls from:
    - backtester_multi_timeframe.py
    - validation_framework.py
    - robustness_tests.py
    - permutation_tests.py
    - regime_classifier.py
    - ftmo_compliance.py
    - cost_adjusted_scoring.py
    """
    
    def __init__(self):
        # Lazy imports to avoid circular dependencies
        self._validation_framework = None
        self._statistical_analysis = None
        self._ftmo_checker = None
        self._cost_scorer = None
    
    @property
    def validation_framework(self):
        if self._validation_framework is None:
            from validation_framework import ValidationFramework
            self._validation_framework = ValidationFramework()
        return self._validation_framework
    
    @property
    def statistical_analysis(self):
        if self._statistical_analysis is None:
            from validation_framework import StatisticalAnalysis
            self._statistical_analysis = StatisticalAnalysis()
        return self._statistical_analysis
    
    @property
    def ftmo_checker(self):
        if self._ftmo_checker is None:
            from ftmo_compliance import FTMOComplianceChecker
            self._ftmo_checker = FTMOComplianceChecker()
        return self._ftmo_checker
    
    @property
    def cost_scorer(self):
        if self._cost_scorer is None:
            from cost_adjusted_scoring import CostAdjustedScorer
            self._cost_scorer = CostAdjustedScorer()
        return self._cost_scorer
    
    def build_features(
        self,
        backtest_result: Dict,
        trades_df: pd.DataFrame = None,
        price_data: pd.DataFrame = None,
        robustness_results: Dict = None,
        permutation_result: Any = None,
        regime_results: Dict = None,
        ftmo_pass_rate: float = None,
        run_statistical_analysis: bool = True,
        run_ftmo_simulation: bool = False,
        ftmo_n_simulations: int = 100
    ) -> StrategyFeatures:
        """
        Build complete feature set for a single strategy.
        
        Args:
            backtest_result: Dict from backtester (must include 'trades' if trades_df not provided)
            trades_df: DataFrame of individual trades (optional if in backtest_result)
            price_data: OHLCV DataFrame for additional analysis
            robustness_results: Dict from robustness_tests (optional)
            permutation_result: PermutationResult from permutation_tests (optional)
            regime_results: Dict with regime performance breakdown (optional)
            ftmo_pass_rate: Pre-computed FTMO pass rate (optional)
            run_statistical_analysis: Whether to run statistical tests
            run_ftmo_simulation: Whether to run FTMO Monte Carlo (slow)
            ftmo_n_simulations: Number of FTMO simulations if running
        
        Returns:
            StrategyFeatures dataclass with all metrics
        """
        
        # Extract trades DataFrame
        if trades_df is None:
            trades_df = pd.DataFrame(backtest_result.get('trades', []))
        
        # Get returns array for statistical analysis
        if len(trades_df) > 0 and 'return_pct' in trades_df.columns:
            returns = trades_df['return_pct'].dropna().values
        else:
            returns = np.array([])
        
        # =====================================================================
        # BASIC PERFORMANCE
        # =====================================================================
        total_return = backtest_result.get('total_return_pct', 0)
        sharpe = backtest_result.get('sharpe_ratio')
        max_dd = backtest_result.get('max_drawdown_pct', 0)
        total_trades = backtest_result.get('total_trades', 0)
        win_rate = backtest_result.get('win_rate')
        profit_factor = backtest_result.get('profit_factor')
        
        # =====================================================================
        # TRADE ANALYTICS
        # =====================================================================
        trades_per_day = backtest_result.get('trades_per_day', 0)
        avg_duration = backtest_result.get('avg_trade_duration_bars', 0)
        avg_trade_return = backtest_result.get('avg_trade_return_pct', 0)
        time_in_market = backtest_result.get('time_in_market_pct', 0)
        
        # =====================================================================
        # STATISTICAL ANALYSIS
        # =====================================================================
        autocorr_lag1 = 0.0
        has_serial_dep = False
        skewness = 0.0
        kurtosis = 0.0
        is_normal = True
        garch_persistence = 0.0
        garch_forecast = 0.0
        var_95_hist = 0.0
        var_95_cf = 0.0
        cvar_95 = 0.0
        
        if run_statistical_analysis and len(returns) >= 20:
            try:
                # Serial dependence
                serial_result = self.statistical_analysis.test_serial_dependence(returns)
                autocorr_lag1 = serial_result.autocorr_lag1
                has_serial_dep = serial_result.has_serial_dependence
                
                # Distribution
                dist_result = self.statistical_analysis.analyze_distribution(returns)
                skewness = dist_result.skewness
                kurtosis = dist_result.kurtosis
                is_normal = dist_result.is_normal
                
                # GARCH
                garch_result = self.statistical_analysis.fit_garch(returns)
                garch_persistence = garch_result.persistence
                garch_forecast = garch_result.forecast_vol_1day
                
                # VaR
                var_result = self.statistical_analysis.calculate_var(returns)
                var_95_hist = var_result.historical_var
                var_95_cf = var_result.cornish_fisher_var
                cvar_95 = var_result.cvar_expected_shortfall
                
            except Exception as e:
                warnings.warn(f"Statistical analysis failed: {e}")
        
        # =====================================================================
        # ROBUSTNESS
        # =====================================================================
        latency_sens = 0.0
        slippage_be = None
        stress_survival = 0.0
        
        if robustness_results:
            if 'latency' in robustness_results:
                latency_sens = robustness_results['latency'].degradation_per_bar
            if 'slippage' in robustness_results:
                slippage_be = robustness_results['slippage'].breakeven_multiplier
            if 'combined' in robustness_results:
                stress_survival = robustness_results['combined'].survival_rate
        
        # =====================================================================
        # PERMUTATION TEST
        # =====================================================================
        perm_pvalue = None
        is_significant = False
        
        if permutation_result:
            perm_pvalue = permutation_result.p_value
            is_significant = permutation_result.is_significant
        
        # =====================================================================
        # REGIME PERFORMANCE
        # =====================================================================
        bull_ret = None
        bear_ret = None
        ranging_ret = None
        high_vol_ret = None
        
        if regime_results:
            bull_ret = regime_results.get('BULL', {}).get('return_pct')
            bear_ret = regime_results.get('BEAR', {}).get('return_pct')
            ranging_ret = regime_results.get('RANGING', {}).get('return_pct')
            high_vol_ret = regime_results.get('HIGH_VOL', {}).get('return_pct')
        
        # =====================================================================
        # FTMO COMPLIANCE
        # =====================================================================
        ftmo_rate = ftmo_pass_rate
        ftmo_fail_reason = None
        
        if run_ftmo_simulation and len(trades_df) >= 4:
            try:
                ftmo_result = self.ftmo_checker.simulate_pass_rate(
                    trades_df, 
                    account_size=100_000,
                    n_simulations=ftmo_n_simulations
                )
                ftmo_rate = ftmo_result['pass_rate']
                ftmo_fail_reason = ftmo_result.get('primary_fail_reason')
            except Exception as e:
                warnings.warn(f"FTMO simulation failed: {e}")
        
        # =====================================================================
        # COST ANALYSIS
        # =====================================================================
        net_return = total_return
        total_cost = 0.0
        cost_ratio = 0.0
        is_viable = True
        
        try:
            adjusted = self.cost_scorer.adjust_result(backtest_result)
            net_return = adjusted.net_return_pct
            total_cost = adjusted.total_cost_pct
            cost_ratio = adjusted.cost_ratio
            is_viable = adjusted.is_viable
        except Exception as e:
            warnings.warn(f"Cost analysis failed: {e}")
        
        # =====================================================================
        # BUILD FEATURES
        # =====================================================================
        return StrategyFeatures(
            # Identifiers
            strategy_name=backtest_result.get('strategy_name', 'Unknown'),
            symbol=backtest_result.get('symbol', 'Unknown'),
            timeframe=backtest_result.get('timeframe', 'Unknown'),
            
            # Basic performance
            total_return_pct=total_return,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            
            # Trade analytics
            trades_per_day=trades_per_day,
            avg_trade_duration_bars=avg_duration,
            avg_trade_return_pct=avg_trade_return,
            time_in_market_pct=time_in_market,
            
            # Statistical analysis
            autocorr_lag1=autocorr_lag1,
            has_serial_dependence=has_serial_dep,
            skewness=skewness,
            kurtosis=kurtosis,
            is_normal_distribution=is_normal,
            garch_persistence=garch_persistence,
            garch_forecast_vol=garch_forecast,
            var_95_historical=var_95_hist,
            var_95_cornish_fisher=var_95_cf,
            cvar_95=cvar_95,
            
            # Robustness
            latency_sensitivity=latency_sens,
            slippage_breakeven=slippage_be,
            combined_stress_survival=stress_survival,
            
            # Significance
            permutation_pvalue=perm_pvalue,
            is_significant=is_significant,
            
            # Regime
            bull_return_pct=bull_ret,
            bear_return_pct=bear_ret,
            ranging_return_pct=ranging_ret,
            high_vol_return_pct=high_vol_ret,
            
            # FTMO
            ftmo_pass_rate=ftmo_rate,
            ftmo_primary_fail_reason=ftmo_fail_reason,
            
            # Cost
            net_return_pct=net_return,
            total_cost_pct=total_cost,
            cost_ratio=cost_ratio,
            is_cost_viable=is_viable,
            
            # Metadata
            feature_timestamp=datetime.now().isoformat(),
            bars_tested=backtest_result.get('bars_tested', 0),
            start_date=backtest_result.get('start_date', ''),
            end_date=backtest_result.get('end_date', '')
        )
    
    def build_feature_table(
        self,
        strategy_results: List[Dict],
        **kwargs
    ) -> pd.DataFrame:
        """
        Build feature table for multiple strategies.
        
        Args:
            strategy_results: List of backtest result dicts
            **kwargs: Additional args passed to build_features
        
        Returns:
            DataFrame with one row per strategy
        """
        features_list = []
        
        for i, result in enumerate(strategy_results):
            print(f"Building features for strategy {i+1}/{len(strategy_results)}: {result.get('strategy_name', 'Unknown')}")
            
            try:
                features = self.build_features(result, **kwargs)
                features_list.append(asdict(features))
            except Exception as e:
                warnings.warn(f"Failed to build features for {result.get('strategy_name')}: {e}")
        
        return pd.DataFrame(features_list)
    
    def features_to_dict(self, features: StrategyFeatures) -> Dict:
        """Convert StrategyFeatures to dictionary"""
        return asdict(features)
    
    def print_feature_summary(self, features: StrategyFeatures):
        """Print formatted feature summary"""
        
        print("\n" + "="*70)
        print(f"FEATURE SUMMARY: {features.strategy_name}")
        print("="*70)
        print(f"Symbol: {features.symbol} | Timeframe: {features.timeframe}")
        print(f"Period: {features.start_date} to {features.end_date}")
        print("-"*70)
        
        print("\n📊 PERFORMANCE:")
        print(f"  Return:        {features.total_return_pct:+.2f}%")
        print(f"  Sharpe:        {features.sharpe_ratio:.2f}" if features.sharpe_ratio else "  Sharpe:        N/A")
        print(f"  Max Drawdown:  {features.max_drawdown_pct:.2f}%")
        print(f"  Win Rate:      {features.win_rate:.1f}%" if features.win_rate else "  Win Rate:      N/A")
        
        print("\n📊 TRADE ANALYTICS:")
        print(f"  Total Trades:      {features.total_trades}")
        print(f"  Trades/Day:        {features.trades_per_day:.2f}")
        print(f"  Avg Duration:      {features.avg_trade_duration_bars:.1f} bars")
        print(f"  Time in Market:    {features.time_in_market_pct:.1f}%")
        
        print("\n📊 STATISTICAL:")
        print(f"  Serial Dependence: {'⚠️  YES' if features.has_serial_dependence else '✅ NO'}")
        print(f"  Skewness:          {features.skewness:+.2f}")
        print(f"  Kurtosis:          {features.kurtosis:+.2f}")
        print(f"  VaR (95%):         {features.var_95_historical:.2f}%")
        print(f"  CVaR:              {features.cvar_95:.2f}%")
        
        if features.ftmo_pass_rate is not None:
            print("\n📊 FTMO:")
            print(f"  Pass Rate:         {features.ftmo_pass_rate*100:.1f}%")
            if features.ftmo_primary_fail_reason:
                print(f"  Primary Fail:      {features.ftmo_primary_fail_reason}")
        
        print("\n📊 COST ANALYSIS:")
        print(f"  Net Return:        {features.net_return_pct:+.2f}%")
        print(f"  Total Costs:       {features.total_cost_pct:.2f}%")
        print(f"  Cost Viable:       {'✅ YES' if features.is_cost_viable else '❌ NO'}")
        
        print("="*70 + "\n")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_features(backtest_result: Dict, trades_df: pd.DataFrame = None) -> StrategyFeatures:
    """Quick feature extraction from a backtest result"""
    engineer = FeatureEngineer()
    return engineer.build_features(backtest_result, trades_df=trades_df)


def feature_table_from_results(results: List[Dict]) -> pd.DataFrame:
    """Build feature table from list of backtest results"""
    engineer = FeatureEngineer()
    return engineer.build_feature_table(results)


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FEATURE ENGINEERING MODULE TEST")
    print("="*70)
    
    # Create sample backtest result
    sample_result = {
        'strategy_name': 'TestStrategy',
        'symbol': 'EUR-USD',
        'timeframe': '1hour',
        'total_return_pct': 15.5,
        'sharpe_ratio': 1.2,
        'max_drawdown_pct': 8.3,
        'total_trades': 45,
        'win_rate': 55.0,
        'profit_factor': 1.8,
        'trades_per_day': 0.5,
        'avg_trade_duration_bars': 12,
        'avg_trade_return_pct': 0.34,
        'time_in_market_pct': 35.0,
        'bars_tested': 5000,
        'start_date': '2023-01-01',
        'end_date': '2024-01-01'
    }
    
    # Create sample trades
    np.random.seed(42)
    sample_trades = pd.DataFrame({
        'return_pct': np.random.normal(0.34, 2.0, 45),
        'duration_bars': np.random.randint(5, 30, 45)
    })
    
    # Build features
    engineer = FeatureEngineer()
    features = engineer.build_features(
        sample_result, 
        trades_df=sample_trades,
        run_statistical_analysis=True,
        run_ftmo_simulation=False
    )
    
    # Print summary
    engineer.print_feature_summary(features)
    
    print("✅ Feature engineering module working!")
    print("="*70)