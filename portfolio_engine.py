# ==============================================================================
# portfolio_engine.py
# ==============================================================================
# Multi-Strategy Portfolio Allocation Engine
#
# Combines multiple validated strategies into a diversified portfolio.
# Uses correlation-aware allocation methods to maximize risk-adjusted returns.
#
# Allocation Methods:
# 1. Equal Weight - Simple 1/N allocation
# 2. Inverse Volatility - Weight by 1/volatility
# 3. Risk Parity - Equal risk contribution
# 4. Minimum Variance - Minimize portfolio variance
# 5. Maximum Sharpe - Maximize risk-adjusted return
# 6. Hierarchical Risk Parity (HRP) - Clustering-based allocation
#
# Usage:
#     from portfolio_engine import PortfolioEngine
#     
#     engine = PortfolioEngine()
#     
#     # Build portfolio from strategy equity curves
#     portfolio = engine.build_portfolio(
#         equity_curves={'strat1': curve1, 'strat2': curve2},
#         method='hrp'
#     )
#     
#     # Analyze portfolio
#     engine.print_portfolio_report(portfolio)
#
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class PortfolioResult:
    """Results from portfolio construction"""
    
    # Weights
    weights: Dict[str, float]
    method: str
    
    # Portfolio metrics
    expected_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    
    # Diversification
    diversification_ratio: float
    effective_n: float  # Effective number of strategies
    max_weight: float
    min_weight: float
    
    # Correlation analysis
    avg_correlation: float
    max_correlation: float
    correlation_matrix: pd.DataFrame
    
    # Risk contribution
    risk_contributions: Dict[str, float]
    marginal_risk: Dict[str, float]
    
    # Combined equity curve
    portfolio_equity: pd.Series
    portfolio_returns: pd.Series
    portfolio_drawdown: pd.Series
    max_drawdown: float


class PortfolioEngine:
    """
    Multi-strategy portfolio allocation engine.
    
    Takes equity curves from multiple strategies and allocates capital
    using various optimization methods.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annual
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        rebalance_threshold: float = 0.1  # Rebalance if weights drift >10%
    ):
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rebalance_threshold = rebalance_threshold
    
    def build_portfolio(
        self,
        equity_curves: Dict[str, pd.Series],
        method: str = 'hrp',
        target_volatility: float = None
    ) -> PortfolioResult:
        """
        Build portfolio from strategy equity curves.
        
        Args:
            equity_curves: Dict mapping strategy names to equity Series
            method: Allocation method ('equal', 'inverse_vol', 'risk_parity', 
                    'min_variance', 'max_sharpe', 'hrp')
            target_volatility: Optional target portfolio volatility (for scaling)
        
        Returns:
            PortfolioResult with weights and analysis
        """
        
        # Convert to returns DataFrame
        returns_df = self._equity_to_returns(equity_curves)
        
        if returns_df.empty or len(returns_df.columns) < 2:
            raise ValueError("Need at least 2 strategies for portfolio construction")
        
        # Calculate correlation and covariance
        corr_matrix = returns_df.corr()
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        # Calculate expected returns and volatilities
        expected_returns = returns_df.mean() * 252
        volatilities = returns_df.std() * np.sqrt(252)
        
        # Get weights based on method
        if method == 'equal':
            weights = self._equal_weight(returns_df)
        elif method == 'inverse_vol':
            weights = self._inverse_volatility(volatilities)
        elif method == 'risk_parity':
            weights = self._risk_parity(cov_matrix)
        elif method == 'min_variance':
            weights = self._minimum_variance(cov_matrix)
        elif method == 'max_sharpe':
            weights = self._maximum_sharpe(expected_returns, cov_matrix)
        elif method == 'hrp':
            weights = self._hierarchical_risk_parity(returns_df)
        else:
            raise ValueError(f"Unknown method: {method}. Use: equal, inverse_vol, risk_parity, min_variance, max_sharpe, hrp")
        
        # Apply constraints
        weights = self._apply_constraints(weights)
        
        # Calculate portfolio metrics
        port_return = np.sum(expected_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Scale to target volatility if specified
        if target_volatility and port_vol > 0:
            scale = target_volatility / port_vol
            weights = weights * scale
            weights = self._apply_constraints(weights)  # Re-apply constraints
            port_vol = target_volatility
        
        # Diversification metrics
        weighted_vol = np.sum(volatilities * weights)
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        
        # Effective N (inverse of HHI)
        hhi = np.sum(weights ** 2)
        effective_n = 1 / hhi if hhi > 0 else len(weights)
        
        # Risk contributions
        risk_contrib = self._calculate_risk_contributions(weights, cov_matrix)
        marginal_risk = self._calculate_marginal_risk(weights, cov_matrix)
        
        # Build combined equity curve
        weights_dict = dict(zip(returns_df.columns, weights))
        port_equity, port_returns, port_dd, max_dd = self._build_portfolio_equity(
            equity_curves, weights_dict
        )
        
        # Average and max correlation
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        avg_corr = np.mean(corr_values) if len(corr_values) > 0 else 0
        max_corr = np.max(corr_values) if len(corr_values) > 0 else 0
        
        return PortfolioResult(
            weights=weights_dict,
            method=method,
            expected_return=port_return,
            portfolio_volatility=port_vol,
            portfolio_sharpe=port_sharpe,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            max_weight=np.max(weights),
            min_weight=np.min(weights),
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            correlation_matrix=corr_matrix,
            risk_contributions=dict(zip(returns_df.columns, risk_contrib)),
            marginal_risk=dict(zip(returns_df.columns, marginal_risk)),
            portfolio_equity=port_equity,
            portfolio_returns=port_returns,
            portfolio_drawdown=port_dd,
            max_drawdown=max_dd
        )
    
    def _equity_to_returns(self, equity_curves: Dict[str, pd.Series]) -> pd.DataFrame:
        """Convert equity curves to returns DataFrame"""
        returns = {}
        for name, equity in equity_curves.items():
            equity = pd.Series(equity)
            ret = equity.pct_change().dropna()
            returns[name] = ret
        
        return pd.DataFrame(returns).dropna()
    
    def _equal_weight(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Equal weight allocation"""
        n = len(returns_df.columns)
        return np.ones(n) / n
    
    def _inverse_volatility(self, volatilities: pd.Series) -> np.ndarray:
        """Inverse volatility weighting"""
        inv_vol = 1 / volatilities
        return (inv_vol / inv_vol.sum()).values
    
    def _risk_parity(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Risk parity allocation - equal risk contribution from each strategy.
        Uses iterative approach.
        """
        n = len(cov_matrix)
        weights = np.ones(n) / n
        
        for _ in range(100):  # Iterative optimization
            risk_contrib = self._calculate_risk_contributions(weights, cov_matrix)
            target_risk = 1 / n
            
            # Adjust weights based on risk contribution
            adjustments = target_risk / (risk_contrib + 1e-10)
            weights = weights * adjustments
            weights = weights / weights.sum()
        
        return weights
    
    def _minimum_variance(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Minimum variance portfolio"""
        try:
            from scipy.optimize import minimize
            
            n = len(cov_matrix)
            
            def portfolio_variance(w):
                return np.dot(w.T, np.dot(cov_matrix.values, w))
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
            
            result = minimize(
                portfolio_variance,
                np.ones(n) / n,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x
            
        except ImportError:
            warnings.warn("scipy not available, using inverse vol instead")
            return self._inverse_volatility(pd.Series(np.diag(cov_matrix) ** 0.5))
    
    def _maximum_sharpe(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Maximum Sharpe ratio portfolio"""
        try:
            from scipy.optimize import minimize
            
            n = len(cov_matrix)
            
            def neg_sharpe(w):
                port_ret = np.sum(expected_returns.values * w)
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))
                return -(port_ret - self.risk_free_rate) / (port_vol + 1e-10)
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
            
            result = minimize(
                neg_sharpe,
                np.ones(n) / n,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x
            
        except ImportError:
            warnings.warn("scipy not available, using equal weight instead")
            return self._equal_weight(pd.DataFrame(index=range(n)))
    
    def _hierarchical_risk_parity(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) allocation.
        
        Uses hierarchical clustering to build a diversified portfolio
        that doesn't rely on unstable covariance matrix inversion.
        """
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            
            # Calculate correlation matrix
            corr = returns_df.corr()
            
            # Convert correlation to distance
            dist = np.sqrt(0.5 * (1 - corr))
            
            # Hierarchical clustering
            link = linkage(squareform(dist.values), method='ward')
            sort_idx = leaves_list(link)
            
            # Reorder correlation matrix
            sorted_corr = corr.iloc[sort_idx, sort_idx]
            
            # Recursive bisection for weights
            weights = self._hrp_recursive_bisection(returns_df, sort_idx)
            
            return weights
            
        except ImportError:
            warnings.warn("scipy not available for HRP, using risk parity instead")
            cov_matrix = returns_df.cov() * 252
            return self._risk_parity(cov_matrix)
    
    def _hrp_recursive_bisection(self, returns_df: pd.DataFrame, sort_idx: np.ndarray) -> np.ndarray:
        """Recursive bisection for HRP weights"""
        n = len(sort_idx)
        weights = np.ones(n)
        
        cov = returns_df.cov().values * 252
        
        def cluster_variance(indices):
            cov_slice = cov[np.ix_(indices, indices)]
            w = np.ones(len(indices)) / len(indices)
            return np.dot(w.T, np.dot(cov_slice, w))
        
        def recursive_bisect(indices, weight):
            if len(indices) == 1:
                weights[indices[0]] = weight
                return
            
            mid = len(indices) // 2
            left = indices[:mid]
            right = indices[mid:]
            
            var_left = cluster_variance(left)
            var_right = cluster_variance(right)
            
            alpha = 1 - var_left / (var_left + var_right)
            
            recursive_bisect(left, weight * alpha)
            recursive_bisect(right, weight * (1 - alpha))
        
        recursive_bisect(list(sort_idx), 1.0)
        
        # Reorder back to original
        final_weights = np.zeros(n)
        for i, idx in enumerate(sort_idx):
            final_weights[idx] = weights[i]
        
        return final_weights / final_weights.sum()
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply min/max weight constraints"""
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()  # Renormalize
        return weights
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate each strategy's contribution to portfolio risk"""
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        
        if port_vol == 0:
            return np.zeros(len(weights))
        
        marginal_contrib = np.dot(cov_matrix.values, weights)
        risk_contrib = weights * marginal_contrib / port_vol
        
        # Normalize to sum to 1
        return risk_contrib / risk_contrib.sum()
    
    def _calculate_marginal_risk(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate marginal risk contribution"""
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        
        if port_vol == 0:
            return np.zeros(len(weights))
        
        return np.dot(cov_matrix.values, weights) / port_vol
    
    def _build_portfolio_equity(
        self,
        equity_curves: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> Tuple[pd.Series, pd.Series, pd.Series, float]:
        """Build combined portfolio equity curve"""
        
        # Align all equity curves
        aligned = pd.DataFrame(equity_curves)
        aligned = aligned.ffill().bfill()
        
        # Normalize to start at 1
        normalized = aligned / aligned.iloc[0]
        
        # Apply weights
        portfolio = sum(normalized[name] * weight for name, weight in weights.items())
        
        # Calculate returns
        returns = portfolio.pct_change().dropna()
        
        # Calculate drawdown
        peak = portfolio.expanding().max()
        drawdown = (portfolio - peak) / peak
        max_dd = drawdown.min() * 100
        
        return portfolio, returns, drawdown, max_dd
    
    def compare_methods(
        self,
        equity_curves: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Compare all allocation methods"""
        
        methods = ['equal', 'inverse_vol', 'risk_parity', 'min_variance', 'max_sharpe', 'hrp']
        results = []
        
        for method in methods:
            try:
                portfolio = self.build_portfolio(equity_curves, method=method)
                results.append({
                    'method': method,
                    'expected_return': portfolio.expected_return * 100,
                    'volatility': portfolio.portfolio_volatility * 100,
                    'sharpe': portfolio.portfolio_sharpe,
                    'max_drawdown': portfolio.max_drawdown,
                    'diversification_ratio': portfolio.diversification_ratio,
                    'effective_n': portfolio.effective_n,
                    'max_weight': portfolio.max_weight * 100,
                    'avg_correlation': portfolio.avg_correlation
                })
            except Exception as e:
                warnings.warn(f"Method {method} failed: {e}")
        
        return pd.DataFrame(results)
    
    def print_portfolio_report(self, result: PortfolioResult):
        """Print formatted portfolio report"""
        
        print("\n" + "="*70)
        print(f"PORTFOLIO REPORT - {result.method.upper()} ALLOCATION")
        print("="*70)
        
        print("\n[STATS] WEIGHTS:")
        print("-"*70)
        for name, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(weight * 50)
            print(f"  {name:<20} {weight*100:>6.1f}%  {bar}")
        
        print("\n[STATS] PORTFOLIO METRICS:")
        print("-"*70)
        print(f"  Expected Return:      {result.expected_return*100:>8.2f}% annual")
        print(f"  Volatility:           {result.portfolio_volatility*100:>8.2f}% annual")
        print(f"  Sharpe Ratio:         {result.portfolio_sharpe:>8.2f}")
        print(f"  Max Drawdown:         {result.max_drawdown:>8.2f}%")
        
        print("\n[STATS] DIVERSIFICATION:")
        print("-"*70)
        print(f"  Diversification Ratio: {result.diversification_ratio:>7.2f}")
        print(f"  Effective N:           {result.effective_n:>7.2f} / {len(result.weights)}")
        print(f"  Avg Correlation:       {result.avg_correlation:>7.2f}")
        print(f"  Max Correlation:       {result.max_correlation:>7.2f}")
        
        print("\n[STATS] RISK CONTRIBUTIONS:")
        print("-"*70)
        for name, contrib in sorted(result.risk_contributions.items(), key=lambda x: -x[1]):
            bar = "█" * int(contrib * 50)
            print(f"  {name:<20} {contrib*100:>6.1f}%  {bar}")
        
        print("\n" + "="*70)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_portfolio(equity_curves: Dict[str, pd.Series], method: str = 'hrp') -> PortfolioResult:
    """Quick portfolio construction"""
    engine = PortfolioEngine()
    return engine.build_portfolio(equity_curves, method=method)


def compare_allocations(equity_curves: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compare all allocation methods"""
    engine = PortfolioEngine()
    return engine.compare_methods(equity_curves)


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PORTFOLIO ENGINE TEST")
    print("="*70)
    
    # Create sample equity curves
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Strategy 1: Steady performer
    returns1 = np.random.normal(0.0003, 0.01, 252)
    equity1 = pd.Series((1 + returns1).cumprod() * 10000, index=dates)
    
    # Strategy 2: Volatile performer
    returns2 = np.random.normal(0.0005, 0.02, 252)
    equity2 = pd.Series((1 + returns2).cumprod() * 10000, index=dates)
    
    # Strategy 3: Low correlation
    returns3 = np.random.normal(0.0002, 0.015, 252)
    equity3 = pd.Series((1 + returns3).cumprod() * 10000, index=dates)
    
    equity_curves = {
        'SteadyStrategy': equity1,
        'VolatileStrategy': equity2,
        'DiversifierStrategy': equity3
    }
    
    # Build portfolio
    engine = PortfolioEngine()
    
    print("\nTesting HRP allocation...")
    portfolio = engine.build_portfolio(equity_curves, method='hrp')
    engine.print_portfolio_report(portfolio)
    
    print("\nComparing all methods...")
    comparison = engine.compare_methods(equity_curves)
    print(comparison.to_string(index=False))
    
    print("\n" + "="*70)
    print("Portfolio engine working!")
    print("="*70)