# ==============================================================================
# cost_adjusted_scoring.py
# ==============================================================================
# Week 4: Cost-Adjusted Mutation Scoring
#
# Scores strategy mutations after applying realistic costs, not gross returns.
# This prevents selecting strategies that look good on paper but fail in reality.
#
# Realistic costs include:
# - Commission/fees (per trade)
# - Spread (bid-ask)
# - Slippage (market impact)
# - Overnight financing (for positions held overnight)
#
# Usage:
#     from cost_adjusted_scoring import CostAdjustedScorer
#     
#     scorer = CostAdjustedScorer()
#     
#     # Score a single result
#     adjusted = scorer.adjust_result(backtest_result)
#     
#     # Score and rank multiple variants
#     ranked = scorer.rank_variants(variant_results)
#
# ==============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CostProfile:
    """Cost profile for an asset class"""
    name: str
    commission_pct: float      # Per-trade commission (%)
    spread_pct: float          # Typical spread (%)
    slippage_pct: float        # Expected slippage (%)
    overnight_rate: float      # Daily financing rate (%)
    min_commission: float      # Minimum commission per trade ($)


@dataclass
class AdjustedResult:
    """Result after cost adjustment"""
    strategy_name: str
    symbol: str
    timeframe: str
    
    # Original metrics
    gross_return_pct: float
    gross_sharpe: Optional[float]
    
    # Cost breakdown
    commission_cost_pct: float
    spread_cost_pct: float
    slippage_cost_pct: float
    financing_cost_pct: float
    total_cost_pct: float
    
    # Adjusted metrics
    net_return_pct: float
    net_sharpe: Optional[float]
    cost_ratio: float  # Total cost as % of gross profit
    
    # Trade statistics
    total_trades: int
    avg_holding_period: float
    turnover: float
    
    # Viability assessment
    is_viable: bool
    viability_reason: str


# ==============================================================================
# DEFAULT COST PROFILES
# ==============================================================================

COST_PROFILES = {
    'forex': CostProfile(
        name='Forex',
        commission_pct=0.002,      # 0.2 bps typical for retail
        spread_pct=0.01,           # 1 pip on majors ≈ 0.01%
        slippage_pct=0.005,        # 0.5 pip typical
        overnight_rate=0.00008,    # ~3% annual / 365
        min_commission=0
    ),
    'crypto': CostProfile(
        name='Crypto',
        commission_pct=0.1,        # 0.1% typical exchange fee
        spread_pct=0.05,           # Wider spreads on crypto
        slippage_pct=0.02,         # More volatile
        overnight_rate=0,          # No overnight for spot
        min_commission=0
    ),
    'stock': CostProfile(
        name='Stocks',
        commission_pct=0,          # Most brokers commission-free now
        spread_pct=0.01,           # 1 cent on $100 stock
        slippage_pct=0.01,         # 
        overnight_rate=0.0002,     # Margin interest if leveraged
        min_commission=0
    ),
    'futures': CostProfile(
        name='Futures',
        commission_pct=0.005,      # Low commissions
        spread_pct=0.005,          # Tight spreads
        slippage_pct=0.01,         # Can be significant in size
        overnight_rate=0,          # Embedded in contract
        min_commission=2.50        # Per contract minimum
    ),
    'conservative': CostProfile(
        name='Conservative',
        commission_pct=0.1,        # Assume worst case
        spread_pct=0.1,            # Wide spreads
        slippage_pct=0.05,         # Significant slippage
        overnight_rate=0.0003,     # High financing
        min_commission=5
    ),
}


class CostAdjustedScorer:
    """
    Adjusts backtest results for realistic trading costs.
    
    Most backtests only account for commission. Real trading has:
    - Spread (you always buy high, sell low)
    - Slippage (your order moves the market)
    - Financing (overnight positions cost money)
    """
    
    def __init__(
        self,
        default_profile: str = 'forex',
        custom_profiles: Dict[str, CostProfile] = None
    ):
        """
        Args:
            default_profile: Which cost profile to use by default
            custom_profiles: Additional custom cost profiles
        """
        self.profiles = COST_PROFILES.copy()
        if custom_profiles:
            self.profiles.update(custom_profiles)
        
        self.default_profile = default_profile
    
    def get_profile(self, symbol: str) -> CostProfile:
        """Get appropriate cost profile for a symbol"""
        
        symbol_upper = symbol.upper()
        
        # Auto-detect asset class
        if any(x in symbol_upper for x in ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE']):
            return self.profiles['crypto']
        elif any(x in symbol_upper for x in ['EUR', 'GBP', 'JPY', 'USD', 'CHF', 'AUD', 'NZD', 'CAD']):
            return self.profiles['forex']
        elif any(x in symbol_upper for x in ['SPX', 'NDX', 'DJI', 'ES', 'NQ']):
            return self.profiles['futures']
        else:
            return self.profiles[self.default_profile]
    
    def adjust_result(
        self,
        result: Dict,
        profile: CostProfile = None,
        avg_holding_bars: float = None
    ) -> AdjustedResult:
        """
        Adjust a single backtest result for realistic costs.
        
        Args:
            result: Backtest result dictionary
            profile: Cost profile to use (auto-detected if None)
            avg_holding_bars: Average bars per trade (estimated if None)
        
        Returns:
            AdjustedResult with net metrics
        """
        
        symbol = result.get('symbol', 'UNKNOWN')
        
        if profile is None:
            profile = self.get_profile(symbol)
        
        # Extract metrics
        gross_return = result.get('total_return_pct', 0)
        gross_sharpe = result.get('sharpe_ratio')
        total_trades = result.get('total_trades', 0)
        bars_tested = result.get('bars_tested', 1000)
        
        # Estimate holding period if not provided
        if avg_holding_bars is None:
            if total_trades > 0:
                avg_holding_bars = bars_tested / total_trades / 2  # Rough estimate
            else:
                avg_holding_bars = 0
        
        # Calculate round-trip cost per trade
        # Each trade = buy + sell, so costs apply twice for spread/slippage
        cost_per_trade = (
            profile.commission_pct * 2 +  # Commission both ways
            profile.spread_pct * 2 +       # Spread both ways
            profile.slippage_pct * 2       # Slippage both ways
        )
        
        # Total trading costs
        commission_cost = total_trades * profile.commission_pct * 2
        spread_cost = total_trades * profile.spread_pct * 2
        slippage_cost = total_trades * profile.slippage_pct * 2
        
        # Financing costs (for overnight positions)
        # Estimate: assume 50% of time in position, held overnight
        if avg_holding_bars > 24:  # Likely holding overnight (for hourly data)
            overnight_periods = total_trades * (avg_holding_bars / 24)
            financing_cost = overnight_periods * profile.overnight_rate * 100
        else:
            financing_cost = 0
        
        # Total costs
        total_cost = commission_cost + spread_cost + slippage_cost + financing_cost
        
        # Net return
        net_return = gross_return - total_cost
        
        # Adjusted Sharpe (rough approximation)
        # Reduce Sharpe proportionally to return reduction
        if gross_sharpe is not None and gross_return != 0:
            sharpe_adjustment = net_return / gross_return if gross_return > 0 else 0
            net_sharpe = gross_sharpe * max(0, sharpe_adjustment)
        else:
            net_sharpe = None
        
        # Cost ratio (what % of gross profit goes to costs)
        if gross_return > 0:
            cost_ratio = (total_cost / gross_return) * 100
        else:
            cost_ratio = float('inf') if total_cost > 0 else 0
        
        # Turnover (trades per bar)
        turnover = total_trades / bars_tested if bars_tested > 0 else 0
        
        # Viability assessment
        is_viable, reason = self._assess_viability(
            net_return, net_sharpe, cost_ratio, total_trades, turnover
        )
        
        return AdjustedResult(
            strategy_name=result.get('strategy_name', 'Unknown'),
            symbol=symbol,
            timeframe=result.get('timeframe', 'Unknown'),
            gross_return_pct=gross_return,
            gross_sharpe=gross_sharpe,
            commission_cost_pct=commission_cost,
            spread_cost_pct=spread_cost,
            slippage_cost_pct=slippage_cost,
            financing_cost_pct=financing_cost,
            total_cost_pct=total_cost,
            net_return_pct=net_return,
            net_sharpe=net_sharpe,
            cost_ratio=cost_ratio,
            total_trades=total_trades,
            avg_holding_period=avg_holding_bars,
            turnover=turnover,
            is_viable=is_viable,
            viability_reason=reason
        )
    
    def _assess_viability(
        self,
        net_return: float,
        net_sharpe: Optional[float],
        cost_ratio: float,
        total_trades: int,
        turnover: float
    ) -> tuple:
        """Assess if strategy is viable after costs"""
        
        issues = []
        
        if net_return < 0:
            issues.append("Negative net return")
        
        if net_sharpe is not None and net_sharpe < 0.5:
            issues.append(f"Low net Sharpe ({net_sharpe:.2f})")
        
        if cost_ratio > 50:
            issues.append(f"Costs consume {cost_ratio:.0f}% of profits")
        
        if total_trades < 30:
            issues.append(f"Too few trades ({total_trades}) for significance")
        
        if turnover > 0.1:
            issues.append(f"Very high turnover ({turnover:.3f} trades/bar)")
        
        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Passes all viability checks"
    
    def rank_variants(
        self,
        results: List[Dict],
        sort_by: str = 'net_return',
        ascending: bool = False
    ) -> List[AdjustedResult]:
        """
        Adjust and rank multiple variant results.
        
        Args:
            results: List of backtest result dictionaries
            sort_by: Metric to sort by ('net_return', 'net_sharpe', 'cost_ratio')
            ascending: Sort order
        
        Returns:
            List of AdjustedResult, sorted by specified metric
        """
        
        adjusted = [self.adjust_result(r) for r in results]
        
        # Sort
        if sort_by == 'net_return':
            adjusted.sort(key=lambda x: x.net_return_pct, reverse=not ascending)
        elif sort_by == 'net_sharpe':
            adjusted.sort(key=lambda x: x.net_sharpe or -999, reverse=not ascending)
        elif sort_by == 'cost_ratio':
            adjusted.sort(key=lambda x: x.cost_ratio, reverse=ascending)
        
        return adjusted
    
    def print_comparison(
        self,
        results: List[Dict],
        top_n: int = 10
    ):
        """Print comparison of gross vs net performance"""
        
        adjusted = self.rank_variants(results)
        
        print(f"\n{'='*80}")
        print(f"COST-ADJUSTED RANKING (Top {top_n})")
        print(f"{'='*80}")
        print(f"{'Strategy':<25} {'Gross':>10} {'Costs':>10} {'Net':>10} {'Viable':>8}")
        print(f"{'':<25} {'Return%':>10} {'%':>10} {'Return%':>10} {'':<8}")
        print(f"{'-'*80}")
        
        for i, adj in enumerate(adjusted[:top_n]):
            viable = "[OK]" if adj.is_viable else "[FAIL]"
            name = adj.strategy_name[:24]
            print(f"{name:<25} {adj.gross_return_pct:>+10.2f} {adj.total_cost_pct:>10.2f} {adj.net_return_pct:>+10.2f} {viable:>8}")
        
        # Summary stats
        viable_count = sum(1 for a in adjusted if a.is_viable)
        avg_cost = np.mean([a.total_cost_pct for a in adjusted])
        
        print(f"{'-'*80}")
        print(f"Viable strategies: {viable_count}/{len(adjusted)}")
        print(f"Average cost impact: {avg_cost:.2f}%")
        print(f"{'='*80}")
    
    def generate_report(
        self,
        result: AdjustedResult
    ) -> str:
        """Generate detailed cost analysis report"""
        
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"COST ANALYSIS REPORT")
        lines.append(f"{'='*60}")
        lines.append(f"Strategy: {result.strategy_name}")
        lines.append(f"Symbol: {result.symbol} | Timeframe: {result.timeframe}")
        lines.append(f"{'='*60}")
        
        lines.append(f"\n[STATS] PERFORMANCE COMPARISON:")
        lines.append(f"   Gross Return:  {result.gross_return_pct:+.2f}%")
        lines.append(f"   Net Return:    {result.net_return_pct:+.2f}%")
        lines.append(f"   Difference:    {result.net_return_pct - result.gross_return_pct:+.2f}%")
        
        if result.gross_sharpe:
            lines.append(f"\n   Gross Sharpe:  {result.gross_sharpe:.2f}")
            if result.net_sharpe:
                lines.append(f"   Net Sharpe:    {result.net_sharpe:.2f}")
        
        lines.append(f"\n[COST] COST BREAKDOWN:")
        lines.append(f"   Commission:    {result.commission_cost_pct:.3f}%")
        lines.append(f"   Spread:        {result.spread_cost_pct:.3f}%")
        lines.append(f"   Slippage:      {result.slippage_cost_pct:.3f}%")
        lines.append(f"   Financing:     {result.financing_cost_pct:.3f}%")
        lines.append(f"   ---------------------")
        lines.append(f"   TOTAL COSTS:   {result.total_cost_pct:.3f}%")
        
        if result.gross_return_pct > 0:
            lines.append(f"\n   Cost Ratio:    {result.cost_ratio:.1f}% of gross profit")
        
        lines.append(f"\n[UP] TRADE STATISTICS:")
        lines.append(f"   Total Trades:  {result.total_trades}")
        lines.append(f"   Avg Holding:   {result.avg_holding_period:.1f} bars")
        lines.append(f"   Turnover:      {result.turnover:.4f} trades/bar")
        
        lines.append(f"\n[TARGET] VIABILITY ASSESSMENT:")
        if result.is_viable:
            lines.append(f"   [OK] VIABLE - {result.viability_reason}")
        else:
            lines.append(f"   [FAIL] NOT VIABLE - {result.viability_reason}")
        
        lines.append(f"\n{'='*60}")
        
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def adjust_for_costs(result: Dict, asset_class: str = 'forex') -> AdjustedResult:
    """Quick cost adjustment for a single result"""
    
    scorer = CostAdjustedScorer(default_profile=asset_class)
    return scorer.adjust_result(result)


def rank_by_net_return(results: List[Dict]) -> List[AdjustedResult]:
    """Rank results by net return after costs"""
    
    scorer = CostAdjustedScorer()
    return scorer.rank_variants(results, sort_by='net_return')


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("COST-ADJUSTED SCORING MODULE")
    print("="*70)
    
    # Example with sample data
    sample_results = [
        {
            'strategy_name': 'MA_Crossover_Fast',
            'symbol': 'EUR-USD',
            'timeframe': '1hour',
            'total_return_pct': 15.5,
            'sharpe_ratio': 1.2,
            'total_trades': 450,
            'bars_tested': 10000
        },
        {
            'strategy_name': 'MA_Crossover_Slow',
            'symbol': 'EUR-USD',
            'timeframe': '1hour',
            'total_return_pct': 8.2,
            'sharpe_ratio': 1.5,
            'total_trades': 85,
            'bars_tested': 10000
        },
        {
            'strategy_name': 'RSI_Scalper',
            'symbol': 'EUR-USD',
            'timeframe': '1hour',
            'total_return_pct': 25.0,
            'sharpe_ratio': 0.9,
            'total_trades': 1200,
            'bars_tested': 10000
        },
    ]
    
    scorer = CostAdjustedScorer()
    
    print("\n[STATS] Analyzing sample strategies...")
    
    # Print comparison
    scorer.print_comparison(sample_results)
    
    # Detailed report for first one
    adjusted = scorer.adjust_result(sample_results[0])
    print(scorer.generate_report(adjusted))
    
    print("\n[OK] Cost-adjusted scoring module working!")
    print("="*70)