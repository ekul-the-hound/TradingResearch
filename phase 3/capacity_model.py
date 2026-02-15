# ==============================================================================
# capacity_model.py
# ==============================================================================
# Phase 3, Module 2 (Week 12): Strategy Capacity Estimation
#
# Estimates the maximum capital a strategy can manage before market impact
# erodes returns below a minimum threshold. A strategy with Sharpe 2.0 at
# $100K might only have Sharpe 0.5 at $10M.
#
# Methods:
#   1. Volume-based: max % of daily volume per trade
#   2. Impact-based: find capital where impact reduces Sharpe below threshold
#   3. Decay curve: fit impact-vs-AUM curve, find inflection
#
# Consumed by:
#   - optimization_pipeline.py (capacity-weighted portfolio)
#   - portfolio_engine.py (allocation constraints)
#   - kill_switch.py (position size limits)
#
# Usage:
#     from capacity_model import CapacityEstimator
#     cap = CapacityEstimator()
#     result = cap.estimate(strategy, daily_volume=5e6, volatility=0.02)
#     print(f"Max AUM: ${result.max_aum:,.0f}")
# ==============================================================================

import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from market_impact import MarketImpactModel, ImpactEstimate


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class CapacityResult:
    """Capacity estimation output."""
    max_aum: float                     # Max capital before unacceptable decay
    max_position_size: float           # Max single position
    sharpe_at_max: float               # Sharpe at max AUM
    impact_at_max_bps: float           # Impact at max AUM
    participation_at_max: float        # % of daily volume at max
    decay_curve: Dict[str, List]       # {aum: [], sharpe: [], impact: []}
    min_sharpe_threshold: float
    method: str

    def __str__(self):
        return (
            f"Capacity [{self.method}]: Max AUM = ${self.max_aum:,.0f}\n"
            f"  Sharpe at max = {self.sharpe_at_max:.3f}\n"
            f"  Impact at max = {self.impact_at_max_bps:.1f} bps\n"
            f"  Participation = {self.participation_at_max:.2%}"
        )


@dataclass
class RegimeCapacity:
    """Capacity estimates per market regime."""
    normal: CapacityResult
    low_vol: Optional[CapacityResult] = None
    high_vol: Optional[CapacityResult] = None
    crisis: Optional[CapacityResult] = None

    @property
    def conservative(self) -> float:
        """Use minimum across regimes."""
        vals = [self.normal.max_aum]
        if self.high_vol:
            vals.append(self.high_vol.max_aum)
        if self.crisis:
            vals.append(self.crisis.max_aum)
        return min(vals)


# ==============================================================================
# CAPACITY ESTIMATOR
# ==============================================================================

class CapacityEstimator:
    """Estimates max capital for a strategy."""

    def __init__(
        self,
        impact_model: Optional[MarketImpactModel] = None,
        max_participation: float = 0.05,     # Max 5% of daily volume
        min_sharpe_threshold: float = 0.5,   # Stop when Sharpe drops below
        n_aum_points: int = 50,              # Resolution of decay curve
    ):
        self.impact_model = impact_model or MarketImpactModel()
        self.max_participation = max_participation
        self.min_sharpe_threshold = min_sharpe_threshold
        self.n_aum_points = n_aum_points

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def estimate(
        self,
        strategy: Dict[str, Any],
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        spread: float = 0.0002,
        price: float = 1.0,
        method: str = "impact",
    ) -> CapacityResult:
        """
        Estimate capacity for a strategy.

        Args:
            strategy: Dict with sharpe_ratio, total_trades, avg_trade_pct, etc.
            daily_volume: Average daily volume (units).
            volatility: Daily volatility.
            spread: Bid-ask spread.
            price: Asset price.
            method: "volume" (simple), "impact" (full decay curve).
        """
        if method == "volume":
            return self._volume_based(strategy, daily_volume, price)
        elif method == "impact":
            return self._impact_based(
                strategy, daily_volume, volatility, spread, price,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # VOLUME-BASED (simple)
    # ------------------------------------------------------------------
    def _volume_based(
        self, strategy: Dict, daily_volume: float, price: float,
    ) -> CapacityResult:
        """Max AUM = max_participation × daily_volume × price."""
        trades_per_day = strategy.get("trades_per_day",
                                       strategy.get("total_trades", 100) / 252)
        trades_per_day = max(trades_per_day, 0.1)

        max_order = self.max_participation * daily_volume
        max_aum = max_order * price / max(trades_per_day, 0.01)

        return CapacityResult(
            max_aum=max_aum,
            max_position_size=max_order * price,
            sharpe_at_max=strategy.get("sharpe_ratio", 0),
            impact_at_max_bps=0,
            participation_at_max=self.max_participation,
            decay_curve={"aum": [max_aum], "sharpe": [strategy.get("sharpe_ratio", 0)]},
            min_sharpe_threshold=self.min_sharpe_threshold,
            method="volume",
        )

    # ------------------------------------------------------------------
    # IMPACT-BASED (full decay curve)
    # ------------------------------------------------------------------
    def _impact_based(
        self,
        strategy: Dict,
        daily_volume: float,
        volatility: float,
        spread: float,
        price: float,
    ) -> CapacityResult:
        """
        Sweep AUM from small to large, compute Sharpe after impact at each.
        Find the AUM where Sharpe crosses the threshold.
        """
        base_sharpe = strategy.get("sharpe_ratio", 1.0) or 1.0
        base_return_pct = strategy.get("total_return_pct", 10.0) or 10.0
        n_trades = strategy.get("total_trades", 100) or 100
        avg_trade_pct = strategy.get("avg_trade_pct", base_return_pct / max(n_trades, 1))

        # AUM range: $1K to $100M
        aum_range = np.geomspace(1_000, 100_000_000, self.n_aum_points)
        sharpe_curve = []
        impact_curve = []

        for aum in aum_range:
            # Position size per trade
            position_value = aum  # Assume full capital per trade (worst case)
            order_units = position_value / max(price, 0.01)

            # Impact per round-trip
            impact = self.impact_model.estimate_roundtrip(
                order_units, daily_volume, volatility, spread,
            )
            impact_pct = impact.total_cost_pct

            # Adjusted return per trade
            adj_trade_return = avg_trade_pct - impact_pct * 100
            # Scale Sharpe proportionally
            if abs(avg_trade_pct) > 1e-6:
                ratio = adj_trade_return / avg_trade_pct
            else:
                ratio = 1.0
            adj_sharpe = base_sharpe * max(ratio, -2.0)

            sharpe_curve.append(adj_sharpe)
            impact_curve.append(impact.total_impact_bps)

        sharpe_arr = np.array(sharpe_curve)
        impact_arr = np.array(impact_curve)

        # Find capacity: last AUM where Sharpe >= threshold
        above = sharpe_arr >= self.min_sharpe_threshold
        if np.any(above):
            max_idx = np.where(above)[0][-1]
        else:
            max_idx = 0

        max_aum = float(aum_range[max_idx])
        max_position = max_aum / max(price, 0.01)

        return CapacityResult(
            max_aum=max_aum,
            max_position_size=max_position * price,
            sharpe_at_max=float(sharpe_arr[max_idx]),
            impact_at_max_bps=float(impact_arr[max_idx]),
            participation_at_max=float(max_position / max(daily_volume, 1)),
            decay_curve={
                "aum": aum_range.tolist(),
                "sharpe": sharpe_arr.tolist(),
                "impact_bps": impact_arr.tolist(),
            },
            min_sharpe_threshold=self.min_sharpe_threshold,
            method="impact",
        )

    # ------------------------------------------------------------------
    # REGIME-ADAPTED CAPACITY
    # ------------------------------------------------------------------
    def estimate_by_regime(
        self,
        strategy: Dict,
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        spread: float = 0.0002,
        price: float = 1.0,
    ) -> RegimeCapacity:
        """
        Estimate capacity under different market regimes.
        Crisis: volume drops 70%, vol spikes 3x, spread widens 5x.
        """
        normal = self.estimate(strategy, daily_volume, volatility, spread, price)

        # Low vol: volume -20%, vol -50%
        low_vol = self.estimate(
            strategy, daily_volume * 0.8, volatility * 0.5, spread * 0.8, price,
        )
        # High vol: volume -30%, vol +100%
        high_vol = self.estimate(
            strategy, daily_volume * 0.7, volatility * 2.0, spread * 2.0, price,
        )
        # Crisis: volume -70%, vol +300%, spread +500%
        crisis = self.estimate(
            strategy, daily_volume * 0.3, volatility * 3.0, spread * 5.0, price,
        )

        return RegimeCapacity(
            normal=normal, low_vol=low_vol, high_vol=high_vol, crisis=crisis,
        )
