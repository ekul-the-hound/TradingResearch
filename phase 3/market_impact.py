# ==============================================================================
# market_impact.py
# ==============================================================================
# Phase 3, Module 1 (Week 11): Market Impact Modeling
#
# Models the price impact of executing trades, critical for realistic
# backtesting and capacity estimation. A strategy that looks great on
# paper may be unprofitable once you account for moving the market.
#
# Models:
#   1. Square-Root Law (empirical, most widely used)
#   2. Almgren-Chriss (optimal execution with temporary + permanent impact)
#   3. Linear Impact (simple, for small orders)
#   4. Kyle's Lambda (single parameter impact coefficient)
#
# References:
#   - Almgren & Chriss (2000): Optimal execution of portfolio transactions
#   - Bouchaud et al. (2009): How markets slowly digest changes in supply
#   - Kyle (1985): Continuous auctions and insider trading
#
# Consumed by:
#   - capacity_model.py (impact-aware capacity estimation)
#   - liquidity_stress.py (impact under stress scenarios)
#   - optimization_pipeline.py (impact-adjusted fitness)
#
# Usage:
#     from market_impact import MarketImpactModel
#     model = MarketImpactModel()
#     impact = model.estimate(order_size=50000, daily_volume=5_000_000,
#                             volatility=0.02, spread=0.0002)
# ==============================================================================

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class ImpactEstimate:
    """Market impact estimate for a single order."""
    total_impact_bps: float        # Total cost in basis points
    temporary_impact_bps: float    # Reverts after execution
    permanent_impact_bps: float    # Persists (information leakage)
    spread_cost_bps: float         # Half-spread crossing
    total_cost_pct: float          # As percentage of order value
    participation_rate: float      # Order size / daily volume
    model: str

    def __str__(self):
        return (f"Impact [{self.model}]: {self.total_impact_bps:.1f} bps "
                f"(temp={self.temporary_impact_bps:.1f}, perm={self.permanent_impact_bps:.1f}, "
                f"spread={self.spread_cost_bps:.1f}) | "
                f"participation={self.participation_rate:.2%}")


@dataclass
class ExecutionPlan:
    """Optimal execution schedule from Almgren-Chriss."""
    n_slices: int
    slice_sizes: np.ndarray       # Shares per time slice
    slice_times: np.ndarray       # Time points
    expected_cost: float          # Expected total cost
    cost_variance: float          # Variance of execution cost
    urgency: float                # Trade-off parameter (risk aversion)


# ==============================================================================
# MARKET IMPACT MODEL
# ==============================================================================

class MarketImpactModel:
    """
    Estimates market impact using multiple models.
    """

    def __init__(
        self,
        default_model: str = "sqrt",
        impact_coef: float = 0.1,     # Calibration coefficient
        permanent_frac: float = 0.3,  # Fraction of impact that's permanent
    ):
        self.default_model = default_model
        self.impact_coef = impact_coef
        self.permanent_frac = permanent_frac

    # ------------------------------------------------------------------
    # MAIN API
    # ------------------------------------------------------------------
    def estimate(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float,
        spread: float = 0.0002,
        model: Optional[str] = None,
        price: float = 1.0,
    ) -> ImpactEstimate:
        """
        Estimate market impact for a single order.

        Args:
            order_size: Order size in units (shares, lots, contracts).
            daily_volume: Average daily volume in same units.
            volatility: Daily volatility (decimal, e.g., 0.02 = 2%).
            spread: Bid-ask spread (decimal, e.g., 0.0002 = 2 pips).
            model: "sqrt", "almgren", "linear", "kyle". Default from init.
            price: Asset price (for converting to currency impact).
        """
        model = model or self.default_model
        participation = order_size / max(daily_volume, 1)
        spread_bps = spread * 10000

        if model == "sqrt":
            impact_bps = self._sqrt_impact(participation, volatility)
        elif model == "almgren":
            impact_bps = self._almgren_impact(participation, volatility)
        elif model == "linear":
            impact_bps = self._linear_impact(participation, volatility)
        elif model == "kyle":
            impact_bps = self._kyle_impact(participation, volatility)
        else:
            raise ValueError(f"Unknown model: {model}")

        temp = impact_bps * (1 - self.permanent_frac)
        perm = impact_bps * self.permanent_frac
        total = impact_bps + spread_bps / 2  # Half spread for one-way

        return ImpactEstimate(
            total_impact_bps=total,
            temporary_impact_bps=temp,
            permanent_impact_bps=perm,
            spread_cost_bps=spread_bps / 2,
            total_cost_pct=total / 10000,
            participation_rate=participation,
            model=model,
        )

    def estimate_roundtrip(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float,
        spread: float = 0.0002,
        **kwargs,
    ) -> ImpactEstimate:
        """
        Estimate round-trip (entry + exit) impact.
        Entry and exit both incur temporary impact; permanent only once.
        """
        one_way = self.estimate(order_size, daily_volume, volatility, spread, **kwargs)
        # Round-trip: 2x temporary + 1x permanent + 2x half-spread
        total_bps = (2 * one_way.temporary_impact_bps +
                     one_way.permanent_impact_bps +
                     2 * one_way.spread_cost_bps)
        return ImpactEstimate(
            total_impact_bps=total_bps,
            temporary_impact_bps=2 * one_way.temporary_impact_bps,
            permanent_impact_bps=one_way.permanent_impact_bps,
            spread_cost_bps=2 * one_way.spread_cost_bps,
            total_cost_pct=total_bps / 10000,
            participation_rate=one_way.participation_rate,
            model=one_way.model,
        )

    # ------------------------------------------------------------------
    # IMPACT MODELS
    # ------------------------------------------------------------------
    def _sqrt_impact(self, participation: float, volatility: float) -> float:
        """
        Square-root law: Impact ∝ σ × √(Q/V)
        Most empirically validated model (Bouchaud et al. 2009).
        """
        return self.impact_coef * volatility * 10000 * np.sqrt(abs(participation))

    def _almgren_impact(self, participation: float, volatility: float) -> float:
        """
        Almgren-Chriss: temporary ∝ σ × (Q/V)^0.6, permanent ∝ σ × (Q/V)
        """
        temp = self.impact_coef * volatility * 10000 * abs(participation) ** 0.6
        perm = 0.1 * volatility * 10000 * abs(participation)
        return temp + perm

    def _linear_impact(self, participation: float, volatility: float) -> float:
        """
        Linear impact: Impact ∝ σ × (Q/V)
        Simple model, valid for small orders (<1% of volume).
        """
        return self.impact_coef * volatility * 10000 * abs(participation)

    def _kyle_impact(self, participation: float, volatility: float) -> float:
        """
        Kyle's lambda: Impact = λ × order_flow
        λ calibrated from volatility and volume.
        """
        kyle_lambda = volatility * 10000 * self.impact_coef
        return kyle_lambda * abs(participation) ** 0.5

    # ------------------------------------------------------------------
    # OPTIMAL EXECUTION (Almgren-Chriss)
    # ------------------------------------------------------------------
    def optimal_execution(
        self,
        total_shares: float,
        n_slices: int = 10,
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        urgency: float = 1e-6,
        time_horizon_days: float = 1.0,
    ) -> ExecutionPlan:
        """
        Compute optimal TWAP/VWAP-like execution schedule.

        Higher urgency → front-load (minimize risk).
        Lower urgency → spread evenly (minimize impact).
        """
        dt = time_horizon_days / n_slices
        times = np.linspace(0, time_horizon_days, n_slices + 1)

        # Almgren-Chriss decay rate
        kappa = np.sqrt(urgency / max(self.impact_coef * volatility, 1e-10))

        # Optimal trajectory: x(t) = X * sinh(κ(T-t)) / sinh(κT)
        T = time_horizon_days
        trajectory = np.zeros(n_slices + 1)
        sinh_kT = np.sinh(kappa * T) if kappa * T < 50 else 1e20
        for i, t in enumerate(times):
            if sinh_kT > 0:
                trajectory[i] = total_shares * np.sinh(kappa * (T - t)) / sinh_kT
            else:
                trajectory[i] = total_shares * (1 - t / T)  # Linear fallback

        # Slice sizes (how much to trade in each period)
        slices = -np.diff(trajectory)  # Positive = sell
        slices = np.maximum(slices, 0)
        if slices.sum() > 0:
            slices = slices * total_shares / slices.sum()

        # Expected cost
        participation = total_shares / max(daily_volume, 1)
        expected_cost = self.estimate(
            total_shares, daily_volume, volatility,
        ).total_cost_pct * total_shares

        return ExecutionPlan(
            n_slices=n_slices,
            slice_sizes=slices,
            slice_times=times[:-1],
            expected_cost=expected_cost,
            cost_variance=volatility ** 2 * total_shares ** 2 * dt,
            urgency=urgency,
        )

    # ------------------------------------------------------------------
    # BATCH: impact for all trades in a backtest
    # ------------------------------------------------------------------
    def adjust_backtest(
        self,
        trades: List[Dict],
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        spread: float = 0.0002,
    ) -> Tuple[List[Dict], Dict]:
        """
        Apply market impact to each trade in a backtest result.
        Returns adjusted trades and summary stats.
        """
        adjusted = []
        total_impact = 0.0
        for t in trades:
            size = abs(t.get("size", t.get("order_size", 1)))
            impact = self.estimate_roundtrip(size, daily_volume, volatility, spread)
            adj = dict(t)
            pnl = adj.get("pnl", adj.get("profit", 0.0))
            cost = impact.total_cost_pct * size * t.get("entry_price", 1.0)
            adj["impact_cost"] = cost
            adj["pnl_after_impact"] = pnl - cost
            adj["impact_bps"] = impact.total_impact_bps
            adjusted.append(adj)
            total_impact += cost

        n = len(trades)
        summary = {
            "total_trades": n,
            "total_impact_cost": total_impact,
            "avg_impact_bps": np.mean([a["impact_bps"] for a in adjusted]) if n > 0 else 0,
            "max_impact_bps": max([a["impact_bps"] for a in adjusted]) if n > 0 else 0,
        }
        return adjusted, summary
