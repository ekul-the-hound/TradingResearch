# ==============================================================================
# liquidity_stress.py
# ==============================================================================
# Phase 3, Module 4 (Week 13): Liquidity Stress Testing
#
# Tests strategies under adverse liquidity conditions that are rare but
# catastrophic. Simulates flash crashes, low volume, partial fills,
# gap risk, and correlated drawdowns.
#
# Stress Scenarios:
#   1. Flash Crash: volume -90%, slippage 10x, instant gap
#   2. Low Liquidity: volume -70%, slippage 3x, wider spreads
#   3. Gap Risk: overnight/weekend gaps of 2-5σ
#   4. Partial Fill: only 30-60% of orders fill
#   5. Correlated Selloff: all assets drop simultaneously
#   6. Volatility Regime Shift: sudden vol spike
#
# Consumed by:
#   - optimization_pipeline.py (stress-adjusted fitness)
#   - kill_switch.py (scenario-based thresholds)
#   - capacity_model.py (stress capacity estimation)
#
# Usage:
#     from liquidity_stress import LiquidityStressTest
#     stress = LiquidityStressTest()
#     result = stress.run_all(trades, returns, account_size=100000)
# ==============================================================================

import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from market_impact import MarketImpactModel


# ==============================================================================
# SCENARIOS
# ==============================================================================

@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    volume_multiplier: float = 1.0     # 0.1 = 90% less volume
    slippage_multiplier: float = 1.0   # 10.0 = 10x slippage
    spread_multiplier: float = 1.0     # 5.0 = 5x wider spread
    gap_sigma: float = 0.0            # Gap size in std devs
    fill_rate: float = 1.0            # 0.3 = only 30% fills
    return_shock_pct: float = 0.0     # -5.0 = instant 5% drop
    vol_multiplier: float = 1.0       # 3.0 = 3x volatility
    correlation_override: Optional[float] = None  # Force correlation to this


DEFAULT_SCENARIOS = [
    StressScenario(
        "flash_crash", "2010-style flash crash: liquidity evaporates",
        volume_multiplier=0.1, slippage_multiplier=10.0,
        spread_multiplier=10.0, gap_sigma=5.0, fill_rate=0.3,
        return_shock_pct=-5.0, vol_multiplier=5.0,
    ),
    StressScenario(
        "low_liquidity", "Holiday/off-hours low volume",
        volume_multiplier=0.3, slippage_multiplier=3.0,
        spread_multiplier=3.0, fill_rate=0.7,
    ),
    StressScenario(
        "gap_risk", "Overnight gap (earnings, geopolitical)",
        gap_sigma=3.0, return_shock_pct=-3.0,
    ),
    StressScenario(
        "partial_fill", "Thin book, orders only partially filled",
        fill_rate=0.4, slippage_multiplier=2.0,
    ),
    StressScenario(
        "correlated_selloff", "All assets decline simultaneously",
        return_shock_pct=-8.0, vol_multiplier=3.0,
        correlation_override=0.9,
    ),
    StressScenario(
        "vol_regime_shift", "Sudden volatility spike",
        vol_multiplier=4.0, spread_multiplier=2.0,
        slippage_multiplier=2.0,
    ),
]


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class ScenarioResult:
    """Result of one stress scenario."""
    scenario: str
    original_pnl: float
    stressed_pnl: float
    pnl_impact_pct: float         # How much worse (negative = bad)
    max_drawdown_pct: float
    trades_affected: int
    trades_unfilled: int
    impact_cost_total: float
    survival: bool                # Strategy survives (doesn't breach limits)


@dataclass
class StressTestResult:
    """Complete stress test output."""
    scenarios: List[ScenarioResult]
    overall_survival_rate: float   # % of scenarios survived
    worst_scenario: str
    worst_pnl_impact_pct: float
    original_sharpe: float
    stressed_sharpe: float         # Worst-case Sharpe
    summary: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        lines = [
            f"\n{'='*60}",
            f"  LIQUIDITY STRESS TEST RESULTS",
            f"{'='*60}",
            f"  Scenarios tested:    {len(self.scenarios)}",
            f"  Survival rate:       {self.overall_survival_rate:.0%}",
            f"  Worst scenario:      {self.worst_scenario}",
            f"  Worst PnL impact:    {self.worst_pnl_impact_pct:.1f}%",
            f"  Original Sharpe:     {self.original_sharpe:.3f}",
            f"  Stressed Sharpe:     {self.stressed_sharpe:.3f}",
            f"\n  Per-scenario results:",
        ]
        for s in self.scenarios:
            icon = "[OK]" if s.survival else "[FAIL]"
            lines.append(f"    {icon} {s.scenario:20s} | PnL impact: {s.pnl_impact_pct:+.1f}% "
                         f"| DD: {s.max_drawdown_pct:.1f}%")
        return "\n".join(lines)


# ==============================================================================
# STRESS TEST ENGINE
# ==============================================================================

class LiquidityStressTest:
    """
    Applies stress scenarios to strategy trades/returns.
    """

    def __init__(
        self,
        scenarios: Optional[List[StressScenario]] = None,
        impact_model: Optional[MarketImpactModel] = None,
        max_drawdown_limit: float = 30.0,  # Survival limit
        account_size: float = 100_000,
    ):
        self.scenarios = scenarios or DEFAULT_SCENARIOS
        self.impact_model = impact_model or MarketImpactModel()
        self.max_drawdown_limit = max_drawdown_limit
        self.account_size = account_size

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def run_all(
        self,
        trades: Optional[List[Dict]] = None,
        returns: Optional[np.ndarray] = None,
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        spread: float = 0.0002,
    ) -> StressTestResult:
        """
        Run all stress scenarios.

        Args:
            trades: List of trade dicts with size, pnl, entry_price.
            returns: Daily return series (alternative to trades).
            daily_volume: Normal daily volume.
            volatility: Normal daily volatility.
            spread: Normal bid-ask spread.
        """
        results = []
        for scenario in self.scenarios:
            r = self._run_scenario(
                scenario, trades, returns,
                daily_volume, volatility, spread,
            )
            results.append(r)

        # Summary
        survival_rate = sum(1 for r in results if r.survival) / max(len(results), 1)
        worst = min(results, key=lambda r: r.pnl_impact_pct)

        # Original vs stressed Sharpe
        if returns is not None and len(returns) > 10:
            orig_sharpe = float(np.mean(returns) / max(np.std(returns, ddof=1), 1e-10) * np.sqrt(252))
            worst_rets = self._apply_return_stress(returns, worst.scenario)
            stress_sharpe = float(np.mean(worst_rets) / max(np.std(worst_rets, ddof=1), 1e-10) * np.sqrt(252))
        else:
            orig_sharpe = 0.0
            stress_sharpe = 0.0

        return StressTestResult(
            scenarios=results,
            overall_survival_rate=survival_rate,
            worst_scenario=worst.scenario,
            worst_pnl_impact_pct=worst.pnl_impact_pct,
            original_sharpe=orig_sharpe,
            stressed_sharpe=stress_sharpe,
        )

    # ------------------------------------------------------------------
    # SINGLE SCENARIO
    # ------------------------------------------------------------------
    def _run_scenario(
        self,
        scenario: StressScenario,
        trades: Optional[List[Dict]],
        returns: Optional[np.ndarray],
        daily_volume: float,
        volatility: float,
        spread: float,
    ) -> ScenarioResult:
        """Apply one stress scenario."""
        # Stressed parameters
        s_vol = daily_volume * scenario.volume_multiplier
        s_volatility = volatility * scenario.vol_multiplier
        s_spread = spread * scenario.spread_multiplier

        original_pnl = 0.0
        stressed_pnl = 0.0
        trades_affected = 0
        trades_unfilled = 0
        impact_total = 0.0
        max_dd = 0.0

        if trades and len(trades) > 0:
            original_pnl = sum(t.get("pnl", 0) for t in trades)

            rng = np.random.RandomState(42)
            equity = self.account_size
            peak = equity
            for t in trades:
                size = abs(t.get("size", 1))
                pnl = t.get("pnl", 0)

                # Partial fill
                if rng.random() > scenario.fill_rate:
                    trades_unfilled += 1
                    continue

                fill_frac = scenario.fill_rate + rng.random() * (1 - scenario.fill_rate)
                adj_size = size * fill_frac

                # Impact cost
                impact = self.impact_model.estimate_roundtrip(
                    adj_size, s_vol, s_volatility, s_spread,
                )
                cost = impact.total_cost_pct * adj_size * t.get("entry_price", 1.0)
                cost *= scenario.slippage_multiplier
                impact_total += cost

                # Gap shock
                if scenario.gap_sigma > 0:
                    gap = rng.normal(0, scenario.gap_sigma * s_volatility)
                    pnl += gap * adj_size * t.get("entry_price", 1.0)

                # Return shock
                if scenario.return_shock_pct != 0:
                    shock = scenario.return_shock_pct / 100 * adj_size * t.get("entry_price", 1.0)
                    if rng.random() < 0.3:  # 30% of trades hit by shock
                        pnl += shock

                adj_pnl = pnl * fill_frac - cost
                stressed_pnl += adj_pnl
                trades_affected += 1

                equity += adj_pnl
                peak = max(peak, equity)
                dd = (peak - equity) / max(peak, 1) * 100
                max_dd = max(max_dd, dd)

        elif returns is not None and len(returns) > 0:
            original_pnl = float(np.sum(returns)) * self.account_size
            stressed_rets = self._apply_return_stress_scenario(returns, scenario)
            stressed_pnl = float(np.sum(stressed_rets)) * self.account_size
            eq = np.cumprod(1 + stressed_rets) * self.account_size
            peak_eq = np.maximum.accumulate(eq)
            dd_series = (peak_eq - eq) / np.maximum(peak_eq, 1) * 100
            max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 0
            trades_affected = len(returns)

        # PnL impact
        if abs(original_pnl) > 1e-6:
            impact_pct = (stressed_pnl - original_pnl) / abs(original_pnl) * 100
        else:
            impact_pct = 0.0

        survival = max_dd < self.max_drawdown_limit

        return ScenarioResult(
            scenario=scenario.name,
            original_pnl=original_pnl,
            stressed_pnl=stressed_pnl,
            pnl_impact_pct=impact_pct,
            max_drawdown_pct=max_dd,
            trades_affected=trades_affected,
            trades_unfilled=trades_unfilled,
            impact_cost_total=impact_total,
            survival=survival,
        )

    # ------------------------------------------------------------------
    def _apply_return_stress(self, returns: np.ndarray, scenario_name: str) -> np.ndarray:
        for s in self.scenarios:
            if s.name == scenario_name:
                return self._apply_return_stress_scenario(returns, s)
        return returns

    def _apply_return_stress_scenario(
        self, returns: np.ndarray, scenario: StressScenario,
    ) -> np.ndarray:
        """Apply stress to a return series."""
        stressed = returns.copy()
        rng = np.random.RandomState(42)

        # Vol scaling
        if scenario.vol_multiplier != 1.0:
            mean = np.mean(stressed)
            stressed = mean + (stressed - mean) * scenario.vol_multiplier

        # Return shock (applied to random 10% of days)
        if scenario.return_shock_pct != 0:
            shock_days = rng.random(len(stressed)) < 0.1
            stressed[shock_days] += scenario.return_shock_pct / 100

        # Gap injection
        if scenario.gap_sigma > 0:
            gap_days = rng.random(len(stressed)) < 0.05
            gaps = rng.normal(0, scenario.gap_sigma * np.std(returns), len(stressed))
            stressed[gap_days] += gaps[gap_days]

        # Slippage drag
        if scenario.slippage_multiplier > 1.0:
            drag = abs(np.mean(returns)) * (scenario.slippage_multiplier - 1) * 0.1
            stressed -= drag

        return stressed

    # ------------------------------------------------------------------
    # CUSTOM SCENARIO
    # ------------------------------------------------------------------
    def add_scenario(self, scenario: StressScenario):
        """Add a custom stress scenario."""
        self.scenarios.append(scenario)
