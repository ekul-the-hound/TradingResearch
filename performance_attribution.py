# ==============================================================================
# performance_attribution.py
# ==============================================================================
# Phase 6, Module 4: Performance Attribution
#
# Answers: WHERE did the returns come from? Decomposes strategy PnL into
# explainable components so the system can learn what actually works.
#
# Attribution layers:
#   1. Alpha vs Beta: how much is market exposure vs true skill?
#   2. Factor decomposition: momentum, value, volatility, carry
#   3. Regime contribution: which regimes generated profit/loss?
#   4. Timing attribution: entry timing vs exit timing vs hold duration
#   5. Cost drag: commission, slippage, spread, market impact
#   6. Luck vs skill: bootstrap to separate genuine alpha from noise
#
# Consumed by:
#   - lineage_analytics.py (understand WHY mutations succeed/fail)
#   - learning_loop.py (target improvements at weakest component)
#   - dashboard (attribution waterfall charts)
#
# Usage:
#     from performance_attribution import PerformanceAttributor
#     attr = PerformanceAttributor()
#     result = attr.attribute(strategy_returns, benchmark_returns,
#                             trades=trades, regimes=regime_labels)
# ==============================================================================

import numpy as np
from scipy import stats as sp_stats
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class AttributionResult:
    """Complete performance attribution."""
    # Alpha vs Beta
    total_return: float
    alpha: float                    # Return not explained by benchmark
    beta: float                     # Market exposure
    beta_return: float              # Return from market exposure
    alpha_return: float             # Return from skill
    r_squared: float                # How much benchmark explains

    # Factor decomposition
    factor_exposures: Dict[str, float]    # Factor -> beta
    factor_returns: Dict[str, float]      # Factor -> attributed return
    residual_return: float                # Unexplained by factors

    # Regime attribution
    regime_returns: Dict[str, float]      # Regime -> total return in that regime
    regime_trade_counts: Dict[str, int]   # Regime -> number of trades
    best_regime: str
    worst_regime: str

    # Timing
    entry_timing_value: float       # Value from entry timing vs random
    exit_timing_value: float        # Value from exit timing vs random
    hold_duration_value: float      # Value from hold duration selection

    # Cost drag
    gross_return: float
    net_return: float
    total_costs: float
    cost_breakdown: Dict[str, float]  # commission, spread, slippage, impact

    # Luck vs Skill
    skill_pvalue: float             # p-value: is alpha statistically significant?
    bootstrap_alpha_ci: Tuple[float, float]  # 95% CI for alpha

    def __str__(self):
        lines = [
            f"\n{'='*60}",
            f"  PERFORMANCE ATTRIBUTION",
            f"{'='*60}",
            f"  Total Return:  {self.total_return:.2%}",
            f"  Alpha:         {self.alpha_return:.2%} (β={self.beta:.3f}, R²={self.r_squared:.3f})",
            f"  Beta Return:   {self.beta_return:.2%}",
            f"  Gross:         {self.gross_return:.2%}",
            f"  Net:           {self.net_return:.2%}",
            f"  Cost Drag:     {self.total_costs:.2%}",
            f"  Skill p-value: {self.skill_pvalue:.4f}",
            f"\n  Regime Attribution:",
        ]
        for regime, ret in sorted(self.regime_returns.items(), key=lambda x: -x[1]):
            n = self.regime_trade_counts.get(regime, 0)
            lines.append(f"    {regime:15s} -> {ret:.2%} ({n} trades)")
        if self.factor_exposures:
            lines.append(f"\n  Factor Exposures:")
            for f, b in sorted(self.factor_exposures.items(), key=lambda x: -abs(x[1])):
                lines.append(f"    {f:15s} -> β={b:.3f}, return={self.factor_returns.get(f, 0):.2%}")
        return "\n".join(lines)


# ==============================================================================
# PERFORMANCE ATTRIBUTOR
# ==============================================================================

class PerformanceAttributor:
    """Decomposes strategy returns into explainable components."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def attribute(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        factor_returns: Optional[Dict[str, np.ndarray]] = None,
        trades: Optional[List[Dict]] = None,
        regimes: Optional[np.ndarray] = None,
        regime_labels: Optional[Dict[int, str]] = None,
        costs: Optional[Dict[str, float]] = None,
        n_bootstrap: int = 1000,
    ) -> AttributionResult:
        """
        Full attribution analysis.

        Args:
            strategy_returns: Daily returns (decimal).
            benchmark_returns: Market/benchmark daily returns.
            factor_returns: Dict of factor name -> daily returns.
            trades: List of trade dicts for timing analysis.
            regimes: Array of regime labels per day (same length as returns).
            regime_labels: Map int -> name, e.g., {0: "bull", 1: "bear"}.
            costs: Dict with commission, spread, slippage, impact totals.
            n_bootstrap: Number of bootstrap samples for skill test.
        """
        r = np.asarray(strategy_returns, dtype=np.float64)
        n = len(r)
        total_return = float(np.prod(1 + r) - 1)

        # Alpha vs Beta
        alpha, beta, beta_ret, alpha_ret, r2 = self._alpha_beta(r, benchmark_returns)

        # Factor decomposition
        f_exp, f_ret, residual = self._factor_decomposition(r, factor_returns)

        # Regime attribution
        reg_rets, reg_counts, best_reg, worst_reg = self._regime_attribution(
            r, regimes, regime_labels,
        )

        # Timing attribution
        entry_val, exit_val, hold_val = self._timing_attribution(trades, r)

        # Cost drag
        gross, net, total_cost, cost_bk = self._cost_attribution(r, costs)

        # Luck vs Skill
        skill_p, alpha_ci = self._skill_test(r, benchmark_returns, n_bootstrap)

        return AttributionResult(
            total_return=total_return,
            alpha=alpha, beta=beta,
            beta_return=beta_ret, alpha_return=alpha_ret,
            r_squared=r2,
            factor_exposures=f_exp, factor_returns=f_ret,
            residual_return=residual,
            regime_returns=reg_rets, regime_trade_counts=reg_counts,
            best_regime=best_reg, worst_regime=worst_reg,
            entry_timing_value=entry_val, exit_timing_value=exit_val,
            hold_duration_value=hold_val,
            gross_return=gross, net_return=net,
            total_costs=total_cost, cost_breakdown=cost_bk,
            skill_pvalue=skill_p, bootstrap_alpha_ci=alpha_ci,
        )

    # ------------------------------------------------------------------
    # ALPHA / BETA
    # ------------------------------------------------------------------
    def _alpha_beta(
        self, r: np.ndarray, bench: Optional[np.ndarray],
    ) -> Tuple[float, float, float, float, float]:
        if bench is None or len(bench) < len(r):
            total = float(np.prod(1 + r) - 1)
            return 0.0, 0.0, 0.0, total, 0.0

        b = bench[:len(r)]
        # OLS: r = alpha + beta * bench + epsilon
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(b, r)
        beta = float(slope)
        alpha_daily = float(intercept)
        r2 = float(r_val ** 2)

        beta_return = float(np.prod(1 + b * beta) - 1)
        total = float(np.prod(1 + r) - 1)
        alpha_return = total - beta_return

        return alpha_daily, beta, beta_return, alpha_return, r2

    # ------------------------------------------------------------------
    # FACTOR DECOMPOSITION
    # ------------------------------------------------------------------
    def _factor_decomposition(
        self, r: np.ndarray, factors: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        if not factors:
            return {}, {}, float(np.prod(1 + r) - 1)

        n = len(r)
        # Build factor matrix
        fnames = list(factors.keys())
        F = np.column_stack([factors[f][:n] for f in fnames])

        # Multi-factor regression
        F_with_const = np.column_stack([np.ones(n), F])
        try:
            betas, residuals, _, _ = np.linalg.lstsq(F_with_const, r, rcond=None)
        except np.linalg.LinAlgError:
            return {}, {}, float(np.prod(1 + r) - 1)

        exposures = {}
        attributed = {}
        for i, fname in enumerate(fnames):
            exposures[fname] = float(betas[i + 1])
            factor_contrib = betas[i + 1] * factors[fname][:n]
            attributed[fname] = float(np.sum(factor_contrib))

        residual_returns = r - F_with_const @ betas
        residual = float(np.sum(residual_returns))

        return exposures, attributed, residual

    # ------------------------------------------------------------------
    # REGIME ATTRIBUTION
    # ------------------------------------------------------------------
    def _regime_attribution(
        self,
        r: np.ndarray,
        regimes: Optional[np.ndarray],
        labels: Optional[Dict[int, str]],
    ) -> Tuple[Dict[str, float], Dict[str, int], str, str]:
        if regimes is None or len(regimes) < len(r):
            return {}, {}, "unknown", "unknown"

        labels = labels or {}
        reg = regimes[:len(r)]
        unique = np.unique(reg)

        ret_by_regime = {}
        count_by_regime = {}
        for u in unique:
            name = labels.get(int(u), str(u))
            mask = reg == u
            regime_r = r[mask]
            ret_by_regime[name] = float(np.sum(regime_r))
            count_by_regime[name] = int(np.sum(mask))

        best = max(ret_by_regime, key=ret_by_regime.get) if ret_by_regime else "unknown"
        worst = min(ret_by_regime, key=ret_by_regime.get) if ret_by_regime else "unknown"

        return ret_by_regime, count_by_regime, best, worst

    # ------------------------------------------------------------------
    # TIMING ATTRIBUTION
    # ------------------------------------------------------------------
    def _timing_attribution(
        self, trades: Optional[List[Dict]], r: np.ndarray,
    ) -> Tuple[float, float, float]:
        if not trades or len(trades) < 5:
            return 0.0, 0.0, 0.0

        # Entry timing: compare actual entry return vs average
        entry_returns = []
        exit_returns = []
        hold_durations = []

        for t in trades:
            pnl = t.get("pnl", t.get("profit", 0))
            size = abs(t.get("size", 1))
            price = t.get("entry_price", 1)
            trade_return = pnl / max(size * price, 1e-6)
            entry_returns.append(trade_return)

            duration = t.get("duration_bars", t.get("hold_bars", 1))
            hold_durations.append(duration)

        avg_return = float(np.mean(r))
        avg_trade = float(np.mean(entry_returns))

        # Entry timing = excess of actual entry timing vs random
        entry_value = avg_trade - avg_return
        # Exit timing = correlation of exit with subsequent move
        exit_value = 0.0  # Would need price data after exit

        # Hold duration: do longer holds perform better?
        if len(hold_durations) > 5:
            corr, _ = sp_stats.pearsonr(
                hold_durations[:len(entry_returns)],
                entry_returns[:len(hold_durations)],
            )
            hold_value = float(corr) if np.isfinite(corr) else 0.0
        else:
            hold_value = 0.0

        return entry_value, exit_value, hold_value

    # ------------------------------------------------------------------
    # COST ATTRIBUTION
    # ------------------------------------------------------------------
    def _cost_attribution(
        self, r: np.ndarray, costs: Optional[Dict[str, float]],
    ) -> Tuple[float, float, float, Dict[str, float]]:
        total_return = float(np.prod(1 + r) - 1)

        if costs is None:
            return total_return, total_return, 0.0, {}

        total_cost = sum(costs.values())
        gross = total_return + total_cost
        return gross, total_return, total_cost, dict(costs)

    # ------------------------------------------------------------------
    # SKILL TEST (bootstrap)
    # ------------------------------------------------------------------
    def _skill_test(
        self,
        r: np.ndarray,
        bench: Optional[np.ndarray],
        n_bootstrap: int,
    ) -> Tuple[float, Tuple[float, float]]:
        """Bootstrap test: is alpha statistically significant?"""
        if bench is None or len(bench) < len(r) or n_bootstrap < 10:
            return 1.0, (0.0, 0.0)

        b = bench[:len(r)]
        n = len(r)

        # Observed alpha
        slope, intercept, _, _, _ = sp_stats.linregress(b, r)
        obs_alpha = intercept

        # Bootstrap
        rng = np.random.RandomState(42)
        boot_alphas = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            slope_b, int_b, _, _, _ = sp_stats.linregress(b[idx], r[idx])
            boot_alphas.append(int_b)

        boot_alphas = np.array(boot_alphas)
        # p-value: fraction of bootstrap alphas <= 0
        p_value = float(np.mean(boot_alphas <= 0))
        ci_lo = float(np.percentile(boot_alphas, 2.5))
        ci_hi = float(np.percentile(boot_alphas, 97.5))

        return p_value, (ci_lo, ci_hi)
