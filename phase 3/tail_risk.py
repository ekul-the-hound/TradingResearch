# ==============================================================================
# tail_risk.py
# ==============================================================================
# Phase 3, Module 5 (Week 14): Tail Risk Analysis
#
# Analyzes extreme loss behavior that normal (Gaussian) models miss.
# Uses Extreme Value Theory, CVaR, and copula-based tail dependence
# to quantify how bad things can really get.
#
# Methods:
#   1. VaR / CVaR (Historical, Parametric, Cornish-Fisher)
#   2. Extreme Value Theory (Block Maxima, Peaks Over Threshold)
#   3. Tail dependence (lower/upper tail coefficients)
#   4. Copula analysis (Clayton/Gumbel for tail dependence structure)
#   5. Drawdown-at-Risk (DaR): probability of extreme drawdowns
#
# References:
#   - McNeil, Frey & Embrechts: Quantitative Risk Management
#   - Embrechts, Klüppelberg & Mikosch: Modelling Extremal Events
#
# Consumed by:
#   - kill_switch.py (tail-risk-aware thresholds)
#   - portfolio_engine.py (CVaR-optimized allocation)
#   - optimization_pipeline.py (tail-risk objective)
#
# Usage:
#     from tail_risk import TailRiskAnalyzer
#     analyzer = TailRiskAnalyzer()
#     result = analyzer.analyze(returns)
#     print(f"CVaR 95%: {result.cvar_95:.2%}")
# ==============================================================================

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class TailRiskResult:
    """Complete tail risk analysis output."""
    # VaR / CVaR
    var_95: float              # 5th percentile loss
    var_99: float              # 1st percentile loss
    cvar_95: float             # Expected loss beyond VaR 95
    cvar_99: float             # Expected loss beyond VaR 99
    var_95_cf: float           # Cornish-Fisher adjusted VaR
    cvar_95_cf: float          # CF-adjusted CVaR

    # Distribution stats
    mean: float
    std: float
    skewness: float
    kurtosis: float           # Excess kurtosis (normal = 0)
    jarque_bera_p: float      # p-value for normality test

    # EVT (Peaks Over Threshold)
    evt_shape: float           # ξ (shape parameter of GPD)
    evt_scale: float           # σ (scale parameter)
    evt_threshold: float       # u (threshold)
    evt_var_99: float          # VaR from EVT
    evt_cvar_99: float         # CVaR from EVT

    # Tail dependence
    lower_tail_coef: float     # λ_L: probability of joint extreme losses
    upper_tail_coef: float     # λ_U: probability of joint extreme gains

    # Drawdown at Risk
    max_drawdown: float
    dar_95: float              # 95th percentile drawdown
    expected_shortfall_dd: float  # Expected DD beyond DaR

    # Summary
    tail_risk_score: float     # 0-1 composite (higher = riskier tails)

    def __str__(self):
        lines = [
            f"\n{'='*60}",
            f"  TAIL RISK ANALYSIS",
            f"{'='*60}",
            f"  VaR 95%:       {self.var_95:.4f} ({self.var_95*100:.2f}%)",
            f"  CVaR 95%:      {self.cvar_95:.4f} ({self.cvar_95*100:.2f}%)",
            f"  VaR 99%:       {self.var_99:.4f}",
            f"  CVaR 99%:      {self.cvar_99:.4f}",
            f"  CF VaR 95%:    {self.var_95_cf:.4f}",
            f"  Skewness:      {self.skewness:.3f}",
            f"  Ex. Kurtosis:  {self.kurtosis:.3f}",
            f"  Normal? (JB):  p={self.jarque_bera_p:.4f}",
            f"  EVT ξ:         {self.evt_shape:.4f}",
            f"  EVT VaR 99%:   {self.evt_var_99:.4f}",
            f"  Lower tail λ:  {self.lower_tail_coef:.4f}",
            f"  Max DD:        {self.max_drawdown:.2%}",
            f"  DaR 95%:       {self.dar_95:.2%}",
            f"  Tail Score:    {self.tail_risk_score:.3f}",
        ]
        return "\n".join(lines)


# ==============================================================================
# TAIL RISK ANALYZER
# ==============================================================================

class TailRiskAnalyzer:
    """Comprehensive tail risk analysis."""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def analyze(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> TailRiskResult:
        """
        Full tail risk analysis on a return series.

        Args:
            returns: Daily returns (decimal, e.g., 0.01 = 1%).
            benchmark_returns: Optional, for tail dependence.
        """
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]

        if len(r) < 20:
            return self._empty_result()

        # Basic stats
        mean = float(np.mean(r))
        std = float(np.std(r, ddof=1))
        skew = float(sp_stats.skew(r))
        kurt = float(sp_stats.kurtosis(r))  # Excess kurtosis
        jb_stat, jb_p = sp_stats.jarque_bera(r)

        # VaR / CVaR (Historical)
        var_95 = float(np.percentile(r, 5))
        var_99 = float(np.percentile(r, 1))
        cvar_95 = float(np.mean(r[r <= var_95]))
        cvar_99 = float(np.mean(r[r <= var_99])) if np.any(r <= var_99) else var_99

        # Cornish-Fisher VaR
        var_95_cf = self._cornish_fisher_var(mean, std, skew, kurt, 0.95)
        cvar_95_cf = self._cornish_fisher_cvar(r, var_95_cf)

        # EVT (Peaks Over Threshold)
        evt_shape, evt_scale, evt_threshold = self._fit_gpd(r)
        evt_var_99 = self._evt_var(evt_shape, evt_scale, evt_threshold, r, 0.99)
        evt_cvar_99 = self._evt_cvar(evt_shape, evt_scale, evt_threshold, evt_var_99)

        # Tail dependence
        if benchmark_returns is not None and len(benchmark_returns) >= len(r):
            lower_tail, upper_tail = self._tail_dependence(r, benchmark_returns[:len(r)])
        else:
            lower_tail, upper_tail = 0.0, 0.0

        # Drawdown analysis
        dd_series = self._drawdown_series(r)
        max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
        dar_95 = float(np.percentile(dd_series, 95)) if len(dd_series) > 0 else 0.0
        es_dd = float(np.mean(dd_series[dd_series >= dar_95])) if np.any(dd_series >= dar_95) else dar_95

        # Composite tail risk score
        score = self._compute_tail_score(
            skew, kurt, jb_p, cvar_95, var_95, evt_shape, max_dd,
        )

        return TailRiskResult(
            var_95=var_95, var_99=var_99,
            cvar_95=cvar_95, cvar_99=cvar_99,
            var_95_cf=var_95_cf, cvar_95_cf=cvar_95_cf,
            mean=mean, std=std, skewness=skew, kurtosis=kurt,
            jarque_bera_p=float(jb_p),
            evt_shape=evt_shape, evt_scale=evt_scale, evt_threshold=evt_threshold,
            evt_var_99=evt_var_99, evt_cvar_99=evt_cvar_99,
            lower_tail_coef=lower_tail, upper_tail_coef=upper_tail,
            max_drawdown=max_dd, dar_95=dar_95, expected_shortfall_dd=es_dd,
            tail_risk_score=score,
        )

    # ------------------------------------------------------------------
    # CORNISH-FISHER
    # ------------------------------------------------------------------
    def _cornish_fisher_var(
        self, mu: float, sigma: float, skew: float, kurt: float, conf: float,
    ) -> float:
        """VaR adjusted for skewness and kurtosis."""
        z = sp_stats.norm.ppf(1 - conf)
        z_cf = (z + (z**2 - 1) * skew / 6
                + (z**3 - 3*z) * kurt / 24
                - (2*z**3 - 5*z) * skew**2 / 36)
        return mu + z_cf * sigma

    def _cornish_fisher_cvar(self, returns: np.ndarray, var_cf: float) -> float:
        """CVaR using CF threshold."""
        tail = returns[returns <= var_cf]
        return float(np.mean(tail)) if len(tail) > 0 else var_cf

    # ------------------------------------------------------------------
    # EVT (Generalized Pareto Distribution)
    # ------------------------------------------------------------------
    def _fit_gpd(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit GPD to tail losses using Peaks Over Threshold.
        Threshold = 10th percentile of losses.
        """
        losses = -returns  # Work with positive losses
        threshold = float(np.percentile(losses, 90))  # Top 10% of losses
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 5:
            return 0.0, float(np.std(returns)), threshold

        try:
            shape, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)
            shape = np.clip(shape, -0.5, 0.5)
        except Exception:
            shape = 0.0
            scale = float(np.mean(exceedances))

        return float(shape), float(scale), float(threshold)

    def _evt_var(
        self, shape: float, scale: float, threshold: float,
        returns: np.ndarray, conf: float,
    ) -> float:
        """VaR from EVT/GPD."""
        n = len(returns)
        n_exceed = np.sum(-returns > threshold)
        p = 1 - conf

        if n_exceed == 0 or scale <= 0:
            return float(np.percentile(returns, (1 - conf) * 100))

        ratio = n_exceed / n
        if abs(shape) > 1e-6:
            var = threshold + (scale / shape) * ((p / ratio) ** (-shape) - 1)
        else:
            var = threshold + scale * np.log(ratio / p)

        return -float(var)  # Return as negative (loss)

    def _evt_cvar(
        self, shape: float, scale: float, threshold: float, var: float,
    ) -> float:
        """CVaR from EVT."""
        if abs(1 - shape) < 1e-6:
            return var * 1.5
        return (var + scale - shape * threshold) / (1 - shape)

    # ------------------------------------------------------------------
    # TAIL DEPENDENCE
    # ------------------------------------------------------------------
    def _tail_dependence(
        self, r1: np.ndarray, r2: np.ndarray, quantile: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Empirical tail dependence coefficients.
        λ_L = P(r2 < q | r1 < q) — joint extreme losses
        λ_U = P(r2 > 1-q | r1 > 1-q) — joint extreme gains
        """
        n = min(len(r1), len(r2))
        r1, r2 = r1[:n], r2[:n]

        q_low_1 = np.percentile(r1, quantile * 100)
        q_low_2 = np.percentile(r2, quantile * 100)
        q_high_1 = np.percentile(r1, (1 - quantile) * 100)
        q_high_2 = np.percentile(r2, (1 - quantile) * 100)

        # Lower tail
        both_low = np.sum((r1 <= q_low_1) & (r2 <= q_low_2))
        one_low = np.sum(r1 <= q_low_1)
        lambda_l = both_low / max(one_low, 1)

        # Upper tail
        both_high = np.sum((r1 >= q_high_1) & (r2 >= q_high_2))
        one_high = np.sum(r1 >= q_high_1)
        lambda_u = both_high / max(one_high, 1)

        return float(lambda_l), float(lambda_u)

    # ------------------------------------------------------------------
    # DRAWDOWN SERIES
    # ------------------------------------------------------------------
    def _drawdown_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute running drawdown from peak."""
        equity = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-10)
        return dd

    # ------------------------------------------------------------------
    # COMPOSITE SCORE
    # ------------------------------------------------------------------
    def _compute_tail_score(
        self, skew, kurt, jb_p, cvar_95, var_95, evt_shape, max_dd,
    ) -> float:
        """
        Composite tail risk score [0, 1]. Higher = riskier tails.
        """
        score = 0.0

        # Negative skew = left tail risk
        if skew < -0.5:
            score += min(abs(skew) / 3.0, 0.25)

        # High kurtosis = fat tails
        if kurt > 1.0:
            score += min(kurt / 10.0, 0.25)

        # Non-normal (JB rejects)
        if jb_p < 0.05:
            score += 0.1

        # CVaR much worse than VaR = tail concentration
        if abs(var_95) > 1e-6:
            cvar_var_ratio = abs(cvar_95 / var_95)
            if cvar_var_ratio > 1.5:
                score += min((cvar_var_ratio - 1) / 3.0, 0.15)

        # Positive EVT shape = heavy tails
        if evt_shape > 0.1:
            score += min(evt_shape / 0.5, 0.15)

        # Max drawdown
        if max_dd > 0.20:
            score += min((max_dd - 0.20) / 0.30, 0.10)

        return min(score, 1.0)

    # ------------------------------------------------------------------
    def _empty_result(self) -> TailRiskResult:
        return TailRiskResult(
            var_95=0, var_99=0, cvar_95=0, cvar_99=0,
            var_95_cf=0, cvar_95_cf=0,
            mean=0, std=0, skewness=0, kurtosis=0, jarque_bera_p=1.0,
            evt_shape=0, evt_scale=0, evt_threshold=0,
            evt_var_99=0, evt_cvar_99=0,
            lower_tail_coef=0, upper_tail_coef=0,
            max_drawdown=0, dar_95=0, expected_shortfall_dd=0,
            tail_risk_score=0,
        )
