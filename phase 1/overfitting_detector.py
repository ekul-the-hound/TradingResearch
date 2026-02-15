# ==============================================================================
# overfitting_detector.py
# ==============================================================================
# Module 2 of 4 — Phase 1: Foundation Completion
#
# Probability of Backtest Overfitting & Deflated Sharpe Ratio
#
# Implements the Bailey, Borwein, Lopez de Prado & Zhu (2015) framework:
#   1. PBO via Combinatorially Symmetric Cross-Validation (CSCV)
#   2. Deflated Sharpe Ratio (DSR) — corrects for selection bias
#   3. Probabilistic Sharpe Ratio (PSR) — minimum track record length
#   4. quantstats integration for HTML tearsheet generation
#
# GitHub repos:
#   - pypbo algorithm reimplemented from Bailey et al. paper directly.
#     (esvhd/pypbo is unmaintained and not on PyPI — implementing from
#     the source paper is more robust and eliminates a dead dependency.)
#   - ranaroussi/quantstats (https://github.com/ranaroussi/quantstats)
#     for one-liner HTML performance tearsheets.
#
# Consumed by:
#   - filtering_pipeline.py (calls compute_pbo on strategy batches,
#     deflated_sharpe_ratio on individual strategies)
#   - lineage_tracker.py (PBO/DSR scores stored in backtest_metrics)
#
# Usage:
#     from overfitting_detector import OverfittingDetector
#
#     detector = OverfittingDetector()
#
#     pbo = detector.compute_pbo(returns_df, n_partitions=16)
#     print(f"PBO = {pbo.probability:.2%}")
#
#     dsr = detector.deflated_sharpe_ratio(
#         observed_sharpe=1.5, n_trials=200, T=252,
#         skewness=-0.3, kurtosis=3.5
#     )
#     print(dsr)
#
# ==============================================================================

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from itertools import combinations
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from pathlib import Path
from joblib import Parallel, delayed

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False


# ==============================================================================
# RESULT DATACLASSES
# ==============================================================================

@dataclass
class PBOResult:
    """Result of Probability of Backtest Overfitting analysis."""
    probability: float              # PBO: P(best IS underperforms median OOS)
    logits: np.ndarray             # Distribution of logit values from CSCV
    logit_mean: float              # Mean logit (negative → overfitting)
    logit_std: float
    n_partitions: int
    n_combinations: int
    performance_degradation: float # Rank correlation IS vs OOS
    stochastic_dominance: float    # Fraction of negative logits
    is_overfit: bool               # True if PBO > 0.5

    def __str__(self) -> str:
        tag = "⚠️  OVERFIT" if self.is_overfit else "✅ OK"
        return (
            f"PBO = {self.probability:.2%} {tag}\n"
            f"  Logit mean:       {self.logit_mean:.4f}\n"
            f"  Logit std:        {self.logit_std:.4f}\n"
            f"  Rank degradation: {self.performance_degradation:.4f}\n"
            f"  Partitions: {self.n_partitions}, Combos: {self.n_combinations}"
        )


@dataclass
class DSRResult:
    """Result of Deflated Sharpe Ratio analysis."""
    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    p_value: float
    n_trials: int
    track_length: int
    min_track_record_length: int
    is_significant: bool

    def __str__(self) -> str:
        tag = "✅ SIGNIFICANT" if self.is_significant else "⚠️  NOT SIGNIFICANT"
        return (
            f"DSR = {self.deflated_sharpe:.4f} {tag}\n"
            f"  Observed SR:  {self.observed_sharpe:.4f}\n"
            f"  E[max(SR)]:   {self.expected_max_sharpe:.4f}\n"
            f"  p-value:      {self.p_value:.4f}\n"
            f"  Trials:       {self.n_trials}\n"
            f"  Min TRL:      {self.min_track_record_length} periods"
        )


@dataclass
class PSRResult:
    """Result of Probabilistic Sharpe Ratio analysis."""
    psr: float
    observed_sharpe: float
    benchmark_sharpe: float
    track_length: int
    skewness: float
    kurtosis: float
    min_track_record_length: int

    def __str__(self) -> str:
        return (
            f"PSR = {self.psr:.2%}\n"
            f"  Observed SR:  {self.observed_sharpe:.4f}\n"
            f"  Benchmark:    {self.benchmark_sharpe:.4f}\n"
            f"  Track len:    {self.track_length}\n"
            f"  Min TRL:      {self.min_track_record_length}"
        )


# ==============================================================================
# OVERFITTING DETECTOR
# ==============================================================================

class OverfittingDetector:
    """
    Detects backtest overfitting using statistical methods from
    Bailey, Borwein, Lopez de Prado & Zhu (2015).
    """

    def __init__(self, random_seed: int = 42, n_jobs: int = -1):
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # PBO via CSCV
    # ------------------------------------------------------------------
    def compute_pbo(
        self,
        returns_df: pd.DataFrame,
        n_partitions: int = 16,
        metric_func: Optional[callable] = None,
        threshold: float = 0.0,
    ) -> PBOResult:
        """
        Compute Probability of Backtest Overfitting via CSCV.

        Partitions return matrix into S subsets. For each C(S, S/2)
        combination of IS vs OOS, checks whether the optimal IS strategy
        underperforms OOS.

        Args:
            returns_df: (T, N) DataFrame — columns are strategy returns.
            n_partitions: Even integer >= 4. Higher = more precise, slower.
            metric_func: f(Series) → float. Default: annualized Sharpe.
            threshold: OOS floor. Default 0.0.

        Returns:
            PBOResult
        """
        if metric_func is None:
            metric_func = self._annualized_sharpe

        T, N = returns_df.shape
        assert n_partitions % 2 == 0 and n_partitions >= 4, "n_partitions must be even >= 4"
        assert N >= 2, "Need >= 2 strategies"

        indices = np.arange(T)
        partitions = np.array_split(indices, n_partitions)
        half = n_partitions // 2
        combos = list(combinations(range(n_partitions), half))

        def _eval(combo):
            is_idx = np.concatenate([partitions[i] for i in combo])
            oos_idx = np.concatenate([partitions[i] for i in range(n_partitions) if i not in combo])

            is_m = returns_df.iloc[is_idx].apply(metric_func).values
            oos_m = returns_df.iloc[oos_idx].apply(metric_func).values

            best = np.argmax(is_m)
            rank = sp_stats.rankdata(oos_m)[best]
            rank_pct = np.clip(rank / N, 1e-6, 1 - 1e-6)
            logit = np.log(rank_pct / (1 - rank_pct))
            corr = sp_stats.spearmanr(is_m, oos_m).statistic
            under = 1.0 if oos_m[best] < threshold else 0.0
            return logit, corr, under

        results = Parallel(n_jobs=self.n_jobs)(delayed(_eval)(c) for c in combos)

        logits = np.array([r[0] for r in results])
        corrs  = np.array([r[1] for r in results])
        unders = np.array([r[2] for r in results])

        return PBOResult(
            probability=float(np.mean(unders)),
            logits=logits,
            logit_mean=float(np.mean(logits)),
            logit_std=float(np.std(logits)) if len(logits) > 1 else 0.0,
            n_partitions=n_partitions,
            n_combinations=len(combos),
            performance_degradation=float(np.mean(corrs)),
            stochastic_dominance=float(np.mean(logits < 0)),
            is_overfit=float(np.mean(unders)) > 0.5,
        )

    # ------------------------------------------------------------------
    # DEFLATED SHARPE RATIO
    # ------------------------------------------------------------------
    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        T: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        sharpe_std: float = 1.0,
    ) -> DSRResult:
        """
        Correct an observed Sharpe for the number of trials tested.

        DSR = (SR_obs − E[max(SR)]) / SE(SR)
        """
        e_max = self._expected_max_sharpe(n_trials, T, sharpe_std)

        se = np.sqrt(
            (1 - skewness * observed_sharpe
             + (kurtosis - 1) / 4 * observed_sharpe ** 2) / max(T - 1, 1)
        )
        se = max(se, 1e-10)

        dsr = (observed_sharpe - e_max) / se
        p_val = 1 - float(sp_stats.norm.cdf(dsr))

        min_trl = self._min_track_record_length(observed_sharpe, n_trials, sharpe_std)

        return DSRResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=dsr,
            expected_max_sharpe=e_max,
            p_value=p_val,
            n_trials=n_trials,
            track_length=T,
            min_track_record_length=min_trl,
            is_significant=dsr > 0,
        )

    def _expected_max_sharpe(self, n: int, T: int, std: float = 1.0) -> float:
        if n <= 1:
            return 0.0
        gamma = 0.5772156649  # Euler-Mascheroni
        z1 = sp_stats.norm.ppf(1 - 1.0 / n)
        z2 = sp_stats.norm.ppf(1 - 1.0 / (n * np.e))
        return std * ((1 - gamma) * z1 + gamma * z2) / np.sqrt(T)

    def _min_track_record_length(self, sr: float, n: int, std: float = 1.0) -> int:
        if abs(sr) < 1e-10:
            return 999999
        e_max = self._expected_max_sharpe(n, 1, std)
        return max(int(np.ceil((e_max / sr) ** 2)), 1)

    # ------------------------------------------------------------------
    # PROBABILISTIC SHARPE RATIO
    # ------------------------------------------------------------------
    def probabilistic_sharpe_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_sharpe: float = 0.0,
    ) -> PSRResult:
        """PSR = Φ((SR − SR*) / SE(SR))"""
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        T = len(r)
        if T < 3:
            return PSRResult(0.0, 0.0, benchmark_sharpe, T, 0.0, 3.0, 999999)

        sr = np.mean(r) / max(np.std(r, ddof=1), 1e-10)
        sr_ann = sr * np.sqrt(252)
        skew = float(sp_stats.skew(r))
        kurt = float(sp_stats.kurtosis(r, fisher=False))

        se = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / max(T - 1, 1))
        se = max(se, 1e-10)

        bench = benchmark_sharpe / np.sqrt(252)
        stat = (sr - bench) / se
        prob = float(sp_stats.norm.cdf(stat))

        if abs(sr - bench) < 1e-10:
            min_trl = 999999
        else:
            min_trl = max(
                int(np.ceil((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (sr - bench) ** 2)),
                1,
            )

        return PSRResult(
            psr=prob, observed_sharpe=sr_ann, benchmark_sharpe=benchmark_sharpe,
            track_length=T, skewness=skew, kurtosis=kurt,
            min_track_record_length=min_trl,
        )

    # ------------------------------------------------------------------
    # CONVENIENCE: combined single-strategy analysis
    # ------------------------------------------------------------------
    def analyze_strategy(
        self,
        returns: Union[pd.Series, np.ndarray],
        n_trials: int = 1,
        benchmark_sharpe: float = 0.0,
    ) -> Dict:
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        T = len(r)

        psr = self.probabilistic_sharpe_ratio(r, benchmark_sharpe)
        skew = float(sp_stats.skew(r)) if T >= 3 else 0.0
        kurt = float(sp_stats.kurtosis(r, fisher=False)) if T >= 3 else 3.0

        dsr = self.deflated_sharpe_ratio(
            psr.observed_sharpe, max(n_trials, 1), T, skew, kurt,
        )
        return {
            "psr": psr, "dsr": dsr,
            "sharpe_annualized": psr.observed_sharpe,
            "skewness": skew, "kurtosis": kurt,
            "track_length": T, "n_trials": n_trials,
        }

    # ------------------------------------------------------------------
    # QUANTSTATS INTEGRATION
    # ------------------------------------------------------------------
    def generate_tearsheet(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        output_path: str = "tearsheet.html",
        title: str = "Strategy Performance",
    ) -> Optional[str]:
        if not QUANTSTATS_AVAILABLE:
            print("⚠️  quantstats not installed. pip install quantstats")
            return None
        try:
            qs.reports.html(returns, benchmark=benchmark, output=output_path, title=title)
            print(f"✓ Tearsheet → {output_path}")
            return output_path
        except Exception as e:
            print(f"⚠️  Tearsheet failed: {e}")
            return None

    def get_quantstats_metrics(self, returns: pd.Series) -> Dict:
        if not QUANTSTATS_AVAILABLE:
            return {}
        try:
            m = {
                "sharpe": qs.stats.sharpe(returns),
                "sortino": qs.stats.sortino(returns),
                "max_drawdown": qs.stats.max_drawdown(returns),
                "cagr": qs.stats.cagr(returns),
                "calmar": qs.stats.calmar(returns),
                "volatility": qs.stats.volatility(returns),
                "win_rate": qs.stats.win_rate(returns),
                "value_at_risk": qs.stats.value_at_risk(returns),
            }
            return {k: float(v) if v is not None and np.isfinite(v) else None
                    for k, v in m.items()}
        except Exception as e:
            print(f"⚠️  quantstats metrics failed: {e}")
            return {}

    # ------------------------------------------------------------------
    @staticmethod
    def _annualized_sharpe(s: pd.Series) -> float:
        r = s.values
        r = r[~np.isnan(r)]
        if len(r) < 2 or np.std(r, ddof=1) < 1e-10:
            return 0.0
        return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
