# ==============================================================================
# drift_detector.py
# ==============================================================================
# Phase 4, Module 1 (Week 15): Statistical Drift Detection
#
# Detects when live strategy performance diverges from its backtest
# distribution. This is the early warning system — if drift is detected,
# the strategy may need retraining, parameter adjustment, or retirement.
#
# Detection methods:
#   1. Kolmogorov-Smirnov test (distribution shift)
#   2. Population Stability Index (binned distribution comparison)
#   3. CUSUM (cumulative sum change detection)
#   4. Page-Hinkley (online change detection, lighter than CUSUM)
#   5. Rolling Sharpe degradation
#   6. Return distribution moments (mean, var, skew, kurt shifts)
#
# Severity levels: NONE → WARNING → CRITICAL
#
# Consumed by:
#   - live_monitor.py (real-time alerting)
#   - strategy_lifecycle.py (demotion triggers)
#   - kill_switch.py (drift-based halting)
#
# Usage:
#     from drift_detector import DriftDetector
#     dd = DriftDetector(reference_returns=backtest_returns)
#     result = dd.check(live_returns)
#     if result.drift_detected:
#         print(f"DRIFT: {result.severity} — {result.message}")
# ==============================================================================

import numpy as np
from scipy import stats as sp_stats
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# ==============================================================================
# ENUMS
# ==============================================================================

class DriftSeverity(Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class DriftResult:
    """Output from drift detection."""
    drift_detected: bool
    severity: DriftSeverity
    message: str
    timestamp: str

    # Per-test results
    ks_stat: float = 0.0
    ks_pvalue: float = 1.0
    psi: float = 0.0
    cusum_signal: bool = False
    cusum_value: float = 0.0
    page_hinkley_signal: bool = False
    sharpe_ratio_live: float = 0.0
    sharpe_ratio_ref: float = 0.0
    sharpe_degradation_pct: float = 0.0
    mean_shift: float = 0.0
    var_shift: float = 0.0

    # Triggered tests
    triggered_tests: List[str] = field(default_factory=list)

    def __str__(self):
        icon = {"none": "✅", "warning": "⚠️", "critical": "🔴"}[self.severity.value]
        tests = ", ".join(self.triggered_tests) if self.triggered_tests else "none"
        return f"{icon} Drift [{self.severity.value}]: {self.message} | Triggered: {tests}"


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class DriftConfig:
    """Thresholds for drift detection."""
    # KS test
    ks_warning_pvalue: float = 0.05
    ks_critical_pvalue: float = 0.01

    # PSI
    psi_warning: float = 0.10
    psi_critical: float = 0.25

    # CUSUM
    cusum_threshold: float = 5.0     # Std devs from mean
    cusum_drift_length: int = 10     # Consecutive signals

    # Page-Hinkley
    ph_delta: float = 0.005          # Minimum detectable change
    ph_threshold: float = 50.0       # Detection threshold

    # Sharpe degradation
    sharpe_warning_pct: float = 30.0
    sharpe_critical_pct: float = 50.0

    # Moments
    mean_shift_warning: float = 2.0   # Z-score of mean shift
    var_shift_warning: float = 2.0    # Ratio of variance change

    # Minimum data
    min_observations: int = 20


# ==============================================================================
# DRIFT DETECTOR
# ==============================================================================

class DriftDetector:
    """
    Statistical drift detection for strategy monitoring.
    """

    def __init__(
        self,
        reference_returns: Optional[np.ndarray] = None,
        config: Optional[DriftConfig] = None,
    ):
        self.config = config or DriftConfig()
        self._ref: Optional[np.ndarray] = None
        self._ref_stats: Dict[str, float] = {}

        # CUSUM / Page-Hinkley state (online)
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._cusum_count = 0
        self._ph_sum = 0.0
        self._ph_min = float("inf")
        self._ph_count = 0
        self._ph_mean = 0.0

        self.history: List[DriftResult] = []

        if reference_returns is not None:
            self.set_reference(reference_returns)

    # ------------------------------------------------------------------
    # REFERENCE
    # ------------------------------------------------------------------
    def set_reference(self, returns: np.ndarray):
        """Set the reference (backtest) return distribution."""
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        self._ref = r
        self._ref_stats = {
            "mean": float(np.mean(r)),
            "std": float(np.std(r, ddof=1)),
            "skew": float(sp_stats.skew(r)),
            "kurt": float(sp_stats.kurtosis(r)),
            "sharpe": float(np.mean(r) / max(np.std(r, ddof=1), 1e-10) * np.sqrt(252)),
            "n": len(r),
        }

    # ------------------------------------------------------------------
    # MAIN CHECK (batch)
    # ------------------------------------------------------------------
    def check(self, live_returns: np.ndarray) -> DriftResult:
        """
        Run all drift tests on a batch of live returns.
        """
        ts = datetime.now().isoformat()
        live = np.asarray(live_returns, dtype=np.float64)
        live = live[np.isfinite(live)]

        if self._ref is None or len(live) < self.config.min_observations:
            return DriftResult(False, DriftSeverity.NONE, "Insufficient data", ts)

        cfg = self.config
        triggered = []
        severity = DriftSeverity.NONE

        # 1. KS test
        ks_stat, ks_p = sp_stats.ks_2samp(self._ref, live)
        if ks_p < cfg.ks_critical_pvalue:
            triggered.append("ks_critical")
            severity = DriftSeverity.CRITICAL
        elif ks_p < cfg.ks_warning_pvalue:
            triggered.append("ks_warning")
            severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # 2. PSI
        psi = self._compute_psi(self._ref, live)
        if psi > cfg.psi_critical:
            triggered.append("psi_critical")
            severity = DriftSeverity.CRITICAL
        elif psi > cfg.psi_warning:
            triggered.append("psi_warning")
            severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # 3. CUSUM
        cusum_val, cusum_signal = self._cusum_batch(live)
        if cusum_signal:
            triggered.append("cusum")
            severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # 4. Sharpe degradation
        live_sharpe = float(np.mean(live) / max(np.std(live, ddof=1), 1e-10) * np.sqrt(252))
        ref_sharpe = self._ref_stats["sharpe"]
        if ref_sharpe > 0:
            degrad = (1 - live_sharpe / ref_sharpe) * 100
        else:
            degrad = 0.0
        if degrad > cfg.sharpe_critical_pct:
            triggered.append("sharpe_critical")
            severity = DriftSeverity.CRITICAL
        elif degrad > cfg.sharpe_warning_pct:
            triggered.append("sharpe_warning")
            severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # 5. Mean shift
        mean_shift_z = abs(np.mean(live) - self._ref_stats["mean"]) / max(self._ref_stats["std"] / np.sqrt(len(live)), 1e-10)
        if mean_shift_z > cfg.mean_shift_warning:
            triggered.append("mean_shift")
            severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # 6. Variance shift
        var_ratio = np.var(live, ddof=1) / max(np.var(self._ref, ddof=1), 1e-10)
        if var_ratio > cfg.var_shift_warning or var_ratio < 1 / cfg.var_shift_warning:
            triggered.append("var_shift")
            severity = max(severity, DriftSeverity.WARNING, key=lambda s: s.value)

        # Build result
        drift_detected = len(triggered) > 0
        msg = f"{len(triggered)} test(s) triggered" if drift_detected else "No drift"

        result = DriftResult(
            drift_detected=drift_detected,
            severity=severity,
            message=msg,
            timestamp=ts,
            ks_stat=float(ks_stat), ks_pvalue=float(ks_p),
            psi=psi,
            cusum_signal=cusum_signal, cusum_value=cusum_val,
            sharpe_ratio_live=live_sharpe, sharpe_ratio_ref=ref_sharpe,
            sharpe_degradation_pct=degrad,
            mean_shift=mean_shift_z, var_shift=var_ratio,
            triggered_tests=triggered,
        )
        self.history.append(result)
        return result

    # ------------------------------------------------------------------
    # ONLINE UPDATE (streaming)
    # ------------------------------------------------------------------
    def update(self, new_return: float) -> DriftResult:
        """
        Process a single new return (online/streaming).
        Uses CUSUM + Page-Hinkley for change detection.
        """
        ts = datetime.now().isoformat()
        if self._ref is None:
            return DriftResult(False, DriftSeverity.NONE, "No reference", ts)

        ref_mean = self._ref_stats["mean"]
        ref_std = max(self._ref_stats["std"], 1e-10)

        # CUSUM update
        z = (new_return - ref_mean) / ref_std
        self._cusum_pos = max(0, self._cusum_pos + z - 0.5)
        self._cusum_neg = max(0, self._cusum_neg - z - 0.5)
        cusum_signal = (self._cusum_pos > self.config.cusum_threshold or
                        self._cusum_neg > self.config.cusum_threshold)

        # Page-Hinkley update
        self._ph_count += 1
        self._ph_mean += (new_return - self._ph_mean) / self._ph_count
        self._ph_sum += new_return - self._ph_mean - self.config.ph_delta
        self._ph_min = min(self._ph_min, self._ph_sum)
        ph_signal = (self._ph_sum - self._ph_min) > self.config.ph_threshold

        triggered = []
        severity = DriftSeverity.NONE
        if cusum_signal:
            triggered.append("cusum_online")
            severity = DriftSeverity.WARNING
        if ph_signal:
            triggered.append("page_hinkley")
            severity = DriftSeverity.CRITICAL if cusum_signal else DriftSeverity.WARNING

        result = DriftResult(
            drift_detected=len(triggered) > 0,
            severity=severity,
            message=f"Online: {len(triggered)} signal(s)" if triggered else "OK",
            timestamp=ts,
            cusum_signal=cusum_signal,
            cusum_value=max(self._cusum_pos, self._cusum_neg),
            page_hinkley_signal=ph_signal,
            triggered_tests=triggered,
        )
        self.history.append(result)
        return result

    def reset_online(self):
        """Reset online detectors (e.g., after strategy adjustment)."""
        self._cusum_pos = self._cusum_neg = 0.0
        self._ph_sum = 0.0
        self._ph_min = float("inf")
        self._ph_count = 0
        self._ph_mean = 0.0

    # ------------------------------------------------------------------
    # PSI (Population Stability Index)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """
        PSI = Σ (p_i - q_i) × ln(p_i / q_i)
        Measures how much the current distribution has shifted.
        < 0.10 = no shift, 0.10-0.25 = moderate, > 0.25 = significant
        """
        eps = 1e-6
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

        ref_counts = np.maximum(ref_counts, eps)
        cur_counts = np.maximum(cur_counts, eps)

        psi = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
        return psi

    # ------------------------------------------------------------------
    # CUSUM (batch)
    # ------------------------------------------------------------------
    def _cusum_batch(self, returns: np.ndarray) -> Tuple[float, bool]:
        """Batch CUSUM on full live series."""
        ref_mean = self._ref_stats["mean"]
        ref_std = max(self._ref_stats["std"], 1e-10)
        z = (returns - ref_mean) / ref_std
        cusum_pos = np.maximum.accumulate(np.concatenate([[0], np.cumsum(z - 0.5)]))
        cusum_neg = np.maximum.accumulate(np.concatenate([[0], np.cumsum(-z - 0.5)]))
        max_val = max(cusum_pos.max(), cusum_neg.max())
        signal = max_val > self.config.cusum_threshold
        return float(max_val), signal
