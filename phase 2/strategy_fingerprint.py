# ==============================================================================
# strategy_fingerprint.py
# ==============================================================================
# Phase 2, Module 1 (Week 5): Strategy Feature Extraction
#
# Converts strategy metadata + backtest results into fixed-length numeric
# feature vectors ("fingerprints"). These vectors are the input space for
# surrogate models — enabling cheap performance prediction without running
# full backtests.
#
# Feature categories:
#   1. Performance metrics (Sharpe, return, drawdown, trades)
#   2. Parameter signature (normalized indicator params)
#   3. Structural descriptors (entry/exit type, indicator count, regime)
#   4. Lineage features (generation, mutation type, parent performance)
#   5. Robustness indicators (PBO, DSR, walk-forward degradation)
#
# GitHub repos: scikit-learn (StandardScaler, PCA)
#
# Consumed by:
#   - surrogate_model.py (training data X)
#   - acquisition_function.py (candidate evaluation)
#   - multi_objective_optimizer.py (population fitness)
#
# Usage:
#     from strategy_fingerprint import StrategyFingerprinter
#
#     fp = StrategyFingerprinter()
#     X, feature_names = fp.transform(strategies)
#     # X: (N, D) numpy array, feature_names: list of str
#
# ==============================================================================

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import hashlib
import json


# ==============================================================================
# FEATURE SPEC
# ==============================================================================

# Each tuple: (key_path, default, normalization)
PERFORMANCE_FEATURES = [
    ("sharpe_ratio", 0.0),
    ("total_return_pct", 0.0),
    ("max_drawdown_pct", 20.0),
    ("total_trades", 50),
    ("win_rate", 0.5),
    ("profit_factor", 1.0),
    ("avg_trade_pct", 0.0),
    ("max_consecutive_losses", 5),
    ("recovery_factor", 0.0),
    ("calmar_ratio", 0.0),
    ("sortino_ratio", 0.0),
    ("volatility_annual", 0.15),
]

STRUCTURAL_FEATURES = [
    ("n_indicators", 2),
    ("n_entry_conditions", 1),
    ("n_exit_conditions", 1),
    ("uses_trailing_stop", 0),
    ("uses_take_profit", 0),
    ("uses_stop_loss", 0),
    ("uses_time_exit", 0),
    ("is_trend_following", 0),
    ("is_mean_reversion", 0),
    ("is_momentum", 0),
    ("is_breakout", 0),
    ("timeframe_minutes", 60),
]

LINEAGE_FEATURES = [
    ("generation", 0),
    ("parent_sharpe", 0.0),
    ("parent_return", 0.0),
    ("siblings_count", 0),
    ("mutation_distance", 1.0),
]

ROBUSTNESS_FEATURES = [
    ("pbo_probability", 0.5),
    ("deflated_sharpe", 0.0),
    ("psr", 0.5),
    ("wf_degradation", 0.5),
    ("regime_consistency", 0.5),
    ("latency_sensitivity", 0.0),
    ("slippage_survival", 1.0),
]

# Mutation type one-hot categories
MUTATION_TYPES = [
    "add_indicator", "remove_indicator", "change_params",
    "add_filter", "change_exit", "change_entry",
    "add_trailing_stop", "change_timeframe", "crossover",
    "discovered", "manual",
]


# ==============================================================================
# FINGERPRINTER
# ==============================================================================

@dataclass
class FingerprintResult:
    """Output of fingerprinting a batch of strategies."""
    X: np.ndarray              # (N, D) feature matrix
    feature_names: List[str]   # length D
    strategy_ids: List[str]    # length N
    scaler: Optional[StandardScaler] = None


class StrategyFingerprinter:
    """
    Converts strategy dicts into fixed-length numeric feature vectors.
    """

    def __init__(self, fit_scaler: bool = True):
        self.fit_scaler = fit_scaler
        self.scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            self._feature_names = self._build_feature_names()
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    # ------------------------------------------------------------------
    # MAIN API
    # ------------------------------------------------------------------
    def transform(
        self,
        strategies: List[Dict[str, Any]],
        fit: bool = True,
    ) -> FingerprintResult:
        """
        Convert a list of strategy dicts into a feature matrix.

        Args:
            strategies: List of dicts with metric keys.
            fit: If True and fit_scaler is True, fit the scaler on this data.

        Returns:
            FingerprintResult with (N, D) matrix.
        """
        rows = []
        ids = []
        for s in strategies:
            row = self._extract_features(s)
            rows.append(row)
            ids.append(s.get("strategy_id", s.get("name", "unknown")))

        X = np.array(rows, dtype=np.float64)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

        # Optionally scale
        if self.fit_scaler:
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        return FingerprintResult(
            X=X_scaled,
            feature_names=self.feature_names,
            strategy_ids=ids,
            scaler=self.scaler,
        )

    def transform_single(self, strategy: Dict[str, Any]) -> np.ndarray:
        """Transform a single strategy into a feature vector."""
        row = self._extract_features(strategy)
        x = np.array([row], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x[0]

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------------
    def _extract_features(self, s: Dict[str, Any]) -> List[float]:
        """Extract all feature groups from a strategy dict."""
        features = []

        # Performance
        for key, default in PERFORMANCE_FEATURES:
            features.append(float(s.get(key, default) or default))

        # Structural
        for key, default in STRUCTURAL_FEATURES:
            features.append(float(s.get(key, default) or default))

        # Lineage
        for key, default in LINEAGE_FEATURES:
            features.append(float(s.get(key, default) or default))

        # Robustness
        for key, default in ROBUSTNESS_FEATURES:
            features.append(float(s.get(key, default) or default))

        # Mutation type one-hot
        mt = s.get("mutation_type", s.get("origin", "discovered"))
        for t in MUTATION_TYPES:
            features.append(1.0 if mt == t else 0.0)

        # Parameter hash (deterministic numeric encoding of params)
        params = s.get("strategy_params", s.get("mutation_params", {}))
        features.append(self._param_hash_feature(params))

        # Parameter count
        if isinstance(params, dict):
            features.append(float(len(params)))
        elif isinstance(params, str):
            try:
                features.append(float(len(json.loads(params))))
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)

        return features

    # ------------------------------------------------------------------
    # FEATURE NAMES
    # ------------------------------------------------------------------
    def _build_feature_names(self) -> List[str]:
        names = []
        for key, _ in PERFORMANCE_FEATURES:
            names.append(f"perf_{key}")
        for key, _ in STRUCTURAL_FEATURES:
            names.append(f"struct_{key}")
        for key, _ in LINEAGE_FEATURES:
            names.append(f"lin_{key}")
        for key, _ in ROBUSTNESS_FEATURES:
            names.append(f"rob_{key}")
        for t in MUTATION_TYPES:
            names.append(f"mut_{t}")
        names.append("param_hash")
        names.append("param_count")
        return names

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    @staticmethod
    def _param_hash_feature(params: Any) -> float:
        """Deterministic numeric hash of parameter dict."""
        if params is None:
            return 0.0
        try:
            s = json.dumps(params, sort_keys=True, default=str)
        except Exception:
            s = str(params)
        h = hashlib.md5(s.encode()).hexdigest()[:8]
        return float(int(h, 16)) / float(0xFFFFFFFF)

    @staticmethod
    def compute_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Cosine similarity between two fingerprint vectors."""
        n1 = np.linalg.norm(fp1)
        n2 = np.linalg.norm(fp2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(fp1, fp2) / (n1 * n2))

    @staticmethod
    def compute_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Euclidean distance between two fingerprint vectors."""
        return float(np.linalg.norm(fp1 - fp2))

    def get_feature_importance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Quick feature importance via correlation with target."""
        importances = {}
        for i, name in enumerate(self.feature_names):
            col = X[:, i]
            if np.std(col) < 1e-10:
                importances[name] = 0.0
            else:
                importances[name] = abs(float(np.corrcoef(col, y)[0, 1]))
        return dict(sorted(importances.items(), key=lambda x: -x[1]))
