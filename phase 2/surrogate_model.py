# ==============================================================================
# surrogate_model.py
# ==============================================================================
# Phase 2, Module 2 (Week 6): Surrogate Performance Model
#
# Trains a cheap statistical model to predict backtest outcomes from strategy
# fingerprints. This avoids running expensive full backtests on every candidate
# in the optimization loop — only the most promising candidates (selected by
# the acquisition function) actually get backtested.
#
# Model types:
#   1. Random Forest (default — fast, robust, handles non-linear)
#   2. Gaussian Process (provides uncertainty estimates for Bayesian optimization)
#   3. Gradient Boosting (high accuracy, no uncertainty)
#
# GitHub repos:
#   - scikit-learn (RandomForestRegressor, GaussianProcessRegressor,
#                   GradientBoostingRegressor, cross_val_score)
#
# Consumed by:
#   - acquisition_function.py (predicts μ and σ for EI/UCB/PI)
#   - optimization_pipeline.py (evaluates candidates cheaply)
#   - multi_objective_optimizer.py (fitness function)
#
# Usage:
#     from surrogate_model import SurrogateModel
#
#     model = SurrogateModel(model_type="gp")
#     model.fit(X_train, y_train)
#
#     # Predict with uncertainty
#     mu, sigma = model.predict(X_test, return_std=True)
#
# ==============================================================================

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import json
from pathlib import Path


# ==============================================================================
# RESULT DATACLASS
# ==============================================================================

@dataclass
class SurrogateMetrics:
    """Training/validation metrics for the surrogate model."""
    mse: float
    rmse: float
    mae: float
    r2: float
    cv_mean: float          # Cross-validated R²
    cv_std: float
    n_train: int
    n_features: int
    model_type: str

    def __str__(self) -> str:
        return (
            f"Surrogate [{self.model_type}] — n={self.n_train}, features={self.n_features}\n"
            f"  R² = {self.r2:.4f} (CV: {self.cv_mean:.4f} ± {self.cv_std:.4f})\n"
            f"  RMSE = {self.rmse:.4f}, MAE = {self.mae:.4f}"
        )


# ==============================================================================
# SURROGATE MODEL
# ==============================================================================

class SurrogateModel:
    """
    Surrogate model that predicts backtest performance from strategy fingerprints.

    Supports three backends:
      - "rf": Random Forest (fast, robust, provides variance via tree disagreement)
      - "gp": Gaussian Process (analytical uncertainty, best for Bayesian opt)
      - "gb": Gradient Boosting (highest accuracy, no native uncertainty)
    """

    def __init__(
        self,
        model_type: str = "rf",
        n_estimators: int = 200,
        random_state: int = 42,
        gp_noise: float = 0.1,
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.gp_noise = gp_noise

        self.model = None
        self.is_fitted = False
        self.metrics: Optional[SurrogateMetrics] = None
        self._feature_names: Optional[List[str]] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

        self._build_model()

    def _build_model(self):
        if self.model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=15,
                min_samples_leaf=3,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gp":
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(self.gp_noise)
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                random_state=self.random_state,
                normalize_y=True,
                alpha=1e-6,
            )
        elif self.model_type == "gb":
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                subsample=0.8,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'rf', 'gp', or 'gb'.")

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
    ) -> SurrogateMetrics:
        """
        Fit the surrogate model.

        Args:
            X: (N, D) feature matrix from StrategyFingerprinter.
            y: (N,) target values (e.g., Sharpe ratio).
            feature_names: Optional feature name list.
            cv_folds: Number of cross-validation folds.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Limit GP to reasonable size (GP is O(N³))
        if self.model_type == "gp" and len(X) > 500:
            idx = np.random.RandomState(self.random_state).choice(
                len(X), 500, replace=False
            )
            X_fit, y_fit = X[idx], y[idx]
        else:
            X_fit, y_fit = X, y

        self.model.fit(X_fit, y_fit)
        self._X_train = X
        self._y_train = y
        self._feature_names = feature_names
        self.is_fitted = True

        # Compute metrics
        y_pred = self.model.predict(X)
        n_cv = min(cv_folds, len(X))
        if n_cv >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    self._clone_model(), X, y, cv=n_cv, scoring="r2"
                )
        else:
            cv_scores = np.array([0.0])

        self.metrics = SurrogateMetrics(
            mse=float(mean_squared_error(y, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y, y_pred))),
            mae=float(mean_absolute_error(y, y_pred)),
            r2=float(r2_score(y, y_pred)),
            cv_mean=float(np.mean(cv_scores)),
            cv_std=float(np.std(cv_scores)),
            n_train=len(X),
            n_features=X.shape[1],
            model_type=self.model_type,
        )
        return self.metrics

    def _clone_model(self):
        """Clone the model for cross-validation."""
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=15,
                min_samples_leaf=3, random_state=self.random_state, n_jobs=-1,
            )
        elif self.model_type == "gp":
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(self.gp_noise)
            return GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=3,
                random_state=self.random_state, normalize_y=True,
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators, max_depth=5,
                learning_rate=0.1, random_state=self.random_state,
            )

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Any:
        """
        Predict performance for new strategy fingerprints.

        Args:
            X: (N, D) or (D,) feature matrix.
            return_std: If True, return (mu, sigma) for uncertainty.

        Returns:
            mu: (N,) predicted values.
            sigma: (N,) standard deviations (if return_std=True).
        """
        assert self.is_fitted, "Model not fitted. Call fit() first."
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        if return_std:
            if self.model_type == "gp":
                mu, sigma = self.model.predict(X, return_std=True)
                return mu, sigma
            elif self.model_type == "rf":
                # Use tree variance as uncertainty proxy
                preds = np.array([t.predict(X) for t in self.model.estimators_])
                mu = preds.mean(axis=0)
                sigma = preds.std(axis=0)
                return mu, sigma
            else:
                # GB: no native uncertainty, use residual std as constant
                mu = self.model.predict(X)
                if self._y_train is not None:
                    y_pred_train = self.model.predict(self._X_train)
                    residual_std = float(np.std(self._y_train - y_pred_train))
                else:
                    residual_std = 1.0
                return mu, np.full_like(mu, residual_std)
        else:
            return self.model.predict(X)

    # ------------------------------------------------------------------
    # UPDATE (incremental)
    # ------------------------------------------------------------------
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Add new data points and refit."""
        X_new = np.atleast_2d(X_new)
        y_new = np.atleast_1d(y_new)
        if self._X_train is not None:
            X_all = np.vstack([self._X_train, X_new])
            y_all = np.concatenate([self._y_train, y_new])
        else:
            X_all = X_new
            y_all = y_new
        self.fit(X_all, y_all, self._feature_names)

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------
    def feature_importance(self) -> Dict[str, float]:
        """Return feature importances (RF/GB) or None (GP)."""
        if not self.is_fitted:
            return {}
        if self.model_type in ("rf", "gb"):
            imp = self.model.feature_importances_
            names = self._feature_names or [f"f{i}" for i in range(len(imp))]
            return dict(sorted(zip(names, imp), key=lambda x: -x[1]))
        return {}

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save model to disk."""
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "feature_names": self._feature_names,
            "X_train": self._X_train,
            "y_train": self._y_train,
            "is_fitted": self.is_fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        """Load model from disk."""
        import joblib
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.model_type = data["model_type"]
        obj.metrics = data["metrics"]
        obj._feature_names = data["feature_names"]
        obj._X_train = data["X_train"]
        obj._y_train = data["y_train"]
        obj.is_fitted = data["is_fitted"]
        return obj


# ==============================================================================
# MULTI-OBJECTIVE SURROGATE
# ==============================================================================

class MultiObjectiveSurrogate:
    """
    Wraps multiple SurrogateModels — one per objective.

    Objectives: sharpe_ratio, max_drawdown_pct, profit_factor, etc.
    """

    def __init__(
        self,
        objectives: List[str],
        model_type: str = "rf",
        **kwargs,
    ):
        self.objectives = objectives
        self.models: Dict[str, SurrogateModel] = {
            obj: SurrogateModel(model_type=model_type, **kwargs)
            for obj in objectives
        }
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        Y: Dict[str, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, SurrogateMetrics]:
        """
        Fit all objective models.

        Args:
            X: (N, D) feature matrix.
            Y: {objective_name: (N,) target array}
        """
        metrics = {}
        for obj in self.objectives:
            if obj in Y:
                m = self.models[obj].fit(X, Y[obj], feature_names)
                metrics[obj] = m
        self.is_fitted = True
        return metrics

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Dict[str, Any]:
        """Predict all objectives."""
        results = {}
        for obj in self.objectives:
            if self.models[obj].is_fitted:
                results[obj] = self.models[obj].predict(X, return_std=return_std)
        return results
