# ==============================================================================
# acquisition_function.py
# ==============================================================================
# Phase 2, Module 3 (Week 7): Bayesian Acquisition Functions
#
# Decides WHICH candidate strategies should actually be backtested.
# Uses surrogate model predictions (μ, σ) to balance exploitation
# (pick strategies predicted to perform well) vs. exploration
# (pick strategies where the surrogate is uncertain).
#
# Acquisition functions:
#   1. Expected Improvement (EI) -- maximize E[max(f(x) - f_best, 0)]
#   2. Upper Confidence Bound (UCB) -- μ + κσ
#   3. Probability of Improvement (PI) -- P(f(x) > f_best + ξ)
#   4. Thompson Sampling (TS) -- sample from posterior
#   5. Multi-Objective Expected Improvement (for Pareto)
#
# No external GitHub repos -- pure numpy/scipy.
#
# Consumed by:
#   - optimization_pipeline.py (candidate selection)
#   - multi_objective_optimizer.py (NSGA-II fitness evaluation)
#
# Usage:
#     from acquisition_function import AcquisitionOptimizer
#
#     acq = AcquisitionOptimizer(surrogate=model, method="ei")
#     indices = acq.select(X_candidates, n_select=10, f_best=1.5)
#
# ==============================================================================

import numpy as np
from scipy.stats import norm
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class AcquisitionResult:
    """Output of acquisition function optimization."""
    selected_indices: List[int]
    scores: np.ndarray          # Acquisition values for ALL candidates
    selected_scores: np.ndarray  # Acquisition values for selected
    method: str
    f_best: float
    n_candidates: int
    n_selected: int


# ==============================================================================
# ACQUISITION FUNCTIONS (standalone, stateless)
# ==============================================================================

def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement: E[max(f(x) - f_best, 0)]

    Higher EI = candidate likely to beat current best by a lot OR
    high uncertainty (exploration).
    """
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def upper_confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float = 2.0,
) -> np.ndarray:
    """
    UCB: μ + κσ

    Higher κ = more exploration. Default κ=2 is standard.
    """
    return mu + kappa * sigma


def probability_of_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Probability of Improvement: P(f(x) > f_best + ξ)
    """
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - f_best - xi) / sigma
    return norm.cdf(z)


def thompson_sampling(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_samples: int = 1,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Thompson Sampling: sample from posterior, pick highest.
    Returns averaged scores across n_samples draws.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    sigma = np.maximum(sigma, 1e-9)
    total = np.zeros(len(mu))
    for _ in range(n_samples):
        samples = rng.normal(mu, sigma)
        total += samples
    return total / n_samples


# ==============================================================================
# ACQUISITION OPTIMIZER
# ==============================================================================

class AcquisitionOptimizer:
    """
    Selects the most promising candidates to evaluate next.
    """

    METHODS = {"ei", "ucb", "pi", "ts", "random"}

    def __init__(
        self,
        surrogate=None,
        method: str = "ei",
        kappa: float = 2.0,
        xi: float = 0.01,
        random_state: int = 42,
    ):
        self.surrogate = surrogate
        self.method = method
        self.kappa = kappa
        self.xi = xi
        self.rng = np.random.RandomState(random_state)

    def select(
        self,
        X_candidates: np.ndarray,
        n_select: int = 10,
        f_best: Optional[float] = None,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
    ) -> AcquisitionResult:
        """
        Select the top n_select candidates to evaluate.

        Args:
            X_candidates: (N, D) candidate features.
            n_select: How many to select.
            f_best: Current best observed value (for EI/PI).
            mu, sigma: Pre-computed predictions (if None, uses surrogate).
        """
        X = np.atleast_2d(X_candidates)
        n = len(X)
        n_select = min(n_select, n)

        # Get predictions
        if mu is None or sigma is None:
            assert self.surrogate is not None, "Need surrogate or pre-computed mu/sigma"
            mu, sigma = self.surrogate.predict(X, return_std=True)

        if f_best is None:
            f_best = float(np.max(mu))

        # Compute acquisition scores
        scores = self._compute_scores(mu, sigma, f_best)

        # Select top n
        idx = np.argsort(-scores)[:n_select]

        return AcquisitionResult(
            selected_indices=idx.tolist(),
            scores=scores,
            selected_scores=scores[idx],
            method=self.method,
            f_best=f_best,
            n_candidates=n,
            n_selected=n_select,
        )

    def _compute_scores(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        f_best: float,
    ) -> np.ndarray:
        if self.method == "ei":
            return expected_improvement(mu, sigma, f_best, self.xi)
        elif self.method == "ucb":
            return upper_confidence_bound(mu, sigma, self.kappa)
        elif self.method == "pi":
            return probability_of_improvement(mu, sigma, f_best, self.xi)
        elif self.method == "ts":
            return thompson_sampling(mu, sigma, n_samples=10, rng=self.rng)
        elif self.method == "random":
            return self.rng.random(len(mu))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ------------------------------------------------------------------
    # MULTI-OBJECTIVE
    # ------------------------------------------------------------------
    def select_multi_objective(
        self,
        X_candidates: np.ndarray,
        multi_surrogate: Any,  # MultiObjectiveSurrogate
        n_select: int = 10,
        f_best: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> AcquisitionResult:
        """
        Multi-objective acquisition via weighted scalarization of per-objective EI.

        Args:
            multi_surrogate: MultiObjectiveSurrogate with .predict()
            f_best: {objective: best_value}
            weights: {objective: weight} (default: equal)
        """
        X = np.atleast_2d(X_candidates)
        preds = multi_surrogate.predict(X, return_std=True)

        if weights is None:
            weights = {obj: 1.0 for obj in preds}

        total = np.zeros(len(X))
        total_w = 0.0

        for obj, (mu, sigma) in preds.items():
            w = weights.get(obj, 1.0)
            fb = (f_best or {}).get(obj, float(np.max(mu)))
            ei = expected_improvement(mu, sigma, fb, self.xi)
            total += w * ei
            total_w += w

        scores = total / max(total_w, 1e-10)
        idx = np.argsort(-scores)[:n_select]

        return AcquisitionResult(
            selected_indices=idx.tolist(),
            scores=scores,
            selected_scores=scores[idx],
            method=f"multi_ei_{self.method}",
            f_best=max(f_best.values()) if f_best else 0.0,
            n_candidates=len(X),
            n_selected=min(n_select, len(X)),
        )


# ==============================================================================
# ADAPTIVE EXPLORATION SCHEDULER
# ==============================================================================

class ExplorationScheduler:
    """
    Adjusts exploration vs. exploitation over the optimization run.
    Early: explore (high κ). Late: exploit (low κ).
    """

    def __init__(
        self,
        initial_kappa: float = 4.0,
        final_kappa: float = 0.5,
        decay: str = "linear",
        total_iterations: int = 100,
    ):
        self.initial_kappa = initial_kappa
        self.final_kappa = final_kappa
        self.decay = decay
        self.total_iterations = total_iterations

    def get_kappa(self, iteration: int) -> float:
        """Return κ for the current iteration."""
        t = min(iteration / max(self.total_iterations, 1), 1.0)
        if self.decay == "linear":
            return self.initial_kappa + t * (self.final_kappa - self.initial_kappa)
        elif self.decay == "exponential":
            r = self.final_kappa / max(self.initial_kappa, 1e-10)
            return self.initial_kappa * (r ** t)
        elif self.decay == "cosine":
            return self.final_kappa + 0.5 * (self.initial_kappa - self.final_kappa) * (
                1 + np.cos(np.pi * t)
            )
        return self.initial_kappa
