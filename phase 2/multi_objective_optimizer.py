# ==============================================================================
# multi_objective_optimizer.py
# ==============================================================================
# Phase 2, Module 4 (Week 8): NSGA-II Multi-Objective Optimizer
#
# Uses pymoo's NSGA-II implementation to find the Pareto frontier of
# strategies that are simultaneously optimal across multiple objectives
# (Sharpe, drawdown, profit factor, trade count, regime consistency).
#
# Why multi-objective? A strategy with Sharpe=2.0 and DD=40% is not
# strictly better than one with Sharpe=1.5 and DD=15%. NSGA-II finds
# the full Pareto frontier so you can pick the best tradeoff.
#
# GitHub repos:
#   - pymoo (https://github.com/anyoptimization/pymoo)
#     NSGA-II, non-dominated sorting, crowding distance
#
# Consumed by:
#   - optimization_pipeline.py (drives the optimization loop)
#   - genetic_operators.py (produces children for NSGA-II population)
#
# Usage:
#     from multi_objective_optimizer import StrategyOptimizer, ObjectiveConfig
#
#     opt = StrategyOptimizer(objectives=[
#         ObjectiveConfig("sharpe_ratio", "maximize"),
#         ObjectiveConfig("max_drawdown_pct", "minimize"),
#     ])
#     result = opt.optimize(strategies, n_generations=50)
#     pareto = result.pareto_front
#
# ==============================================================================

import numpy as np
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize as pymoo_minimize
import warnings


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ObjectiveConfig:
    """One optimization objective."""
    name: str
    direction: str = "maximize"   # "maximize" or "minimize"
    weight: float = 1.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    @property
    def sign(self) -> float:
        """pymoo minimizes. Return -1 for maximize, +1 for minimize."""
        return -1.0 if self.direction == "maximize" else 1.0


DEFAULT_OBJECTIVES = [
    ObjectiveConfig("sharpe_ratio", "maximize", 1.0),
    ObjectiveConfig("max_drawdown_pct", "minimize", 1.0),
    ObjectiveConfig("profit_factor", "maximize", 0.8),
    ObjectiveConfig("total_trades", "maximize", 0.3),
    ObjectiveConfig("regime_consistency", "maximize", 0.5),
]


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class OptimizationResult:
    """Output of NSGA-II optimization."""
    pareto_front: List[Dict[str, Any]]   # Non-dominated strategies
    pareto_X: np.ndarray                  # Decision variables for Pareto front
    pareto_F: np.ndarray                  # Objective values for Pareto front
    all_strategies: List[Dict[str, Any]]
    n_generations: int
    n_evaluations: int
    hypervolume: Optional[float] = None
    generation_history: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  NSGA-II OPTIMIZATION RESULTS",
            f"{'='*60}",
            f"  Generations:     {self.n_generations}",
            f"  Total evals:     {self.n_evaluations}",
            f"  Pareto front:    {len(self.pareto_front)} strategies",
        ]
        if self.hypervolume is not None:
            lines.append(f"  Hypervolume:     {self.hypervolume:.4f}")
        if self.pareto_front:
            lines.append(f"\n  Top 5 Pareto strategies:")
            for i, s in enumerate(self.pareto_front[:5]):
                sr = s.get("sharpe_ratio", 0)
                dd = s.get("max_drawdown_pct", 0)
                lines.append(f"    {i+1}. {s.get('name','?')} "
                             f"(Sharpe={sr:.3f}, DD={dd:.1f}%)")
        return "\n".join(lines)


# ==============================================================================
# PYMOO PROBLEM WRAPPER
# ==============================================================================

class StrategyProblem(Problem):
    """
    pymoo Problem that evaluates strategies via surrogate or lookup table.
    """

    def __init__(
        self,
        n_var: int,
        objectives: List[ObjectiveConfig],
        evaluate_fn: Callable,
        xl: Optional[np.ndarray] = None,
        xu: Optional[np.ndarray] = None,
    ):
        self.objectives_cfg = objectives
        self.evaluate_fn = evaluate_fn
        self._eval_count = 0

        n_obj = len(objectives)
        if xl is None:
            xl = np.zeros(n_var)
        if xu is None:
            xu = np.ones(n_var)

        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=0,
            xl=xl, xu=xu,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), self.n_obj))
        for i, x in enumerate(X):
            obj_vals = self.evaluate_fn(x)
            for j, cfg in enumerate(self.objectives_cfg):
                val = obj_vals.get(cfg.name, 0.0)
                F[i, j] = cfg.sign * val  # pymoo minimizes, flip for maximize
            self._eval_count += 1
        out["F"] = F


# ==============================================================================
# LOOKUP-BASED PROBLEM (for discrete strategy pools)
# ==============================================================================

class StrategyPoolProblem(Problem):
    """
    Problem over a discrete pool of strategies (no continuous optimization).
    Each "variable" is an index into the pool.
    Used for subset selection from existing strategies.
    """

    def __init__(
        self,
        strategies: List[Dict],
        objectives: List[ObjectiveConfig],
        fingerprints: Optional[np.ndarray] = None,
        surrogate: Optional[Any] = None,
    ):
        self.strategies = strategies
        self.objectives_cfg = objectives
        self.fingerprints = fingerprints
        self.surrogate = surrogate
        self._eval_count = 0
        n = len(strategies)

        n_var = fingerprints.shape[1] if fingerprints is not None else 1
        super().__init__(
            n_var=n_var, n_obj=len(objectives), n_constr=0,
            xl=np.zeros(n_var), xu=np.ones(n_var),
        )
        self._precompute_objectives()

    def _precompute_objectives(self):
        """Pre-extract objective values from strategies."""
        self._obj_matrix = np.zeros((len(self.strategies), len(self.objectives_cfg)))
        for i, s in enumerate(self.strategies):
            for j, cfg in enumerate(self.objectives_cfg):
                val = float(s.get(cfg.name, 0) or 0)
                self._obj_matrix[i, j] = cfg.sign * val

    def _evaluate(self, X, out, *args, **kwargs):
        # Map continuous X to nearest strategy via fingerprint distance
        if self.fingerprints is not None:
            F = np.zeros((len(X), self.n_obj))
            for i, x in enumerate(X):
                dists = np.linalg.norm(self.fingerprints - x, axis=1)
                nearest = np.argmin(dists)
                F[i] = self._obj_matrix[nearest]
                self._eval_count += 1
            out["F"] = F
        else:
            out["F"] = self._obj_matrix[:len(X)]


# ==============================================================================
# STRATEGY OPTIMIZER
# ==============================================================================

class StrategyOptimizer:
    """
    NSGA-II optimizer for trading strategies.
    """

    def __init__(
        self,
        objectives: Optional[List[ObjectiveConfig]] = None,
        pop_size: int = 100,
        random_state: int = 42,
    ):
        self.objectives = objectives or DEFAULT_OBJECTIVES
        self.pop_size = pop_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # MAIN: optimize over a pool of strategies
    # ------------------------------------------------------------------
    def optimize(
        self,
        strategies: List[Dict[str, Any]],
        fingerprints: Optional[np.ndarray] = None,
        n_generations: int = 50,
        surrogate: Optional[Any] = None,
    ) -> OptimizationResult:
        """
        Run NSGA-II over a pool of strategies.

        Args:
            strategies: List of strategy dicts with objective metric keys.
            fingerprints: (N, D) feature matrix from StrategyFingerprinter.
            n_generations: Number of NSGA-II generations.
            surrogate: Optional SurrogateModel for fitness evaluation.
        """
        n = len(strategies)
        if n == 0:
            return OptimizationResult(
                pareto_front=[], pareto_X=np.array([]),
                pareto_F=np.array([]), all_strategies=strategies,
                n_generations=0, n_evaluations=0,
            )

        # Build problem
        if fingerprints is not None:
            problem = StrategyPoolProblem(
                strategies, self.objectives, fingerprints, surrogate,
            )
            n_var = fingerprints.shape[1]
        else:
            # Direct evaluation mode — build objective matrix
            return self._optimize_direct(strategies, n_generations)

        # Configure NSGA-II
        algorithm = NSGA2(
            pop_size=min(self.pop_size, n * 2),
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", n_generations)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pymoo_minimize(
                problem, algorithm, termination,
                seed=self.random_state, verbose=False,
            )

        # Map results back to strategies
        pareto_strats = self._map_to_strategies(
            res.X, strategies, fingerprints
        )
        pareto_F = res.F if res.F is not None else np.array([])

        return OptimizationResult(
            pareto_front=pareto_strats,
            pareto_X=res.X if res.X is not None else np.array([]),
            pareto_F=pareto_F,
            all_strategies=strategies,
            n_generations=n_generations,
            n_evaluations=problem._eval_count,
        )

    def _optimize_direct(
        self,
        strategies: List[Dict],
        n_generations: int,
    ) -> OptimizationResult:
        """
        Direct non-dominated sorting on the existing pool.
        No continuous optimization needed — just extract the Pareto front.
        """
        F = np.zeros((len(strategies), len(self.objectives)))
        for i, s in enumerate(strategies):
            for j, cfg in enumerate(self.objectives):
                val = float(s.get(cfg.name, 0) or 0)
                F[i, j] = cfg.sign * val

        # Non-dominated sorting
        pareto_mask = self._fast_non_dominated_sort(F)
        pareto_strats = [strategies[i] for i in range(len(strategies)) if pareto_mask[i]]
        pareto_F_vals = F[pareto_mask]

        return OptimizationResult(
            pareto_front=pareto_strats,
            pareto_X=np.array([]),
            pareto_F=pareto_F_vals,
            all_strategies=strategies,
            n_generations=1,
            n_evaluations=len(strategies),
        )

    # ------------------------------------------------------------------
    # NON-DOMINATED SORTING
    # ------------------------------------------------------------------
    @staticmethod
    def _fast_non_dominated_sort(F: np.ndarray) -> np.ndarray:
        """Return boolean mask of Pareto-optimal (rank 0) solutions."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # j dominates i if j <= i on all objectives and j < i on at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return ~is_dominated

    # ------------------------------------------------------------------
    # MAPPING
    # ------------------------------------------------------------------
    def _map_to_strategies(
        self,
        X: np.ndarray,
        strategies: List[Dict],
        fingerprints: np.ndarray,
    ) -> List[Dict]:
        """Map solution vectors back to nearest strategies."""
        if X is None or len(X) == 0:
            return []
        X = np.atleast_2d(X)
        seen = set()
        result = []
        for x in X:
            dists = np.linalg.norm(fingerprints - x, axis=1)
            nearest = int(np.argmin(dists))
            if nearest not in seen:
                seen.add(nearest)
                result.append(strategies[nearest])
        return result

    # ------------------------------------------------------------------
    # PARETO UTILITIES
    # ------------------------------------------------------------------
    @staticmethod
    def compute_crowding_distance(F: np.ndarray) -> np.ndarray:
        """Compute crowding distance for a Pareto front."""
        n, m = F.shape
        dist = np.zeros(n)
        for j in range(m):
            idx = np.argsort(F[:, j])
            dist[idx[0]] = dist[idx[-1]] = np.inf
            f_range = F[idx[-1], j] - F[idx[0], j]
            if f_range < 1e-10:
                continue
            for k in range(1, n - 1):
                dist[idx[k]] += (F[idx[k+1], j] - F[idx[k-1], j]) / f_range
        return dist

    @staticmethod
    def knee_point(F: np.ndarray) -> int:
        """Find the 'knee' of a 2D Pareto front (max distance from line)."""
        if len(F) < 2:
            return 0
        p1 = F[0]
        p2 = F[-1]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            return 0
        dists = np.abs(np.cross(line_vec, p1 - F)) / line_len
        return int(np.argmax(dists))
