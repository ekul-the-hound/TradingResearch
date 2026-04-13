# ==============================================================================
# optimization_pipeline.py
# ==============================================================================
# Phase 2, Module 6 (Week 10): Full Surrogate-GA Optimization Pipeline
#
# End-to-end orchestrator: fingerprint -> surrogate -> acquisition ->
# backtest -> update -> NSGA-II -> evolve -> repeat -> checkpoint.
#
# Usage:
#     from optimization_pipeline import OptimizationPipeline, PipelineConfig
#     pipeline = OptimizationPipeline(config=PipelineConfig(n_generations=50))
#     result = pipeline.run(initial_strategies, returns_dict)
#
# ==============================================================================

import numpy as np
import json
import time
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from strategy_fingerprint import StrategyFingerprinter, FingerprintResult
from surrogate_model import SurrogateModel
from acquisition_function import AcquisitionOptimizer, ExplorationScheduler
from multi_objective_optimizer import StrategyOptimizer, ObjectiveConfig
from genetic_operators import GeneticEngine, GeneticConfig


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PipelineConfig:
    pop_size: int = 100
    n_generations: int = 50
    backtest_budget: int = 500
    surrogate_type: str = "rf"
    surrogate_retrain_every: int = 10
    acquisition_method: str = "ei"
    n_acquire_per_gen: int = 10
    initial_kappa: float = 4.0
    final_kappa: float = 0.5
    objectives: List[ObjectiveConfig] = field(default_factory=lambda: [
        ObjectiveConfig("sharpe_ratio", "maximize", 1.0),
        ObjectiveConfig("max_drawdown_pct", "minimize", 1.0),
        ObjectiveConfig("profit_factor", "maximize", 0.8),
    ])
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    elite_frac: float = 0.1
    checkpoint_dir: str = "data/optimization"
    checkpoint_every: int = 5
    random_seed: int = 42
    verbose: bool = True


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class PipelineResult:
    pareto_front: List[Dict[str, Any]]
    best_strategy: Optional[Dict[str, Any]]
    total_generations: int
    total_backtests: int
    total_surrogate_evals: int
    surrogate_r2: float
    final_diversity: float
    generation_history: List[Dict]
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  OPTIMIZATION PIPELINE -- COMPLETE",
            f"{'='*70}",
            f"  Generations:       {self.total_generations}",
            f"  Backtests used:    {self.total_backtests}",
            f"  Surrogate evals:   {self.total_surrogate_evals}",
            f"  Surrogate R²:      {self.surrogate_r2:.4f}",
            f"  Pareto front:      {len(self.pareto_front)} strategies",
            f"  Final diversity:   {self.final_diversity:.4f}",
            f"  Elapsed:           {self.elapsed_seconds:.1f}s",
        ]
        if self.best_strategy:
            sr = self.best_strategy.get("sharpe_ratio", 0)
            dd = self.best_strategy.get("max_drawdown_pct", 0)
            lines.append(f"\n  Best: {self.best_strategy.get('name','?')} "
                         f"(Sharpe={sr:.3f}, DD={dd:.1f}%)")
        return "\n".join(lines)


# ==============================================================================
# PIPELINE
# ==============================================================================

class OptimizationPipeline:
    """End-to-end surrogate-assisted evolutionary optimization."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        backtest_fn: Optional[Callable] = None,
    ):
        self.config = config or PipelineConfig()
        self.backtest_fn = backtest_fn
        cfg = self.config

        self.fingerprinter = StrategyFingerprinter(fit_scaler=True)
        self.surrogate = SurrogateModel(model_type=cfg.surrogate_type)
        self.acquisition = AcquisitionOptimizer(
            surrogate=self.surrogate, method=cfg.acquisition_method,
        )
        self.exploration = ExplorationScheduler(
            initial_kappa=cfg.initial_kappa, final_kappa=cfg.final_kappa,
            total_iterations=cfg.n_generations,
        )
        self.genetic = GeneticEngine(config=GeneticConfig(
            pop_size=cfg.pop_size, crossover_prob=cfg.crossover_prob,
            mutation_prob=cfg.mutation_prob, elite_frac=cfg.elite_frac,
            random_seed=cfg.random_seed,
        ))
        self.optimizer = StrategyOptimizer(
            objectives=cfg.objectives, pop_size=cfg.pop_size,
        )

        self._backtests_used = 0
        self._surrogate_evals = 0
        self._gen_history: List[Dict] = []

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def run(
        self,
        strategies: List[Dict[str, Any]],
        returns_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> PipelineResult:
        start = time.time()
        cfg = self.config
        _log = print if cfg.verbose else lambda *a, **k: None

        _log(f"\n{'='*70}")
        _log(f"  OPTIMIZATION PIPELINE -- STARTING")
        _log(f"  Pop={cfg.pop_size}, Gen={cfg.n_generations}, Budget={cfg.backtest_budget}")
        _log(f"{'='*70}")

        # Step 1: Fingerprint
        fp = self.fingerprinter.transform(strategies, fit=True)
        population = fp.X.copy()
        pop_strategies = list(strategies)

        # Step 2: Build training data from existing metrics
        X_train, y_train = self._build_training_data(strategies, fp)
        best_sharpe = float(np.max(y_train)) if len(y_train) > 0 else 0.0

        if len(X_train) >= 5:
            m = self.surrogate.fit(X_train, y_train, fp.feature_names)
            _log(f"  [STATS] Initial surrogate: R²={m.r2:.4f} (CV={m.cv_mean:.4f})")

        # Step 3: Evolution loop
        gen = 0
        for gen in range(cfg.n_generations):
            if self._backtests_used >= cfg.backtest_budget:
                _log(f"\n  🛑 Budget exhausted ({self._backtests_used})")
                break

            # Predict with surrogate
            if self.surrogate.is_fitted:
                mu, sigma = self.surrogate.predict(population, return_std=True)
                self._surrogate_evals += len(population)
            else:
                mu = np.array([s.get("sharpe_ratio", 0.0) or 0.0
                               for s in pop_strategies[:len(population)]])
                sigma = np.ones_like(mu) * 0.5

            # Acquisition
            kappa = self.exploration.get_kappa(gen)
            self.acquisition.kappa = kappa
            n_acq = min(cfg.n_acquire_per_gen,
                        cfg.backtest_budget - self._backtests_used,
                        len(population))

            acq = self.acquisition.select(
                population, n_select=max(1, n_acq), f_best=best_sharpe,
                mu=mu, sigma=sigma,
            )

            # Backtest selected
            for idx in acq.selected_indices:
                if self._backtests_used >= cfg.backtest_budget:
                    break
                strat = pop_strategies[idx] if idx < len(pop_strategies) else {}
                result = self._run_backtest(strat, returns_dict)
                if result is not None:
                    sr = result.get("sharpe_ratio", 0.0)
                    X_train = np.vstack([X_train, population[idx:idx+1]])
                    y_train = np.append(y_train, sr)
                    if sr > best_sharpe:
                        best_sharpe = sr
                    self._backtests_used += 1

            # Retrain surrogate
            if (gen + 1) % cfg.surrogate_retrain_every == 0 and len(X_train) >= 5:
                self.surrogate.fit(X_train, y_train, fp.feature_names)

            # Fitness for evolution
            fitness = self.surrogate.predict(population) if self.surrogate.is_fitted else mu

            # Evolve
            children, stats = self.genetic.evolve_generation(population, fitness, gen)

            # Combine + survive
            combined = np.vstack([population, children])
            combined_f = np.concatenate([fitness, np.zeros(len(children))])
            if self.surrogate.is_fitted and len(children) > 0:
                combined_f[len(population):] = self.surrogate.predict(children)
                self._surrogate_evals += len(children)

            population, pop_fitness = self.genetic.survive(
                combined, combined_f, cfg.pop_size,
            )

            # Extend strategy list
            while len(pop_strategies) < len(combined):
                pop_strategies.append({
                    "name": f"gen{gen}_child_{len(pop_strategies)}",
                    "strategy_id": f"gen{gen}_{len(pop_strategies)}",
                    "sharpe_ratio": 0.0, "origin": "genetic",
                })
            top_idx = np.argsort(-combined_f)[:cfg.pop_size]
            pop_strategies = [pop_strategies[i] if i < len(pop_strategies)
                              else {} for i in top_idx]

            self._gen_history.append({
                "generation": gen,
                "best": float(np.max(pop_fitness)),
                "mean": float(np.mean(pop_fitness)),
                "diversity": stats.diversity,
                "backtests": self._backtests_used,
                "kappa": kappa,
            })

            if cfg.verbose and gen % 5 == 0:
                _log(f"  Gen {gen:3d} | best={np.max(pop_fitness):.3f} "
                     f"mean={np.mean(pop_fitness):.3f} "
                     f"bt={self._backtests_used}/{cfg.backtest_budget} "
                     f"κ={kappa:.2f}")

            if cfg.checkpoint_dir and (gen + 1) % cfg.checkpoint_every == 0:
                self._checkpoint(gen, population, X_train, y_train)

        # Step 4: Pareto selection
        final_strats = []
        for i, s in enumerate(pop_strategies[:len(population)]):
            d = dict(s)
            if self.surrogate.is_fitted:
                d["predicted_sharpe"] = float(self.surrogate.predict(population[i:i+1])[0])
            final_strats.append(d)

        pareto = self.optimizer.optimize(final_strats, population, n_generations=10)

        # Result
        best = max(final_strats,
                   key=lambda s: s.get("sharpe_ratio", s.get("predicted_sharpe", 0)),
                   default=None)

        result = PipelineResult(
            pareto_front=pareto.pareto_front,
            best_strategy=best,
            total_generations=gen + 1,
            total_backtests=self._backtests_used,
            total_surrogate_evals=self._surrogate_evals,
            surrogate_r2=self.surrogate.metrics.r2 if self.surrogate.metrics else 0.0,
            final_diversity=self.genetic.compute_diversity(population),
            generation_history=self._gen_history,
            elapsed_seconds=time.time() - start,
        )

        if cfg.verbose:
            _log(result.summary())

        if cfg.checkpoint_dir:
            self._save_final(result)

        return result

    # ------------------------------------------------------------------
    def _run_backtest(self, strategy, returns_dict):
        if self.backtest_fn:
            return self.backtest_fn(strategy)
        sid = strategy.get("strategy_id", strategy.get("name", ""))
        sharpe = strategy.get("sharpe_ratio", 0.0) or 0.0
        if returns_dict and sid in returns_dict:
            r = returns_dict[sid]
            if len(r) > 10:
                sr = float(np.mean(r) / max(np.std(r, ddof=1), 1e-10) * np.sqrt(252))
                return {**strategy, "sharpe_ratio": sr}
        return {**strategy, "sharpe_ratio": sharpe + np.random.normal(0, 0.1)}

    def _build_training_data(self, strategies, fp):
        X, y = [], []
        for i, s in enumerate(strategies):
            sr = s.get("sharpe_ratio")
            if sr is not None:
                X.append(fp.X[i])
                y.append(float(sr))
        if not X:
            return np.empty((0, fp.X.shape[1])), np.empty(0)
        return np.array(X), np.array(y)

    def _checkpoint(self, gen, pop, X, y):
        d = Path(self.config.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"checkpoint_gen{gen}.npz", population=pop, X_train=X, y_train=y)

    def _save_final(self, result):
        d = Path(self.config.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_generations": result.total_generations,
            "total_backtests": result.total_backtests,
            "surrogate_r2": result.surrogate_r2,
            "pareto_size": len(result.pareto_front),
            "best": {
                "name": result.best_strategy.get("name") if result.best_strategy else None,
                "sharpe": result.best_strategy.get("sharpe_ratio") if result.best_strategy else None,
            },
            "generation_history": result.generation_history,
        }
        with open(d / "final_results.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  [OK] Final results -> {d / 'final_results.json'}")
