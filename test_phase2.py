# ==============================================================================
# test_phase2.py
# ==============================================================================
# Phase 2 Test Suite -- 36 tests across 6 modules + integration
# Run: python test_phase2.py
# ==============================================================================

import sys
import os
import time
import tempfile
import shutil
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from strategy_fingerprint import StrategyFingerprinter
from surrogate_model import SurrogateModel, MultiObjectiveSurrogate
from acquisition_function import (
    AcquisitionOptimizer, ExplorationScheduler,
    expected_improvement, upper_confidence_bound,
    probability_of_improvement, thompson_sampling,
)
from multi_objective_optimizer import StrategyOptimizer, ObjectiveConfig
from genetic_operators import GeneticEngine, GeneticConfig
from optimization_pipeline import OptimizationPipeline, PipelineConfig


# ==============================================================================
# HARNESS
# ==============================================================================
_passed = 0
_failed = 0
_errors = []

def run_test(name, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  [OK] {name}")
    except Exception as e:
        _failed += 1
        _errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


def _synth_strategies(n=30, seed=42):
    np.random.seed(seed)
    strats = []
    for i in range(n):
        strats.append({
            "strategy_id": f"S_{i:03d}", "name": f"Strategy_{i:03d}",
            "origin": "discovered" if i < n//2 else "mutation",
            "mutation_type": "add_indicator" if i % 3 == 0 else "change_params",
            "sharpe_ratio": round(np.random.uniform(-0.5, 2.5), 3),
            "max_drawdown_pct": round(np.random.uniform(5, 40), 1),
            "total_trades": int(np.random.uniform(20, 300)),
            "total_return_pct": round(np.random.uniform(-20, 80), 1),
            "win_rate": round(np.random.uniform(0.3, 0.7), 3),
            "profit_factor": round(np.random.uniform(0.5, 2.5), 2),
            "generation": i % 5,
            "pbo_probability": round(np.random.uniform(0.1, 0.8), 2),
            "regime_consistency": round(np.random.uniform(0.3, 0.9), 2),
        })
    return strats


def _synth_returns(n=30, T=500, seed=42):
    np.random.seed(seed)
    ids = [f"S_{i:03d}" for i in range(n)]
    ret = np.random.normal(0.0003, 0.012, (T, n))
    for i in range(n):
        ret[:, i] += (i - n/2) * 0.00003
    return {ids[i]: ret[:, i] for i in range(n)}


# ==============================================================================
# MODULE 1 TESTS: StrategyFingerprinter
# ==============================================================================

def test_fp_1_basic():
    fp = StrategyFingerprinter()
    strats = _synth_strategies(10)
    r = fp.transform(strats)
    assert r.X.shape[0] == 10
    assert r.X.shape[1] == fp.n_features
    assert len(r.feature_names) == fp.n_features

def test_fp_2_scaling():
    fp = StrategyFingerprinter(fit_scaler=True)
    strats = _synth_strategies(20)
    r = fp.transform(strats)
    means = np.abs(r.X.mean(axis=0))
    assert np.mean(means) < 1.0, "Scaled data should have small means"

def test_fp_3_single():
    fp = StrategyFingerprinter()
    strats = _synth_strategies(20)
    fp.transform(strats)  # fit scaler
    v = fp.transform_single(strats[0])
    assert v.shape == (fp.n_features,)

def test_fp_4_similarity():
    fp = StrategyFingerprinter(fit_scaler=False)
    s1 = {"sharpe_ratio": 1.5, "max_drawdown_pct": 10, "total_trades": 100,
           "total_return_pct": 40, "win_rate": 0.6, "profit_factor": 1.8}
    s2 = dict(s1)  # identical
    s3 = {"sharpe_ratio": -0.5, "max_drawdown_pct": 45, "total_trades": 10,
           "total_return_pct": -30, "win_rate": 0.3, "profit_factor": 0.5}
    r = fp.transform([s1, s2, s3], fit=False)
    sim_same = fp.compute_similarity(r.X[0], r.X[1])
    sim_diff = fp.compute_similarity(r.X[0], r.X[2])
    assert sim_same > sim_diff

def test_fp_5_importance():
    fp = StrategyFingerprinter(fit_scaler=False)
    strats = _synth_strategies(50)
    r = fp.transform(strats, fit=False)
    y = np.array([s["sharpe_ratio"] for s in strats])
    imp = fp.get_feature_importance(r.X, y)
    assert "perf_sharpe_ratio" in imp
    assert imp["perf_sharpe_ratio"] > 0.5  # Should be highly correlated

def test_fp_6_deterministic():
    fp1 = StrategyFingerprinter(fit_scaler=False)
    fp2 = StrategyFingerprinter(fit_scaler=False)
    strats = _synth_strategies(10)
    r1 = fp1.transform(strats, fit=False)
    r2 = fp2.transform(strats, fit=False)
    assert np.allclose(r1.X, r2.X)


# ==============================================================================
# MODULE 2 TESTS: SurrogateModel
# ==============================================================================

def test_sm_1_rf_fit():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] * 2 + X[:, 1] + np.random.normal(0, 0.1, 100)
    m = SurrogateModel("rf")
    met = m.fit(X, y)
    assert met.r2 > 0.8

def test_sm_2_rf_predict_std():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] + np.random.normal(0, 0.1, 100)
    m = SurrogateModel("rf")
    m.fit(X, y)
    mu, sigma = m.predict(X[:5], return_std=True)
    assert len(mu) == 5
    assert len(sigma) == 5
    assert np.all(sigma >= 0)

def test_sm_3_gp_uncertainty():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X[:, 0] ** 2 + np.random.normal(0, 0.1, 50)
    m = SurrogateModel("gp")
    m.fit(X, y, cv_folds=3)
    # Far-away point should have high uncertainty
    far = np.array([[10.0, 10.0, 10.0]])
    close = X[:1]
    _, sig_far = m.predict(far, return_std=True)
    _, sig_close = m.predict(close, return_std=True)
    assert sig_far[0] > sig_close[0]

def test_sm_4_gb_fit():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 3 + np.random.normal(0, 0.2, 100)
    m = SurrogateModel("gb")
    met = m.fit(X, y)
    assert met.r2 > 0.7

def test_sm_5_update():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = X[:, 0] + np.random.normal(0, 0.1, 50)
    m = SurrogateModel("rf")
    m.fit(X, y)
    r2_before = m.metrics.r2
    X_new = np.random.randn(10, 5)
    y_new = X_new[:, 0] + np.random.normal(0, 0.1, 10)
    m.update(X_new, y_new)
    assert m.metrics.n_train == 60

def test_sm_6_multi_objective():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    Y = {
        "sharpe": X[:, 0] + np.random.normal(0, 0.1, 100),
        "drawdown": -X[:, 1] + np.random.normal(0, 0.1, 100),
    }
    ms = MultiObjectiveSurrogate(["sharpe", "drawdown"], model_type="rf")
    metrics = ms.fit(X, Y)
    assert "sharpe" in metrics
    preds = ms.predict(X[:5], return_std=True)
    assert "sharpe" in preds


# ==============================================================================
# MODULE 3 TESTS: AcquisitionFunction
# ==============================================================================

def test_acq_1_ei():
    mu = np.array([1.0, 1.5, 0.5, 2.0])
    sigma = np.array([0.1, 0.3, 0.5, 0.1])
    ei = expected_improvement(mu, sigma, f_best=1.5)
    assert ei[3] > ei[0]  # μ=2.0 should have highest EI

def test_acq_2_ucb():
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.1, 0.5, 1.0])
    ucb = upper_confidence_bound(mu, sigma, kappa=2.0)
    assert ucb[2] > ucb[1] > ucb[0]

def test_acq_3_pi():
    mu = np.array([0.5, 1.0, 1.5, 2.0])
    sigma = np.array([0.1] * 4)
    pi = probability_of_improvement(mu, sigma, f_best=1.2)
    assert pi[3] > pi[2] > pi[1]

def test_acq_4_optimizer():
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = X[:, 0] + np.random.normal(0, 0.1, 50)
    m = SurrogateModel("rf")
    m.fit(X, y)
    acq = AcquisitionOptimizer(surrogate=m, method="ei")
    result = acq.select(X, n_select=5, f_best=1.0)
    assert len(result.selected_indices) == 5
    assert result.method == "ei"

def test_acq_5_scheduler():
    sched = ExplorationScheduler(initial_kappa=4.0, final_kappa=0.5, total_iterations=100)
    k0 = sched.get_kappa(0)
    k50 = sched.get_kappa(50)
    k100 = sched.get_kappa(100)
    assert k0 > k50 > k100
    assert abs(k0 - 4.0) < 0.01
    assert abs(k100 - 0.5) < 0.01

def test_acq_6_thompson():
    mu = np.array([1.0, 1.0, 1.0])
    sigma = np.array([0.01, 0.5, 2.0])
    scores = thompson_sampling(mu, sigma, n_samples=100)
    assert len(scores) == 3


# ==============================================================================
# MODULE 4 TESTS: MultiObjectiveOptimizer
# ==============================================================================

def test_mo_1_pareto_sort():
    strats = [
        {"name": "A", "sharpe_ratio": 2.0, "max_drawdown_pct": 30},
        {"name": "B", "sharpe_ratio": 1.5, "max_drawdown_pct": 10},
        {"name": "C", "sharpe_ratio": 1.0, "max_drawdown_pct": 5},
        {"name": "D", "sharpe_ratio": 0.5, "max_drawdown_pct": 35},  # Dominated
    ]
    opt = StrategyOptimizer(objectives=[
        ObjectiveConfig("sharpe_ratio", "maximize"),
        ObjectiveConfig("max_drawdown_pct", "minimize"),
    ])
    result = opt.optimize(strats, n_generations=1)
    names = {s["name"] for s in result.pareto_front}
    assert "D" not in names  # D is dominated by A (worse SR, worse DD)
    assert "A" in names or "B" in names or "C" in names

def test_mo_2_with_fingerprints():
    strats = _synth_strategies(20)
    fp = StrategyFingerprinter(fit_scaler=False)
    r = fp.transform(strats, fit=False)
    opt = StrategyOptimizer(objectives=[
        ObjectiveConfig("sharpe_ratio", "maximize"),
        ObjectiveConfig("max_drawdown_pct", "minimize"),
    ])
    result = opt.optimize(strats, r.X, n_generations=10)
    assert len(result.pareto_front) > 0

def test_mo_3_crowding():
    F = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
    cd = StrategyOptimizer.compute_crowding_distance(F)
    assert cd[0] == np.inf  # Boundary points
    assert cd[-1] == np.inf
    assert cd[2] > 0  # Interior point has finite distance

def test_mo_4_knee_point():
    F = np.array([[0, 5], [1, 3], [2, 2.5], [3, 2.4], [5, 2.3]])
    knee = StrategyOptimizer.knee_point(F)
    assert knee in [1, 2]  # Should be near the bend

def test_mo_5_empty():
    opt = StrategyOptimizer()
    result = opt.optimize([])
    assert len(result.pareto_front) == 0

def test_mo_6_single():
    opt = StrategyOptimizer()
    result = opt.optimize([{"name": "solo", "sharpe_ratio": 1.5, "max_drawdown_pct": 10}])
    assert len(result.pareto_front) == 1


# ==============================================================================
# MODULE 5 TESTS: GeneticOperators
# ==============================================================================

def test_ga_1_tournament():
    np.random.seed(42)
    engine = GeneticEngine()
    pop = np.random.randn(20, 5)
    fitness = np.arange(20, dtype=float)
    selected = engine.select_tournament(pop, fitness, 10)
    assert selected.shape == (10, 5)

def test_ga_2_crossover_sbx():
    engine = GeneticEngine()
    parents = np.random.randn(10, 5)
    children = engine.crossover_sbx(parents)
    assert children.shape[0] == 10
    assert children.shape[1] == 5

def test_ga_3_mutate():
    engine = GeneticEngine(GeneticConfig(mutation_prob=1.0, mutation_scale=0.5))
    pop = np.ones((10, 5))
    mutated = engine.mutate_gaussian(pop)
    assert not np.allclose(pop, mutated)

def test_ga_4_survive():
    engine = GeneticEngine(GeneticConfig(pop_size=10, elite_frac=0.2))
    pop = np.random.randn(30, 5)
    fitness = np.arange(30, dtype=float)
    survivors, surv_fit = engine.survive(pop, fitness, 10)
    assert len(survivors) == 10
    assert 29.0 in surv_fit  # Best should survive

def test_ga_5_diversity():
    engine = GeneticEngine()
    pop = np.random.randn(20, 5)
    d = engine.compute_diversity(pop)
    assert d > 0
    identical = np.ones((20, 5))
    d2 = engine.compute_diversity(identical)
    assert d2 < d

def test_ga_6_evolve_generation():
    engine = GeneticEngine(GeneticConfig(pop_size=20))
    pop = np.random.randn(20, 5)
    fitness = np.random.randn(20)
    children, stats = engine.evolve_generation(pop, fitness, generation=0)
    assert children.shape[1] == 5
    assert stats.generation == 0
    assert stats.best_fitness == float(np.max(fitness))


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_int_1_full_pipeline():
    d = tempfile.mkdtemp(prefix="phase2_test_")
    try:
        strats = _synth_strategies(30)
        rets = _synth_returns(30)

        pipe = OptimizationPipeline(
            config=PipelineConfig(
                pop_size=20, n_generations=5, backtest_budget=50,
                n_acquire_per_gen=5, surrogate_retrain_every=3,
                checkpoint_dir=f"{d}/opt", verbose=False,
            )
        )
        result = pipe.run(strats, rets)
        assert result.total_generations > 0
        assert result.total_backtests <= 50
        assert result.total_surrogate_evals > 0
    finally:
        shutil.rmtree(d)

def test_int_2_pipeline_no_returns():
    d = tempfile.mkdtemp(prefix="phase2_test_")
    try:
        strats = _synth_strategies(20)
        pipe = OptimizationPipeline(
            config=PipelineConfig(
                pop_size=15, n_generations=3, backtest_budget=30,
                n_acquire_per_gen=3, checkpoint_dir=f"{d}/opt",
                verbose=False,
            )
        )
        result = pipe.run(strats)
        assert result.total_generations > 0
    finally:
        shutil.rmtree(d)

def test_int_3_pareto_output():
    strats = _synth_strategies(30)
    pipe = OptimizationPipeline(
        config=PipelineConfig(
            pop_size=20, n_generations=3, backtest_budget=20,
            n_acquire_per_gen=3, checkpoint_dir="", verbose=False,
        )
    )
    result = pipe.run(strats)
    assert result.pareto_front is not None

def test_int_4_checkpoint():
    d = tempfile.mkdtemp(prefix="phase2_test_")
    try:
        strats = _synth_strategies(20)
        pipe = OptimizationPipeline(
            config=PipelineConfig(
                pop_size=15, n_generations=6, backtest_budget=30,
                checkpoint_dir=f"{d}/opt", checkpoint_every=3,
                verbose=False,
            )
        )
        pipe.run(strats)
        opt_dir = Path(d) / "opt"
        files = list(opt_dir.glob("*.npz")) + list(opt_dir.glob("*.json"))
        assert len(files) > 0
    finally:
        shutil.rmtree(d)


# ==============================================================================
# MAIN
# ==============================================================================

ALL_TESTS = [
    # Module 1: Fingerprinter
    ("FP.1 Basic transform",         test_fp_1_basic),
    ("FP.2 Scaling",                 test_fp_2_scaling),
    ("FP.3 Single transform",        test_fp_3_single),
    ("FP.4 Similarity",             test_fp_4_similarity),
    ("FP.5 Feature importance",      test_fp_5_importance),
    ("FP.6 Deterministic",          test_fp_6_deterministic),
    # Module 2: Surrogate
    ("SM.1 RF fit",                  test_sm_1_rf_fit),
    ("SM.2 RF predict w/ std",       test_sm_2_rf_predict_std),
    ("SM.3 GP uncertainty",          test_sm_3_gp_uncertainty),
    ("SM.4 GB fit",                  test_sm_4_gb_fit),
    ("SM.5 Incremental update",      test_sm_5_update),
    ("SM.6 Multi-objective",         test_sm_6_multi_objective),
    # Module 3: Acquisition
    ("ACQ.1 EI",                     test_acq_1_ei),
    ("ACQ.2 UCB",                    test_acq_2_ucb),
    ("ACQ.3 PI",                     test_acq_3_pi),
    ("ACQ.4 Optimizer",              test_acq_4_optimizer),
    ("ACQ.5 Scheduler",              test_acq_5_scheduler),
    ("ACQ.6 Thompson sampling",      test_acq_6_thompson),
    # Module 4: Multi-Objective
    ("MO.1 Pareto sort",            test_mo_1_pareto_sort),
    ("MO.2 With fingerprints",       test_mo_2_with_fingerprints),
    ("MO.3 Crowding distance",       test_mo_3_crowding),
    ("MO.4 Knee point",             test_mo_4_knee_point),
    ("MO.5 Empty input",            test_mo_5_empty),
    ("MO.6 Single strategy",         test_mo_6_single),
    # Module 5: Genetic Operators
    ("GA.1 Tournament select",       test_ga_1_tournament),
    ("GA.2 SBX crossover",          test_ga_2_crossover_sbx),
    ("GA.3 Gaussian mutation",       test_ga_3_mutate),
    ("GA.4 Elitist survival",        test_ga_4_survive),
    ("GA.5 Diversity metric",        test_ga_5_diversity),
    ("GA.6 Evolve generation",       test_ga_6_evolve_generation),
    # Integration
    ("INT.1 Full pipeline",          test_int_1_full_pipeline),
    ("INT.2 No returns mode",        test_int_2_pipeline_no_returns),
    ("INT.3 Pareto output",          test_int_3_pareto_output),
    ("INT.4 Checkpointing",          test_int_4_checkpoint),
]

if __name__ == "__main__":
    start = time.time()
    modules = [
        ("Module 1: Strategy Fingerprint", 0, 6),
        ("Module 2: Surrogate Model", 6, 12),
        ("Module 3: Acquisition Function", 12, 18),
        ("Module 4: Multi-Objective Optimizer", 18, 24),
        ("Module 5: Genetic Operators", 24, 30),
        ("Module 6: Integration Pipeline", 30, 34),
    ]
    for mod_name, lo, hi in modules:
        print(f"\n{'-'*60}")
        print(f"  {mod_name}")
        print(f"{'-'*60}")
        for name, fn in ALL_TESTS[lo:hi]:
            run_test(name, fn)

    elapsed = time.time() - start
    print(f"\n  ⏱️  Completed in {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print(f"  PHASE 2 TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total:   {_passed + _failed}")
    print(f"  Passed:  {_passed} [OK]")
    print(f"  Failed:  {_failed} [FAIL]")
    if _errors:
        print(f"\n  Failures:")
        for n, e in _errors:
            print(f"    {n}: {e}")
    print(f"{'='*60}")
    sys.exit(0 if _failed == 0 else 1)
