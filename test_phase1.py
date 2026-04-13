# ==============================================================================
# test_phase1.py
# ==============================================================================
# Phase 1 Test Suite -- 32 tests across all 4 modules + integration
#
# All tests are fully offline. No external services needed.
# Each test uses an isolated temp directory to prevent cross-contamination.
#
# Run:   python test_phase1.py
# Quick: python test_phase1.py --quick    (8 critical tests only)
#
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

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from lineage_tracker import LineageTracker
from overfitting_detector import OverfittingDetector
from filtering_pipeline import FilteringPipeline, FilterConfig
from diversification_filter import DiversificationFilter, DiversityConfig
from phase1_pipeline import Phase1Pipeline


# ==============================================================================
# TEST HARNESS
# ==============================================================================

_passed = 0
_failed = 0
_errors = []

def run_test(name: str, fn, quick_list: set = None, quick_mode: bool = False):
    global _passed, _failed
    if quick_mode and quick_list and name not in quick_list:
        return
    try:
        fn()
        _passed += 1
        print(f"  [OK] {name}")
    except Exception as e:
        _failed += 1
        _errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


def _make_tmp():
    d = tempfile.mkdtemp(prefix="phase1_test_")
    return d


def _synth_returns(n_strategies=10, n_days=500, seed=42):
    """Generate synthetic daily returns for N strategies."""
    np.random.seed(seed)
    ret = np.random.normal(0.0003, 0.012, (n_days, n_strategies))
    # Make some strategies better than others
    for i in range(n_strategies):
        ret[:, i] += (i - n_strategies / 2) * 0.00005
    ids = [f"Strategy_{i:03d}" for i in range(n_strategies)]
    return ret, ids


def _synth_strategies(n=20, seed=42):
    """Generate synthetic strategy dicts with metrics."""
    np.random.seed(seed)
    strats = []
    for i in range(n):
        strats.append({
            "strategy_id": f"Strategy_{i:03d}",
            "name": f"Strategy_{i:03d}",
            "origin": "discovered" if i < n // 2 else "mutation",
            "sharpe_ratio": round(np.random.uniform(-0.5, 2.5), 3),
            "max_drawdown_pct": round(np.random.uniform(5, 40), 1),
            "total_trades": int(np.random.uniform(10, 300)),
            "total_return_pct": round(np.random.uniform(-20, 80), 1),
            "win_rate": round(np.random.uniform(0.3, 0.7), 3),
            "profit_factor": round(np.random.uniform(0.5, 2.5), 2),
        })
    return strats


# ==============================================================================
# MODULE 1 TESTS: LineageTracker
# ==============================================================================

def test_m1_1_register_root():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        sid = t.register_strategy(name="Root_SMA", origin="discovered")
        assert sid, "No strategy_id returned"
        rec = t.get_strategy(sid)
        assert rec is not None, "Strategy not found"
        assert rec.name == "Root_SMA"
        assert rec.generation == 0
        assert rec.origin == "discovered"
    finally:
        shutil.rmtree(d)


def test_m1_2_register_child():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        root = t.register_strategy(name="Root", origin="discovered")
        child = t.register_strategy(
            name="Child_RSI", origin="mutation", parent_id=root,
            mutation_type="add_indicator", mutation_params={"indicator": "RSI"},
        )
        rec = t.get_strategy(child)
        assert rec.generation == 1
        assert rec.parent_id == root
        assert rec.mutation_type == "add_indicator"
        children = t.get_children(root)
        assert len(children) == 1
        assert children[0].strategy_id == child
    finally:
        shutil.rmtree(d)


def test_m1_3_deep_genealogy():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        ids = [t.register_strategy(name="Gen0", origin="discovered")]
        for g in range(1, 4):
            ids.append(t.register_strategy(
                name=f"Gen{g}", origin="mutation", parent_id=ids[-1],
                mutation_type="tweak",
            ))
        rec = t.get_strategy(ids[-1])
        assert rec.generation == 3
        desc = t.get_descendants(ids[0])
        assert len(desc) == 3
    finally:
        shutil.rmtree(d)


def test_m1_4_log_backtest():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        sid = t.register_strategy(name="S1", origin="discovered")
        row_id = t.log_backtest(sid, {
            "sharpe_ratio": 1.5, "max_drawdown_pct": 12.0,
            "total_return_pct": 45.0, "total_trades": 150,
        })
        assert row_id > 0
        m = t.get_best_metrics(sid)
        assert m is not None
        assert m["sharpe_ratio"] == 1.5
    finally:
        shutil.rmtree(d)


def test_m1_5_status_lifecycle():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        sid = t.register_strategy(name="S1", origin="discovered")
        assert t.get_strategy(sid).status == "pending"
        for status in ["backtested", "filtered", "promoted", "retired"]:
            t.update_status(sid, status)
            assert t.get_strategy(sid).status == status
    finally:
        shutil.rmtree(d)


def test_m1_6_mutation_success_rates():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        root = t.register_strategy(name="Root", origin="discovered")
        for mt in ["add_indicator", "change_params", "add_indicator"]:
            c = t.register_strategy(name=f"C_{mt}", origin="mutation",
                                    parent_id=root, mutation_type=mt)
            t.log_backtest(c, {"sharpe_ratio": np.random.uniform(0.5, 2.0)})
        rates = t.get_mutation_success_rates()
        assert "add_indicator" in rates
        assert rates["add_indicator"]["count"] == 2
    finally:
        shutil.rmtree(d)


def test_m1_7_family_tree():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        root = t.register_strategy(name="Root", origin="discovered")
        c1 = t.register_strategy(name="C1", origin="mutation", parent_id=root)
        c2 = t.register_strategy(name="C2", origin="mutation", parent_id=root)
        tree = t.get_family_tree(root)
        assert tree is not None
        assert len(tree.children) == 2
    finally:
        shutil.rmtree(d)


def test_m1_8_lineage_summary():
    d = _make_tmp()
    try:
        t = LineageTracker(db_path=f"{d}/lineage.db", enable_mlflow=False)
        for i in range(5):
            t.register_strategy(name=f"S{i}", origin="discovered" if i < 3 else "mutation")
        s = t.get_lineage_summary()
        assert s["total_strategies"] == 5
        assert s["discovered"] == 3
        assert s["mutated"] == 2
    finally:
        shutil.rmtree(d)


# ==============================================================================
# MODULE 2 TESTS: OverfittingDetector
# ==============================================================================

def test_m2_1_pbo_random():
    """Random data should produce PBO around 0.5 (no signal)."""
    det = OverfittingDetector(random_seed=42)
    np.random.seed(42)
    df = pd.DataFrame(np.random.normal(0, 0.01, (500, 10)),
                       columns=[f"s{i}" for i in range(10)])
    pbo = det.compute_pbo(df, n_partitions=8)
    assert 0.0 <= pbo.probability <= 1.0
    assert pbo.n_combinations > 0
    # Random data -> PBO should be moderate-to-high
    assert pbo.probability >= 0.2, f"PBO={pbo.probability} too low for random data"


def test_m2_2_pbo_with_signal():
    """Data with genuine signal should produce lower PBO."""
    det = OverfittingDetector(random_seed=42)
    np.random.seed(42)
    df = pd.DataFrame(np.random.normal(0, 0.01, (500, 10)),
                       columns=[f"s{i}" for i in range(10)])
    # Add strong signal to strategy 0
    df["s0"] += 0.003
    pbo = det.compute_pbo(df, n_partitions=8)
    assert pbo.probability < 0.8, f"PBO={pbo.probability} too high for signal data"


def test_m2_3_dsr_basic():
    det = OverfittingDetector()
    dsr = det.deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=1, T=252)
    assert dsr.is_significant  # Single trial, SR=2 should be significant
    assert dsr.deflated_sharpe > 0


def test_m2_4_dsr_many_trials():
    det = OverfittingDetector()
    dsr1 = det.deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=10, T=252)
    dsr2 = det.deflated_sharpe_ratio(observed_sharpe=1.5, n_trials=1000, T=252)
    # More trials -> higher expected max -> lower deflated sharpe
    assert dsr2.deflated_sharpe < dsr1.deflated_sharpe


def test_m2_5_psr_positive():
    det = OverfittingDetector()
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 500)  # positive drift
    psr = det.probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
    assert psr.psr > 0.5  # Should beat zero benchmark


def test_m2_6_psr_no_signal():
    det = OverfittingDetector()
    np.random.seed(42)
    returns = np.random.normal(0.0, 0.01, 500)  # no drift
    psr = det.probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)
    assert psr.psr < 0.9  # Should be close to 0.5


def test_m2_7_analyze_strategy():
    det = OverfittingDetector()
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 300)
    result = det.analyze_strategy(returns, n_trials=50)
    assert "psr" in result
    assert "dsr" in result
    assert result["track_length"] == 300


def test_m2_8_quantstats():
    det = OverfittingDetector()
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0003, 0.01, 252),
                         index=pd.date_range("2024-01-01", periods=252))
    m = det.get_quantstats_metrics(returns)
    assert "sharpe" in m
    assert "max_drawdown" in m


# ==============================================================================
# MODULE 3 TESTS: FilteringPipeline
# ==============================================================================

def test_m3_1_pass():
    pipe = FilteringPipeline()
    result = pipe.run(
        [{"strategy_id": "G", "name": "GoodStrategy",
          "sharpe_ratio": 1.5, "max_drawdown_pct": 15, "total_trades": 100,
          "total_return_pct": 30}],
        config=FilterConfig(top_n=10),
    )
    assert result.total_survivors == 1


def test_m3_2_reject():
    pipe = FilteringPipeline()
    result = pipe.run(
        [{"strategy_id": "B", "name": "BadStrategy",
          "sharpe_ratio": 0.1, "max_drawdown_pct": 50, "total_trades": 5,
          "total_return_pct": -10}],
        config=FilterConfig(min_sharpe=0.3, max_drawdown=30, min_trades=30),
    )
    assert result.total_survivors == 0
    assert len(result.rejected) == 1
    assert len(result.rejected[0].rejection_reasons) >= 3


def test_m3_3_top_n():
    strats = [
        {"strategy_id": f"S_{i}", "name": f"Strategy_{i}",
         "sharpe_ratio": 1.0 + i * 0.1, "max_drawdown_pct": 15,
         "total_trades": 100, "total_return_pct": 20}
        for i in range(20)
    ]
    pipe = FilteringPipeline()
    result = pipe.run(strats, config=FilterConfig(top_n=5))
    assert result.total_survivors == 5
    # Best Sharpe should be ranked #1
    assert abs(result.survivors[0].metrics["sharpe_ratio"] - 2.9) < 0.01


def test_m3_4_composite_scoring():
    pipe = FilteringPipeline()
    strats = [
        {"strategy_id": "H", "name": "HighSharpe",
         "sharpe_ratio": 2.0, "max_drawdown_pct": 25, "total_trades": 50,
         "total_return_pct": 40},
        {"strategy_id": "L", "name": "LowSharpe",
         "sharpe_ratio": 0.5, "max_drawdown_pct": 10, "total_trades": 200,
         "total_return_pct": 10},
    ]
    result = pipe.run(strats, config=FilterConfig(top_n=2))
    assert result.survivors[0].strategy_id == "H"  # Higher Sharpe wins
    assert result.survivors[0].composite_score > result.survivors[1].composite_score


def test_m3_5_with_pbo():
    det = OverfittingDetector(random_seed=42)
    pipe = FilteringPipeline(overfitting_detector=det)

    np.random.seed(42)
    n = 10
    ret = np.random.normal(0.0003, 0.01, (500, n))
    ret_df = pd.DataFrame(ret, columns=[f"S{i}" for i in range(n)])

    strats = [
        {"strategy_id": f"S{i}", "name": f"Strategy_{i}",
         "sharpe_ratio": 1.0 + np.random.uniform(-0.5, 1.5),
         "max_drawdown_pct": 15, "total_trades": 100, "total_return_pct": 20}
        for i in range(n)
    ]
    result = pipe.run(strats, config=FilterConfig(top_n=10), returns_matrix=ret_df)
    assert result.total_input == n


def test_m3_6_save_load():
    d = _make_tmp()
    try:
        pipe = FilteringPipeline()
        result = pipe.run(
            [{"strategy_id": "S1", "name": "S1",
              "sharpe_ratio": 1.0, "max_drawdown_pct": 15,
              "total_trades": 100, "total_return_pct": 20}],
            config=FilterConfig(top_n=10),
        )
        path = f"{d}/results.json"
        pipe.save_results(result, path)
        assert os.path.exists(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["total_survivors"] == 1
    finally:
        shutil.rmtree(d)


# ==============================================================================
# MODULE 4 TESTS: DiversificationFilter
# ==============================================================================

def test_m4_1_uncorrelated_pass():
    """Uncorrelated strategies should all pass."""
    np.random.seed(42)
    filt = DiversificationFilter()
    strats = [
        {"strategy_id": f"S{i}", "name": f"Strategy_{i}", "composite_score": 0.8 - i * 0.05}
        for i in range(5)
    ]
    returns = {f"S{i}": np.random.normal(0, 0.01, 500) for i in range(5)}
    result = filt.run(strats, returns_dict=returns, config=DiversityConfig(max_correlation=0.5))
    assert len(result.selected) == 5
    assert len(result.removed) == 0


def test_m4_2_correlated_removal():
    """Perfectly correlated strategies should be removed."""
    np.random.seed(42)
    filt = DiversificationFilter()
    base = np.random.normal(0, 0.01, 500)
    strats = [
        {"strategy_id": f"S{i}", "name": f"Correlated_{i}", "composite_score": 0.9 - i * 0.05}
        for i in range(5)
    ]
    # All share the same returns (perfect correlation)
    returns = {f"S{i}": base + np.random.normal(0, 0.0001, 500) for i in range(5)}
    result = filt.run(strats, returns_dict=returns,
                      config=DiversityConfig(max_correlation=0.5, min_strategies=1))
    assert len(result.selected) < 5  # Some must be removed


def test_m4_3_greedy_order():
    """Greedy should pick highest-scored first."""
    np.random.seed(42)
    filt = DiversificationFilter()
    strats = [
        {"strategy_id": "A", "name": "Best", "composite_score": 0.95},
        {"strategy_id": "B", "name": "Medium", "composite_score": 0.70},
        {"strategy_id": "C", "name": "Worst", "composite_score": 0.30},
    ]
    returns = {k: np.random.normal(0, 0.01, 500) for k in ["A", "B", "C"]}
    result = filt.run(strats, returns_dict=returns)
    assert result.selected[0]["name"] == "Best"


def test_m4_4_trade_overlap():
    """Trade overlap utility."""
    filt = DiversificationFilter()
    overlap = filt.compute_trade_overlap(
        {"2024-01-01", "2024-01-02", "2024-01-03"},
        {"2024-01-02", "2024-01-03", "2024-01-04"},
    )
    # Jaccard: intersection=2, union=4 -> 0.5
    assert abs(overlap - 0.5) < 0.01


def test_m4_5_diversity_stats():
    """Check diversity statistics are computed."""
    np.random.seed(42)
    filt = DiversificationFilter()
    strats = [
        {"strategy_id": f"S_{i}", "name": f"S_{i}",
         "composite_score": np.random.uniform(0.5, 0.9)}
        for i in range(10)
    ]
    returns = {f"S_{i}": np.random.normal(0, 0.01, 500) for i in range(10)}
    result = filt.run(strats, returns_dict=returns, config=DiversityConfig(max_strategies=5))
    assert result.effective_n > 0
    assert result.avg_pairwise_corr >= 0


def test_m4_6_clustering():
    """Clustering should produce valid labels."""
    np.random.seed(42)
    filt = DiversificationFilter()
    strats = [
        {"strategy_id": f"S{i}", "name": f"S{i}", "composite_score": 0.8}
        for i in range(6)
    ]
    returns = {f"S{i}": np.random.normal(0, 0.01, 500) for i in range(6)}
    result = filt.run(strats, returns_dict=returns)
    if result.cluster_labels:
        assert result.n_clusters >= 1


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_int_1_full_pipeline():
    d = _make_tmp()
    try:
        strats = _synth_strategies(20, seed=42)
        ret, ids = _synth_returns(20, 500, seed=42)
        returns_dict = {ids[i]: ret[:, i] for i in range(20)}
        for i, s in enumerate(strats):
            s["strategy_id"] = ids[i]
            s["name"] = ids[i]

        pipe = Phase1Pipeline(
            db_path=f"{d}/lineage.db",
            enable_mlflow=False,
            filter_config=FilterConfig(top_n=15, min_sharpe=0.0, min_trades=0),
            diversity_config=DiversityConfig(max_strategies=10),
        )
        result = pipe.run(strats, returns_dict=returns_dict)
        assert result.total_registered == 20
        assert result.total_diversified <= 10
        assert result.total_diversified > 0
        assert result.pbo_result is not None
    finally:
        shutil.rmtree(d)


def test_int_2_output_files():
    d = _make_tmp()
    try:
        strats = _synth_strategies(10, seed=123)
        ret, ids = _synth_returns(10, 500, seed=123)
        returns_dict = {ids[i]: ret[:, i] for i in range(10)}
        for i, s in enumerate(strats):
            s["strategy_id"] = ids[i]
            s["name"] = ids[i]

        pipe = Phase1Pipeline(
            db_path=f"{d}/lineage.db", enable_mlflow=False,
            filter_config=FilterConfig(top_n=10, min_sharpe=0.0, min_trades=0),
        )
        result = pipe.run(strats, returns_dict=returns_dict)

        output_dir = Path(d) / "output"
        assert (output_dir / "phase1_results.json").exists()
        assert (output_dir / "filter_results.json").exists()
    finally:
        shutil.rmtree(d)


def test_int_3_lineage_populated():
    d = _make_tmp()
    try:
        strats = _synth_strategies(15, seed=99)
        for i, s in enumerate(strats):
            s["strategy_id"] = f"SID_{i}"

        pipe = Phase1Pipeline(
            db_path=f"{d}/lineage.db", enable_mlflow=False,
            filter_config=FilterConfig(top_n=15, min_sharpe=0.0, min_trades=0),
        )
        pipe.run(strats)

        tracker = pipe.tracker
        assert tracker.strategy_count() == 15
        summary = tracker.get_lineage_summary()
        assert summary["total_strategies"] == 15
    finally:
        shutil.rmtree(d)


def test_int_4_no_returns():
    """Pipeline should work without returns (skip PBO)."""
    d = _make_tmp()
    try:
        strats = [
            {"strategy_id": f"S_{i}", "name": f"S_{i}",
             "sharpe_ratio": 1.0 + i * 0.1, "max_drawdown_pct": 15,
             "total_trades": 100, "total_return_pct": 20}
            for i in range(10)
        ]
        pipe = Phase1Pipeline(
            db_path=f"{d}/lineage.db", enable_mlflow=False,
            filter_config=FilterConfig(top_n=5),
        )
        result = pipe.run(strats)
        assert result.pbo_result is None
        assert result.total_diversified <= 5
    finally:
        shutil.rmtree(d)


# ==============================================================================
# MAIN
# ==============================================================================

QUICK_TESTS = {
    "M1.1", "M1.4", "M2.1", "M2.3", "M3.1", "M3.3", "M4.1", "INT.1",
}

ALL_TESTS = [
    # Module 1: LineageTracker
    ("M1.1 Register root strategy",    test_m1_1_register_root),
    ("M1.2 Register child strategy",   test_m1_2_register_child),
    ("M1.3 Deep genealogy (4 gen)",    test_m1_3_deep_genealogy),
    ("M1.4 Log backtest metrics",      test_m1_4_log_backtest),
    ("M1.5 Status lifecycle",          test_m1_5_status_lifecycle),
    ("M1.6 Mutation success rates",    test_m1_6_mutation_success_rates),
    ("M1.7 Family tree build",         test_m1_7_family_tree),
    ("M1.8 Lineage summary",           test_m1_8_lineage_summary),
    # Module 2: OverfittingDetector
    ("M2.1 PBO random data",           test_m2_1_pbo_random),
    ("M2.2 PBO with signal",           test_m2_2_pbo_with_signal),
    ("M2.3 DSR basic",                 test_m2_3_dsr_basic),
    ("M2.4 DSR trial scaling",         test_m2_4_dsr_many_trials),
    ("M2.5 PSR positive",              test_m2_5_psr_positive),
    ("M2.6 PSR no signal",             test_m2_6_psr_no_signal),
    ("M2.7 Analyze strategy",          test_m2_7_analyze_strategy),
    ("M2.8 quantstats metrics",        test_m2_8_quantstats),
    # Module 3: FilteringPipeline
    ("M3.1 Hard filter pass",          test_m3_1_pass),
    ("M3.2 Hard filter reject",        test_m3_2_reject),
    ("M3.3 Top-N selection",           test_m3_3_top_n),
    ("M3.4 Composite scoring",         test_m3_4_composite_scoring),
    ("M3.5 Filter with PBO",           test_m3_5_with_pbo),
    ("M3.6 Save/load results",         test_m3_6_save_load),
    # Module 4: DiversificationFilter
    ("M4.1 Uncorrelated pass",         test_m4_1_uncorrelated_pass),
    ("M4.2 Correlated removal",        test_m4_2_correlated_removal),
    ("M4.3 Greedy score order",        test_m4_3_greedy_order),
    ("M4.4 Trade overlap sim",         test_m4_4_trade_overlap),
    ("M4.5 Diversity stats",           test_m4_5_diversity_stats),
    ("M4.6 Strategy clustering",       test_m4_6_clustering),
    # Integration
    ("INT.1 Full pipeline",            test_int_1_full_pipeline),
    ("INT.2 Output files",             test_int_2_output_files),
    ("INT.3 Lineage populated",        test_int_3_lineage_populated),
    ("INT.4 No returns mode",          test_int_4_no_returns),
]


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    start = time.time()

    modules = [
        "Module 1: LineageTracker",
        "Module 2: OverfittingDetector",
        "Module 3: FilteringPipeline",
        "Module 4: DiversificationFilter",
        "Module 5: Integration",
    ]
    mod_ranges = [(0, 8), (8, 16), (16, 22), (22, 28), (28, 32)]

    for mod_name, (lo, hi) in zip(modules, mod_ranges):
        print(f"\n{'-'*60}")
        print(f"  {mod_name}")
        print(f"{'-'*60}")
        for name, fn in ALL_TESTS[lo:hi]:
            run_test(name, fn, QUICK_TESTS, quick)

    elapsed = time.time() - start
    print(f"\n  ⏱️  Completed in {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print(f"  PHASE 1 TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total:   {_passed + _failed}")
    print(f"  Passed:  {_passed} [OK]")
    print(f"  Failed:  {_failed} [FAIL]")
    if _errors:
        print(f"\n  Failures:")
        for name, err in _errors:
            print(f"    {name}: {err}")
    print(f"{'='*60}")

    sys.exit(0 if _failed == 0 else 1)
