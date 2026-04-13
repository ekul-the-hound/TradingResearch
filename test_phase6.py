#!/usr/bin/env python3
# test_phase6.py -- Phase 6 Learning Loop FULL Test Suite (50 tests)
import sys, time, traceback, tempfile, shutil
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from learning_loop import LearningLoop, LoopConfig, TriggerType
from lineage_analytics import LineageAnalyzer, StrategyLineage
from retraining_scheduler import (
    RetrainingScheduler, ScheduleConfig, WindowType, RetrainTrigger, RetrainDecision,
)
from performance_attribution import PerformanceAttributor
from experiment_tracker import ExperimentTracker, RunStatus

_p, _f, _e = 0, 0, []

def run_test(name, fn):
    global _p, _f
    try:
        fn(); _p += 1; print(f"  [OK] {name}")
    except Exception as ex:
        _f += 1; _e.append((name, str(ex))); print(f"  [FAIL] {name}: {ex}")
        traceback.print_exc()


def _make_strategies(n=20, seed=42):
    np.random.seed(seed)
    out = []
    for i in range(n):
        out.append(StrategyLineage(
            strategy_id=f"S_{i:03d}",
            parent_id=f"S_{i-1:03d}" if i > 0 else None,
            mutation_type=["add_indicator", "change_params", "add_filter", "change_exit"][i % 4],
            hypothesis_id=f"H_{i % 3}", generation=i // 4,
            created_at="2025-06-01T00:00:00",
            backtest_sharpe=round(np.random.uniform(0.5, 2.5), 3),
            live_sharpe=round(np.random.uniform(0.2, 2.0), 3),
            max_drawdown=round(np.random.uniform(0.05, 0.3), 3),
            total_trades=int(np.random.uniform(20, 200)),
            profit_factor=round(np.random.uniform(0.8, 2.0), 2),
            compute_seconds=round(np.random.uniform(5, 60), 1),
            api_cost_usd=round(np.random.uniform(0.01, 0.20), 3),
            regime_sharpes={"bull": round(np.random.uniform(0.5, 2.5), 2),
                           "bear": round(np.random.uniform(-0.5, 1.0), 2)},
            is_active=i < 15,
            final_state=["research", "paper", "live", "retired"][i % 4],
        ))
    return out


# ==============================================================================
# MODULE 1: Learning Loop (12 tests)
# ==============================================================================

def test_ll_01_register():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""))
    lp.register_strategy("S_001", backtest_sharpe=1.5)
    assert "S_001" in lp.get_strategy_states()

def test_ll_02_update():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""))
    lp.register_strategy("S_001", backtest_sharpe=2.0)
    lp.update_live_data("S_001", np.random.normal(0.0003, 0.01, 100))
    assert lp.get_strategy_states()["S_001"].live_sharpe != 0

def test_ll_03_basic_cycle():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""))
    lp.register_strategy("S_001", backtest_sharpe=1.5)
    assert lp.run_cycle(TriggerType.MANUAL).cycle_id == 1

def test_ll_04_drift():
    retrained = []
    demoted = []
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""),
                       retrain_fn=lambda sid: retrained.append(sid),
                       demote_fn=lambda sid: demoted.append(sid))
    lp.register_strategy("S_001", backtest_sharpe=2.0)
    # Mild degradation so it retrains rather than demotes
    lp.update_live_data("S_001", np.random.normal(0.0005, 0.012, 100), drift_detected=True)
    lp.run_cycle(TriggerType.DRIFT)
    assert len(retrained) > 0 or len(demoted) > 0  # Either action is valid

def test_ll_05_degradation():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir="", sharpe_degradation_trigger_pct=30))
    lp.register_strategy("S_001", backtest_sharpe=2.0)
    lp.update_live_data("S_001", np.random.normal(-0.002, 0.01, 100))
    assert lp.run_cycle().strategies_flagged > 0

def test_ll_06_surrogate():
    refreshed = [False]
    lp = LearningLoop(
        LoopConfig(verbose=False, state_dir="", min_new_backtests_for_refresh=10),
        surrogate_refresh_fn=lambda: refreshed.__setitem__(0, True))
    lp.add_backtest_results(15)
    r = lp.run_cycle()
    assert r.surrogate_refreshed and refreshed[0]

def test_ll_07_kappa_adapt():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""))
    for i in range(5):
        lp.register_strategy(f"S_{i}", backtest_sharpe=2.0)
        lp.update_live_data(f"S_{i}", np.random.normal(-0.003, 0.01, 100))
    k0 = lp.kappa
    lp.run_cycle()
    assert lp.kappa >= k0

def test_ll_08_prune():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir="", hypothesis_min_improvement_rate=0.5))
    for i in range(5):
        lp.register_strategy(f"S_{i}", backtest_sharpe=2.0, hypothesis_id="H_bad")
        lp.update_live_data(f"S_{i}", np.random.normal(-0.005, 0.01, 50))
    assert lp.run_cycle().hypotheses_pruned > 0

def test_ll_09_mutation_eff():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""))
    lp.register_strategy("parent", backtest_sharpe=1.0)
    lp.update_live_data("parent", np.random.normal(0.001, 0.01, 100))
    lp.register_strategy("child", backtest_sharpe=1.5, mutation_type="add_indicator", parent_id="parent")
    lp.update_live_data("child", np.random.normal(0.002, 0.01, 100))
    assert "add_indicator" in lp.get_mutation_effectiveness()

def test_ll_10_checkpoint():
    d = tempfile.mkdtemp()
    try:
        lp = LearningLoop(LoopConfig(verbose=False, state_dir=d))
        lp.register_strategy("S_001", backtest_sharpe=1.5)
        lp.run_cycle()
        assert len(list(Path(d).glob("*.json"))) > 0
    finally:
        shutil.rmtree(d)

def test_ll_11_multi_cycle():
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""))
    lp.register_strategy("S_001", backtest_sharpe=1.5)
    for _ in range(5):
        lp.run_cycle()
    assert lp.cycle_count == 5 and len(lp.get_history()) == 5

def test_ll_12_severe_demote():
    demoted = []
    lp = LearningLoop(LoopConfig(verbose=False, state_dir=""),
                       demote_fn=lambda sid: demoted.append(sid))
    lp.register_strategy("S_001", backtest_sharpe=2.0)
    s = lp.get_strategy_states()["S_001"]
    s.drift_detected = True
    s.degradation_pct = 70
    s.live_sharpe = 0.5
    assert lp.run_cycle(TriggerType.DRIFT).strategies_demoted > 0
    assert "S_001" in demoted


# ==============================================================================
# MODULE 2: Lineage Analytics (14 tests)
# ==============================================================================

def test_la_01_add():
    la = LineageAnalyzer()
    la.add_strategy(StrategyLineage("S_001", backtest_sharpe=1.5, live_sharpe=1.2))
    assert "S_001" in la._strategies

def test_la_02_batch():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(10))
    assert len(la._strategies) == 10

def test_la_03_report():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    r = la.generate_report()
    assert r.total_strategies == 20 and r.active_strategies == 15

def test_la_04_mutation_reports():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    for mr in la.generate_report().mutation_reports:
        assert mr.total_count > 0 and 0 <= mr.success_rate <= 1

def test_la_05_hypothesis_decay():
    la = LineageAnalyzer()
    strats = _make_strategies(20)
    for s in strats:
        if s.hypothesis_id == "H_0":
            s.live_sharpe = 0.1
    la.add_batch(strats)
    h0 = [h for h in la.generate_report().hypothesis_reports if h.hypothesis_id == "H_0"]
    assert len(h0) == 1 and h0[0].avg_sharpe < 0.5

def test_la_06_generation():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    assert len(la.generate_report().generation_reports) > 0

def test_la_07_weights():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    w = la.generate_report().mutation_weights
    assert len(w) > 0 and abs(sum(w.values()) - 1.0) < 0.01

def test_la_08_regime():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    la.generate_report()  # No crash

def test_la_09_tree():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    tree = la.get_family_tree("S_003")
    assert len(tree) > 0 and tree[0]["strategy_id"] == "S_003"

def test_la_10_descendants():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    assert "S_001" in la.get_descendants("S_000")

def test_la_11_top_lineages():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    top = la.get_top_lineages(3)
    assert len(top) <= 3
    for lin in top:
        assert "best_sharpe" in lin

def test_la_12_from_dict():
    la = LineageAnalyzer()
    la.add_from_dict({"strategy_id": "S_t", "backtest_sharpe": 1.5, "live_sharpe": 1.2})
    assert "S_t" in la._strategies

def test_la_13_empty():
    assert LineageAnalyzer().generate_report().total_strategies == 0

def test_la_14_summary():
    la = LineageAnalyzer()
    la.add_batch(_make_strategies(20))
    assert "LINEAGE" in la.generate_report().summary()


# ==============================================================================
# MODULE 3: Retraining Scheduler (10 tests)
# ==============================================================================

def test_rs_01_register():
    rs = RetrainingScheduler(ScheduleConfig(state_dir=""))
    rs.register("S_001", data_start="2023-01-01", current_sharpe=1.5)
    assert rs.get_schedule("S_001") is not None

def test_rs_02_rolling_window():
    rs = RetrainingScheduler(ScheduleConfig(
        window_type=WindowType.ROLLING, rolling_window_days=252, state_dir=""))
    rs.register("S_001", data_start="2020-01-01")
    w = rs.compute_window("S_001")
    assert w.n_days <= 253 and w.window_type == WindowType.ROLLING

def test_rs_03_expanding_window():
    rs = RetrainingScheduler(ScheduleConfig(
        window_type=WindowType.EXPANDING, state_dir=""))
    rs.register("S_001", data_start="2020-01-01")
    assert rs.compute_window("S_001").n_days > 252

def test_rs_04_trigger():
    rs = RetrainingScheduler(ScheduleConfig(state_dir=""))
    rs.register("S_001")
    job = rs.trigger_retrain("S_001", RetrainTrigger.DRIFT, priority=10)
    assert job.priority == 10 and job.trigger == RetrainTrigger.DRIFT

def test_rs_05_execute():
    rs = RetrainingScheduler(ScheduleConfig(state_dir=""))
    rs.register("S_001", current_sharpe=1.5)
    result = rs.execute_job(rs.trigger_retrain("S_001"))
    assert result.decision in list(RetrainDecision)
    assert result.old_sharpe == 1.5

def test_rs_06_adopt():
    rs = RetrainingScheduler(
        ScheduleConfig(state_dir="", min_improvement_pct=5),
        backtest_fn=lambda sid, s, e: {"sharpe_ratio": 2.5, "params": {"improved": True}},
    )
    rs.register("S_001", current_sharpe=1.0)
    result = rs.execute_job(rs.trigger_retrain("S_001"))
    assert result.decision == RetrainDecision.ADOPT_NEW
    assert rs.get_schedule("S_001").current_sharpe == 2.5

def test_rs_07_walk_forward():
    rs = RetrainingScheduler(ScheduleConfig(state_dir=""))
    rs.register("S_001", data_start="2022-01-01", current_sharpe=1.5)
    results = rs.walk_forward_retrain("S_001", n_folds=3)
    assert len(results) == 3

def test_rs_08_summary():
    rs = RetrainingScheduler(ScheduleConfig(state_dir=""))
    rs.register("S_001", current_sharpe=1.5)
    rs.execute_job(rs.trigger_retrain("S_001"))
    s = rs.summary()
    assert s["total_retrains"] == 1 and s["registered"] == 1

def test_rs_09_adaptive():
    rs = RetrainingScheduler(ScheduleConfig(adaptive=True, state_dir=""))
    rs.register("S_001")
    rs.update_volatility("S_001", current_vol=0.04, normal_vol=0.01)
    assert rs.get_schedule("S_001").next_retrain is not None

def test_rs_10_checkpoint():
    d = tempfile.mkdtemp()
    try:
        rs = RetrainingScheduler(ScheduleConfig(state_dir=d))
        rs.register("S_001", current_sharpe=1.5)
        rs.execute_job(rs.trigger_retrain("S_001"))
        assert len(list(Path(d).glob("*.json"))) > 0
    finally:
        shutil.rmtree(d)


# ==============================================================================
# MODULE 4: Performance Attribution (8 tests)
# ==============================================================================

def test_pa_01_basic():
    np.random.seed(42)
    result = PerformanceAttributor().attribute(np.random.normal(0.001, 0.01, 500))
    assert abs(result.total_return) < 5

def test_pa_02_alpha_beta():
    np.random.seed(42)
    bench = np.random.normal(0.0005, 0.01, 500)
    strat = 0.5 * bench + np.random.normal(0.0003, 0.005, 500)
    result = PerformanceAttributor().attribute(strat, bench)
    assert result.beta > 0.3 and result.r_squared > 0.1

def test_pa_03_factors():
    np.random.seed(42)
    n = 500
    mom = np.random.normal(0.0003, 0.01, n)
    vol = np.random.normal(0, 0.008, n)
    strat = 0.5 * mom + 0.3 * vol + np.random.normal(0.0002, 0.005, n)
    result = PerformanceAttributor().attribute(
        strat, factor_returns={"momentum": mom, "volatility": vol})
    assert "momentum" in result.factor_exposures
    assert result.factor_exposures["momentum"] > 0.2

def test_pa_04_regime():
    np.random.seed(42)
    r = np.random.normal(0.001, 0.01, 500)
    regimes = np.array([0] * 200 + [1] * 200 + [2] * 100)
    result = PerformanceAttributor().attribute(
        r, regimes=regimes, regime_labels={0: "bull", 1: "bear", 2: "ranging"})
    assert "bull" in result.regime_returns

def test_pa_05_costs():
    np.random.seed(42)
    r = np.random.normal(0.001, 0.01, 200)
    costs = {"commission": 0.005, "spread": 0.003, "slippage": 0.002}
    result = PerformanceAttributor().attribute(r, costs=costs)
    assert result.total_costs == 0.01
    assert result.gross_return > result.net_return

def test_pa_06_skill():
    np.random.seed(42)
    bench = np.random.normal(0, 0.01, 500)
    strat = bench + np.random.normal(0.002, 0.005, 500)
    result = PerformanceAttributor().attribute(strat, bench, n_bootstrap=200)
    assert result.skill_pvalue < 0.5

def test_pa_07_timing():
    np.random.seed(42)
    trades = [{"pnl": 100, "size": 10000, "entry_price": 1.1, "hold_bars": b}
              for b in range(5, 25)]
    result = PerformanceAttributor().attribute(
        np.random.normal(0.001, 0.01, 100), trades=trades)
    assert isinstance(result.entry_timing_value, float)

def test_pa_08_no_bench():
    np.random.seed(42)
    result = PerformanceAttributor().attribute(np.random.normal(0.001, 0.01, 200))
    assert result.beta == 0 and result.r_squared == 0


# ==============================================================================
# MODULE 5: Experiment Tracker (6 tests)
# ==============================================================================

def test_et_01_create():
    d = tempfile.mkdtemp()
    try:
        et = ExperimentTracker(d)
        eid = et.create_experiment("test_exp", "description")
        assert et.get_experiment(eid).name == "test_exp"
    finally:
        shutil.rmtree(d)

def test_et_02_lifecycle():
    et = ExperimentTracker(tempfile.mkdtemp())
    eid = et.create_experiment("exp1")
    rid = et.start_run(eid, "run1")
    et.log_params(rid, {"sma_fast": 10})
    et.log_metrics(rid, {"sharpe": 1.8, "max_dd": 0.12})
    et.end_run(rid)
    run = et.get_run(rid)
    assert run.status == RunStatus.COMPLETED
    assert run.metrics["sharpe"] == 1.8 and run.params["sma_fast"] == 10

def test_et_03_search():
    et = ExperimentTracker(tempfile.mkdtemp())
    eid = et.create_experiment("exp1")
    for i in range(10):
        tag = "add_indicator" if i < 5 else "change_params"
        rid = et.start_run(eid, f"run_{i}", tags={"mutation": tag})
        et.log_metrics(rid, {"sharpe": 0.5 + i * 0.2})
        et.end_run(rid)
    results = et.search_runs(experiment_id=eid, filter_tags={"mutation": "add_indicator"})
    assert len(results) == 5
    best = et.get_best_run(eid, "sharpe")
    assert abs(best.metrics["sharpe"] - (0.5 + 9 * 0.2)) < 0.01

def test_et_04_compare():
    et = ExperimentTracker(tempfile.mkdtemp())
    eid = et.create_experiment("exp1")
    rids = []
    for i in range(5):
        rid = et.start_run(eid)
        et.log_metrics(rid, {"sharpe": 1.0 + i * 0.3})
        et.end_run(rid)
        rids.append(rid)
    comp = et.compare_runs(rids, "sharpe")
    assert abs(comp.best_value - (1.0 + 4 * 0.3)) < 0.01
    assert comp.std > 0

def test_et_05_save_load():
    d = tempfile.mkdtemp()
    try:
        et = ExperimentTracker(d)
        eid = et.create_experiment("exp1")
        rid = et.start_run(eid, "run1")
        et.log_metrics(rid, {"sharpe": 1.5})
        et.end_run(rid)
        et.save()
        # Reload from disk
        et2 = ExperimentTracker(d)
        assert et2.get_experiment(eid) is not None
        assert et2.get_run(rid).metrics["sharpe"] == 1.5
    finally:
        shutil.rmtree(d)

def test_et_06_summary():
    et = ExperimentTracker(tempfile.mkdtemp())
    eid = et.create_experiment("exp1")
    rid1 = et.start_run(eid)
    et.end_run(rid1)
    rid2 = et.start_run(eid)
    et.fail_run(rid2, "error")
    s = et.summary()
    assert s["completed"] == 1 and s["failed"] == 1


# ==============================================================================
# RUNNER
# ==============================================================================

ALL = [
    # Module 1: Learning Loop (12)
    ("LL.01 Register",              test_ll_01_register),
    ("LL.02 Update live data",      test_ll_02_update),
    ("LL.03 Basic cycle",           test_ll_03_basic_cycle),
    ("LL.04 Drift trigger",         test_ll_04_drift),
    ("LL.05 Degradation flag",      test_ll_05_degradation),
    ("LL.06 Surrogate refresh",     test_ll_06_surrogate),
    ("LL.07 Kappa adaptation",      test_ll_07_kappa_adapt),
    ("LL.08 Hypothesis prune",      test_ll_08_prune),
    ("LL.09 Mutation effectiveness", test_ll_09_mutation_eff),
    ("LL.10 Checkpoint",            test_ll_10_checkpoint),
    ("LL.11 Multiple cycles",       test_ll_11_multi_cycle),
    ("LL.12 Severe drift demote",   test_ll_12_severe_demote),
    # Module 2: Lineage Analytics (14)
    ("LA.01 Add strategy",          test_la_01_add),
    ("LA.02 Batch add",             test_la_02_batch),
    ("LA.03 Full report",           test_la_03_report),
    ("LA.04 Mutation reports",      test_la_04_mutation_reports),
    ("LA.05 Hypothesis decay",      test_la_05_hypothesis_decay),
    ("LA.06 Generation analysis",   test_la_06_generation),
    ("LA.07 Mutation weights",      test_la_07_weights),
    ("LA.08 Regime bias",           test_la_08_regime),
    ("LA.09 Family tree",           test_la_09_tree),
    ("LA.10 Descendants",           test_la_10_descendants),
    ("LA.11 Top lineages",          test_la_11_top_lineages),
    ("LA.12 From dict",             test_la_12_from_dict),
    ("LA.13 Empty analyzer",        test_la_13_empty),
    ("LA.14 Summary output",        test_la_14_summary),
    # Module 3: Retraining Scheduler (10)
    ("RS.01 Register",              test_rs_01_register),
    ("RS.02 Rolling window",        test_rs_02_rolling_window),
    ("RS.03 Expanding window",      test_rs_03_expanding_window),
    ("RS.04 Trigger retrain",       test_rs_04_trigger),
    ("RS.05 Execute job",           test_rs_05_execute),
    ("RS.06 Adopt new params",      test_rs_06_adopt),
    ("RS.07 Walk-forward",          test_rs_07_walk_forward),
    ("RS.08 Summary",               test_rs_08_summary),
    ("RS.09 Adaptive schedule",     test_rs_09_adaptive),
    ("RS.10 Checkpoint",            test_rs_10_checkpoint),
    # Module 4: Performance Attribution (8)
    ("PA.01 Basic attribution",     test_pa_01_basic),
    ("PA.02 Alpha/beta",            test_pa_02_alpha_beta),
    ("PA.03 Factor decomposition",  test_pa_03_factors),
    ("PA.04 Regime attribution",    test_pa_04_regime),
    ("PA.05 Cost drag",             test_pa_05_costs),
    ("PA.06 Skill test",            test_pa_06_skill),
    ("PA.07 Timing attribution",    test_pa_07_timing),
    ("PA.08 No benchmark",          test_pa_08_no_bench),
    # Module 5: Experiment Tracker (6)
    ("ET.01 Create experiment",     test_et_01_create),
    ("ET.02 Run lifecycle",         test_et_02_lifecycle),
    ("ET.03 Search & filter",       test_et_03_search),
    ("ET.04 Compare runs",          test_et_04_compare),
    ("ET.05 Save/load",             test_et_05_save_load),
    ("ET.06 Summary",               test_et_06_summary),
]

if __name__ == "__main__":
    start = time.time()
    mods = [
        ("Module 1: Learning Loop",          0, 12),
        ("Module 2: Lineage Analytics",      12, 26),
        ("Module 3: Retraining Scheduler",   26, 36),
        ("Module 4: Performance Attribution",36, 44),
        ("Module 5: Experiment Tracker",     44, 50),
    ]
    for name, lo, hi in mods:
        print(f"\n{'-'*60}\n  {name}\n{'-'*60}")
        for n, fn in ALL[lo:hi]:
            run_test(n, fn)
    print(f"\n  ⏱️  {time.time()-start:.1f}s\n{'='*60}")
    print(f"  PHASE 6: {_p} passed, {_f} failed")
    if _e:
        for n, e in _e:
            print(f"    {n}: {e}")
    print(f"{'='*60}")
    sys.exit(0 if _f == 0 else 1)
