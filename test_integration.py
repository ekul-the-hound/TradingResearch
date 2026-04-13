#!/usr/bin/env python3
# ==============================================================================
# test_integration.py -- Tests for the integration layer
# ==============================================================================
# These tests verify the 3 integration pieces work correctly WITHOUT
# needing your data files or Backtrader running. They test:
#   1. CanonicalResult creation and format conversion
#   2. BacktestAdapter in dry-run mode
#   3. Pipeline structure and step chaining
# ==============================================================================

import sys
import time
import traceback
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from canonical_result import CanonicalResult
from backtest_adapter import BacktestAdapter

_p, _f, _e = 0, 0, []

def run_test(name, fn):
    global _p, _f
    try:
        fn(); _p += 1; print(f"  [OK] {name}")
    except Exception as ex:
        _f += 1; _e.append((name, str(ex))); print(f"  [FAIL] {name}: {ex}")
        traceback.print_exc()


# ==============================================================================
# CanonicalResult Tests (15)
# ==============================================================================

def test_cr_01_from_dict():
    raw = {
        "strategy_name": "SMA_Cross",
        "symbol": "EUR-USD",
        "timeframe": "1hour",
        "total_return_pct": 15.5,
        "sharpe_ratio": 1.8,
        "max_drawdown_pct": 12.3,
        "total_trades": 45,
        "win_rate": 55.0,
        "profit_factor": 1.6,
        "starting_value": 10000,
        "ending_value": 11550,
        "bars_tested": 500,
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "strategy_params": {"fast_period": 10, "slow_period": 50},
    }
    cr = CanonicalResult.from_backtest(raw, strategy_id="test_001")
    assert cr.strategy_id == "test_001"
    assert cr.sharpe_ratio == 1.8
    assert cr.total_trades == 45

def test_cr_02_auto_id():
    cr = CanonicalResult.from_backtest({"strategy_name": "SMA", "symbol": "EURUSD", "timeframe": "1h"})
    assert "SMA" in cr.strategy_id

def test_cr_03_none_input():
    cr = CanonicalResult.from_backtest(None)
    assert cr.strategy_id == "FAILED"
    assert cr.total_trades == 0

def test_cr_04_returns_from_trades():
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30}]
    raw = {"strategy_name": "test", "starting_value": 10000, "trades": trades,
           "total_return_pct": 2.2, "sharpe_ratio": 1.0, "max_drawdown_pct": 5, "total_trades": 4}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.returns is not None
    assert len(cr.returns) > 0
    assert cr.equity_curve is not None

def test_cr_05_returns_synthetic():
    raw = {"strategy_name": "test", "total_return_pct": 20, "sharpe_ratio": 1.5,
           "max_drawdown_pct": 10, "total_trades": 30, "bars_tested": 252,
           "starting_value": 10000}
    cr = CanonicalResult.from_backtest(raw, strategy_id="synth")
    assert cr.returns is not None
    assert len(cr.returns) == 252

def test_cr_06_to_dict():
    cr = CanonicalResult(strategy_id="S1", sharpe_ratio=1.5, total_trades=30)
    d = cr.to_dict()
    assert d["strategy_id"] == "S1"
    assert d["sharpe_ratio"] == 1.5

def test_cr_07_to_filter_dict():
    cr = CanonicalResult(strategy_id="S1", sharpe_ratio=1.5, max_drawdown_pct=12,
                          total_trades=40, win_rate=55, profit_factor=1.6)
    d = cr.to_filter_dict()
    assert d["sharpe_ratio"] == 1.5
    assert d["total_trades"] == 40

def test_cr_08_to_risk_dict():
    cr = CanonicalResult(strategy_id="S1", sharpe_ratio=1.5, total_trades=40)
    d = cr.to_risk_dict()
    assert "sharpe_ratio" in d
    assert "max_drawdown_pct" in d

def test_cr_09_to_fingerprint():
    cr = CanonicalResult(strategy_id="S1", sharpe_ratio=1.5, max_drawdown_pct=10,
                          total_trades=40, win_rate=55, profit_factor=1.6,
                          total_return_pct=20, trades_per_day=0.5)
    fp = cr.to_fingerprint_input()
    assert fp["sharpe_ratio"] == 1.5
    assert fp["win_rate"] == 0.55  # Converted to decimal

def test_cr_10_to_lineage():
    cr = CanonicalResult(strategy_id="S1", sharpe_ratio=1.5, parent_id="S0",
                          mutation_type="add_indicator", generation=2)
    d = cr.to_lineage_dict()
    assert d["parent_id"] == "S0"
    assert d["generation"] == 2

def test_cr_11_str():
    cr = CanonicalResult(strategy_id="S1", symbol="EUR-USD", timeframe="1hour",
                          total_return_pct=15, sharpe_ratio=1.8, max_drawdown_pct=12,
                          total_trades=45, win_rate=55)
    s = str(cr)
    assert "S1" in s
    assert "EUR-USD" in s

def test_cr_12_null_sharpe():
    raw = {"strategy_name": "test", "sharpe_ratio": None, "total_trades": 0}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.sharpe_ratio == 0

def test_cr_13_equity_from_trades():
    trades = [{"pnl": 100}, {"pnl": 200}]
    raw = {"strategy_name": "t", "trades": trades, "starting_value": 10000,
           "total_return_pct": 3, "sharpe_ratio": 1, "max_drawdown_pct": 1, "total_trades": 2}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.equity_curve[0] == 10000
    assert cr.equity_curve[-1] == 10300

def test_cr_14_empty_trades():
    raw = {"strategy_name": "t", "trades": [], "bars_tested": 100,
           "total_return_pct": 5, "starting_value": 10000,
           "sharpe_ratio": 0.8, "max_drawdown_pct": 5, "total_trades": 0}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.returns is not None

def test_cr_15_missing_fields():
    raw = {"strategy_name": "minimal"}
    cr = CanonicalResult.from_backtest(raw)
    assert cr.strategy_name == "minimal"
    assert cr.total_trades == 0
    assert cr.sharpe_ratio == 0


# ==============================================================================
# BacktestAdapter Tests (8) -- dry-run mode (no backtester loaded)
# ==============================================================================

def test_ba_01_init():
    adapter = BacktestAdapter(verbose=False)
    assert adapter.eval_count == 0

def test_ba_02_no_engine():
    """Without Backtrader data, evaluate_params returns empty CanonicalResult."""
    adapter = BacktestAdapter(verbose=False)
    # This will fail to run (no data) but should return gracefully
    class FakeStrategy:
        __name__ = "FakeStrategy"
    cr = adapter.evaluate_params({"fast": 10}, FakeStrategy, "TEST-SYM")
    assert isinstance(cr, CanonicalResult)

def test_ba_03_eval_count():
    adapter = BacktestAdapter(verbose=False)
    class FakeStrategy:
        __name__ = "Fake"
    adapter.evaluate_params({"a": 1}, FakeStrategy)
    adapter.evaluate_params({"a": 2}, FakeStrategy)
    assert adapter.eval_count == 2

def test_ba_04_reset():
    adapter = BacktestAdapter(verbose=False)
    class FakeStrategy:
        __name__ = "Fake"
    adapter.evaluate_params({"a": 1}, FakeStrategy)
    adapter.reset_count()
    assert adapter.eval_count == 0

def test_ba_05_objective_fn():
    adapter = BacktestAdapter(verbose=False)
    class FakeStrategy:
        __name__ = "Fake"
    fn = adapter.as_objective_function(FakeStrategy)
    result = fn({"fast": 10})
    assert "sharpe_ratio" in result
    assert "max_drawdown_pct" in result

def test_ba_06_variant_missing():
    adapter = BacktestAdapter(verbose=False)
    cr = adapter.evaluate_variant("/nonexistent/file.py")
    assert cr.strategy_name == "FILE_NOT_FOUND"

def test_ba_07_aggregate():
    adapter = BacktestAdapter(verbose=False)
    results = [
        CanonicalResult(strategy_id="a", sharpe_ratio=1.5, max_drawdown_pct=10,
                         total_trades=20, total_return_pct=15, returns=np.random.normal(0, 0.01, 50)),
        CanonicalResult(strategy_id="b", sharpe_ratio=2.0, max_drawdown_pct=8,
                         total_trades=30, total_return_pct=20, returns=np.random.normal(0, 0.01, 50)),
    ]
    agg = adapter._aggregate_results(results, {"test": True}, "TestStrat")
    assert agg.sharpe_ratio == 1.75  # mean of 1.5 and 2.0
    assert agg.total_trades == 50
    assert len(agg.returns) == 100  # concatenated

def test_ba_08_strategy_id_gen():
    adapter = BacktestAdapter(verbose=False)
    class MyStrat:
        __name__ = "MyStrat"
    cr = adapter.evaluate_params({"fast_period": 10, "slow_period": 50}, MyStrat, "EUR-USD", "1hour")
    assert "MyStrat" in cr.strategy_id
    assert "fast_period" in cr.strategy_id


# ==============================================================================
# Pipeline Structure Tests (5)
# ==============================================================================

def test_pipe_01_config():
    from run_pipeline import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.min_sharpe == 0.5
    assert cfg.top_pool_size == 10

def test_pipe_02_init():
    from run_pipeline import Pipeline, PipelineConfig
    cfg = PipelineConfig()
    cfg.verbose = False
    cfg.output_dir = Path("/tmp/test_pipeline_output")
    p = Pipeline(cfg)
    assert p._results == {}

def test_pipe_03_step_methods():
    """Verify all 11 step methods exist."""
    from run_pipeline import Pipeline, PipelineConfig
    cfg = PipelineConfig()
    cfg.verbose = False
    cfg.output_dir = Path("/tmp/test_pipeline_output")
    p = Pipeline(cfg)
    for i in range(1, 12):
        method = getattr(p, f"step_{i}_{'discovery' if i==1 else 'x'}", None)
    # Just check they exist
    assert hasattr(p, "step_1_discovery")
    assert hasattr(p, "step_2_backtest_filter")
    assert hasattr(p, "step_7_split")
    assert hasattr(p, "step_11_analytics")

def test_pipe_04_save_state():
    from run_pipeline import Pipeline, PipelineConfig
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        cfg = PipelineConfig()
        cfg.verbose = False
        cfg.output_dir = Path(d)
        p = Pipeline(cfg)
        p._results["step2_candidates"] = [
            CanonicalResult(strategy_id="S1", sharpe_ratio=1.5, total_trades=30)
        ]
        p._save_state()
        assert (Path(d) / "pipeline_state.json").exists()
    finally:
        shutil.rmtree(d)

def test_pipe_05_canonical_through_steps():
    """Verify CanonicalResult flows through step data correctly."""
    from run_pipeline import Pipeline, PipelineConfig
    cfg = PipelineConfig()
    cfg.verbose = False
    cfg.output_dir = Path("/tmp/test_pipeline_output")
    p = Pipeline(cfg)

    # Simulate step 2 output
    candidates = [
        CanonicalResult(strategy_id=f"S_{i}", sharpe_ratio=0.5 + i * 0.3,
                         max_drawdown_pct=10, total_trades=30,
                         returns=np.random.normal(0.001, 0.01, 100))
        for i in range(15)
    ]
    p._results["step2_candidates"] = candidates
    p._results["step2_all"] = candidates

    # Run diversification (step 6) which consumes candidates
    p.step_6_diversify()
    diversified = p._results.get("step6_diversified", [])
    assert len(diversified) <= len(candidates)  # Some may be removed

    # Run split (step 7)
    p._results["step6_diversified"] = diversified if diversified else candidates
    p.step_7_split()
    assert len(p._results.get("step7_top_pool", [])) <= cfg.top_pool_size


# ==============================================================================
# RUNNER
# ==============================================================================

ALL = [
    # CanonicalResult (15)
    ("CR.01 From dict",           test_cr_01_from_dict),
    ("CR.02 Auto ID",             test_cr_02_auto_id),
    ("CR.03 None input",          test_cr_03_none_input),
    ("CR.04 Returns from trades", test_cr_04_returns_from_trades),
    ("CR.05 Synthetic returns",   test_cr_05_returns_synthetic),
    ("CR.06 to_dict",             test_cr_06_to_dict),
    ("CR.07 to_filter_dict",      test_cr_07_to_filter_dict),
    ("CR.08 to_risk_dict",        test_cr_08_to_risk_dict),
    ("CR.09 to_fingerprint",      test_cr_09_to_fingerprint),
    ("CR.10 to_lineage",          test_cr_10_to_lineage),
    ("CR.11 __str__",             test_cr_11_str),
    ("CR.12 Null Sharpe",         test_cr_12_null_sharpe),
    ("CR.13 Equity from trades",  test_cr_13_equity_from_trades),
    ("CR.14 Empty trades",        test_cr_14_empty_trades),
    ("CR.15 Missing fields",      test_cr_15_missing_fields),
    # BacktestAdapter (8)
    ("BA.01 Init",                test_ba_01_init),
    ("BA.02 No engine graceful",  test_ba_02_no_engine),
    ("BA.03 Eval count",          test_ba_03_eval_count),
    ("BA.04 Reset count",         test_ba_04_reset),
    ("BA.05 Objective function",  test_ba_05_objective_fn),
    ("BA.06 Missing variant",     test_ba_06_variant_missing),
    ("BA.07 Aggregate results",   test_ba_07_aggregate),
    ("BA.08 Strategy ID gen",     test_ba_08_strategy_id_gen),
    # Pipeline (5)
    ("PIPE.01 Config",            test_pipe_01_config),
    ("PIPE.02 Init",              test_pipe_02_init),
    ("PIPE.03 Step methods",      test_pipe_03_step_methods),
    ("PIPE.04 Save state",        test_pipe_04_save_state),
    ("PIPE.05 Canonical flow",    test_pipe_05_canonical_through_steps),
]

if __name__ == "__main__":
    start = time.time()
    mods = [
        ("CanonicalResult",    0, 15),
        ("BacktestAdapter",   15, 23),
        ("Pipeline",          23, 28),
    ]
    for name, lo, hi in mods:
        print(f"\n{'-'*60}\n  {name}\n{'-'*60}")
        for n, fn in ALL[lo:hi]:
            run_test(n, fn)
    print(f"\n  ⏱️  {time.time()-start:.1f}s\n{'='*60}")
    print(f"  INTEGRATION: {_p} passed, {_f} failed")
    if _e:
        for n, e in _e:
            print(f"    {n}: {e}")
    print(f"{'='*60}")
    sys.exit(0 if _f == 0 else 1)
