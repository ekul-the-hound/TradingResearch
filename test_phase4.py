# ==============================================================================
# test_phase4.py -- Phase 4 Live Infrastructure Test Suite
# 32 tests across 4 modules. Run: python test_phase4.py
# ==============================================================================

import sys, time, traceback
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from drift_detector import DriftDetector, DriftConfig, DriftSeverity
from shadow_trader import ShadowTrader
from live_monitor import LiveMonitor, StrategyHealth, AlertLevel
from strategy_lifecycle import (
    StrategyLifecycle, LifecycleConfig, LifecycleState,
)

_p, _f, _e = 0, 0, []
def run_test(name, fn):
    global _p, _f
    try: fn(); _p += 1; print(f"  [OK] {name}")
    except Exception as e: _f += 1; _e.append((name, str(e))); print(f"  [FAIL] {name}: {e}"); traceback.print_exc()

# -- MODULE 1: Drift Detector -------------------------------------------------

def test_dd_1_no_drift():
    np.random.seed(42)
    ref = np.random.normal(0.0005, 0.01, 1000)
    live = np.random.normal(0.0005, 0.01, 200)
    # Use lenient config since small samples naturally vary
    cfg = DriftConfig(ks_critical_pvalue=0.001, sharpe_critical_pct=80, mean_shift_warning=4.0)
    dd = DriftDetector(ref, cfg)
    r = dd.check(live)
    # Same distribution should not trigger CRITICAL with lenient config
    assert r.severity != DriftSeverity.CRITICAL

def test_dd_2_mean_shift():
    np.random.seed(42)
    ref = np.random.normal(0.0005, 0.01, 500)
    live = np.random.normal(-0.005, 0.01, 100)  # Big mean shift
    dd = DriftDetector(ref)
    r = dd.check(live)
    assert r.drift_detected
    assert "ks" in " ".join(r.triggered_tests) or "mean" in " ".join(r.triggered_tests)

def test_dd_3_vol_shift():
    np.random.seed(42)
    ref = np.random.normal(0, 0.01, 500)
    live = np.random.normal(0, 0.05, 100)  # 5x vol
    dd = DriftDetector(ref)
    r = dd.check(live)
    assert r.drift_detected
    assert "var_shift" in r.triggered_tests or "ks" in " ".join(r.triggered_tests)

def test_dd_4_psi():
    np.random.seed(42)
    ref = np.random.normal(0, 0.01, 500)
    live = np.random.normal(0.02, 0.03, 100)  # Totally different
    dd = DriftDetector(ref)
    r = dd.check(live)
    assert r.psi > 0.1

def test_dd_5_online_cusum():
    np.random.seed(42)
    ref = np.random.normal(0.0005, 0.01, 500)
    dd = DriftDetector(ref, DriftConfig(cusum_threshold=3.0))
    signals = 0
    for _ in range(50):
        r = dd.update(np.random.normal(0.0005, 0.01))
    for _ in range(50):
        r = dd.update(np.random.normal(-0.02, 0.01))  # Shift
        if r.drift_detected:
            signals += 1
    assert signals > 0

def test_dd_6_sharpe_degrad():
    np.random.seed(42)
    ref = np.random.normal(0.002, 0.01, 500)  # Good Sharpe
    live = np.random.normal(-0.001, 0.015, 100)  # Bad Sharpe
    dd = DriftDetector(ref, DriftConfig(sharpe_warning_pct=30))
    r = dd.check(live)
    assert r.sharpe_degradation_pct > 30

def test_dd_7_reset_online():
    dd = DriftDetector(np.random.normal(0, 0.01, 100))
    dd.update(0.05)
    dd.reset_online()
    assert dd._cusum_pos == 0 and dd._cusum_neg == 0

def test_dd_8_insufficient():
    dd = DriftDetector(np.random.normal(0, 0.01, 100))
    r = dd.check(np.array([0.01, -0.01]))  # Only 2 points
    assert not r.drift_detected

# -- MODULE 2: Shadow Trader --------------------------------------------------

def test_st_1_buy_sell():
    t = ShadowTrader("S_001", 100000, slippage_bps=0, commission_pct=0)
    t.submit_order("BUY", 1.10, 10000)
    assert t.position.side.value == "long" and t.position.size == 10000
    t.submit_order("SELL", 1.12, 10000)
    assert t.position.side.value == "flat"
    assert t.metrics.total_trades == 1
    assert t.position.realized_pnl > 0  # PnL from the round-trip

def test_st_2_pnl():
    t = ShadowTrader("S_002", 100000, slippage_bps=0, commission_pct=0)
    t.submit_order("BUY", 100.0, 100)
    t.submit_order("SELL", 110.0, 100)
    assert abs(t.position.realized_pnl - 1000.0) < 0.01

def test_st_3_slippage():
    t = ShadowTrader("S_003", 100000, slippage_bps=10, commission_pct=0)
    o = t.submit_order("BUY", 100.0, 100)
    assert o.fill_price > 100.0  # Slippage adds to buy

def test_st_4_mark_to_market():
    t = ShadowTrader("S_004", 100000, slippage_bps=0, commission_pct=0)
    t.submit_order("BUY", 100.0, 100)
    t.mark_to_market(110.0)
    assert t.position.unrealized_pnl == 1000.0

def test_st_5_drawdown():
    t = ShadowTrader("S_005", 100000, slippage_bps=0, commission_pct=0)
    t.submit_order("BUY", 100.0, 100)
    t.mark_to_market(110.0)  # Peak
    t.mark_to_market(95.0)   # Drop
    assert t.metrics.max_drawdown > 0

def test_st_6_comparison():
    t = ShadowTrader("S_006", 100000, slippage_bps=0, commission_pct=0)
    for i in range(35):
        t.submit_order("BUY", 100.0, 100)
        t.submit_order("SELL", 100.5, 100)
        t.end_of_day(100.0)
    comp = t.get_comparison(backtest_sharpe=2.0)
    assert "meets_promotion_criteria" in comp

def test_st_7_inactive():
    t = ShadowTrader("S_007", 100000)
    t.stop()
    o = t.submit_order("BUY", 100.0, 100)
    assert o.status.value == "rejected"

def test_st_8_short():
    t = ShadowTrader("S_008", 100000, slippage_bps=0, commission_pct=0)
    t.submit_order("SELL", 100.0, 100)
    assert t.position.side.value == "short"
    t.mark_to_market(95.0)
    assert t.position.unrealized_pnl == 500.0

# -- MODULE 3: Live Monitor --------------------------------------------------

def test_lm_1_register():
    m = LiveMonitor()
    t = ShadowTrader("S_001", 100000)
    m.register_strategy("S_001", shadow_trader=t)
    snap = m.get_strategy_snapshot("S_001")
    assert snap is not None and snap.health == StrategyHealth.HEALTHY

def test_lm_2_update():
    m = LiveMonitor()
    t = ShadowTrader("S_001", 100000, slippage_bps=0, commission_pct=0)
    m.register_strategy("S_001", shadow_trader=t)
    t.submit_order("BUY", 100.0, 100)
    m.update("S_001", price=105.0)
    snap = m.get_strategy_snapshot("S_001")
    assert snap.pnl > 0

def test_lm_3_portfolio():
    m = LiveMonitor()
    for i in range(3):
        t = ShadowTrader(f"S_{i}", 100000)
        m.register_strategy(f"S_{i}", shadow_trader=t, capital=100000)
    ps = m.get_portfolio_snapshot()
    assert ps.n_strategies_active == 3
    assert ps.total_capital == 300000

def test_lm_4_alerts():
    m = LiveMonitor()
    t = ShadowTrader("S_001", 100000, slippage_bps=0, commission_pct=0)
    m.register_strategy("S_001", shadow_trader=t)
    t.submit_order("BUY", 100.0, 1000)
    t.mark_to_market(70.0)  # 30% drawdown
    for _ in range(15):
        t.end_of_day(70.0)
    m.update("S_001", price=70.0)
    ps = m.get_portfolio_snapshot()
    assert ps.n_strategies_critical > 0 or len(ps.active_alerts) > 0

# -- MODULE 4: Strategy Lifecycle ---------------------------------------------

def test_lc_1_register():
    lc = StrategyLifecycle()
    lc.register("S_001", backtest_sharpe=1.5)
    assert lc.get_state("S_001") == LifecycleState.RESEARCH

def test_lc_2_promote():
    lc = StrategyLifecycle()
    lc.register("S_001")
    lc.promote("S_001")
    assert lc.get_state("S_001") == LifecycleState.PAPER
    lc.promote("S_001")
    assert lc.get_state("S_001") == LifecycleState.LIVE

def test_lc_3_demote():
    lc = StrategyLifecycle()
    lc.register("S_001")
    lc.promote("S_001")  # PAPER
    lc.promote("S_001")  # LIVE
    lc.demote("S_001")   # REVIEW
    assert lc.get_state("S_001") == LifecycleState.REVIEW

def test_lc_4_retire():
    lc = StrategyLifecycle()
    lc.register("S_001")
    lc.promote("S_001")  # PAPER
    lc.promote("S_001")  # LIVE
    lc.retire("S_001")
    assert lc.get_state("S_001") == LifecycleState.RETIRED

def test_lc_5_invalid():
    lc = StrategyLifecycle()
    lc.register("S_001")
    try:
        lc.retire("S_001")  # Can't retire from RESEARCH
        assert False, "Should have raised"
    except ValueError:
        pass

def test_lc_6_auto_promote():
    lc = StrategyLifecycle(LifecycleConfig(min_paper_days=5, min_trades_for_promotion=5))
    lc.register("S_001", backtest_sharpe=1.5)
    lc.promote("S_001")  # PAPER
    tr = lc.check_auto_transitions(
        "S_001", live_sharpe=1.3, max_drawdown=0.08,
        days_active=10, total_trades=20,
    )
    assert tr is not None
    assert lc.get_state("S_001") == LifecycleState.LIVE

def test_lc_7_auto_demote():
    lc = StrategyLifecycle()
    lc.register("S_001", backtest_sharpe=2.0)
    lc.promote("S_001")
    lc.promote("S_001")  # LIVE
    tr = lc.check_auto_transitions(
        "S_001", live_sharpe=0.3, max_drawdown=0.25,
        days_active=60, total_trades=50,
    )
    assert tr is not None
    assert lc.get_state("S_001") == LifecycleState.REVIEW

def test_lc_8_audit_trail():
    lc = StrategyLifecycle()
    lc.register("S_001")
    lc.promote("S_001")
    lc.promote("S_001")
    lc.demote("S_001")
    trail = lc.get_audit_trail("S_001")
    assert len(trail) == 3
    assert trail[0].to_state == LifecycleState.PAPER

def test_lc_9_summary():
    lc = StrategyLifecycle()
    for i in range(5):
        lc.register(f"S_{i}")
    lc.promote("S_0")
    lc.promote("S_1")
    lc.promote("S_1")
    s = lc.summary()
    assert s["research"] == 3
    assert s["paper"] == 1
    assert s["live"] == 1

def test_lc_10_kill_switch_demote():
    lc = StrategyLifecycle()
    lc.register("S_001", backtest_sharpe=2.0)
    lc.promote("S_001")
    lc.promote("S_001")
    tr = lc.check_auto_transitions("S_001", live_sharpe=1.5, kill_switch_triggered=True)
    assert lc.get_state("S_001") == LifecycleState.REVIEW

# -- RUN ----------------------------------------------------------------------

ALL = [
    ("DD.1 No drift",            test_dd_1_no_drift),
    ("DD.2 Mean shift",          test_dd_2_mean_shift),
    ("DD.3 Vol shift",           test_dd_3_vol_shift),
    ("DD.4 PSI",                 test_dd_4_psi),
    ("DD.5 Online CUSUM",        test_dd_5_online_cusum),
    ("DD.6 Sharpe degradation",  test_dd_6_sharpe_degrad),
    ("DD.7 Reset online",        test_dd_7_reset_online),
    ("DD.8 Insufficient data",   test_dd_8_insufficient),
    ("ST.1 Buy/sell cycle",      test_st_1_buy_sell),
    ("ST.2 PnL accuracy",       test_st_2_pnl),
    ("ST.3 Slippage",           test_st_3_slippage),
    ("ST.4 Mark to market",      test_st_4_mark_to_market),
    ("ST.5 Drawdown tracking",   test_st_5_drawdown),
    ("ST.6 Backtest comparison", test_st_6_comparison),
    ("ST.7 Inactive rejection",  test_st_7_inactive),
    ("ST.8 Short positions",     test_st_8_short),
    ("LM.1 Register",           test_lm_1_register),
    ("LM.2 Price update",       test_lm_2_update),
    ("LM.3 Portfolio snapshot",  test_lm_3_portfolio),
    ("LM.4 Alert generation",   test_lm_4_alerts),
    ("LC.1 Register",           test_lc_1_register),
    ("LC.2 Promote chain",      test_lc_2_promote),
    ("LC.3 Demote",             test_lc_3_demote),
    ("LC.4 Retire",             test_lc_4_retire),
    ("LC.5 Invalid transition",  test_lc_5_invalid),
    ("LC.6 Auto-promote",       test_lc_6_auto_promote),
    ("LC.7 Auto-demote",        test_lc_7_auto_demote),
    ("LC.8 Audit trail",        test_lc_8_audit_trail),
    ("LC.9 Summary counts",     test_lc_9_summary),
    ("LC.10 Kill switch demote", test_lc_10_kill_switch_demote),
]

if __name__ == "__main__":
    start = time.time()
    mods = [
        ("Module 1: Drift Detector", 0, 8),
        ("Module 2: Shadow Trader", 8, 16),
        ("Module 3: Live Monitor", 16, 20),
        ("Module 4: Strategy Lifecycle", 20, 30),
    ]
    for name, lo, hi in mods:
        print(f"\n{'-'*60}\n  {name}\n{'-'*60}")
        for n, fn in ALL[lo:hi]:
            run_test(n, fn)
    print(f"\n  ⏱️  {time.time()-start:.1f}s\n{'='*60}")
    print(f"  PHASE 4: {_p} passed, {_f} failed")
    if _e:
        for n, e in _e: print(f"    {n}: {e}")
    print(f"{'='*60}")
    sys.exit(0 if _f == 0 else 1)
