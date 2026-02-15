# ==============================================================================
# test_phase3.py — Phase 3 Risk & Validation Test Suite
# 35 tests across 5 modules. Run: python test_phase3.py
# ==============================================================================

import sys, time, traceback
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from market_impact import MarketImpactModel
from capacity_model import CapacityEstimator
from kill_switch import KillSwitch, KillSwitchConfig, KillAction
from liquidity_stress import LiquidityStressTest, StressScenario
from tail_risk import TailRiskAnalyzer

_p, _f, _e = 0, 0, []

def run_test(name, fn):
    global _p, _f
    try: fn(); _p += 1; print(f"  ✅ {name}")
    except Exception as e: _f += 1; _e.append((name, str(e))); print(f"  ❌ {name}: {e}"); traceback.print_exc()

# ── MODULE 1: Market Impact ──────────────────────────────────────────────────

def test_mi_1_sqrt():
    r = MarketImpactModel("sqrt").estimate(10000, 1_000_000, 0.02)
    assert r.total_impact_bps > 0 and r.participation_rate == 0.01

def test_mi_2_scaling():
    m = MarketImpactModel()
    assert m.estimate(100_000, 1e6, 0.02).total_impact_bps > m.estimate(1000, 1e6, 0.02).total_impact_bps

def test_mi_3_all_models():
    m = MarketImpactModel()
    for model in ["sqrt", "almgren", "linear", "kyle"]:
        r = m.estimate(10000, 1e6, 0.02, model=model)
        assert r.total_impact_bps > 0 and r.model == model

def test_mi_4_roundtrip():
    m = MarketImpactModel()
    assert m.estimate_roundtrip(10000, 1e6, 0.02).total_impact_bps > m.estimate(10000, 1e6, 0.02).total_impact_bps

def test_mi_5_execution():
    plan = MarketImpactModel().optimal_execution(50000, 10, 1e6, 0.02)
    assert plan.n_slices == 10 and abs(sum(plan.slice_sizes) - 50000) < 1

def test_mi_6_batch():
    trades = [{"size": s, "pnl": p, "entry_price": 1.1} for s, p in [(5000,100),(10000,-50),(2000,30)]]
    adj, summ = MarketImpactModel().adjust_backtest(trades, 500_000)
    assert len(adj) == 3 and all("impact_cost" in a for a in adj)

def test_mi_7_zero_vol():
    r = MarketImpactModel().estimate(10000, 1e6, 0.0)
    assert r.total_impact_bps >= 0  # spread still costs

# ── MODULE 2: Capacity Model ─────────────────────────────────────────────────

def test_cap_1_volume():
    r = CapacityEstimator(max_participation=0.05).estimate(
        {"sharpe_ratio": 1.5, "total_trades": 252}, daily_volume=1e6, price=1.0, method="volume")
    assert r.max_aum > 0 and r.method == "volume"

def test_cap_2_impact():
    r = CapacityEstimator(min_sharpe_threshold=0.5).estimate(
        {"sharpe_ratio": 2.0, "total_trades": 100, "total_return_pct": 30, "avg_trade_pct": 0.3},
        daily_volume=1e6, volatility=0.02, method="impact")
    assert r.max_aum > 0 and r.sharpe_at_max >= 0.5

def test_cap_3_strong_beats_weak():
    cap = CapacityEstimator(min_sharpe_threshold=0.5)
    weak = cap.estimate({"sharpe_ratio": 0.8, "total_trades": 100, "total_return_pct": 10, "avg_trade_pct": 0.1}, 1e6, 0.02)
    strong = cap.estimate({"sharpe_ratio": 2.5, "total_trades": 100, "total_return_pct": 40, "avg_trade_pct": 0.4}, 1e6, 0.02)
    assert strong.max_aum >= weak.max_aum

def test_cap_4_regime():
    r = CapacityEstimator().estimate_by_regime(
        {"sharpe_ratio": 1.5, "total_trades": 100, "total_return_pct": 20, "avg_trade_pct": 0.2}, 1e6, 0.02)
    assert r.crisis.max_aum <= r.normal.max_aum and r.conservative <= r.normal.max_aum

# ── MODULE 3: Kill Switch ────────────────────────────────────────────────────

def test_ks_1_clear():
    r = KillSwitch().check(current_pnl=100, account_size=100000, drawdown_pct=2)
    assert not r.triggered and r.action == KillAction.NONE

def test_ks_2_daily():
    r = KillSwitch(KillSwitchConfig(max_daily_loss_pct=3.0)).check(daily_loss_pct=4.0)
    assert r.triggered and r.action == KillAction.HALT_DAY

def test_ks_3_dd_tiers():
    cfg = KillSwitchConfig(reduce_drawdown_pct=10, halt_drawdown_pct=20, liquidate_drawdown_pct=30)
    ks = KillSwitch(cfg)
    assert ks.check(drawdown_pct=12).action == KillAction.REDUCE
    ks.reset()
    assert ks.check(drawdown_pct=22).action == KillAction.HALT
    ks.reset()
    assert ks.check(drawdown_pct=35).action == KillAction.LIQUIDATE

def test_ks_4_streak():
    ks = KillSwitch(KillSwitchConfig(max_consecutive_losses=8, review_consecutive_losses=5))
    assert ks.check(consecutive_losses=6).action == KillAction.REVIEW
    ks.reset()
    assert ks.check(consecutive_losses=10).action == KillAction.HALT

def test_ks_5_sharpe():
    r = KillSwitch(KillSwitchConfig(sharpe_degradation_pct=50)).check(live_sharpe=0.4, backtest_sharpe=1.5)
    assert r.triggered  # 73% degradation

def test_ks_6_can_trade():
    ks = KillSwitch()
    assert ks.can_trade()
    ks.check(drawdown_pct=25)
    assert not ks.can_trade() and ks.position_size_multiplier() == 0.0

def test_ks_7_ftmo():
    r = KillSwitch(KillSwitchConfig(ftmo_mode=True)).check(daily_loss_pct=5.5, drawdown_pct=3)
    assert r.triggered and "ftmo" in r.rule_name.lower()

def test_ks_8_reset():
    ks = KillSwitch()
    ks.check(drawdown_pct=25)
    ks.reset()
    assert ks.can_trade() and ks.position_size_multiplier() == 1.0

def test_ks_9_vol_spike():
    r = KillSwitch(KillSwitchConfig(vol_spike_mult=3.0)).check(current_vol=0.06, normal_vol=0.015)
    assert r.triggered and r.action == KillAction.REDUCE

# ── MODULE 4: Liquidity Stress ───────────────────────────────────────────────

def test_ls_1_all():
    np.random.seed(42)
    result = LiquidityStressTest().run_all(returns=np.random.normal(0.0003, 0.01, 500))
    assert len(result.scenarios) == 6 and result.overall_survival_rate >= 0

def test_ls_2_flash():
    np.random.seed(42)
    stress = LiquidityStressTest(scenarios=[
        StressScenario("flash_crash", "test", volume_multiplier=0.1,
                       slippage_multiplier=10, return_shock_pct=-5, vol_multiplier=5)])
    result = stress.run_all(returns=np.random.normal(0.001, 0.008, 300))
    assert result.scenarios[0].stressed_pnl < result.scenarios[0].original_pnl

def test_ls_3_trades():
    trades = [{"size": s, "pnl": p, "entry_price": 1.1} for s, p in [(10000,100),(5000,-50),(8000,80)]]
    result = LiquidityStressTest().run_all(trades=trades, daily_volume=500_000)
    assert len(result.scenarios) == 6

def test_ls_4_custom():
    stress = LiquidityStressTest()
    stress.add_scenario(StressScenario("custom", "test", volume_multiplier=0.2, return_shock_pct=-10))
    assert len(stress.scenarios) == 7

# ── MODULE 5: Tail Risk ──────────────────────────────────────────────────────

def test_tr_1_basic():
    np.random.seed(42)
    result = TailRiskAnalyzer().analyze(np.random.normal(0.0003, 0.01, 1000))
    assert result.var_95 < 0 and result.cvar_95 <= result.var_95

def test_tr_2_fat():
    np.random.seed(42)
    result = TailRiskAnalyzer().analyze(np.random.standard_t(3, 1000) * 0.01)
    assert result.kurtosis > 1 and result.tail_risk_score > 0

def test_tr_3_skewed():
    np.random.seed(42)
    r = -np.abs(np.random.normal(0, 0.01, 1000)) + 0.002
    assert TailRiskAnalyzer().analyze(r).skewness < 0

def test_tr_4_evt():
    np.random.seed(42)
    r = np.random.normal(0.0003, 0.01, 2000)
    r[np.random.choice(len(r), 20)] = -0.05
    result = TailRiskAnalyzer().analyze(r)
    assert result.evt_shape != 0 or result.evt_scale > 0

def test_tr_5_dependence():
    np.random.seed(42)
    r1 = np.random.normal(0, 0.01, 500)
    r2 = r1 * 0.8 + np.random.normal(0, 0.005, 500)
    assert TailRiskAnalyzer().analyze(r1, r2).lower_tail_coef > 0

def test_tr_6_dar():
    np.random.seed(42)
    result = TailRiskAnalyzer().analyze(np.random.normal(0.0005, 0.012, 500))
    assert result.max_drawdown > 0 and result.dar_95 > 0

def test_tr_7_short():
    result = TailRiskAnalyzer().analyze(np.array([0.01, -0.01]))
    assert result.tail_risk_score == 0  # Too short

# ── RUN ──────────────────────────────────────────────────────────────────────

ALL = [
    ("MI.1 Square-root impact",       test_mi_1_sqrt),
    ("MI.2 Size scaling",             test_mi_2_scaling),
    ("MI.3 All models",               test_mi_3_all_models),
    ("MI.4 Round-trip",               test_mi_4_roundtrip),
    ("MI.5 Execution plan",           test_mi_5_execution),
    ("MI.6 Batch adjust",             test_mi_6_batch),
    ("MI.7 Zero volatility",          test_mi_7_zero_vol),
    ("CAP.1 Volume method",           test_cap_1_volume),
    ("CAP.2 Impact method",           test_cap_2_impact),
    ("CAP.3 Strong > weak",           test_cap_3_strong_beats_weak),
    ("CAP.4 Regime capacity",         test_cap_4_regime),
    ("KS.1 All clear",                test_ks_1_clear),
    ("KS.2 Daily loss halt",          test_ks_2_daily),
    ("KS.3 Drawdown tiers",           test_ks_3_dd_tiers),
    ("KS.4 Streak review/halt",       test_ks_4_streak),
    ("KS.5 Sharpe degradation",       test_ks_5_sharpe),
    ("KS.6 Can trade gate",           test_ks_6_can_trade),
    ("KS.7 FTMO compliance",          test_ks_7_ftmo),
    ("KS.8 Reset state",              test_ks_8_reset),
    ("KS.9 Vol spike",                test_ks_9_vol_spike),
    ("LS.1 All scenarios",            test_ls_1_all),
    ("LS.2 Flash crash",              test_ls_2_flash),
    ("LS.3 Trade-level stress",       test_ls_3_trades),
    ("LS.4 Custom scenario",          test_ls_4_custom),
    ("TR.1 Basic VaR/CVaR",           test_tr_1_basic),
    ("TR.2 Fat tails",                test_tr_2_fat),
    ("TR.3 Skewed returns",           test_tr_3_skewed),
    ("TR.4 EVT GPD fit",              test_tr_4_evt),
    ("TR.5 Tail dependence",          test_tr_5_dependence),
    ("TR.6 Drawdown-at-Risk",         test_tr_6_dar),
    ("TR.7 Short series",             test_tr_7_short),
]

if __name__ == "__main__":
    start = time.time()
    mods = [
        ("Module 1: Market Impact", 0, 7),
        ("Module 2: Capacity Model", 7, 11),
        ("Module 3: Kill Switch", 11, 20),
        ("Module 4: Liquidity Stress", 20, 24),
        ("Module 5: Tail Risk", 24, 31),
    ]
    for name, lo, hi in mods:
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")
        for n, fn in ALL[lo:hi]:
            run_test(n, fn)
    print(f"\n  ⏱️  {time.time()-start:.1f}s\n{'='*60}")
    print(f"  PHASE 3: {_p} passed, {_f} failed")
    if _e:
        for n, e in _e: print(f"    {n}: {e}")
    print(f"{'='*60}")
    sys.exit(0 if _f == 0 else 1)
