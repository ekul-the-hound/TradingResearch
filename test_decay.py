# ==============================================================================
# test_decay.py
# ==============================================================================
# Tests for decay_calculator.py
#
# Run:
#     python test_decay.py
#
# Pattern matches test_system.py: returns (passed, failed) counts.
# ==============================================================================

import os
import sys
import tempfile
import shutil
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decay_calculator import (
    DecayCalculator,
    HARD_MIN_TOTAL_TRADES,
    BASELINE_FRAC,
    RECENT_FRAC,
)


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def _make_trade(pnl: float, exit_dt: datetime, dur_hours: float = 4.0) -> Dict[str, Any]:
    return {
        "entry_date": (exit_dt - timedelta(hours=dur_hours)).isoformat(),
        "exit_date": exit_dt.isoformat(),
        "pnl": pnl,
        "pnlcomm": pnl - 0.5,
        "size": 1.0,
        "return_pct": pnl / 100.0,
        "duration_bars": int(dur_hours),
        "is_long": pnl > 0,
    }


def _series(
    win_rate: float, n: int, start: datetime,
    avg_win: float = 100.0, avg_loss: float = 50.0,
    interval_hours: float = 6.0,
) -> List[Dict[str, Any]]:
    """Build a deterministic trade list matching a target win rate."""
    rng = random.Random(42 + n)
    n_wins = int(round(n * win_rate / 100.0))
    pnls = [avg_win] * n_wins + [-avg_loss] * (n - n_wins)
    rng.shuffle(pnls)
    trades = []
    t = start
    for p in pnls:
        trades.append(_make_trade(p, t))
        t += timedelta(hours=interval_hours)
    return trades


def _passed(name: str):
    print(f"  [PASS] {name}")
    return 1, 0


def _failed(name: str, msg: str):
    print(f"  [FAIL] {name}: {msg}")
    return 0, 1


# ------------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------------

def test_metrics_basic() -> Tuple[int, int]:
    print("\n[Test] compute_metrics basic")
    trades = _series(60.0, 100, datetime(2024, 1, 1))
    m = DecayCalculator.compute_metrics(trades)

    if m["trade_count"] != 100:
        return _failed("trade_count", f"got {m['trade_count']}")
    if not (55.0 <= m["win_rate"] <= 65.0):
        return _failed("win_rate", f"got {m['win_rate']}")
    if m["expectancy"] is None:
        return _failed("expectancy", "got None")
    if m["max_consecutive_losses"] < 1:
        return _failed("max_consecutive_losses", "expected >= 1")
    return _passed("compute_metrics basic")


def test_metrics_empty() -> Tuple[int, int]:
    print("\n[Test] compute_metrics empty")
    m = DecayCalculator.compute_metrics([])
    if m["trade_count"] != 0 or m["win_rate"] is not None:
        return _failed("empty", f"got {m}")
    return _passed("compute_metrics empty")


def test_scoring_standard_baseline() -> Tuple[int, int]:
    print("\n[Test] scoring -- identical baseline = 100")
    base = {"win_rate": 60.0, "trade_frequency": 2.0,
            "profit_factor": 1.5, "win_loss_ratio": 1.2,
            "max_consecutive_losses": 3, "avg_trade_duration_hours": 4.0,
            "expectancy": 25.0}
    scores = DecayCalculator.compute_decay_scores(base, dict(base))
    for k in ("win_rate", "trade_frequency", "profit_factor",
              "win_loss_ratio", "expectancy", "composite"):
        if scores[k] is None or abs(scores[k] - 100.0) > 0.01:
            return _failed("identical baseline", f"{k}={scores[k]}")
    return _passed("identical baseline scores 100")


def test_scoring_decay_detected() -> Tuple[int, int]:
    print("\n[Test] scoring -- decayed performance")
    base = {"win_rate": 60.0, "trade_frequency": 2.0,
            "profit_factor": 1.5, "win_loss_ratio": 1.2,
            "max_consecutive_losses": 3, "avg_trade_duration_hours": 4.0,
            "expectancy": 25.0}
    recent = {"win_rate": 30.0, "trade_frequency": 1.0,
              "profit_factor": 0.75, "win_loss_ratio": 0.6,
              "max_consecutive_losses": 6, "avg_trade_duration_hours": 8.0,
              "expectancy": 5.0}
    scores = DecayCalculator.compute_decay_scores(base, recent)
    if scores["composite"] is None or scores["composite"] >= 70.0:
        return _failed("decay composite", f"got {scores['composite']}")
    # inverted (lower better)
    if scores["max_consecutive_losses"] >= 100:
        return _failed("max_consec_losses inverted", "should drop below 100")
    if scores["avg_trade_duration"] >= 100:
        return _failed("avg_duration inverted", "should drop below 100")
    return _passed("decay correctly detected")


def test_expectancy_special_cases() -> Tuple[int, int]:
    print("\n[Test] scoring -- expectancy special handling")
    s = DecayCalculator._score_expectancy

    if s(-10.0, 5.0) != 110.0:
        return _failed("neg->pos", "expected 110.0")
    s1 = s(-10.0, -5.0)
    if s1 is None or s1 < 100.0:
        return _failed("both negative", f"got {s1}")
    s2 = s(10.0, -5.0)
    if s2 is None or s2 > 50.0:
        return _failed("pos->neg", f"got {s2}")
    s3 = s(10.0, 5.0)
    if s3 is None or abs(s3 - 50.0) > 0.01:
        return _failed("standard ratio", f"got {s3}")
    return _passed("expectancy special cases")


def test_score_clamping() -> Tuple[int, int]:
    print("\n[Test] scoring -- 0-110 clamp")
    s = DecayCalculator._score_standard(10.0, 1000.0)
    if s != 110.0:
        return _failed("high clamp", f"got {s}")
    s = DecayCalculator._score_standard(10.0, -50.0)
    if s != 0.0:
        return _failed("low clamp", f"got {s}")
    return _passed("0-110 clamp")


def test_status_classification() -> Tuple[int, int]:
    print("\n[Test] classify_status")
    C = DecayCalculator.classify_status
    cases = [(95.0, "excellent"), (75.0, "good"),
             (55.0, "warning"), (30.0, "poor"), (None, "unknown")]
    for val, expected in cases:
        got = C(val)
        if got != expected:
            return _failed(f"classify({val})", f"got {got}, expected {expected}")
    return _passed("classify_status")


def test_persistence_and_snapshot() -> Tuple[int, int]:
    print("\n[Test] end-to-end persistence + snapshot")
    tmp = tempfile.mkdtemp(prefix="decay_test_")
    try:
        db_path = os.path.join(tmp, "test.db")
        dc = DecayCalculator(db_path=db_path)

        # Build degrading trade history: strong early, weak late
        start = datetime(2024, 1, 1)
        baseline_trades = _series(70.0, 200, start)
        last = start + timedelta(hours=6 * 200)
        recent_trades = _series(35.0, 60, last)
        all_trades = baseline_trades + recent_trades

        n_saved = dc.save_trades("test_strat", "EUR-USD", all_trades)
        if n_saved != len(all_trades):
            return _failed("save_trades count", f"got {n_saved}")

        snap = dc.generate_snapshot("test_strat", "EUR-USD")
        if snap is None:
            return _failed("snapshot", "got None")
        if snap["total_trades"] != len(all_trades):
            return _failed("total_trades", f"got {snap['total_trades']}")
        if snap["decay_score_composite"] is None:
            return _failed("composite", "got None")
        if snap["decay_score_composite"] >= 90:
            return _failed("decay detected",
                f"composite should be < 90 for degrading strat, got {snap['decay_score_composite']}")

        # Retrieve
        snaps = dc.get_snapshots("test_strat")
        if len(snaps) != 1:
            return _failed("get_snapshots", f"got {len(snaps)}")

        latest = dc.latest_per_strategy()
        if len(latest) != 1:
            return _failed("latest_per_strategy", f"got {len(latest)}")

        return _passed("persistence + snapshot pipeline")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_insufficient_trades() -> Tuple[int, int]:
    print("\n[Test] insufficient trades returns None")
    tmp = tempfile.mkdtemp(prefix="decay_test_")
    try:
        dc = DecayCalculator(db_path=os.path.join(tmp, "t.db"))
        few = _series(50.0, HARD_MIN_TOTAL_TRADES - 1, datetime(2024, 1, 1))
        dc.save_trades("tiny", "EUR-USD", few)
        snap = dc.generate_snapshot("tiny", "EUR-USD")
        if snap is not None:
            return _failed("insufficient", "should be None")
        return _passed("insufficient trades guarded")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_windowing_fractions() -> Tuple[int, int]:
    print("\n[Test] windowing splits at correct fractions")
    tmp = tempfile.mkdtemp(prefix="decay_test_")
    try:
        dc = DecayCalculator(db_path=os.path.join(tmp, "t.db"))
        trades = _series(60.0, 1000, datetime(2024, 1, 1))
        dc.save_trades("wnd", "EUR-USD", trades)
        snap = dc.generate_snapshot("wnd", "EUR-USD")
        expected_base = int(1000 * BASELINE_FRAC)
        expected_rec = int(1000 * RECENT_FRAC)
        if snap["baseline_trade_count"] != expected_base:
            return _failed("baseline_n",
                f"got {snap['baseline_trade_count']}, expected {expected_base}")
        if snap["rolling_trade_count"] != expected_rec:
            return _failed("recent_n",
                f"got {snap['rolling_trade_count']}, expected {expected_rec}")
        return _passed("windowing fractions correct")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_replace_trades() -> Tuple[int, int]:
    print("\n[Test] save_trades(replace=True) overwrites")
    tmp = tempfile.mkdtemp(prefix="decay_test_")
    try:
        dc = DecayCalculator(db_path=os.path.join(tmp, "t.db"))
        t1 = _series(50.0, 30, datetime(2024, 1, 1))
        dc.save_trades("s", "EUR-USD", t1)
        if len(dc.get_trades("s", "EUR-USD")) != 30:
            return _failed("initial count", "expected 30")

        t2 = _series(60.0, 20, datetime(2024, 6, 1))
        dc.save_trades("s", "EUR-USD", t2, replace=True)
        if len(dc.get_trades("s", "EUR-USD")) != 20:
            return _failed("replaced count", "expected 20")
        return _passed("replace=True works")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------
# RUNNER
# ------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Edge Decay Calculator -- Test Suite")
    print("=" * 70)

    tests = [
        test_metrics_basic,
        test_metrics_empty,
        test_scoring_standard_baseline,
        test_scoring_decay_detected,
        test_expectancy_special_cases,
        test_score_clamping,
        test_status_classification,
        test_persistence_and_snapshot,
        test_insufficient_trades,
        test_windowing_fractions,
        test_replace_trades,
    ]

    passed = failed = 0
    for t in tests:
        try:
            p, f = t()
            passed += p; failed += f
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
