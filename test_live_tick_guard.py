# ==============================================================================
# test_live_tick_guard.py -- Tests for the live bad-tick filter / spread guard
# ==============================================================================
# Convention (project-wide): import failures are HARD errors, not skips.
# ==============================================================================

import math
import unittest
from dataclasses import dataclass

from live_tick_guard import LiveTickGuard, TickGuardConfig, TickVerdict


@dataclass
class Tick:
    """Minimal stand-in for broker_base.BrokerTick."""
    symbol: str
    bid: float
    ask: float
    last: float = 0.0


def clean(symbol="EURUSD", bid=1.10000, ask=1.10002):
    return Tick(symbol, bid, ask)


class TestStructuralValidity(unittest.TestCase):
    def setUp(self):
        self.g = LiveTickGuard(TickGuardConfig())

    def test_clean_tick_passes(self):
        v = self.g.check(clean())
        self.assertTrue(v.ok)
        self.assertEqual(v.reasons, [])
        self.assertAlmostEqual(v.mid, 1.10001, places=6)

    def test_zero_bid_rejected(self):
        v = self.g.check(Tick("EURUSD", 0.0, 1.10002))
        self.assertFalse(v.ok)
        self.assertTrue(any("bid" in r for r in v.reasons))

    def test_zero_ask_rejected(self):
        v = self.g.check(Tick("EURUSD", 1.10000, 0.0))
        self.assertFalse(v.ok)
        self.assertTrue(any("ask" in r for r in v.reasons))

    def test_negative_price_rejected(self):
        v = self.g.check(Tick("EURUSD", -1.1, 1.10002))
        self.assertFalse(v.ok)

    def test_nan_price_rejected(self):
        v = self.g.check(Tick("EURUSD", float("nan"), 1.10002))
        self.assertFalse(v.ok)

    def test_inf_price_rejected(self):
        v = self.g.check(Tick("EURUSD", 1.10000, float("inf")))
        self.assertFalse(v.ok)

    def test_both_sides_bad_rejected(self):
        v = self.g.check(Tick("EURUSD", 0.0, 0.0))
        self.assertFalse(v.ok)
        self.assertTrue(any("no valid bid or ask" in r for r in v.reasons))


class TestMarketStructure(unittest.TestCase):
    def setUp(self):
        self.g = LiveTickGuard(TickGuardConfig())

    def test_crossed_market_rejected(self):
        v = self.g.check(Tick("EURUSD", 1.10010, 1.10005))
        self.assertFalse(v.ok)
        self.assertTrue(any("crossed" in r for r in v.reasons))

    def test_locked_market_rejected(self):
        v = self.g.check(Tick("EURUSD", 1.10000, 1.10000))
        self.assertFalse(v.ok)
        self.assertTrue(any("locked" in r for r in v.reasons))


class TestSpreadGuard(unittest.TestCase):
    def setUp(self):
        self.g = LiveTickGuard(TickGuardConfig(max_spread_bps=5.0))

    def test_narrow_spread_passes(self):
        # ~1.8 bps
        self.assertTrue(self.g.check(Tick("EURUSD", 1.10000, 1.10002)).ok)

    def test_wide_spread_rejected(self):
        # 0.00100 / ~1.10 * 1e4 ~= 9 bps > 5 bps limit
        v = self.g.check(Tick("EURUSD", 1.10000, 1.10100))
        self.assertFalse(v.ok)
        self.assertTrue(any("spread" in r for r in v.reasons))

    def test_spread_check_disabled(self):
        g = LiveTickGuard(TickGuardConfig(max_spread_bps=0.0))
        self.assertTrue(g.check(Tick("EURUSD", 1.10000, 1.10100)).ok)


class TestStaleness(unittest.TestCase):
    def test_frozen_feed_rejected_after_threshold(self):
        g = LiveTickGuard(TickGuardConfig(max_frozen_ticks=3, outlier_sigma=0))
        t = Tick("EURUSD", 1.10000, 1.10002)
        # First tick establishes baseline.
        self.assertTrue(g.check(t).ok)
        # Next identical mids: counts 1, 2, 3 -> reject at 3.
        self.assertTrue(g.check(Tick("EURUSD", 1.10000, 1.10002)).ok)   # count 1
        self.assertTrue(g.check(Tick("EURUSD", 1.10000, 1.10002)).ok)   # count 2
        v = g.check(Tick("EURUSD", 1.10000, 1.10002))                   # count 3
        self.assertFalse(v.ok)
        self.assertTrue(any("stale" in r for r in v.reasons))

    def test_moving_feed_never_stale(self):
        g = LiveTickGuard(TickGuardConfig(max_frozen_ticks=3, outlier_sigma=0))
        for i in range(10):
            bid = 1.10000 + i * 0.00001
            self.assertTrue(g.check(Tick("EURUSD", bid, bid + 0.00002)).ok)

    def test_staleness_disabled(self):
        g = LiveTickGuard(TickGuardConfig(max_frozen_ticks=0, outlier_sigma=0))
        for _ in range(50):
            self.assertTrue(g.check(Tick("EURUSD", 1.10000, 1.10002)).ok)


class TestOutlier(unittest.TestCase):
    def _prime(self, g, n=25, base=1.10000):
        # Feed n slightly-varying clean ticks to build a stable median.
        for i in range(n):
            jitter = (i % 5) * 0.00001
            bid = base + jitter
            v = g.check(Tick("EURUSD", bid, bid + 0.00002))
            assert v.ok, v.reasons

    def test_outlier_rejected(self):
        g = LiveTickGuard(TickGuardConfig(outlier_sigma=8.0,
                                          min_history_for_outlier=20,
                                          max_frozen_ticks=0))
        self._prime(g)
        # A print 200 pips away is a gross outlier.
        v = g.check(Tick("EURUSD", 1.12000, 1.12002))
        self.assertFalse(v.ok)
        self.assertTrue(any("outlier" in r for r in v.reasons))

    def test_no_outlier_check_before_min_history(self):
        g = LiveTickGuard(TickGuardConfig(outlier_sigma=8.0,
                                          min_history_for_outlier=20,
                                          max_frozen_ticks=0))
        # Only a few ticks: outlier rule must not fire yet.
        g.check(Tick("EURUSD", 1.10000, 1.10002))
        v = g.check(Tick("EURUSD", 1.15000, 1.15002))
        self.assertTrue(v.ok)  # accepted because history too short to judge

    def test_rejected_outlier_excluded_from_history(self):
        g = LiveTickGuard(TickGuardConfig(outlier_sigma=8.0,
                                          min_history_for_outlier=20,
                                          max_frozen_ticks=0,
                                          exclude_rejected_from_history=True))
        self._prime(g)
        before = len(g._history["EURUSD"])
        g.check(Tick("EURUSD", 1.12000, 1.12002))  # rejected outlier
        after = len(g._history["EURUSD"])
        self.assertEqual(before, after)  # history unchanged by the bad tick


class TestStateAPI(unittest.TestCase):
    def test_rejection_tally(self):
        g = LiveTickGuard(TickGuardConfig())
        g.check(Tick("EURUSD", 1.10010, 1.10005))  # crossed -> reject
        g.check(Tick("EURUSD", 1.10000, 1.10000))  # locked -> reject
        self.assertEqual(g.rejections.get("EURUSD"), 2)

    def test_reset_one_symbol(self):
        g = LiveTickGuard(TickGuardConfig())
        g.check(clean("EURUSD"))
        g.check(clean("GBPUSD", 1.25000, 1.25003))
        g.reset("EURUSD")
        self.assertNotIn("EURUSD", g._history)
        self.assertIn("GBPUSD", g._history)

    def test_reset_all(self):
        g = LiveTickGuard(TickGuardConfig())
        g.check(clean("EURUSD"))
        g.check(clean("GBPUSD", 1.25000, 1.25003))
        g.reset()
        self.assertEqual(g._history, {})

    def test_stats_shape(self):
        g = LiveTickGuard(TickGuardConfig())
        g.check(clean())
        s = g.stats()
        self.assertIn("symbols_tracked", s)
        self.assertIn("rejections", s)
        self.assertIn("EURUSD", s["symbols_tracked"])

    def test_per_symbol_isolation(self):
        # A frozen EURUSD feed must not mark GBPUSD stale.
        g = LiveTickGuard(TickGuardConfig(max_frozen_ticks=2, outlier_sigma=0))
        for _ in range(5):
            g.check(Tick("EURUSD", 1.10000, 1.10002))
        v = g.check(Tick("GBPUSD", 1.25000, 1.25003))
        self.assertTrue(v.ok)


class TestVerdictBool(unittest.TestCase):
    def test_truthiness(self):
        g = LiveTickGuard(TickGuardConfig())
        self.assertTrue(bool(g.check(clean())))
        self.assertFalse(bool(g.check(Tick("EURUSD", 1.10010, 1.10005))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
