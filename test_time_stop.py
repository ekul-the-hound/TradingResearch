# ==============================================================================
# test_time_stop.py -- Tests for the maximum holding-period exit
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# Uses injected `now`/`bar_index` so no real waiting is required.
# ==============================================================================

import unittest
from datetime import datetime, timezone, timedelta

from time_stop import TimeStop, TimeStopConfig, TimeStopVerdict


T0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


class TestRegistry(unittest.TestCase):
    def test_register_and_clear(self):
        ts = TimeStop()
        ts.register("EURUSD", now=T0)
        self.assertTrue(ts.is_registered("EURUSD"))
        ts.clear("EURUSD")
        self.assertFalse(ts.is_registered("EURUSD"))

    def test_registered_symbols(self):
        ts = TimeStop()
        ts.register("EURUSD", now=T0)
        ts.register("GBPUSD", now=T0)
        self.assertEqual(set(ts.registered_symbols()), {"EURUSD", "GBPUSD"})

    def test_reregister_resets_age(self):
        ts = TimeStop(TimeStopConfig(max_hold_seconds=3600))
        ts.register("EURUSD", now=T0)
        # 2h later would be expired...
        self.assertTrue(ts.expired("EURUSD", now=T0 + timedelta(hours=2)))
        # ...but re-registering at that time resets the clock.
        ts.register("EURUSD", now=T0 + timedelta(hours=2))
        self.assertFalse(ts.expired("EURUSD", now=T0 + timedelta(hours=2)))

    def test_clear_missing_symbol_is_safe(self):
        ts = TimeStop()
        ts.clear("NOPE")  # must not raise


class TestWallMode(unittest.TestCase):
    def setUp(self):
        self.ts = TimeStop(TimeStopConfig(max_hold_seconds=3600))
        self.ts.register("EURUSD", now=T0)

    def test_within_limit_not_expired(self):
        v = self.ts.check("EURUSD", now=T0 + timedelta(minutes=30))
        self.assertFalse(v.expired)

    def test_at_limit_expired(self):
        v = self.ts.check("EURUSD", now=T0 + timedelta(seconds=3600))
        self.assertTrue(v.expired)

    def test_past_limit_expired(self):
        v = self.ts.check("EURUSD", now=T0 + timedelta(hours=3))
        self.assertTrue(v.expired)
        self.assertIn("max", v.reason)

    def test_age_reported(self):
        v = self.ts.check("EURUSD", now=T0 + timedelta(seconds=1800))
        self.assertAlmostEqual(v.age_seconds, 1800, places=0)  # type: ignore[arg-type]


class TestBarMode(unittest.TestCase):
    def setUp(self):
        self.ts = TimeStop(TimeStopConfig(max_bars=10))
        self.ts.register("EURUSD", now=T0, bar_index=0)

    def test_within_bars_not_expired(self):
        v = self.ts.check("EURUSD", now=T0, bar_index=5)
        self.assertFalse(v.expired)

    def test_at_bar_limit_expired(self):
        v = self.ts.check("EURUSD", now=T0, bar_index=10)
        self.assertTrue(v.expired)

    def test_past_bar_limit_expired(self):
        v = self.ts.check("EURUSD", now=T0, bar_index=25)
        self.assertTrue(v.expired)
        self.assertIn("bars", v.reason)

    def test_bars_held_reported(self):
        v = self.ts.check("EURUSD", now=T0, bar_index=7)
        self.assertEqual(v.bars_held, 7)

    def test_entry_bar_offset(self):
        ts = TimeStop(TimeStopConfig(max_bars=5))
        ts.register("EURUSD", now=T0, bar_index=100)
        self.assertFalse(ts.expired("EURUSD", now=T0, bar_index=104))
        self.assertTrue(ts.expired("EURUSD", now=T0, bar_index=105))


class TestDailyCutoff(unittest.TestCase):
    def setUp(self):
        self.ts = TimeStop(TimeStopConfig(daily_cutoff_hhmm="21:00"))
        self.ts.register("EURUSD", now=T0)  # opened 10:00

    def test_before_cutoff_ok(self):
        v = self.ts.check("EURUSD",
                          now=datetime(2026, 1, 1, 20, 59, tzinfo=timezone.utc))
        self.assertFalse(v.expired)

    def test_at_cutoff_expired(self):
        v = self.ts.check("EURUSD",
                          now=datetime(2026, 1, 1, 21, 0, tzinfo=timezone.utc))
        self.assertTrue(v.expired)
        self.assertIn("cutoff", v.reason)

    def test_after_cutoff_expired(self):
        v = self.ts.check("EURUSD",
                          now=datetime(2026, 1, 1, 22, 30, tzinfo=timezone.utc))
        self.assertTrue(v.expired)

    def test_bad_cutoff_string_ignored(self):
        ts = TimeStop(TimeStopConfig(daily_cutoff_hhmm="not-a-time"))
        ts.register("EURUSD", now=T0)
        # invalid cutoff must not crash and must not expire
        v = ts.check("EURUSD", now=T0 + timedelta(hours=5))
        self.assertFalse(v.expired)


class TestUnknownPosition(unittest.TestCase):
    def test_unknown_expired_by_default(self):
        ts = TimeStop(TimeStopConfig(max_hold_seconds=3600))
        v = ts.check("EURUSD", now=T0)  # never registered
        self.assertTrue(v.expired)
        self.assertIn("unknown age", v.reason)

    def test_unknown_permissive_when_configured(self):
        ts = TimeStop(TimeStopConfig(max_hold_seconds=3600, expire_unknown=False))
        v = ts.check("EURUSD", now=T0)
        self.assertFalse(v.expired)


class TestCombinedModes(unittest.TestCase):
    def test_first_limit_hit_wins_wall(self):
        # wall limit (1h) hit before bar limit (100 bars)
        ts = TimeStop(TimeStopConfig(max_hold_seconds=3600, max_bars=100))
        ts.register("EURUSD", now=T0, bar_index=0)
        v = ts.check("EURUSD", now=T0 + timedelta(hours=2), bar_index=5)
        self.assertTrue(v.expired)
        self.assertIn("s >=", v.reason)  # wall reason

    def test_first_limit_hit_wins_bar(self):
        # bar limit (5) hit before wall limit (10h)
        ts = TimeStop(TimeStopConfig(max_hold_seconds=36000, max_bars=5))
        ts.register("EURUSD", now=T0, bar_index=0)
        v = ts.check("EURUSD", now=T0 + timedelta(minutes=5), bar_index=6)
        self.assertTrue(v.expired)
        self.assertIn("bars", v.reason)


class TestCheckAll(unittest.TestCase):
    def test_returns_only_expired(self):
        ts = TimeStop(TimeStopConfig(max_hold_seconds=3600))
        ts.register("EURUSD", now=T0)
        ts.register("GBPUSD", now=T0 + timedelta(minutes=50))
        # At T0+70min: EURUSD is 70min old (expired), GBPUSD is 20min (ok)
        now = T0 + timedelta(minutes=70)
        expired = ts.check_all(["EURUSD", "GBPUSD"], now=now)
        symbols = [v.symbol for v in expired]
        self.assertIn("EURUSD", symbols)
        self.assertNotIn("GBPUSD", symbols)

    def test_empty_when_all_fresh(self):
        ts = TimeStop(TimeStopConfig(max_hold_seconds=3600))
        ts.register("EURUSD", now=T0)
        self.assertEqual(ts.check_all(["EURUSD"], now=T0), [])


class TestVerdictBool(unittest.TestCase):
    def test_truthiness(self):
        ts = TimeStop(TimeStopConfig(max_hold_seconds=3600))
        ts.register("EURUSD", now=T0)
        self.assertFalse(bool(ts.check("EURUSD", now=T0)))
        self.assertTrue(bool(ts.check("EURUSD", now=T0 + timedelta(hours=2))))


class TestIntradayBenefit(unittest.TestCase):
    """The daily-anchor-exactness use case: no position survives to midnight."""

    def test_position_forced_flat_before_midnight(self):
        # cutoff 23:00, opened 22:00 -> expired by 23:00, never crosses midnight
        ts = TimeStop(TimeStopConfig(daily_cutoff_hhmm="23:00"))
        ts.register("EURUSD",
                    now=datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc))
        v = ts.check("EURUSD",
                     now=datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc))
        self.assertTrue(v.expired)


if __name__ == "__main__":
    unittest.main(verbosity=2)