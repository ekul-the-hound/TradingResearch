# ==============================================================================
# test_weekend_policy.py -- Tests for the weekend/EOD flatten policy + gap stress
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# The Prague-midnight tests require pytz + ftmo_daily_anchor (the project's
# canonical DST-correct helpers); they assert real seasonal boundaries.
# ==============================================================================

import unittest
from datetime import datetime, timezone, timedelta

from weekend_policy import (
    WeekendPolicy, WeekendPolicyConfig, FlattenDecision,
    weekend_gaps, GapStats, _HAVE_PRAGUE,
)


def utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class TestPragueMidnightDST(unittest.TestCase):
    """These assert the DST-correct boundary; skip only if pytz truly absent."""

    def setUp(self):
        if not _HAVE_PRAGUE:
            self.skipTest("pytz/ftmo_daily_anchor unavailable in this env")
        self.pol = WeekendPolicy(WeekendPolicyConfig(
            flatten_before_midnight_minutes=5, enforce_friday_close=False))

    def test_winter_boundary_2257_flattens(self):
        # Winter: Prague midnight = 23:00 UTC. 22:57 = 3 min before.
        d = self.pol.check(utc(2026, 1, 7, 22, 57))
        self.assertTrue(d.should_flatten)
        self.assertEqual(d.boundary, "prague_midnight")

    def test_winter_2240_does_not_flatten(self):
        # 20 min before midnight, buffer is 5 -> no flatten.
        d = self.pol.check(utc(2026, 1, 7, 22, 40))
        self.assertFalse(d.should_flatten)

    def test_summer_boundary_2157_flattens(self):
        # Summer: Prague midnight = 22:00 UTC. 21:57 = 3 min before.
        d = self.pol.check(utc(2026, 7, 7, 21, 57))
        self.assertTrue(d.should_flatten)

    def test_summer_2257_is_past_midnight_no_flatten(self):
        # In summer 22:57 UTC is already ~00:57 Prague -> not near next midnight.
        d = self.pol.check(utc(2026, 7, 7, 22, 57))
        self.assertFalse(d.should_flatten)

    def test_midday_far_from_boundary(self):
        d = self.pol.check(utc(2026, 1, 7, 12, 0))
        self.assertFalse(d.should_flatten)
        self.assertIsNotNone(d.minutes_to_boundary)


class TestFridayClose(unittest.TestCase):
    def setUp(self):
        self.pol = WeekendPolicy(WeekendPolicyConfig(
            enforce_prague_midnight=False,
            enforce_friday_close=True,
            friday_close_hhmm_utc="21:00",
            flatten_before_friday_close_minutes=30))

    def test_flatten_near_friday_close(self):
        # 2026-01-09 is a Friday. 20:40 UTC = 20 min before 21:00 close.
        d = self.pol.check(utc(2026, 1, 9, 20, 40))
        self.assertTrue(d.should_flatten)
        self.assertEqual(d.boundary, "friday_close")

    def test_no_flatten_friday_morning(self):
        d = self.pol.check(utc(2026, 1, 9, 9, 0))
        self.assertFalse(d.should_flatten)

    def test_no_flatten_midweek(self):
        # Wednesday -> Friday close is days away.
        d = self.pol.check(utc(2026, 1, 7, 20, 45))
        self.assertFalse(d.should_flatten)

    def test_past_friday_close(self):
        d = self.pol.check(utc(2026, 1, 9, 21, 30))
        self.assertFalse(d.should_flatten)


class TestNearestBoundaryWins(unittest.TestCase):
    def test_reports_nearest_when_none_trigger(self):
        pol = WeekendPolicy()
        d = pol.check(utc(2026, 1, 7, 12, 0))  # Wednesday midday
        self.assertFalse(d.should_flatten)
        self.assertIn("min to", d.reason)


class TestConfigToggles(unittest.TestCase):
    def test_all_rules_off(self):
        pol = WeekendPolicy(WeekendPolicyConfig(
            enforce_prague_midnight=False, enforce_friday_close=False))
        d = pol.check(utc(2026, 1, 9, 20, 55))
        self.assertFalse(d.should_flatten)
        self.assertIn("no boundary rules", d.reason)

    def test_naive_datetime_assumed_utc(self):
        pol = WeekendPolicy(WeekendPolicyConfig(enforce_friday_close=False))
        # naive datetime should not raise
        d = pol.check(datetime(2026, 1, 7, 12, 0))
        self.assertIsInstance(d, FlattenDecision)


class TestFlattenDecisionBool(unittest.TestCase):
    def test_truthiness(self):
        self.assertTrue(bool(FlattenDecision(True, "x")))
        self.assertFalse(bool(FlattenDecision(False, "y")))


class TestWeekendGaps(unittest.TestCase):
    def _make_bars(self, n_weeks=20, gap_pct=0.5):
        import pandas as pd
        import numpy as np
        rows = []
        dt = pd.Timestamp("2026-01-05 00:00")  # a Monday
        price = 1.10
        for w in range(n_weeks):
            # 5 weekday bars
            for day in range(5):
                rows.append((dt, price, price + 0.001, price - 0.001, price))
                dt = dt + pd.Timedelta(days=1)
                price += 0.0005
            # weekend jump (2 days) with a gap on Monday open
            dt = dt + pd.Timedelta(days=2)
            price = price * (1 + gap_pct / 100.0)  # gap up
        df = pd.DataFrame(rows, columns=["dt", "open", "high", "low", "close"])
        df = df.set_index("dt")
        return df

    def test_gaps_detected(self):
        df = self._make_bars(n_weeks=20, gap_pct=0.5)
        stats = weekend_gaps(df, min_weekends=10)
        self.assertTrue(stats.sufficient)
        self.assertGreaterEqual(stats.n_weekends, 15)
        # mean absolute gap should be near 0.5%
        self.assertGreater(stats.mean_gap_pct, 0.2)

    def test_insufficient_weekends(self):
        df = self._make_bars(n_weeks=3, gap_pct=0.5)
        stats = weekend_gaps(df, min_weekends=10)
        self.assertFalse(stats.sufficient)
        self.assertIn("weekend gaps", stats.note)

    def test_empty_bars(self):
        import pandas as pd
        stats = weekend_gaps(pd.DataFrame(), min_weekends=10)
        self.assertFalse(stats.sufficient)

    def test_none_bars(self):
        stats = weekend_gaps(None)
        self.assertFalse(stats.sufficient)

    def test_worst_gap_signed(self):
        import pandas as pd
        import numpy as np
        # Build bars with a big gap DOWN to check signed worst.
        rows = []
        dt = pd.Timestamp("2026-01-05 00:00")
        price = 1.10
        for w in range(15):
            for day in range(5):
                rows.append((dt, price, price, price, price))
                dt += pd.Timedelta(days=1)
            dt += pd.Timedelta(days=2)
            price = price * (1 - 0.01)  # 1% gap DOWN each weekend
        df = pd.DataFrame(rows, columns=["dt", "open", "high", "low",
                                         "close"]).set_index("dt")
        stats = weekend_gaps(df, min_weekends=10)
        self.assertTrue(stats.sufficient)
        self.assertLess(stats.worst_gap_pct, 0)  # negative = gap down


if __name__ == "__main__":
    unittest.main(verbosity=2)
