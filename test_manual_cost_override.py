# ==============================================================================
# test_manual_cost_override.py -- Tests for the manual pessimistic cost override
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import os
import json
import tempfile
import unittest

from manual_cost_override import (
    ManualCostOverride, ManualCosts, pips_to_pct, pct_to_pips,
)


class OverrideTestBase(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".json")
        self.ov = ManualCostOverride(path=self.path)

    def tearDown(self):
        for p in (self.path, self.path + ".tmp"):
            if os.path.exists(p):
                os.remove(p)


class TestPipConversion(unittest.TestCase):
    def test_pips_to_pct_roundtrip(self):
        pct = pips_to_pct(2.0)
        self.assertAlmostEqual(pct_to_pips(pct), 2.0, places=6)

    def test_two_pips_reasonable(self):
        # ~2 pips should be well under 0.05% of notional.
        self.assertLess(pips_to_pct(2.0), 0.05)
        self.assertGreater(pips_to_pct(2.0), 0.0)


class TestSetGet(OverrideTestBase):
    def test_starts_not_set(self):
        self.assertFalse(self.ov.is_set())
        self.assertIsNone(self.ov.get())

    def test_set_and_get(self):
        self.ov.set_costs(ManualCosts(spread_pct=0.02, slippage_pct=0.01))
        got = self.ov.get()
        self.assertIsNotNone(got)
        self.assertEqual(got.spread_pct, 0.02)

    def test_set_stamps_updated_at(self):
        c = self.ov.set_costs(ManualCosts(spread_pct=0.02))
        self.assertTrue(c.updated_at)

    def test_set_from_pips(self):
        self.ov.set_from_pips(spread_pips=2.0, slippage_pips=1.0)
        got = self.ov.get()
        self.assertAlmostEqual(pct_to_pips(got.spread_pct), 2.0, places=4)
        self.assertAlmostEqual(pct_to_pips(got.slippage_pct), 1.0, places=4)

    def test_clear(self):
        self.ov.set_costs(ManualCosts())
        self.ov.clear()
        self.assertFalse(self.ov.is_set())


class TestIntradayNoSwaps(OverrideTestBase):
    def test_default_overnight_zero(self):
        self.ov.set_costs(ManualCosts())
        self.assertEqual(self.ov.get().overnight_rate, 0.0)

    def test_set_from_pips_zero_overnight(self):
        self.ov.set_from_pips(2.0)
        self.assertEqual(self.ov.get().overnight_rate, 0.0)


class TestCostProfileEmission(OverrideTestBase):
    def test_emits_cost_profile(self):
        self.ov.set_from_pips(2.0, 1.0)
        prof = self.ov.to_cost_profile()
        # Must be a CostProfile with the right fields.
        self.assertTrue(hasattr(prof, "spread_pct"))
        self.assertTrue(hasattr(prof, "overnight_rate"))
        self.assertEqual(prof.overnight_rate, 0.0)

    def test_profile_values_match(self):
        self.ov.set_costs(ManualCosts(spread_pct=0.02, commission_pct=0.003,
                                      slippage_pct=0.01))
        prof = self.ov.to_cost_profile()
        self.assertEqual(prof.spread_pct, 0.02)
        self.assertEqual(prof.commission_pct, 0.003)
        self.assertEqual(prof.slippage_pct, 0.01)

    def test_no_silent_fallback(self):
        # The core discipline: no override set -> raise, never a silent default.
        with self.assertRaises(RuntimeError):
            self.ov.to_cost_profile()


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_cost_sums_correctly(self):
        c = ManualCosts(spread_pct=0.02, commission_pct=0.002, slippage_pct=0.01)
        # spread + 2*(commission + slippage) = 0.02 + 2*0.012 = 0.044
        self.assertAlmostEqual(c.total_round_trip_pct(), 0.044, places=6)


class TestPersistence(OverrideTestBase):
    def test_survives_reload(self):
        self.ov.set_from_pips(2.5, 1.5)
        ov2 = ManualCostOverride(path=self.path)
        got = ov2.get()
        self.assertAlmostEqual(pct_to_pips(got.spread_pct), 2.5, places=4)

    def test_tolerates_extra_keys(self):
        # A file with an unknown key should still load (forward-compat).
        data = {"asset_class": "forex", "spread_pct": 0.02,
                "commission_pct": 0.002, "slippage_pct": 0.01,
                "overnight_rate": 0.0, "min_commission": 0.0,
                "note": "x", "updated_at": "t", "future_field": 123}
        with open(self.path, "w") as f:
            json.dump(data, f)
        got = self.ov.get()
        self.assertEqual(got.spread_pct, 0.02)

    def test_corrupt_file_reads_none(self):
        with open(self.path, "w") as f:
            f.write("{ not json")
        self.assertIsNone(self.ov.get())


class TestDescribe(OverrideTestBase):
    def test_describe_not_set(self):
        self.assertIn("NOT SET", self.ov.describe())

    def test_describe_set(self):
        self.ov.set_from_pips(2.0)
        d = self.ov.describe()
        self.assertIn("spread", d)
        self.assertIn("intraday", d)  # overnight 0 -> intraday note


if __name__ == "__main__":
    unittest.main(verbosity=2)
