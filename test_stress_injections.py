# ==============================================================================
# test_stress_injections.py -- Tests for Monte Carlo stress injections
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import unittest
import numpy as np

from stress_injections import StressInjector, StressConfig, stress_and_summarize


class TestNoOp(unittest.TestCase):
    def test_default_config_is_identity(self):
        sims = np.random.RandomState(0).normal(0, 100, (50, 10))
        out = StressInjector(StressConfig()).apply(sims)
        np.testing.assert_allclose(out, sims)

    def test_input_not_mutated(self):
        sims = np.random.RandomState(1).normal(0, 100, (50, 10))
        snapshot = sims.copy()
        StressInjector(StressConfig(worst_day_factor=3.0)).apply(sims)
        np.testing.assert_allclose(sims, snapshot)  # original untouched


class TestWorstDayAmplification(unittest.TestCase):
    def test_worst_day_amplified(self):
        sims = np.array([[10.0, -100.0, 5.0, -20.0]])
        out = StressInjector(StressConfig(worst_day_factor=2.0)).apply(sims)
        # worst day (-100) doubled to -200; others unchanged.
        self.assertEqual(out[0, 1], -200.0)
        self.assertEqual(out[0, 0], 10.0)
        self.assertEqual(out[0, 3], -20.0)

    def test_positive_days_never_amplified(self):
        # An all-positive path has no negative worst day to amplify.
        sims = np.array([[10.0, 20.0, 30.0]])
        out = StressInjector(StressConfig(worst_day_factor=5.0)).apply(sims)
        np.testing.assert_allclose(out, sims)

    def test_multiple_worst_days(self):
        sims = np.array([[-50.0, -100.0, 5.0, -75.0]])
        out = StressInjector(
            StressConfig(worst_day_factor=2.0, worst_day_count=2)).apply(sims)
        # two most-negative: -100 and -75 doubled; -50 untouched.
        self.assertEqual(out[0, 1], -200.0)
        self.assertEqual(out[0, 3], -150.0)
        self.assertEqual(out[0, 0], -50.0)


class TestShockInjection(unittest.TestCase):
    def test_shock_applied_with_certainty(self):
        sims = np.zeros((100, 10))  # all-flat paths
        cfg = StressConfig(shock_probability=1.0, shock_loss_pct=0.04,
                           worst_day_factor=1.0)
        out = StressInjector(cfg, account_size=100_000).apply(sims)
        # every path should now contain exactly one -4000 day.
        for i in range(100):
            self.assertAlmostEqual(out[i].min(), -4000.0)

    def test_no_shock_when_prob_zero(self):
        sims = np.zeros((100, 10))
        out = StressInjector(StressConfig(shock_probability=0.0,
                                          shock_loss_pct=0.04)).apply(sims)
        np.testing.assert_allclose(out, sims)

    def test_shock_never_softens_worse_day(self):
        # A path already has a -10000 day; a -4000 shock must not replace it.
        sims = np.full((1, 10), 0.0)
        sims[0, 3] = -10_000.0
        cfg = StressConfig(shock_probability=1.0, shock_loss_pct=0.04)
        out = StressInjector(cfg, account_size=100_000).apply(sims)
        self.assertEqual(out[0].min(), -10_000.0)  # still the worst

    def test_shock_reproducible(self):
        sims = np.zeros((100, 10))
        cfg = StressConfig(shock_probability=0.5, shock_loss_pct=0.04,
                           random_seed=99)
        a = StressInjector(cfg).apply(sims)
        b = StressInjector(cfg).apply(sims)
        np.testing.assert_allclose(a, b)  # same seed -> same result


class TestSpreadDrag(unittest.TestCase):
    def test_extra_cost_subtracted_from_every_day(self):
        sims = np.full((5, 10), 100.0)
        out = StressInjector(StressConfig(extra_daily_cost=20.0)).apply(sims)
        np.testing.assert_allclose(out, np.full((5, 10), 80.0))

    def test_zero_cost_is_noop(self):
        sims = np.full((5, 10), 100.0)
        out = StressInjector(StressConfig(extra_daily_cost=0.0)).apply(sims)
        np.testing.assert_allclose(out, sims)


class TestLossDirectionalInvariant(unittest.TestCase):
    """The core safety property: stress can only ever make paths worse."""

    def test_all_injections_reduce_totals(self):
        sims = np.random.RandomState(3).normal(50, 400, (500, 20))
        original_totals = sims.sum(axis=1)
        cfg = StressConfig(worst_day_factor=2.5, shock_probability=0.2,
                           shock_loss_pct=0.03, extra_daily_cost=15.0)
        out = StressInjector(cfg).apply(sims)
        new_totals = out.sum(axis=1)
        self.assertTrue(np.all(new_totals <= original_totals + 1e-6))

    def test_stressed_worst_day_not_better(self):
        sims = np.random.RandomState(4).normal(0, 300, (200, 15))
        out = StressInjector(StressConfig(worst_day_factor=3.0)).apply(sims)
        self.assertLessEqual(out.min(), sims.min() + 1e-6)


class TestValidation(unittest.TestCase):
    def test_rejects_1d_array(self):
        with self.assertRaises(ValueError):
            StressInjector(StressConfig()).apply(np.array([1.0, 2.0, 3.0]))

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            StressInjector(StressConfig()).apply(np.empty((0, 0)))

    def test_accepts_list_input(self):
        out = StressInjector(StressConfig(extra_daily_cost=1.0)).apply(
            [[10.0, 20.0], [30.0, 40.0]])  # type: ignore[arg-type]
        self.assertEqual(out.shape, (2, 2))


class TestSummary(unittest.TestCase):
    def test_summary_fields(self):
        sims = np.random.RandomState(5).normal(50, 400, (300, 20))
        s = stress_and_summarize(
            sims, StressConfig(worst_day_factor=2.0, extra_daily_cost=10.0))
        self.assertIn("stressed", s)
        self.assertEqual(s["n_paths"], 300)
        self.assertEqual(s["n_days"], 20)
        # stressed worst day should be <= base worst day
        self.assertLessEqual(s["stressed_worst_day"], s["base_worst_day"])
        # stressed mean total should be <= base mean total
        self.assertLessEqual(s["stressed_mean_total"], s["base_mean_total"])


if __name__ == "__main__":
    unittest.main(verbosity=2)