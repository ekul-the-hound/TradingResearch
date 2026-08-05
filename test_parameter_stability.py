# ==============================================================================
# test_parameter_stability.py
# ==============================================================================
# Phase 2, Item 16.
#
#   python test_parameter_stability.py
#
# Built around two synthetic surfaces with known shape: a broad plateau that
# must pass, and a noise spike that must fail. A gate that cannot separate
# those two is doing nothing.
#
# Import failures are HARD ERRORS. A skip is not a pass.
# ==============================================================================

import sys
import unittest

import numpy as np

import parameter_stability as ps


# --------------------------------------------------------------------------
# Surfaces with known shape
# --------------------------------------------------------------------------

def plateau_1d():
    """A broad hill: neighbours of the peak stay close to it."""
    return [0.2, 0.6, 1.1, 1.5, 1.7, 1.8, 1.7, 1.5, 1.1, 0.6, 0.2]


def spike_1d():
    """One lucky combination surrounded by nothing."""
    return [0.1, 0.1, 0.1, 0.1, 0.1, 2.4, 0.1, 0.1, 0.1, 0.1, 0.1]


def cliff_1d():
    """Decent scores that fall through the floor one step away."""
    return [1.6, 1.7, 1.8, -0.9, -1.2, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9]


def plateau_2d():
    x, y = np.meshgrid(np.linspace(-2.5, 2.5, 11), np.linspace(-2.5, 2.5, 11))
    return 1.8 * np.exp(-(x ** 2 + y ** 2) / 9.0)


def spike_2d():
    m = np.full((11, 11), 0.1)
    m[5, 5] = 2.4
    return m


def ridge_2d():
    """Stable along one axis, a cliff along the other."""
    m = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            m[i, j] = 1.6 if j == 5 else -1.0
    return m


# --------------------------------------------------------------------------

class TestSeparatesPlateauFromSpike(unittest.TestCase):
    """If this fails, nothing else in the module matters."""

    def test_plateau_passes(self):
        r = ps.analyze_1d(plateau_1d())
        self.assertTrue(r.passed, f"false negative: {r.failures}")

    def test_spike_fails(self):
        r = ps.analyze_1d(spike_1d())
        self.assertFalse(r.passed, "an isolated peak must not be promoted")
        self.assertTrue(any('spike_index' in f for f in r.failures))

    def test_cliff_fails(self):
        r = ps.analyze_1d(cliff_1d(), chosen_index=2)
        self.assertFalse(r.passed)
        self.assertTrue(any('cliff_distance' in f or 'sign_consistency' in f
                            for f in r.failures))

    def test_gate_returns_bool(self):
        self.assertTrue(ps.gate(plateau_1d()))
        self.assertFalse(ps.gate(spike_1d()))


class TestScaleInvariance(unittest.TestCase):
    """
    The defect in the existing stability_score, pinned so it cannot return.

    stability_score = mean(abs(diff(returns))) scales with the magnitude of the
    returns, so a smooth curve at 40% scored WORSE than a jagged one at 1%.
    Every metric here must be indifferent to the units.
    """

    def test_existing_metric_ranks_backwards(self):
        jagged_small = [1.0, 0.2, 1.1, 0.1, 1.2]
        smooth_large = [40.0, 42.0, 44.0, 46.0, 48.0]
        old = lambda v: float(np.mean(np.abs(np.diff(v))))
        self.assertLess(old(jagged_small), old(smooth_large),
                        "documents the bug: jagged scores 'better' than smooth")

    def test_roughness_ranks_correctly(self):
        jagged_small = [1.0, 0.2, 1.1, 0.1, 1.2]
        smooth_large = [40.0, 42.0, 44.0, 46.0, 48.0]
        self.assertGreater(ps.roughness(jagged_small), ps.roughness(smooth_large),
                           "scale-free roughness must call the jagged one rougher")

    def test_verdict_survives_rescaling(self):
        base = plateau_1d()
        for factor in (0.01, 1.0, 100.0, 10_000.0):
            scaled = [v * factor for v in base]
            self.assertTrue(ps.gate(scaled),
                            f"verdict changed at scale {factor}")

    def test_spike_stays_a_spike_at_any_scale(self):
        for factor in (0.01, 1.0, 1000.0):
            self.assertFalse(ps.gate([v * factor for v in spike_1d()]),
                             f"spike passed at scale {factor}")

    def test_metrics_are_dimensionless(self):
        a = ps.analyze_1d(plateau_1d())
        b = ps.analyze_1d([v * 500 for v in plateau_1d()])
        for f in ('plateau_ratio', 'spike_index', 'sign_consistency', 'roughness'):
            self.assertAlmostEqual(getattr(a, f), getattr(b, f), places=6,
                                   msg=f"{f} changed with scale")


class TestRobustRecommendation(unittest.TestCase):
    """The peak is usually the spike. The plateau centre is the tradeable point."""

    def test_recommends_the_plateau_centre_over_the_peak(self):
        # Narrow peak at index 1, broad hill centred at 7.
        scores = [0.2, 2.2, 0.2, 1.2, 1.4, 1.5, 1.6, 1.6, 1.5, 1.3, 1.1]
        best = int(np.argmax(scores))
        idx, _ = ps.recommend_robust_point(scores)
        self.assertEqual(best, 1, "the peak really is the narrow one")
        self.assertGreater(idx, 3, "recommendation must move to the broad hill")

    def test_recommendation_appears_in_the_summary(self):
        scores = [0.2, 2.2, 0.2, 1.2, 1.4, 1.5, 1.6, 1.6, 1.5, 1.3, 1.1]
        s = ps.analyze_1d(scores).summary()
        self.assertIn('Plateau centre', s)

    def test_flat_surface_recommends_something_valid(self):
        idx, score = ps.recommend_robust_point([1.0] * 9)
        self.assertTrue(0 <= idx < 9)
        self.assertAlmostEqual(score, 1.0)


class TestChosenPointNotBest(unittest.TestCase):
    """The gate must judge the point you will trade, not the best one."""

    def test_chosen_index_is_respected(self):
        scores = spike_1d()
        at_peak = ps.analyze_1d(scores, chosen_index=5)
        elsewhere = ps.analyze_1d(scores, chosen_index=1)
        self.assertEqual(at_peak.chosen_index, 5)
        self.assertEqual(elsewhere.chosen_index, 1)
        self.assertNotEqual(at_peak.spike_index, elsewhere.spike_index)

    def test_defaults_to_the_best_point(self):
        r = ps.analyze_1d(spike_1d())
        self.assertEqual(r.chosen_index, 5)

    def test_out_of_range_index_is_an_error(self):
        r = ps.analyze_1d(plateau_1d(), chosen_index=99)
        self.assertIsNotNone(r.error)
        self.assertFalse(r.passed)


class TestRefusesToGuess(unittest.TestCase):
    """Unmeasurable must not read as fine."""

    def test_too_few_points_fails(self):
        r = ps.analyze_1d([1.0, 1.1])
        self.assertIsNotNone(r.error)
        self.assertFalse(r.passed)
        self.assertFalse(ps.gate([1.0, 1.1]))

    def test_empty_fails(self):
        self.assertFalse(ps.gate([]))

    def test_nan_fails_rather_than_flattening(self):
        """
        A failed backtest returning NaN must not be read as a flat -- and
        therefore stable -- neighbourhood.
        """
        r = ps.analyze_1d([1.0, 1.2, np.nan, 1.3, 1.1])
        self.assertIsNotNone(r.error)
        self.assertFalse(r.passed)

    def test_inf_fails(self):
        r = ps.analyze_1d([1.0, 1.2, np.inf, 1.3, 1.1])
        self.assertIsNotNone(r.error)


class TestTwoDimensional(unittest.TestCase):

    def test_plateau_surface_passes(self):
        r = ps.analyze_2d(plateau_2d())
        self.assertTrue(r.passed, f"false negative: {r.failures}")

    def test_spike_surface_fails(self):
        r = ps.analyze_2d(spike_2d())
        self.assertFalse(r.passed)

    def test_ridge_fails_because_one_axis_is_a_cliff(self):
        """
        Stable along one axis, a cliff along the other. Two independent 1D
        sweeps would each look acceptable; the block neighbourhood catches it.
        """
        r = ps.analyze_2d(ridge_2d(), chosen=(5, 5))
        self.assertFalse(r.passed, "a ridge is not a plateau")

    def test_2d_recommendation_is_in_range(self):
        r = ps.analyze_2d(plateau_2d())
        i, j = r.recommended_index
        self.assertTrue(0 <= i < 11 and 0 <= j < 11)

    def test_non_matrix_is_an_error(self):
        self.assertIsNotNone(ps.analyze_2d([1, 2, 3]).error)

    def test_gate_dispatches_on_dimension(self):
        self.assertTrue(ps.gate(plateau_2d()))
        self.assertFalse(ps.gate(spike_2d()))


class TestThresholds(unittest.TestCase):

    def test_thresholds_are_configurable(self):
        scores = spike_1d()
        self.assertFalse(ps.gate(scores))
        lax = {'max_spike_index': 1.01, 'min_plateau_ratio': 0.0,
               'min_sign_consistency': 0.0, 'min_cliff_distance': 0}
        self.assertTrue(ps.gate(scores, thresholds=lax),
                        "thresholds must actually be adjustable")

    def test_failures_name_the_criterion_and_the_numbers(self):
        r = ps.analyze_1d(spike_1d())
        self.assertTrue(r.failures)
        for f in r.failures:
            self.assertTrue(any(c.isdigit() for c in f),
                            "a failure without numbers is not actionable")


class TestScatter(unittest.TestCase):
    """
    Irregular sampling -- what the mutation loop and the strategy pool actually
    produce. There is no grid to sweep: multi_objective_optimizer selects over
    already-backtested strategies and never varies a parameter.
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        cluster = rng.normal(0, 0.3, (20, 2))
        outlier = np.array([[8.0, 8.0]])
        self.points = np.vstack([cluster, outlier])
        self.scores = np.concatenate([rng.normal(1.5, 0.1, 20), [3.0]])

    def test_cluster_member_passes(self):
        r = ps.analyze_scatter(self.points, self.scores, chosen_index=0)
        self.assertTrue(r.passed, f"false negative: {r.failures}")

    def test_isolated_outlier_fails(self):
        r = ps.analyze_scatter(self.points, self.scores, chosen_index=20)
        self.assertFalse(r.passed, "a lucky isolated point must not be promoted")

    def test_dimensions_are_normalised_before_measuring_distance(self):
        """
        Without per-dimension normalisation, a parameter measured in bars
        (10-200) swamps one measured as a fraction (0.01-0.05), and 'nearest
        neighbour' silently means 'nearest in the widest parameter'.
        """
        raw = np.array([[10, 0.01], [200, 0.01], [10, 0.05], [200, 0.05]], dtype=float)
        norm = ps._normalise_points(raw)
        self.assertAlmostEqual(float(norm[:, 0].max()), 1.0)
        self.assertAlmostEqual(float(norm[:, 1].max()), 1.0)
        self.assertAlmostEqual(float(norm[:, 0].min()), 0.0)
        self.assertAlmostEqual(float(norm[:, 1].min()), 0.0)

    def test_constant_dimension_does_not_divide_by_zero(self):
        raw = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
        norm = ps._normalise_points(raw)
        self.assertTrue(np.all(np.isfinite(norm)))
        self.assertTrue(np.allclose(norm[:, 1], 0.0))

    def test_mismatched_lengths_is_an_error(self):
        r = ps.analyze_scatter(np.zeros((5, 2)), [1.0, 2.0])
        self.assertIsNotNone(r.error)
        self.assertFalse(r.passed)

    def test_too_few_points_is_an_error(self):
        r = ps.analyze_scatter(np.zeros((3, 2)), [1.0, 2.0, 3.0])
        self.assertIsNotNone(r.error)

    def test_nan_is_an_error_not_a_flat_neighbourhood(self):
        pts = np.zeros((5, 2))
        r = ps.analyze_scatter(pts, [1.0, 2.0, np.nan, 3.0, 1.5])
        self.assertIsNotNone(r.error)

    def test_1d_points_are_accepted(self):
        r = ps.analyze_scatter([1.0, 2.0, 3.0, 4.0, 5.0],
                               [1.0, 1.1, 1.2, 1.1, 1.0], chosen_index=2)
        self.assertIsNone(r.error)

    def test_recommends_a_valid_point(self):
        r = ps.analyze_scatter(self.points, self.scores, chosen_index=20)
        self.assertTrue(0 <= r.recommended_index < len(self.scores))
        self.assertNotEqual(r.recommended_index, 20,
                            "must not recommend the isolated outlier")


def main():
    print("=" * 70)
    print("PARAMETER STABILITY - TEST SUITE")
    print("=" * 70)
    print("The best-scoring parameter set is usually the luckiest one. A point")
    print("whose neighbours collapse is describing this dataset, not the market.")
    print("")
    print("The existing stability_score cannot be thresholded: it is")
    print("unnormalised, so a smooth curve at 40% scores worse than a jagged")
    print("one at 1%. Every metric here is scale-free, and that is pinned.")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    n_skip = len(getattr(result, 'skipped', []))
    print("\n" + "=" * 70)
    if n_skip:
        print(f"[WARN] {n_skip} test(s) SKIPPED - a skip is not a pass")
    if result.wasSuccessful() and not n_skip:
        print(f"ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}  "
              f"SKIPPED: {n_skip}")
    print("=" * 70)
    return 0 if (result.wasSuccessful() and not n_skip) else 1


if __name__ == '__main__':
    sys.exit(main())