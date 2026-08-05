# ==============================================================================
# test_consistency_rule.py
# ==============================================================================

import sys
import unittest
from typing import Optional, TypeVar

import numpy as np
import pandas as pd

import consistency_rule
from consistency_rule import (
    NOT_EVALUATED_NO_DAYS, NOT_EVALUATED_NO_PROFIT, NOT_EVALUATED_NO_THRESHOLD,
    VARIANTS_NOT_MODELLED, ConsistencyResult, check_consistency,
    check_consistency_frame, consistency_breach_mask, consistency_stats,
)


# ------------------------------------------------------------------
# Narrowing helper
# ------------------------------------------------------------------
# Many values under test are Optional by design -- best_day_share is None when
# the rule could not be evaluated, DeltaCell.raw is None when there is no
# baseline, apply_firm_form returns None on rejection. Passing those straight
# into assertAlmostEqual fails to type-check, and `or 0.0` would paper over a
# None by turning it into a passing comparison against zero.
#
# This asserts non-None and hands back the narrowed value, so the check is
# real and the following assertion is well-typed.
_T = TypeVar('_T')


def not_none(value: Optional[_T], msg: str = 'expected a value, got None') -> _T:
    assert value is not None, msg
    return value



class TestBasicArithmetic(unittest.TestCase):

    def test_even_distribution_passes(self):
        r = check_consistency([100, 100, 100, 100], threshold=0.30)
        self.assertTrue(r.evaluated)
        self.assertTrue(r.passed)
        self.assertAlmostEqual(not_none(r.best_day_share), 0.25)

    def test_concentrated_profit_fails(self):
        """900 of 1000 total from one day is 90%, far over a 30% cap."""
        r = check_consistency([900, 50, 30, 20], threshold=0.30)
        self.assertTrue(r.evaluated)
        self.assertFalse(r.passed)
        self.assertAlmostEqual(not_none(r.best_day_share), 0.90)
        self.assertAlmostEqual(r.best_day_profit, 900.0)

    def test_exact_threshold_passes(self):
        """<= is the comparison, so landing exactly on the cap complies."""
        r = check_consistency([300, 300, 200, 200], threshold=0.30)
        self.assertTrue(r.passed)
        self.assertAlmostEqual(not_none(r.best_day_share), 0.30)

    def test_losing_days_shrink_the_denominator(self):
        """
        Net profit, not gross. A big losing day makes the best day a LARGER
        share of what is left, which is the stricter and more common reading.
        """
        r = check_consistency([500, 200, -400], threshold=0.60)
        self.assertAlmostEqual(r.total_profit, 300.0)
        self.assertAlmostEqual(not_none(r.best_day_share), 500.0 / 300.0)
        self.assertFalse(r.passed)

    def test_single_day_is_always_all_of_it(self):
        r = check_consistency([1000], threshold=0.50)
        self.assertTrue(r.evaluated)
        self.assertFalse(r.passed)
        self.assertAlmostEqual(not_none(r.best_day_share), 1.0)

    def test_best_day_date_reported(self):
        r = check_consistency([100, 900, 50],
                              threshold=0.30,
                              dates=['Mon', 'Tue', 'Wed'])
        self.assertEqual(r.best_day_date, 'Tue')


class TestUndefinedCases(unittest.TestCase):
    """
    The cases where a confident answer would be a fabricated one.

    Neither pass nor fail is correct when there is no profit to take a share
    of, so the result must report that it could not evaluate.
    """

    def test_zero_total_profit_is_not_evaluated(self):
        r = check_consistency([100, -100], threshold=0.30)
        self.assertFalse(r.evaluated)
        self.assertIsNone(r.passed)
        self.assertEqual(r.reason, NOT_EVALUATED_NO_PROFIT)

    def test_net_loss_is_not_evaluated(self):
        r = check_consistency([100, -500], threshold=0.30)
        self.assertFalse(r.evaluated)
        self.assertIsNone(r.passed)

    def test_all_losing_days_is_not_evaluated(self):
        r = check_consistency([-50, -20, -30], threshold=0.30)
        self.assertFalse(r.evaluated)
        self.assertEqual(r.reason, NOT_EVALUATED_NO_PROFIT)

    def test_empty_series_is_not_evaluated(self):
        r = check_consistency([], threshold=0.30)
        self.assertFalse(r.evaluated)
        self.assertEqual(r.reason, NOT_EVALUATED_NO_DAYS)

    def test_no_threshold_means_no_rule(self):
        r = check_consistency([100, 200], threshold=None)
        self.assertFalse(r.evaluated)
        self.assertEqual(r.reason, NOT_EVALUATED_NO_THRESHOLD)

    def test_unevaluated_is_not_a_pass(self):
        """
        is_pass must be strict. Treating 'could not evaluate' as success is
        exactly the confident-wrong-answer failure this guards against.
        """
        r = check_consistency([100, -100], threshold=0.30)
        self.assertFalse(r.is_pass)
        self.assertFalse(r.is_fail)

    def test_no_division_by_zero(self):
        for series in ([0, 0, 0], [0], [5, -5]):
            r = check_consistency(series, threshold=0.30)
            self.assertFalse(r.evaluated)
            self.assertIsNone(r.best_day_share)


class TestReporting(unittest.TestCase):

    def test_summary_states_the_numbers(self):
        r = check_consistency([900, 100], threshold=0.30)
        s = r.summary()
        self.assertIn('FAIL', s)
        self.assertIn('90.0%', s)

    def test_summary_of_unevaluated_says_so(self):
        r = check_consistency([100, -100], threshold=0.30)
        self.assertIn('NOT evaluated', r.summary())

    def test_unmodelled_variants_are_named(self):
        """
        The threshold is a number; these are different computations. Naming
        them keeps 'we model the consistency rule' from overclaiming.
        """
        self.assertTrue(VARIANTS_NOT_MODELLED)
        joined = ' '.join(VARIANTS_NOT_MODELLED).lower()
        self.assertIn('profit target', joined)
        self.assertIn('trade', joined)

    def test_counts_reported(self):
        r = check_consistency([100, -50, 200, 0], threshold=0.90)
        self.assertEqual(r.n_days, 4)
        self.assertEqual(r.n_profitable_days, 2)


class TestFrameWrapper(unittest.TestCase):

    def setUp(self):
        idx = pd.date_range('2024-01-01', periods=4, freq='D')
        self.df = pd.DataFrame({'A': [400., 50., 30., 20.],
                                'B': [500., 50., 20., 30.]}, index=idx)

    def test_sums_across_strategies_first(self):
        """
        The rule applies to the ACCOUNT's daily profit. Neither strategy
        alone contributes 90%, but together day one does.
        """
        r = check_consistency_frame(self.df, threshold=0.30)
        self.assertTrue(r.evaluated)
        self.assertAlmostEqual(r.best_day_profit, 900.0)
        self.assertFalse(r.passed)

    def test_empty_frame(self):
        r = check_consistency_frame(pd.DataFrame(), threshold=0.30)
        self.assertFalse(r.evaluated)

    def test_none_frame(self):
        self.assertFalse(check_consistency_frame(None, threshold=0.3).evaluated)


class TestBootstrapMask(unittest.TestCase):

    def setUp(self):
        # path 0: even -> complies. path 1: concentrated -> breaches.
        # path 2: net loss -> rule does not apply.
        self.sims = np.array([
            [100., 100., 100., 100.],
            [900., 50., 30., 20.],
            [-100., -100., 50., 20.],
        ])

    def test_mask_flags_only_the_concentrated_path(self):
        m = consistency_breach_mask(self.sims, 0.30)
        self.assertFalse(m[0])
        self.assertTrue(m[1])

    def test_losing_paths_are_not_breaches(self):
        """
        A path that never made money did not violate the consistency rule --
        it failed the profit target, which is counted separately. Marking it
        here too would double-count the same failure.
        """
        m = consistency_breach_mask(self.sims, 0.30)
        self.assertFalse(m[2])

    def test_no_threshold_flags_nothing(self):
        m = consistency_breach_mask(self.sims, None)
        self.assertFalse(m.any())

    def test_mask_shape(self):
        self.assertEqual(consistency_breach_mask(self.sims, 0.3).shape, (3,))

    def test_no_warnings_on_zero_totals(self):
        sims = np.array([[0., 0.], [100., 100.]])
        with np.errstate(all='raise'):
            m = consistency_breach_mask(sims, 0.30)
        self.assertFalse(m[0])


class TestBootstrapStats(unittest.TestCase):

    def setUp(self):
        self.sims = np.array([
            [100., 100., 100., 100.],
            [900., 50., 30., 20.],
            [800., 100., 60., 40.],
            [-100., -100., 50., 20.],
        ])

    def test_breach_rate_excludes_inapplicable_paths(self):
        """
        Denominator is the paths that made money. 2 of 3 evaluable paths
        breach; dividing by all 4 would understate it as 0.5.
        """
        s = consistency_stats(self.sims, 0.30)
        self.assertEqual(s['n_evaluable'], 3)
        self.assertAlmostEqual(s['breach_rate'], 2.0 / 3.0)
        self.assertAlmostEqual(s['breach_rate_all_paths'], 0.5)

    def test_no_threshold_reports_unevaluated(self):
        s = consistency_stats(self.sims, None)
        self.assertFalse(s['evaluated'])
        self.assertIsNone(s['breach_rate'])

    def test_variants_travel_with_the_stats(self):
        s = consistency_stats(self.sims, 0.30)
        self.assertTrue(s['variants_not_modelled'])


class TestBurstVsSteady(unittest.TestCase):
    """
    The System A concern, made concrete: a burst strategy and a steady one
    with identical total profit are treated very differently.
    """

    def test_identical_totals_opposite_verdicts(self):
        burst = [2000., 100., 50., 50., 50.]
        steady = [450., 450., 450., 450., 450.]
        self.assertAlmostEqual(sum(burst), sum(steady))

        rb = check_consistency(burst, threshold=0.30)
        rs = check_consistency(steady, threshold=0.30)

        self.assertFalse(rb.passed)
        self.assertTrue(rs.passed)

    def test_burst_needs_more_days_to_comply(self):
        """A single big day can only be diluted by adding other days."""
        big = 1000.
        for n_extra, expect_pass in [(2, False), (30, True)]:
            series = [big] + [200.] * n_extra
            r = check_consistency(series, threshold=0.30)
            self.assertEqual(r.passed, expect_pass,
                             f"{n_extra} extra days -> {r.best_day_share:.3f}")


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('\n' + '=' * 68)
    print(f"  ran {result.testsRun} | failures {len(result.failures)} | "
          f"errors {len(result.errors)} | skipped {len(result.skipped)}")
    print('=' * 68)
    if result.skipped:
        print("  SKIPS ARE FAILURES:")
        for case, reason in result.skipped:
            print(f"    - {case}: {reason}")
    return 0 if not (result.failures or result.errors or result.skipped) else 1


if __name__ == '__main__':
    sys.exit(main())