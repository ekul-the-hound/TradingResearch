# ==============================================================================
# test_synthetic_returns.py
# ==============================================================================
# Proves Phase 0 Item 5. No market data required.
#
#   python test_synthetic_returns.py
#
# Central property: a statistic can never be computed on data that was not
# derived from executed trades, unless someone opted in on purpose.
# ==============================================================================

import sys
import unittest

import numpy as np

import canonical_result as cr_mod
from canonical_result import CanonicalResult, SyntheticReturnsError


def summary_only_result(strategy_id='synth', bars=252):
    """A backtest result with summary stats but NO trade list."""
    return {
        'strategy_name': 'test',
        'total_return_pct': 20,
        'sharpe_ratio': 1.5,
        'max_drawdown_pct': 10,
        'total_trades': 30,
        'bars_tested': bars,
        'starting_value': 10000,
    }


def result_with_trades(n=40, pnl=25.0):
    return {
        'strategy_name': 'test',
        'total_return_pct': 10,
        'sharpe_ratio': 1.2,
        'max_drawdown_pct': 5,
        'total_trades': n,
        'bars_tested': 252,
        'starting_value': 10000,
        'trades': [{'pnl': pnl} for _ in range(n)],
    }


class TestFabricationIsGone(unittest.TestCase):

    def setUp(self):
        cr_mod.ALLOW_SYNTHETIC_RETURNS = False

    def test_no_trades_yields_no_returns(self):
        """Was: 252 Gaussian draws. Now: nothing."""
        cr = CanonicalResult.from_backtest(summary_only_result(), strategy_id='s1')
        self.assertIsNone(cr.returns)
        self.assertEqual(cr.returns_source, 'none')
        self.assertFalse(cr.has_real_returns)

    def test_returns_is_none_not_zeros(self):
        """
        Zeros would let a caller compute a Sharpe of 0/0 and carry on.
        None makes accidental use fail immediately.
        """
        cr = CanonicalResult.from_backtest(summary_only_result(), strategy_id='s1')
        self.assertIsNone(cr.returns, "must be None, not an array of zeros")

    def test_trade_list_still_produces_real_returns(self):
        cr = CanonicalResult.from_backtest(result_with_trades(), strategy_id='s2')
        self.assertIsNotNone(cr.returns)
        self.assertEqual(cr.returns_source, 'trade_list')
        self.assertTrue(cr.has_real_returns)
        self.assertEqual(len(cr.returns), 40)

    def test_opt_in_switch_restores_old_behaviour_but_marks_it(self):
        cr_mod.ALLOW_SYNTHETIC_RETURNS = True
        try:
            cr = CanonicalResult.from_backtest(summary_only_result(), strategy_id='s3')
            self.assertIsNotNone(cr.returns)
            self.assertEqual(len(cr.returns), 252)
            self.assertTrue(cr.returns_synthetic)
            self.assertEqual(cr.returns_source, 'synthetic')
            self.assertFalse(cr.has_real_returns,
                             "opting in must not make fabricated data look real")
        finally:
            cr_mod.ALLOW_SYNTHETIC_RETURNS = False

    def test_switch_defaults_off(self):
        """
        Read the source rather than reloading the module. importlib.reload
        rebinds SyntheticReturnsError and CanonicalResult to new class objects,
        which silently breaks isinstance/assertRaises in every test that runs
        afterwards -- a nasty way for a suite to lie to you.
        """
        import inspect, re
        src = inspect.getsource(cr_mod)
        # Match module-level assignments only. A naive substring check trips on
        # the usage example inside the module's own comment block.
        self.assertTrue(re.search(r'^ALLOW_SYNTHETIC_RETURNS\s*=\s*False\s*$', src, re.M),
                        "the switch must be defined at module level and default to off")
        self.assertIsNone(re.search(r'^ALLOW_SYNTHETIC_RETURNS\s*=\s*True\s*$', src, re.M),
                          "the module must not enable fabrication for itself")


class TestWhyFabricationWasDangerous(unittest.TestCase):
    """
    Documents the mechanism, so nobody re-adds it thinking it was harmless.
    Fabricated returns are Gaussian, which is the most favourable possible
    input to every detector in the overfitting stack.
    """

    def test_fabricated_series_has_no_skew_or_excess_kurtosis(self):
        cr_mod.ALLOW_SYNTHETIC_RETURNS = True
        try:
            cr = CanonicalResult.from_backtest(
                summary_only_result(bars=5000), strategy_id='s4')
            r = cr.returns
            m, s = r.mean(), r.std()
            skew = float(np.mean(((r - m) / s) ** 3))
            exkurt = float(np.mean(((r - m) / s) ** 4)) - 3.0
            # Deflated / Probabilistic Sharpe penalise negative skew and fat
            # tails. Gaussian draws exhibit neither, so they are never deflated.
            self.assertAlmostEqual(skew, 0.0, delta=0.15)
            self.assertAlmostEqual(exkurt, 0.0, delta=0.30)
        finally:
            cr_mod.ALLOW_SYNTHETIC_RETURNS = False

    def test_fabricated_series_is_stationary_across_splits(self):
        """
        CSCV/PBO compare in-sample and out-of-sample block rankings. Draws from
        a constant mean and variance agree across every split by construction,
        which drives measured overfitting toward zero.
        """
        cr_mod.ALLOW_SYNTHETIC_RETURNS = True
        try:
            cr = CanonicalResult.from_backtest(
                summary_only_result(bars=4000), strategy_id='s5')
            blocks = np.array_split(cr.returns, 8)
            means = [b.mean() for b in blocks]
            spread = (max(means) - min(means)) / (abs(np.mean(means)) + 1e-12)
            self.assertLess(spread, 5.0,
                            "block means barely differ - splits cannot disagree")
        finally:
            cr_mod.ALLOW_SYNTHETIC_RETURNS = False

    def test_fabrication_was_deterministic_so_it_looked_reproducible(self):
        cr_mod.ALLOW_SYNTHETIC_RETURNS = True
        try:
            a = CanonicalResult.from_backtest(summary_only_result(), strategy_id='same')
            b = CanonicalResult.from_backtest(summary_only_result(), strategy_id='same')
            np.testing.assert_array_equal(a.returns, b.returns)
        finally:
            cr_mod.ALLOW_SYNTHETIC_RETURNS = False


class TestRequireReturnsGate(unittest.TestCase):

    def setUp(self):
        cr_mod.ALLOW_SYNTHETIC_RETURNS = False

    def test_raises_when_returns_missing(self):
        cr = CanonicalResult.from_backtest(summary_only_result(), strategy_id='s6')
        with self.assertRaises(SyntheticReturnsError) as ctx:
            cr.require_returns('CSCV')
        self.assertIn('CSCV', str(ctx.exception))
        self.assertIn('trade extraction', str(ctx.exception))

    def test_raises_on_synthetic_returns(self):
        cr_mod.ALLOW_SYNTHETIC_RETURNS = True
        try:
            cr = CanonicalResult.from_backtest(summary_only_result(), strategy_id='s7')
        finally:
            cr_mod.ALLOW_SYNTHETIC_RETURNS = False
        with self.assertRaises(SyntheticReturnsError):
            cr.require_returns('Deflated Sharpe Ratio')

    def test_returns_the_array_when_real(self):
        cr = CanonicalResult.from_backtest(result_with_trades(), strategy_id='s8')
        out = cr.require_returns('bootstrap')
        np.testing.assert_array_equal(out, cr.returns)

    def test_min_length_enforced(self):
        cr = CanonicalResult.from_backtest(result_with_trades(n=5), strategy_id='s9')
        with self.assertRaises(ValueError):
            cr.require_returns('CSCV', min_length=30)

    def test_error_message_says_what_to_do(self):
        cr = CanonicalResult.from_backtest(summary_only_result(), strategy_id='s10')
        try:
            cr.require_returns('PBO')
        except SyntheticReturnsError as e:
            msg = str(e)
            self.assertIn('s10', msg)
            self.assertTrue(len(msg) > 60, "must be actionable, not just 'error'")


class TestProvenanceSurvivesAggregation(unittest.TestCase):
    """
    The laundering path: _aggregate_results concatenated arrays and built the aggregate
    without carrying returns_synthetic, so one synthetic input vanished into a
    clean-looking aggregate.
    """

    def setUp(self):
        cr_mod.ALLOW_SYNTHETIC_RETURNS = False

    def _mixed_inputs(self):
        real = CanonicalResult.from_backtest(result_with_trades(), strategy_id='real')
        cr_mod.ALLOW_SYNTHETIC_RETURNS = True
        try:
            fake = CanonicalResult.from_backtest(summary_only_result(), strategy_id='fake')
        finally:
            cr_mod.ALLOW_SYNTHETIC_RETURNS = False
        return real, fake

    def test_synthetic_input_is_excluded_and_flagged(self):
        try:
            from backtest_adapter import BacktestAdapter
        except Exception:
            self.skipTest("backtest_adapter not importable in this environment")

        real, fake = self._mixed_inputs()
        agg = BacktestAdapter._aggregate_results(
            BacktestAdapter.__new__(BacktestAdapter), [real, fake], {}, 'mix')

        self.assertEqual(agg.returns_source, 'mixed',
                         "aggregate must not claim to be clean")
        self.assertEqual(len(agg.returns), len(real.returns),
                         "synthetic series must be excluded, not concatenated")
        with self.assertRaises(SyntheticReturnsError):
            agg.require_returns('CSCV')

    def test_all_real_aggregate_stays_usable(self):
        try:
            from backtest_adapter import BacktestAdapter
        except Exception:
            self.skipTest("backtest_adapter not importable in this environment")

        a = CanonicalResult.from_backtest(result_with_trades(n=20), strategy_id='a')
        b = CanonicalResult.from_backtest(result_with_trades(n=30), strategy_id='b')
        agg = BacktestAdapter._aggregate_results(
            BacktestAdapter.__new__(BacktestAdapter), [a, b], {}, 'clean')

        self.assertEqual(agg.returns_source, 'trade_list')
        self.assertEqual(len(agg.returns), 50)
        self.assertIsNotNone(agg.require_returns('CSCV'))


def main():
    print("=" * 70)
    print("SYNTHETIC RETURNS - TEST SUITE")
    print("=" * 70)
    print("Old: no trade list -> rng.normal(...) passed off as a return series,")
    print("     then laundered clean through aggregation.")
    print("New: no trade list -> returns is None; require_returns() raises.")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}")
    print("=" * 70)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
