# ==============================================================================
# test_holdout_guard.py
# ==============================================================================
# Phase 2, Item 10.
#
#   python test_holdout_guard.py
#
# Organised around the ways a holdout normally gets destroyed, because those
# are the properties that matter -- not that the happy path works.
#
# Import failures are HARD ERRORS. A skip is not a pass.
# ==============================================================================

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from holdout_guard import (DEFAULT_HOLDOUT_FRACTION, HoldoutExhausted,
                           HoldoutGuard, HoldoutViolation)


def frame(n=1000, start='2020-01-01'):
    idx = pd.date_range(start, periods=n, freq='D')
    close = 100 * np.exp(np.cumsum(np.random.RandomState(0).normal(0, 0.01, n)))
    return pd.DataFrame({'open': close, 'high': close * 1.01,
                         'low': close * 0.99, 'close': close,
                         'volume': 1000.0}, index=idx)


class Base(unittest.TestCase):
    def setUp(self):
        fd, self.ledger = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        os.unlink(self.ledger)
        self.df = frame()
        self.cutoff = HoldoutGuard.suggest_cutoff(self.df.index)

    def tearDown(self):
        for p in (self.ledger, f"{self.ledger}.tmp"):
            try:
                os.unlink(p)
            except OSError:
                pass

    def guard(self, max_peeks=3):
        return HoldoutGuard.initialise(self.cutoff, max_peeks=max_peeks,
                                       ledger_path=self.ledger)


class TestDefaultDeny(Base):
    """The protection has to be the default, not an option."""

    def test_data_is_truncated_without_a_token(self):
        g = self.guard()
        out = g.enforce(self.df, symbol='EUR-USD')
        self.assertLess(len(out), len(self.df))
        self.assertTrue((out.index < self.cutoff).all())

    def test_truncation_keeps_roughly_the_training_fraction(self):
        g = self.guard()
        out = g.enforce(self.df, symbol='EUR-USD')
        kept = len(out) / len(self.df)
        self.assertAlmostEqual(kept, 1 - DEFAULT_HOLDOUT_FRACTION, delta=0.02)

    def test_token_grants_the_full_series(self):
        g = self.guard()
        tok = g.request_access('final validation', 'variant_07')
        out = g.enforce(self.df, symbol='EUR-USD', token=tok)
        self.assertEqual(len(out), len(self.df))

    def test_unconfigured_guard_does_not_silently_truncate(self):
        """No cutoff pinned means no protection -- and must not look like protection."""
        g = HoldoutGuard.load(self.ledger)
        self.assertFalse(g.is_configured)
        self.assertEqual(len(g.enforce(self.df)), len(self.df))
        self.assertIn('NOT CONFIGURED', g.report())

    def test_non_datetime_index_is_passed_through(self):
        g = self.guard()
        plain = self.df.reset_index(drop=True)
        self.assertEqual(len(g.enforce(plain)), len(plain))


class TestPinnedCutoff(Base):
    """
    A boundary that moves as data arrives quietly promotes holdout into
    training, and every individual run still looks correct.
    """

    def test_cutoff_persists_across_instances(self):
        g1 = self.guard()
        g2 = HoldoutGuard.load(self.ledger)
        self.assertEqual(g1.cutoff, g2.cutoff)

    def test_reinitialising_with_a_different_date_is_refused(self):
        self.guard()
        with self.assertRaises(HoldoutViolation):
            HoldoutGuard.initialise(self.cutoff + pd.Timedelta(days=100),
                                    ledger_path=self.ledger)

    def test_reinitialising_with_the_same_date_is_idempotent(self):
        self.guard()
        g = HoldoutGuard.initialise(self.cutoff, ledger_path=self.ledger)
        self.assertEqual(g.cutoff, self.cutoff)

    def test_cutoff_does_not_move_when_more_data_arrives(self):
        g = self.guard()
        original = g.cutoff
        longer = frame(n=2000)
        g.enforce(longer, symbol='EUR-USD')
        self.assertEqual(HoldoutGuard.load(self.ledger).cutoff, original)

    def test_force_is_refused_after_peeks_without_explicit_force(self):
        g = self.guard()
        g.request_access('look', 's1')
        with self.assertRaises(HoldoutViolation):
            HoldoutGuard.initialise(self.cutoff + pd.Timedelta(days=5),
                                    ledger_path=self.ledger)


class TestBudget(Base):
    """An unlimited holdout you are trusted not to overuse is just a test set."""

    def test_peeks_decrement(self):
        g = self.guard(max_peeks=3)
        self.assertEqual(g.peeks_remaining, 3)
        g.request_access('r1', 's1')
        self.assertEqual(g.peeks_remaining, 2)

    def test_exhaustion_is_refused_and_permanent(self):
        g = self.guard(max_peeks=2)
        g.request_access('r1', 's1')
        g.request_access('r2', 's2')
        self.assertTrue(g.is_burned)
        with self.assertRaises(HoldoutExhausted):
            g.request_access('r3', 's3')

    def test_restarting_the_process_does_not_reset_the_budget(self):
        """The whole point of an on-disk ledger."""
        g = self.guard(max_peeks=2)
        g.request_access('r1', 's1')
        fresh = HoldoutGuard.load(self.ledger)
        self.assertEqual(fresh.peeks_used, 1)
        self.assertEqual(fresh.peeks_remaining, 1)

    def test_exhaustion_message_names_what_spent_it(self):
        g = self.guard(max_peeks=1)
        g.request_access('final check', 'variant_07')
        try:
            g.request_access('another', 'variant_08')
            self.fail("should have raised")
        except HoldoutExhausted as e:
            self.assertIn('variant_07', str(e))

    def test_reason_and_strategy_are_mandatory(self):
        g = self.guard()
        for bad in ('', '   ', None):
            with self.assertRaises(ValueError):
                g.request_access(bad, 's1')
            with self.assertRaises(ValueError):
                g.request_access('reason', bad)


class TestLedger(Base):

    def test_every_peek_is_recorded(self):
        g = self.guard()
        g.request_access('validating the momentum variant', 'variant_07')
        with open(self.ledger, encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(len(data['peeks']), 1)
        self.assertEqual(data['peeks'][0]['strategy_id'], 'variant_07')
        self.assertIn('momentum', data['peeks'][0]['reason'])

    def test_outcome_can_be_attached(self):
        g = self.guard()
        tok = g.request_access('r', 's1')
        g.record_outcome(tok, sharpe=1.4, passed=True)
        with open(self.ledger, encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['peeks'][0]['outcome']['sharpe'], 1.4)

    def test_fabricated_token_is_rejected(self):
        from holdout_guard import HoldoutToken
        g = self.guard()
        fake = HoldoutToken('made_up', 's1', 'r', '2026-01-01')
        with self.assertRaises(HoldoutViolation):
            g.enforce(self.df, symbol='EUR-USD', token=fake)

    def test_token_cannot_be_reused_for_the_same_symbol(self):
        g = self.guard()
        tok = g.request_access('r', 's1')
        g.enforce(self.df, symbol='EUR-USD', token=tok)
        with self.assertRaises(HoldoutViolation):
            g.enforce(self.df, symbol='EUR-USD', token=tok)

    def test_corrupt_ledger_raises_rather_than_resetting(self):
        """Silently treating a broken ledger as empty would un-burn the holdout."""
        self.guard()
        with open(self.ledger, 'w', encoding='utf-8') as f:
            f.write('{not valid json')
        with self.assertRaises(RuntimeError):
            HoldoutGuard.load(self.ledger)

    def test_report_lists_history(self):
        g = self.guard()
        g.request_access('checking variant_07 out of sample', 'variant_07')
        r = g.report()
        self.assertIn('variant_07', r)
        self.assertIn('1/3', r)


class TestDeflation(Base):
    """Looking N times is N trials, whether or not a search was intended."""

    def test_no_deflation_before_any_peek(self):
        g = self.guard()
        d = g.deflate_sharpe(1.5, n_obs=500)
        self.assertAlmostEqual(d['deflated_sharpe'], 1.5)

    def test_deflation_grows_with_peeks(self):
        g = self.guard(max_peeks=10)
        g.request_access('r1', 's1')
        one = g.deflate_sharpe(1.5, n_obs=500)['deflated_sharpe']
        for i in range(2, 6):
            g.request_access(f'r{i}', f's{i}')
        five = g.deflate_sharpe(1.5, n_obs=500)['deflated_sharpe']
        self.assertLess(five, one, "more looks must mean a larger haircut")
        self.assertLess(five, 1.5)

    def test_more_observations_mean_a_smaller_haircut(self):
        g = self.guard(max_peeks=10)
        for i in range(4):
            g.request_access(f'r{i}', f's{i}')
        short = g.deflate_sharpe(1.5, n_obs=100)['haircut']
        long_ = g.deflate_sharpe(1.5, n_obs=5000)['haircut']
        self.assertLess(long_, short)

    def test_extra_trials_are_counted(self):
        g = self.guard()
        a = g.deflate_sharpe(1.5, n_obs=500, extra_trials=0)['haircut']
        b = g.deflate_sharpe(1.5, n_obs=500, extra_trials=200)['haircut']
        self.assertGreater(b, a, "an explicit search must also be deflated")


class TestSuggestCutoff(Base):

    def test_fraction_is_respected(self):
        c = HoldoutGuard.suggest_cutoff(self.df.index, fraction=0.30)
        held = (self.df.index >= c).sum()
        self.assertAlmostEqual(held / len(self.df), 0.30, delta=0.02)

    def test_too_little_data_raises(self):
        with self.assertRaises(ValueError):
            HoldoutGuard.suggest_cutoff(pd.date_range('2024-01-01', periods=5))

    def test_suggestion_alone_does_not_pin(self):
        HoldoutGuard.suggest_cutoff(self.df.index)
        self.assertFalse(HoldoutGuard.load(self.ledger).is_configured)


def main():
    print("=" * 70)
    print("HOLDOUT GUARD - TEST SUITE")
    print("=" * 70)
    print("A research loop destroys its own OOS set by using it, not by cheating.")
    print("Tested: default deny, a pinned boundary, a finite budget, an")
    print("append-only ledger, and deflation for the looking already done.")
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
