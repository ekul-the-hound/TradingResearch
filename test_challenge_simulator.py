# ==============================================================================
# test_challenge_simulator.py
# ==============================================================================

import sys
import unittest
from typing import Optional, TypeVar

import numpy as np
import pandas as pd

import challenge_simulator as CS
from challenge_simulator import (
    FAIL_CONSISTENCY, FAIL_DAILY_LOSS, FAIL_DRAWDOWN, FAIL_NOT_REACHED,
    FAIL_TIME_LIMIT, MIN_CONDITIONAL_SAMPLE, PASSED, ChallengeResult,
    StageSpec, StageStats, simulate_challenge, walk_stage,
)
from firm_rules import FirmRules, ftmo, generic_trailing

ACCOUNT = 100_000.0

_T = TypeVar('_T')


def not_none(value: Optional[_T], msg: str = 'expected a value, got None') -> _T:
    assert value is not None, msg
    return value


def rules_no_min_days(**kw):
    """Most walk tests want to isolate one rule, not trip over min days."""
    kw.setdefault('firm_name', 'Test')
    return FirmRules(**kw)


STAGE = StageSpec(name='challenge', profit_target_pct=0.10,
                  max_days=None, min_trading_days=1)


# ==============================================================================
# WALKING ONE STAGE
# ==============================================================================

class TestEarlyStopping(unittest.TestCase):
    """
    A trader who reaches the target stops. The remainder of the path never
    happens, so it must not be evaluated.
    """

    def test_stops_the_day_the_target_is_hit(self):
        daily = np.array([4000., 4000., 4000., 4000., 4000.])
        out = walk_stage(daily, ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], PASSED)
        self.assertEqual(out['days'], 3)          # 12k >= 10k on day 3

    def test_later_disaster_is_never_experienced(self):
        """
        THE POINT. Hit +10% on day 3, then the path craters. A fixed-window
        evaluation would see the final equity and fail it. The trader had
        already stopped.
        """
        daily = np.array([4000., 4000., 4000., -50_000., -50_000.])
        out = walk_stage(daily, ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], PASSED)
        self.assertEqual(len(out['realised']), 3)

    def test_min_trading_days_delays_the_stop(self):
        stage = StageSpec(name='c', profit_target_pct=0.10, min_trading_days=4)
        daily = np.array([11_000., 100., 100., 100., 100.])
        out = walk_stage(daily, ACCOUNT, stage, ftmo())
        self.assertEqual(out['outcome'], PASSED)
        self.assertEqual(out['days'], 4)

    def test_zero_pnl_days_do_not_count_as_trading_days(self):
        stage = StageSpec(name='c', profit_target_pct=0.10, min_trading_days=3)
        daily = np.array([11_000., 0., 0., 50., 50.])
        out = walk_stage(daily, ACCOUNT, stage, ftmo())
        self.assertEqual(out['days'], 5)


class TestTargetBoundary(unittest.TestCase):
    """
    100_000 * (1.0 + 0.10) is 110000.00000000001 in IEEE 754. A plain >=
    therefore judges a path landing on exactly +10.00% to be short, by one
    hundred-billionth of a cent. Money is counted to the cent.
    """

    def test_exactly_on_target_passes(self):
        out = walk_stage(np.array([10_000.0]), ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], PASSED)

    def test_one_cent_short_does_not_pass(self):
        out = walk_stage(np.array([9_999.98]), ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], FAIL_NOT_REACHED)

    def test_helper_is_cent_accurate(self):
        target = ACCOUNT * (1.0 + 0.10)
        self.assertGreater(target, 110_000.0)          # the artifact itself
        self.assertTrue(CS._reached(110_000.0, target))
        self.assertFalse(CS._reached(109_999.98, target))

    def test_boundary_holds_across_account_sizes(self):
        for size in (10_000.0, 25_000.0, 50_000.0, 100_000.0, 200_000.0):
            stage = StageSpec(name='c', profit_target_pct=0.10,
                              min_trading_days=1)
            out = walk_stage(np.array([size * 0.10]), size, stage, ftmo())
            self.assertEqual(out['outcome'], PASSED, f'failed at {size}')


class TestFailureAttribution(unittest.TestCase):

    def test_daily_loss_breach(self):
        out = walk_stage(np.array([-6000., 100.]), ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], FAIL_DAILY_LOSS)
        self.assertEqual(out['days'], 1)

    def test_daily_loss_at_exactly_the_limit_breaches(self):
        out = walk_stage(np.array([-5000.]), ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], FAIL_DAILY_LOSS)

    def test_just_inside_the_limit_survives(self):
        out = walk_stage(np.array([-4999.]), ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], FAIL_NOT_REACHED)

    def test_drawdown_breach(self):
        """Accumulated losses, none of which breaches the daily rule."""
        daily = np.array([-3000.] * 4)
        out = walk_stage(daily, ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], FAIL_DRAWDOWN)
        self.assertEqual(out['days'], 4)

    def test_daily_loss_wins_when_both_would_trigger(self):
        """
        A single day that breaks the daily limit AND drops through the floor
        is attributed to the daily limit -- that is the one that fires first
        intraday.
        """
        out = walk_stage(np.array([-20_000.]), ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], FAIL_DAILY_LOSS)

    def test_time_limit(self):
        stage = StageSpec(name='c', profit_target_pct=0.10, max_days=3,
                          min_trading_days=1)
        out = walk_stage(np.array([100.] * 30), ACCOUNT, stage, ftmo())
        self.assertEqual(out['outcome'], FAIL_TIME_LIMIT)
        self.assertEqual(out['days'], 3)

    def test_no_time_limit_runs_the_path_out(self):
        out = walk_stage(np.array([100.] * 10), ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], FAIL_NOT_REACHED)
        self.assertEqual(out['days'], 10)


class TestConsistencyInteraction(unittest.TestCase):
    """
    Early stopping concentrates profit into fewer days, which makes the best
    day a larger share of the total. The two rules pull against each other.
    """

    def test_fast_win_violates_the_cap(self):
        daily = np.array([10_500., 100., 100., 100.])
        r = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        out = walk_stage(daily, ACCOUNT, STAGE, r)
        self.assertEqual(out['outcome'], FAIL_CONSISTENCY)

    def test_same_path_passes_without_the_cap(self):
        daily = np.array([10_500., 100., 100., 100.])
        out = walk_stage(daily, ACCOUNT, STAGE, ftmo())
        self.assertEqual(out['outcome'], PASSED)

    def test_even_accumulation_satisfies_the_cap(self):
        daily = np.array([2100.] * 5)
        r = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        out = walk_stage(daily, ACCOUNT, STAGE, r)
        self.assertEqual(out['outcome'], PASSED)

    def test_consistency_only_checked_at_the_win(self):
        """A path that never reaches the target fails on the target, not here."""
        r = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        out = walk_stage(np.array([500., 10., 10.]), ACCOUNT, STAGE, r)
        self.assertEqual(out['outcome'], FAIL_NOT_REACHED)


# ==============================================================================
# STAGE SPEC
# ==============================================================================

class TestStageSpec(unittest.TestCase):

    def test_from_rules_reads_the_profile(self):
        s = StageSpec.from_rules(ftmo(), 'challenge')
        self.assertAlmostEqual(s.profit_target_pct, 0.10)
        self.assertEqual(s.min_trading_days, 4)

    def test_verification_has_the_lower_target(self):
        c = StageSpec.from_rules(ftmo(), 'challenge')
        v = StageSpec.from_rules(ftmo(), 'verification')
        self.assertLess(v.profit_target_pct, c.profit_target_pct)

    def test_unknown_phase_rejected(self):
        with self.assertRaises(ValueError):
            StageSpec.from_rules(ftmo(), 'nonexistent')

    def test_inherits_max_calendar_days(self):
        r = FirmRules(firm_name='Timed', max_calendar_days=30)
        self.assertEqual(StageSpec.from_rules(r, 'challenge').max_days, 30)


# ==============================================================================
# FULL SIMULATION
# ==============================================================================

def winning_sims(n=200, days=20, per_day=800.0):
    return np.full((n, days), per_day)


def losing_sims(n=200, days=20, per_day=-100.0):
    return np.full((n, days), per_day)


class TestSimulation(unittest.TestCase):

    def test_stages_run_in_order(self):
        r = simulate_challenge(winning_sims(), ACCOUNT, ftmo())
        self.assertEqual([s.name for s in r.stages],
                         ['challenge', 'verification'])

    def test_everyone_passes_a_generous_path(self):
        r = simulate_challenge(winning_sims(), ACCOUNT, ftmo())
        self.assertAlmostEqual(r.p_funded, 1.0)
        self.assertEqual(r.n_funded, 200)

    def test_nobody_passes_a_losing_path(self):
        r = simulate_challenge(losing_sims(), ACCOUNT, ftmo())
        self.assertAlmostEqual(r.p_funded, 0.0)
        self.assertEqual(r.n_funded, 0)

    def test_stage_two_only_sees_stage_one_survivors(self):
        r = simulate_challenge(losing_sims(), ACCOUNT, ftmo())
        self.assertEqual(r.stages[1].n_entered, r.stages[0].n_passed)

    def test_pass_rate_is_none_when_nobody_entered(self):
        """None, not 0.0 -- nobody failed, nobody tried."""
        r = simulate_challenge(losing_sims(), ACCOUNT, ftmo())
        self.assertEqual(r.stages[1].n_entered, 0)
        self.assertIsNone(r.stages[1].pass_rate)

    def test_outcomes_sum_to_entrants(self):
        r = simulate_challenge(winning_sims(n=50), ACCOUNT, ftmo())
        for s in r.stages:
            if s.n_entered:
                self.assertEqual(sum(s.outcomes.values()), s.n_entered)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            simulate_challenge(np.array([[]]), ACCOUNT, ftmo())
        with self.assertRaises(ValueError):
            simulate_challenge(np.array([1.0, 2.0]), ACCOUNT, ftmo())

    def test_deterministic_under_seed(self):
        sims = np.random.RandomState(1).normal(200, 1500, (400, 25))
        a = simulate_challenge(sims, ACCOUNT, ftmo(), random_seed=9)
        b = simulate_challenge(sims, ACCOUNT, ftmo(), random_seed=9)
        self.assertEqual(a.p_funded, b.p_funded)


class TestHonestReporting(unittest.TestCase):

    def test_zero_pass_rate_gives_undefined_attempts(self):
        """
        Infinite, not merely large. Returning a big finite number would read
        as 'expensive' when the answer is 'never'.
        """
        r = simulate_challenge(losing_sims(), ACCOUNT, ftmo())
        self.assertIsNone(r.expected_attempts())
        self.assertIsNone(r.expected_fee(155.0))
        self.assertTrue(any('undefined' in w for w in r.warnings))

    def test_expected_attempts_is_the_reciprocal(self):
        sims = np.vstack([winning_sims(n=20), losing_sims(n=80)])
        r = simulate_challenge(sims, ACCOUNT, ftmo())
        p = r.p_funded
        self.assertGreater(p, 0)
        self.assertAlmostEqual(not_none(r.expected_attempts()), 1.0 / p)

    def test_small_conditional_sample_warns(self):
        """A pass rate off a handful of survivors is noise, and says so."""
        n_win = MIN_CONDITIONAL_SAMPLE - 5
        sims = np.vstack([winning_sims(n=n_win), losing_sims(n=300)])
        r = simulate_challenge(sims, ACCOUNT, ftmo())
        self.assertFalse(r.stages[1].reliable)
        self.assertTrue(any('too small a sample' in w for w in r.warnings))

    def test_large_sample_does_not_warn(self):
        r = simulate_challenge(winning_sims(n=500), ACCOUNT, ftmo())
        self.assertTrue(r.stages[1].reliable)
        self.assertFalse(any('too small' in w for w in r.warnings))

    def test_unchecked_rules_propagate(self):
        r = simulate_challenge(winning_sims(), ACCOUNT, generic_trailing())
        self.assertFalse(r.is_complete)
        self.assertIn('trailing_drawdown_eod', r.unchecked_rules)

    def test_complete_profile_reports_complete(self):
        self.assertTrue(
            simulate_challenge(winning_sims(), ACCOUNT, ftmo()).is_complete)


class TestDayAccounting(unittest.TestCase):

    def test_median_is_over_per_path_totals(self):
        """
        Not the sum of per-stage medians: the path at the median of stage one
        is generally not the one at the median of stage two.
        """
        r = simulate_challenge(winning_sims(per_day=800.0), ACCOUNT, ftmo())
        self.assertEqual(len(r.funded_path_days), r.n_funded)
        # 13 days for +10%, then 7 more for +5%, at 800/day.
        self.assertAlmostEqual(not_none(r.median_days_to_funded()), 20.0)

    def test_no_funded_paths_means_no_median(self):
        r = simulate_challenge(losing_sims(), ACCOUNT, ftmo())
        self.assertIsNone(r.median_days_to_funded())
        self.assertIsNone(r.days_to_funded_percentile(90))

    def test_p_funded_within_uses_all_attempts_as_denominator(self):
        """
        'If I start today, am I funded in N days' -- failing does not stop
        the clock, so the denominator is every attempt.
        """
        sims = np.vstack([winning_sims(n=50), losing_sims(n=50)])
        r = simulate_challenge(sims, ACCOUNT, ftmo())
        self.assertAlmostEqual(r.p_funded_within(365), r.p_funded)
        self.assertEqual(r.p_funded_within(1), 0.0)

    def test_percentiles_ordered(self):
        r = simulate_challenge(
            np.random.RandomState(3).normal(400, 1200, (600, 40)),
            ACCOUNT, ftmo())
        if r.n_funded > 10:
            self.assertLessEqual(not_none(r.days_to_funded_percentile(50)),
                                 not_none(r.days_to_funded_percentile(90)))


class TestSerialisation(unittest.TestCase):

    def test_to_dict_shape(self):
        d = simulate_challenge(winning_sims(n=60), ACCOUNT, ftmo()).to_dict()
        for k in ('firm', 'p_funded', 'stages', 'is_complete',
                  'p_funded_within_90d', 'median_days_to_funded'):
            self.assertIn(k, d)
        self.assertEqual(len(d['stages']), 2)

    def test_summary_mentions_every_stage(self):
        s = simulate_challenge(winning_sims(n=60), ACCOUNT, ftmo()).summary()
        self.assertIn('challenge', s)
        self.assertIn('verification', s)
        self.assertIn('P(funded)', s)

    def test_partial_profile_summary_says_partial(self):
        s = simulate_challenge(winning_sims(n=60), ACCOUNT,
                               generic_trailing()).summary()
        self.assertIn('PARTIAL', s)


# ==============================================================================
# EARLY STOPPING FOR SINGLE STRATEGIES
# ==============================================================================

def stats_frame(end_equities, start='2024-01-01'):
    dates = pd.date_range(start, periods=len(end_equities), freq='D')
    return pd.DataFrame({
        'date': [d.date() for d in dates],
        'end_equity': [float(e) for e in end_equities],
    })


class TestFindEarlyStopDate(unittest.TestCase):

    def test_finds_the_first_qualifying_close(self):
        f = stats_frame([102_000, 105_000, 109_000, 111_000, 112_000])
        d = CS.find_early_stop_date(f, ACCOUNT, 0.10, min_trading_days=1)
        self.assertEqual(str(d), '2024-01-04')

    def test_min_trading_days_defers_the_stop(self):
        f = stats_frame([111_000, 111_500, 112_000, 112_500])
        d = CS.find_early_stop_date(f, ACCOUNT, 0.10, min_trading_days=4)
        self.assertEqual(str(d), '2024-01-04')

    def test_none_when_target_never_closed_above(self):
        f = stats_frame([101_000, 103_000, 108_000])
        self.assertIsNone(
            CS.find_early_stop_date(f, ACCOUNT, 0.10, min_trading_days=1))

    def test_none_on_empty_or_missing_column(self):
        self.assertIsNone(CS.find_early_stop_date(None, ACCOUNT, 0.1, 1))
        self.assertIsNone(
            CS.find_early_stop_date(pd.DataFrame(), ACCOUNT, 0.1, 1))
        self.assertIsNone(CS.find_early_stop_date(
            pd.DataFrame({'date': [1], 'other': [2]}), ACCOUNT, 0.1, 1))

    def test_exact_target_qualifies(self):
        f = stats_frame([110_000])
        self.assertIsNotNone(
            CS.find_early_stop_date(f, ACCOUNT, 0.10, min_trading_days=1))


class TestEarlyStopResult(unittest.TestCase):

    def test_rescue_share_none_when_nothing_ran(self):
        r = CS.EarlyStopResult()
        self.assertIsNone(r.rescue_share)

    def test_rescue_share_computed(self):
        r = CS.EarlyStopResult(n_evaluated=200, n_rescued=20)
        self.assertAlmostEqual(not_none(r.rescue_share), 0.1)

    def test_error_summary_says_unavailable(self):
        self.assertIn('unavailable',
                      CS.EarlyStopResult(error='boom').summary())


class TestEarlyStopGuards(unittest.TestCase):
    """Refusals, not zeros. A 0.0 pass rate must mean 'it failed', never
    'we could not run it'."""

    def test_too_few_trades_is_an_error_not_a_zero(self):
        df = pd.DataFrame({'exit_date': pd.to_datetime(['2024-01-01'])})
        r = CS.simulate_pass_rate_early_stop(None, df)
        self.assertIsNotNone(r.error)
        self.assertIn('Insufficient trades', not_none(r.error))

    def test_none_trades_is_an_error(self):
        self.assertIsNotNone(
            CS.simulate_pass_rate_early_stop(None, None).error)

    def test_unknown_phase_is_an_error(self):
        df = pd.DataFrame({'exit_date': pd.to_datetime(
            ['2024-01-0%d' % i for i in range(1, 9)])})
        r = CS.simulate_pass_rate_early_stop(None, df, phase='nope')
        self.assertIn('no phase', not_none(r.error))

    def test_unchecked_rules_propagate(self):
        df = pd.DataFrame({'exit_date': pd.to_datetime(
            ['2024-01-0%d' % i for i in range(1, 9)])})
        r = CS.simulate_pass_rate_early_stop(
            None, df, phase='nope', rules=generic_trailing())
        self.assertIn('trailing_drawdown_eod', r.unchecked_rules)


class TestConsistencyInEarlyStopPath(unittest.TestCase):
    """
    checker.validate() implements daily loss, drawdown, min days and profit
    target. It knows nothing about consistency. Without a separate step, a
    caller could supply a cap, be told unchecked_rules was empty, and get a
    pass rate in which the rule never ran.
    """

    def test_daily_pnl_reconstructed_from_closes(self):
        f = stats_frame([101_000, 100_500, 103_000])
        pnl = CS.daily_pnl_from_stats(f, ACCOUNT)
        self.assertEqual([round(x) for x in pnl], [1000, -500, 2500])

    def test_daily_pnl_empty_when_unavailable(self):
        self.assertEqual(CS.daily_pnl_from_stats(None, ACCOUNT), [])
        self.assertEqual(CS.daily_pnl_from_stats(pd.DataFrame(), ACCOUNT), [])

    def test_no_cap_always_ok(self):
        f = stats_frame([110_500, 110_600])
        self.assertTrue(CS._consistency_ok(f, ACCOUNT, ftmo()))

    def test_concentrated_win_breaches_the_cap(self):
        f = stats_frame([110_400, 110_500, 110_600])   # day 1 is ~98%
        r = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        self.assertFalse(CS._consistency_ok(f, ACCOUNT, r))

    def test_even_win_satisfies_the_cap(self):
        f = stats_frame([102_500, 105_000, 107_500, 110_000])
        r = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        self.assertTrue(CS._consistency_ok(f, ACCOUNT, r))

    def test_unreconstructable_series_is_not_a_breach(self):
        """
        A path only reaches this check by having met the target. If the day
        series cannot be rebuilt, failing it would invent a breach.
        """
        r = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        self.assertTrue(CS._consistency_ok(None, ACCOUNT, r))


class TestFirmLocalDates(unittest.TestCase):
    """One implementation, shared. Two copies drifting apart would silently
    assign trades to different days in different modules."""

    def test_late_utc_rolls_to_next_prague_day(self):
        s = pd.Series(pd.to_datetime(['2024-01-01T23:30:00']))
        self.assertEqual(str(CS.firm_local_dates(s, 'Europe/Prague')[0]),
                         '2024-01-02')

    def test_utc_keeps_the_same_day(self):
        s = pd.Series(pd.to_datetime(['2024-01-01T23:30:00']))
        self.assertEqual(str(CS.firm_local_dates(s, 'UTC')[0]), '2024-01-01')

    def test_portfolio_merge_delegates_here(self):
        import portfolio_merge
        s = pd.Series(pd.to_datetime(['2024-01-01T23:30:00']))
        self.assertEqual(
            list(portfolio_merge._firm_local_dates(s, 'Europe/Prague')),
            list(CS.firm_local_dates(s, 'Europe/Prague')))


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