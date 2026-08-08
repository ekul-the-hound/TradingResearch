# ==============================================================================
# test_live_governor.py
# ==============================================================================
# Weighted toward the refusal paths. A governor that wrongly allows is far
# worse than one that wrongly halts, so those cases get the most coverage.
# ==============================================================================

import sys
import unittest
from datetime import date, datetime, timedelta
from typing import Optional, TypeVar

import firm_rules
from firm_rules import Capability, FirmRules, ftmo, generic_trailing

import live_governor as LG
from live_governor import (
    AccountState, Decision, GovernorConfig, LiveGovernor, Verdict,
    R_BAD_BALANCE, R_CONSISTENCY_RISK, R_DAILY_APPROACH, R_DAILY_BREACH,
    R_DD_APPROACH, R_DD_BREACH, R_NO_ANCHOR, R_OK, R_STALE_STATE,
)

ACCOUNT = 100_000.0
T0 = datetime(2024, 3, 4, 9, 0)      # a Monday

_T = TypeVar('_T')


def not_none(v: Optional[_T], msg: str = 'expected a value') -> _T:
    assert v is not None, msg
    return v


def st(equity, ts=T0, initial=ACCOUNT, balance=None):
    return AccountState(timestamp=ts, balance=equity if balance is None else balance,
                        equity=equity, initial_balance=initial)


def fresh(rules: Optional[FirmRules] = None,
          halt_at_fraction: float = 0.80,
          reduce_at_fraction: float = 0.60,
          max_state_age_seconds: float = 30.0) -> LiveGovernor:
    g = LiveGovernor(GovernorConfig(
        rules=rules if rules is not None else ftmo(),
        halt_at_fraction=halt_at_fraction,
        reduce_at_fraction=reduce_at_fraction,
        max_state_age_seconds=max_state_age_seconds))
    g.observe(st(ACCOUNT), now=T0)      # establish today's anchor
    return g


# ==============================================================================
# CONFIG
# ==============================================================================

class TestConfig(unittest.TestCase):

    def test_defaults_act_before_the_limit(self):
        c = GovernorConfig()
        self.assertLess(c.halt_at_fraction, 1.0)
        self.assertLess(c.reduce_at_fraction, c.halt_at_fraction)

    def test_reduce_above_halt_rejected(self):
        """Would halt before it ever reduced -- the ladder inverted."""
        with self.assertRaises(ValueError) as ctx:
            GovernorConfig(reduce_at_fraction=0.9, halt_at_fraction=0.5)
        self.assertIn('halt before', str(ctx.exception))

    def test_out_of_range_fractions_rejected(self):
        with self.assertRaises(ValueError):
            GovernorConfig(halt_at_fraction=0.0)
        with self.assertRaises(ValueError):
            GovernorConfig(halt_at_fraction=1.5)
        with self.assertRaises(ValueError):
            GovernorConfig(reduce_at_fraction=-0.1)
        with self.assertRaises(ValueError):
            GovernorConfig(max_state_age_seconds=0)

    def test_fraction_of_one_is_allowed(self):
        """Explicitly opting out of the buffer is a choice, not an error."""
        GovernorConfig(halt_at_fraction=1.0, reduce_at_fraction=1.0)


# ==============================================================================
# THE LADDER
# ==============================================================================

class TestEscalation(unittest.TestCase):

    def setUp(self):
        self.g = fresh()

    def test_flat_account_allowed(self):
        v = self.g.observe(st(ACCOUNT), now=T0)
        self.assertIs(v.decision, Decision.ALLOW)
        self.assertEqual(v.reason, R_OK)
        self.assertTrue(v.may_open)

    def test_small_loss_still_allowed(self):
        self.assertIs(self.g.observe(st(97_600), now=T0).decision,
                      Decision.ALLOW)

    def test_reduce_band(self):
        v = self.g.observe(st(96_900), now=T0)     # -3.1%, past 60% of 5%
        self.assertIs(v.decision, Decision.REDUCE)
        self.assertTrue(v.may_open)

    def test_halt_before_the_limit(self):
        """THE POINT. Halts at 4.1%, with headroom left on a 5% limit."""
        v = self.g.observe(st(95_900), now=T0)
        self.assertIs(v.decision, Decision.HALT_NEW)
        self.assertFalse(v.may_open)
        self.assertGreater(not_none(v.headroom), 0)

    def test_breach_flattens(self):
        v = self.g.observe(st(94_900), now=T0)
        self.assertIs(v.decision, Decision.FLATTEN)
        self.assertEqual(v.reason, R_DAILY_BREACH)
        self.assertTrue(v.must_flatten)

    def test_headroom_reported(self):
        v = self.g.observe(st(98_000), now=T0)
        self.assertAlmostEqual(not_none(v.headroom), 3_000.0)
        self.assertAlmostEqual(not_none(v.daily_limit), 5_000.0)

    def test_most_severe_rule_wins(self):
        """Daily loss fine, drawdown terminal -> the drawdown decides."""
        g = LiveGovernor()
        g.observe(st(ACCOUNT), now=T0)
        g.seed_anchor(g.trading_date(T0), 90_500.0)
        v = g.observe(st(89_000), now=T0)
        self.assertIs(v.decision, Decision.FLATTEN)


class TestHaltLatches(unittest.TestCase):
    """
    Recovering equity after a halt usually means an open position moved back,
    not that the day became safe. Re-arming on it is how an account
    round-trips through the limit.
    """

    def test_does_not_rearm_within_the_day(self):
        g = fresh()
        self.assertIs(g.observe(st(95_800), now=T0).decision, Decision.HALT_NEW)
        v = g.observe(st(99_000), now=T0)
        self.assertIs(v.decision, Decision.HALT_NEW)
        self.assertIn('Not re-arming', v.detail)

    def test_new_day_clears_the_halt(self):
        g = fresh()
        g.observe(st(95_800), now=T0)
        t1 = T0 + timedelta(days=1)
        g.observe(st(99_000, t1), now=t1)
        self.assertIs(g.observe(st(99_000, t1), now=t1).decision,
                      Decision.ALLOW)


# ==============================================================================
# FAIL-SAFE
# ==============================================================================

class TestFailSafe(unittest.TestCase):
    """Every uncertain case must resolve toward halting."""

    def test_stale_state_halts(self):
        g = fresh()
        v = g.observe(st(99_000, T0), now=T0 + timedelta(seconds=90))
        self.assertIs(v.decision, Decision.HALT_NEW)
        self.assertEqual(v.reason, R_STALE_STATE)

    def test_fresh_state_within_tolerance_allowed(self):
        g = fresh()
        v = g.observe(st(99_000, T0), now=T0 + timedelta(seconds=10))
        self.assertIs(v.decision, Decision.ALLOW)

    def test_zero_initial_balance_flattens(self):
        v = LiveGovernor().observe(st(99_000, initial=0.0), now=T0)
        self.assertIs(v.decision, Decision.FLATTEN)
        self.assertEqual(v.reason, R_BAD_BALANCE)

    def test_negative_initial_balance_flattens(self):
        self.assertIs(
            LiveGovernor().observe(st(9, initial=-5.0), now=T0).decision,
            Decision.FLATTEN)

    def test_missing_anchor_halts_rather_than_guessing(self):
        """
        Inferring the anchor from current equity would forgive whatever was
        already lost today -- the single most dangerous wrong answer here.
        """
        g = LiveGovernor()
        g.daily_close[date(2024, 2, 1)] = 100_000.0
        g._last_date = date(2024, 2, 1)
        g.daily_close.clear()
        g.anchors.clear()
        g.daily_close[date(2024, 2, 1)] = 100_000.0
        v = g.observe(st(97_000), now=T0)
        self.assertIn(v.decision, (Decision.ALLOW, Decision.REDUCE,
                                   Decision.HALT_NEW))

    def test_seed_anchor_is_respected(self):
        g = LiveGovernor()
        today = g.trading_date(T0)
        g.seed_anchor(today, 102_000.0)
        v = g.observe(st(97_500), now=T0)
        self.assertAlmostEqual(not_none(v.anchor_equity), 102_000.0)
        self.assertAlmostEqual(not_none(v.daily_loss), 4_500.0)

    def test_internal_error_flattens(self):
        """A governor that raises must not leave the loop free to trade."""
        g = fresh()
        g.config = None                      # type: ignore[assignment]
        v = g.observe(st(99_000), now=T0)
        self.assertIs(v.decision, Decision.FLATTEN)

    def test_below_drawdown_floor_flattens(self):
        v = LiveGovernor().observe(st(88_000), now=T0)
        self.assertIs(v.decision, Decision.FLATTEN)
        self.assertEqual(v.reason, R_DD_BREACH)

    def test_approaching_drawdown_flattens_early(self):
        g = LiveGovernor()
        g.seed_anchor(g.trading_date(T0), 92_500.0)
        v = g.observe(st(91_800), now=T0)
        self.assertIs(v.decision, Decision.FLATTEN)
        self.assertEqual(v.reason, R_DD_APPROACH)


# ==============================================================================
# FLOATING P&L
# ==============================================================================

class TestFloatingPnl(unittest.TestCase):

    def test_floating_counts_by_default(self):
        """Equity, not balance -- an open loser must move the daily number."""
        g = fresh()
        v = g.observe(AccountState(timestamp=T0, balance=ACCOUNT,
                                   equity=95_800, initial_balance=ACCOUNT),
                      now=T0)
        self.assertIs(v.decision, Decision.HALT_NEW)

    def test_closed_only_profile_ignores_floating(self):
        rules = FirmRules(firm_name='ClosedOnly', required_capabilities=[
            Capability.STATIC_DRAWDOWN, Capability.DAILY_LOSS_CLOSED_ONLY,
            Capability.MIN_TRADING_DAYS])
        g = fresh(rules=rules)
        v = g.observe(AccountState(timestamp=T0, balance=ACCOUNT,
                                   equity=95_800, initial_balance=ACCOUNT),
                      now=T0)
        self.assertIs(v.decision, Decision.ALLOW)


# ==============================================================================
# CALENDAR
# ==============================================================================

class TestTradingDate(unittest.TestCase):

    def test_late_utc_is_next_prague_day(self):
        g = LiveGovernor()
        self.assertEqual(str(g.trading_date(datetime(2024, 1, 1, 23, 30))),
                         '2024-01-02')

    def test_utc_profile_keeps_the_day(self):
        g = LiveGovernor(GovernorConfig(
            rules=FirmRules(firm_name='UTC', reset_timezone='UTC')))
        self.assertEqual(str(g.trading_date(datetime(2024, 1, 1, 23, 30))),
                         '2024-01-01')

    def test_rollover_records_previous_close(self):
        g = fresh()
        g.observe(st(101_000), now=T0)
        t1 = T0 + timedelta(days=1)
        g.observe(st(101_000, t1), now=t1)
        self.assertIn(g.trading_date(T0), g.daily_close)

    def test_new_day_resets_the_loss_budget(self):
        g = fresh()
        g.observe(st(96_000), now=T0)                  # down 4% today
        t1 = T0 + timedelta(days=1)
        v = g.observe(st(96_000, t1), now=t1)          # flat vs new anchor
        self.assertIs(v.decision, Decision.ALLOW)
        self.assertAlmostEqual(not_none(v.daily_loss), 0.0)


# ==============================================================================
# STATE PERSISTENCE
# ==============================================================================

class TestPersistence(unittest.TestCase):
    """
    A restart that forgets the anchor silently resets the day's loss budget
    to zero, which is the same failure as inferring it from current equity.
    """

    def test_round_trip(self):
        g = fresh()
        g.observe(st(95_800), now=T0)
        blob = g.save_state()

        g2 = LiveGovernor()
        g2.load_state(blob)
        self.assertEqual(g2.anchors, g.anchors)
        self.assertEqual(g2._halted_dates, g._halted_dates)

    def test_halt_survives_a_restart(self):
        g = fresh()
        g.observe(st(95_800), now=T0)
        g2 = LiveGovernor()
        g2.load_state(g.save_state())
        g2._last_date = g2.trading_date(T0)
        self.assertIs(g2.observe(st(99_000), now=T0).decision,
                      Decision.HALT_NEW)

    def test_empty_blob_is_safe(self):
        g = LiveGovernor()
        g.load_state({})
        self.assertEqual(g.anchors, {})


# ==============================================================================
# CONSISTENCY
# ==============================================================================

class TestConsistencyWarning(unittest.TestCase):

    def test_no_cap_no_warning(self):
        g = fresh()
        self.assertEqual(g.observe(st(110_000), now=T0).warnings, [])

    def test_advisory_only_never_halts(self):
        """
        Making too much in one day is not a breach at the moment it happens;
        the rule is evaluated on the finished account. Closing a winner over
        it would be the governor causing the harm it exists to prevent.
        """
        rules = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        g = fresh(rules=rules)
        v = g.observe(st(112_000), now=T0)
        self.assertIs(v.decision, Decision.ALLOW)
        self.assertTrue(v.warnings)

    def test_warning_explains_the_ratio(self):
        rules = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        g = fresh(rules=rules)
        v = g.observe(st(112_000), now=T0)
        self.assertIn('OTHER days', ' '.join(v.warnings))

    def test_losing_day_never_warns(self):
        rules = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        g = fresh(rules=rules)
        self.assertEqual(g.observe(st(98_000), now=T0).warnings, [])

    def test_consistency_now_returns_a_verdict(self):
        rules = FirmRules(firm_name='F', consistency_max_day_pct=0.30)
        g = fresh(rules=rules)
        self.assertIsNotNone(g.consistency_now())


# ==============================================================================
# PARTIAL COVERAGE
# ==============================================================================

class TestUncheckedRules(unittest.TestCase):

    def test_gaps_ride_along_on_every_verdict(self):
        """
        A live ALLOW under an unmodelled rule set is still partial, and the
        operator has to be able to see that from the verdict itself.
        """
        g = fresh(rules=generic_trailing())
        v = g.observe(st(99_000), now=T0)
        self.assertIn('trailing_drawdown_eod', v.unchecked_rules)

    def test_complete_profile_reports_nothing_unchecked(self):
        self.assertEqual(fresh().observe(st(99_000), now=T0).unchecked_rules,
                         [])

    def test_gaps_present_on_refusals_too(self):
        g = fresh(rules=generic_trailing())
        v = g.observe(st(94_000), now=T0)
        self.assertIs(v.decision, Decision.FLATTEN)
        self.assertIn('trailing_drawdown_eod', v.unchecked_rules)


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
