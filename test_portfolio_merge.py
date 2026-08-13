# ==============================================================================
# test_portfolio_merge.py
# ==============================================================================
# Phase 3 test suite: firm_rules.py + portfolio_merge.py
#
# Imports are HARD ERRORS. No skipTest in setUp -- a missing dependency must
# not produce a green suite that ran nothing. The runner at the bottom counts
# skips and exits non-zero if any occurred.
# ==============================================================================

import sys
import unittest
from typing import Optional, TypeVar, cast
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from canonical_result import CanonicalResult
import firm_rules
from firm_rules import (
    Capability, FirmRules, IMPLEMENTED, ftmo, generic_trailing, load_profile,
)
import portfolio_merge
from portfolio_merge import (
    PortfolioMergeError, bootstrap_summary, daily_pnl_matrix, extract_ledger,
    joint_block_bootstrap, merge_strategies, OVERLAP_UNION,
)

ACCOUNT = 100_000.0


# ------------------------------------------------------------------
# Narrowing helper
# ------------------------------------------------------------------
# Some values under test are Optional by design. Passing them straight into
# assertAlmostEqual fails to type-check, and `or 0.0` would paper over a None
# by turning it into a passing comparison against zero. This asserts non-None
# and returns the narrowed value, so the check is real and the following
# assertion is well-typed.
_T = TypeVar('_T')


def not_none(value: Optional[_T], msg: str = 'expected a value, got None') -> _T:
    assert value is not None, msg
    return value



# ==============================================================================
# HELPERS
# ==============================================================================

def make_result(strategy_id, day_pnls, start='2024-01-01', symbol='EURUSD'):
    """
    Build a CanonicalResult with one trade per day.

    day_pnls: list of daily P&L in currency. Index 0 is `start`.
    Trades exit at 15:00 UTC so they land unambiguously on the same Prague day.
    """
    base = cast(pd.Timestamp, pd.Timestamp(start))
    trades = []
    for i, pnl in enumerate(day_pnls):
        exit_dt = base + timedelta(days=i, hours=15)
        trades.append({
            'entry_date': (exit_dt - timedelta(hours=2)).isoformat(),
            'exit_date': exit_dt.isoformat(),
            'entry_price': 1.1000,
            'exit_price': 1.1000 + pnl / 100000.0,
            'size': 1.0,
            'symbol': symbol,
            'pnl': float(pnl),
        })
    return CanonicalResult(
        strategy_id=strategy_id,
        strategy_name=strategy_id,
        symbol=symbol,
        timeframe='M15',
        starting_value=ACCOUNT,
        total_trades=len(trades),
        trade_list=trades,
    )


# ==============================================================================
# FIRM RULES -- numbers
# ==============================================================================

class TestFirmRulesNumbers(unittest.TestCase):

    def test_default_profile_is_ftmo_shaped(self):
        r = ftmo()
        self.assertAlmostEqual(r.max_daily_loss_pct, 0.05)
        self.assertAlmostEqual(r.max_total_drawdown_pct, 0.10)
        self.assertEqual(r.min_trading_days, 4)
        self.assertAlmostEqual(r.profit_targets['challenge'], 0.10)
        self.assertEqual(r.reset_timezone, 'Europe/Prague')

    def test_percent_typo_rejected(self):
        """5 instead of 0.05 is the most likely form-entry mistake."""
        with self.assertRaises(ValueError) as ctx:
            FirmRules(max_daily_loss_pct=5.0)
        self.assertIn('0.05', str(ctx.exception))

    def test_daily_limit_above_total_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            FirmRules(max_daily_loss_pct=0.15, max_total_drawdown_pct=0.10)
        self.assertIn('unreachable', str(ctx.exception))

    def test_negative_and_zero_targets_rejected(self):
        with self.assertRaises(ValueError):
            FirmRules(profit_targets={'challenge': 0.0})
        with self.assertRaises(ValueError):
            FirmRules(min_trading_days=-1)
        with self.assertRaises(ValueError):
            FirmRules(max_calendar_days=0)

    def test_conflicting_drawdown_modes_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            FirmRules(required_capabilities=[
                Capability.STATIC_DRAWDOWN,
                Capability.TRAILING_DRAWDOWN_EOD,
            ])
        self.assertIn('more than one drawdown mode', str(ctx.exception))

    def test_conflicting_daily_modes_rejected(self):
        with self.assertRaises(ValueError):
            FirmRules(required_capabilities=[
                Capability.DAILY_LOSS_INCLUDES_FLOATING,
                Capability.DAILY_LOSS_CLOSED_ONLY,
            ])

    def test_derived_values(self):
        r = ftmo()
        self.assertAlmostEqual(r.daily_loss_limit(100_000), 5_000)
        self.assertAlmostEqual(r.drawdown_floor(100_000), 90_000)
        self.assertAlmostEqual(
            r.profit_target_value(100_000, 'challenge'), 110_000)
        with self.assertRaises(ValueError):
            r.profit_target_value(100_000, 'nonexistent_phase')

    def test_editing_numbers_needs_no_code_change(self):
        """The whole point: a different firm is a different set of floats."""
        r = FirmRules(
            firm_name='OtherFirm',
            max_daily_loss_pct=0.04,
            max_total_drawdown_pct=0.08,
            min_trading_days=5,
            profit_targets={'challenge': 0.08},
        )
        self.assertAlmostEqual(r.daily_loss_limit(50_000), 2_000)
        self.assertAlmostEqual(r.drawdown_floor(50_000), 46_000)


# ==============================================================================
# FIRM RULES -- semantics / honest absence
# ==============================================================================

class TestFirmRulesCapabilities(unittest.TestCase):

    def test_ftmo_profile_is_fully_modelled(self):
        r = ftmo()
        self.assertEqual(r.unsupported(), [])
        self.assertTrue(r.is_fully_modelled)

    def test_trailing_profile_reports_gap_instead_of_pretending(self):
        r = generic_trailing()
        gaps = r.unsupported()
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].capability, Capability.TRAILING_DRAWDOWN_EOD)
        self.assertFalse(r.is_fully_modelled)
        self.assertIn('PARTIAL', r.caveat_line())

    def test_consistency_pct_alone_is_now_honoured(self):
        """
        The threshold is read directly by the checker, independent of
        required_capabilities, so setting it no longer produces a gap.
        """
        r = FirmRules(consistency_max_day_pct=0.30)
        caps = [g.capability for g in r.unsupported()]
        self.assertNotIn(Capability.CONSISTENCY_RULE, caps)
        self.assertTrue(r.is_fully_modelled)

    def test_consistency_rule_is_implemented(self):
        """
        Backed by consistency_rule.check_consistency.

        The assertion is paired deliberately: whitelisted AND callable.
        A capability in IMPLEMENTED with no code behind it is the exact
        false confidence the split exists to prevent.
        """
        import consistency_rule
        self.assertIn(Capability.CONSISTENCY_RULE, IMPLEMENTED)
        self.assertTrue(callable(consistency_rule.check_consistency))

    def test_every_implemented_capability_is_reachable(self):
        """Nothing in IMPLEMENTED should be missing from the enum."""
        for cap in IMPLEMENTED:
            self.assertIsInstance(cap, Capability)

    def test_consistency_pct_range_validated(self):
        with self.assertRaises(ValueError):
            FirmRules(consistency_max_day_pct=30)  # meant 0.30

    def test_every_unimplemented_capability_has_a_note(self):
        for cap in Capability:
            if cap not in IMPLEMENTED:
                self.assertIn(cap, firm_rules.CAPABILITY_NOTES,
                              f"{cap} has no explanation for the dashboard")

    def test_caveat_line_names_the_missing_rules(self):
        r = generic_trailing()
        self.assertIn('trailing_drawdown_eod', r.caveat_line())


class TestFirmRulesSerialisation(unittest.TestCase):

    def test_round_trip(self):
        original = FirmRules(
            firm_name='RoundTrip',
            max_daily_loss_pct=0.03,
            consistency_max_day_pct=0.25,
        )
        restored = FirmRules.from_dict(original.to_dict())
        self.assertEqual(restored.firm_name, 'RoundTrip')
        self.assertAlmostEqual(restored.max_daily_loss_pct, 0.03)
        self.assertAlmostEqual(
            not_none(restored.consistency_max_day_pct), 0.25)
        self.assertEqual(
            [c for c in restored.required_capabilities],
            [c for c in original.required_capabilities],
        )

    def test_to_dict_exposes_gaps_for_the_dashboard(self):
        d = generic_trailing().to_dict()
        self.assertFalse(d['is_fully_modelled'])
        self.assertEqual(len(d['unsupported']), 1)
        self.assertIn('reason', d['unsupported'][0])

    def test_unknown_field_rejected_not_ignored(self):
        d = ftmo().to_dict()
        d['max_dialy_loss_pct'] = 0.02       # typo
        with self.assertRaises(ValueError) as ctx:
            FirmRules.from_dict(d)
        self.assertIn('max_dialy_loss_pct', str(ctx.exception))

    def test_load_profile_rejects_unknown(self):
        self.assertIsInstance(load_profile('ftmo'), FirmRules)
        with self.assertRaises(ValueError):
            load_profile('not_a_firm')


# ==============================================================================
# LEDGER EXTRACTION -- refusals
# ==============================================================================

class TestExtractLedger(unittest.TestCase):

    def test_extracts_real_ledger(self):
        cr = make_result('A', [100, -50, 200])
        led = extract_ledger(cr)
        self.assertEqual(len(led), 3)
        self.assertListEqual(list(led['pnl']), [100.0, -50.0, 200.0])
        self.assertTrue((led['strategy_id'] == 'A').all())

    def test_refuses_empty_trade_list(self):
        cr = CanonicalResult(strategy_id='NoTrades', total_return_pct=15.0)
        with self.assertRaises(PortfolioMergeError) as ctx:
            extract_ledger(cr)
        self.assertIn('empty trade_list', str(ctx.exception))

    def test_refuses_synthetic_returns(self):
        cr = make_result('Synth', [10, 20])
        cr.returns_source = 'synthetic'
        cr.returns_synthetic = True
        with self.assertRaises(PortfolioMergeError) as ctx:
            extract_ledger(cr)
        self.assertIn('synthetic', str(ctx.exception))

    def test_refuses_trade_without_exit_timestamp(self):
        cr = make_result('B', [100])
        del cr.trade_list[0]['exit_date']
        with self.assertRaises(PortfolioMergeError) as ctx:
            extract_ledger(cr)
        self.assertIn('exit timestamp', str(ctx.exception))

    def test_refuses_trade_without_pnl(self):
        cr = make_result('C', [100])
        del cr.trade_list[0]['pnl']
        with self.assertRaises(PortfolioMergeError) as ctx:
            extract_ledger(cr)
        self.assertIn('P&L', str(ctx.exception))


# ==============================================================================
# THE CENTRAL CASE
# ==============================================================================

class TestCombinedBreach(unittest.TestCase):
    """
    Neither strategy breaches the 5% daily limit alone. Together they do.
    This is the entire justification for merging at trade level.
    """

    def setUp(self):
        #            Mon    Tue     Wed
        self.a = make_result('A', [1200, -3000,  500])
        self.b = make_result('B', [-400, -2800, 1100])
        self.rules = ftmo()

    def test_neither_strategy_breaches_alone(self):
        limit = self.rules.daily_loss_limit(ACCOUNT)   # 5000
        for cr in (self.a, self.b):
            worst = min(t['pnl'] for t in cr.trade_list)
            self.assertGreater(abs(worst), 0)
            self.assertLess(abs(worst), limit,
                            f"{cr.strategy_id} breaches alone; bad fixture")

    def test_portfolio_breaches(self):
        res = merge_strategies([self.a, self.b], rules=self.rules,
                               account_size=ACCOUNT)
        worst = res.diagnostics.worst_combined_day_pct
        self.assertLess(worst, -5.0,
                        "combined Tuesday should breach the 5% daily limit")
        self.assertEqual(res.diagnostics.worst_combined_day_date, '2024-01-02')

    def test_breach_is_surfaced_as_a_warning(self):
        res = merge_strategies([self.a, self.b], rules=self.rules,
                               account_size=ACCOUNT)
        self.assertTrue(
            any('breaches' in w for w in res.diagnostics.warnings),
            "a combination-induced breach must be called out explicitly")

    def test_same_day_loss_clustering_counted(self):
        res = merge_strategies([self.a, self.b], rules=self.rules,
                               account_size=ACCOUNT)
        self.assertEqual(res.diagnostics.same_day_loss_days, 1)

    def test_weights_can_model_capital_splitting(self):
        """50/50 dilutes the same trades below the limit."""
        res = merge_strategies(
            [self.a, self.b], rules=self.rules, account_size=ACCOUNT,
            weights={'A': 0.5, 'B': 0.5},
        )
        self.assertGreater(res.diagnostics.worst_combined_day_pct, -5.0)


# ==============================================================================
# MERGE MECHANICS
# ==============================================================================

class TestMergeMechanics(unittest.TestCase):

    def test_requires_two_strategies(self):
        with self.assertRaises(PortfolioMergeError):
            merge_strategies([make_result('A', [10])])

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(PortfolioMergeError) as ctx:
            merge_strategies([make_result('A', [10, 20]),
                              make_result('A', [30, 40])])
        self.assertIn('Duplicate', str(ctx.exception))

    def test_rejects_non_overlapping_windows(self):
        a = make_result('A', [10] * 5, start='2020-01-01')
        b = make_result('B', [10] * 5, start='2023-01-01')
        with self.assertRaises(PortfolioMergeError) as ctx:
            merge_strategies([a, b])
        self.assertIn('no overlapping', str(ctx.exception))

    def test_intersection_truncates_and_reports(self):
        a = make_result('A', [10] * 40, start='2024-01-01')
        b = make_result('B', [10] * 10, start='2024-01-20')
        res = merge_strategies([a, b])
        self.assertLess(res.diagnostics.trades_after_truncation,
                        res.diagnostics.trades_before_truncation)
        self.assertGreater(res.diagnostics.trades_dropped_pct, 0)

    def test_union_warns_about_solo_periods(self):
        a = make_result('A', [10] * 40, start='2024-01-01')
        b = make_result('B', [10] * 10, start='2024-01-20')
        res = merge_strategies([a, b], overlap=OVERLAP_UNION)
        self.assertTrue(any('union' in w for w in res.diagnostics.warnings))

    def test_rejects_bad_overlap_mode(self):
        with self.assertRaises(ValueError):
            merge_strategies([make_result('A', [1, 2]),
                              make_result('B', [3, 4])], overlap='whatever')

    def test_rejects_non_positive_weight(self):
        with self.assertRaises(PortfolioMergeError):
            merge_strategies([make_result('A', [1, 2]),
                              make_result('B', [3, 4])],
                             weights={'A': 0.0})

    def test_unsupported_rules_propagate_into_diagnostics(self):
        res = merge_strategies(
            [make_result('A', [10, 20]), make_result('B', [30, 40])],
            rules=generic_trailing(),
        )
        self.assertFalse(res.is_fully_modelled)
        self.assertEqual(len(res.diagnostics.unsupported_rules), 1)


class TestCanonicalOutput(unittest.TestCase):
    """The merged result must be indistinguishable in kind from a single one."""

    def setUp(self):
        self.res = merge_strategies(
            [make_result('A', [100, -50, 200, 30]),
             make_result('B', [-20, 80, -60, 140])],
            account_size=ACCOUNT,
        )

    def test_returns_provenance_is_real(self):
        cr = self.res.canonical
        self.assertEqual(cr.returns_source, 'trade_list')
        self.assertFalse(cr.returns_synthetic)
        self.assertTrue(cr.has_real_returns)

    def test_require_returns_does_not_raise(self):
        self.res.canonical.require_returns('portfolio test')

    def test_totals_reconcile_with_constituents(self):
        expected = sum([100, -50, 200, 30]) + sum([-20, 80, -60, 140])
        self.assertAlmostEqual(
            self.res.canonical.ending_value, ACCOUNT + expected, places=6)

    def test_trade_count_is_the_sum(self):
        self.assertEqual(self.res.canonical.total_trades, 8)

    def test_members_recorded_in_params(self):
        self.assertListEqual(
            self.res.canonical.strategy_params['members'], ['A', 'B'])

    def test_daily_matrix_shape(self):
        self.assertEqual(list(self.res.daily_pnl.columns), ['A', 'B'])
        self.assertEqual(len(self.res.daily_pnl), 4)


# ==============================================================================
# JOINT BOOTSTRAP
# ==============================================================================

class TestJointBootstrap(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(7)
        n = 200
        # Correlated: both strategies share a common shock component.
        common = rng.normal(0, 600, n)
        a = common + rng.normal(50, 200, n)
        b = common + rng.normal(50, 200, n)
        idx = pd.date_range('2024-01-01', periods=n, freq='D')
        self.daily = pd.DataFrame({'A': a, 'B': b}, index=idx)

    def test_shape(self):
        sims = joint_block_bootstrap(self.daily, n_simulations=200,
                                     window_days=30)
        self.assertEqual(sims.shape, (200, 30))

    def test_not_degenerate(self):
        """The old permutation bug produced identical paths every time."""
        sims = joint_block_bootstrap(self.daily, n_simulations=100,
                                     window_days=30)
        totals = sims.sum(axis=1)
        self.assertGreater(totals.std(), 0,
                           "every simulation identical -- resampler is a no-op")

    def test_deterministic_under_seed(self):
        kw = dict(n_simulations=50, window_days=20, random_seed=123)
        np.testing.assert_array_equal(
            joint_block_bootstrap(self.daily, **kw),
            joint_block_bootstrap(self.daily, **kw),
        )

    def test_joint_resampling_preserves_correlation(self):
        """
        THE POINT OF THE MODULE.

        Joint resampling must produce a fatter left tail than resampling each
        strategy independently, because independent draws quietly assume the
        strategies never lose on the same day.
        """
        sims_joint = joint_block_bootstrap(self.daily, n_simulations=3000,
                                           window_days=30, random_seed=1)

        # Independent: shuffle each strategy's days separately, destroying the
        # same-day link, then re-run the identical bootstrap.
        rng = np.random.RandomState(99)
        broken = self.daily.copy()
        broken['B'] = broken['B'].values[rng.permutation(len(broken))]
        sims_indep = joint_block_bootstrap(broken, n_simulations=3000,
                                           window_days=30, random_seed=1)

        worst_joint = np.percentile(sims_joint.min(axis=1), 5)
        worst_indep = np.percentile(sims_indep.min(axis=1), 5)

        self.assertLess(
            worst_joint, worst_indep,
            "joint resampling should produce worse tail days than independent; "
            "if not, the same-day correlation is being lost"
        )

    def test_refuses_empty_matrix(self):
        with self.assertRaises(PortfolioMergeError):
            joint_block_bootstrap(pd.DataFrame(), n_simulations=10)

    def test_refuses_single_day_history(self):
        one = self.daily.iloc[:1]
        with self.assertRaises(PortfolioMergeError) as ctx:
            joint_block_bootstrap(one, n_simulations=10)
        self.assertIn('single point', str(ctx.exception))

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            joint_block_bootstrap(self.daily, window_days=0)
        with self.assertRaises(ValueError):
            joint_block_bootstrap(self.daily, mean_block_days=0)


class TestBootstrapSummary(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(3)
        idx = pd.date_range('2024-01-01', periods=120, freq='D')
        self.daily = pd.DataFrame({
            'A': rng.normal(120, 700, 120),
            'B': rng.normal(90, 650, 120),
        }, index=idx)
        self.sims = joint_block_bootstrap(self.daily, n_simulations=500,
                                          window_days=30, random_seed=5)

    def test_rates_are_probabilities(self):
        s = bootstrap_summary(self.sims, ACCOUNT, ftmo())
        for k in ('daily_breach_rate', 'drawdown_breach_rate',
                  'survived_rate', 'reached_target_rate',
                  'modelled_pass_rate'):
            self.assertGreaterEqual(s[k], 0.0)
            self.assertLessEqual(s[k], 1.0)

    def test_complete_when_all_rules_modelled(self):
        s = bootstrap_summary(self.sims, ACCOUNT, ftmo())
        self.assertTrue(s['is_complete'])
        self.assertEqual(s['unsupported_rules'], [])

    def test_incomplete_carries_the_caveat(self):
        """
        A pass rate computed under an unmodelled rule set must not be
        presentable as a plain percentage.
        """
        rules = generic_trailing('PartialFirm')
        s = bootstrap_summary(self.sims, ACCOUNT, rules)
        self.assertFalse(s['is_complete'])
        self.assertIn('trailing_drawdown_eod', s['unsupported_rules'])
        self.assertIn('PARTIAL', s['caveat'])

    def test_consistency_lowers_the_pass_rate(self):
        """
        The whole point. A firm with a consistency cap must not report
        the same pass rate as one without.
        """
        loose = ftmo()
        strict = FirmRules(firm_name='BurstHostile',
                           consistency_max_day_pct=0.25)
        a = bootstrap_summary(self.sims, ACCOUNT, loose)
        b = bootstrap_summary(self.sims, ACCOUNT, strict)
        self.assertLessEqual(b['modelled_pass_rate'],
                             a['modelled_pass_rate'])
        self.assertAlmostEqual(b['pass_rate_ignoring_consistency'],
                               a['modelled_pass_rate'], places=9)

    def test_no_threshold_means_no_consistency_penalty(self):
        s = bootstrap_summary(self.sims, ACCOUNT, ftmo())
        self.assertFalse(s['consistency']['evaluated'])
        self.assertAlmostEqual(s['modelled_pass_rate'],
                               s['pass_rate_ignoring_consistency'],
                               places=9)

    def test_agrees_exactly_with_the_challenge_walk(self):
        """
        REGRESSION GUARD. bootstrap_summary and challenge_simulator must not
        drift apart on P(pass) -- they once disagreed by more than a factor of
        two, because this function scanned a fixed window while the walk stops
        at the target. The rule mechanics now live in one place; this pins it.
        """
        import challenge_simulator as CS
        rules = ftmo()
        s = bootstrap_summary(self.sims, ACCOUNT, rules)
        stage = CS.StageSpec.from_rules(rules, 'challenge')
        walked = sum(
            1 for row in self.sims
            if CS.walk_stage(row, ACCOUNT, stage, rules)['outcome'] == CS.PASSED
        ) / len(self.sims)
        self.assertAlmostEqual(s['modelled_pass_rate'], walked, places=12)

    def test_fixed_window_figure_is_kept_but_distinct(self):
        """The old number stays available, under a name that says what it is."""
        s = bootstrap_summary(self.sims, ACCOUNT, ftmo())
        self.assertIn('fixed_window_pass_rate', s)
        self.assertLessEqual(s['fixed_window_pass_rate'],
                             s['modelled_pass_rate'])

    def test_ignoring_consistency_equals_no_cap_run(self):
        """
        passed + consistency-failures should equal the pass rate the same
        paths score with the rule switched off, since walk_stage only raises
        FAIL_CONSISTENCY at the moment a path would otherwise have won.
        """
        capped = bootstrap_summary(
            self.sims, ACCOUNT,
            FirmRules(firm_name='F', consistency_max_day_pct=0.30))
        uncapped = bootstrap_summary(self.sims, ACCOUNT, ftmo())
        self.assertAlmostEqual(capped['pass_rate_ignoring_consistency'],
                               uncapped['modelled_pass_rate'], places=12)

    def test_key_is_named_modelled_pass_rate(self):
        """Naming guard: it is not 'pass_rate', because it isn't one."""
        s = bootstrap_summary(self.sims, ACCOUNT, ftmo())
        self.assertIn('modelled_pass_rate', s)
        self.assertNotIn('pass_rate', s)


# ==============================================================================
# TIMEZONE ANCHORING
# ==============================================================================

class TestDailyAnchoring(unittest.TestCase):

    def test_late_utc_trade_lands_on_next_prague_day(self):
        """23:30 UTC on 1 Jan is 00:30 Prague on 2 Jan -- a different day."""
        cr = CanonicalResult(
            strategy_id='TZ', starting_value=ACCOUNT,
            trade_list=[{
                'entry_date': '2024-01-01T22:00:00',
                'exit_date': '2024-01-01T23:30:00',
                'size': 1.0, 'symbol': 'EURUSD', 'pnl': -100.0,
            }],
        )
        daily = daily_pnl_matrix(extract_ledger(cr), 'Europe/Prague')
        self.assertEqual(str(pd.Timestamp(str(daily.index[0])).date()), '2024-01-02')

    def test_utc_profile_keeps_same_day(self):
        cr = CanonicalResult(
            strategy_id='TZ2', starting_value=ACCOUNT,
            trade_list=[{
                'entry_date': '2024-01-01T22:00:00',
                'exit_date': '2024-01-01T23:30:00',
                'size': 1.0, 'symbol': 'EURUSD', 'pnl': -100.0,
            }],
        )
        daily = daily_pnl_matrix(extract_ledger(cr), 'UTC')
        self.assertEqual(str(pd.Timestamp(str(daily.index[0])).date()), '2024-01-01')


# ==============================================================================
# RUNNER -- skips are failures
# ==============================================================================

def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print('\n' + '=' * 68)
    print(f"  ran {result.testsRun} | failures {len(result.failures)} | "
          f"errors {len(result.errors)} | skipped {len(result.skipped)}")
    print('=' * 68)

    if result.skipped:
        print("  SKIPS ARE TREATED AS FAILURES -- a skipped test is a test "
              "that did not run:")
        for case, reason in result.skipped:
            print(f"    - {case}: {reason}")

    ok = (not result.failures and not result.errors and not result.skipped)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())