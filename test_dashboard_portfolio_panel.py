# ==============================================================================
# test_dashboard_portfolio_panel.py
# ==============================================================================
# Imports are hard errors; the runner treats skips as failures.
# ==============================================================================

import sys
import unittest
from typing import Optional, TypeVar

from canonical_result import CanonicalResult
import firm_rules
from firm_rules import Capability, FirmRules, IMPLEMENTED, ftmo, generic_trailing

import dashboard_portfolio_panel as P
from dashboard_portfolio_panel import (
    ComparisonTable, MetricCell, apply_firm_form, build_comparison,
    build_firm_form, comparison_caption, firm_status_line, try_merge,
    unavailable_column,
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



def cr(sid, ret=10.0, sharpe=1.2, dd=7.0, wr=55.0, pf=1.4, n=100,
       source='trade_list'):
    c = CanonicalResult(
        strategy_id=sid, strategy_name=sid, total_return_pct=ret,
        sharpe_ratio=sharpe, max_drawdown_pct=dd, win_rate=wr,
        profit_factor=pf, total_trades=n, starting_value=100_000.0,
    )
    c.returns_source = source
    return c


# ==============================================================================
# COMPARISON
# ==============================================================================

class TestComparisonBasics(unittest.TestCase):

    def test_columns_in_order_with_portfolio_last(self):
        t = build_comparison([cr('A'), cr('B')], portfolio=cr('portfolio'))
        self.assertEqual([c.key for c in t.columns], ['A', 'B', 'portfolio'])
        self.assertTrue(t.columns[-1].is_portfolio)
        self.assertTrue(t.has_portfolio)

    def test_no_portfolio_means_no_deltas(self):
        t = build_comparison([cr('A'), cr('B')])
        self.assertFalse(t.has_portfolio)
        self.assertEqual(t.deltas, {})

    def test_every_metric_has_a_cell(self):
        t = build_comparison([cr('A')])
        for k in t.metric_keys:
            self.assertIn(k, t.columns[0].cells)

    def test_missing_metric_renders_as_unavailable_not_zero(self):
        c = cr('A')
        c.sharpe_ratio = None
        t = build_comparison([c])
        cell = t.columns[0].cells['sharpe_ratio']
        self.assertFalse(cell.available)
        self.assertIsNone(cell.raw)
        self.assertNotEqual(cell.text, '0.00')

    def test_label_override(self):
        t = build_comparison([cr('A')], labels={'A': 'Trend Follower'})
        self.assertEqual(t.columns[0].label, 'Trend Follower')


class TestDeltas(unittest.TestCase):
    """Delta is portfolio minus BEST individual, direction-aware."""

    def setUp(self):
        self.a = cr('A', ret=12.4, sharpe=1.35, dd=8.2, wr=54.0, pf=1.42)
        self.b = cr('B', ret=9.1, sharpe=1.62, dd=5.4, wr=61.0, pf=1.55)

    def test_higher_is_better_metric(self):
        p = cr('P', ret=21.0)
        t = build_comparison([self.a, self.b], portfolio=p)
        d = t.deltas['total_return_pct']
        self.assertAlmostEqual(not_none(d.raw), 21.0 - 12.4, places=6)
        self.assertTrue(d.better)

    def test_lower_is_better_metric_inverts(self):
        """Drawdown: a positive delta is WORSE."""
        p = cr('P', dd=6.1)
        t = build_comparison([self.a, self.b], portfolio=p)
        d = t.deltas['max_drawdown_pct']
        self.assertAlmostEqual(not_none(d.raw), 6.1 - 5.4, places=6)
        self.assertFalse(d.better)

    def test_lower_is_better_improvement(self):
        p = cr('P', dd=4.0)
        t = build_comparison([self.a, self.b], portfolio=p)
        self.assertTrue(t.deltas['max_drawdown_pct'].better)

    def test_baseline_is_best_not_mean(self):
        """
        Against the mean return (10.75) a 12.0 portfolio would look good.
        Against the best (12.4) it is worse. The stricter comparison is the
        honest one.
        """
        p = cr('P', ret=12.0)
        t = build_comparison([self.a, self.b], portfolio=p)
        d = t.deltas['total_return_pct']
        self.assertLess(not_none(d.raw), 0)
        self.assertFalse(d.better)

    def test_exact_tie_is_neither(self):
        p = cr('P', ret=12.4)
        t = build_comparison([self.a, self.b], portfolio=p)
        self.assertIsNone(t.deltas['total_return_pct'].better)

    def test_descriptive_metrics_get_no_verdict(self):
        """Trade count is not better or worse, just different."""
        t = build_comparison([self.a, self.b], portfolio=cr('P', n=500))
        self.assertIsNone(t.deltas['total_trades'].better)

    def test_delta_unavailable_when_no_individual_has_the_metric(self):
        a, b = cr('A'), cr('B')
        a.sharpe_ratio = b.sharpe_ratio = None
        t = build_comparison([a, b], portfolio=cr('P', sharpe=2.0))
        self.assertIsNone(t.deltas['sharpe_ratio'].raw)
        self.assertEqual(t.deltas['sharpe_ratio'].text, '--')

    def test_unavailable_individuals_excluded_from_baseline(self):
        bad = unavailable_column('C', 'no trade ledger')
        t = build_comparison([self.a, self.b], portfolio=cr('P', ret=21.0))
        t.columns.insert(2, bad)
        self.assertAlmostEqual(
            not_none(t.deltas['total_return_pct'].raw), 8.6, places=6)


class TestUnavailableColumns(unittest.TestCase):

    def test_unavailable_column_keeps_its_slot(self):
        """
        Dropping a failed strategy would make 'failed to load' look identical
        to 'never selected'.
        """
        col = unavailable_column('X', 'empty trade_list')
        self.assertFalse(col.available)
        self.assertEqual(col.reason, 'empty trade_list')
        self.assertTrue(col.cells)

    def test_result_without_id_becomes_unavailable(self):
        anon = CanonicalResult(total_return_pct=5.0)
        t = build_comparison([anon])
        self.assertFalse(t.columns[0].available)
        self.assertIn('strategy_id', t.columns[0].reason)

    def test_caption_counts_failures(self):
        t = build_comparison([cr('A')])
        t.columns.append(unavailable_column('B', 'boom'))
        self.assertIn('could not be loaded', comparison_caption(t))

    def test_empty_selection_caption(self):
        self.assertIn('No strategies', comparison_caption(ComparisonTable()))


class TestProvenance(unittest.TestCase):

    def test_synthetic_source_raises_a_note(self):
        t = build_comparison([cr('A', source='synthetic')])
        self.assertTrue(any('synthetic' in n for n in t.notes))
        self.assertFalse(t.columns[0].provenance_ok)

    def test_real_source_is_silent(self):
        t = build_comparison([cr('A', source='trade_list')])
        self.assertEqual(t.notes, [])
        self.assertTrue(t.columns[0].provenance_ok)


# ==============================================================================
# FIRM RULES FORM
# ==============================================================================

class TestFirmForm(unittest.TestCase):

    def test_fields_carry_current_values(self):
        fields, _ = build_firm_form(ftmo())
        by = {f.name: f for f in fields}
        self.assertAlmostEqual(by['max_daily_loss_pct'].value, 0.05)
        self.assertEqual(by['reset_timezone'].value, 'Europe/Prague')

    def test_unimplemented_capabilities_are_locked(self):
        _, toggles = build_firm_form(ftmo())
        locked = {t.capability for t in toggles if t.locked}
        self.assertIn('trailing_drawdown_eod', locked)
        self.assertIn('weekend_holding_ban', locked)

    def test_consistency_rule_is_now_unlocked(self):
        """It moved from locked to editable when the code landed."""
        _, toggles = build_firm_form(ftmo())
        by = {t.capability: t for t in toggles}
        self.assertFalse(by['consistency_rule'].locked)

    def test_implemented_capabilities_are_unlocked(self):
        _, toggles = build_firm_form(ftmo())
        unlocked = {t.capability for t in toggles if not t.locked}
        for cap in IMPLEMENTED:
            self.assertIn(cap.value, unlocked)

    def test_locked_toggles_explain_themselves(self):
        _, toggles = build_firm_form(ftmo())
        for t in toggles:
            if t.locked:
                self.assertTrue(t.reason,
                                f"{t.capability} is locked with no reason shown")

    def test_active_capabilities_reflected(self):
        _, toggles = build_firm_form(ftmo())
        on = {t.capability for t in toggles if t.enabled}
        self.assertIn('static_drawdown', on)
        self.assertNotIn('trailing_drawdown_eod', on)

    def test_mutually_exclusive_toggles_are_grouped(self):
        _, toggles = build_firm_form(ftmo())
        groups = {t.capability: t.group for t in toggles}
        self.assertEqual(groups['static_drawdown'], 'drawdown')
        self.assertEqual(groups['trailing_drawdown_eod'], 'drawdown')


class TestApplyForm(unittest.TestCase):

    def test_valid_submission(self):
        rules, fields = apply_firm_form({
            'firm_name': 'NewFirm',
            'max_daily_loss_pct': '0.04',
            'max_total_drawdown_pct': '0.08',
            'min_trading_days': '5',
            'max_calendar_days': '',
            'consistency_max_day_pct': '',
            'reset_timezone': 'UTC',
        })
        rules = not_none(rules, 'valid submission should build rules')
        self.assertEqual(rules.firm_name, 'NewFirm')
        self.assertAlmostEqual(rules.max_daily_loss_pct, 0.04)
        self.assertIsNone(rules.max_calendar_days)
        self.assertFalse(any(f.error for f in fields))

    def test_blank_optional_becomes_none(self):
        rules, _ = apply_firm_form({'max_calendar_days': '',
                                    'consistency_max_day_pct': ''})
        rules = not_none(rules)
        self.assertIsNone(rules.max_calendar_days)
        self.assertIsNone(rules.consistency_max_day_pct)

    def test_non_numeric_flagged_on_the_right_field(self):
        rules, fields = apply_firm_form({'max_daily_loss_pct': 'five percent'})
        self.assertIsNone(rules)
        bad = [f for f in fields if f.error]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0].name, 'max_daily_loss_pct')

    def test_percent_typo_surfaces_as_field_error_not_crash(self):
        rules, fields = apply_firm_form({'max_daily_loss_pct': '5'})
        self.assertIsNone(rules)
        self.assertTrue(any(f.error for f in fields))

    def test_cross_field_error_lands_somewhere_visible(self):
        rules, fields = apply_firm_form({
            'max_daily_loss_pct': '0.15',
            'max_total_drawdown_pct': '0.10',
        })
        self.assertIsNone(rules)
        errs = [f for f in fields if f.error]
        self.assertTrue(errs)
        self.assertIn('unreachable', errs[0].error)

    def test_capability_selection_applied(self):
        rules, _ = apply_firm_form(
            {'max_daily_loss_pct': '0.05'},
            capabilities=['static_drawdown', 'daily_loss_closed_only'],
        )
        rules = not_none(rules)
        self.assertIn(Capability.DAILY_LOSS_CLOSED_ONLY,
                      rules.required_capabilities)
        self.assertFalse(rules.includes_floating_pnl)

    def test_unknown_capability_rejected(self):
        rules, fields = apply_firm_form({}, capabilities=['does_not_exist'])
        self.assertIsNone(rules)
        self.assertTrue(any(f.name == 'required_capabilities' and f.error
                            for f in fields))

    def test_conflicting_capabilities_rejected(self):
        rules, fields = apply_firm_form(
            {}, capabilities=['static_drawdown', 'trailing_drawdown_eod'])
        self.assertIsNone(rules)
        self.assertTrue(any(f.error for f in fields))

    def test_form_and_direct_construction_reject_identically(self):
        """The dashboard must not be a laxer door than the constructor."""
        with self.assertRaises(ValueError):
            FirmRules(max_daily_loss_pct=5.0)
        rules, _ = apply_firm_form({'max_daily_loss_pct': '5.0'})
        self.assertIsNone(rules)


class TestStatusLine(unittest.TestCase):

    def test_complete_profile_is_green(self):
        s = firm_status_line(ftmo())
        self.assertTrue(s['complete'])
        self.assertEqual(s['tone'], 'green')
        self.assertEqual(s['n_unchecked'], 0)

    def test_partial_profile_is_amber_and_names_gaps(self):
        s = firm_status_line(generic_trailing())
        self.assertFalse(s['complete'])
        self.assertEqual(s['tone'], 'amber')
        self.assertIn('trailing_drawdown_eod', s['unchecked'])
        self.assertIn('PARTIAL', s['text'])

    def test_consistency_number_no_longer_downgrades_status(self):
        """A configured consistency cap is now actually checked."""
        s = firm_status_line(FirmRules(consistency_max_day_pct=0.30))
        self.assertTrue(s['complete'])
        self.assertEqual(s['tone'], 'green')
        self.assertNotIn('consistency_rule', s['unchecked'])


# ==============================================================================
# MERGE WRAPPER
# ==============================================================================

class TestTryMerge(unittest.TestCase):

    def test_failure_is_data_not_exception(self):
        out = try_merge([cr('only_one')], ftmo())
        self.assertFalse(out['ok'])
        self.assertIn('at least 2', out['reason'])

    def test_missing_ledger_reported_with_reason(self):
        out = try_merge([cr('A'), cr('B')], ftmo())
        self.assertFalse(out['ok'])
        self.assertIn('trade_list', out['reason'])

    def test_successful_merge_payload(self):
        from test_portfolio_merge import make_result, ACCOUNT
        a = make_result('A', [1200, -3000, 500, 800])
        b = make_result('B', [-400, -2800, 1100, -300])
        out = try_merge([a, b], ftmo(), ACCOUNT)
        self.assertTrue(out['ok'])
        self.assertEqual(out['same_day_loss_days'], 1)
        self.assertLess(out['worst_day_pct'], -5.0)
        self.assertTrue(out['warnings'])

    def test_unchecked_rules_propagate_to_the_page(self):
        from test_portfolio_merge import make_result, ACCOUNT
        out = try_merge([make_result('A', [10, 20, 30]),
                         make_result('B', [40, 50, 60])],
                        generic_trailing(), ACCOUNT)
        self.assertTrue(out['ok'])
        self.assertIn('trailing_drawdown_eod', out['unchecked'])


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
    ok = not (result.failures or result.errors or result.skipped)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())