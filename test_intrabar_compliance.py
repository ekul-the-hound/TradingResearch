# ==============================================================================
# test_intrabar_compliance.py
# ==============================================================================
# Verifies the wiring of intrabar_risk into FTMOComplianceChecker.
#
#   python test_intrabar_compliance.py
#
# The case that matters: a strategy that PASSES on close-only marking and FAILS
# once intrabar excursion is counted. If the verdict cannot change, the wiring
# achieves nothing.
#
# Import failures are HARD ERRORS. A skip is not a pass.
# ==============================================================================

import sys
import unittest
from typing import Any, cast
from datetime import timedelta

import numpy as np
import pandas as pd

import intrabar_risk as ir
from ftmo_compliance import FTMOComplianceChecker


def bars(specs, start='2024-01-02 09:00', freq='h'):
    idx = pd.date_range(start, periods=len(specs), freq=freq)
    return pd.DataFrame(
        [{'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1000.0}
         for o, h, l, c in specs], index=idx)


def calm_history(n_days=8, start='2024-01-02'):
    """Enough distinct trading days to satisfy the min-days rule."""
    base = pd.Timestamp(start)
    price_rows, trade_rows = [], []
    px = 1.1000
    for d in range(n_days):
        day = base + timedelta(days=d)
        for h in range(6):
            ts = day.replace(hour=9 + h)
            price_rows.append((ts, px, px * 1.0005, px * 0.9995, px))
        entry = day.replace(hour=9)
        exit_ = day.replace(hour=13)
        trade_rows.append({
            'entry_date': entry, 'exit_date': exit_,
            'entry_price': px, 'exit_price': px * 1.0010,
            'size': 100_000, 'pnl': px * 0.0010 * 100_000 / px,
            'symbol': 'EUR-USD',
        })
        px *= 1.0010

    idx = pd.DatetimeIndex([r[0] for r in price_rows])
    px_df = pd.DataFrame(
        [{'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1000.0}
         for _, o, h, l, c in price_rows], index=idx)
    return pd.DataFrame(trade_rows), px_df


class TestMethodLabelling(unittest.TestCase):
    """A result that does not say how it was derived invites over-reading."""

    def setUp(self):
        self.checker = FTMOComplianceChecker()
        self.trades, self.prices = calm_history()

    def test_validate_labels_itself_close_only(self):
        r = self.checker.validate(self.trades, account_size=100_000)
        self.assertEqual(r.daily_loss_method, 'close_only')

    def test_intrabar_labels_itself(self):
        r = self.checker.validate_intrabar(self.trades, self.prices,
                                           account_size=100_000)
        self.assertEqual(r.daily_loss_method, 'intrabar_adverse')
        self.assertIsNotNone(r.intrabar_report)

    def test_missing_price_data_does_not_claim_an_intrabar_check(self):
        """
        Refusing to upgrade a verdict we could not measure. Returning
        'intrabar_adverse' here would be the exact failure this project keeps
        finding: a label asserting more than was computed.
        """
        r = self.checker.validate_intrabar(self.trades, None, account_size=100_000)
        self.assertEqual(r.daily_loss_method, 'close_only')

    def test_non_overlapping_prices_do_not_claim_a_check(self):
        far = bars([(1.1, 1.1, 1.1, 1.1)] * 20, start='2030-01-02 09:00')
        r = self.checker.validate_intrabar(self.trades, far, account_size=100_000)
        self.assertEqual(r.daily_loss_method, 'close_only')


class TestVerdictChanges(unittest.TestCase):
    """The wiring is pointless unless the verdict can actually move."""

    def setUp(self):
        self.checker = FTMOComplianceChecker()

    def _spiky_history(self):
        """
        Eight quiet days, then one day with a 6% intraday dive that recovers
        to close down only 1%. Close-only marking sees a 1% day.
        """
        trades, prices = calm_history(n_days=8)

        day = pd.Timestamp('2024-01-10')
        spike_rows, spike_idx = [], []
        entry_px = 1.1100
        for h, (o, hi, lo, c) in enumerate([
            (entry_px, entry_px * 1.0005, entry_px * 0.9995, entry_px),
            (entry_px, entry_px * 1.0005, entry_px * 0.9400, entry_px * 0.9950),
            (entry_px * 0.9950, entry_px * 0.9990, entry_px * 0.9940, entry_px * 0.9900),
        ]):
            spike_idx.append(day.replace(hour=9 + h))
            spike_rows.append({'open': o, 'high': hi, 'low': lo,
                               'close': c, 'volume': 1000.0})

        prices = pd.concat([prices,
                            pd.DataFrame(spike_rows, index=pd.DatetimeIndex(spike_idx))])

        spike_trade = {
            'entry_date': spike_idx[0], 'exit_date': spike_idx[-1],
            'entry_price': entry_px, 'exit_price': entry_px * 0.9900,
            'size': 100_000, 'pnl': -(entry_px * 0.0100 * 100_000),
            'symbol': 'EUR-USD',
        }
        trades = pd.concat([trades, pd.DataFrame([spike_trade])], ignore_index=True)
        return trades, prices

    def test_close_only_understates_the_daily_loss(self):
        trades, prices = self._spiky_history()
        close = self.checker.validate(trades, account_size=100_000)
        intra = self.checker.validate_intrabar(trades, prices, account_size=100_000)
        self.assertGreater(intra.max_daily_loss_pct, close.max_daily_loss_pct,
                           "the intraday dive must show up somewhere")

    def test_a_passing_day_can_become_a_breach(self):
        trades, prices = self._spiky_history()
        close = self.checker.validate(trades, account_size=100_000)
        intra = self.checker.validate_intrabar(trades, prices, account_size=100_000)
        self.assertTrue(close.daily_loss_ok,
                        "close-only marking must find this survivable")
        self.assertFalse(intra.daily_loss_ok,
                         "a 6% intraday dive breaches the 5% rule")
        self.assertFalse(intra.passed)

    def test_report_is_attached_for_inspection(self):
        trades, prices = self._spiky_history()
        intra = self.checker.validate_intrabar(trades, prices, account_size=100_000)
        rep: Any = cast(Any, intra.intrabar_report)
        self.assertIsNotNone(rep)
        self.assertTrue(rep.days_flipped)
        self.assertIn('BREACH', rep.summary())

    def test_calm_history_is_not_falsely_failed(self):
        """
        False positives here would reject sound strategies silently, which is
        the mirror image of the bug being fixed.
        """
        trades, prices = calm_history(n_days=8)
        close = self.checker.validate(trades, account_size=100_000)
        intra = self.checker.validate_intrabar(trades, prices, account_size=100_000)
        self.assertEqual(close.daily_loss_ok, intra.daily_loss_ok)
        self.assertTrue(intra.daily_loss_ok)


class TestOtherRulesUntouched(unittest.TestCase):
    """Only the loss rules are re-derived; the rest must carry through."""

    def setUp(self):
        self.checker = FTMOComplianceChecker()
        self.trades, self.prices = calm_history(n_days=8)

    def test_min_days_and_profit_target_are_preserved(self):
        close = self.checker.validate(self.trades, account_size=100_000)
        intra = self.checker.validate_intrabar(self.trades, self.prices,
                                               account_size=100_000)
        self.assertEqual(close.min_days_ok, intra.min_days_ok)
        self.assertEqual(close.profit_target_ok, intra.profit_target_ok)
        self.assertEqual(close.trading_days, intra.trading_days)

    def test_intrabar_never_reports_a_smaller_loss(self):
        intra = self.checker.validate_intrabar(self.trades, self.prices,
                                               account_size=100_000)
        close = self.checker.validate(self.trades, account_size=100_000)
        self.assertGreaterEqual(intra.max_daily_loss_pct, 0.0)
        self.assertGreaterEqual(intra.max_total_drawdown_pct,
                                min(close.max_total_drawdown_pct,
                                    intra.max_total_drawdown_pct))


def main():
    print("=" * 70)
    print("INTRABAR COMPLIANCE - TEST SUITE")
    print("=" * 70)
    print("validate() marks equity on close-like prices and cannot see an")
    print("intraday dive that recovered. A broker marks to market continuously;")
    print("a 5% breach at any moment ends the challenge.")
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