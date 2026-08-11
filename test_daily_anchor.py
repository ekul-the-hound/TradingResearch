# ==============================================================================
# test_daily_anchor.py
# ==============================================================================
# Proves the Phase 0 Item 2 fix. Runs entirely on synthetic equity curves --
# no market data, no E:\TradingData, no Docker. Safe to run anywhere.
#
#   python test_daily_anchor.py
#
# Each test states the FTMO rule it encodes and the hand-computed expectation,
# so a failure tells you which rule broke rather than just which assert fired.
# ==============================================================================

import sys
import unittest
from datetime import datetime, timedelta

import pandas as pd
from typing import cast, Any
import pytz

import ftmo_daily_anchor as anchor

PRAGUE = pytz.timezone('Europe/Prague')
UTC = pytz.UTC
INITIAL = 100_000.0


def utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm)


def curve(rows):
    """rows: list of (timestamp, balance, equity)"""
    return pd.DataFrame(
        [{'timestamp': t, 'balance': b, 'equity': e} for t, b, e in rows]
    )


class TestPragueMidnight(unittest.TestCase):
    """The checkpoint must land on midnight CE(S)T, not 00:00 UTC."""

    def test_winter_midnight_is_2300_utc(self):
        m = anchor.prague_midnight_utc(datetime(2024, 1, 15).date())
        self.assertEqual(m, utc(2024, 1, 14, 23, 0),
                         "Prague midnight in CET is 23:00 UTC the previous day")

    def test_summer_midnight_is_2200_utc(self):
        m = anchor.prague_midnight_utc(datetime(2024, 7, 15).date())
        self.assertEqual(m, utc(2024, 7, 14, 22, 0),
                         "Prague midnight in CEST is 22:00 UTC the previous day")

    def test_old_utc_normalize_was_off_by_1_to_2_hours(self):
        """Documents the size of the bug being fixed."""
        for date, expected_gap_h in ((datetime(2024, 1, 15), 1), (datetime(2024, 7, 15), 2)):
            correct = anchor.prague_midnight_utc(date.date())
            _ts = cast(Any, pd.Timestamp(date)).normalize()
            old = _ts.to_pydatetime()  # 00:00 UTC
            gap = (old - correct).total_seconds() / 3600
            self.assertEqual(gap, expected_gap_h,
                             f"old checkpoint sat {gap}h inside the Prague day")

    def test_checkpoints_span_dst_transition(self):
        """Crossing the March DST switch, the UTC offset must change."""
        cps = anchor.prague_midnight_checkpoints(utc(2024, 3, 28), utc(2024, 4, 3))
        offsets = {(c.hour) for c in cps}
        self.assertIn(23, offsets, "CET midnights are 23:00 UTC")
        self.assertIn(22, offsets, "CEST midnights are 22:00 UTC")

    def test_no_duplicate_or_missing_days(self):
        cps = anchor.prague_midnight_checkpoints(utc(2024, 1, 1), utc(2024, 1, 11))
        self.assertEqual(len(cps), len(set(cps)), "no duplicate checkpoints")
        self.assertEqual(len(cps), 10, "one checkpoint per intervening Prague day")


class TestBalanceAnchor(unittest.TestCase):
    """FTMO anchors the limit on BALANCE at midnight, and measures EQUITY."""

    def test_day_one_anchors_on_initial_capital(self):
        """
        FTMO: 'On the first day, the account balance used for the calculation
        is the Initial Simulated Capital.' Entry fees must not lower it.
        """
        c = curve([
            (utc(2024, 1, 15, 9), 99_950, 99_950),   # after $50 entry fee
            (utc(2024, 1, 15, 12), 99_950, 96_000),  # floating loss
        ])
        out = anchor.calculate_daily_stats_anchored(c, INITIAL)
        self.assertEqual(out.iloc[0]['anchor_balance'], INITIAL)
        self.assertEqual(out.iloc[0]['anchor_source'], 'initial_capital')
        # Loss vs initial capital = 100,000 - 96,000 = 4,000 = 4.00%
        self.assertAlmostEqual(out.iloc[0]['daily_loss_pct'], 4.00, places=2)

    def test_floating_loss_across_midnight_no_longer_hides_a_breach(self):
        """
        THE CORE BUG.

        Balance at Prague midnight = $100,000 (nothing closed yet).
        A position carried overnight is $3,000 underwater at the reset,
        then loses $2,500 more during the day -> equity low $94,500.

        FTMO limit  = 100,000 - 5,000 = 95,000. Equity hit 94,500 -> BREACH.
        Old code    = anchored on equity at first event (97,000),
                      so measured only 97,000 - 94,500 = 2,500 = 2.5% -> PASS.
        """
        c = curve([
            (utc(2024, 1, 14, 20, 0), 100_000, 98_000),   # day 1, position open
            (utc(2024, 1, 14, 23, 0), 100_000, 97_000),   # PRAGUE MIDNIGHT of Jan 15
            (utc(2024, 1, 15, 10, 0), 100_000, 96_000),
            (utc(2024, 1, 15, 14, 0), 100_000, 94_500),   # low of the day
            (utc(2024, 1, 15, 16, 0), 94_500, 94_500),    # closed out
        ])
        out = anchor.calculate_daily_stats_anchored(c, INITIAL)
        jan15 = out[out['date'] == datetime(2024, 1, 15).date()].iloc[0]

        self.assertEqual(jan15['anchor_balance'], 100_000,
                         "anchor is BALANCE at midnight, not equity")
        self.assertEqual(jan15['daily_loss_limit'], 95_000)
        self.assertAlmostEqual(jan15['daily_loss_pct'], 5.50, places=2)
        self.assertTrue(jan15['breached'], "5.5% > 5% must breach")

        # And confirm the old anchor would have missed it.
        cmp = anchor.compare_anchors(c, INITIAL)
        old = cmp[cmp['date'] == datetime(2024, 1, 15).date()].iloc[0]
        self.assertLess(old['old_daily_loss_pct'], 5.0,
                        "old anchor understated this day - the bug")

    def test_floating_profit_across_midnight_does_not_inflate_the_limit(self):
        """
        FTMO: 'Intraday changes resulting from open positions do not affect
        the Maximum Daily Loss Limit.' Floating profit at midnight is NOT
        added to the anchor under FTMO's own documentation.
        """
        # Jan 14 row makes Jan 15 a genuine later day, so the midnight-balance
        # branch is exercised rather than the day-one initial-capital branch.
        c = curve([
            (utc(2024, 1, 14, 10, 0), 100_000, 100_000),  # Prague Jan 14
            (utc(2024, 1, 14, 23, 0), 100_000, 103_000),  # +3k floating at reset
            (utc(2024, 1, 15, 12, 0), 100_000, 95_500),
        ])
        out = anchor.calculate_daily_stats_anchored(c, INITIAL)
        jan15 = out[out['date'] == datetime(2024, 1, 15).date()].iloc[0]
        self.assertEqual(jan15['anchor_balance'], 100_000)
        self.assertEqual(jan15['anchor_source'], 'balance@prague_midnight')
        self.assertAlmostEqual(jan15['daily_loss_pct'], 4.50, places=2)
        self.assertFalse(jan15['breached'])

    def test_max_mode_models_the_third_party_reading(self):
        """The alternative 'balance or equity, whichever is higher' reading."""
        c = curve([
            (utc(2024, 1, 14, 10, 0), 100_000, 100_000),
            (utc(2024, 1, 14, 23, 0), 100_000, 103_000),
            (utc(2024, 1, 15, 12, 0), 100_000, 95_500),
        ])
        out = anchor.calculate_daily_stats_anchored(
            c, INITIAL, anchor_mode=anchor.ANCHOR_MAX)
        jan15 = out[out['date'] == datetime(2024, 1, 15).date()].iloc[0]
        self.assertEqual(jan15['anchor_balance'], 103_000)
        self.assertAlmostEqual(jan15['daily_loss_pct'], 7.50, places=2)

    def test_anchor_follows_a_growing_balance(self):
        """Limit recalculates daily: profitable day raises the next day's floor."""
        c = curve([
            (utc(2024, 1, 14, 23, 0), 100_000, 100_000),
            (utc(2024, 1, 15, 12, 0), 104_000, 104_000),
            (utc(2024, 1, 15, 23, 0), 104_000, 104_000),  # Prague midnight of Jan 16
            (utc(2024, 1, 16, 12, 0), 104_000, 99_500),
        ])
        out = anchor.calculate_daily_stats_anchored(c, INITIAL)
        jan16 = out[out['date'] == datetime(2024, 1, 16).date()].iloc[0]
        self.assertEqual(jan16['anchor_balance'], 104_000)
        self.assertEqual(jan16['daily_loss_limit'], 99_000,
                         "104,000 - 5,000; equity of 99,500 survives")
        self.assertFalse(jan16['breached'])

    def test_exactly_at_limit_is_not_a_breach(self):
        """FTMO: violated only if equity drops BELOW the limit."""
        c = curve([
            (utc(2024, 1, 14, 23, 0), 100_000, 100_000),
            (utc(2024, 1, 15, 12, 0), 100_000, 95_000),  # exactly the limit
        ])
        out = anchor.calculate_daily_stats_anchored(c, INITIAL)
        jan15 = out[out['date'] == datetime(2024, 1, 15).date()].iloc[0]
        self.assertFalse(jan15['breached'])


class TestContracts(unittest.TestCase):
    """The replacement must not break callers of _calculate_daily_stats."""

    REQUIRED = ['date', 'start_equity', 'end_equity', 'min_equity', 'max_equity',
                'daily_pnl', 'daily_loss_from_start', 'daily_loss_pct']

    def test_original_columns_preserved(self):
        c = curve([
            (utc(2024, 1, 15, 9), 100_000, 100_000),
            (utc(2024, 1, 15, 15), 99_000, 99_000),
        ])
        out = anchor.calculate_daily_stats_anchored(c, INITIAL)
        for col in self.REQUIRED:
            self.assertIn(col, out.columns, f"caller contract needs '{col}'")

    def test_empty_curve_returns_empty_frame(self):
        out = anchor.calculate_daily_stats_anchored(pd.DataFrame(), INITIAL)
        self.assertTrue(out.empty)

    def test_missing_balance_column_fails_loudly(self):
        """Silently falling back to the equity anchor is what we are fixing."""
        bad = pd.DataFrame([{'timestamp': utc(2024, 1, 15), 'equity': 100_000}])
        with self.assertRaises(KeyError):
            anchor.calculate_daily_stats_anchored(bad, INITIAL)

    def test_tz_aware_input_is_normalised(self):
        c = pd.DataFrame([
            {'timestamp': pd.Timestamp('2024-01-15 09:00', tz='UTC'),
             'balance': 100_000, 'equity': 100_000},
            {'timestamp': pd.Timestamp('2024-01-15 15:00', tz='UTC'),
             'balance': 97_000, 'equity': 97_000},
        ])
        out = anchor.calculate_daily_stats_anchored(c, INITIAL)
        self.assertEqual(len(out), 1)


def main():
    print("=" * 70)
    print("FTMO DAILY-LOSS ANCHOR - TEST SUITE")
    print("=" * 70)
    print("Encodes FTMO's published rule:")
    print("  limit = balance at midnight CE(S)T - 5% of initial capital")
    print("  breach when EQUITY (balance + floating) drops below that limit")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED - anchor matches FTMO's rule")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}")
    print("=" * 70)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())