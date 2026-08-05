# ==============================================================================
# test_intrabar_risk.py
# ==============================================================================
# Phase 2, Item 14.
#
#   python test_intrabar_risk.py
#
# The central case is a day that passes the 5% daily-loss rule on close prices
# and breaches it intrabar. That is not a modelling preference -- it is a real
# breach that the current compliance check cannot see.
#
# Import failures are HARD ERRORS. A skip is not a pass.
# ==============================================================================

import sys
import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

import intrabar_risk as ir


def bars(specs, start='2024-01-02 09:00', freq='h'):
    """specs: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(specs), freq=freq)
    return pd.DataFrame(
        [{'open': o, 'high': h, 'low': l, 'close': c, 'volume': 1000.0}
         for o, h, l, c in specs], index=idx)


def flat_bars(n=48, price=1.1000, start='2024-01-02 09:00'):
    return bars([(price, price, price, price)] * n, start=start)


class TestExcursions(unittest.TestCase):

    def test_long_mae_uses_the_window_low(self):
        px = bars([(1.1000, 1.1010, 1.0900, 1.1005),
                   (1.1005, 1.1020, 1.0950, 1.1015)])
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1000, 'exit_price': 1.1015,
              'size': 100_000, 'pnl': 150.0, 'symbol': 'EUR-USD'}]
        e = ir.trade_excursions(t, px)[0]
        # Low of 1.0900 is 100 pips against a 100k long = -$1,000
        self.assertAlmostEqual(e.mae, -1000.0, places=2)
        self.assertLess(e.mae, 0)

    def test_short_mae_uses_the_window_high(self):
        px = bars([(1.1000, 1.1150, 1.0990, 1.1010)])
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1000, 'exit_price': 1.1010,
              'size': -100_000, 'pnl': -100.0, 'symbol': 'EUR-USD'}]
        e = ir.trade_excursions(t, px)[0]
        self.assertAlmostEqual(e.mae, -1500.0, places=2)

    def test_winner_that_first_went_underwater_is_exposed(self):
        """
        The trade a close-only curve reports as riskless. Booked +$150, but it
        held -$1,000 at one point and consumed daily-loss budget doing so.
        """
        px = bars([(1.1000, 1.1010, 1.0900, 1.1015)])
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1000, 'exit_price': 1.1015,
              'size': 100_000, 'pnl': 150.0, 'symbol': 'EUR-USD'}]
        e = ir.trade_excursions(t, px)[0]
        self.assertGreater(e.realised_pnl, 0)
        self.assertAlmostEqual(e.hidden_loss, 1000.0, places=2)

    def test_trade_that_never_went_against_you_hides_nothing(self):
        px = bars([(1.1000, 1.1050, 1.1000, 1.1040)])
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1000, 'exit_price': 1.1040,
              'size': 100_000, 'pnl': 400.0, 'symbol': 'EUR-USD'}]
        e = ir.trade_excursions(t, px)[0]
        self.assertAlmostEqual(e.mae, 0.0, places=6)
        self.assertAlmostEqual(e.hidden_loss, 0.0, places=6)

    def test_unmatched_trades_are_skipped_not_zeroed(self):
        """Assigning zero excursion to an unmeasured trade is the bug, not the fix."""
        px = flat_bars(4, start='2024-06-01 09:00')
        t = [{'entry_date': pd.Timestamp('2024-01-02 09:00'),
              'exit_date': pd.Timestamp('2024-01-02 12:00'),
              'entry_price': 1.10, 'exit_price': 1.11, 'size': 100_000,
              'pnl': 100.0, 'symbol': 'EUR-USD'}]
        self.assertEqual(ir.trade_excursions(t, px), [])

    def test_missing_high_low_columns_skips(self):
        px = pd.DataFrame({'close': [1.1, 1.2]},
                          index=pd.date_range('2024-01-02 09:00', periods=2, freq='h'))
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1, 'exit_price': 1.2, 'size': 100_000,
              'pnl': 100.0, 'symbol': 'EUR-USD'}]
        self.assertEqual(ir.trade_excursions(t, px), [])


class TestEquityPath(unittest.TestCase):

    def setUp(self):
        self.px = bars([(1.1000, 1.1010, 1.0900, 1.1005),
                        (1.1005, 1.1020, 1.0990, 1.1015),
                        (1.1015, 1.1025, 1.1000, 1.1020)])
        self.t = [{'entry_date': self.px.index[0], 'exit_date': self.px.index[-1],
                   'entry_price': 1.1000, 'exit_price': 1.1020,
                   'size': 100_000, 'pnl': 200.0, 'symbol': 'EUR-USD'}]

    def test_adverse_path_dips_below_close_path(self):
        c = ir.equity_path(self.t, self.px, 100_000, ir.MODE_CLOSE)
        a = ir.equity_path(self.t, self.px, 100_000, ir.MODE_ADVERSE)
        self.assertLess(a['equity'].min(), c['equity'].min(),
                        "the low of the first bar must show up somewhere")

    def test_adverse_low_matches_the_bar_low(self):
        a = ir.equity_path(self.t, self.px, 100_000, ir.MODE_ADVERSE)
        # 1.0900 vs 1.1000 entry on 100k long = -$1,000
        self.assertAlmostEqual(a['equity'].min(), 99_000.0, delta=1.0)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            ir.equity_path(self.t, self.px, 100_000, mode='optimistic')

    def test_empty_trades_returns_empty(self):
        self.assertTrue(ir.equity_path([], self.px, 100_000).empty)


class TestDailyLossFlip(unittest.TestCase):
    """The headline: a day that passes on close and breaches intrabar."""

    def test_day_flips_from_pass_to_breach(self):
        # A long that dives 6% intraday and closes down only 1%.
        px = bars([
            (1.1000, 1.1010, 1.1000, 1.1000),
            (1.1000, 1.1005, 1.0340, 1.0950),   # -6.0% at the low
            (1.0950, 1.0990, 1.0940, 1.0890),   # closes -1.0% overall
        ])
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1000, 'exit_price': 1.0890,
              'size': 100_000, 'pnl': -1100.0, 'symbol': 'EUR-USD'}]

        rep = ir.analyze(t, px, account_size=100_000, max_daily_loss_pct=5.0)
        self.assertIsNone(rep.error, rep.error)
        self.assertLess(rep.close_only_max_daily_loss_pct, 5.0,
                        "close prices must look survivable")
        self.assertGreater(rep.adverse_max_daily_loss_pct, 5.0,
                           "the intraday low must breach")
        self.assertTrue(rep.verdict_changes)
        self.assertIn('BREACH', rep.summary())

    def test_quiet_history_does_not_flip(self):
        px = bars([(1.1000, 1.1005, 1.0995, 1.1000)] * 6)
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1000, 'exit_price': 1.1000,
              'size': 100_000, 'pnl': 0.0, 'symbol': 'EUR-USD'}]
        rep = ir.analyze(t, px, account_size=100_000)
        self.assertFalse(rep.verdict_changes, f"false positive: {rep.days_flipped}")

    def test_adverse_is_never_rosier_than_close(self):
        rng = np.random.RandomState(3)
        close = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.002, 60)))
        spec = [(c, c * 1.002, c * 0.998, c) for c in close]
        px = bars(spec)
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': float(close[0]), 'exit_price': float(close[-1]),
              'size': 100_000, 'pnl': float((close[-1] - close[0]) * 100_000),
              'symbol': 'EUR-USD'}]
        rep = ir.analyze(t, px, account_size=100_000)
        self.assertGreaterEqual(rep.adverse_max_daily_loss_pct,
                                rep.close_only_max_daily_loss_pct)
        self.assertGreaterEqual(rep.adverse_max_drawdown_pct,
                                rep.close_only_max_drawdown_pct)


class TestReporting(unittest.TestCase):

    def test_no_price_data_is_an_error_not_a_pass(self):
        t = [{'entry_date': pd.Timestamp('2024-01-02 09:00'),
              'exit_date': pd.Timestamp('2024-01-02 12:00'),
              'entry_price': 1.1, 'exit_price': 1.11, 'size': 100_000,
              'pnl': 100.0, 'symbol': 'EUR-USD'}]
        rep = ir.analyze(t, None, account_size=100_000)
        self.assertIsNotNone(rep.error)
        assert rep.error is not None      # narrow Optional for the checker
        self.assertIn('price data', rep.error.lower())

    def test_no_trades_is_an_error(self):
        rep = ir.analyze([], flat_bars(), account_size=100_000)
        self.assertIsNotNone(rep.error)

    def test_non_overlapping_prices_is_an_error(self):
        px = flat_bars(10, start='2024-06-01 09:00')
        t = [{'entry_date': pd.Timestamp('2024-01-02 09:00'),
              'exit_date': pd.Timestamp('2024-01-02 12:00'),
              'entry_price': 1.1, 'exit_price': 1.11, 'size': 100_000,
              'pnl': 100.0, 'symbol': 'EUR-USD'}]
        rep = ir.analyze(t, px, account_size=100_000)
        self.assertIsNotNone(rep.error)

    def test_summary_shows_both_columns(self):
        px = bars([(1.1000, 1.1010, 1.0900, 1.1005)] * 4)
        t = [{'entry_date': px.index[0], 'exit_date': px.index[-1],
              'entry_price': 1.1000, 'exit_price': 1.1005,
              'size': 100_000, 'pnl': 50.0, 'symbol': 'EUR-USD'}]
        s = ir.analyze(t, px, account_size=100_000).summary()
        self.assertIn('close-only', s)
        self.assertIn('adverse', s)

    def test_multi_symbol_prices_are_accepted(self):
        a = bars([(1.1000, 1.1010, 1.0950, 1.1005)] * 4)
        b = bars([(1.2700, 1.2710, 1.2650, 1.2705)] * 4)
        t = [
            {'entry_date': a.index[0], 'exit_date': a.index[-1],
             'entry_price': 1.1000, 'exit_price': 1.1005, 'size': 100_000,
             'pnl': 50.0, 'symbol': 'EUR-USD'},
            {'entry_date': b.index[0], 'exit_date': b.index[-1],
             'entry_price': 1.2700, 'exit_price': 1.2705, 'size': 100_000,
             'pnl': 50.0, 'symbol': 'GBP-USD'},
        ]
        rep = ir.analyze(t, {'EUR-USD': a, 'GBP-USD': b}, account_size=100_000)
        self.assertIsNone(rep.error, rep.error)
        self.assertEqual(rep.trades_analysed, 2)


def main():
    print("=" * 70)
    print("INTRABAR RISK - TEST SUITE")
    print("=" * 70)
    print("ftmo_compliance marks equity on close-like prices only and falls back")
    print("to a stale last_price at checkpoints, so a position that goes deeply")
    print("underwater and recovers registers no drawdown. A broker marks to")
    print("market continuously; recovering before the candle closed earns")
    print("no credit.")
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
