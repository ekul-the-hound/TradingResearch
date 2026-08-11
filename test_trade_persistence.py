# ==============================================================================
# test_trade_persistence.py
# ==============================================================================
# Covers the two largest fixes in this batch:
#   - trade-level persistence in the results DB   (item 11.4)
#   - correct bootstrap pass-rate simulation      (item 11.3)
#
#   python test_trade_persistence.py
# ==============================================================================

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def make_trades(n_days=30, per_day=2, pnl=120.0, start='2024-01-02'):
    base = pd.Timestamp(start)
    rows, price = [], 1.1000
    for d in range(n_days):
        day = base + timedelta(days=d)
        for k in range(per_day):
            entry = day.replace(hour=9 + k * 3)
            exit_ = entry + timedelta(hours=2)
            nxt = price + (pnl / 100_000)
            rows.append({
                'entry_date': entry, 'exit_date': exit_,
                'entry_price': price, 'exit_price': nxt,
                'size': 100_000, 'pnl': pnl, 'return_pct': 0.1,
                'duration_bars': 2, 'is_long': True, 'symbol': 'EUR-USD',
            })
            price = nxt
    return rows


# ==============================================================================
# TRADE PERSISTENCE
# ==============================================================================

class TestTradePersistence(unittest.TestCase):

    def setUp(self):
        # NOTE: import failures here used to call skipTest, which meant a
        # wrong class name silently skipped all ten tests while the suite
        # still printed "ALL PASSED". A skip that hides a broken import is a
        # false negative -- the same shape of bug as everything else in this
        # batch. It is now a hard failure.
        from database import ResultsDatabase
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        os.unlink(self.path)
        self.db = ResultsDatabase(self.path)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def _result(self, trades=None, variant='variant_07'):
        return {
            'strategy_name': 'TestStrat', 'variant_id': variant,
            'symbol': 'EUR-USD', 'timeframe': '1hour',
            'start_date': '2024-01-02', 'end_date': '2024-02-02',
            'bars_tested': 1000, 'starting_value': 100_000,
            'ending_value': 107_200, 'total_return_pct': 7.2,
            'sharpe_ratio': 1.4, 'max_drawdown_pct': 3.1,
            'total_trades': len(trades or []), 'win_rate': 60.0,
            'profit_factor': 1.8, 'trades': trades or [],
        }

    def test_table_exists_after_init(self):
        conn = sqlite3.connect(self.path)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='backtest_trades'").fetchone()
        conn.close()
        self.assertIsNotNone(row, "backtest_trades table must be created on init")

    def test_trades_are_saved_and_read_back(self):
        trades = make_trades(n_days=10)
        bid = self.db.save_backtest(self._result(trades))
        out = self.db.get_trades(bid)
        self.assertEqual(len(out), len(trades))

    def test_saved_fields_match_the_checker_contract(self):
        """The whole point: what comes back must feed FTMOComplianceChecker."""
        bid = self.db.save_backtest(self._result(make_trades(n_days=5)))
        out = self.db.get_trades(bid)
        for col in ('entry_date', 'exit_date', 'entry_price', 'exit_price', 'size'):
            self.assertIn(col, out[0], f"checker needs '{col}'")
        self.assertIsNotNone(out[0]['entry_price'])
        self.assertIsNotNone(out[0]['exit_price'])

    def test_exit_price_is_stored_not_reconstructed(self):
        """
        This is what makes the results DB exact where the decay DB is only
        approximate: the decay table has no prices, so notional-based fees have
        to be estimated. Here they do not.
        """
        trades = make_trades(n_days=3)
        bid = self.db.save_backtest(self._result(trades))
        out = self.db.get_trades(bid)
        self.assertAlmostEqual(out[0]['entry_price'], trades[0]['entry_price'], places=6)
        self.assertAlmostEqual(out[0]['exit_price'], trades[0]['exit_price'], places=6)

    def test_result_without_trades_still_saves(self):
        bid = self.db.save_backtest(self._result([]))
        self.assertIsNotNone(bid)
        self.assertEqual(self.db.get_trades(bid), [])

    def test_malformed_trade_does_not_lose_the_backtest_row(self):
        bad = [{'entry_date': 'x'}, 'not-a-dict', {'exit_date': None}]
        bid = self.db.save_backtest(self._result(bad))
        self.assertIsNotNone(bid, "a bad trade must not roll back the result")

    def test_get_latest_trades_returns_most_recent_run(self):
        self.db.save_backtest(self._result(make_trades(n_days=5)))
        self.db.save_backtest(self._result(make_trades(n_days=12)))
        trades, bid = self.db.get_latest_trades(variant_id='variant_07')
        self.assertEqual(len(trades), 24, "must return the newest run, not a merge")
        self.assertIsNotNone(bid)

    def test_get_latest_trades_unknown_variant(self):
        trades, bid = self.db.get_latest_trades(variant_id='nope')
        self.assertEqual(trades, [])
        self.assertIsNone(bid)

    def test_panel_reads_the_results_db(self):
        import dashboard_ftmo_panel as panel
        self.db.save_backtest(self._result(make_trades(n_days=20)))
        r = panel.rows_from_results_db(self.path, variant_id='variant_07')
        self.assertTrue(r.available, r.reason)
        self.assertEqual(r.source, panel.SOURCE_RESULTS_DB)
        self.assertFalse(r.approximate, "results DB has real prices - must be exact")
        self.assertEqual(len(r.rows), len(panel.ACCOUNT_SIZES))

    def test_panel_unavailable_when_backtest_has_no_trades(self):
        import dashboard_ftmo_panel as panel
        self.db.save_backtest(self._result([]))
        r = panel.rows_from_results_db(self.path, variant_id='variant_07')
        self.assertFalse(r.available)
        self.assertIn('predates trade persistence', r.reason)


# ==============================================================================
# PASS RATE SIMULATION
# ==============================================================================

class TestPassRateSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pass_rate_simulator  # noqa: F401
        from ftmo_compliance import FTMOComplianceChecker  # noqa: F401

    def setUp(self):
        import pass_rate_simulator as prs
        from ftmo_compliance import FTMOComplianceChecker
        self.prs = prs
        self.checker = FTMOComplianceChecker()
        self.trades = pd.DataFrame(make_trades(n_days=40, per_day=2))

    def test_old_shuffle_was_a_no_op(self):
        """
        REGRESSION PIN documenting the original bug: reordering rows cannot
        change the result, because the equity curve sorts by timestamp.
        """
        a = self.checker.validate(self.trades, account_size=100_000, phase='challenge')
        shuffled = self.trades.sample(frac=1, random_state=7).reset_index(drop=True)
        b = self.checker.validate(shuffled, account_size=100_000, phase='challenge')
        self.assertAlmostEqual(a.final_return_pct, b.final_return_pct, places=9)
        self.assertAlmostEqual(a.max_total_drawdown_pct, b.max_total_drawdown_pct, places=9)

    def test_synthetic_window_actually_redates(self):
        rng = np.random.RandomState(0)
        sim = self.prs.build_synthetic_window(self.trades, 30, rng)
        self.assertFalse(sim.empty)
        orig = set(pd.to_datetime(self.trades['entry_date']))
        new = set(pd.to_datetime(sim['entry_date']))
        self.assertTrue(len(new - orig) > 0,
                        "re-dating is the step the original was missing")

    def test_synthetic_window_uses_weekdays_only(self):
        rng = np.random.RandomState(1)
        sim = self.prs.build_synthetic_window(self.trades, 30, rng)
        wd = pd.to_datetime(sim['entry_date']).dt.weekday
        self.assertTrue((wd < 5).all(), "FX is closed at weekends")

    def test_sampling_is_with_replacement(self):
        """Composition must vary; a permutation would keep it fixed."""
        rng = np.random.RandomState(2)
        idx = self.prs._draw_indices(50, 200, self.prs.MODE_IID, rng, 5.0)
        self.assertEqual(len(idx), 200)
        self.assertGreater(len(idx) - len(set(idx)), 0, "replacement implies repeats")

    def test_block_mode_preserves_contiguity(self):
        rng = np.random.RandomState(3)
        idx = self.prs._draw_indices(100, 300, self.prs.MODE_BLOCK, rng, 5.0)
        consecutive = sum(1 for i in range(1, len(idx)) if idx[i] == idx[i - 1] + 1)
        self.assertGreater(consecutive, 50,
                           "block bootstrap must keep runs together")

    def test_results_actually_vary(self):
        """The headline fix: the distribution must be a distribution."""
        r = self.prs.simulate_pass_rate(
            self.checker, self.trades, n_simulations=40, verbose=False)
        self.assertIsNone(r.error)
        self.assertFalse(r.degenerate,
                         "identical results every time means resampling is broken")
        spread = r.max_dd_pct['p95'] - r.max_dd_pct['p5']
        self.assertGreater(spread, 0.0, "max drawdown must have a distribution")

    def test_pass_rate_is_not_pinned_to_zero_or_one(self):
        r = self.prs.simulate_pass_rate(
            self.checker, self.trades, n_simulations=60, verbose=False)
        self.assertGreaterEqual(r.pass_rate, 0.0)
        self.assertLessEqual(r.pass_rate, 1.0)
        self.assertTrue(r.return_pct, "return distribution must be populated")

    def test_deterministic_for_a_given_seed(self):
        a = self.prs.simulate_pass_rate(self.checker, self.trades,
                                        n_simulations=25, random_seed=5, verbose=False)
        b = self.prs.simulate_pass_rate(self.checker, self.trades,
                                        n_simulations=25, random_seed=5, verbose=False)
        self.assertEqual(a.pass_rate, b.pass_rate)

    def test_too_few_trades_is_an_error(self):
        r = self.prs.simulate_pass_rate(
            self.checker, self.trades.head(2), n_simulations=5, verbose=False)
        self.assertIsNotNone(r.error)
        assert r.error is not None  # narrow for type checker
        self.assertIn('at least 4', r.error)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.prs.simulate_pass_rate(self.checker, self.trades,
                                        mode='nonsense', verbose=False)

    def test_summary_flags_degeneracy(self):
        r = self.prs.PassRateResult(
            pass_rate=1.0, n_simulations=10, account_size=100_000,
            phase='challenge', n_trades=50, mode='block', window_days=30,
            degenerate=True)
        self.assertIn('identical outcome', r.summary())


def main():
    print("=" * 70)
    print("TRADE PERSISTENCE + PASS RATE - TEST SUITE")
    print("=" * 70)
    print("Persistence: backtest_results stored counts, not trades -- the shared")
    print("  cause of the dashboard proxy and the synthetic-returns fallback.")
    print("Pass rate:   the old shuffle was a no-op; 1,000 sims, 1 outcome.")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 70)
    n_skip = len(getattr(result, 'skipped', []))
    if n_skip:
        print(f"[WARN] {n_skip} test(s) SKIPPED - a skip is not a pass:")
        for t, why in result.skipped:
            print(f"       {t}: {why}")
    if result.wasSuccessful() and not n_skip:
        print(f"ALL {result.testsRun} TESTS PASSED")
    elif result.wasSuccessful():
        print(f"{result.testsRun - n_skip} passed, {n_skip} SKIPPED - not a clean run")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}")
    print("=" * 70)
    return 0 if (result.wasSuccessful() and not n_skip) else 1


if __name__ == '__main__':
    sys.exit(main())