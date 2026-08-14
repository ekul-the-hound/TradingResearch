# ==============================================================================
# test_parity_harness.py -- Tests for the backtest<->live parity checker
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# These test the pure comparison engine, which carries all the correctness.
# ==============================================================================

import unittest

from parity_harness import (
    Trade, ParityConfig, ParityChecker, ParityResult, TradeDiff,
)


def t(direction="long", size=1.0, entry=1.1000, exit=1.1050,
      entry_time=None, exit_time=None, pnl=None):
    return Trade(direction=direction, size=size, entry_price=entry,
                 exit_price=exit, entry_time=entry_time, exit_time=exit_time,
                 pnl=pnl)


class TestExactMatch(unittest.TestCase):
    def test_identical_lists_match(self):
        a = [t(), t("short", 2.0, 1.2, 1.19)]
        b = [t(), t("short", 2.0, 1.2, 1.19)]
        res = ParityChecker().compare(a, b)
        self.assertTrue(res.matched)
        self.assertEqual(res.hard_mismatches, [])
        self.assertEqual(res.soft_diffs, [])

    def test_empty_lists_match(self):
        res = ParityChecker().compare([], [])
        self.assertTrue(res.matched)
        self.assertEqual(res.n_backtest, 0)

    def test_bool_protocol(self):
        self.assertTrue(bool(ParityChecker().compare([t()], [t()])))


class TestCountMismatch(unittest.TestCase):
    def test_different_counts_hard_fail(self):
        res = ParityChecker().compare([t(), t()], [t()])
        self.assertFalse(res.matched)
        self.assertEqual(len(res.hard_mismatches), 1)
        self.assertIn("count", res.hard_mismatches[0].fields)

    def test_count_fail_does_not_truncate(self):
        # A 3-vs-1 mismatch must not silently compare only the first trade.
        res = ParityChecker().compare([t(), t(), t()], [t()])
        self.assertFalse(res.matched)
        self.assertEqual(res.n_backtest, 3)
        self.assertEqual(res.n_live, 1)


class TestHardMismatches(unittest.TestCase):
    def test_direction_mismatch_is_hard(self):
        res = ParityChecker().compare([t("long")], [t("short")])
        self.assertFalse(res.matched)
        self.assertIn("direction", res.hard_mismatches[0].fields)

    def test_size_mismatch_is_hard(self):
        res = ParityChecker().compare([t(size=1.0)], [t(size=2.0)])
        self.assertFalse(res.matched)
        self.assertIn("size", res.hard_mismatches[0].fields)

    def test_tiny_size_noise_is_not_hard(self):
        # Floating-point noise below rel tolerance must pass.
        res = ParityChecker().compare([t(size=1.0)], [t(size=1.0 + 1e-9)])
        self.assertTrue(res.matched)

    def test_multiple_hard_fields_recorded(self):
        res = ParityChecker().compare([t("long", 1.0)], [t("short", 5.0)])
        fields = res.hard_mismatches[0].fields
        self.assertIn("direction", fields)
        self.assertIn("size", fields)


class TestSoftDiffs(unittest.TestCase):
    def test_fill_within_tolerance_is_soft_but_matched(self):
        # entry off by 1e-5, tolerance 1e-4 -> within tolerance, no diff at all
        res = ParityChecker().compare([t(entry=1.1000)], [t(entry=1.10001)])
        self.assertTrue(res.matched)
        self.assertEqual(res.soft_diffs, [])  # within tolerance -> not flagged

    def test_fill_outside_tolerance_is_soft_diff(self):
        # entry off by 5e-4, tolerance 1e-4 -> flagged soft, still matched
        res = ParityChecker().compare([t(entry=1.1000)], [t(entry=1.1005)])
        self.assertTrue(res.matched)  # soft diffs don't fail parity
        self.assertEqual(len(res.soft_diffs), 1)
        self.assertIn("entry_price", res.soft_diffs[0].fields)

    def test_exit_fill_soft_diff(self):
        res = ParityChecker().compare([t(exit=1.1050)], [t(exit=1.1099)])
        self.assertTrue(res.matched)
        self.assertIn("exit_price", res.soft_diffs[0].fields)

    def test_custom_price_tolerance(self):
        cfg = ParityConfig(price_tolerance=1e-6)
        res = ParityChecker(cfg).compare([t(entry=1.1000)], [t(entry=1.10001)])
        # now 1e-5 exceeds 1e-6 -> soft diff appears
        self.assertEqual(len(res.soft_diffs), 1)


class TestTimestamps(unittest.TestCase):
    def test_close_timestamps_ok(self):
        res = ParityChecker(ParityConfig(time_tolerance_seconds=60)).compare(
            [t(entry_time="2026-01-01T10:00:00")],
            [t(entry_time="2026-01-01T10:00:30")])
        self.assertEqual(res.soft_diffs, [])

    def test_far_timestamps_soft_diff(self):
        res = ParityChecker(ParityConfig(time_tolerance_seconds=60)).compare(
            [t(entry_time="2026-01-01T10:00:00")],
            [t(entry_time="2026-01-01T10:05:00")])
        self.assertTrue(res.matched)  # still soft
        self.assertIn("entry_time", res.soft_diffs[0].fields)

    def test_missing_timestamp_skipped(self):
        # One side has no timestamp -> comparison skipped, no diff.
        res = ParityChecker().compare(
            [t(entry_time="2026-01-01T10:00:00")], [t(entry_time=None)])
        self.assertEqual(res.soft_diffs, [])

    def test_timestamp_check_disabled(self):
        res = ParityChecker(ParityConfig(time_tolerance_seconds=0)).compare(
            [t(entry_time="2026-01-01T10:00:00")],
            [t(entry_time="2026-06-01T10:00:00")])
        self.assertEqual(res.soft_diffs, [])


class TestPnlComparison(unittest.TestCase):
    def test_pnl_off_by_default(self):
        res = ParityChecker().compare([t(pnl=100)], [t(pnl=999)])
        self.assertTrue(res.matched)  # pnl not compared by default
        self.assertEqual(res.soft_diffs, [])

    def test_pnl_compared_when_enabled(self):
        cfg = ParityConfig(compare_pnl=True, price_tolerance=1e-4)
        res = ParityChecker(cfg).compare([t(size=1.0, pnl=100)],
                                         [t(size=1.0, pnl=200)])
        self.assertIn("pnl", res.soft_diffs[0].fields)


class TestRawMapping(unittest.TestCase):
    def test_from_backtest_record(self):
        rec = {"is_long": True, "size": 1.5, "entry_price": 1.10,
               "exit_price": 1.11, "entry_date": "2026-01-01T10:00:00",
               "exit_date": "2026-01-01T12:00:00", "pnl": 150}
        tr = Trade.from_backtest(rec)
        self.assertEqual(tr.direction, "long")
        self.assertEqual(tr.size, 1.5)
        self.assertEqual(tr.entry_price, 1.10)

    def test_from_backtest_short(self):
        rec = {"is_long": False, "size": 2.0, "entry_price": 1.2,
               "exit_price": 1.19}
        self.assertEqual(Trade.from_backtest(rec).direction, "short")

    def test_from_live_side_buy(self):
        rec = {"side": "buy", "size": 1.0, "entry_price": 1.1,
               "exit_price": 1.11}
        self.assertEqual(Trade.from_live(rec).direction, "long")

    def test_from_live_side_sell(self):
        rec = {"side": "sell", "size": 1.0, "entry": 1.2, "exit": 1.19}
        tr = Trade.from_live(rec)
        self.assertEqual(tr.direction, "short")
        self.assertEqual(tr.entry_price, 1.2)  # 'entry' alias

    def test_compare_raw_end_to_end(self):
        bt = [{"is_long": True, "size": 1.0, "entry_price": 1.10,
               "exit_price": 1.105}]
        lv = [{"side": "buy", "size": 1.0, "entry_price": 1.10,
               "exit_price": 1.105}]
        res = ParityChecker().compare_raw(bt, lv)
        self.assertTrue(res.matched)


class TestRealisticScenario(unittest.TestCase):
    def test_engines_agree_with_slippage(self):
        # Same 3 trades, live has realistic sub-pip slippage on fills.
        bt = [t("long", 1.0, 1.1000, 1.1050),
              t("short", 2.0, 1.2000, 1.1950),
              t("long", 1.5, 1.0800, 1.0850)]
        lv = [t("long", 1.0, 1.10002, 1.10498),
              t("short", 2.0, 1.20001, 1.19502),
              t("long", 1.5, 1.08003, 1.08497)]
        res = ParityChecker().compare(bt, lv)
        self.assertTrue(res.matched)  # structure identical, fills within pip

    def test_engine_bug_extra_trade_caught(self):
        # Live took an extra trade the backtest didn't -> hard fail.
        bt = [t("long"), t("short", 2.0, 1.2, 1.19)]
        lv = [t("long"), t("short", 2.0, 1.2, 1.19), t("long", 1.0, 1.1, 1.1)]
        res = ParityChecker().compare(bt, lv)
        self.assertFalse(res.matched)

    def test_engine_bug_wrong_direction_caught(self):
        # A sign-flip bug: live went short where backtest went long.
        bt = [t("long", 1.0, 1.10, 1.105)]
        lv = [t("short", 1.0, 1.10, 1.105)]
        res = ParityChecker().compare(bt, lv)
        self.assertFalse(res.matched)
        self.assertIn("direction", res.hard_mismatches[0].fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
