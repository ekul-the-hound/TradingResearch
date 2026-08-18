# ==============================================================================
# test_slippage_recorder.py -- Tests for the slippage/spread recorder
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import os
import tempfile
import unittest
from dataclasses import dataclass

from slippage_recorder import SlippageRecorder, FillObservation, SymbolStats


class RecorderTestBase(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.rec = SlippageRecorder(db_path=self.db, min_fills=30)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            p = self.db + suffix
            if os.path.exists(p):
                os.remove(p)


class TestSignedSlippage(unittest.TestCase):
    def test_buy_filled_above_is_adverse_positive(self):
        obs = FillObservation("EURUSD", "buy", 1.10000, 1.10010, 0.00002)
        # +0.0091% adverse
        self.assertGreater(obs.slippage_pct, 0)

    def test_buy_filled_below_is_favourable_negative(self):
        obs = FillObservation("EURUSD", "buy", 1.10000, 1.09990, 0.00002)
        self.assertLess(obs.slippage_pct, 0)

    def test_sell_filled_below_is_adverse_positive(self):
        # A sell filled BELOW signal is adverse -> positive slippage.
        obs = FillObservation("EURUSD", "sell", 1.10000, 1.09990, 0.00002)
        self.assertGreater(obs.slippage_pct, 0)

    def test_sell_filled_above_is_favourable_negative(self):
        obs = FillObservation("EURUSD", "sell", 1.10000, 1.10010, 0.00002)
        self.assertLess(obs.slippage_pct, 0)

    def test_zero_signal_price_safe(self):
        obs = FillObservation("EURUSD", "buy", 0.0, 1.10, 0.0)
        self.assertEqual(obs.slippage_pct, 0.0)

    def test_spread_pct_computed(self):
        obs = FillObservation("EURUSD", "buy", 1.10000, 1.10000, 0.00011)
        self.assertAlmostEqual(obs.spread_pct, 0.01, places=3)


class TestRecording(RecorderTestBase):
    def test_record_persists(self):
        self.rec.record("EURUSD", "buy", 1.10000, 1.10010, 0.00002)
        stats = self.rec.stats_for("EURUSD")
        self.assertEqual(stats.n_fills, 1)

    def test_multiple_symbols(self):
        self.rec.record("EURUSD", "buy", 1.10, 1.1001)
        self.rec.record("GBPUSD", "sell", 1.25, 1.2499)
        self.assertEqual(set(self.rec.all_symbols()), {"EURUSD", "GBPUSD"})

    def test_record_observation_object(self):
        obs = FillObservation("EURUSD", "buy", 1.10, 1.1001, 0.00002)
        self.rec.record_observation(obs)
        self.assertEqual(self.rec.stats_for("EURUSD").n_fills, 1)


class TestAggregation(RecorderTestBase):
    def _fill_n(self, n, symbol="EURUSD", side="buy", slip=0.00006):
        for _ in range(n):
            sig = 1.10000
            fill = sig + slip if side == "buy" else sig - slip
            self.rec.record(symbol, side, sig, fill, quoted_spread=0.00002)

    def test_sufficient_threshold(self):
        self._fill_n(29)
        self.assertFalse(self.rec.stats_for("EURUSD").sufficient)
        self._fill_n(1)  # now 30
        self.assertTrue(self.rec.stats_for("EURUSD").sufficient)

    def test_mean_adverse_slippage_positive(self):
        self._fill_n(40, side="buy", slip=0.00006)  # all adverse buys
        stats = self.rec.stats_for("EURUSD")
        self.assertGreater(stats.mean_adverse_slippage_pct, 0)

    def test_favourable_fills_excluded_from_adverse(self):
        # All favourable buys (fill below signal) -> no adverse slippage.
        for _ in range(40):
            self.rec.record("EURUSD", "buy", 1.10000, 1.09994, 0.00002)
        stats = self.rec.stats_for("EURUSD")
        self.assertEqual(stats.mean_adverse_slippage_pct, 0.0)  # none adverse

    def test_spread_aggregated(self):
        self._fill_n(35)
        stats = self.rec.stats_for("EURUSD")
        self.assertGreater(stats.mean_spread_pct, 0)

    def test_no_fills_stats(self):
        stats = self.rec.stats_for("NOPE")
        self.assertEqual(stats.n_fills, 0)
        self.assertFalse(stats.sufficient)


class TestObservedProfile(RecorderTestBase):
    def _fill_n(self, n, slip=0.00006):
        for _ in range(n):
            self.rec.record("EURUSD", "buy", 1.10000, 1.10000 + slip,
                            quoted_spread=0.00002)

    def test_sufficient_emits_values(self):
        self._fill_n(40)
        prof = self.rec.observed_profile("EURUSD")
        self.assertTrue(prof["sufficient"])
        self.assertIsNotNone(prof["slippage_pct"])
        self.assertGreater(prof["slippage_pct"], 0)

    def test_insufficient_returns_none_and_fallback(self):
        self._fill_n(5)

        @dataclass
        class Profile:
            slippage_pct: float = 0.005
            spread_pct: float = 0.01

        prof = self.rec.observed_profile("EURUSD", base_profile=Profile())
        self.assertFalse(prof["sufficient"])
        self.assertIsNone(prof["slippage_pct"])  # not enough to override
        self.assertEqual(prof["fallback_slippage_pct"], 0.005)  # keep assumption

    def test_never_overwrites_with_noise(self):
        # The whole point: <min_fills must not produce a confident number.
        self._fill_n(3)
        prof = self.rec.observed_profile("EURUSD")
        self.assertIsNone(prof["slippage_pct"])
        self.assertIsNone(prof["spread_pct"])


class TestClear(RecorderTestBase):
    def test_clear_symbol(self):
        self.rec.record("EURUSD", "buy", 1.10, 1.1001)
        self.rec.record("GBPUSD", "buy", 1.25, 1.2501)
        self.rec.clear("EURUSD")
        self.assertEqual(self.rec.stats_for("EURUSD").n_fills, 0)
        self.assertEqual(self.rec.stats_for("GBPUSD").n_fills, 1)

    def test_clear_all(self):
        self.rec.record("EURUSD", "buy", 1.10, 1.1001)
        self.rec.clear()
        self.assertEqual(self.rec.all_symbols(), [])


class TestPersistence(RecorderTestBase):
    def test_survives_reopen(self):
        for _ in range(35):
            self.rec.record("EURUSD", "buy", 1.10000, 1.10006, 0.00002)
        rec2 = SlippageRecorder(db_path=self.db, min_fills=30)
        self.assertTrue(rec2.stats_for("EURUSD").sufficient)


if __name__ == "__main__":
    unittest.main(verbosity=2)
