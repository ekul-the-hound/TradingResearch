# ==============================================================================
# test_mfe_mae_analyzer.py -- Tests for the MFE/MAE placement analyzer
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import unittest
from dataclasses import dataclass

from mfe_mae_analyzer import MFEMAEAnalyzer, MFEMAEReport, ExcursionRow, _percentile


def row(realised, mae, mfe, size=1.0, symbol="EURUSD"):
    return {"realised_pnl": realised, "mae": mae, "mfe": mfe,
            "size": size, "symbol": symbol}


def make_trades(n_winners, n_losers, winner_dip=50, winner_gain=100):
    rows = []
    for _ in range(n_winners):
        rows.append(row(winner_gain, -winner_dip, winner_gain + 50))
    for _ in range(n_losers):
        rows.append(row(-80, -100, 20))
    return rows


class TestPercentile(unittest.TestCase):
    def test_median_odd(self):
        self.assertEqual(_percentile([1, 2, 3], 50), 2)

    def test_median_even(self):
        self.assertEqual(_percentile([1, 2, 3, 4], 50), 2.5)

    def test_p75(self):
        self.assertAlmostEqual(_percentile([1, 2, 3, 4, 5], 75), 4.0)  # type: ignore[arg-type]

    def test_single_value(self):
        self.assertEqual(_percentile([7], 90), 7)

    def test_empty(self):
        self.assertIsNone(_percentile([], 50))


class TestInsufficientSample(unittest.TestCase):
    def test_too_few_trades_withholds(self):
        rep = MFEMAEAnalyzer(min_trades=20).analyze(make_trades(3, 3))
        self.assertFalse(rep.sufficient)
        self.assertIsNone(rep.winner_mae_p75)
        self.assertTrue(any("noise" in n for n in rep.notes))

    def test_enough_trades_but_few_winners_withholds_winner_stats(self):
        # 25 trades total but only 5 winners -> winner guidance withheld,
        # overall still "sufficient" for other stats.
        rep = MFEMAEAnalyzer(min_trades=20, min_winners=10).analyze(
            make_trades(5, 20))
        self.assertTrue(rep.sufficient)
        self.assertIsNone(rep.winner_mae_p75)
        self.assertTrue(any("winners" in n for n in rep.notes))

    def test_empty_input(self):
        rep = MFEMAEAnalyzer().analyze([])
        self.assertFalse(rep.sufficient)
        self.assertEqual(rep.n_trades, 0)


class TestWinnerMAE(unittest.TestCase):
    def test_winner_dip_distribution(self):
        # 30 winners all dipping exactly 50 before working out.
        rep = MFEMAEAnalyzer().analyze(make_trades(30, 5, winner_dip=50))
        self.assertEqual(rep.winner_mae_median, 50)
        self.assertEqual(rep.winner_mae_p75, 50)

    def test_winner_count(self):
        rep = MFEMAEAnalyzer().analyze(make_trades(25, 10))
        self.assertEqual(rep.n_winners, 25)
        self.assertEqual(rep.n_losers, 10)

    def test_varied_dips_percentiles_ordered(self):
        rows = []
        for dip in range(10, 41):  # 31 winners, dips 10..40
            rows.append(row(100, -dip, 200))
        rep = MFEMAEAnalyzer().analyze(rows)
        self.assertLessEqual(rep.winner_mae_median, rep.winner_mae_p75)  # type: ignore[arg-type]
        self.assertLessEqual(rep.winner_mae_p75, rep.winner_mae_p90)  # type: ignore[arg-type]


class TestLeftOnTable(unittest.TestCase):
    def test_run_left_computed(self):
        # winner realises 100, MFE 250 -> 150 left on table each.
        rows = [row(100, -20, 250) for _ in range(25)]
        rep = MFEMAEAnalyzer().analyze(rows)
        self.assertAlmostEqual(rep.left_on_table_median, 150, places=1)  # type: ignore[arg-type]
        self.assertAlmostEqual(rep.left_on_table_total, 150 * 25, places=1)  # type: ignore[arg-type]

    def test_no_run_when_mfe_equals_realised(self):
        rows = [row(100, -20, 100) for _ in range(25)]
        rep = MFEMAEAnalyzer().analyze(rows)
        self.assertAlmostEqual(rep.left_on_table_median, 0, places=6)  # type: ignore[arg-type]


class TestHiddenAdverse(unittest.TestCase):
    def test_winner_that_went_underwater(self):
        # Winner realised +50 but MAE was -120: hidden adverse = 120 - 0 = 120
        # (realised is positive so the max(0,-realised) term is 0).
        rows = make_trades(20, 5)
        rows.append(row(50, -120, 200))
        rep = MFEMAEAnalyzer().analyze(rows)
        self.assertGreater(rep.total_hidden_adverse, 0)


class TestInputTypes(unittest.TestCase):
    def test_accepts_objects(self):
        @dataclass
        class Exc:
            realised_pnl: float
            mae: float
            mfe: float
            size: float = 1.0
            symbol: str = "EURUSD"

        objs = [Exc(100, -50, 200) for _ in range(25)]
        rep = MFEMAEAnalyzer().analyze(objs)
        self.assertTrue(rep.sufficient)
        self.assertEqual(rep.n_winners, 25)

    def test_realized_spelling_alias(self):
        # American 'realized_pnl' spelling should also map.
        rows = [{"realized_pnl": 100, "mae": -50, "mfe": 200} for _ in range(25)]
        rep = MFEMAEAnalyzer().analyze(rows)
        self.assertEqual(rep.n_winners, 25)

    def test_pnl_fallback(self):
        rows = [{"pnl": 100, "mae": -50, "mfe": 200} for _ in range(25)]
        rep = MFEMAEAnalyzer().analyze(rows)
        self.assertEqual(rep.n_winners, 25)


class TestConvenienceWrapper(unittest.TestCase):
    def test_missing_dependency_returns_insufficient(self):
        # analyze_from_trades must not raise even if intrabar_risk/data absent.
        rep = MFEMAEAnalyzer().analyze_from_trades(trades=None, price_data=None)
        self.assertIsInstance(rep, MFEMAEReport)
        self.assertFalse(rep.sufficient)


class TestSummaryRenders(unittest.TestCase):
    def test_summary_sufficient(self):
        rep = MFEMAEAnalyzer().analyze(make_trades(25, 10))
        s = rep.summary()
        self.assertIn("placement analysis", s)
        self.assertIn("out-of-sample", s)  # the caveat must be present

    def test_summary_insufficient(self):
        rep = MFEMAEAnalyzer().analyze(make_trades(2, 2))
        self.assertIn("insufficient", rep.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)