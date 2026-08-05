# ==============================================================================
# test_data_fingerprint.py
# ==============================================================================
# Phase 2, Item 18.
#
#   python test_data_fingerprint.py
#
# The property that matters: the fingerprint must CHANGE when the data changes
# in a way that would change a result, and stay the same when it would not.
# A hash that never differs is decoration.
#
# Import failures are HARD ERRORS. A skip is not a pass.
# ==============================================================================

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

import data_fingerprint as dfp


def fp_of(df, symbol='', timeframe=''):
    """
    fingerprint_frame returns Optional by design -- it must never raise, so it
    returns None on bad input. In tests the input is always valid, so assert
    that up front. This is better than indexing through an Optional: a None
    here reports as a clear failure instead of an AttributeError twenty lines
    later, and it silences the reportOptionalMemberAccess noise honestly
    rather than with a blanket ignore.
    """
    fp = dfp.fingerprint_frame(df, symbol, timeframe)
    assert fp is not None, "fingerprint_frame returned None for a valid frame"
    return fp


def reg_of(symbol, timeframe):
    """Same idea for the registry lookup, which is Optional on a miss."""
    fp = dfp.lookup(symbol, timeframe)
    assert fp is not None, f"no fingerprint registered for {symbol} {timeframe}"
    return fp


def frame(n=500, start='2020-01-01', seed=0, base=100.0):
    rng = np.random.RandomState(seed)
    close = base * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range(start, periods=n, freq='h')
    return pd.DataFrame({'open': close, 'high': close * 1.001,
                         'low': close * 0.999, 'close': close,
                         'volume': 1000.0}, index=idx)


class TestFingerprintSensitivity(unittest.TestCase):
    """Must change when it matters, hold when it does not."""

    def setUp(self):
        dfp.clear()
        self.df = frame()

    def test_identical_frames_hash_the_same(self):
        a = fp_of(frame(), 'EUR-USD', '1hour')
        b = fp_of(frame(), 'EUR-USD', '1hour')
        self.assertEqual(a.hash, b.hash)

    def test_different_values_change_the_hash(self):
        a = fp_of(frame(seed=0))
        b = fp_of(frame(seed=1))
        self.assertNotEqual(a.hash, b.hash)

    def test_truncation_changes_the_hash(self):
        a = fp_of(self.df)
        b = fp_of(self.df.head(400))
        self.assertNotEqual(a.hash, b.hash,
                            "holdout truncation must be visible in provenance")

    def test_a_time_shift_changes_the_hash(self):
        """
        THE CASE THAT MOTIVATED THIS. The timezone fix moved every timestamp
        +5h without altering a single close value. A hash over prices alone
        would call the old and new data identical.
        """
        shifted = self.df.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=5)
        a = fp_of(self.df)
        b = fp_of(shifted)
        self.assertNotEqual(a.hash, b.hash,
                            "a pure re-dating must change the fingerprint")

    def test_column_change_changes_the_hash(self):
        a = fp_of(self.df)
        b = fp_of(self.df.drop(columns=['volume']))
        self.assertNotEqual(a.hash, b.hash)

    def test_empty_and_none_return_none(self):
        # Passing None is the point of the test, so the type complaint is
        # expected and scoped rather than suppressed file-wide.
        self.assertIsNone(dfp.fingerprint_frame(pd.DataFrame()))
        self.assertIsNone(dfp.fingerprint_frame(None))  # pyright: ignore[reportArgumentType]

    def test_fingerprinting_never_raises(self):
        """A missing fingerprint is recoverable; a crashed backtest is not."""
        # Deliberately wrong types -- that is what is being tested.
        bad_inputs = (None, pd.DataFrame(), 'not a frame', 42, [])
        for bad in bad_inputs:
            try:
                dfp.fingerprint_frame(bad)  # pyright: ignore[reportArgumentType]
            except Exception as e:
                self.fail(f"raised on {bad!r}: {e}")

    def test_metadata_is_recorded(self):
        fp = fp_of(self.df, 'EUR-USD', '1hour')
        self.assertEqual(fp.rows, len(self.df))
        self.assertEqual(fp.symbol, 'EUR-USD')
        self.assertEqual(fp.timeframe, '1hour')
        self.assertIn('close', fp.columns)


class TestRegistry(unittest.TestCase):

    def setUp(self):
        dfp.clear()

    def test_record_then_lookup(self):
        dfp.record('EUR-USD', '1hour', frame())
        self.assertEqual(reg_of('EUR-USD', '1hour')['symbol'], 'EUR-USD')

    def test_lookup_miss_returns_none(self):
        self.assertIsNone(dfp.lookup('GBP-USD', '4hour'))

    def test_symbols_do_not_collide(self):
        dfp.record('EUR-USD', '1hour', frame(seed=0))
        dfp.record('GBP-USD', '1hour', frame(seed=1))
        self.assertNotEqual(reg_of('EUR-USD', '1hour')['hash'],
                            reg_of('GBP-USD', '1hour')['hash'])

    def test_timeframes_do_not_collide(self):
        dfp.record('EUR-USD', '1hour', frame(n=500))
        dfp.record('EUR-USD', '4hour', frame(n=125))
        self.assertNotEqual(reg_of('EUR-USD', '1hour')['hash'],
                            reg_of('EUR-USD', '4hour')['hash'])

    def test_rerecording_overwrites(self):
        dfp.record('EUR-USD', '1hour', frame(seed=0))
        first = reg_of('EUR-USD', '1hour')['hash']
        dfp.record('EUR-USD', '1hour', frame(seed=9))
        self.assertNotEqual(reg_of('EUR-USD', '1hour')['hash'], first)

    def test_clear_empties(self):
        dfp.record('EUR-USD', '1hour', frame())
        dfp.clear()
        self.assertEqual(dfp.registry_size(), 0)


class TestComparison(unittest.TestCase):

    def test_same_data_is_comparable(self):
        a = fp_of(frame()).to_dict()
        b = fp_of(frame()).to_dict()
        self.assertTrue(dfp.compare(a, b)['comparable'])

    def test_different_data_is_not_comparable(self):
        a = fp_of(frame(seed=0)).to_dict()
        b = fp_of(frame(seed=1)).to_dict()
        self.assertFalse(dfp.compare(a, b)['comparable'])

    def test_missing_fingerprint_is_not_comparable(self):
        a = fp_of(frame()).to_dict()
        self.assertFalse(dfp.compare(a, None)['comparable'])
        self.assertFalse(dfp.compare(None, None)['comparable'])

    def test_reason_names_what_differs(self):
        a = fp_of(frame(n=500)).to_dict()
        b = fp_of(frame(n=400)).to_dict()
        self.assertIn('rows', dfp.compare(a, b)['reason'])


class TestTimezoneHeuristic(unittest.TestCase):
    """
    Positively identifies stale rows rather than merely failing to vouch
    for them -- which is what makes the audit worth running.
    """

    def test_pre_fix_forex_timestamp_detected(self):
        self.assertTrue(dfp.looks_pre_timezone_fix('2000-05-30 17:27:00', 'EUR-USD'))

    def test_post_fix_forex_timestamp_detected(self):
        self.assertFalse(dfp.looks_pre_timezone_fix('2000-05-30 22:27:00', 'EUR-USD'))

    def test_ambiguous_hour_returns_none(self):
        """An unknown must not read as a clean bill of health."""
        self.assertIsNone(dfp.looks_pre_timezone_fix('2020-01-02 09:00:00', 'EUR-USD'))

    def test_non_forex_returns_none(self):
        self.assertIsNone(dfp.looks_pre_timezone_fix('2020-01-02 17:00:00', 'BTC-USD'))

    def test_garbage_returns_none(self):
        for bad in ('not a date', None, ''):
            self.assertIsNone(dfp.looks_pre_timezone_fix(bad, 'EUR-USD'))


class TestCodeFingerprint(unittest.TestCase):

    def test_returns_library_versions(self):
        fp = dfp.code_fingerprint()
        self.assertIn('pandas', fp)
        self.assertIn('numpy', fp)

    def test_string_form_is_stable(self):
        self.assertEqual(dfp.code_fingerprint_str(), dfp.code_fingerprint_str())

    def test_string_form_is_not_empty(self):
        self.assertTrue(dfp.code_fingerprint_str())


class TestDatabaseIntegration(unittest.TestCase):
    """End to end: a saved result must carry its provenance."""

    def setUp(self):
        from database import ResultsDatabase
        dfp.clear()
        fd, self.path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        os.unlink(self.path)
        # Pass a Path, not a str. apply_typing_patch.py widens the annotation
        # in database.py to Union[str, os.PathLike], but this test should not
        # depend on whether that patch has been applied -- and mkstemp handing
        # back a str is an implementation detail, not a contract.
        self.db = ResultsDatabase(Path(self.path))

    def tearDown(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def _result(self):
        return {
            'strategy_name': 'S', 'variant_id': 'v1', 'symbol': 'EUR-USD',
            'timeframe': '1hour', 'start_date': '2020-01-01',
            'end_date': '2020-02-01', 'bars_tested': 500,
            'starting_value': 100000, 'ending_value': 105000,
            'total_return_pct': 5.0, 'sharpe_ratio': 1.1,
            'max_drawdown_pct': 2.0, 'total_trades': 20,
            'win_rate': 55.0, 'profit_factor': 1.4, 'trades': [],
        }

    def test_provenance_columns_exist(self):
        conn = sqlite3.connect(self.path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(backtest_results)")}
        conn.close()
        for c in ('data_fingerprint', 'data_rows', 'data_first',
                  'data_last', 'code_fingerprint'):
            self.assertIn(c, cols)

    def test_fingerprint_is_recorded_when_registered(self):
        dfp.record('EUR-USD', '1hour', frame())
        bid = self.db.save_backtest(self._result())
        conn = sqlite3.connect(self.path)
        row = conn.execute(
            "SELECT data_fingerprint, data_rows FROM backtest_results WHERE id=?",
            (bid,)).fetchone()
        conn.close()
        self.assertIsNotNone(row[0])
        self.assertEqual(row[1], 500)

    def test_missing_registration_leaves_fingerprint_null(self):
        """NULL is the signal that provenance is unknown. It must not be faked."""
        bid = self.db.save_backtest(self._result())
        conn = sqlite3.connect(self.path)
        row = conn.execute(
            "SELECT data_fingerprint, code_fingerprint FROM backtest_results WHERE id=?",
            (bid,)).fetchone()
        conn.close()
        self.assertIsNone(row[0], "an unknown fingerprint must stay NULL")
        self.assertIsNotNone(row[1], "code fingerprint is always knowable")


def main():
    print("=" * 70)
    print("DATA FINGERPRINT - TEST SUITE")
    print("=" * 70)
    print("The timezone fix moved every timestamp +5h without changing a single")
    print("close value. A hash over prices alone would call old and new data")
    print("identical -- so the index is part of the hash.")
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