# ==============================================================================
# test_phase1_gates.py
# ==============================================================================
# Phase 1 items 7 and 8.
#
#   python test_phase1_gates.py
#
# Both suites are built around deliberately non-compliant strategies. A
# detector that only clears good input proves nothing.
#
# Import failures are HARD ERRORS. An earlier suite in this project used
# try/except -> skipTest and printed "ALL TESTS PASSED" while silently skipping
# ten tests on a wrong class name. A skip is not a pass.
# ==============================================================================

import sys
import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

import prohibited_patterns as pp
import property_crash_tests as pct


# ==============================================================================
# TRADE FIXTURES
# ==============================================================================

def trades(specs, symbol='EUR-USD', start='2024-01-02', hold_hours=2):
    """specs: list of (size, pnl) in order."""
    base = pd.Timestamp(start)
    rows, price = [], 1.1000
    for i, (size, pnl) in enumerate(specs):
        entry = base + timedelta(days=i, hours=9)
        rows.append({
            'entry_date': entry,
            'exit_date': entry + timedelta(hours=hold_hours),
            'entry_price': price,
            'exit_price': price + (pnl / abs(size) if size else 0),
            'size': size, 'pnl': pnl, 'symbol': symbol,
        })
        price += 0.0005
    return pd.DataFrame(rows)


def clean_trades(n=20):
    rng = np.random.RandomState(0)
    return trades([(100_000, float(rng.normal(60, 300))) for _ in range(n)])


# ==============================================================================
# ITEM 8 -- BEHAVIOURAL
# ==============================================================================

class TestMartingaleDetection(unittest.TestCase):

    def test_classic_martingale_is_caught(self):
        """Double after every loss -- the canonical banned pattern."""
        specs, size = [], 10_000
        for i in range(10):
            pnl = -200.0 if i % 2 == 0 else 400.0
            specs.append((size, pnl))
            size = size * 2 if pnl < 0 else 10_000
        r = pp.scan_trades(trades(specs), name='martingale')
        self.assertTrue(r.failed)
        self.assertIn('martingale', r.patterns)

    def test_constant_size_is_clean(self):
        r = pp.scan_trades(clean_trades(), name='clean')
        self.assertNotIn('martingale', {f.pattern for f in r.critical})

    def test_pyramiding_is_not_flagged_as_martingale(self):
        """
        Anti-martingale (size up after WINS) is usually permitted. Conflating
        the two would reject a legitimate strategy class outright.
        """
        specs, size = [], 10_000
        for i in range(10):
            pnl = 400.0 if i % 2 == 0 else -150.0
            specs.append((size, pnl))
            size = size * 2 if pnl > 0 else 10_000
        r = pp.scan_trades(trades(specs), name='pyramid')
        self.assertNotIn('martingale', {f.pattern for f in r.critical},
                         "sizing up after WINS is not martingale")

    def test_isolated_size_increase_is_not_enough(self):
        specs = [(10_000, -100.0), (20_000, 300.0)] + [(10_000, 50.0)] * 10
        r = pp.scan_trades(trades(specs), name='oneoff')
        self.assertNotIn('martingale', {f.pattern for f in r.critical})


class TestHedgingDetection(unittest.TestCase):

    def test_simultaneous_long_and_short_caught(self):
        base = pd.Timestamp('2024-01-02 09:00')
        df = pd.DataFrame([
            {'entry_date': base, 'exit_date': base + timedelta(hours=6),
             'entry_price': 1.10, 'exit_price': 1.101, 'size': 100_000,
             'pnl': 100.0, 'symbol': 'EUR-USD'},
            {'entry_date': base + timedelta(hours=1),
             'exit_date': base + timedelta(hours=4),
             'entry_price': 1.1005, 'exit_price': 1.10, 'size': -100_000,
             'pnl': 50.0, 'symbol': 'EUR-USD'},
        ])
        r = pp.scan_trades(df, name='hedged')
        self.assertTrue(r.failed)
        self.assertIn('hedging', r.patterns)

    def test_sequential_long_then_short_is_fine(self):
        specs = [(100_000, 100.0), (-100_000, 80.0), (100_000, -50.0)]
        r = pp.scan_trades(trades(specs), name='sequential')
        self.assertNotIn('hedging', {f.pattern for f in r.critical},
                         "non-overlapping direction changes are normal trading")

    def test_opposite_directions_different_symbols_is_fine(self):
        base = pd.Timestamp('2024-01-02 09:00')
        df = pd.DataFrame([
            {'entry_date': base, 'exit_date': base + timedelta(hours=6),
             'entry_price': 1.10, 'exit_price': 1.101, 'size': 100_000,
             'pnl': 100.0, 'symbol': 'EUR-USD'},
            {'entry_date': base, 'exit_date': base + timedelta(hours=6),
             'entry_price': 1.27, 'exit_price': 1.269, 'size': -100_000,
             'pnl': 100.0, 'symbol': 'GBP-USD'},
        ])
        r = pp.scan_trades(df, name='two-symbols')
        self.assertNotIn('hedging', {f.pattern for f in r.critical})


class TestSubThresholdHolds(unittest.TestCase):

    def test_tick_scalping_caught(self):
        base = pd.Timestamp('2024-01-02 09:00')
        rows = []
        for i in range(20):
            e = base + timedelta(minutes=i * 10)
            rows.append({'entry_date': e, 'exit_date': e + timedelta(seconds=8),
                         'entry_price': 1.10, 'exit_price': 1.1001,
                         'size': 100_000, 'pnl': 10.0, 'symbol': 'EUR-USD'})
        r = pp.scan_trades(pd.DataFrame(rows), name='scalper')
        self.assertTrue(r.failed)
        self.assertIn('sub_threshold', r.patterns)

    def test_normal_holds_are_clean(self):
        r = pp.scan_trades(clean_trades(), name='normal')
        self.assertNotIn('sub_threshold', {f.pattern for f in r.critical})

    def test_threshold_is_configurable_per_firm(self):
        base = pd.Timestamp('2024-01-02 09:00')
        rows = [{'entry_date': base + timedelta(hours=i),
                 'exit_date': base + timedelta(hours=i, seconds=90),
                 'entry_price': 1.10, 'exit_price': 1.1001,
                 'size': 100_000, 'pnl': 10.0, 'symbol': 'EUR-USD'}
                for i in range(20)]
        df = pd.DataFrame(rows)
        lax = pp.scan_trades(df, thresholds={'min_hold_seconds': 60})
        strict = pp.scan_trades(df, thresholds={'min_hold_seconds': 300})
        self.assertNotIn('sub_threshold', {f.pattern for f in lax.critical})
        self.assertIn('sub_threshold', {f.pattern for f in strict.critical})


class TestGridDetection(unittest.TestCase):

    def test_averaging_down_caught(self):
        base = pd.Timestamp('2024-01-02 09:00')
        rows, price = [], 1.1000
        for i in range(6):
            rows.append({'entry_date': base + timedelta(minutes=i * 5),
                         'exit_date': base + timedelta(hours=10),
                         'entry_price': price, 'exit_price': 1.0950,
                         'size': 100_000, 'pnl': -200.0, 'symbol': 'EUR-USD'})
            price -= 0.0020          # adding as it falls
        r = pp.scan_trades(pd.DataFrame(rows), name='grid')
        self.assertTrue(r.failed)
        self.assertIn('grid', r.patterns)

    def test_sequential_non_overlapping_is_not_grid(self):
        r = pp.scan_trades(clean_trades(), name='clean')
        self.assertNotIn('grid', {f.pattern for f in r.critical})


class TestStaticScan(unittest.TestCase):

    def test_martingale_identifier_caught(self):
        src = '''
import backtrader as bt
class M(bt.Strategy):
    def next(self):
        self.martingale = self.martingale * 2
        self.buy()
'''
        r = pp.scan_source(src, 'm')
        self.assertTrue(r.failed)
        self.assertIn('martingale', r.patterns)

    def test_size_doubling_caught(self):
        src = '''
import backtrader as bt
class D(bt.Strategy):
    def next(self):
        if self.last_loss:
            self.stake_size *= 2
        self.buy(size=self.stake_size)
'''
        r = pp.scan_source(src, 'd')
        self.assertTrue(r.failed)
        self.assertIn('martingale', r.patterns)

    def test_clean_strategy_passes(self):
        src = '''
import backtrader as bt
class C(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy(size=self.p.fixed_size)
        else:
            self.close()
'''
        self.assertFalse(pp.scan_source(src, 'c').failed)

    def test_syntax_error_fails_the_gate(self):
        r = pp.scan_source("def broken(:", 'bad')
        self.assertTrue(r.failed)

    def test_static_alone_does_not_clear_a_strategy(self):
        """Behaviour is authoritative; syntax can hide anything."""
        src = '''
import backtrader as bt
class Sneaky(bt.Strategy):
    def next(self):
        self.buy(size=self.compute_next_allocation())
'''
        self.assertFalse(pp.scan_source(src, 'sneaky').failed)
        specs, size = [], 10_000
        for i in range(10):
            pnl = -200.0 if i % 2 == 0 else 400.0
            specs.append((size, pnl))
            size = size * 2 if pnl < 0 else 10_000
        self.assertTrue(pp.scan_trades(trades(specs)).failed,
                        "behavioural layer must catch what static missed")


class TestReporting(unittest.TestCase):

    def test_empty_trades_is_an_error_not_a_pass(self):
        r = pp.scan_trades(pd.DataFrame(), name='none')
        self.assertTrue(r.failed)
        self.assertIsNotNone(r.error)

    def test_summary_names_the_pattern(self):
        specs, size = [], 10_000
        for i in range(8):
            pnl = -200.0 if i % 2 == 0 else 400.0
            specs.append((size, pnl))
            size = size * 2 if pnl < 0 else 10_000
        s = pp.scan_trades(trades(specs), name='mg').summary()
        self.assertIn('martingale', s)
        self.assertIn('FAIL', s)


# ==============================================================================
# ITEM 7 -- PROPERTY CRASH TESTS
# ==============================================================================

def _strategies():
    import backtrader as bt

    class Robust(bt.Strategy):
        params = (('period', 10),)

        def __init__(self):
            self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)

        def next(self):
            if len(self) < self.p.period:
                return
            if not self.position and self.data.close[0] > self.sma[0]:
                self.buy(size=1)
            elif self.position and self.data.close[0] < self.sma[0]:
                self.close()

    class DividesByVolatility(bt.Strategy):
        """Crashes on a flat series, where high - low is zero."""

        def next(self):
            if len(self) < 2:
                return
            rng = self.data.high[0] - self.data.low[0]
            size = int(1000 / rng)          # ZeroDivisionError on a flat bar
            if not self.position and size > 0:
                self.buy(size=size)

    class BadIndexing(bt.Strategy):
        """
        Crashes on viable-length data by indexing a fixed list with the bar
        count.

        NOTE: an earlier version used self.data.close[-50] expecting an
        IndexError. It does not raise -- Backtrader preloads and Python wraps
        the negative index to the END of the array, so it silently returns
        FUTURE data instead. That is a lookahead bug, not a crash, and it is
        now caught by lookahead_detector's unguarded-deep-lookback rule.
        """

        def __init__(self):
            self.levels = [1.0, 2.0, 3.0]

        def next(self):
            _ = self.levels[len(self)]      # IndexError from bar 4 onward

    return Robust, DividesByVolatility, BadIndexing


class TestPathologicalCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import backtrader  # noqa: F401  hard error if missing
        cls.Robust, cls.DivVol, cls.BadIndex = _strategies()

    def test_generated_frames_are_valid_ohlc(self):
        """Invalid bars would produce crashes that mean nothing."""
        for name, df in pct.pathological_cases().items():
            with self.subTest(case=name):
                self.assertTrue((df['high'] >= df[['open', 'close']].max(axis=1) - 1e-9).all())
                self.assertTrue((df['low'] <= df[['open', 'close']].min(axis=1) + 1e-9).all())
                self.assertTrue((df['low'] > 0).all())

    def test_robust_strategy_survives_everything(self):
        r = pct.run_pathological(self.Robust)
        self.assertTrue(r.clean, f"false positive: {r.failures}")
        # passed + insufficient_data accounts for every case: sub-viable frames
        # are neither passes nor defects, so they are counted in their own bucket.
        self.assertEqual(r.passed + len(r.insufficient_data), r.total)

    def test_zero_division_strategy_is_caught(self):
        r = pct.run_pathological(self.DivVol)
        self.assertFalse(r.clean, "a flat series must expose the division")
        self.assertTrue(any('ZeroDivision' in f['error'] for f in r.failures))

    def test_crashing_strategy_is_caught_on_viable_data(self):
        r = pct.run_pathological(self.BadIndex)
        self.assertFalse(r.clean, "an IndexError on viable-length data is a real defect")
        self.assertTrue(any(f['bars'] >= pct.MIN_VIABLE_BARS for f in r.failures))

    def test_tiny_data_is_not_counted_as_a_strategy_defect(self):
        """
        Backtrader raises for ANY strategy whose indicator period exceeds the
        series length -- verified with a bare SMA(10) on 1 and 3 bars. Counting
        that as a defect would reject essentially every real strategy, which is
        the false-positive failure mode this gate must avoid.
        """
        r = pct.run_pathological(self.Robust)
        self.assertTrue(r.clean, f"false positive: {r.failures}")
        cases = {f['case'] for f in r.insufficient_data}
        self.assertTrue({'single_bar', 'minimal'} & cases,
                        "short series must be surfaced, just not as failures")

    def test_named_cases_cover_known_hazards(self):
        cases = set(pct.pathological_cases())
        for expected in ('flat_line', 'zero_volume', 'huge_gap',
                         'single_bar', 'minimal', 'extreme_vol'):
            self.assertIn(expected, cases)

    def test_gate_returns_bool(self):
        self.assertTrue(pct.gate(self.Robust))
        self.assertFalse(pct.gate(self.DivVol))

    def test_summary_reports_the_failing_case(self):
        s = pct.run_pathological(self.DivVol).summary()
        self.assertIn('FAIL', s)
        self.assertIn('flat_line', s)


class TestFuzzing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not pct.HAS_HYPOTHESIS:
            raise unittest.SkipTest("hypothesis not installed")
        import backtrader  # noqa: F401
        cls.Robust, cls.DivVol, _ = _strategies()

    def test_fuzz_clears_a_robust_strategy(self):
        r = pct.run_fuzz(self.Robust, max_examples=15)
        self.assertTrue(r.clean, f"false positive under fuzzing: {r.failures}")

    def test_fuzz_finds_the_division_bug(self):
        r = pct.run_fuzz(self.DivVol, max_examples=40)
        self.assertFalse(r.clean, "fuzzing should reach a flat or near-flat bar")


def main():
    print("=" * 70)
    print("PHASE 1 GATES - TEST SUITE")
    print("=" * 70)
    print("Item 8: prohibited patterns -- martingale, grid, hedging, tick scalping.")
    print("        mutation_config actively tells the LLM to generate these.")
    print("Item 7: property crash tests -- pathological OHLC the sample data lacks.")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    n_skip = len(getattr(result, 'skipped', []))
    print("\n" + "=" * 70)
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
