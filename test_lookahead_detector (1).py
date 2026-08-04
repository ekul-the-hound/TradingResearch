# ==============================================================================
# test_lookahead_detector.py
# ==============================================================================
# Phase 1, Item 6 verification.
#
#   python test_lookahead_detector.py
#
# The suite is built around deliberately cheating strategies. A detector that
# only passes clean code proves nothing -- what matters is that a strategy which
# genuinely reads the future gets caught, and that an honest one is not
# falsely accused.
# ==============================================================================

import sys
import unittest

import numpy as np
import pandas as pd

import ast as _ast

import lookahead_detector as ld
from lookahead_detector import LookaheadDetector, CRITICAL, WARNING


def _Visitor_const(value):
    """Parse a literal and run it through the visitor's constant extractor."""
    node = _ast.parse(f"x[{value}]", mode='eval').body
    return ld._Visitor._const_int(node.slice)


# ==============================================================================
# TEST DATA
# ==============================================================================

def make_prices(n=400, seed=7):
    rng = np.random.RandomState(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    openp = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    return pd.DataFrame({
        'open': openp,
        'high': np.maximum(openp, close) + spread,
        'low': np.minimum(openp, close) - spread,
        'close': close,
        'volume': np.abs(rng.normal(1000, 200, n)),
    }, index=pd.date_range('2022-01-03', periods=n, freq='D'))


# ==============================================================================
# STRATEGY SOURCES (for the static scanner)
# ==============================================================================

CLEAN_SRC = '''
import backtrader as bt

class Clean(bt.Strategy):
    params = (('fast', 10), ('slow', 30))

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.fast)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        prev = self.data.close[-1]
        if not self.position and self.crossover > 0 and self.data.close[0] > prev:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()
'''

PEEK_SRC = '''
import backtrader as bt

class Peeker(bt.Strategy):
    def next(self):
        if self.data.close[1] > self.data.close[0]:
            self.buy()
'''

PEEK_FAR_SRC = '''
import backtrader as bt

class FarPeeker(bt.Strategy):
    def next(self):
        future = self.data.high[3]
        if future > self.data.close[0] * 1.02:
            self.buy()
'''

CHEAT_COC_SRC = '''
import backtrader as bt

class Coc(bt.Strategy):
    def next(self):
        self.buy()

def run():
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    return cerebro
'''

NEG_SHIFT_SRC = '''
import backtrader as bt
import pandas as pd

class Shifty(bt.Strategy):
    def __init__(self):
        df = pd.DataFrame({'c': [1, 2, 3]})
        self.target = df['c'].shift(-1)

    def next(self):
        self.buy()
'''

ARRAY_SRC = '''
import backtrader as bt

class ArrayUser(bt.Strategy):
    def __init__(self):
        self.ceiling = max(self.data.close.array)

    def next(self):
        if self.data.close[0] < self.ceiling * 0.9:
            self.buy()
'''

AGO_SRC = '''
import backtrader as bt

class AgoPeeker(bt.Strategy):
    def next(self):
        nxt = self.data.close.get(ago=2, size=1)
        if nxt:
            self.buy()
'''


# ==============================================================================
# LAYER 1 -- STATIC SCAN
# ==============================================================================

class TestStaticScanCatchesCheaters(unittest.TestCase):

    def setUp(self):
        self.d = LookaheadDetector()

    def _rules(self, src):
        return {f.rule for f in self.d.scan_source(src).findings}

    def test_positive_index_one_bar_ahead(self):
        r = self.d.scan_source(PEEK_SRC, 'peeker')
        self.assertTrue(r.failed)
        self.assertIn('positive-line-index', {f.rule for f in r.critical})

    def test_positive_index_several_bars_ahead(self):
        r = self.d.scan_source(PEEK_FAR_SRC, 'far')
        self.assertTrue(r.failed)
        self.assertTrue(any('3 bar' in f.message for f in r.critical),
                        "message should state how far ahead it reads")

    def test_cheat_on_close(self):
        r = self.d.scan_source(CHEAT_COC_SRC, 'coc')
        self.assertTrue(r.failed)
        self.assertIn('cheat-on-close-or-open', {f.rule for f in r.critical})

    def test_negative_pandas_shift(self):
        r = self.d.scan_source(NEG_SHIFT_SRC, 'shifty')
        self.assertTrue(r.failed)
        self.assertIn('negative-shift', {f.rule for f in r.critical})

    def test_positive_ago(self):
        r = self.d.scan_source(AGO_SRC, 'ago')
        self.assertTrue(r.failed)
        self.assertIn('positive-ago', {f.rule for f in r.critical})

    def test_whole_series_statistic_warns(self):
        r = self.d.scan_source(ARRAY_SRC, 'array')
        rules = {f.rule for f in r.findings}
        self.assertTrue({'whole-series-statistic', 'raw-array-access'} & rules)


class TestStaticScanDoesNotFalselyAccuse(unittest.TestCase):
    """
    False positives are not harmless: a gate that rejects good strategies
    quietly shrinks the search space and nobody notices.
    """

    def setUp(self):
        self.d = LookaheadDetector()

    def test_clean_strategy_passes(self):
        r = self.d.scan_source(CLEAN_SRC, 'clean')
        self.assertFalse(r.failed, f"clean strategy flagged: {[f.rule for f in r.critical]}")

    def test_negative_indexing_is_fine(self):
        src = '''
import backtrader as bt
class Backward(bt.Strategy):
    def next(self):
        if self.data.close[-1] < self.data.close[0] and self.data.high[-5] > 0:
            self.buy()
'''
        self.assertFalse(self.d.scan_source(src).failed)

    def test_zero_index_is_fine(self):
        src = '''
import backtrader as bt
class Now(bt.Strategy):
    def next(self):
        if self.data.close[0] > self.data.open[0]:
            self.buy()
'''
        self.assertFalse(self.d.scan_source(src).failed)

    def test_positive_shift_is_fine(self):
        src = '''
import pandas as pd
import backtrader as bt
class LagOnly(bt.Strategy):
    def __init__(self):
        df = pd.DataFrame({'c': [1, 2, 3]})
        self.lagged = df['c'].shift(1)
    def next(self):
        self.buy()
'''
        self.assertFalse(self.d.scan_source(src).failed)

    def test_indexing_a_plain_list_is_not_flagged(self):
        """A list of parameters is not a Backtrader line."""
        src = '''
import backtrader as bt
class Params(bt.Strategy):
    def next(self):
        periods = [10, 20, 30]
        chosen = periods[1]
        if chosen > 5:
            self.buy()
'''
        r = self.d.scan_source(src)
        self.assertFalse(r.failed, f"flagged: {[f.rule for f in r.critical]}")


class TestScanReporting(unittest.TestCase):

    def setUp(self):
        self.d = LookaheadDetector()

    def test_syntax_error_fails_the_gate(self):
        r = self.d.scan_source("def broken(:\n  pass", 'bad')
        self.assertTrue(r.failed)
        self.assertIsNotNone(r.parse_error)

    def test_summary_is_actionable(self):
        s = self.d.scan_source(PEEK_SRC, 'peeker').summary()
        self.assertIn('peeker', s)
        self.assertIn('CRITICAL', s)
        self.assertIn('FAIL', s)
        self.assertIn('FUTURE', s.upper())

    def test_clean_summary_admits_it_proves_nothing(self):
        s = self.d.scan_source(CLEAN_SRC, 'clean').summary()
        self.assertIn('does not prove absence', s)


# ==============================================================================
# LAYER 2 -- EMPIRICAL PERTURBATION
# ==============================================================================

def _strategies():
    """Built lazily so the module imports without backtrader."""
    import backtrader as bt

    class CleanSMA(bt.Strategy):
        params = (('fast', 10), ('slow', 30))

        def __init__(self):
            self.f = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.s = bt.indicators.SMA(self.data.close, period=self.p.slow)
            self.x = bt.indicators.CrossOver(self.f, self.s)

        def next(self):
            if not self.position and self.x > 0:
                self.buy()
            elif self.position and self.x < 0:
                self.close()

    class Peeker(bt.Strategy):
        """Buys only when the NEXT bar closes higher. Cannot lose."""

        def next(self):
            # buflen() is the total preloaded bar count; len(self.data) is how
            # many have been processed. Stop one short so close[1] exists.
            if len(self.data) >= self.data.buflen() - 1:
                return
            nxt = self.data.close[1]
            if not self.position and nxt > self.data.close[0]:
                self.buy()
            elif self.position and nxt < self.data.close[0]:
                self.close()

    return CleanSMA, Peeker


class TestPerturbation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import backtrader  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("backtrader not installed")
        cls.CleanSMA, cls.Peeker = _strategies()
        cls.data = make_prices()

    def setUp(self):
        self.d = LookaheadDetector()

    def test_clean_strategy_is_not_flagged(self):
        r = self.d.perturbation_test(self.CleanSMA, self.data)
        self.assertIsNone(r.error, r.error)
        self.assertGreater(r.baseline_entries, 0, "strategy must actually trade")
        self.assertTrue(r.clean, f"false positive: {r.mismatches}")

    def test_peeking_strategy_is_caught(self):
        r = self.d.perturbation_test(self.Peeker, self.data)
        self.assertIsNone(r.error, r.error)
        self.assertFalse(r.clean, "a strategy reading close[1] must be detected")
        self.assertTrue(r.mismatches)

    def test_reports_which_cut_diverged(self):
        r = self.d.perturbation_test(self.Peeker, self.data)
        self.assertTrue(all('cut' in m and 'detail' in m for m in r.mismatches))

    def test_short_data_is_an_error_not_a_pass(self):
        r = self.d.perturbation_test(self.CleanSMA, self.data.head(30))
        self.assertFalse(r.clean, "insufficient data must not silently pass")
        self.assertIn('60 bars', r.error)

    def test_missing_columns_is_an_error(self):
        r = self.d.perturbation_test(self.CleanSMA, self.data[['close']])
        self.assertFalse(r.clean)
        self.assertIn('columns', r.error)

    def test_orders_timestamped_at_submission_not_notification(self):
        """
        REGRESSION PIN.

        Backtrader delivers the Submitted notification at the start of the NEXT
        bar, so reading data.datetime.datetime(0) inside notify_order gives
        submission_bar + 1. That one-bar lag pushes the last decision before a
        cut past the cut, filtering out precisely the decision the perturbation
        was built to contaminate -- and a strategy reading close[1] gets
        reported as clean. Timestamps must come from order.created.dt.
        """
        import backtrader as bt
        import io
        from contextlib import redirect_stdout

        fired = {}

        class BuyOnce(bt.Strategy):
            def next(self):
                if len(self) - 1 == 50 and not self.position:
                    fired['bar_dt'] = self.data.datetime.datetime(0)
                    self.buy()

        Rec = ld._make_entry_analyzer()
        c = bt.Cerebro(stdstats=False)
        c.adddata(bt.feeds.PandasData(dataname=self.data))
        c.addstrategy(BuyOnce)
        c.addanalyzer(Rec, _name='e')
        with redirect_stdout(io.StringIO()):
            s = c.run()[0]
        entries = s.analyzers.e.get_analysis()

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['dt'], fired['bar_dt'],
                         "recorded time must be the bar the order was created on")
        self.assertEqual(entries[0]['dt'], self.data.index[50].to_pydatetime())

    def test_multi_bar_peek_is_caught(self):
        """A strategy reading 3 bars ahead must also be detected."""
        import backtrader as bt

        class FarPeeker(bt.Strategy):
            def next(self):
                if len(self.data) >= self.data.buflen() - 3:
                    return
                ahead = self.data.close[3]
                if not self.position and ahead > self.data.close[0] * 1.005:
                    self.buy()
                elif self.position and ahead < self.data.close[0]:
                    self.close()

        r = self.d.perturbation_test(FarPeeker, self.data, name='FarPeeker')
        self.assertIsNone(r.error, r.error)
        self.assertFalse(r.clean, "a 3-bar peek must be detected")

    def test_power_is_reported_and_gates_the_verdict(self):
        """A test that exercised nothing must not read as a pass."""
        import backtrader as bt

        class NeverTrades(bt.Strategy):
            def next(self):
                pass

        r = self.d.perturbation_test(NeverTrades, self.data, name='NeverTrades')
        self.assertEqual(r.targeted_cuts, 0)
        self.assertEqual(r.power, 'none')
        self.assertIn('INCONCLUSIVE', r.summary())


class TestPerturbationHelper(unittest.TestCase):

    def test_prefix_is_untouched(self):
        df = make_prices(200)
        out = ld.perturb_future(df, 120)
        pd.testing.assert_frame_equal(df.iloc[:120], out.iloc[:120])

    def test_suffix_actually_changes(self):
        df = make_prices(200)
        out = ld.perturb_future(df, 120)
        self.assertFalse(np.allclose(df['close'].iloc[120:], out['close'].iloc[120:]),
                         "a weak perturbation would let real lookahead slip through")

    def test_generated_bars_are_valid_ohlc(self):
        """Invalid bars would cause failures unrelated to lookahead."""
        out = ld.perturb_future(make_prices(300), 150)
        tail = out.iloc[150:]
        self.assertTrue((tail['high'] >= tail[['open', 'close']].max(axis=1) - 1e-9).all())
        self.assertTrue((tail['low'] <= tail[['open', 'close']].min(axis=1) + 1e-9).all())
        self.assertTrue((tail['low'] > 0).all())

    def test_deterministic_for_a_given_seed(self):
        df = make_prices(200)
        a = ld.perturb_future(df, 100, seed=42)
        b = ld.perturb_future(df, 100, seed=42)
        pd.testing.assert_frame_equal(a, b)


class TestCombinedGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import backtrader  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("backtrader not installed")
        cls.CleanSMA, cls.Peeker = _strategies()
        cls.data = make_prices()

    def test_static_failure_short_circuits(self):
        """No backtest should be spent on code already known to peek."""
        self.assertFalse(LookaheadDetector().gate(PEEK_SRC, name='peeker'))

    def test_clean_source_only_passes(self):
        self.assertTrue(LookaheadDetector().gate(CLEAN_SRC, name='clean'))

    def test_clean_source_and_data_passes(self):
        self.assertTrue(LookaheadDetector().gate(
            CLEAN_SRC, strategy_class=self.CleanSMA, data=self.data, name='clean'))


def main():
    print("=" * 70)
    print("LOOKAHEAD DETECTOR - TEST SUITE")
    print("=" * 70)
    print("Layer 1: AST scan  -- fast, catches authored mistakes, may false-positive")
    print("Layer 2: perturbation -- ground truth, cannot false-positive, slower")
    print("=" * 70)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}")
    print("=" * 70)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
