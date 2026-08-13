# ==============================================================================
# test_position_sizing.py
# ==============================================================================
# Proves Phase 0 Item 4. Pure arithmetic -- no market data, no broker.
#
#   python test_position_sizing.py
# ==============================================================================

import sys
import unittest

import position_sizing as ps

EQUITY = 100_000.0


class TestOldBugCannotRecur(unittest.TestCase):
    """Regression pins against the exact defect being removed."""

    def test_old_formula_collapsed_to_equity_over_price(self):
        """Documents the bug: risk_pct and stop_pct cancel."""
        for price in (1.10, 150.0, 60_000.0):
            old_size = (EQUITY * 0.02) / (price * 0.02)
            self.assertAlmostEqual(old_size, EQUITY / price, places=6,
                                   msg="the two 2%s cancel - no risk parameter survives")

    def test_risk_pct_now_actually_changes_the_size(self):
        """Under the old code this was impossible."""
        a = ps.size_position('EUR-USD', 1.10, EQUITY, risk_per_trade=0.005,
                             stop_price=1.0965, max_leverage_pct=None)
        b = ps.size_position('EUR-USD', 1.10, EQUITY, risk_per_trade=0.010,
                             stop_price=1.0965, max_leverage_pct=None)
        self.assertAlmostEqual(b.size / a.size, 2.0, places=6,
                               msg="doubling risk must double size")

    def test_stop_distance_actually_changes_the_size(self):
        tight = ps.size_position('EUR-USD', 1.10, EQUITY, stop_distance=0.0010,
                                 max_leverage_pct=None)
        wide = ps.size_position('EUR-USD', 1.10, EQUITY, stop_distance=0.0020,
                                max_leverage_pct=None)
        self.assertAlmostEqual(tight.size / wide.size, 2.0, places=6,
                               msg="a tighter stop must permit a larger position")

    def test_not_every_position_is_20_percent_notional(self):
        """
        The old code produced exactly 20% of equity notional every time,
        because the leverage cap always bound. Real risk sizing must not.
        """
        r = ps.size_position('EUR-USD', 1.10, EQUITY,
                             risk_per_trade=0.005, stop_price=1.0500)
        self.assertLess(r.notional_pct_of_equity, 0.20)
        self.assertEqual(r.bound_by, ps.BOUND_RISK)

    def test_tight_fx_stops_still_hit_the_leverage_cap(self):
        """
        Operational reality worth knowing: on FX, a 0.5% risk budget with a
        stop tighter than ~400 pips wants more notional than the 20% cap
        allows. The cap then silently reduces actual risk below the request --
        which is safe, but means the risk number is not the one you asked for.
        This is exactly what bound_by and is_real_risk_sizing exist to surface.
        """
        r = ps.size_position('EUR-USD', 1.10, EQUITY,
                             risk_per_trade=0.005, stop_price=1.0800)
        self.assertEqual(r.bound_by, ps.BOUND_LEVERAGE)
        self.assertFalse(r.is_real_risk_sizing)
        self.assertLess(r.risk_pct_of_equity, 0.005)
        self.assertTrue(any('capped' in w.lower() for w in r.warnings))


class TestRiskIsHonoured(unittest.TestCase):
    """Loss at the stop must equal the requested fraction of equity."""

    def test_loss_at_stop_equals_requested_risk(self):
        r = ps.size_position('EUR-USD', 1.10, EQUITY, risk_per_trade=0.005,
                             stop_price=1.0950, max_leverage_pct=None)
        loss = r.size * abs(1.10 - 1.0950)
        self.assertAlmostEqual(loss, EQUITY * 0.005, places=4)
        self.assertAlmostEqual(r.risk_pct_of_equity, 0.005, places=6)

    def test_works_across_asset_classes_with_real_stops(self):
        cases = [
            ('EUR-USD', 1.10, 1.0950),
            ('USD-JPY', 150.0, 149.25),
            ('BTC-USD', 60_000.0, 58_200.0),
        ]
        for sym, px, stop in cases:
            r = ps.size_position(sym, px, EQUITY, risk_per_trade=0.005,
                                 stop_price=stop, max_leverage_pct=None,
                                 max_position_size=None)
            loss = r.size * abs(px - stop)
            self.assertAlmostEqual(loss, EQUITY * 0.005, places=3,
                                   msg=f"{sym} risk must be equity-relative, not notional-relative")


class TestStopResolution(unittest.TestCase):
    """Priority order and honest reporting of where the stop came from."""

    def test_explicit_distance_wins(self):
        r = ps.size_position('EUR-USD', 1.10, EQUITY,
                             stop_distance=0.0050, stop_price=1.0000)
        self.assertEqual(r.stop_source, ps.STOP_EXPLICIT_DISTANCE)
        self.assertAlmostEqual(r.stop_distance, 0.0050)

    def test_stop_price_used_when_no_distance(self):
        r = ps.size_position('EUR-USD', 1.10, EQUITY, stop_price=1.0950)
        self.assertEqual(r.stop_source, ps.STOP_EXPLICIT_PRICE)
        self.assertAlmostEqual(r.stop_distance, 0.0050, places=6)

    def test_volatility_fallback_when_history_exists(self):
        vt = ps.VolatilityTracker()
        px = 1.10
        for i in range(60):
            px *= 1.0 + (0.001 if i % 2 else -0.001)
            vt.update('EUR-USD', px)
        r = ps.size_position('EUR-USD', px, EQUITY, vol_tracker=vt)
        self.assertEqual(r.stop_source, ps.STOP_VOLATILITY)
        self.assertTrue(any('volatility' in w.lower() for w in r.warnings),
                        "must warn that risk was estimated, not measured")

    def test_asset_default_when_no_history(self):
        r = ps.size_position('BTC-USD', 60_000.0, EQUITY)
        self.assertEqual(r.stop_source, ps.STOP_ASSET_DEFAULT)
        self.assertTrue(r.warnings)

    def test_flat_two_percent_default_is_gone(self):
        """
        The specific defect: a flat 2% of price for every instrument. FX and
        crypto defaults must differ, because 2% means different things.
        """
        fx = ps.size_position('EUR-USD', 1.10, EQUITY)
        crypto = ps.size_position('BTC-USD', 60_000.0, EQUITY)
        fx_frac = fx.stop_distance / 1.10
        crypto_frac = crypto.stop_distance / 60_000.0
        self.assertNotAlmostEqual(fx_frac, crypto_frac, places=4)
        self.assertGreater(crypto_frac, fx_frac,
                           "crypto is more volatile and needs a wider default stop")

    def test_is_real_risk_sizing_flag(self):
        real = ps.size_position('EUR-USD', 1.10, EQUITY, stop_price=1.0950,
                                max_leverage_pct=None)
        self.assertTrue(real.is_real_risk_sizing)

        guessed = ps.size_position('EUR-USD', 1.10, EQUITY)
        self.assertFalse(guessed.is_real_risk_sizing,
                         "a fallback stop is not real risk sizing")


class TestCapsAndSafety(unittest.TestCase):

    def test_leverage_cap_is_reported_not_hidden(self):
        # A very tight stop wants a huge position; the cap must bind AND say so.
        r = ps.size_position('EUR-USD', 1.10, EQUITY, risk_per_trade=0.005,
                             stop_distance=0.00001, max_leverage_pct=0.20)
        self.assertEqual(r.bound_by, ps.BOUND_LEVERAGE)
        self.assertTrue(any('capped' in w.lower() for w in r.warnings))
        self.assertLess(r.risk_pct_of_equity, 0.005,
                        "capping reduces actual risk below the request")

    def test_max_position_size_cap(self):
        r = ps.size_position('EUR-USD', 1.10, EQUITY, stop_distance=0.00001,
                             max_position_size=1000, max_leverage_pct=None)
        self.assertEqual(r.bound_by, ps.BOUND_MAX_SIZE)
        self.assertEqual(r.size, 1000)

    def test_zero_stop_distance_cannot_produce_infinite_size(self):
        r = ps.size_position('EUR-USD', 1.10, EQUITY, stop_distance=0)
        self.assertTrue(r.size > 0 and r.size != float('inf'))
        self.assertNotEqual(r.stop_source, ps.STOP_EXPLICIT_DISTANCE)

    def test_stop_equal_to_entry_is_rejected(self):
        r = ps.size_position('EUR-USD', 1.10, EQUITY, stop_price=1.10)
        self.assertNotEqual(r.stop_source, ps.STOP_EXPLICIT_PRICE)
        self.assertTrue(any('equals entry' in w for w in r.warnings))

    def test_bad_inputs_return_zero_not_garbage(self):
        for kwargs in ({'price': 0}, {'price': -1}, {'equity': 0}, {'equity': -5}):
            args = {'symbol': 'EUR-USD', 'price': 1.10, 'equity': EQUITY}
            args.update(kwargs)
            r = ps.size_position(**args)
            self.assertEqual(r.size, 0.0)
            self.assertEqual(r.bound_by, ps.BOUND_ZERO)
            self.assertTrue(r.warnings)


class TestVolatilityTracker(unittest.TestCase):

    def test_needs_minimum_samples(self):
        vt = ps.VolatilityTracker()
        for i in range(5):
            vt.update('EUR-USD', 1.10 + i * 0.001)
        self.assertIsNone(vt.sigma('EUR-USD'))

    def test_measures_higher_vol_as_higher_sigma(self):
        calm, wild = ps.VolatilityTracker(), ps.VolatilityTracker()
        p1 = p2 = 100.0
        for i in range(60):
            p1 *= 1.0 + (0.0005 if i % 2 else -0.0005)
            p2 *= 1.0 + (0.02 if i % 2 else -0.02)
            calm.update('X', p1)
            wild.update('X', p2)
        _cs, _ws = calm.sigma('X'), wild.sigma('X')
        assert _cs is not None and _ws is not None
        self.assertLess(_cs, _ws)

    def test_ignores_bad_prices(self):
        vt = ps.VolatilityTracker()
        for p in (1.10, 0, -1, None, 1.11):
            vt.update('EUR-USD', p)
        self.assertIsNone(vt.sigma('EUR-USD'))  # too few valid samples


def main():
    print("=" * 70)
    print("POSITION SIZING - TEST SUITE")
    print("=" * 70)
    print("Old: size = (equity*0.02)/(price*0.02) = equity/price, then always")
    print("     capped to 20% notional. No risk parameter survived.")
    print("New: size = (equity*risk_pct)/distance_to_stop, stop from strategy.")
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