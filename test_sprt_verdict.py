# ==============================================================================
# test_sprt_verdict.py -- Tests for the SPRT demo verdict engine
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# Uses deterministic outcome sequences (no RNG) so verdicts are reproducible.
# ==============================================================================

import unittest

from sprt_verdict import (
    SPRT, SPRTConfig, SPRTState, verdict_from_outcomes,
    KEEP, KILL, WATCH,
)


def stream(win_rate_pattern):
    """Expand a pattern like [True,True,False] repeated into a long list."""
    return win_rate_pattern * 60


class TestValidation(unittest.TestCase):
    def test_p1_must_be_below_p0(self):
        with self.assertRaises(ValueError):
            SPRT(SPRTConfig(p0_win_rate=0.45, p1_win_rate=0.55))

    def test_p0_below_one(self):
        with self.assertRaises(ValueError):
            SPRT(SPRTConfig(p0_win_rate=1.0, p1_win_rate=0.5))

    def test_alpha_in_range(self):
        with self.assertRaises(ValueError):
            SPRT(SPRTConfig(alpha=0.0))


class TestBoundaries(unittest.TestCase):
    def test_symmetric_for_equal_error_rates(self):
        sprt = SPRT(SPRTConfig(alpha=0.05, beta=0.05))
        b = sprt.boundaries()
        self.assertAlmostEqual(b["keep_boundary"], -b["kill_boundary"], places=6)

    def test_win_decrements_lr_toward_keep(self):
        # With p1 < p0, a win makes H1 (kill) LESS likely -> negative increment.
        sprt = SPRT(SPRTConfig())
        self.assertLess(sprt.boundaries()["win_increment"], 0)

    def test_loss_increments_lr_toward_kill(self):
        sprt = SPRT(SPRTConfig())
        self.assertGreater(sprt.boundaries()["loss_increment"], 0)


class TestKeepVerdict(unittest.TestCase):
    def test_strong_edge_keeps(self):
        # 70% win rate, well above p0=0.55 -> KEEP.
        s = verdict_from_outcomes(stream([True, True, True, False]),
                                  SPRTConfig(min_trades=20))
        self.assertEqual(s.verdict, KEEP)

    def test_at_p0_tends_keep(self):
        # Exactly the backtested rate should lean KEEP, not KILL.
        s = verdict_from_outcomes(
            stream([True, True, True, True, True, True, True, True, True, True,
                    True, False, False, False, False, False, False, False,
                    False]),  # ~55%
            SPRTConfig(min_trades=20))
        self.assertIn(s.verdict, (KEEP, WATCH))


class TestKillVerdict(unittest.TestCase):
    def test_degraded_edge_kills(self):
        # 30% win rate, well below p1=0.45 -> KILL.
        s = verdict_from_outcomes(stream([True, False, False, False]),
                                  SPRTConfig(min_trades=20))
        self.assertEqual(s.verdict, KILL)

    def test_all_losses_kills_fast(self):
        s = verdict_from_outcomes([False] * 40, SPRTConfig(min_trades=20))
        self.assertEqual(s.verdict, KILL)


class TestWatchVerdict(unittest.TestCase):
    def test_below_min_trades_watches(self):
        sprt = SPRT(SPRTConfig(min_trades=20))
        for _ in range(10):
            sprt.update(False)  # even all losses can't decide before floor
        self.assertEqual(sprt.state.verdict, WATCH)

    def test_ambiguous_evidence_watches(self):
        # Right at the p0/p1 midpoint (0.50), evidence accumulates slowly.
        sprt = SPRT(SPRTConfig(p0_win_rate=0.55, p1_win_rate=0.45,
                               min_trades=20))
        # 25 trades alternating -> unlikely to cross either boundary yet
        for i in range(25):
            sprt.update(i % 2 == 0)
        self.assertEqual(sprt.state.verdict, WATCH)


class TestMinTradesFloor(unittest.TestCase):
    def test_no_verdict_before_floor_even_with_streak(self):
        sprt = SPRT(SPRTConfig(min_trades=30))
        for _ in range(29):
            sprt.update(False)
        self.assertEqual(sprt.state.verdict, WATCH)
        # 30th loss can now allow a KILL.
        sprt.update(False)
        self.assertIn(sprt.state.verdict, (KILL, WATCH))


class TestMaxTradesCap(unittest.TestCase):
    def test_max_trades_forces_verdict(self):
        # Alternating outcomes never cross a boundary; max_trades forces one.
        sprt = SPRT(SPRTConfig(min_trades=10, max_trades=40))
        for i in range(40):
            sprt.update(i % 2 == 0)
        self.assertIn(sprt.state.verdict, (KEEP, KILL))
        self.assertIn("max_trades", sprt.state.reason)

    def test_no_cap_stays_watching(self):
        sprt = SPRT(SPRTConfig(min_trades=10, max_trades=0))
        for i in range(60):
            sprt.update(i % 2 == 0)
        self.assertEqual(sprt.state.verdict, WATCH)


class TestStickyVerdict(unittest.TestCase):
    def test_verdict_does_not_change_after_decided(self):
        sprt = SPRT(SPRTConfig(min_trades=20))
        for _ in range(40):
            sprt.update(False)  # -> KILL
        self.assertEqual(sprt.state.verdict, KILL)
        n_at_decision = sprt.state.n
        # Feeding wins after the decision must not flip or advance it.
        for _ in range(50):
            sprt.update(True)
        self.assertEqual(sprt.state.verdict, KILL)
        self.assertEqual(sprt.state.n, n_at_decision)


class TestReturnInterface(unittest.TestCase):
    def test_positive_return_is_win(self):
        sprt = SPRT(SPRTConfig(min_trades=5))
        sprt.update_return(1.5)
        self.assertEqual(sprt.state.wins, 1)

    def test_negative_return_is_loss(self):
        sprt = SPRT(SPRTConfig(min_trades=5))
        sprt.update_return(-0.8)
        self.assertEqual(sprt.state.losses, 1)

    def test_zero_return_is_loss(self):
        # Not > 0 -> counted as a loss (breakeven is not an edge).
        sprt = SPRT(SPRTConfig(min_trades=5))
        sprt.update_return(0.0)
        self.assertEqual(sprt.state.losses, 1)


class TestErrorRateBehaviour(unittest.TestCase):
    def test_tighter_error_rates_need_more_evidence(self):
        # Lower alpha/beta -> wider boundaries -> more trades to decide.
        loose = SPRT(SPRTConfig(alpha=0.1, beta=0.1))
        tight = SPRT(SPRTConfig(alpha=0.01, beta=0.01))
        self.assertGreater(abs(tight.boundaries()["kill_boundary"]),
                           abs(loose.boundaries()["kill_boundary"]))


class TestReset(unittest.TestCase):
    def test_reset_clears_state(self):
        sprt = SPRT(SPRTConfig(min_trades=5))
        for _ in range(10):
            sprt.update(False)
        sprt.reset()
        self.assertEqual(sprt.state.n, 0)
        self.assertEqual(sprt.state.verdict, WATCH)


class TestWinRate(unittest.TestCase):
    def test_win_rate_computed(self):
        sprt = SPRT(SPRTConfig(min_trades=100))  # high floor to stay in WATCH
        sprt.update_many([True, True, True, False])
        self.assertAlmostEqual(sprt.state.win_rate, 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
