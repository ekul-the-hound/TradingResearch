# ==============================================================================
# test_entry_guard.py -- Tests for the pre-order entry feasibility guard
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import unittest
from dataclasses import dataclass

from entry_guard import (
    EntryGuard, EntryGuardConfig, EntryRequest, EntryDecision,
    PASS, WARN, BLOCK,
)


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    last: float = 0.0


@dataclass
class Bal:
    total_equity: float
    free_margin: float
    used_margin: float = 0.0


def good_request(**overrides):
    base = dict(
        symbol="EURUSD", side="long", size=10_000, price=1.10,
        stop_distance=0.0020,
        tick=Tick("EURUSD", 1.09998, 1.10002),
        balance=Bal(total_equity=100_000, free_margin=95_000),
        remaining_daily_budget=4000.0, current_open_risk=500.0,
    )
    base.update(overrides)
    return EntryRequest(**base)  # type: ignore[arg-type]


def sev(decision: EntryDecision, name: str) -> str:
    for c in decision.checks:
        if c.name == name:
            return c.severity
    raise AssertionError(f"no check named {name}")


class TestHappyPath(unittest.TestCase):
    def test_clean_entry_allowed(self):
        d = EntryGuard().check(good_request())
        self.assertTrue(d.allowed)
        self.assertEqual(sev(d, "size_valid"), PASS)
        self.assertEqual(sev(d, "spread"), PASS)
        self.assertEqual(sev(d, "notional_cap"), PASS)
        self.assertEqual(sev(d, "free_margin"), PASS)
        self.assertEqual(sev(d, "daily_risk"), PASS)

    def test_bool_protocol(self):
        self.assertTrue(bool(EntryGuard().check(good_request())))


class TestSizeValidity(unittest.TestCase):
    def test_zero_size_blocks(self):
        d = EntryGuard().check(good_request(size=0))
        self.assertFalse(d.allowed)
        self.assertEqual(sev(d, "size_valid"), BLOCK)

    def test_negative_size_blocks(self):
        d = EntryGuard().check(good_request(size=-5))
        self.assertEqual(sev(d, "size_valid"), BLOCK)

    def test_zero_price_blocks(self):
        d = EntryGuard().check(good_request(price=0))
        self.assertEqual(sev(d, "size_valid"), BLOCK)

    def test_bad_side_blocks(self):
        d = EntryGuard().check(good_request(side="sideways"))
        self.assertEqual(sev(d, "size_valid"), BLOCK)


class TestSpread(unittest.TestCase):
    def test_wide_spread_blocks(self):
        # 0.0020 spread on ~1.10 mid ~= 18 bps > 10 default
        d = EntryGuard().check(good_request(
            tick=Tick("EURUSD", 1.09900, 1.10100)))
        self.assertEqual(sev(d, "spread"), BLOCK)
        self.assertFalse(d.allowed)

    def test_crossed_market_blocks(self):
        d = EntryGuard().check(good_request(
            tick=Tick("EURUSD", 1.10010, 1.10005)))
        self.assertEqual(sev(d, "spread"), BLOCK)

    def test_missing_tick_warns_not_blocks(self):
        d = EntryGuard().check(good_request(tick=None))
        self.assertEqual(sev(d, "spread"), WARN)
        self.assertTrue(d.allowed)  # warn allows when not strict

    def test_missing_tick_blocks_under_strict(self):
        d = EntryGuard(EntryGuardConfig(strict=True)).check(
            good_request(tick=None))
        self.assertFalse(d.allowed)


class TestNotionalCap(unittest.TestCase):
    def test_over_leveraged_blocks(self):
        # size 30k @ 1.10 = 33k notional = 33% of 100k > 20% cap
        d = EntryGuard().check(good_request(size=30_000))
        self.assertEqual(sev(d, "notional_cap"), BLOCK)
        self.assertFalse(d.allowed)

    def test_at_cap_allowed(self):
        # exactly 20%: notional 20000 -> size ~18181.8 @ 1.10
        d = EntryGuard().check(good_request(size=18_181, current_open_risk=0,
                                            stop_distance=0.0001))
        self.assertEqual(sev(d, "notional_cap"), PASS)

    def test_no_balance_warns(self):
        d = EntryGuard().check(good_request(balance=None))
        self.assertEqual(sev(d, "notional_cap"), WARN)

    def test_custom_cap(self):
        g = EntryGuard(EntryGuardConfig(max_notional_pct_of_equity=0.05))
        d = g.check(good_request())  # 11% > 5%
        self.assertEqual(sev(d, "notional_cap"), BLOCK)


class TestFreeMargin(unittest.TestCase):
    def test_insufficient_free_margin_blocks(self):
        d = EntryGuard().check(good_request(
            balance=Bal(total_equity=100_000, free_margin=5_000)))
        # notional 11000 * 1.1 buffer = 12100 > 5000 free
        self.assertEqual(sev(d, "free_margin"), BLOCK)
        self.assertFalse(d.allowed)

    def test_sufficient_free_margin_passes(self):
        d = EntryGuard().check(good_request())
        self.assertEqual(sev(d, "free_margin"), PASS)

    def test_no_balance_warns(self):
        d = EntryGuard().check(good_request(balance=None))
        self.assertEqual(sev(d, "free_margin"), WARN)


class TestDailyRisk(unittest.TestCase):
    def test_over_budget_blocks(self):
        # size 10000 * stop 0.05 = 500 risk + 3800 open = 4300 > 4000 budget
        d = EntryGuard().check(good_request(
            stop_distance=0.05, current_open_risk=3800.0))
        self.assertEqual(sev(d, "daily_risk"), BLOCK)
        self.assertFalse(d.allowed)

    def test_within_budget_passes(self):
        d = EntryGuard().check(good_request())
        self.assertEqual(sev(d, "daily_risk"), PASS)

    def test_no_stop_warns(self):
        d = EntryGuard().check(good_request(stop_distance=0))
        self.assertEqual(sev(d, "daily_risk"), WARN)

    def test_no_budget_warns(self):
        d = EntryGuard().check(good_request(remaining_daily_budget=None))
        self.assertEqual(sev(d, "daily_risk"), WARN)

    def test_no_stop_blocks_under_strict(self):
        d = EntryGuard(EntryGuardConfig(strict=True)).check(
            good_request(stop_distance=0))
        self.assertFalse(d.allowed)


class TestBrokerMarginHonesty(unittest.TestCase):
    def test_broker_margin_always_unverifiable_warn(self):
        d = EntryGuard().check(good_request())
        self.assertEqual(sev(d, "broker_margin"), WARN)
        # It must not silently pass -- honesty about the limitation.
        detail = next(c.detail for c in d.checks if c.name == "broker_margin")
        self.assertIn("UNVERIFIABLE", detail)

    def test_unverifiable_blocks_under_strict(self):
        # With everything else clean, the only WARN is broker_margin.
        d = EntryGuard(EntryGuardConfig(strict=True)).check(good_request())
        self.assertFalse(d.allowed)  # strict blocks on the unverifiable dimension


class TestDecisionAggregation(unittest.TestCase):
    def test_reasons_excludes_pass(self):
        d = EntryGuard().check(good_request())
        # Only the broker_margin WARN should surface as a reason.
        self.assertTrue(all("PASS" not in r for r in d.reasons))
        self.assertTrue(any("broker_margin" in r for r in d.reasons))

    def test_multiple_blocks_all_recorded(self):
        d = EntryGuard().check(good_request(
            size=30_000,  # notional block
            tick=Tick("EURUSD", 1.09900, 1.10100),  # spread block
        ))
        severities = {c.name: c.severity for c in d.checks}
        self.assertEqual(severities["notional_cap"], BLOCK)
        self.assertEqual(severities["spread"], BLOCK)
        self.assertFalse(d.allowed)


if __name__ == "__main__":
    unittest.main(verbosity=2)