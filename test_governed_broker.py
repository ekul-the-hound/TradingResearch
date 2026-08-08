# ==============================================================================
# test_governed_broker.py
# ==============================================================================

import sys
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from broker_adapter import (
    BaseBroker, BrokerBalance, BrokerOrder, BrokerPosition, BrokerTick,
    OrderSide, OrderStatus, OrderType,
)
from firm_rules import FirmRules, ftmo, generic_trailing
from live_governor import (
    AccountState, Decision, GovernorConfig, LiveGovernor,
)

from governed_broker import (
    GovernedBroker, account_state_from_broker,
)

ACCOUNT = 100_000.0


class FakeBroker(BaseBroker):
    """Records every call so tests can assert the gate stopped things."""

    def __init__(self, equity=ACCOUNT, unrealized=0.0, positions=None,
                 raise_on_balance=False):
        super().__init__(name='fake')
        self.equity = equity
        self.unrealized = unrealized
        self._positions = positions or []
        self.raise_on_balance = raise_on_balance
        self.submitted: List[Dict[str, Any]] = []
        self.cancelled: List[str] = []

    def connect(self):
        self.is_connected = True
        return True

    def disconnect(self):
        self.is_connected = False

    def get_balance(self) -> BrokerBalance:
        if self.raise_on_balance:
            raise RuntimeError('broker offline')
        return BrokerBalance(
            total_equity=self.equity, free_margin=self.equity,
            used_margin=0.0, unrealized_pnl=self.unrealized,
            currency='USD', timestamp='')

    def get_positions(self) -> List[BrokerPosition]:
        return list(self._positions)

    def get_position(self, symbol) -> Optional[BrokerPosition]:
        for p in self._positions:
            if p.symbol == symbol:
                return p
        return None

    def get_tick(self, symbol) -> Optional[BrokerTick]:
        return BrokerTick(symbol=symbol, bid=1.0, ask=1.0, last=1.0)

    def get_order(self, order_id) -> Optional[BrokerOrder]:
        return None

    def cancel_order(self, order_id) -> bool:
        self.cancelled.append(order_id)
        return True

    def submit_order(self, side, symbol, size, order_type='market',
                     price=None, stop_price=None) -> BrokerOrder:
        self.submitted.append(
            {'side': side, 'symbol': symbol, 'size': size})
        return BrokerOrder(
            order_id='X1', symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET, size=size,
            status=OrderStatus.FILLED, fill_price=1.0, filled_size=size,
            timestamp='')


def pos(symbol='EURUSD', side='long', size=1.0):
    return BrokerPosition(symbol=symbol, side=side, size=size,
                          entry_price=1.0, current_price=1.0,
                          unrealized_pnl=0.0, realized_pnl=0.0)


def governed(equity=ACCOUNT, rules=None, positions=None, **kw):
    inner = FakeBroker(equity=equity, positions=positions)
    g = LiveGovernor(GovernorConfig(rules=rules or ftmo()))
    g.seed_anchor(g.trading_date(datetime.utcnow()), ACCOUNT)
    return GovernedBroker(inner, g, initial_balance=ACCOUNT, **kw), inner, g


# ==============================================================================
# CONSTRUCTION
# ==============================================================================

class TestConstruction(unittest.TestCase):

    def test_rejects_non_positive_initial_balance(self):
        inner = FakeBroker()
        g = LiveGovernor()
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                GovernedBroker(inner, g, initial_balance=bad)

    def test_rejects_bad_reduce_factor(self):
        inner = FakeBroker()
        g = LiveGovernor()
        for bad in (0.0, 1.5, -0.2):
            with self.assertRaises(ValueError):
                GovernedBroker(inner, g, initial_balance=ACCOUNT,
                               reduce_size_factor=bad)

    def test_name_shows_the_wrapping(self):
        b, _, _ = governed()
        self.assertIn('governed', b.name)
        self.assertIn('fake', b.name)


# ==============================================================================
# THE GATE
# ==============================================================================

class TestGate(unittest.TestCase):

    def test_healthy_account_passes_through(self):
        b, inner, _ = governed(equity=ACCOUNT)
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.FILLED)
        self.assertEqual(len(inner.submitted), 1)

    def test_halt_blocks_and_inner_broker_never_called(self):
        """The order must not reach the broker at all."""
        b, inner, _ = governed(equity=95_900)      # past the 80% halt band
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(inner.submitted, [])
        self.assertEqual(b.blocked_count, 1)

    def test_rejection_carries_the_reason(self):
        b, _, _ = governed(equity=95_900)
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIn('governor', o.raw['error'].lower())
        self.assertIn('governor_decision', o.raw)
        self.assertIsNotNone(o.raw['headroom'])

    def test_reduce_band_scales_the_size(self):
        b, inner, _ = governed(equity=96_900)      # reduce band
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertEqual(len(inner.submitted), 1)
        self.assertAlmostEqual(inner.submitted[0]['size'], 50_000)

    def test_reduce_factor_configurable(self):
        b, inner, _ = governed(equity=96_900, reduce_size_factor=0.25)
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertAlmostEqual(inner.submitted[0]['size'], 25_000)

    def test_breach_flattens_and_blocks(self):
        b, inner, _ = governed(equity=94_000, positions=[pos()])
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(b.flatten_count, 1)
        # the only order sent was the closing one
        self.assertEqual(len(inner.submitted), 1)
        self.assertEqual(inner.submitted[0]['side'], 'sell')

    def test_auto_flatten_can_be_disabled(self):
        b, inner, _ = governed(equity=94_000, positions=[pos()],
                               auto_flatten=False)
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertEqual(b.flatten_count, 0)
        self.assertEqual(inner.submitted, [])


class TestFailClosed(unittest.TestCase):
    """
    A wrapper that falls back to the inner broker on error is worse than no
    wrapper: the operator believes orders are being checked.
    """

    def test_unreadable_account_blocks_the_order(self):
        inner = FakeBroker(raise_on_balance=True)
        g = LiveGovernor()
        b = GovernedBroker(inner, g, initial_balance=ACCOUNT,
                           auto_flatten=False)
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(inner.submitted, [])

    def test_unreadable_account_is_a_flatten_verdict(self):
        inner = FakeBroker(raise_on_balance=True)
        b = GovernedBroker(inner, LiveGovernor(), initial_balance=ACCOUNT,
                           auto_flatten=False)
        v = b.check()
        self.assertIs(v.decision, Decision.FLATTEN)
        self.assertEqual(v.reason, 'account_state_unavailable')

    def test_check_never_raises(self):
        """The caller is a trading loop; an exception there is one `except`
        away from being read as 'carry on'."""
        inner = FakeBroker(raise_on_balance=True)
        b = GovernedBroker(inner, LiveGovernor(), initial_balance=ACCOUNT,
                           auto_flatten=False)
        b.check()          # must not raise

    def test_governor_internal_error_blocks(self):
        b, inner, g = governed()
        g.config = None                     # type: ignore[assignment]
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(inner.submitted, [])


class TestHeartbeat(unittest.TestCase):
    """
    A breach can happen with no signal firing: an open position drifting
    against you moves equity without anyone submitting anything.
    """

    def test_heartbeat_flattens_without_an_order(self):
        b, inner, _ = governed(equity=94_000, positions=[pos()])
        v = b.heartbeat()
        self.assertTrue(v.must_flatten)
        self.assertEqual(len(inner.submitted), 1)

    def test_healthy_heartbeat_does_nothing(self):
        b, inner, _ = governed(equity=ACCOUNT, positions=[pos()])
        v = b.heartbeat()
        self.assertIs(v.decision, Decision.ALLOW)
        self.assertEqual(inner.submitted, [])

    def test_heartbeat_respects_auto_flatten_off(self):
        b, inner, _ = governed(equity=94_000, positions=[pos()],
                               auto_flatten=False)
        b.heartbeat()
        self.assertEqual(inner.submitted, [])


# ==============================================================================
# PASS-THROUGH
# ==============================================================================

class TestPassThrough(unittest.TestCase):

    def test_reads_are_not_gated(self):
        b, inner, _ = governed(equity=94_000)     # would block an order
        self.assertIsNotNone(b.get_tick('EURUSD'))
        self.assertIsNotNone(b.get_balance())
        self.assertEqual(b.get_positions(), [])

    def test_cancel_is_never_gated(self):
        """Cancelling can only reduce exposure."""
        b, inner, _ = governed(equity=94_000)
        self.assertTrue(b.cancel_order('X1'))
        self.assertEqual(inner.cancelled, ['X1'])

    def test_connect_delegates(self):
        b, inner, _ = governed()
        b.connect()
        self.assertTrue(inner.is_connected)
        self.assertTrue(b.is_connected)
        b.disconnect()
        self.assertFalse(inner.is_connected)


# ==============================================================================
# STATE BRIDGE
# ==============================================================================

class TestAccountStateBridge(unittest.TestCase):

    def test_equity_and_balance_derived_separately(self):
        """
        Collapsing them disables intraday protection for firms whose daily
        rule counts floating P&L.
        """
        inner = FakeBroker(equity=98_000, unrealized=-2_000)
        s = account_state_from_broker(inner, ACCOUNT)
        self.assertAlmostEqual(s.equity, 98_000)
        self.assertAlmostEqual(s.balance, 100_000)

    def test_prefers_native_to_account_state(self):
        class Native(FakeBroker):
            def to_account_state(self, initial_balance):
                return AccountState(timestamp=datetime.utcnow(),
                                    balance=1.0, equity=2.0,
                                    initial_balance=initial_balance)
        s = account_state_from_broker(Native(), ACCOUNT)
        self.assertAlmostEqual(s.equity, 2.0)

    def test_counts_open_positions(self):
        inner = FakeBroker(positions=[pos('EURUSD'), pos('GBPUSD')])
        s = account_state_from_broker(inner, ACCOUNT)
        self.assertEqual(s.open_positions, 2)
        self.assertIn('GBPUSD', s.symbol_exposure)


# ==============================================================================
# AUDIT
# ==============================================================================

class TestAudit(unittest.TestCase):

    def test_every_decision_recorded(self):
        b, _, _ = governed(equity=ACCOUNT)
        b.submit_order('buy', 'EURUSD', 100_000)
        b.submit_order('sell', 'EURUSD', 50_000)
        self.assertEqual(len(b.events), 2)
        self.assertEqual(b.events[0].symbol, 'EURUSD')

    def test_blocked_orders_recorded_with_zero_executed(self):
        b, _, _ = governed(equity=95_900)
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertEqual(b.events[0].executed_size, 0.0)
        self.assertEqual(b.events[0].requested_size, 100_000)

    def test_reduced_order_records_both_sizes(self):
        b, _, _ = governed(equity=96_900)
        b.submit_order('buy', 'EURUSD', 100_000)
        e = b.events[0]
        self.assertEqual(e.requested_size, 100_000)
        self.assertAlmostEqual(e.executed_size, 50_000)

    def test_summary_shape(self):
        b, _, _ = governed(equity=95_900)
        b.submit_order('buy', 'EURUSD', 100_000)
        s = b.summary()
        self.assertEqual(s['blocked_orders'], 1)
        self.assertIsNotNone(s['last_decision'])

    def test_unchecked_rules_reach_the_summary(self):
        b, _, _ = governed(rules=generic_trailing())
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIn('trailing_drawdown_eod', b.summary()['unchecked_rules'])

    def test_log_fn_called(self):
        seen = []
        b, _, _ = governed(equity=95_900, log_fn=seen.append)
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertTrue(any('GOVERNOR' in m for m in seen))


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('\n' + '=' * 68)
    print(f"  ran {result.testsRun} | failures {len(result.failures)} | "
          f"errors {len(result.errors)} | skipped {len(result.skipped)}")
    print('=' * 68)
    if result.skipped:
        for case, reason in result.skipped:
            print(f"    SKIPPED {case}: {reason}")
    return 0 if not (result.failures or result.errors or result.skipped) else 1


if __name__ == '__main__':
    sys.exit(main())
