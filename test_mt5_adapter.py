# ==============================================================================
# test_mt5_adapter.py
# ==============================================================================
# Everything here runs against an injected fake terminal. That covers the
# translation logic completely and the real terminal's behaviour not at all --
# see mt5_adapter.selftest_against_terminal() for the half that needs Windows.
# ==============================================================================

import sys
import unittest
from typing import Any, Dict, List, Optional

from broker_adapter import OrderSide, OrderStatus, OrderType

import mt5_adapter as M
from mt5_adapter import (
    MT5Broker, MT5Error, ORDER_FILLING_FOK, ORDER_FILLING_IOC,
    ORDER_FILLING_RETURN, SymbolSpec, choose_filling_mode, describe_retcode,
    normalize_volume, order_succeeded, resolve_symbol, units_to_lots,
)


# ==============================================================================
# FAKE TERMINAL
# ==============================================================================

class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeMT5:
    """Minimal stand-in shaped like the MetaTrader5 module."""

    def __init__(self, symbols=('EURUSD.raw', 'GBPUSD.raw', 'EURUSDCHF.raw'),
                 retcode=M.TRADE_RETCODE_DONE, tick=True, visible=True,
                 filling_mode=2, equity=100_000.0, balance=100_000.0,
                 init_ok=True, account_ok=True):
        self._symbols = list(symbols)
        self._retcode = retcode
        self._tick = tick
        self._visible = visible
        self._filling = filling_mode
        self._equity = equity
        self._balance = balance
        self.sent: List[Dict[str, Any]] = []
        self.selected: List[str] = []
        self.initialized = False
        self.positions: List[Any] = []
        self._init_ok = init_ok
        self._account_ok = account_ok

    def initialize(self, **kw):
        self.initialized = self._init_ok
        return self._init_ok

    def shutdown(self):
        self.initialized = False

    def last_error(self):
        return (0, 'ok')

    def symbols_get(self):
        return [Obj(name=n) for n in self._symbols]

    def symbol_info(self, name):
        if name not in self._symbols:
            return None
        return Obj(name=name, volume_min=0.01, volume_max=50.0,
                   volume_step=0.01, trade_contract_size=100_000.0,
                   filling_mode=self._filling, digits=5,
                   visible=self._visible)

    def symbol_select(self, name, enable=True):
        self.selected.append(name)
        self._visible = True
        return True

    def symbol_info_tick(self, name):
        if not self._tick:
            return None
        return Obj(bid=1.09995, ask=1.10005)

    def account_info(self):
        if not self._account_ok:
            return None
        return Obj(balance=self._balance, equity=self._equity,
                   margin_free=self._equity * 0.9, margin=100.0,
                   profit=self._equity - self._balance, currency='USD')

    def positions_get(self, symbol=None):
        if symbol is None:
            return list(self.positions)
        return [p for p in self.positions if p.symbol == symbol]

    def order_send(self, request):
        self.sent.append(dict(request))
        return Obj(retcode=self._retcode, order=12345, deal=999,
                   volume=request['volume'], price=request['price'],
                   comment='ok' if self._retcode == M.TRADE_RETCODE_DONE
                   else 'rejected')


def broker(**kw):
    fake = FakeMT5(**kw)
    b = MT5Broker(mt5_module=fake)
    return b, fake


SPEC = SymbolSpec(name='EURUSD', volume_min=0.01, volume_max=50.0,
                  volume_step=0.01, trade_contract_size=100_000.0)


# ==============================================================================
# 1. VOLUME -- lots, not units
# ==============================================================================

class TestVolume(unittest.TestCase):

    def test_units_to_lots(self):
        self.assertAlmostEqual(units_to_lots(100_000, SPEC), 1.0)
        self.assertAlmostEqual(units_to_lots(10_000, SPEC), 0.1)

    def test_zero_contract_size_refused(self):
        bad = SymbolSpec(name='X', trade_contract_size=0.0)
        with self.assertRaises(MT5Error):
            units_to_lots(100_000, bad)

    def test_exact_multiple_untouched(self):
        v, notes = normalize_volume(1.0, SPEC)
        self.assertAlmostEqual(v, 1.0)
        self.assertEqual(notes, [])

    def test_snaps_down_to_step(self):
        """
        DOWN, not nearest. Rounding up trades more than the sizer asked for,
        which makes the risk calculation advisory.
        """
        v, notes = normalize_volume(0.1235, SPEC)
        self.assertAlmostEqual(v, 0.12)
        self.assertTrue(notes)

    def test_never_rounds_up_even_when_closer(self):
        v, _ = normalize_volume(0.1999, SPEC)
        self.assertAlmostEqual(v, 0.19)

    def test_below_minimum_returns_zero_not_minimum(self):
        """
        Opening the smallest allowed trade when the model asked for less is
        the plumbing overriding the sizer.
        """
        v, notes = normalize_volume(0.004, SPEC)
        self.assertEqual(v, 0.0)
        self.assertTrue(any('below' in n for n in notes))

    def test_above_maximum_capped_and_flagged(self):
        v, notes = normalize_volume(90.0, SPEC)
        self.assertAlmostEqual(v, 50.0)
        self.assertTrue(any('NOT split' in n for n in notes))

    def test_zero_and_negative(self):
        for bad in (0.0, -1.0):
            v, notes = normalize_volume(bad, SPEC)
            self.assertEqual(v, 0.0)
            self.assertTrue(notes)

    def test_zero_step_refused(self):
        with self.assertRaises(MT5Error):
            normalize_volume(1.0, SymbolSpec(name='X', volume_step=0.0))

    def test_no_floating_point_dust(self):
        """0.1 + 0.2 territory: MT5 rejects 0.30000000000000004."""
        for lots in (0.3, 0.7, 1.1, 2.9):
            v, _ = normalize_volume(lots, SPEC)
            self.assertEqual(v, round(v, 2))

    def test_coarse_step_broker(self):
        coarse = SymbolSpec(name='IDX', volume_min=0.1, volume_max=10.0,
                            volume_step=0.1, trade_contract_size=1.0)
        v, _ = normalize_volume(0.37, coarse)
        self.assertAlmostEqual(v, 0.3)


# ==============================================================================
# 2. FILLING MODE -- a bitmask, not a value
# ==============================================================================

class TestFillingMode(unittest.TestCase):

    def test_ioc_preferred_when_available(self):
        """
        A partial fill beats no fill: the governor can size down next time,
        but cannot act on a position that never opened.
        """
        self.assertEqual(choose_filling_mode(
            SymbolSpec(name='X', filling_mode=3)), ORDER_FILLING_IOC)

    def test_fok_when_ioc_absent(self):
        self.assertEqual(choose_filling_mode(
            SymbolSpec(name='X', filling_mode=1)), ORDER_FILLING_FOK)

    def test_return_when_neither(self):
        self.assertEqual(choose_filling_mode(
            SymbolSpec(name='X', filling_mode=0)), ORDER_FILLING_RETURN)

    def test_mask_is_read_as_bits(self):
        """Bit positions are not the ORDER_FILLING_* values; 10030 lives here."""
        self.assertEqual(choose_filling_mode(
            SymbolSpec(name='X', filling_mode=2)), ORDER_FILLING_IOC)


# ==============================================================================
# 3. SYMBOL RESOLUTION
# ==============================================================================

class TestResolveSymbol(unittest.TestCase):

    def test_exact_match_wins(self):
        self.assertEqual(resolve_symbol('EURUSD', ['EURUSD', 'EURUSD.raw']),
                         'EURUSD')

    def test_suffix_matched(self):
        self.assertEqual(resolve_symbol('EURUSD', ['EURUSD.raw']), 'EURUSD.raw')

    def test_prefers_shortest_over_longer_pair(self):
        """EURUSD must not resolve to EURUSDCHF."""
        self.assertEqual(
            resolve_symbol('EURUSD', ['EURUSDCHF.raw', 'EURUSD.raw']),
            'EURUSD.raw')

    def test_separator_insensitive(self):
        self.assertEqual(resolve_symbol('EUR-USD', ['EURUSD']), 'EURUSD')

    def test_unknown_returns_none(self):
        self.assertIsNone(resolve_symbol('XAUUSD', ['EURUSD', 'GBPUSD']))


# ==============================================================================
# 4. RETCODES -- a result object is returned on failure too
# ==============================================================================

class TestRetcodes(unittest.TestCase):

    def test_done_is_success(self):
        self.assertTrue(order_succeeded(Obj(retcode=M.TRADE_RETCODE_DONE)))

    def test_partial_is_success(self):
        self.assertTrue(
            order_succeeded(Obj(retcode=M.TRADE_RETCODE_DONE_PARTIAL)))

    def test_rejection_is_not_success_despite_being_truthy(self):
        result = Obj(retcode=10014, order=0)
        self.assertTrue(bool(result))
        self.assertFalse(order_succeeded(result))

    def test_none_is_not_success(self):
        self.assertFalse(order_succeeded(None))

    def test_missing_retcode_is_not_success(self):
        self.assertFalse(order_succeeded(Obj(order=1)))

    def test_known_codes_are_named(self):
        self.assertIn('Invalid volume', describe_retcode(10014))
        self.assertIn('filling mode', describe_retcode(10030))

    def test_unknown_code_still_reported(self):
        self.assertIn('99999', describe_retcode(99999))


# ==============================================================================
# ADAPTER BEHAVIOUR
# ==============================================================================

class TestConnection(unittest.TestCase):

    def test_connect_and_disconnect(self):
        b, fake = broker()
        self.assertTrue(b.connect())
        self.assertTrue(b.is_connected)
        b.disconnect()
        self.assertFalse(b.is_connected)

    def test_failed_initialize_raises(self):
        b, fake = broker(init_ok=False)
        with self.assertRaises(MT5Error):
            b.connect()
        self.assertFalse(b.is_connected)

    def test_missing_package_message_is_specific(self):
        """
        Blocks the import rather than relying on the package being absent.

        The original asserted that `import MetaTrader5` fails, which is a
        statement about the machine, not the code. It passed on a box without
        the package and failed the moment one was installed -- a test whose
        result depends on the environment tests the environment.
        """
        import sys as _sys
        # sys.modules bound through an Any local: setting an entry to None to
        # force ImportError is documented CPython behaviour that the typeshed
        # stubs do not model (they require a ModuleType).
        mods: Any = _sys.modules
        saved = mods.get('MetaTrader5', '<<absent>>')
        mods['MetaTrader5'] = None              # forces ImportError
        try:
            b = MT5Broker()
            with self.assertRaises(MT5Error) as ctx:
                _ = b.mt5
            self.assertIn('Windows-only', str(ctx.exception))
        finally:
            if saved == '<<absent>>':
                mods.pop('MetaTrader5', None)
            else:
                mods['MetaTrader5'] = saved

    def test_injected_module_bypasses_the_import(self):
        """An injected terminal must not touch sys.modules at all."""
        b, fake = broker()
        self.assertIs(b.mt5, fake)


class TestSymbols(unittest.TestCase):

    def test_resolves_and_caches(self):
        b, fake = broker()
        self.assertEqual(b.broker_symbol('EURUSD'), 'EURUSD.raw')
        self.assertEqual(b._symbol_map['EURUSD'], 'EURUSD.raw')

    def test_unknown_symbol_raises_with_a_hint(self):
        b, fake = broker()
        with self.assertRaises(MT5Error) as ctx:
            b.broker_symbol('XAUUSD')
        self.assertIn('suffix', str(ctx.exception))

    def test_hidden_symbol_is_selected_into_market_watch(self):
        """Hidden means untradeable, and would fail later as an opaque reject."""
        b, fake = broker(visible=False)
        b.spec('EURUSD')
        self.assertIn('EURUSD.raw', fake.selected)


class TestSubmitOrder(unittest.TestCase):

    def test_market_buy_translates(self):
        b, fake = broker()
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.FILLED)
        req = fake.sent[0]
        self.assertEqual(req['symbol'], 'EURUSD.raw')
        self.assertAlmostEqual(req['volume'], 1.0)
        self.assertEqual(req['type'], M.ORDER_TYPE_BUY)
        self.assertEqual(req['type_filling'], ORDER_FILLING_IOC)

    def test_sell_uses_bid(self):
        b, fake = broker()
        b.submit_order('sell', 'EURUSD', 100_000)
        self.assertAlmostEqual(fake.sent[0]['price'], 1.09995)

    def test_buy_uses_ask(self):
        b, fake = broker()
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertAlmostEqual(fake.sent[0]['price'], 1.10005)

    def test_rejection_reported_not_swallowed(self):
        b, fake = broker(retcode=10014)
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertIn('Invalid volume', o.raw['error'])

    def test_sub_minimum_size_does_not_reach_the_broker(self):
        b, fake = broker()
        o = b.submit_order('buy', 'EURUSD', 200)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(fake.sent, [])

    def test_no_tick_rejects(self):
        b, fake = broker(tick=False)
        o = b.submit_order('buy', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertIn('market may be closed', o.raw['error'])

    def test_limit_order_rejected_not_silently_marketed(self):
        """
        Substituting a market order for a limit is how a limit strategy ends
        up chasing price.
        """
        b, fake = broker()
        o = b.submit_order('buy', 'EURUSD', 100_000, order_type='limit',
                           price=1.09)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(fake.sent, [])
        self.assertIn('not implemented', o.raw['error'])

    def test_unknown_side_rejected(self):
        b, fake = broker()
        o = b.submit_order('sideways', 'EURUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)

    def test_unknown_symbol_rejected_without_sending(self):
        b, fake = broker()
        o = b.submit_order('buy', 'XAUUSD', 100_000)
        self.assertIs(o.status, OrderStatus.REJECTED)
        self.assertEqual(fake.sent, [])

    def test_rejected_orders_are_recorded(self):
        b, fake = broker(retcode=10019)
        b.submit_order('buy', 'EURUSD', 100_000)
        self.assertEqual(len(b._order_history), 1)

    def test_cancel_returns_false_not_a_fake_success(self):
        b, _ = broker()
        self.assertFalse(b.cancel_order('12345'))


class TestPositions(unittest.TestCase):

    def test_reads_positions(self):
        b, fake = broker()
        fake.positions = [Obj(symbol='EURUSD.raw', volume=1.0, type=0,
                              price_open=1.1, price_current=1.11, profit=100.0)]
        pos = b.get_positions()
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0].side, 'long')

    def test_hedged_positions_net_out(self):
        """
        A hedging account can hold both directions at once; the risk rules
        care about the net.
        """
        b, fake = broker()
        fake.positions = [
            Obj(symbol='EURUSD.raw', volume=2.0, type=0, price_open=1.10,
                price_current=1.11, profit=200.0),
            Obj(symbol='EURUSD.raw', volume=0.5, type=1, price_open=1.12,
                price_current=1.11, profit=50.0),
        ]
        net = b.get_position('EURUSD')
        self.assertIsNotNone(net)
        assert net is not None
        self.assertEqual(net.side, 'long')
        self.assertAlmostEqual(net.size, 1.5)
        self.assertAlmostEqual(net.unrealized_pnl, 250.0)

    def test_no_position_returns_none(self):
        b, _ = broker()
        self.assertIsNone(b.get_position('EURUSD'))


class TestGovernorBridge(unittest.TestCase):

    def test_builds_account_state(self):
        b, fake = broker(equity=98_500.0, balance=100_000.0)
        state = b.to_account_state(initial_balance=100_000.0)
        self.assertAlmostEqual(state.equity, 98_500.0)
        self.assertAlmostEqual(state.balance, 100_000.0)

    def test_equity_and_balance_stay_distinct(self):
        """
        Collapsing them silently disables intraday protection: an open loser
        would stop moving the daily number.
        """
        b, fake = broker(equity=97_000.0, balance=100_000.0)
        state = b.to_account_state(100_000.0)
        self.assertNotEqual(state.equity, state.balance)

    def test_governor_accepts_the_state(self):
        from live_governor import Decision, LiveGovernor
        b, fake = broker(equity=95_800.0, balance=100_000.0)
        g = LiveGovernor()
        g.seed_anchor(g.trading_date(b.to_account_state(100_000.0).timestamp),
                      100_000.0)
        v = g.observe(b.to_account_state(100_000.0))
        self.assertIs(v.decision, Decision.HALT_NEW)

    def test_missing_account_info_raises(self):
        b, fake = broker(account_ok=False)
        with self.assertRaises(MT5Error):
            b.to_account_state(100_000.0)


class TestFlattenAll(unittest.TestCase):

    def test_closes_each_position_in_the_opposite_direction(self):
        b, fake = broker()
        fake.positions = [
            Obj(symbol='EURUSD.raw', volume=1.0, type=0, price_open=1.1,
                price_current=1.11, profit=100.0),
            Obj(symbol='GBPUSD.raw', volume=0.5, type=1, price_open=1.27,
                price_current=1.26, profit=50.0),
        ]
        orders = b.flatten_all()
        self.assertEqual(len(orders), 2)
        self.assertEqual(fake.sent[0]['type'], M.ORDER_TYPE_SELL)
        self.assertEqual(fake.sent[1]['type'], M.ORDER_TYPE_BUY)

    def test_nothing_open_sends_nothing(self):
        b, fake = broker()
        self.assertEqual(b.flatten_all(), [])
        self.assertEqual(fake.sent, [])


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('\n' + '=' * 68)
    print(f"  ran {result.testsRun} | failures {len(result.failures)} | "
          f"errors {len(result.errors)} | skipped {len(result.skipped)}")
    print('=' * 68)
    print("  NOTE: fake terminal only. Real-terminal assumptions are checked")
    print("  by mt5_adapter.selftest_against_terminal() on Windows.")
    if result.skipped:
        for case, reason in result.skipped:
            print(f"    SKIPPED {case}: {reason}")
    return 0 if not (result.failures or result.errors or result.skipped) else 1


if __name__ == '__main__':
    sys.exit(main())