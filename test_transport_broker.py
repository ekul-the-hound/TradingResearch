# ==============================================================================
# test_transport_broker.py -- Tests for the transport layer + broker adapter
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# Everything here runs against FakeTransport / a temp-dir FileIPCTransport --
# no MetaTrader5 package or terminal required. This proves the ADAPTER LOGIC;
# the real transport still needs demo-account validation.
# ==============================================================================

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from broker_base import OrderStatus
from mt5_transport import (
    FakeTransport, FileIPCTransport, FileIPCConfig,
    TransportTick, TransportPosition, TransportAccount, TransportOrder,
    TransportOrderResult, TransportStale, TransportNotConnected,
)
from transport_broker import TransportBroker, ReconcileReport


def primed_fake():
    ft = FakeTransport()
    ft.connect()
    ft.ticks["EURUSD"] = TransportTick("EURUSD", 1.09998, 1.10002)
    ft.ticks["GBPUSD"] = TransportTick("GBPUSD", 1.25000, 1.25004)
    return ft


# ==============================================================================
# TRANSPORT-LEVEL TESTS
# ==============================================================================
class TestFakeTransport(unittest.TestCase):
    def test_requires_connect(self):
        ft = FakeTransport()
        with self.assertRaises(TransportNotConnected):
            ft.get_ticks(["EURUSD"])

    def test_order_fills_and_creates_position(self):
        ft = primed_fake()
        r = ft.place_order(TransportOrder("EURUSD", "buy", 0.10, sl=1.0980))
        self.assertTrue(r.ok)
        self.assertEqual(r.fill_price, 1.10002)  # buy fills at ask
        self.assertEqual(len(ft.get_positions()), 1)

    def test_programmable_result(self):
        ft = primed_fake()
        ft.next_result = TransportOrderResult(ok=False, comment="rejected by test")
        r = ft.place_order(TransportOrder("EURUSD", "buy", 0.1))
        self.assertFalse(r.ok)


class TestFileIPCTransport(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.t = FileIPCTransport(FileIPCConfig(directory=self.d,
                                                max_state_age_seconds=5.0))
        self.t.connect()

    def _write_state(self, age_seconds=0, **over):
        ts = (datetime.now(timezone.utc)
              - timedelta(seconds=age_seconds)).isoformat()
        state = {
            "seq": 1, "timestamp": ts,
            "account": {"balance": 100_000, "equity": 100_500,
                        "margin_free": 99_000, "margin_used": 1_000},
            "ticks": {"EURUSD": {"bid": 1.09998, "ask": 1.10002}},
            "positions": [{"ticket": 7, "symbol": "EURUSD", "type": "buy",
                           "volume": 0.10, "price_open": 1.10, "sl": 1.098}],
        }
        state.update(over)
        Path(self.d, "state.json").write_text(json.dumps(state))

    def test_fresh_state_reads(self):
        self._write_state(age_seconds=0)
        self.assertTrue(self.t.is_alive())
        ticks = self.t.get_ticks(["EURUSD"])
        self.assertIn("EURUSD", ticks)
        self.assertEqual(self.t.get_account().equity, 100_500)
        self.assertEqual(len(self.t.get_positions()), 1)

    def test_stale_state_raises(self):
        self._write_state(age_seconds=60)
        self.assertFalse(self.t.is_alive())
        with self.assertRaises(TransportStale):
            self.t.get_ticks(["EURUSD"])

    def test_missing_state_raises(self):
        with self.assertRaises(TransportStale):
            self.t.get_positions()

    def test_order_timeout_reports_unknown(self):
        t = FileIPCTransport(FileIPCConfig(directory=self.d,
                                           result_wait_seconds=0.2,
                                           result_poll_seconds=0.05))
        t.connect()
        r = t.place_order(TransportOrder("EURUSD", "buy", 0.1))
        self.assertFalse(r.ok)
        self.assertIn("UNKNOWN", r.comment)

    def test_order_result_matched_by_id(self):
        t = FileIPCTransport(FileIPCConfig(directory=self.d,
                                           result_wait_seconds=1.0,
                                           result_poll_seconds=0.02))
        t.connect()
        # Simulate the EA: place order, then write a matching result line.
        import threading
        import time

        def ea_responder():
            time.sleep(0.1)
            cmds = Path(self.d, "commands.jsonl").read_text().splitlines()
            last = json.loads(cmds[-1])
            result = {"id": last["id"], "ok": True, "ticket": 55,
                      "retcode": 10009, "fill_price": 1.10002,
                      "filled_volume": 0.1}
            with Path(self.d, "results.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")

        th = threading.Thread(target=ea_responder)
        th.start()
        r = t.place_order(TransportOrder("EURUSD", "buy", 0.1, sl=1.098))
        th.join()
        self.assertTrue(r.ok)
        self.assertEqual(r.ticket, 55)


# ==============================================================================
# BROKER ADAPTER TESTS
# ==============================================================================
class TestTransportBrokerData(unittest.TestCase):
    def setUp(self):
        self.ft = primed_fake()
        self.broker = TransportBroker(self.ft, symbols=["EURUSD", "GBPUSD"])
        self.broker.connect()

    def test_get_tick_maps_fields(self):
        t = self.broker.get_tick("EURUSD")
        self.assertIsNotNone(t)
        self.assertEqual(t.bid, 1.09998)
        self.assertEqual(t.ask, 1.10002)

    def test_get_tick_stale_returns_none(self):
        # Use a file transport with stale data to exercise the None path.
        d = tempfile.mkdtemp()
        ft = FileIPCTransport(FileIPCConfig(directory=d, max_state_age_seconds=1))
        ft.connect()
        old = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        Path(d, "state.json").write_text(json.dumps(
            {"timestamp": old, "ticks": {"EURUSD": {"bid": 1.1, "ask": 1.1}},
             "positions": [], "account": {}}))
        b = TransportBroker(ft)
        b.connect()
        self.assertIsNone(b.get_tick("EURUSD"))  # stale -> None, not old quote

    def test_get_balance_maps(self):
        self.ft.account = TransportAccount(balance=100_000, equity=101_000,
                                           margin_free=98_000, margin_used=2_000)
        bal = self.broker.get_balance()
        self.assertEqual(bal.total_equity, 101_000)
        self.assertEqual(bal.free_margin, 98_000)

    def test_positions_side_translation(self):
        self.ft.positions = [
            TransportPosition(1, "EURUSD", "buy", 0.1),
            TransportPosition(2, "GBPUSD", "sell", 0.2),
        ]
        positions = {p.symbol: p for p in self.broker.get_positions()}
        self.assertEqual(positions["EURUSD"].side, "long")
        self.assertEqual(positions["GBPUSD"].side, "short")


class TestServerSideStops(unittest.TestCase):
    def setUp(self):
        self.ft = primed_fake()

    def test_order_with_stop_succeeds(self):
        broker = TransportBroker(self.ft, require_stops=True)
        broker.connect()
        o = broker.submit_order("buy", "EURUSD", 0.1, stop_loss=1.0980)
        self.assertEqual(o.status, OrderStatus.FILLED)
        # The stop was passed to the transport.
        self.assertEqual(self.ft.placed_orders[-1].sl, 1.0980)

    def test_stopless_order_rejected_when_required(self):
        broker = TransportBroker(self.ft, require_stops=True)
        broker.connect()
        o = broker.submit_order("buy", "EURUSD", 0.1)  # no stop
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertIn("stop-loss required", o.reason)
        # Nothing was sent to the transport.
        self.assertEqual(len(self.ft.placed_orders), 0)

    def test_stopless_order_warns_when_not_required(self):
        broker = TransportBroker(self.ft, require_stops=False)
        broker.connect()
        o = broker.submit_order("buy", "EURUSD", 0.1)
        self.assertEqual(o.status, OrderStatus.FILLED)
        self.assertTrue(any("WITHOUT a server-side stop" in n
                            for n in broker.last_notes))

    def test_unknown_result_becomes_rejected(self):
        broker = TransportBroker(self.ft, require_stops=False)
        broker.connect()
        self.ft.next_result = TransportOrderResult(
            ok=False, comment="no result within 10s (order status UNKNOWN)")
        o = broker.submit_order("buy", "EURUSD", 0.1)
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertIn("UNKNOWN", o.reason)


class TestReconciliation(unittest.TestCase):
    def setUp(self):
        self.ft = primed_fake()
        self.broker = TransportBroker(self.ft)
        self.broker.connect()

    def test_clean_when_matched(self):
        self.ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.1)]
        self.broker.snapshot_local()
        report = self.broker.reconcile()
        self.assertTrue(report.clean)
        self.assertIn("EURUSD", report.matched)

    def test_detects_position_only_on_broker(self):
        # Local knows nothing; broker has a position (opened during a crash).
        self.ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.1)]
        # no snapshot_local -> local empty
        report = self.broker.reconcile()
        self.assertFalse(report.clean)
        self.assertIn("EURUSD", report.only_on_broker)

    def test_detects_position_gone_at_broker(self):
        self.ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.1)]
        self.broker.snapshot_local()
        # Position vanished at broker (e.g. hit its server-side stop).
        self.ft.positions = []
        report = self.broker.reconcile()
        self.assertFalse(report.clean)
        self.assertIn("EURUSD", report.only_local)

    def test_detects_size_mismatch(self):
        self.ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.1)]
        self.broker.snapshot_local()
        self.ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.3)]
        report = self.broker.reconcile()
        self.assertFalse(report.clean)
        self.assertIn("EURUSD", report.size_mismatch)

    def test_reconcile_sets_new_baseline(self):
        self.ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.1)]
        self.broker.reconcile()  # broker truth becomes baseline
        report2 = self.broker.reconcile()
        self.assertTrue(report2.clean)  # now matches

    def test_summary_readable(self):
        self.ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.1)]
        report = self.broker.reconcile()
        self.assertIn("UNTRACKED", report.summary())


class TestFlattenAll(unittest.TestCase):
    def test_flatten_sends_opposing_orders(self):
        ft = primed_fake()
        ft.positions = [
            TransportPosition(1, "EURUSD", "buy", 0.1),
            TransportPosition(2, "GBPUSD", "sell", 0.2),
        ]
        broker = TransportBroker(ft, require_stops=True)
        broker.connect()
        orders = broker.flatten_all()
        self.assertEqual(len(orders), 2)
        # Closing a long sends a sell; closing a short sends a buy.
        sides = {o.symbol: o.side.value for o in orders}
        self.assertEqual(sides["EURUSD"], "sell")
        self.assertEqual(sides["GBPUSD"], "buy")

    def test_flatten_bypasses_stop_requirement(self):
        # Closing orders carry no stop; require_stops must not block them.
        ft = primed_fake()
        ft.positions = [TransportPosition(1, "EURUSD", "buy", 0.1)]
        broker = TransportBroker(ft, require_stops=True)
        broker.connect()
        orders = broker.flatten_all()
        self.assertEqual(orders[0].status, OrderStatus.FILLED)
        # require_stops restored afterward.
        self.assertTrue(broker.require_stops)


if __name__ == "__main__":
    unittest.main(verbosity=2)
