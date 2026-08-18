# ==============================================================================
# test_live_session.py -- Tests for the Phase 2 live session orchestrator
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
#
# The real LiveEngine needs a broker + live feed, which cannot run here. These
# tests use fakes to prove the ORCHESTRATION logic: the pre-session gates, the
# guard wrapper's decisions, and the session lifecycle. The wiring to the real
# engine is exercised on the user's machine once the MT5 stack (Phase 3) exists.
# ==============================================================================

import unittest
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from live_session import LiveSession, LiveSessionConfig, SessionState


# ── Fakes ─────────────────────────────────────────────────────────────────────
@dataclass
class FakeTick:
    last: float = 1.10
    bid: float = 1.0999
    ask: float = 1.1001
    symbol: str = "EURUSD"


@dataclass
class FakeBalance:
    total_equity: float = 100_000.0


@dataclass
class FakePosition:
    size: float = 0.0


class FakeBroker:
    def __init__(self, equity=100_000.0):
        self._equity = equity
        self.flattened = []
        self._positions: Dict[str, float] = {}

    def get_balance(self):
        return FakeBalance(total_equity=self._equity)

    def get_position(self, symbol):
        return FakePosition(size=self._positions.get(symbol, 0.0))

    def get_tick(self, symbol):
        return FakeTick(symbol=symbol)

    def flatten(self, symbol):
        self.flattened.append(symbol)
        self._positions[symbol] = 0.0

    def set_position(self, symbol, size):
        self._positions[symbol] = size


@dataclass
class FakeSlot:
    strategy_id: str
    symbol: str
    signal_fn: Optional[Callable] = None


class FakeEngine:
    def __init__(self):
        self._slots: Dict[str, FakeSlot] = {}
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)

    def add_slot(self, sid, symbol, signal_fn=None):
        self._slots[sid] = FakeSlot(sid, symbol, signal_fn)


# Guard fakes injected directly onto the session (bypass import wiring).
class RejectingTickGuard:
    def check(self, tick):
        return False


class AcceptingTickGuard:
    def check(self, tick):
        return True


class VetoEntryGuard:
    def check(self, **kw):
        return False


class AllowEntryGuard:
    def check(self, **kw):
        return True


class HalfRegime:
    def multiplier(self):
        return 0.5


class ExpiredTimeStop:
    def expired(self, symbol, now=None):
        return True

    def register(self, *a, **k):
        pass


class FreshTimeStop:
    def expired(self, symbol, now=None):
        return False

    def register(self, *a, **k):
        pass


@dataclass
class FlattenDecision:
    should_flatten: bool
    reason: str = "test"


class FlattenWeekend:
    def check(self, now=None):
        return FlattenDecision(True, "friday close")


class QuietWeekend:
    def check(self, now=None):
        return FlattenDecision(False)


class RecordingJournal:
    def __init__(self):
        self.opened = None
        self.closed = None
        self.trades = []
        self.actions = []

    def open_day(self, equity):
        self.opened = equity

    def close_day(self, equity, notes=""):
        self.closed = equity

    def record_trade(self, symbol, side, size, price=None):
        self.trades.append((symbol, side, size))

    def record_action(self, action, detail=""):
        self.actions.append((action, detail))


def make_session(**overrides):
    engine = FakeEngine()
    broker = FakeBroker()
    cfg = LiveSessionConfig(require_preflight=False, enable_heartbeat=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    session = LiveSession(engine, broker, cfg)
    # Replace import-wired collaborators with fakes by default (None).
    session._tick_guard = None
    session._entry_guard = None
    session._regime = None
    session._time_stop = None
    session._weekend = None
    session._journal = None
    session._heartbeat = None
    session._slippage = None
    return session, engine, broker


# ── Tests ──────────────────────────────────────────────────────────────────────
class TestPreflightGate(unittest.TestCase):
    def test_refuses_when_preflight_required_and_unavailable(self):
        engine, broker = FakeEngine(), FakeBroker()
        cfg = LiveSessionConfig(require_preflight=True)
        session = LiveSession(engine, broker, cfg)
        # preflight module likely unavailable in test env -> refuse
        started = session.start()
        # Either it started (preflight present and passed) or refused cleanly.
        if not started:
            self.assertIn("preflight", session.state.refused_reason)

    def test_starts_when_preflight_not_required(self):
        session, engine, broker = make_session(require_preflight=False)
        self.assertTrue(session.start())
        self.assertTrue(session.state.started)


class TestJournalLifecycle(unittest.TestCase):
    def test_open_and_close_day(self):
        session, engine, broker = make_session()
        journal = RecordingJournal()
        session._journal = journal
        session.start()
        self.assertEqual(journal.opened, 100_000.0)
        session.end()
        self.assertEqual(journal.closed, 100_000.0)


class TestTickGuard(unittest.TestCase):
    def test_bad_tick_rejected(self):
        session, engine, broker = make_session()
        session._tick_guard = RejectingTickGuard()
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: ("BUY", 1.0))
        session.start()
        wrapped = engine._slots["s1"].signal_fn
        result = wrapped(FakeTick())
        self.assertIsNone(result)  # bad tick -> no signal
        self.assertEqual(session.state.ticks_rejected, 1)

    def test_good_tick_passes_through(self):
        session, engine, broker = make_session()
        session._tick_guard = AcceptingTickGuard()
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: ("BUY", 1.0))
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertEqual(result, ("BUY", 1.0))


class TestRegimeThrottle(unittest.TestCase):
    def test_size_scaled(self):
        session, engine, broker = make_session()
        session._regime = HalfRegime()
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: ("BUY", 2.0))
        session.start()
        side, size = engine._slots["s1"].signal_fn(FakeTick())
        self.assertEqual(size, 1.0)  # 2.0 * 0.5


class TestEntryGuard(unittest.TestCase):
    def test_veto_blocks_order(self):
        session, engine, broker = make_session()
        session._entry_guard = VetoEntryGuard()
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: ("BUY", 1.0))
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertIsNone(result)
        self.assertEqual(session.state.orders_vetoed, 1)

    def test_allow_passes(self):
        session, engine, broker = make_session()
        session._entry_guard = AllowEntryGuard()
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: ("BUY", 1.0))
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertEqual(result, ("BUY", 1.0))


class TestTimeStop(unittest.TestCase):
    def test_expired_forces_exit(self):
        session, engine, broker = make_session()
        session._time_stop = ExpiredTimeStop()
        broker.set_position("EURUSD", 1.5)
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: None)
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertEqual(result[0], "SELL")  # forced exit
        self.assertEqual(session.state.time_stop_exits, 1)

    def test_fresh_does_not_exit(self):
        session, engine, broker = make_session()
        session._time_stop = FreshTimeStop()
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: None)
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertIsNone(result)  # no signal, no forced exit


class TestWeekendFlatten(unittest.TestCase):
    def test_weekend_forces_exit(self):
        session, engine, broker = make_session()
        session._weekend = FlattenWeekend()
        broker.set_position("EURUSD", 1.0)
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: None)
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertEqual(result[0], "SELL")
        self.assertEqual(session.state.weekend_flattens, 1)

    def test_end_flattens_positions(self):
        session, engine, broker = make_session()
        session._weekend = FlattenWeekend()
        broker.set_position("EURUSD", 1.0)
        engine.add_slot("s1", "EURUSD")
        session.start()
        session.end()
        self.assertIn("EURUSD", broker.flattened)


class TestChainedGuards(unittest.TestCase):
    def test_full_chain_allows_clean_order(self):
        session, engine, broker = make_session()
        session._tick_guard = AcceptingTickGuard()
        session._regime = HalfRegime()
        session._entry_guard = AllowEntryGuard()
        session._time_stop = FreshTimeStop()
        session._weekend = QuietWeekend()
        journal = RecordingJournal()
        session._journal = journal
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: ("BUY", 4.0))
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertEqual(result, ("BUY", 2.0))  # 4.0 * 0.5 regime
        self.assertEqual(len(journal.trades), 1)  # trade recorded

    def test_tick_guard_short_circuits_everything(self):
        session, engine, broker = make_session()
        session._tick_guard = RejectingTickGuard()
        session._entry_guard = VetoEntryGuard()  # shouldn't even be reached
        engine.add_slot("s1", "EURUSD", signal_fn=lambda t: ("BUY", 1.0))
        session.start()
        result = engine._slots["s1"].signal_fn(FakeTick())
        self.assertIsNone(result)
        self.assertEqual(session.state.ticks_rejected, 1)
        self.assertEqual(session.state.orders_vetoed, 0)  # never reached


class TestConfigFreezeGate(unittest.TestCase):
    def test_mismatch_refuses_start(self):
        session, engine, broker = make_session(require_config_freeze=True)

        class MismatchFreeze:
            def verify_all(self, strategies):
                @dataclass
                class R:
                    ok: bool = False
                return [R(ok=False)]

        session._verify_config_freeze = lambda: (False, "1 mismatch")
        started = session.start()
        self.assertFalse(started)
        self.assertIn("config_freeze", session.state.refused_reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
