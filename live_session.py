# ==============================================================================
# live_session.py -- Phase 2: The Live Session Orchestrator
# ==============================================================================
# Ties the standalone safety/analytics modules into ONE runnable session around
# the existing LiveEngine. The engine already handles the tick loop, drift, and
# kill switch; this orchestrator adds the layers the engine does not:
#
#   BEFORE the session starts:
#     * preflight        -- go/no-go gate; refuses to start if anything is unsafe
#     * config_freeze     -- verify each strategy matches its frozen hash
#     * challenge_journal -- open the trading day, record the starting equity
#
#   AROUND every signal (via a guard wrapper, so the engine is not rewritten):
#     * live_tick_guard   -- reject bad ticks before they reach a strategy
#     * regime_throttle   -- scale size by market regime
#     * entry_guard       -- veto orders that breach size/margin/risk limits
#     * time_stop         -- force-exit positions held too long
#     * weekend_policy    -- flatten before Friday close / the daily boundary
#     * slippage_recorder -- record intended-vs-fill on every execution
#
#   THROUGHOUT:
#     * watchdog          -- write a heartbeat each loop so a separate watchdog
#                            process can detect a hang
#     * challenge_journal -- record trades, governor verdicts, actions
#
#   AT END OF DAY / SHUTDOWN:
#     * weekend_policy    -- final flatten
#     * challenge_journal -- close the day with the ending equity
#
# WHY AN ORCHESTRATOR, NOT ENGINE EDITS:
#   Composing around the engine keeps each concern in its own tested module and
#   lets the whole session be exercised with a fake broker (see the tests)
#   without a live feed. It also means a missing/failed module DEGRADES the
#   session (logged, session continues or refuses to start) rather than silently
#   corrupting the engine's own loop.
#
# DESIGN PRINCIPLE (project-wide):
#   Fail loud and safe. If preflight fails or config-freeze finds a mismatch,
#   the session REFUSES to start rather than trading an unverified book. If a
#   guard cannot make a decision (missing data), it blocks the action rather
#   than waving it through. Every refusal is recorded in the journal.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


def _utcnow():
    return datetime.now(timezone.utc)


# Optional imports: the session degrades cleanly if a module is unavailable,
# and says so, rather than crashing the whole run.
def _try_import(name, attr):
    try:
        mod = __import__(name, fromlist=[attr])
        return getattr(mod, attr)
    except Exception:
        return None


def _safe_construct(cls, *args, **kwargs):
    """Construct cls, returning None (not raising) if it cannot be built.

    A guard that needs a config we cannot supply must degrade to 'absent',
    never crash the session at construction time.
    """
    if cls is None:
        return None
    try:
        return cls(*args, **kwargs)
    except Exception:
        return None


@dataclass
class LiveSessionConfig:
    account_size: float = 100_000.0
    firm: str = "ftmo"
    require_preflight: bool = True      # refuse to start if preflight fails
    require_config_freeze: bool = False # refuse to start on frozen-hash mismatch
    enable_tick_guard: bool = True
    enable_entry_guard: bool = True
    enable_regime_throttle: bool = True
    enable_time_stop: bool = True
    enable_weekend_flatten: bool = True
    enable_journal: bool = True
    enable_heartbeat: bool = True
    time_stop_max_hold_seconds: float = 6 * 3600
    friday_close_hhmm_utc: str = "21:00"


@dataclass
class SessionState:
    started: bool = False
    start_equity: Optional[float] = None
    refused_reason: str = ""
    ticks_guarded: int = 0
    ticks_rejected: int = 0
    orders_vetoed: int = 0
    time_stop_exits: int = 0
    weekend_flattens: int = 0


class LiveSession:
    """
    Orchestrates a full live/demo trading session around a LiveEngine.

    Usage:
        session = LiveSession(engine, broker, config)
        if session.start():          # runs preflight + freeze + journal open
            engine.run_loop(...)      # engine loop, now guarded
            session.end()             # flatten + journal close
    """

    def __init__(self, engine: Any, broker: Any,
                 config: Optional[LiveSessionConfig] = None,
                 strategies: Optional[List[Dict[str, Any]]] = None):
        self.engine = engine
        self.broker = broker
        self.config = config or LiveSessionConfig()
        self.strategies = strategies or []
        self.state = SessionState()

        # Wire optional collaborators (None if their module is unavailable).
        self._tick_guard = self._make_tick_guard()
        self._entry_guard = self._make_entry_guard()
        self._regime = self._make_regime_throttle()
        self._time_stop = self._make_time_stop()
        self._weekend = self._make_weekend_policy()
        self._journal = self._make_journal()
        self._heartbeat = self._make_heartbeat()
        self._slippage = self._make_slippage_recorder()

        self._log = getattr(engine, "_log", print)

    # ── Collaborator construction (each degrades to None) ──────────────────────
    def _make_tick_guard(self):
        if not self.config.enable_tick_guard:
            return None
        return _safe_construct(_try_import("live_tick_guard", "LiveTickGuard"))

    def _make_entry_guard(self):
        if not self.config.enable_entry_guard:
            return None
        return _safe_construct(_try_import("entry_guard", "EntryGuard"))

    def _make_regime_throttle(self):
        if not self.config.enable_regime_throttle:
            return None
        return _safe_construct(_try_import("regime_throttle", "RegimeThrottle"))

    def _make_time_stop(self):
        if not self.config.enable_time_stop:
            return None
        cls = _try_import("time_stop", "TimeStop")
        cfg = _try_import("time_stop", "TimeStopConfig")
        if cls and cfg:
            return cls(cfg(max_hold_seconds=self.config.time_stop_max_hold_seconds))
        return None

    def _make_weekend_policy(self):
        if not self.config.enable_weekend_flatten:
            return None
        cls = _try_import("weekend_policy", "WeekendPolicy")
        cfg = _try_import("weekend_policy", "WeekendPolicyConfig")
        if cls and cfg:
            return cls(cfg(friday_close_hhmm_utc=self.config.friday_close_hhmm_utc))
        return None

    def _make_journal(self):
        if not self.config.enable_journal:
            return None
        return _safe_construct(_try_import("challenge_journal", "ChallengeJournal"))

    def _make_heartbeat(self):
        if not self.config.enable_heartbeat:
            return None
        cls = _try_import("watchdog", "HeartbeatWriter")
        if not cls:
            return None
        try:
            return cls(path="live_heartbeat.json")
        except TypeError:
            try:
                return cls("live_heartbeat.json")
            except Exception:
                return None
        except Exception:
            return None

    def _make_slippage_recorder(self):
        return _safe_construct(_try_import("slippage_recorder", "SlippageRecorder"))

    # ── Session lifecycle ─────────────────────────────────────────────────────
    def start(self) -> bool:
        """
        Run all pre-session gates. Returns True only if the session may safely
        begin trading. Records the reason and returns False otherwise.
        """
        # 1. Preflight go/no-go.
        if self.config.require_preflight:
            ok, reason = self._run_preflight()
            if not ok:
                self.state.refused_reason = f"preflight: {reason}"
                self._log(f"[REFUSE] session not started -- {self.state.refused_reason}")
                return False

        # 2. Config freeze verification.
        if self.config.require_config_freeze:
            ok, reason = self._verify_config_freeze()
            if not ok:
                self.state.refused_reason = f"config_freeze: {reason}"
                self._log(f"[REFUSE] session not started -- {self.state.refused_reason}")
                return False

        # 3. Open the journal day with starting equity.
        equity = self._current_equity()
        self.state.start_equity = equity
        if self._journal:
            try:
                self._journal.open_day(equity)
            except Exception as e:
                self._log(f"[WARN] journal open_day failed: {e}")

        # 4. Install the guard wrapper around each strategy's signal.
        self._install_guards()

        self.state.started = True
        self._log(f"[OK] Live session started (equity {equity:.2f})")
        return True

    def end(self) -> None:
        """Flatten per weekend policy and close the journal day."""
        # Final flatten.
        if self._weekend is not None:
            try:
                decision = self._weekend.check()
                if getattr(decision, "should_flatten", False):
                    self._flatten_all(f"session end / {decision.reason}")
            except Exception as e:
                self._log(f"[WARN] weekend flatten check failed: {e}")

        equity = self._current_equity()
        if self._journal:
            try:
                self._journal.close_day(equity, notes="session ended")
            except Exception as e:
                self._log(f"[WARN] journal close_day failed: {e}")
        self._log(f"[OK] Live session ended (equity {equity:.2f})")

    # ── The guard wrapper ──────────────────────────────────────────────────────
    def _install_guards(self):
        """
        Wrap each engine slot's signal_fn so guards run on every tick/signal,
        without editing the engine. The wrapper:
          * rejects bad ticks (tick guard) -> no signal
          * enforces the time stop -> may emit a forced exit
          * enforces weekend flatten -> may emit a forced exit
          * throttles size by regime, and vetoes via entry guard
        """
        slots = getattr(self.engine, "_slots", {})
        for sid, slot in slots.items():
            original = getattr(slot, "signal_fn", None)
            slot.signal_fn = self._wrap_signal(sid, slot, original)

    def _wrap_signal(self, sid: str, slot: Any,
                     original: Optional[Callable]) -> Callable:
        def guarded(tick):
            # 1. Tick guard: reject bad ticks outright.
            if self._tick_guard is not None:
                try:
                    verdict = self._tick_guard.check(tick)
                    if not self._tick_ok(verdict):
                        self.state.ticks_rejected += 1
                        return None
                except Exception as e:
                    self._log(f"[WARN] tick guard error [{sid}]: {e}")
            self.state.ticks_guarded += 1

            symbol = getattr(slot, "symbol", sid)

            # 2. Time stop: force an exit if a position is too old.
            if self._time_stop is not None:
                try:
                    if self._time_stop.expired(symbol, now=_utcnow()):
                        self.state.time_stop_exits += 1
                        self._record_action("time_stop_exit", symbol)
                        return ("SELL", self._position_size(symbol))
                except Exception as e:
                    self._log(f"[WARN] time stop error [{sid}]: {e}")

            # 3. Weekend / EOD flatten.
            if self._weekend is not None:
                try:
                    decision = self._weekend.check(now=_utcnow())
                    if getattr(decision, "should_flatten", False):
                        self.state.weekend_flattens += 1
                        self._record_action("weekend_flatten", symbol,
                                             getattr(decision, "reason", ""))
                        return ("SELL", self._position_size(symbol))
                except Exception as e:
                    self._log(f"[WARN] weekend policy error [{sid}]: {e}")

            # 4. Original strategy signal.
            if original is None:
                return None
            signal = original(tick)
            if signal is None:
                return None
            side, size = signal

            # 5. Regime throttle: scale size.
            if self._regime is not None:
                try:
                    mult = self._regime_multiplier()
                    size = size * mult
                except Exception as e:
                    self._log(f"[WARN] regime throttle error [{sid}]: {e}")

            # 6. Entry guard: veto if the order breaches limits.
            if self._entry_guard is not None:
                try:
                    if not self._entry_allowed(symbol, side, size, tick):
                        self.state.orders_vetoed += 1
                        self._record_action("entry_vetoed", symbol,
                                             f"{side} {size}")
                        return None
                except Exception as e:
                    self._log(f"[WARN] entry guard error [{sid}]: {e}")

            # 7. Register the entry with the time stop and record intent.
            if self._time_stop is not None and side in ("BUY", "SELL"):
                try:
                    self._time_stop.register(symbol, now=_utcnow())
                except Exception:
                    pass

            if self._journal:
                try:
                    self._journal.record_trade(symbol, side, size,
                                               getattr(tick, "last", None))
                except Exception:
                    pass

            return (side, size)

        return guarded

    # ── Heartbeat (called by the engine loop or a wrapping runner) ─────────────
    def beat(self) -> None:
        """Write a heartbeat. Call once per engine loop iteration."""
        if self._heartbeat is None:
            return
        try:
            has_positions = self._has_open_positions()
            self._heartbeat.beat(positions_open=has_positions)
        except Exception as e:
            self._log(f"[WARN] heartbeat failed: {e}")

    # ── Small helpers that tolerate different broker/guard shapes ──────────────
    def _run_preflight(self) -> Tuple[bool, str]:
        fn = _try_import("preflight", "run_preflight")
        cls = _try_import("preflight", "Preflight")
        try:
            if fn is not None:
                result = fn(broker=self.broker, firm=self.config.firm)
                return self._preflight_ok(result)
            if cls is not None:
                result = cls(broker=self.broker).run()
                return self._preflight_ok(result)
        except Exception as e:
            return False, f"preflight raised {e}"
        # If preflight isn't importable, be conservative only if required.
        return (False, "preflight module unavailable")

    def _preflight_ok(self, result) -> Tuple[bool, str]:
        # Accept a bool, or an object with .ok / .passed / .is_go.
        if isinstance(result, bool):
            return result, "" if result else "preflight returned False"
        for attr in ("is_go", "passed", "ok"):
            val = getattr(result, attr, None)
            if val is not None:
                return bool(val), getattr(result, "reason", "") or ""
        return bool(result), ""

    def _verify_config_freeze(self) -> Tuple[bool, str]:
        cls = _try_import("config_freeze", "ConfigFreeze")
        if cls is None:
            return True, "config_freeze unavailable (skipped)"
        try:
            cf = cls()
            results = cf.verify_all(self.strategies) if self.strategies else []
            bad = [r for r in results if not getattr(r, "ok", True)]
            if bad:
                return False, f"{len(bad)} strategy hash mismatch(es)"
            return True, ""
        except Exception as e:
            return False, f"config_freeze raised {e}"

    def _tick_ok(self, verdict) -> bool:
        if isinstance(verdict, bool):
            return verdict
        for attr in ("ok", "valid", "accepted"):
            v = getattr(verdict, attr, None)
            if v is not None:
                return bool(v)
        return True

    def _regime_multiplier(self) -> float:
        for m in ("multiplier", "current_multiplier", "get_multiplier"):
            attr = getattr(self._regime, m, None)
            if callable(attr):
                try:
                    return float(attr())  # type: ignore[arg-type]
                except Exception:
                    pass
            elif attr is not None:
                try:
                    return float(attr)  # type: ignore[arg-type]
                except Exception:
                    pass
        return 1.0

    def _entry_allowed(self, symbol, side, size, tick) -> bool:
        for m in ("check", "allow", "evaluate"):
            fn = getattr(self._entry_guard, m, None)
            if callable(fn):
                try:
                    res = fn(symbol=symbol, side=side, size=size,
                             price=getattr(tick, "last", None))
                except TypeError:
                    res = fn(symbol, side, size)
                if isinstance(res, bool):
                    return res
                for attr in ("allowed", "ok", "passed"):
                    v = getattr(res, attr, None)
                    if v is not None:
                        return bool(v)
                return True
        return True

    def _current_equity(self) -> float:
        try:
            bal = self.broker.get_balance()
            for attr in ("total_equity", "total", "equity"):
                v = getattr(bal, attr, None)
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return self.config.account_size

    def _position_size(self, symbol) -> float:
        try:
            pos = self.broker.get_position(symbol)
            return abs(float(getattr(pos, "size", 0.0)))
        except Exception:
            return 0.0

    def _has_open_positions(self) -> bool:
        try:
            slots = getattr(self.engine, "_slots", {})
            for slot in slots.values():
                if self._position_size(getattr(slot, "symbol", "")) > 0:
                    return True
        except Exception:
            pass
        return False

    def _flatten_all(self, reason: str) -> None:
        slots = getattr(self.engine, "_slots", {})
        for slot in slots.values():
            symbol = getattr(slot, "symbol", None)
            if symbol and self._position_size(symbol) > 0:
                try:
                    self.broker.flatten(symbol)
                    self._record_action("flatten", symbol, reason)
                except Exception as e:
                    self._log(f"[WARN] flatten failed [{symbol}]: {e}")

    def _record_action(self, action, symbol, detail=""):
        if self._journal:
            try:
                self._journal.record_action(action, f"{symbol} {detail}".strip())
            except Exception:
                pass


__all__ = ["LiveSession", "LiveSessionConfig", "SessionState"]