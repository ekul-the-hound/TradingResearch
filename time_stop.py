# ==============================================================================
# time_stop.py -- Maximum Holding-Period Exit ("Time Stop")
# ==============================================================================
# Forces an exit when a position has been held too long, regardless of price.
# Purpose:
#   * Kill stalled trades that are neither hitting their stop nor their target
#     and are just tying up risk budget.
#   * Cap overnight exposure.
#   * Keep strategies INTRADAY -- which makes the FTMO daily-loss anchor exact,
#     because a position that never crosses Prague midnight has no overnight
#     floating-P&L ambiguity in the daily calculation.
#
# WHY IT KEEPS ITS OWN CLOCK:
#   BrokerPosition carries no entry-time or bars-held field. So the time stop
#   cannot ask the broker "how old is this position?" -- it must record when a
#   position opened and measure age itself. register() is called on entry;
#   clear() on exit; check()/expired() answer "should this be closed now?".
#
# TWO MODES (a single guard can use either or both):
#   * BAR mode      -- age measured in bars held. Deterministic; ideal for
#                      backtests where "N bars" is unambiguous.
#   * WALL mode     -- age measured in wall-clock seconds. Ideal for live, where
#                      "no position older than 6h" or "flat by 21:00" matters.
#
# DESIGN PRINCIPLE (project-wide):
#   A position the guard has never seen registered is reported as UNKNOWN-age,
#   and (by default) treated as expired-for-safety rather than silently kept.
#   An untracked position is a reconciliation gap, and the safe response to "I
#   don't know how long this has been open" is to flag it for exit, not ignore it.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TimeStopConfig:
    # BAR mode: exit when bars_held >= max_bars. 0 disables bar-mode.
    max_bars: int = 0
    # WALL mode: exit when age >= max_hold_seconds. 0 disables wall-mode.
    max_hold_seconds: float = 0.0
    # Optional hard daily cutoff in "HH:MM" (position's local exchange time as
    # provided by the caller). Any position still open at/after this is expired.
    # Empty disables. This is the "flat by X" intraday enforcement.
    daily_cutoff_hhmm: str = ""
    # How to treat a position with no registered entry (unknown age):
    #   True  -> expired (safe: flag unknown positions for exit)
    #   False -> not expired (permissive)
    expire_unknown: bool = True


@dataclass
class _Entry:
    symbol: str
    entry_time: datetime
    entry_bar: int


@dataclass
class TimeStopVerdict:
    symbol: str
    expired: bool
    reason: str = ""
    age_seconds: Optional[float] = None
    bars_held: Optional[int] = None

    def __bool__(self) -> bool:
        return self.expired


class TimeStop:
    """
    Tracks open-position ages and reports which should be closed on time.

    Usage:
        ts = TimeStop(TimeStopConfig(max_hold_seconds=6*3600))
        ts.register("EURUSD")                 # on entry
        ...
        for v in ts.check_all(open_symbols, now=..., bar_index=...):
            if v.expired: close(v.symbol)     # engine closes, then:
        ts.clear("EURUSD")                    # on exit
    """

    def __init__(self, config: Optional[TimeStopConfig] = None):
        self.config = config or TimeStopConfig()
        self._entries: Dict[str, _Entry] = {}

    # -- Registry --------------------------------------------------------------
    def register(self, symbol: str, now: Optional[datetime] = None,
                 bar_index: int = 0) -> None:
        """Record a position's entry. Call on fill. Re-registering resets age."""
        self._entries[symbol] = _Entry(
            symbol=symbol, entry_time=now or _utcnow(), entry_bar=bar_index)

    def clear(self, symbol: str) -> None:
        """Forget a position. Call on exit."""
        self._entries.pop(symbol, None)

    def is_registered(self, symbol: str) -> bool:
        return symbol in self._entries

    def registered_symbols(self) -> List[str]:
        return list(self._entries.keys())

    # -- Age queries -----------------------------------------------------------
    def age_seconds(self, symbol: str, now: Optional[datetime] = None) -> Optional[float]:
        e = self._entries.get(symbol)
        if e is None:
            return None
        return ((now or _utcnow()) - e.entry_time).total_seconds()

    def bars_held(self, symbol: str, bar_index: int) -> Optional[int]:
        e = self._entries.get(symbol)
        if e is None:
            return None
        return bar_index - e.entry_bar

    # -- Core check ------------------------------------------------------------
    def check(self, symbol: str, now: Optional[datetime] = None,
              bar_index: Optional[int] = None) -> TimeStopVerdict:
        """
        Decide whether one position has exceeded its holding limit.

        `now` drives WALL mode and the daily cutoff; `bar_index` drives BAR mode.
        Whichever limits are configured and have the data they need are checked;
        the first limit hit wins.
        """
        cfg = self.config
        now = now or _utcnow()
        e = self._entries.get(symbol)

        # Unknown position: we have no age. Safe default is to flag it.
        if e is None:
            if cfg.expire_unknown:
                return TimeStopVerdict(
                    symbol, True,
                    "position not registered with time-stop (unknown age); "
                    "flagged for exit as a safety default")
            return TimeStopVerdict(symbol, False, "unregistered; not expired "
                                                  "(expire_unknown=False)")

        age = (now - e.entry_time).total_seconds()
        bars = (bar_index - e.entry_bar) if bar_index is not None else None

        # WALL mode.
        if cfg.max_hold_seconds > 0 and age >= cfg.max_hold_seconds:
            return TimeStopVerdict(
                symbol, True,
                f"held {age:.0f}s >= max {cfg.max_hold_seconds:.0f}s",
                age_seconds=age, bars_held=bars)

        # BAR mode.
        if cfg.max_bars > 0 and bars is not None and bars >= cfg.max_bars:
            return TimeStopVerdict(
                symbol, True,
                f"held {bars} bars >= max {cfg.max_bars}",
                age_seconds=age, bars_held=bars)

        # Daily cutoff.
        if cfg.daily_cutoff_hhmm:
            cutoff = _parse_hhmm(cfg.daily_cutoff_hhmm)
            if cutoff is not None:
                cutoff_minutes = cutoff[0] * 60 + cutoff[1]
                now_minutes = now.hour * 60 + now.minute
                if now_minutes >= cutoff_minutes:
                    return TimeStopVerdict(
                        symbol, True,
                        f"past daily cutoff {cfg.daily_cutoff_hhmm} "
                        f"(now {now.hour:02d}:{now.minute:02d})",
                        age_seconds=age, bars_held=bars)

        return TimeStopVerdict(symbol, False, "within holding limits",
                               age_seconds=age, bars_held=bars)

    def check_all(self, symbols: List[str], now: Optional[datetime] = None,
                  bar_index: Optional[int] = None) -> List[TimeStopVerdict]:
        """Check a list of currently-open symbols; return only the expired ones."""
        now = now or _utcnow()
        out: List[TimeStopVerdict] = []
        for sym in symbols:
            v = self.check(sym, now=now, bar_index=bar_index)
            if v.expired:
                out.append(v)
        return out

    def expired(self, symbol: str, now: Optional[datetime] = None,
                bar_index: Optional[int] = None) -> bool:
        """Boolean convenience wrapper around check()."""
        return self.check(symbol, now=now, bar_index=bar_index).expired


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_hhmm(s: str) -> Optional[tuple]:
    try:
        parts = s.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return (h, m)
    except (ValueError, IndexError):
        pass
    return None


__all__ = ["TimeStop", "TimeStopConfig", "TimeStopVerdict"]


if __name__ == "__main__":
    from datetime import timedelta
    ts = TimeStop(TimeStopConfig(max_hold_seconds=3600, max_bars=10))
    t0 = _utcnow()
    ts.register("EURUSD", now=t0, bar_index=0)
    print("fresh:", ts.check("EURUSD", now=t0, bar_index=0).reason)
    print("2h later:", ts.check("EURUSD", now=t0 + timedelta(hours=2),
                                 bar_index=3).reason)
    print("12 bars later:", ts.check("EURUSD", now=t0 + timedelta(minutes=5),
                                      bar_index=12).reason)
    print("unknown symbol:", ts.check("GBPUSD", now=t0).reason)
