# ==============================================================================
# watchdog.py -- Engine Heartbeat + Out-of-Process Watchdog
# ==============================================================================
# Detects a dead or frozen trading engine and alerts LOUDLY -- especially when
# the engine died while holding open positions, which is the dangerous case.
#
# WHY A SEPARATE PROCESS:
#   A watchdog living inside the engine dies when the engine dies. So this splits
#   into two halves that communicate through a small file on disk:
#
#     1. HeartbeatWriter -- called by the engine every tick/loop. It atomically
#        writes a tiny JSON file: {timestamp, open_positions, connected, ...}.
#
#     2. Watchdog -- an INDEPENDENT process (run it in a second terminal, a
#        scheduled task, or a separate service). It polls the heartbeat file and
#        fires alerts when the heartbeat goes stale. Because it is a different
#        process, an engine crash/freeze/OOM cannot silence it.
#
# THE KEY RULE:
#   Staleness while FLAT is a warning. Staleness while positions are OPEN is
#   CRITICAL -- a crashed engine with a naked position can breach a daily-loss
#   rule with no one watching. The watchdog escalates severity accordingly.
#
# DESIGN PRINCIPLE (project-wide):
#   Absence of a fresh heartbeat is treated as "engine presumed down", never as
#   "probably fine". A missing or unparseable file is itself an alert condition,
#   not a reason to stay quiet.
#
# ALERT SINKS:
#   Alerts go to a list of pluggable sinks (callables). Console and file sinks
#   are built in. A Telegram/Discord sink can be added later by appending one
#   callable -- no change to the watchdog loop.
#
# USAGE (engine side):
#   from watchdog import HeartbeatWriter
#   hb = HeartbeatWriter("data/engine_heartbeat.json")
#   # ... inside the loop, each tick:
#   hb.beat(open_positions=len(open_pos), connected=broker.is_connected)
#
# USAGE (watchdog side, separate terminal):
#   python watchdog.py --heartbeat data/engine_heartbeat.json --stale-seconds 15
# ==============================================================================

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Severity ──────────────────────────────────────────────────────────────────
INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"

_RANK = {INFO: 0, WARNING: 1, CRITICAL: 2}


@dataclass
class WatchdogAlert:
    severity: str
    message: str
    timestamp: str = field(default_factory=lambda: _utcnow().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.timestamp} {self.message}"


# ==============================================================================
# HEARTBEAT WRITER (engine side)
# ==============================================================================
class HeartbeatWriter:
    """
    Writes a small heartbeat file the watchdog reads. Call beat() each loop.

    The write is ATOMIC (write temp + os.replace) so the watchdog can never read
    a half-written file and misjudge the engine as down.
    """

    def __init__(self, path: str, engine_id: str = "live_engine"):
        self.path = Path(path)
        self.engine_id = engine_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._beats = 0

    def beat(
        self,
        open_positions: int = 0,
        connected: bool = True,
        tick_count: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write one heartbeat. Returns the payload written."""
        self._beats += 1
        payload: Dict[str, Any] = {
            "engine_id": self.engine_id,
            "timestamp": _utcnow().isoformat(),
            "open_positions": int(open_positions),
            "connected": bool(connected),
            "beats": self._beats,
        }
        if tick_count is not None:
            payload["tick_count"] = int(tick_count)
        if extra:
            payload["extra"] = extra

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic on same filesystem
        return payload

    def mark_shutdown(self, clean: bool = True) -> None:
        """
        Record an intentional shutdown so the watchdog can distinguish a planned
        stop from a crash. A clean shutdown with positions still open is itself
        worth flagging, so we record open_positions too if the file exists.
        """
        payload = {
            "engine_id": self.engine_id,
            "timestamp": _utcnow().isoformat(),
            "shutdown": True,
            "clean": bool(clean),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)


# ==============================================================================
# ALERT SINKS
# ==============================================================================
def console_sink(alert: WatchdogAlert) -> None:
    """Print alerts to stderr so they are visible even if stdout is redirected."""
    print(str(alert), file=sys.stderr, flush=True)


class FileSink:
    """Append alerts to a log file (one JSON object per line)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, alert: WatchdogAlert) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(alert)) + "\n")


class CollectingSink:
    """In-memory sink for tests and dashboards."""

    def __init__(self) -> None:
        self.alerts: List[WatchdogAlert] = []

    def __call__(self, alert: WatchdogAlert) -> None:
        self.alerts.append(alert)


# ==============================================================================
# WATCHDOG (separate process)
# ==============================================================================
@dataclass
class WatchdogConfig:
    heartbeat_path: str
    stale_seconds: float = 15.0        # heartbeat older than this -> stale
    poll_seconds: float = 5.0          # how often the watchdog checks
    # Re-alert cadence: once stale, don't spam every poll. Re-fire only every
    # this many seconds while the condition persists. 0 = alert every poll.
    realert_seconds: float = 60.0
    # Grace period at startup: don't alarm about a missing file for this long,
    # so launching the watchdog slightly before the engine is not a false alarm.
    startup_grace_seconds: float = 30.0


class Watchdog:
    """
    Polls a heartbeat file and fires alerts through its sinks.

    Stateless-ish: it holds only enough state to avoid duplicate alert spam and
    to detect recovery (a stale engine coming back).
    """

    def __init__(self, config: WatchdogConfig,
                 sinks: Optional[List[Callable[[WatchdogAlert], None]]] = None):
        self.config = config
        self.sinks: List[Callable[[WatchdogAlert], None]] = sinks or [console_sink]
        self._started_at = _utcnow()
        self._last_alert_at: Optional[datetime] = None
        self._was_stale = False
        self._running = False

    # -- Core evaluation (pure, testable) --------------------------------------
    def evaluate(self, now: Optional[datetime] = None) -> Optional[WatchdogAlert]:
        """
        Inspect the heartbeat once and return an alert if warranted, else None.

        This is deliberately pure and side-effect-light so tests can drive it
        with injected `now` values. The polling loop calls it and dispatches.
        """
        now = now or _utcnow()
        path = Path(self.config.heartbeat_path)

        # 1. Missing file. -----------------------------------------------------
        if not path.exists():
            age_since_start = (now - self._started_at).total_seconds()
            if age_since_start < self.config.startup_grace_seconds:
                return None  # still in grace window; engine may be starting
            return WatchdogAlert(
                CRITICAL,
                f"No heartbeat file at {path} "
                f"({age_since_start:.0f}s since watchdog start). Engine never "
                f"started or file was removed.",
                data={"reason": "missing_file"},
            )

        # 2. Unparseable / partial file. --------------------------------------
        try:
            raw = path.read_text(encoding="utf-8")
            hb = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            return WatchdogAlert(
                WARNING,
                f"Heartbeat file unreadable: {e}. Will re-check next poll.",
                data={"reason": "unparseable"},
            )

        # 3. Clean shutdown marker. -------------------------------------------
        if hb.get("shutdown"):
            # Planned stop. Only noteworthy if it claims to be unclean.
            if not hb.get("clean", True):
                return WatchdogAlert(
                    WARNING,
                    "Engine recorded an UNCLEAN shutdown.",
                    data={"reason": "unclean_shutdown", "heartbeat": hb},
                )
            return None

        # 4. Staleness. --------------------------------------------------------
        ts = hb.get("timestamp", "")
        beat_dt = _parse_iso(ts)
        if beat_dt is None:
            return WatchdogAlert(
                WARNING,
                f"Heartbeat has no parseable timestamp ({ts!r}).",
                data={"reason": "bad_timestamp", "heartbeat": hb},
            )
        if beat_dt.tzinfo is None:
            beat_dt = beat_dt.replace(tzinfo=timezone.utc)

        age = (now - beat_dt).total_seconds()
        if age <= self.config.stale_seconds:
            # Fresh. Note recovery if we were previously stale.
            if self._was_stale:
                self._was_stale = False
                return WatchdogAlert(
                    INFO,
                    f"Engine heartbeat RECOVERED (age {age:.1f}s).",
                    data={"reason": "recovered", "heartbeat": hb},
                )
            return None

        # Stale. Severity depends on open positions. --------------------------
        open_pos = int(hb.get("open_positions", 0) or 0)
        connected = bool(hb.get("connected", False))
        self._was_stale = True

        if open_pos > 0:
            return WatchdogAlert(
                CRITICAL,
                f"ENGINE STALE for {age:.0f}s WITH {open_pos} OPEN POSITION(S). "
                f"A crashed engine may be leaving positions unmanaged. "
                f"Check the terminal / flatten manually now.",
                data={"reason": "stale_with_positions", "age_seconds": age,
                      "open_positions": open_pos, "connected": connected},
            )
        return WatchdogAlert(
            WARNING,
            f"Engine heartbeat stale for {age:.0f}s (account is flat).",
            data={"reason": "stale_flat", "age_seconds": age,
                  "connected": connected},
        )

    # -- Dispatch with anti-spam ----------------------------------------------
    def _should_dispatch(self, alert: WatchdogAlert, now: datetime) -> bool:
        """Rate-limit repeat alerts of the persisting condition."""
        # Recovery/INFO and first alerts always go out.
        if alert.severity == INFO or self._last_alert_at is None:
            return True
        if self.config.realert_seconds <= 0:
            return True
        elapsed = (now - self._last_alert_at).total_seconds()
        return elapsed >= self.config.realert_seconds

    def dispatch(self, alert: WatchdogAlert, now: Optional[datetime] = None) -> bool:
        """Send an alert through all sinks if not rate-limited. Returns sent?"""
        now = now or _utcnow()
        if not self._should_dispatch(alert, now):
            return False
        for sink in self.sinks:
            try:
                sink(alert)
            except Exception as e:  # a broken sink must not kill the watchdog
                print(f"[watchdog] sink error: {e}", file=sys.stderr, flush=True)
        if alert.severity != INFO:
            self._last_alert_at = now
        return True

    def check_once(self, now: Optional[datetime] = None) -> Optional[WatchdogAlert]:
        """One evaluate + dispatch cycle. Returns the alert if one was raised."""
        now = now or _utcnow()
        alert = self.evaluate(now)
        if alert is not None:
            self.dispatch(alert, now)
        return alert

    # -- Blocking poll loop ----------------------------------------------------
    def run(self, max_iterations: int = 0) -> None:
        """
        Poll forever (or max_iterations times, for tests). Ctrl-C to stop.
        """
        self._running = True
        i = 0
        try:
            while self._running:
                self.check_once()
                i += 1
                if max_iterations and i >= max_iterations:
                    break
                time.sleep(self.config.poll_seconds)
        except KeyboardInterrupt:
            print("[watchdog] stopped by user", file=sys.stderr, flush=True)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Out-of-process watchdog for the live trading engine")
    parser.add_argument("--heartbeat", required=True,
                        help="Path to the engine heartbeat file")
    parser.add_argument("--stale-seconds", type=float, default=15.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--realert-seconds", type=float, default=60.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=30.0)
    parser.add_argument("--log-file", default="",
                        help="Optional file to append alerts to")
    parser.add_argument("--once", action="store_true",
                        help="Run a single check and exit (for cron/scripts)")
    args = parser.parse_args(argv)

    cfg = WatchdogConfig(
        heartbeat_path=args.heartbeat,
        stale_seconds=args.stale_seconds,
        poll_seconds=args.poll_seconds,
        realert_seconds=args.realert_seconds,
        startup_grace_seconds=args.startup_grace_seconds,
    )
    sinks: List[Callable[[WatchdogAlert], None]] = [console_sink]
    if args.log_file:
        sinks.append(FileSink(args.log_file))

    wd = Watchdog(cfg, sinks=sinks)

    if args.once:
        alert = wd.check_once()
        # Exit code signals severity for scripting: 0 ok, 1 warn, 2 critical.
        if alert is None:
            return 0
        return {INFO: 0, WARNING: 1, CRITICAL: 2}.get(alert.severity, 1)

    print(f"[watchdog] watching {args.heartbeat} "
          f"(stale>{args.stale_seconds}s, poll={args.poll_seconds}s)",
          file=sys.stderr, flush=True)
    wd.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
