# ==============================================================================
# test_watchdog.py -- Tests for the engine heartbeat + watchdog
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# Uses injected `now` timestamps so no real waiting is required.
# ==============================================================================

import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from watchdog import (
    HeartbeatWriter, Watchdog, WatchdogConfig, WatchdogAlert,
    CollectingSink, FileSink,
    INFO, WARNING, CRITICAL, _parse_iso,
)


def iso(dt):
    return dt.isoformat()


class TestHeartbeatWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = str(Path(self.tmp) / "hb.json")

    def test_beat_writes_file(self):
        hb = HeartbeatWriter(self.path)
        payload = hb.beat(open_positions=2, connected=True, tick_count=99)
        self.assertTrue(Path(self.path).exists())
        on_disk = json.loads(Path(self.path).read_text())
        self.assertEqual(on_disk["open_positions"], 2)
        self.assertTrue(on_disk["connected"])
        self.assertEqual(on_disk["tick_count"], 99)
        self.assertEqual(on_disk["beats"], 1)

    def test_beats_increment(self):
        hb = HeartbeatWriter(self.path)
        hb.beat()
        hb.beat()
        p = hb.beat()
        self.assertEqual(p["beats"], 3)

    def test_no_leftover_tmp_file(self):
        hb = HeartbeatWriter(self.path)
        hb.beat()
        tmp = Path(self.path + ".tmp")
        self.assertFalse(tmp.exists())  # atomic replace cleans up

    def test_shutdown_marker(self):
        hb = HeartbeatWriter(self.path)
        hb.beat(open_positions=1)
        hb.mark_shutdown(clean=True)
        on_disk = json.loads(Path(self.path).read_text())
        self.assertTrue(on_disk["shutdown"])
        self.assertTrue(on_disk["clean"])


class TestWatchdogEvaluate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = str(Path(self.tmp) / "hb.json")
        self.cfg = WatchdogConfig(
            heartbeat_path=self.path,
            stale_seconds=15.0,
            startup_grace_seconds=30.0,
            realert_seconds=60.0,
        )

    def _write_hb(self, ts, open_positions=0, connected=True, **extra):
        payload = {
            "engine_id": "test",
            "timestamp": ts,
            "open_positions": open_positions,
            "connected": connected,
        }
        payload.update(extra)
        Path(self.path).write_text(json.dumps(payload))

    def test_fresh_heartbeat_no_alert(self):
        now = datetime.now(timezone.utc)
        self._write_hb(iso(now - timedelta(seconds=2)))
        wd = Watchdog(self.cfg, sinks=[])
        self.assertIsNone(wd.evaluate(now=now))

    def test_stale_with_positions_is_critical(self):
        now = datetime.now(timezone.utc)
        self._write_hb(iso(now - timedelta(seconds=60)), open_positions=3)
        wd = Watchdog(self.cfg, sinks=[])
        alert = wd.evaluate(now=now)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, CRITICAL)
        self.assertEqual(alert.data["open_positions"], 3)

    def test_stale_while_flat_is_warning(self):
        now = datetime.now(timezone.utc)
        self._write_hb(iso(now - timedelta(seconds=60)), open_positions=0)
        wd = Watchdog(self.cfg, sinks=[])
        alert = wd.evaluate(now=now)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, WARNING)

    def test_missing_file_within_grace_no_alert(self):
        wd = Watchdog(self.cfg, sinks=[])
        # started_at is ~now; within grace window -> no alert
        self.assertIsNone(wd.evaluate(now=wd._started_at + timedelta(seconds=5)))

    def test_missing_file_after_grace_is_critical(self):
        wd = Watchdog(self.cfg, sinks=[])
        alert = wd.evaluate(now=wd._started_at + timedelta(seconds=45))
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, CRITICAL)
        self.assertEqual(alert.data["reason"], "missing_file")

    def test_unparseable_file_is_warning(self):
        Path(self.path).write_text("{ this is not json")
        wd = Watchdog(self.cfg, sinks=[])
        alert = wd.evaluate(now=datetime.now(timezone.utc))
        self.assertEqual(alert.severity, WARNING)
        self.assertEqual(alert.data["reason"], "unparseable")

    def test_clean_shutdown_no_alert(self):
        Path(self.path).write_text(json.dumps(
            {"shutdown": True, "clean": True, "timestamp": iso(datetime.now(timezone.utc))}))
        wd = Watchdog(self.cfg, sinks=[])
        self.assertIsNone(wd.evaluate(now=datetime.now(timezone.utc)))

    def test_unclean_shutdown_is_warning(self):
        Path(self.path).write_text(json.dumps(
            {"shutdown": True, "clean": False, "timestamp": iso(datetime.now(timezone.utc))}))
        wd = Watchdog(self.cfg, sinks=[])
        alert = wd.evaluate(now=datetime.now(timezone.utc))
        self.assertEqual(alert.severity, WARNING)

    def test_recovery_emits_info(self):
        now = datetime.now(timezone.utc)
        wd = Watchdog(self.cfg, sinks=[])
        # First: stale
        self._write_hb(iso(now - timedelta(seconds=60)), open_positions=0)
        a1 = wd.evaluate(now=now)
        self.assertEqual(a1.severity, WARNING)
        self.assertTrue(wd._was_stale)
        # Then: fresh again
        self._write_hb(iso(now))
        a2 = wd.evaluate(now=now)
        self.assertIsNotNone(a2)
        self.assertEqual(a2.severity, INFO)
        self.assertEqual(a2.data["reason"], "recovered")
        self.assertFalse(wd._was_stale)

    def test_bad_timestamp_is_warning(self):
        self._write_hb("not-a-date", open_positions=0)
        wd = Watchdog(self.cfg, sinks=[])
        alert = wd.evaluate(now=datetime.now(timezone.utc))
        self.assertEqual(alert.severity, WARNING)
        self.assertEqual(alert.data["reason"], "bad_timestamp")

    def test_naive_timestamp_treated_as_utc(self):
        now = datetime.now(timezone.utc)
        naive = (now - timedelta(seconds=60)).replace(tzinfo=None)
        self._write_hb(naive.isoformat(), open_positions=1)
        wd = Watchdog(self.cfg, sinks=[])
        alert = wd.evaluate(now=now)
        self.assertEqual(alert.severity, CRITICAL)


class TestDispatchAntiSpam(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = str(Path(self.tmp) / "hb.json")
        self.cfg = WatchdogConfig(heartbeat_path=self.path, stale_seconds=15.0,
                                  realert_seconds=60.0, startup_grace_seconds=0.0)

    def _stale_hb(self, now, open_positions=1):
        Path(self.path).write_text(json.dumps({
            "timestamp": iso(now - timedelta(seconds=60)),
            "open_positions": open_positions, "connected": True,
        }))

    def test_first_alert_dispatches(self):
        now = datetime.now(timezone.utc)
        self._stale_hb(now)
        sink = CollectingSink()
        wd = Watchdog(self.cfg, sinks=[sink])
        wd.check_once(now=now)
        self.assertEqual(len(sink.alerts), 1)

    def test_repeat_within_window_suppressed(self):
        now = datetime.now(timezone.utc)
        self._stale_hb(now)
        sink = CollectingSink()
        wd = Watchdog(self.cfg, sinks=[sink])
        wd.check_once(now=now)
        # 10s later, still stale, within 60s realert window -> suppressed
        wd.check_once(now=now + timedelta(seconds=10))
        self.assertEqual(len(sink.alerts), 1)

    def test_repeat_after_window_refires(self):
        now = datetime.now(timezone.utc)
        self._stale_hb(now)
        sink = CollectingSink()
        wd = Watchdog(self.cfg, sinks=[sink])
        wd.check_once(now=now)
        # 70s later, past the 60s window -> re-fires
        # (the stale hb timestamp is relative to `now`, so refresh it)
        later = now + timedelta(seconds=70)
        Path(self.path).write_text(json.dumps({
            "timestamp": iso(later - timedelta(seconds=60)),
            "open_positions": 1, "connected": True,
        }))
        wd.check_once(now=later)
        self.assertEqual(len(sink.alerts), 2)

    def test_broken_sink_does_not_crash(self):
        now = datetime.now(timezone.utc)
        self._stale_hb(now)

        def broken(alert):
            raise RuntimeError("sink is broken")

        good = CollectingSink()
        wd = Watchdog(self.cfg, sinks=[broken, good])
        # Should not raise; good sink still receives.
        wd.check_once(now=now)
        self.assertEqual(len(good.alerts), 1)


class TestFileSink(unittest.TestCase):
    def test_appends_json_lines(self):
        tmp = tempfile.mkdtemp()
        logp = str(Path(tmp) / "alerts.log")
        sink = FileSink(logp)
        sink(WatchdogAlert(CRITICAL, "first"))
        sink(WatchdogAlert(WARNING, "second"))
        lines = Path(logp).read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["message"], "first")
        self.assertEqual(json.loads(lines[1])["severity"], WARNING)


class TestRunLoop(unittest.TestCase):
    def test_max_iterations_stops(self):
        tmp = tempfile.mkdtemp()
        path = str(Path(tmp) / "hb.json")
        # fresh heartbeat so no alerts, loop should just run and stop
        HeartbeatWriter(path).beat(open_positions=0)
        cfg = WatchdogConfig(heartbeat_path=path, stale_seconds=999,
                             poll_seconds=0.001, startup_grace_seconds=999)
        wd = Watchdog(cfg, sinks=[])
        wd.run(max_iterations=3)  # must terminate
        self.assertFalse(wd._running)


class TestEndToEndScenario(unittest.TestCase):
    """Simulate: engine beats, then dies with a position open."""

    def test_crash_with_position_detected(self):
        tmp = tempfile.mkdtemp()
        path = str(Path(tmp) / "hb.json")
        cfg = WatchdogConfig(heartbeat_path=path, stale_seconds=15,
                             startup_grace_seconds=0, realert_seconds=0)
        sink = CollectingSink()
        wd = Watchdog(cfg, sinks=[sink])

        t0 = datetime.now(timezone.utc)
        hb = HeartbeatWriter(path)

        # Engine healthy, position open. Rewrite timestamp to t0 explicitly.
        Path(path).write_text(json.dumps({
            "timestamp": iso(t0), "open_positions": 1, "connected": True}))
        self.assertIsNone(wd.check_once(now=t0))  # fresh -> no alert

        # 20s later the engine has stopped beating (file frozen at t0).
        t1 = t0 + timedelta(seconds=20)
        alert = wd.check_once(now=t1)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, CRITICAL)
        self.assertIn("OPEN POSITION", alert.message.upper())
        self.assertEqual(len(sink.alerts), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
