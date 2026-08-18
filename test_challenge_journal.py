# ==============================================================================
# test_challenge_journal.py -- Tests for the per-day challenge audit journal
# ==============================================================================
# Convention: import failures are HARD errors, not skips.
# ==============================================================================

import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import List, Optional

from challenge_journal import ChallengeJournal, DayRecord


@dataclass
class FakeVerdict:
    decision: str = "allow"
    reason: str = ""
    detail: str = ""
    daily_loss: Optional[float] = None
    daily_loss_pct: Optional[float] = None
    daily_limit: Optional[float] = None
    drawdown_floor: Optional[float] = None
    headroom: Optional[float] = None
    anchor_equity: Optional[float] = None


class JournalTestBase(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.j = ChallengeJournal(db_path=self.db)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            p = self.db + suffix
            if os.path.exists(p):
                os.remove(p)


class TestDayLifecycle(JournalTestBase):
    def test_open_and_close_computes_pnl(self):
        self.j.open_day(100_000, trading_date="2026-01-05")
        day = self.j.close_day(101_500, trading_date="2026-01-05")
        self.assertEqual(day.day_pnl, 1500.0)
        self.assertAlmostEqual(day.day_pnl_pct, 1.5)  # type: ignore[arg-type]
        self.assertTrue(day.complete)

    def test_loss_day(self):
        self.j.open_day(100_000, trading_date="2026-01-06")
        day = self.j.close_day(96_000, trading_date="2026-01-06")
        self.assertEqual(day.day_pnl, -4000.0)
        self.assertAlmostEqual(day.day_pnl_pct, -4.0)  # type: ignore[arg-type]

    def test_open_day_idempotent(self):
        self.j.open_day(100_000, trading_date="2026-01-05")
        self.j.open_day(105_000, trading_date="2026-01-05")  # re-open updates
        day = self.j.get_day("2026-01-05")
        self.assertEqual(day.opening_equity, 105_000)

    def test_incomplete_day_before_close(self):
        self.j.open_day(100_000, trading_date="2026-01-05")
        day = self.j.get_day("2026-01-05")
        self.assertFalse(day.complete)
        self.assertIsNone(day.closing_equity)

    def test_close_without_open_stores_incomplete_pnl(self):
        # Closing a day that was never opened: no opening equity to diff.
        day = self.j.close_day(99_000, trading_date="2026-01-07")
        self.assertTrue(day.complete)
        self.assertIsNone(day.day_pnl)  # honest: cannot compute without opening


class TestVerdictRecording(JournalTestBase):
    def test_verdict_logged_as_event(self):
        self.j.open_day(100_000, trading_date="2026-01-05")
        self.j.record_verdict(FakeVerdict(decision="reduce",
                                          reason="near_limit", headroom=1500),
                              trading_date="2026-01-05")
        events = self.j.get_events("2026-01-05")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "verdict")
        self.assertEqual(events[0]["decision"], "reduce")

    def test_worst_decision_tracked(self):
        d = "2026-01-05"
        self.j.open_day(100_000, trading_date=d)
        self.j.record_verdict(FakeVerdict(decision="allow"), trading_date=d)
        self.j.record_verdict(FakeVerdict(decision="reduce"), trading_date=d)
        self.j.record_verdict(FakeVerdict(decision="halt_new"), trading_date=d)
        self.j.record_verdict(FakeVerdict(decision="allow"), trading_date=d)
        day = self.j.get_day(d)
        self.assertEqual(day.worst_decision, "halt_new")

    def test_flatten_is_worst(self):
        d = "2026-01-05"
        self.j.record_verdict(FakeVerdict(decision="halt_new"), trading_date=d)
        self.j.record_verdict(FakeVerdict(decision="flatten"), trading_date=d)
        self.assertEqual(self.j.get_day(d).worst_decision, "flatten")

    def test_tightest_headroom_tracked(self):
        d = "2026-01-05"
        self.j.record_verdict(FakeVerdict(headroom=3000), trading_date=d)
        self.j.record_verdict(FakeVerdict(headroom=800), trading_date=d)
        self.j.record_verdict(FakeVerdict(headroom=1500), trading_date=d)
        self.assertEqual(self.j.get_day(d).tightest_headroom, 800)

    def test_verdict_as_dict(self):
        d = "2026-01-05"
        self.j.record_verdict({"decision": "reduce", "reason": "x",
                               "headroom": 500}, trading_date=d)
        self.assertEqual(self.j.get_day(d).worst_decision, "reduce")

    def test_missing_headroom_is_null_not_zero(self):
        d = "2026-01-05"
        self.j.record_verdict(FakeVerdict(decision="allow", headroom=None),
                              trading_date=d)
        # tightest_headroom stays None, not fabricated to 0.
        self.assertIsNone(self.j.get_day(d).tightest_headroom)


class TestTrades(JournalTestBase):
    def test_trade_increments_count(self):
        d = "2026-01-05"
        self.j.open_day(100_000, trading_date=d)
        self.j.record_trade("EURUSD", "buy", 0.5, 1.10, trading_date=d)
        self.j.record_trade("GBPUSD", "sell", 0.3, 1.25, trading_date=d)
        self.assertEqual(self.j.get_day(d).trades, 2)

    def test_trade_logged_as_event(self):
        d = "2026-01-05"
        self.j.record_trade("EURUSD", "buy", 0.5, 1.10, trading_date=d)
        events = self.j.get_events(d)
        trade_events = [e for e in events if e["kind"] == "trade"]
        self.assertEqual(len(trade_events), 1)


class TestActionsAndNotes(JournalTestBase):
    def test_action_recorded(self):
        d = "2026-01-05"
        self.j.record_action("flatten_all", "daily limit breached",
                             trading_date=d)
        events = self.j.get_events(d)
        self.assertEqual(events[0]["kind"], "action")
        self.assertEqual(events[0]["reason"], "flatten_all")

    def test_note_recorded(self):
        d = "2026-01-05"
        self.j.record_note("news event at 14:30", trading_date=d)
        events = self.j.get_events(d)
        self.assertEqual(events[0]["kind"], "note")


class TestQueries(JournalTestBase):
    def test_list_days_ordered(self):
        self.j.open_day(100_000, trading_date="2026-01-07")
        self.j.open_day(100_000, trading_date="2026-01-05")
        self.j.open_day(100_000, trading_date="2026-01-06")
        days = self.j.list_days()
        dates = [d.trading_date for d in days]
        self.assertEqual(dates, ["2026-01-05", "2026-01-06", "2026-01-07"])

    def test_get_missing_day_returns_none(self):
        self.assertIsNone(self.j.get_day("2099-01-01"))


class TestMarkdownExport(JournalTestBase):
    def test_export_creates_file(self):
        self.j.open_day(100_000, trading_date="2026-01-05")
        self.j.record_trade("EURUSD", "buy", 0.5, trading_date="2026-01-05")
        self.j.close_day(101_000, trading_date="2026-01-05", notes="good day")
        path = tempfile.mktemp(suffix=".md")
        out = self.j.export_markdown(path=path)
        self.assertTrue(os.path.exists(out))
        with open(out, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("2026-01-05", content)
        self.assertIn("good day", content)
        os.remove(out)

    def test_incomplete_flagged_in_export(self):
        self.j.open_day(100_000, trading_date="2026-01-05")  # never closed
        path = tempfile.mktemp(suffix=".md")
        out = self.j.export_markdown(path=path)
        with open(out, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("INCOMPLETE", content)
        os.remove(out)

    def test_export_is_crlf(self):
        self.j.open_day(100_000, trading_date="2026-01-05")
        path = tempfile.mktemp(suffix=".md")
        out = self.j.export_markdown(path=path)
        with open(out, "rb") as fh:
            raw = fh.read()
        self.assertIn(b"\r\n", raw)
        self.assertEqual(raw.replace(b"\r\n", b"").count(b"\n"), 0)
        os.remove(out)


class TestPersistence(JournalTestBase):
    def test_survives_reopen(self):
        d = "2026-01-05"
        self.j.open_day(100_000, trading_date=d)
        self.j.close_day(102_000, trading_date=d)
        # New journal instance on the same DB file.
        j2 = ChallengeJournal(db_path=self.db)
        day = j2.get_day(d)
        self.assertEqual(day.day_pnl, 2000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)