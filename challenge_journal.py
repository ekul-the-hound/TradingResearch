# ==============================================================================
# challenge_journal.py -- Per-Day Challenge Audit Journal
# ==============================================================================
# A durable, per-day record of a live challenge: closing P&L, how close each
# rule came to breaching, how many trades ran, and every governor action taken.
#
# WHY IT EXISTS:
#   During a challenge you need to answer, days later, "what happened on day 7
#   and how close did I come to a breach?" -- and if a payout review ever
#   questions the account, you need an evidence trail of what the system did and
#   when. A journal is also how you learn between attempts: the days you nearly
#   breached are the days worth studying.
#
# WHAT IT RECORDS:
#   * Per-day summary: date, opening/closing equity, day P&L and %, trades taken,
#     worst governor decision of the day, and the tightest rule headroom reached.
#   * Intraday events: individual governor verdicts and notable actions
#     (REDUCE / HALT_NEW / FLATTEN), timestamped, so the sequence is auditable.
#
# IT READS REAL GOVERNOR OUTPUT:
#   record_verdict() accepts a live_governor.Verdict (or a compatible object /
#   dict) and pulls its actual fields -- decision, daily_loss_pct, daily_limit,
#   drawdown_floor, headroom -- rather than inventing a rule-proximity metric.
#
# STORAGE:
#   SQLite (authoritative), WAL mode, row_factory=Row -- matching database.py.
#   Two tables: journal_days (one row per trading day) and journal_events
#   (many rows per day). export_markdown() renders a human-readable log.
#
# DESIGN PRINCIPLE (project-wide):
#   The journal records what was OBSERVED, including gaps. A day with no closing
#   equity recorded is stored as incomplete, not back-filled with a guess; a
#   missing governor field is stored as NULL, not zero. The audit trail must not
#   fabricate the very numbers it exists to preserve.
# ==============================================================================

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path resolution matching project convention, with graceful fallback.
try:
    from discovery_config import DATA_DIR
    _DB_PATH = str(DATA_DIR / "challenge_journal.db")
except Exception:
    _D = Path(__file__).parent / "data"
    _D.mkdir(parents=True, exist_ok=True)
    _DB_PATH = str(_D / "challenge_journal.db")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    return date.today().isoformat()


# Governor decision severity ranking, so we can track the WORST of the day.
_DECISION_RANK = {"allow": 0, "reduce": 1, "halt_new": 2, "flatten": 3}


def _decision_value(decision: Any) -> str:
    """Normalize a Decision enum / str / dict to its lowercase string value."""
    if decision is None:
        return "allow"
    v = getattr(decision, "value", decision)
    return str(v).lower()


@dataclass
class DayRecord:
    trading_date: str
    opening_equity: Optional[float] = None
    closing_equity: Optional[float] = None
    day_pnl: Optional[float] = None
    day_pnl_pct: Optional[float] = None
    trades: int = 0
    worst_decision: str = "allow"
    tightest_headroom: Optional[float] = None   # smallest headroom seen (closest to breach)
    min_daily_pct_to_limit: Optional[float] = None  # closest approach to daily loss limit
    complete: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChallengeJournal:
    """Per-day audit journal backed by SQLite."""

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return conn

    def _ensure_tables(self) -> None:
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS journal_days (
                trading_date          TEXT PRIMARY KEY,
                opening_equity        REAL,
                closing_equity        REAL,
                day_pnl               REAL,
                day_pnl_pct           REAL,
                trades                INTEGER DEFAULT 0,
                worst_decision        TEXT DEFAULT 'allow',
                tightest_headroom     REAL,
                min_daily_pct_to_limit REAL,
                complete              INTEGER DEFAULT 0,
                notes                 TEXT DEFAULT '',
                updated_at            TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS journal_events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                trading_date  TEXT NOT NULL,
                timestamp     TEXT NOT NULL,
                kind          TEXT NOT NULL,      -- 'verdict' | 'trade' | 'action' | 'note'
                decision      TEXT,
                reason        TEXT,
                detail        TEXT,
                data          TEXT                -- JSON blob of extra fields
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_date "
                     "ON journal_events(trading_date)")
        conn.commit()
        conn.close()

    # -- Day lifecycle ---------------------------------------------------------
    def open_day(self, opening_equity: float,
                 trading_date: Optional[str] = None) -> str:
        """Record the day's opening equity. Idempotent per date."""
        d = trading_date or _today_iso()
        conn = self._conn()
        existing = conn.execute(
            "SELECT trading_date FROM journal_days WHERE trading_date = ?",
            (d,)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO journal_days (trading_date, opening_equity, "
                "updated_at) VALUES (?, ?, ?)",
                (d, float(opening_equity), _utcnow_iso()))
        else:
            conn.execute(
                "UPDATE journal_days SET opening_equity = ?, updated_at = ? "
                "WHERE trading_date = ?",
                (float(opening_equity), _utcnow_iso(), d))
        conn.commit()
        conn.close()
        return d

    def close_day(self, closing_equity: float,
                  trading_date: Optional[str] = None,
                  notes: str = "") -> DayRecord:
        """Finalize the day: compute P&L, mark complete."""
        d = trading_date or _today_iso()
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM journal_days WHERE trading_date = ?", (d,)).fetchone()
        opening = row["opening_equity"] if row else None
        day_pnl = None
        day_pnl_pct = None
        if opening is not None:
            day_pnl = float(closing_equity) - float(opening)
            if opening:
                day_pnl_pct = day_pnl / float(opening) * 100.0
        if row is None:
            conn.execute(
                "INSERT INTO journal_days (trading_date, closing_equity, "
                "day_pnl, day_pnl_pct, complete, notes, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (d, float(closing_equity), day_pnl, day_pnl_pct, notes,
                 _utcnow_iso()))
        else:
            conn.execute(
                "UPDATE journal_days SET closing_equity = ?, day_pnl = ?, "
                "day_pnl_pct = ?, complete = 1, notes = ?, updated_at = ? "
                "WHERE trading_date = ?",
                (float(closing_equity), day_pnl, day_pnl_pct, notes,
                 _utcnow_iso(), d))
        conn.commit()
        conn.close()
        return self.get_day(d)  # type: ignore[return-value]

    # -- Event recording -------------------------------------------------------
    def record_verdict(self, verdict: Any,
                       trading_date: Optional[str] = None) -> None:
        """
        Record a governor Verdict. Updates the day's worst-decision and
        tightest-headroom, and logs the verdict as an event.
        """
        d = trading_date or _today_iso()
        decision = _decision_value(getattr(verdict, "decision", None)
                                   if not isinstance(verdict, dict)
                                   else verdict.get("decision"))
        reason = _get(verdict, "reason", "")
        detail = _get(verdict, "detail", "")
        headroom = _get(verdict, "headroom", None)
        daily_pct = _get(verdict, "daily_loss_pct", None)
        daily_limit = _get(verdict, "daily_limit", None)

        data = {
            "daily_loss": _get(verdict, "daily_loss", None),
            "daily_loss_pct": daily_pct,
            "daily_limit": daily_limit,
            "drawdown_floor": _get(verdict, "drawdown_floor", None),
            "headroom": headroom,
            "anchor_equity": _get(verdict, "anchor_equity", None),
        }
        self._log_event(d, "verdict", decision=decision, reason=str(reason),
                        detail=str(detail), data=data)

        # min_daily_pct_to_limit: how close to the daily limit (as a % of limit
        # consumed). Higher = closer to breach; we track the max seen.
        approach = None
        if (daily_pct is not None and daily_limit not in (None, 0)):
            try:
                # daily_loss_pct is loss as % of equity; convert to fraction of
                # the limit if the limit is also a %. If not comparable, skip.
                approach = float(daily_pct)
            except (TypeError, ValueError):
                approach = None

        self._update_day_from_verdict(d, decision, headroom, approach)

    def record_trade(self, symbol: str, side: str, size: float,
                     price: Optional[float] = None,
                     trading_date: Optional[str] = None,
                     detail: str = "") -> None:
        """Record a single trade and increment the day's trade count."""
        d = trading_date or _today_iso()
        self._log_event(d, "trade", detail=detail or f"{side} {size} {symbol}"
                        + (f" @ {price}" if price is not None else ""),
                        data={"symbol": symbol, "side": side, "size": size,
                              "price": price})
        conn = self._conn()
        self._ensure_day_row(conn, d)
        conn.execute("UPDATE journal_days SET trades = trades + 1, "
                     "updated_at = ? WHERE trading_date = ?",
                     (_utcnow_iso(), d))
        conn.commit()
        conn.close()

    def record_action(self, action: str, detail: str = "",
                      trading_date: Optional[str] = None) -> None:
        """Record a notable action (e.g. 'flatten_all', 'kill_switch')."""
        d = trading_date or _today_iso()
        self._log_event(d, "action", reason=action, detail=detail)

    def record_note(self, note: str, trading_date: Optional[str] = None) -> None:
        d = trading_date or _today_iso()
        self._log_event(d, "note", detail=note)

    # -- Internal writers ------------------------------------------------------
    def _log_event(self, d: str, kind: str, decision: str = "",
                   reason: str = "", detail: str = "",
                   data: Optional[Dict[str, Any]] = None) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO journal_events (trading_date, timestamp, kind, "
            "decision, reason, detail, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (d, _utcnow_iso(), kind, decision, reason, detail,
             json.dumps(data) if data else None))
        conn.commit()
        conn.close()

    def _ensure_day_row(self, conn: sqlite3.Connection, d: str) -> None:
        exists = conn.execute(
            "SELECT 1 FROM journal_days WHERE trading_date = ?", (d,)).fetchone()
        if exists is None:
            conn.execute(
                "INSERT INTO journal_days (trading_date, updated_at) "
                "VALUES (?, ?)", (d, _utcnow_iso()))

    def _update_day_from_verdict(self, d: str, decision: str,
                                 headroom: Optional[float],
                                 approach: Optional[float]) -> None:
        conn = self._conn()
        self._ensure_day_row(conn, d)
        row = conn.execute(
            "SELECT worst_decision, tightest_headroom, min_daily_pct_to_limit "
            "FROM journal_days WHERE trading_date = ?", (d,)).fetchone()

        worst = row["worst_decision"] or "allow"
        if _DECISION_RANK.get(decision, 0) > _DECISION_RANK.get(worst, 0):
            worst = decision

        tightest = row["tightest_headroom"]
        if headroom is not None:
            tightest = headroom if tightest is None else min(tightest, headroom)

        approach_col = row["min_daily_pct_to_limit"]
        if approach is not None:
            approach_col = (approach if approach_col is None
                            else max(approach_col, approach))

        conn.execute(
            "UPDATE journal_days SET worst_decision = ?, tightest_headroom = ?, "
            "min_daily_pct_to_limit = ?, updated_at = ? WHERE trading_date = ?",
            (worst, tightest, approach_col, _utcnow_iso(), d))
        conn.commit()
        conn.close()

    # -- Queries ---------------------------------------------------------------
    def get_day(self, trading_date: str) -> Optional[DayRecord]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM journal_days WHERE trading_date = ?",
            (trading_date,)).fetchone()
        conn.close()
        if row is None:
            return None
        return DayRecord(
            trading_date=row["trading_date"],
            opening_equity=row["opening_equity"],
            closing_equity=row["closing_equity"],
            day_pnl=row["day_pnl"],
            day_pnl_pct=row["day_pnl_pct"],
            trades=row["trades"] or 0,
            worst_decision=row["worst_decision"] or "allow",
            tightest_headroom=row["tightest_headroom"],
            min_daily_pct_to_limit=row["min_daily_pct_to_limit"],
            complete=bool(row["complete"]),
            notes=row["notes"] or "",
        )

    def list_days(self) -> List[DayRecord]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT trading_date FROM journal_days ORDER BY trading_date"
        ).fetchall()
        conn.close()
        return [self.get_day(r["trading_date"]) for r in rows]  # type: ignore

    def get_events(self, trading_date: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM journal_events WHERE trading_date = ? "
            "ORDER BY id", (trading_date,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # -- Export ----------------------------------------------------------------
    def export_markdown(self, path: Optional[str] = None) -> str:
        p = path or str(Path(self.db_path).with_suffix(".md"))
        days = self.list_days()
        lines: List[str] = ["# Challenge Journal", "",
                            f"Exported {_utcnow_iso()}", ""]
        for day in days:
            flag = "" if day.complete else "  (INCOMPLETE)"
            lines.append(f"## {day.trading_date}{flag}")
            if day.day_pnl is not None:
                lines.append(f"- P&L: {day.day_pnl:+.2f} "
                             f"({day.day_pnl_pct:+.2f}%)" if day.day_pnl_pct
                             is not None else f"- P&L: {day.day_pnl:+.2f}")
            lines.append(f"- Trades: {day.trades}")
            lines.append(f"- Worst governor decision: {day.worst_decision}")
            if day.tightest_headroom is not None:
                lines.append(f"- Tightest headroom to breach: "
                             f"{day.tightest_headroom:.2f}")
            if day.notes:
                lines.append(f"- Notes: {day.notes}")
            lines.append("")
        text = "\r\n".join(lines)
        Path(p).write_text(text, encoding="utf-8", newline="")
        return p


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


__all__ = ["ChallengeJournal", "DayRecord"]
