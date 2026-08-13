# ==============================================================================
# algorithm_ideas.py -- AI-Generated Algorithm Ideas Backlog
# ==============================================================================
# A parking lot for strategy ideas that AI (or a human) generates but that the
# current backtesting system CANNOT test yet -- ideas needing order-book depth,
# live-only feeds, cross-exchange data, funding rates, etc.
#
# WHY THIS EXISTS:
#   Untestable ideas must NOT enter the Phase 1 pipeline. If they did, they would
#   either crash for lack of data or -- worse -- appear to "pass" against data
#   that does not actually exercise their edge, polluting backtest results.
#   This module keeps them quarantined but recorded, so they can be revisited
#   when the required data/infrastructure arrives.
#
#   Design principle (project-wide): make the ABSENCE of a testable answer
#   representable and loud, rather than silently faking testability.
#
# STORAGE:
#   SQLite (authoritative) at data/algorithm_ideas.db
#   Markdown export (human-readable) at algorithm_ideas_backlog.md
#
# THREE WAYS TO USE:
#   1. CLI:     python algorithm_ideas.py --add
#               python algorithm_ideas.py --list
#               python algorithm_ideas.py --promote <idea_id>
#               python algorithm_ideas.py --export-md
#   2. Import:  from algorithm_ideas import IdeaBacklog
#               IdeaBacklog().capture(title=..., description=..., ...)
#   3. Pipeline: llm_extractor / mutate_strategy call capture() when an idea is
#               flagged untestable, instead of writing it into discovery.db.
#
# PROMOTION:
#   When the data/infrastructure an idea needs becomes available, promote() hands
#   it to StrategyInbox.add_strategy(), where it re-enters the normal pipeline.
# ==============================================================================

import sys
import json
import hashlib
import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

# ── Path resolution: match project conventions, degrade gracefully ────────────
try:
    from discovery_config import DATA_DIR
    _DB_PATH = str(DATA_DIR / "algorithm_ideas.db")
except Exception:
    _DATA_DIR = Path(__file__).parent / "data"
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _DB_PATH = str(_DATA_DIR / "algorithm_ideas.db")

_MD_EXPORT_PATH = str(Path(__file__).parent / "algorithm_ideas_backlog.md")


# ── Controlled vocabularies (advisory, not enforced -- ideas are freeform) ────
CONFIDENCE_LEVELS = ("speculative", "promising", "ready-to-code")

CATEGORIES = (
    "live-only",              # reactive/feed-dependent/latency-sensitive
    "cross-asset-macro",      # correlated instruments, calendars, vol surface
    "crypto-specific",        # funding rates, liquidations, MEV, basis
    "multi-feed",             # needs multiple reconciled data sources
    "regime-adaptive",        # logic exists; needs live vol/regime data
    "stat-arb",               # pairs, mean-reversion across instruments
    "microstructure",         # spread prediction, order-flow, VWAP/TWAP
    "multi-timeframe",        # cross-timeframe; needs robust alignment
    "risk-hedge",             # portfolio-level protections
    "speculative-rnd",        # unvalidated concepts
    "uncategorized",
)

# Idea lifecycle status
STATUS_OPEN = "open"            # captured, awaiting review
STATUS_PROMISING = "promising"  # reviewed, worth pursuing when unblocked
STATUS_PROMOTED = "promoted"    # sent to strategy_inbox
STATUS_DISCARDED = "discarded"  # reviewed and rejected

VALID_STATUSES = (STATUS_OPEN, STATUS_PROMISING, STATUS_PROMOTED, STATUS_DISCARDED)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class IdeaBacklog:
    """
    Storage and lifecycle management for untestable algorithm ideas.

    The SQLite DB is authoritative. The Markdown file is a human-readable
    export regenerated on demand -- never edited by hand as a source of truth.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # WAL mode: matches project convention for concurrent read/write safety.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return conn

    def _ensure_tables(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ideas (
                idea_id           TEXT PRIMARY KEY,
                title             TEXT NOT NULL,
                description       TEXT NOT NULL,
                why_untestable    TEXT,
                data_needed       TEXT,
                category          TEXT DEFAULT 'uncategorized',
                tags              TEXT,
                confidence        TEXT DEFAULT 'speculative',
                effort            TEXT,
                generated_by      TEXT DEFAULT 'unknown',
                source_context    TEXT,
                asset_class       TEXT,
                timeframe         TEXT,
                status            TEXT DEFAULT 'open',
                promoted_strategy_id TEXT,
                notes             TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ideas_category ON ideas(category)")
        conn.commit()
        conn.close()

    # ── Capture ───────────────────────────────────────────────────────────────
    def capture(
        self,
        title: str,
        description: str,
        why_untestable: str = "",
        data_needed: str = "",
        category: str = "uncategorized",
        tags: str = "",
        confidence: str = "speculative",
        effort: str = "",
        generated_by: str = "unknown",
        source_context: str = "",
        asset_class: str = "",
        timeframe: str = "",
        notes: str = "",
    ) -> str:
        """
        Record a new idea. Returns the assigned idea_id.

        Deduplicates on (title + description) hash: capturing the same idea twice
        updates the existing row's timestamp rather than creating a duplicate,
        so an LLM re-suggesting the same thing doesn't spam the backlog.
        """
        if not title.strip():
            raise ValueError("Idea title cannot be empty")
        if not description.strip():
            raise ValueError("Idea description cannot be empty")

        # Advisory normalization -- unknown values are kept but nudged to defaults
        # so downstream filters have something predictable to work with.
        if confidence not in CONFIDENCE_LEVELS:
            confidence = "speculative"
        if category not in CATEGORIES:
            category = "uncategorized"

        dedup_key = hashlib.sha256(
            f"{title.strip().lower()}|{description.strip().lower()}".encode("utf-8")
        ).hexdigest()
        idea_id = dedup_key[:16]

        now = _utcnow()
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT idea_id, status FROM ideas WHERE idea_id = ?", (idea_id,)
        ).fetchone()

        if existing is not None:
            # Idea already known -- refresh updated_at and any newly supplied
            # fields, but do NOT resurrect a discarded idea automatically.
            conn.execute(
                """
                UPDATE ideas
                   SET why_untestable = COALESCE(NULLIF(?, ''), why_untestable),
                       data_needed    = COALESCE(NULLIF(?, ''), data_needed),
                       tags           = COALESCE(NULLIF(?, ''), tags),
                       confidence     = ?,
                       effort         = COALESCE(NULLIF(?, ''), effort),
                       notes          = COALESCE(NULLIF(?, ''), notes),
                       updated_at     = ?
                 WHERE idea_id = ?
                """,
                (why_untestable, data_needed, tags, confidence, effort,
                 notes, now, idea_id),
            )
            conn.commit()
            conn.close()
            return idea_id

        conn.execute(
            """
            INSERT INTO ideas (
                idea_id, title, description, why_untestable, data_needed,
                category, tags, confidence, effort, generated_by,
                source_context, asset_class, timeframe, status,
                notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (idea_id, title.strip(), description.strip(), why_untestable,
             data_needed, category, tags, confidence, effort, generated_by,
             source_context, asset_class, timeframe, STATUS_OPEN, notes,
             now, now),
        )
        conn.commit()
        conn.close()
        return idea_id

    # ── Query ─────────────────────────────────────────────────────────────────
    def get(self, idea_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM ideas WHERE idea_id = ?", (idea_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row is not None else None

    def list_ideas(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List ideas, optionally filtered. Newest first."""
        clauses = []
        params: List[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if confidence is not None:
            clauses.append("confidence = ?")
            params.append(confidence)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT * FROM ideas {where} ORDER BY created_at DESC", params
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def set_status(self, idea_id: str, status: str) -> bool:
        """Update an idea's lifecycle status. Returns False if idea not found."""
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of {VALID_STATUSES}"
            )
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE ideas SET status = ?, updated_at = ? WHERE idea_id = ?",
            (status, _utcnow(), idea_id),
        )
        conn.commit()
        changed = cur.rowcount > 0
        conn.close()
        return changed

    # ── Promotion ─────────────────────────────────────────────────────────────
    def promote(
        self,
        idea_id: str,
        code: str = "",
        quality_override: float = 60.0,
        force: bool = False,
    ) -> str:
        """
        Promote an idea into the normal pipeline via StrategyInbox.

        An idea only belongs in the pipeline once whatever made it untestable is
        resolved. By default this refuses to promote an idea still marked with a
        blocking `why_untestable` note unless `force=True`, so ideas don't slip
        back into backtesting before their data actually exists.

        Returns the new strategy_id assigned by StrategyInbox.
        """
        idea = self.get(idea_id)
        if idea is None:
            raise KeyError(f"No idea with id '{idea_id}'")
        if idea["status"] == STATUS_PROMOTED and not force:
            raise ValueError(
                f"Idea '{idea_id}' already promoted "
                f"(strategy_id={idea.get('promoted_strategy_id')}). "
                f"Use force=True to promote again."
            )
        if idea.get("why_untestable") and not force:
            raise ValueError(
                f"Idea '{idea_id}' still has a blocking note: "
                f"{idea['why_untestable']!r}. Resolve the blocker (or pass "
                f"force=True) before promoting into the pipeline."
            )

        try:
            from strategy_inbox import StrategyInbox
        except Exception as e:
            raise RuntimeError(
                f"Cannot import StrategyInbox for promotion: {e}"
            ) from e

        inbox = StrategyInbox()
        strategy_id = inbox.add_strategy(
            name=idea["title"],
            description=idea["description"],
            hypothesis=idea.get("source_context", "") or "",
            code=code,
            asset_class=idea.get("asset_class") or "forex",
            timeframe=idea.get("timeframe") or "1hour",
            source_url="",
            tags=(idea.get("tags") or "") + ",promoted-from-ideas-backlog",
            quality_override=quality_override,
        )

        conn = self._get_conn()
        conn.execute(
            """
            UPDATE ideas
               SET status = ?, promoted_strategy_id = ?, updated_at = ?
             WHERE idea_id = ?
            """,
            (STATUS_PROMOTED, strategy_id, _utcnow(), idea_id),
        )
        conn.commit()
        conn.close()
        return strategy_id

    # ── Metrics ───────────────────────────────────────────────────────────────
    def metrics(self) -> Dict[str, Any]:
        """Aggregate counts for dashboards / periodic review."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) AS c FROM ideas").fetchone()["c"]
        by_status = {
            r["status"]: r["c"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS c FROM ideas GROUP BY status"
            ).fetchall()
        }
        by_category = {
            r["category"]: r["c"]
            for r in conn.execute(
                "SELECT category, COUNT(*) AS c FROM ideas GROUP BY category"
            ).fetchall()
        }
        promoted = by_status.get(STATUS_PROMOTED, 0)
        discarded = by_status.get(STATUS_DISCARDED, 0)
        reviewed = promoted + discarded
        conn.close()
        return {
            "total": total,
            "by_status": by_status,
            "by_category": by_category,
            "promoted": promoted,
            "discarded": discarded,
            "promotion_rate": (promoted / reviewed) if reviewed else 0.0,
        }

    # ── Markdown export ───────────────────────────────────────────────────────
    def export_markdown(self, path: str = _MD_EXPORT_PATH) -> str:
        """
        Regenerate the human-readable backlog file from the DB.

        Written with CRLF line endings to match project convention.
        """
        ideas = self.list_ideas()
        m = self.metrics()
        lines: List[str] = []
        lines.append("# Algorithm Ideas Backlog")
        lines.append("")
        lines.append(
            "**Auto-generated from `algorithm_ideas.db` -- do not edit by hand.** "
            "Use `python algorithm_ideas.py` to modify ideas."
        )
        lines.append("")
        lines.append(f"**Last exported:** {_utcnow()}")
        lines.append("")
        lines.append(
            f"**Totals:** {m['total']} ideas "
            f"({m['by_status'].get(STATUS_OPEN, 0)} open, "
            f"{m['by_status'].get(STATUS_PROMISING, 0)} promising, "
            f"{m['promoted']} promoted, {m['discarded']} discarded)"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        if not ideas:
            lines.append("*No ideas captured yet.*")
            lines.append("")
        else:
            # Group by category for readability
            for cat in CATEGORIES:
                cat_ideas = [i for i in ideas if i["category"] == cat]
                if not cat_ideas:
                    continue
                lines.append(f"## {cat}")
                lines.append("")
                for i in cat_ideas:
                    lines.append(f"### {i['title']}  `[{i['status']}]`")
                    lines.append(f"- **id:** `{i['idea_id']}`")
                    lines.append(f"- **confidence:** {i['confidence']}")
                    if i.get("generated_by"):
                        lines.append(f"- **generated by:** {i['generated_by']}")
                    if i.get("tags"):
                        lines.append(f"- **tags:** {i['tags']}")
                    if i.get("effort"):
                        lines.append(f"- **effort:** {i['effort']}")
                    lines.append("")
                    lines.append(f"**Idea:** {i['description']}")
                    lines.append("")
                    if i.get("why_untestable"):
                        lines.append(f"**Why not testable yet:** {i['why_untestable']}")
                        lines.append("")
                    if i.get("data_needed"):
                        lines.append(f"**Data/infrastructure needed:** {i['data_needed']}")
                        lines.append("")
                    if i.get("promoted_strategy_id"):
                        lines.append(
                            f"**Promoted to strategy:** `{i['promoted_strategy_id']}`"
                        )
                        lines.append("")
                    if i.get("notes"):
                        lines.append(f"**Notes:** {i['notes']}")
                        lines.append("")
                    lines.append("---")
                    lines.append("")

        text = "\r\n".join(lines)
        Path(path).write_text(text, encoding="utf-8", newline="")
        return path


# ==============================================================================
# CLI
# ==============================================================================
def _interactive_add(backlog: IdeaBacklog) -> None:
    print("\n== Capture a new algorithm idea ==\n")
    title = input("Title: ").strip()
    description = input("Description: ").strip()
    why = input("Why not testable yet: ").strip()
    data_needed = input("Data/infrastructure needed: ").strip()
    print(f"Categories: {', '.join(CATEGORIES)}")
    category = input("Category [uncategorized]: ").strip() or "uncategorized"
    print(f"Confidence: {', '.join(CONFIDENCE_LEVELS)}")
    confidence = input("Confidence [speculative]: ").strip() or "speculative"
    tags = input("Tags (comma-separated): ").strip()
    generated_by = input("Generated by [human]: ").strip() or "human"
    idea_id = backlog.capture(
        title=title, description=description, why_untestable=why,
        data_needed=data_needed, category=category, confidence=confidence,
        tags=tags, generated_by=generated_by,
    )
    print(f"\nCaptured idea: {idea_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI-generated algorithm ideas backlog"
    )
    parser.add_argument("--add", action="store_true", help="Interactively add an idea")
    parser.add_argument("--add-quick", nargs=2, metavar=("TITLE", "DESCRIPTION"),
                        help="Quick-add an idea by title + description")
    parser.add_argument("--list", action="store_true", help="List ideas")
    parser.add_argument("--status", help="Filter --list by status")
    parser.add_argument("--category", help="Filter --list by category")
    parser.add_argument("--promote", metavar="IDEA_ID",
                        help="Promote an idea into strategy_inbox")
    parser.add_argument("--code-file", help="Code file to attach when promoting")
    parser.add_argument("--force", action="store_true",
                        help="Force promotion past blocking notes")
    parser.add_argument("--set-status", nargs=2, metavar=("IDEA_ID", "STATUS"),
                        help="Set an idea's status")
    parser.add_argument("--metrics", action="store_true", help="Show backlog metrics")
    parser.add_argument("--export-md", action="store_true",
                        help="Regenerate the Markdown backlog file")
    args = parser.parse_args()

    backlog = IdeaBacklog()

    if args.add:
        _interactive_add(backlog)
        backlog.export_markdown()
        return 0

    if args.add_quick:
        title, description = args.add_quick
        idea_id = backlog.capture(title=title, description=description,
                                  generated_by="human")
        print(f"Captured idea: {idea_id}")
        backlog.export_markdown()
        return 0

    if args.promote:
        code = ""
        if args.code_file:
            code = Path(args.code_file).read_text(encoding="utf-8")
        try:
            sid = backlog.promote(args.promote, code=code, force=args.force)
            print(f"Promoted idea {args.promote} -> strategy {sid}")
            backlog.export_markdown()
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"Promotion failed: {e}")
            return 1
        return 0

    if args.set_status:
        idea_id, status = args.set_status
        try:
            ok = backlog.set_status(idea_id, status)
        except ValueError as e:
            print(str(e))
            return 1
        print(f"{'Updated' if ok else 'No idea found:'} {idea_id}")
        backlog.export_markdown()
        return 0 if ok else 1

    if args.metrics:
        m = backlog.metrics()
        print(json.dumps(m, indent=2))
        return 0

    if args.export_md:
        path = backlog.export_markdown()
        print(f"Exported to {path}")
        return 0

    if args.list or True:  # default action
        ideas = backlog.list_ideas(status=args.status, category=args.category)
        if not ideas:
            print("No ideas found.")
            return 0
        for i in ideas:
            print(f"[{i['status']:10}] {i['idea_id']}  ({i['category']}) "
                  f"{i['confidence']:12}  {i['title']}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
