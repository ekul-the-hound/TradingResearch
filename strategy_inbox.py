# ==============================================================================
# strategy_inbox.py -- Manual Strategy Entry + Discovery DB Bridge
# ==============================================================================
# Add strategies manually alongside what the discovery AI finds.
# Strategies entered here go into the SAME discovery.db and appear
# in the same pipeline as scraped strategies.
#
# Three ways to use:
#   1. CLI:        python strategy_inbox.py --add
#   2. Import:     from strategy_inbox import StrategyInbox
#   3. Dashboard:  Called from react_dashboard2.py's entry form
#
# Also provides the bridge that feeds discovery.db into run_pipeline.py
# so discovered + manual strategies flow into backtesting automatically.
#
# Usage:
#   python strategy_inbox.py --add                    # Interactive entry
#   python strategy_inbox.py --add-quick "RSI Mean Reversion" "Buy when RSI<30"
#   python strategy_inbox.py --add-file my_strategy.py
#   python strategy_inbox.py --list                   # Show all strategies
#   python strategy_inbox.py --export                 # Export to strategies/ dir
#   python strategy_inbox.py --export-for-pipeline    # Prepare for run_pipeline.py
# ==============================================================================

import sys
import json
import hashlib
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

try:
    from discovery_config import DISCOVERY_CONFIG as cfg, STRATEGIES_DIR
    DB_PATH = str(cfg.db_path)
except ImportError:
    DB_PATH = str(Path(__file__).parent / "data" / "discovery.db")
    STRATEGIES_DIR = Path(__file__).parent / "strategies" / "discovered"


class StrategyInbox:
    """
    Manual strategy entry into the discovery database.

    Strategies added here get origin_source='manual' and appear
    alongside scraped strategies in the pipeline.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self):
        """Create strategies table if it doesn't exist, and migrate old schemas."""
        conn = self._get_conn()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                doc_id TEXT,
                strategy_name TEXT NOT NULL,
                summary TEXT,
                description TEXT,
                generated_code TEXT,
                code_file_path TEXT,
                origin_source TEXT DEFAULT 'manual',
                source_url TEXT,
                source_type TEXT,
                source_bias TEXT,
                parent_docs TEXT,
                quality_score REAL,
                has_math INTEGER DEFAULT 0,
                has_backtest INTEGER DEFAULT 0,
                has_code INTEGER DEFAULT 0,
                has_explicit_params INTEGER DEFAULT 0,
                extraction_model TEXT,
                extraction_timestamp TEXT,
                extraction_cost_estimate REAL DEFAULT 0.0,
                code_validates INTEGER DEFAULT 0,
                validation_trades INTEGER DEFAULT 0,
                validation_error TEXT,
                is_duplicate INTEGER DEFAULT 0,
                duplicate_of TEXT,
                similarity_score REAL,
                status TEXT DEFAULT 'manual_entry',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                asset_class TEXT,
                timeframe TEXT,
                hypothesis TEXT,
                tags TEXT
            )
        ''')

        # ── Migration: add columns that may be missing from older databases ──
        # If the table was originally created by research_db.py (discovery pipeline),
        # it won't have asset_class, timeframe, hypothesis, or tags columns.
        # ALTER TABLE ADD COLUMN is safe to call — it silently fails if column exists.
        for col, coltype in [
            ("asset_class", "TEXT"),
            ("timeframe", "TEXT"),
            ("hypothesis", "TEXT"),
            ("tags", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE strategies ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # Column already exists — this is fine

        conn.commit()
        conn.close()

    def add_strategy(
        self,
        name: str,
        description: str,
        hypothesis: str = "",
        code: str = "",
        asset_class: str = "forex",
        timeframe: str = "1hour",
        source_url: str = "",
        tags: str = "",
        quality_override: float = 75.0,
    ) -> str:
        """
        Add a strategy manually.

        Returns:
            strategy_id: The ID assigned to this strategy.
        """
        ts = datetime.now().isoformat()
        sid = hashlib.sha256(f"{name}_{ts}".encode()).hexdigest()[:16]

        has_code = 1 if code.strip() else 0
        code_validates = 0
        if has_code:
            try:
                compile(code, f"{name}.py", "exec")
                code_validates = 1
            except SyntaxError:
                code_validates = 0

        conn = self._get_conn()
        conn.execute('''
            INSERT INTO strategies (
                strategy_id, strategy_name, summary, description, generated_code,
                origin_source, source_url, quality_score, has_code,
                code_validates, status, created_at, updated_at,
                asset_class, timeframe, hypothesis, tags, extraction_model
            ) VALUES (?, ?, ?, ?, ?, 'manual', ?, ?, ?, ?, 'manual_entry', ?, ?, ?, ?, ?, ?, 'human')
        ''', (sid, name, description[:200], description, code,
              source_url, quality_override, has_code, code_validates,
              ts, ts, asset_class, timeframe, hypothesis, tags))
        conn.commit()
        conn.close()

        return sid

    def add_from_file(self, filepath: str, name: str = "", description: str = "",
                      hypothesis: str = "", asset_class: str = "forex") -> str:
        """Add a strategy from a .py file."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        code = p.read_text()
        if not name:
            name = p.stem.replace("_", " ").title()

        # Try to extract docstring as description
        if not description:
            import ast
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.Module)):
                        ds = ast.get_docstring(node)
                        if ds:
                            description = ds[:500]
                            break
            except SyntaxError:
                pass

        sid = self.add_strategy(
            name=name, description=description or f"Strategy from {p.name}",
            hypothesis=hypothesis, code=code, asset_class=asset_class,
            quality_override=80.0)

        # Also save the .py file
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
        dest = STRATEGIES_DIR / p.name
        dest.write_text(code)

        return sid

    def list_strategies(self, origin: str = None, limit: int = 50) -> List[Dict]:
        """List strategies from the DB."""
        conn = self._get_conn()
        if origin:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE origin_source=? ORDER BY created_at DESC LIMIT ?",
                (origin, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM strategies ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def list_manual(self, limit: int = 50) -> List[Dict]:
        return self.list_strategies(origin="manual", limit=limit)

    def list_all_for_pipeline(self) -> List[Dict]:
        """Get all non-duplicate, validated strategies ready for backtesting."""
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT * FROM strategies
            WHERE is_duplicate=0
            AND (code_validates=1 OR origin_source='manual')
            AND status NOT IN ('rejected', 'failed')
            ORDER BY quality_score DESC
        ''').fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def export_for_pipeline(self, output_dir: str = None) -> int:
        """
        Export all ready strategies as .py files for run_pipeline.py.

        This is the BRIDGE between discovery and the rest of the system.
        """
        out = Path(output_dir) if output_dir else STRATEGIES_DIR
        out.mkdir(parents=True, exist_ok=True)

        strats = self.list_all_for_pipeline()
        exported = 0

        for s in strats:
            code = s.get("generated_code", "")
            if not code or not code.strip():
                continue

            # Generate safe filename
            name = s.get("strategy_name", "unnamed").lower()
            name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)[:40]
            sid = s.get("strategy_id", "")[:8]
            filename = f"disc_{name}_{sid}.py"

            filepath = out / filename
            filepath.write_text(code)
            exported += 1

            # Update status
            conn = self._get_conn()
            conn.execute("UPDATE strategies SET status='exported', code_file_path=? WHERE strategy_id=?",
                         (str(filepath), s["strategy_id"]))
            conn.commit()
            conn.close()

        return exported

    def delete_strategy(self, strategy_id: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM strategies WHERE strategy_id=?", (strategy_id,))
        conn.commit()
        conn.close()

    def update_strategy(self, strategy_id: str, **kwargs):
        conn = self._get_conn()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [strategy_id]
        conn.execute(f"UPDATE strategies SET {sets}, updated_at=? WHERE strategy_id=?",
                     vals[:-1] + [datetime.now().isoformat(), strategy_id])
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
        manual = conn.execute("SELECT COUNT(*) FROM strategies WHERE origin_source='manual'").fetchone()[0]
        scraped = conn.execute("SELECT COUNT(*) FROM strategies WHERE origin_source='scraped'").fetchone()[0]
        exported = conn.execute("SELECT COUNT(*) FROM strategies WHERE status='exported'").fetchone()[0]
        validated = conn.execute("SELECT COUNT(*) FROM strategies WHERE code_validates=1").fetchone()[0]
        conn.close()
        return {"total": total, "manual": manual, "scraped": scraped,
                "exported": exported, "validated": validated}


# ==============================================================================
# CLI
# ==============================================================================

def interactive_add():
    """Interactive strategy entry."""
    inbox = StrategyInbox()

    print("\n" + "=" * 60)
    print("  Add Strategy Manually")
    print("=" * 60)

    name = input("\n  Strategy name: ").strip()
    if not name:
        print("  Cancelled."); return

    desc = input("  Description: ").strip()
    hypothesis = input("  Hypothesis (why it works): ").strip()
    asset = input("  Asset class [forex/crypto/indices]: ").strip() or "forex"
    tf = input("  Timeframe [1hour]: ").strip() or "1hour"
    url = input("  Source URL (optional): ").strip()
    tags = input("  Tags (comma-separated, optional): ").strip()

    print("\n  Paste Backtrader code below (empty line + 'END' to finish):")
    print("  (Or press Enter twice to skip)")
    code_lines = []
    empty_count = 0
    while True:
        line = input()
        if line.strip() == "END":
            break
        if line.strip() == "" and empty_count > 0:
            break
        if line.strip() == "":
            empty_count += 1
        else:
            empty_count = 0
        code_lines.append(line)
    code = "\n".join(code_lines).strip()

    sid = inbox.add_strategy(
        name=name, description=desc, hypothesis=hypothesis,
        code=code, asset_class=asset, timeframe=tf,
        source_url=url, tags=tags)

    print(f"\n  Added: {name} (ID: {sid})")
    print(f"  Origin: manual | Quality: 75 | Code: {'yes' if code else 'no'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="TradingLab Strategy Inbox")
    parser.add_argument("--add", action="store_true", help="Interactive strategy entry")
    parser.add_argument("--add-quick", nargs=2, metavar=("NAME", "DESC"),
                        help="Quick add: name and description")
    parser.add_argument("--add-file", type=str, help="Add strategy from .py file")
    parser.add_argument("--list", action="store_true", help="List all strategies")
    parser.add_argument("--list-manual", action="store_true", help="List manual entries only")
    parser.add_argument("--export", action="store_true", help="Export strategies to files")
    parser.add_argument("--export-for-pipeline", action="store_true",
                        help="Export ready strategies for run_pipeline.py")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    args = parser.parse_args()

    inbox = StrategyInbox()

    if args.add:
        interactive_add()
    elif args.add_quick:
        sid = inbox.add_strategy(name=args.add_quick[0], description=args.add_quick[1])
        print(f"Added: {args.add_quick[0]} (ID: {sid})")
    elif args.add_file:
        sid = inbox.add_from_file(args.add_file)
        print(f"Added from file: {args.add_file} (ID: {sid})")
    elif args.list:
        strats = inbox.list_strategies()
        for s in strats:
            tag = f"[{s.get('origin_source','?')}]"
            print(f"  {s.get('quality_score',0):5.0f}  {tag:10s}  {s.get('strategy_name','?')[:35]}  ({s.get('status','?')})")
    elif args.list_manual:
        strats = inbox.list_manual()
        for s in strats:
            print(f"  {s.get('strategy_name','?')[:35]}  |  {s.get('description','')[:50]}  |  {s.get('status','?')}")
    elif args.export or args.export_for_pipeline:
        n = inbox.export_for_pipeline()
        print(f"Exported {n} strategies to {STRATEGIES_DIR}")
    elif args.stats:
        st = inbox.get_stats()
        print(f"\n  Total: {st['total']} | Manual: {st['manual']} | Scraped: {st['scraped']}")
        print(f"  Exported: {st['exported']} | Validated: {st['validated']}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()