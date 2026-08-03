# ==============================================================================
# apply_sources_patch.py -- Adds the Source Extraction tab to react_dashboard2.py
# ==============================================================================
# Applies 5 edits:
#   1. Import PgSources (with a safe fallback stub if the module fails to load)
#   2. NAV entry
#   3. TITLES entry
#   4. PAGES entry
#   5. NAV_COLORS accent
#
# Safe to run twice -- detects an already-patched file and exits without
# touching it. Writes a timestamped .bak before making any change.
# Preserves CRLF line endings.
#
# Usage (from the project root):
#   python apply_sources_patch.py
#   python apply_sources_patch.py --revert     # restore the newest .bak
#   python apply_sources_patch.py --dry-run    # show what would change
# ==============================================================================

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

TARGET = Path(__file__).parent / "react_dashboard2.py"

# ------------------------------------------------------------------------
# Edits: (label, find, replace)
# Written with \n; converted to \r\n at apply time if the file is CRLF.
# ------------------------------------------------------------------------

EDITS = [
    (
        "1/5 import PgSources",
        "try:\n"
        "    from strategy_inbox import StrategyInbox\n"
        "    INBOX_AVAILABLE = True\n"
        "except ImportError:\n"
        "    INBOX_AVAILABLE = False\n",

        "try:\n"
        "    from strategy_inbox import StrategyInbox\n"
        "    INBOX_AVAILABLE = True\n"
        "except ImportError:\n"
        "    INBOX_AVAILABLE = False\n"
        "\n"
        "# Source extraction page (paste transcripts/articles)\n"
        "try:\n"
        "    from page_sources import PgSources\n"
        "    SOURCES_AVAILABLE = True\n"
        "except Exception as _src_err:\n"
        "    SOURCES_AVAILABLE = False\n"
        "    _SRC_ERR = str(_src_err)\n"
        "\n"
        "    @component\n"
        "    def PgSources():\n"
        "        return html.div({\"style\": {\"color\": \"#f59e0b\", \"padding\": \"40px\",\n"
        "            \"fontFamily\": \"monospace\", \"fontSize\": \"13px\"}},\n"
        "            html.p(\"page_sources.py failed to load:\"),\n"
        "            html.p({\"style\": {\"color\": \"#ef4444\"}}, _SRC_ERR),\n"
        "            html.p({\"style\": {\"color\": \"#64748b\"}},\n"
        "                \"Check that page_sources.py and source_extractor.py are in \"\n"
        "                \"the project root, then clear __pycache__ and restart.\"))\n",
    ),
    (
        "2/5 NAV entry",
        '    ("inbox",       "[IN]","Strategy Inbox"),\n',
        '    ("inbox",       "[IN]","Strategy Inbox"),\n'
        '    ("sources",     "[DOC]","Source Extraction"),\n',
    ),
    (
        "3/5 TITLES entry",
        '    "inbox":       ("Strategy Inbox","Add strategies manually + view AI discoveries"),\n',
        '    "inbox":       ("Strategy Inbox","Add strategies manually + view AI discoveries"),\n'
        '    "sources":     ("Source Extraction","Paste transcripts/articles -- extract strategies, review, approve to codegen"),\n',
    ),
    (
        "4/5 PAGES entry",
        '    "pipeline":PgPipeline,"inbox":PgInbox,"backtests":PgBacktests,"strategies":PgStrategies,\n',
        '    "pipeline":PgPipeline,"inbox":PgInbox,"sources":PgSources,"backtests":PgBacktests,"strategies":PgStrategies,\n',
    ),
    (
        "5/5 NAV_COLORS accent",
        'NAV_COLORS = {"lineage":T["p1"],"overfit":T["p1"],"optimization":T["p2"],\n',
        'NAV_COLORS = {"sources":T["p5"],"lineage":T["p1"],"overfit":T["p1"],"optimization":T["p2"],\n',
    ),
]

PATCH_MARKER = "from page_sources import PgSources"


def detect_newline(raw: str) -> str:
    return "\r\n" if "\r\n" in raw else "\n"


def revert():
    baks = sorted(TARGET.parent.glob("react_dashboard2.py.bak_*"))
    if not baks:
        print("  No backup found. Nothing to revert.")
        return 1
    newest = baks[-1]
    shutil.copy2(newest, TARGET)
    print(f"  Reverted react_dashboard2.py from {newest.name}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Add the Source Extraction tab to react_dashboard2.py")
    p.add_argument("--dry-run", action="store_true", help="Show what would change, write nothing")
    p.add_argument("--revert", action="store_true", help="Restore the newest .bak")
    args = p.parse_args()

    print()
    print("=" * 64)
    print("  SOURCE EXTRACTION TAB -- DASHBOARD PATCHER")
    print("=" * 64)

    if args.revert:
        return revert()

    if not TARGET.exists():
        print(f"  ERROR: {TARGET.name} not found.")
        print(f"  Run this from the project root: {TARGET.parent}")
        return 1

    # newline="" disables universal-newline translation so we can actually
    # see whether the file is CRLF and write the same endings back.
    with open(TARGET, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = detect_newline(raw)
    print(f"  Target:      {TARGET.name} ({len(raw):,} chars, "
          f"{'CRLF' if nl == chr(13) + chr(10) else 'LF'})")

    # Idempotency
    if PATCH_MARKER in raw:
        print("  Already patched -- PgSources import found. Nothing to do.")
        print("=" * 64 + "\n")
        return 0

    # Companion files
    missing = [f for f in ("page_sources.py", "source_extractor.py")
               if not (TARGET.parent / f).exists()]
    if missing:
        print(f"  WARNING: not in project root: {', '.join(missing)}")
        print("  The tab will show a load error until they're placed there.")

    # Normalize to \n for matching
    work = raw.replace("\r\n", "\n")

    # Verify every anchor before touching anything
    failures = []
    for label, find, _ in EDITS:
        n = work.count(find)
        if n != 1:
            failures.append(f"    [{label}] found {n} times, expected exactly 1")

    if failures:
        print("\n  ANCHOR CHECK FAILED -- no changes made:")
        for f in failures:
            print(f)
        print("\n  Your react_dashboard2.py differs from the expected version.")
        print("  Apply the edits by hand from PATCH_react_dashboard2.md instead.")
        print("=" * 64 + "\n")
        return 1

    print("  Anchors:     5/5 found\n")

    for label, find, replace in EDITS:
        work = work.replace(find, replace, 1)
        added = replace.count("\n") - find.count("\n")
        print(f"  [OK] {label}  (+{added} lines)")

    # Restore original line endings
    out = work.replace("\n", nl) if nl == "\r\n" else work

    if args.dry_run:
        print(f"\n  DRY RUN -- nothing written. Result would be {len(out):,} chars.")
        print("=" * 64 + "\n")
        return 0

    # Backup, then write
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET.with_suffix(f".py.bak_{stamp}")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(out, encoding="utf-8", newline="")

    # Syntax check the result
    import ast
    try:
        ast.parse(out)
        syn = "OK"
    except SyntaxError as e:
        syn = f"FAILED: {e}"
        shutil.copy2(bak, TARGET)
        print(f"\n  SYNTAX CHECK {syn}")
        print("  Reverted automatically. No changes kept.")
        print("=" * 64 + "\n")
        return 1

    print(f"\n  Backup:      {bak.name}")
    print(f"  Syntax:      {syn}")
    print(f"  Written:     {len(out):,} chars")
    print()
    print("  NEXT STEPS:")
    print("    Remove-Item -Recurse -Force __pycache__")
    print("    python react_dashboard2.py")
    print()
    print("  Then click 'Source Extraction' in the sidebar.")
    print("  To undo:  python apply_sources_patch.py --revert")
    print("=" * 64 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
