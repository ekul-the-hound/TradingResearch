#!/usr/bin/env python
# ==============================================================================
# fix_broken_variants.py -- Repair the two variant files that crash at backtest
# ==============================================================================
# Two variants failed in the pipeline run for reasons that are bugs in the
# GENERATED CODE, not the pipeline:
#
#   variant_11: "module 'backtrader.indicators' has no attribute 'OnBalanceVolume'"
#       Your Backtrader version has no OnBalanceVolume/OBV indicator. This fixer
#       injects a small, correct custom OBV indicator class and rewrites the
#       reference to use it.
#
#   variant_07: "unsupported operand type(s) for -: 'float' and 'NoneType'"
#       An indicator line is read during its warmup period, before it has a
#       value, so it is None and the arithmetic blows up. This fixer inserts a
#       warmup guard at the top of next() so the strategy waits until its
#       indicators are ready.
#
# This runs on YOUR machine because the variant files are not in the repo copy
# Claude can see. It reads the real files, shows what it will change, and only
# writes after a backup, with an ast syntax check and auto-rollback.
#
# USAGE:
#   python fix_broken_variants.py --dry-run     # inspect + show planned changes
#   python fix_broken_variants.py               # apply fixes (backs up first)
#   python fix_broken_variants.py --revert      # restore latest backups
# ==============================================================================

import ast
import re
import glob
import shutil
import argparse
from datetime import datetime
from pathlib import Path

VARIANTS_DIR = Path("strategies/variants")

# A minimal, correct OBV implementation for Backtrader, injected when needed.
OBV_CLASS = '''

class _CustomOBV(bt.Indicator):
    """On-Balance Volume (this Backtrader build lacks a built-in OnBalanceVolume)."""
    lines = ('obv',)

    def __init__(self):
        self.addminperiod(2)

    def next(self):
        if len(self) < 2:
            self.lines.obv[0] = 0.0
            return
        prev = self.lines.obv[-1]
        if self.data.close[0] > self.data.close[-1]:
            self.lines.obv[0] = prev + self.data.volume[0]
        elif self.data.close[0] < self.data.close[-1]:
            self.lines.obv[0] = prev - self.data.volume[0]
        else:
            self.lines.obv[0] = prev
'''

# The warmup guard inserted at the start of next().
WARMUP_GUARD = (
    "        # WARMUP GUARD: skip bars where any indicator is still None\n"
    "        for _ind in getattr(self, '_guard_indicators', []):\n"
    "            try:\n"
    "                if _ind[0] is None:\n"
    "                    return\n"
    "            except (IndexError, TypeError):\n"
    "                return\n"
)


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _find(patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(str(VARIANTS_DIR / pat)))
    return sorted(set(out))


def _syntax_ok(path):
    try:
        ast.parse(Path(path).read_text(encoding="utf-8"))
        return True
    except SyntaxError as e:
        print(f"    [FAIL] syntax error after edit: {e}")
        return False


def fix_obv(path, dry_run):
    """Fix a variant that references the missing OnBalanceVolume indicator."""
    text = Path(path).read_text(encoding="utf-8")
    if "OnBalanceVolume" not in text and "indicators.OBV" not in text:
        return False, "no OBV reference found"
    if "_CustomOBV" in text:
        return False, "already fixed"

    new = text
    # Inject the custom class after the imports (after the last 'import' line).
    lines = new.splitlines(keepends=True)
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.startswith("import ") or ln.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, OBV_CLASS)
    new = "".join(lines)

    # Rewrite the reference: bt.indicators.OnBalanceVolume(...) or
    # bt.ind.OnBalanceVolume(...) or indicators.OBV(...) -> _CustomOBV(self.data)
    new = re.sub(r"bt\.indicators\.OnBalanceVolume\([^)]*\)",
                 "_CustomOBV(self.data)", new)
    new = re.sub(r"bt\.ind\.OnBalanceVolume\([^)]*\)",
                 "_CustomOBV(self.data)", new)
    new = re.sub(r"bt\.indicators\.OBV\([^)]*\)",
                 "_CustomOBV(self.data)", new)

    if dry_run:
        return True, "would inject _CustomOBV and rewrite the OnBalanceVolume reference"

    Path(path).write_text(new, encoding="utf-8")
    return True, "injected _CustomOBV and rewrote reference"


def fix_warmup(path, dry_run):
    """Fix a variant that reads an indicator before warmup (float - None)."""
    text = Path(path).read_text(encoding="utf-8")
    if "_guard_indicators" in text or "WARMUP GUARD" in text:
        return False, "already fixed"
    if "def next(self):" not in text:
        return False, "no next() method found"

    # Collect indicator attribute names assigned in __init__ (self.X = bt.ind...).
    ind_names = re.findall(r"self\.(\w+)\s*=\s*bt\.(?:indicators|ind)\.", text)
    ind_names += re.findall(r"self\.(\w+)\s*=\s*_CustomOBV", text)
    ind_names = list(dict.fromkeys(ind_names))  # unique, ordered

    new = text

    # Register the guard list at the end of __init__ if we found indicators.
    if ind_names:
        guard_list = ("        self._guard_indicators = ["
                      + ", ".join(f"self.{n}" for n in ind_names) + "]\n")
        # Insert right before def next(
        new = new.replace("    def next(self):",
                          guard_list + "\n    def next(self):", 1)

    # Insert the guard body as the first statement in next().
    new = new.replace("    def next(self):\n",
                      "    def next(self):\n" + WARMUP_GUARD, 1)

    if dry_run:
        found = ", ".join(ind_names) if ind_names else "(none detected)"
        return True, f"would add warmup guard for indicators: {found}"

    Path(path).write_text(new, encoding="utf-8")
    return True, f"added warmup guard for {len(ind_names)} indicator(s)"


def process(path, dry_run):
    print(f"  [FILE] {path}")
    backup = None
    if not dry_run:
        backup = f"{path}.varfix_{_timestamp()}.bak"
        shutil.copy2(path, backup)
        print(f"    backup: {backup}")

    changed_any = False
    for label, fn in (("OBV", fix_obv), ("warmup", fix_warmup)):
        changed, msg = fn(path, dry_run)
        status = "WOULD FIX" if (changed and dry_run) else ("FIXED" if changed else "skip")
        print(f"    [{status}] {label}: {msg}")
        changed_any = changed_any or changed

    if not dry_run and changed_any:
        if not _syntax_ok(path) and backup is not None:
            shutil.copy2(backup, path)
            print(f"    [OK] restored from backup due to syntax failure.")
            return
        print(f"    [OK] {path} fixed and parses clean.")


def revert():
    backups = sorted(glob.glob(str(VARIANTS_DIR / "*.varfix_*.bak")))
    if not backups:
        print("  [INFO] no variant backups found.")
        return 1
    # Restore each file from its most recent backup.
    latest = {}
    for b in backups:
        original = b.split(".varfix_")[0]
        latest[original] = b  # sorted, so last wins = most recent
    for original, b in latest.items():
        shutil.copy2(b, original)
        print(f"  [OK] reverted {original} from {b}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Fix broken variant strategy files")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print(" FIX: broken variant files (OnBalanceVolume + warmup None)")
    print("=" * 64)

    if args.revert:
        return revert()

    if not VARIANTS_DIR.exists():
        print(f"  [FAIL] {VARIANTS_DIR} not found. Run from the project root.")
        return 2

    # Target the two known-broken variants, but also scan any file that shows
    # the same failure signatures, so this is robust to renumbering.
    candidates = _find(["variant_07*.py", "variant_11*.py"])
    if not candidates:
        # Fall back to scanning all variants for the two signatures.
        for p in _find(["*.py"]):
            t = Path(p).read_text(encoding="utf-8", errors="ignore")
            if "OnBalanceVolume" in t or "indicators.OBV" in t:
                candidates.append(p)
    candidates = sorted(set(candidates))

    if not candidates:
        print("  [INFO] no matching variant files found to fix.")
        return 0

    for path in candidates:
        process(path, args.dry_run)

    print("=" * 64)
    if args.dry_run:
        print("  DRY RUN complete -- no files written. Re-run without --dry-run "
              "to apply.")
    else:
        print("  Done. Re-run the pipeline to confirm the two variants now "
              "backtest.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())