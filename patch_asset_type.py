#!/usr/bin/env python
# ==============================================================================
# patch_asset_type.py -- Fix "Unknown asset type for EURUSD" in data_manager.py
# ==============================================================================
# THE BUG:
#   data_manager._determine_asset_type() only recognises forex symbols that
#   contain a dash (its fallback tests `'-' in symbol`). So 'EUR-USD' works but
#   'EURUSD' falls through to 'unknown', and every backtest on 'EURUSD' is
#   skipped with "no data" -- which zeroed the entire pipeline funnel.
#   (There is also an operator-precedence bug: `A or B and C` parses as
#   `A or (B and C)`, not the intended `(A or B) and C`.)
#
# THE FIX:
#   Replace the forex line in the fallback block so it matches a symbol whether
#   or not it has a dash, by normalising (strip '-', uppercase) and comparing to
#   the known pairs. 'EURUSD' and 'EUR-USD' both resolve to 'forex'.
#
# SAFETY:
#   * --dry-run shows the change without writing.
#   * A timestamped .bak backup is written before any edit.
#   * The result is syntax-checked with ast.parse; if it fails, the backup is
#     restored automatically.
#   * --revert restores the most recent backup.
#   * Idempotent: running twice does nothing the second time.
#
# USAGE:
#   python patch_asset_type.py --dry-run
#   python patch_asset_type.py
#   python patch_asset_type.py --revert
# ==============================================================================

import ast
import sys
import shutil
import argparse
import glob
import os
from datetime import datetime

TARGET = "data_manager.py"

OLD = (
    "        # Fallback pattern detection\n"
    "        if symbol.endswith('=X') or '-' in symbol and symbol.replace('-', '') in "
    "['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']:\n"
    "            return 'forex'\n"
)

NEW = (
    "        # Fallback pattern detection\n"
    "        # Accept forex symbols with OR without a dash: 'EURUSD' and\n"
    "        # 'EUR-USD' both resolve to forex. (Previously only dashed symbols\n"
    "        # matched, so 'EURUSD' fell through to 'unknown' and was skipped.)\n"
    "        _forex_undashed = {'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',\n"
    "                           'USDCAD', 'USDCHF', 'NZDUSD'}\n"
    "        if symbol.endswith('=X') or symbol.replace('-', '').upper() in _forex_undashed:\n"
    "            return 'forex'\n"
)

# A marker that indicates the patch is already applied (idempotency check).
MARKER = "_forex_undashed = {'EURUSD'"


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _syntax_ok(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"  [FAIL] syntax error after edit: {e}")
        return False


def revert():
    backups = sorted(glob.glob(f"{TARGET}.asset_type_*.bak"))
    if not backups:
        print("  [INFO] no backup found to revert.")
        return 1
    latest = backups[-1]
    shutil.copy2(latest, TARGET)
    print(f"  [OK] reverted {TARGET} from {latest}")
    return 0


def patch(dry_run=False):
    if not os.path.exists(TARGET):
        print(f"  [FAIL] {TARGET} not found in current directory. Run this from "
              f"the project root.")
        return 2

    with open(TARGET, "r", encoding="utf-8") as f:
        content = f.read()

    if MARKER in content:
        print("  [INFO] already patched (idempotent no-op).")
        return 0

    if OLD not in content:
        print("  [FAIL] could not find the exact original block to replace.")
        print("         The file may have been edited. No changes made.")
        return 3

    new_content = content.replace(OLD, NEW, 1)

    if dry_run:
        print("  [DRY RUN] would replace the forex fallback block with:")
        print("  " + "-" * 60)
        for line in NEW.splitlines():
            print("  | " + line)
        print("  " + "-" * 60)
        print("  [DRY RUN] no file written.")
        return 0

    backup = f"{TARGET}.asset_type_{_timestamp()}.bak"
    shutil.copy2(TARGET, backup)
    print(f"  [OK] backup written: {backup}")

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)

    if not _syntax_ok(TARGET):
        shutil.copy2(backup, TARGET)
        print(f"  [OK] restored from backup due to syntax failure.")
        return 4

    print(f"  [OK] patched {TARGET}: EURUSD and EUR-USD now both resolve to forex.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Fix EURUSD asset-type detection")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the change without writing")
    ap.add_argument("--revert", action="store_true",
                    help="Restore the most recent backup")
    args = ap.parse_args()

    print("=" * 64)
    print(" PATCH: data_manager asset-type detection (EURUSD fix)")
    print("=" * 64)

    if args.revert:
        return revert()
    return patch(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
