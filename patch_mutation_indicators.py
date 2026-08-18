#!/usr/bin/env python
# ==============================================================================
# patch_mutation_indicators.py -- Fix the indicator list the mutation agent uses
# ==============================================================================
# THE BUG (source-level):
#   mutation_config.INDICATORS lists indicators for the AI to add to strategies.
#   Five of them DO NOT EXIST in this Backtrader version, so any variant the
#   agent builds with them crashes at backtest time:
#       * OBV / OnBalanceVolume  -> "has no attribute 'OnBalanceVolume'"
#       * VWAP                   -> not a base Backtrader indicator
#       * Volume Profile         -> not a Backtrader indicator at all
#       * Keltner Channels       -> KeltnerChannel missing in this version
#       * Donchian Channels      -> DonchianChannel missing in this version
#   This is why variant_11 (OBV) failed. It is a SOURCE bug: the agent did
#   exactly what the config told it to. Fixing the individual variant files
#   treats the symptom; fixing the list stops new broken variants being born.
#
#   (Verified against this environment's backtrader.indicators: RSI, MACD, ADX,
#   BollingerBands, ATR, Stochastic, EMA, CCI, WilliamsR, Ichimoku all exist;
#   the five above do not.)
#
# THE FIX:
#   Replace the INDICATORS block with only indicators confirmed present. If you
#   later add custom implementations of OBV/VWAP/Keltner/Donchian to the engine,
#   add them back to the list.
#
# SAFETY:
#   --dry-run, timestamped .bak backup, ast.parse check with auto-restore,
#   --revert, idempotent.
#
# USAGE:
#   python patch_mutation_indicators.py --dry-run
#   python patch_mutation_indicators.py
#   python patch_mutation_indicators.py --revert
# ==============================================================================

import ast
import shutil
import argparse
import glob
import os
from datetime import datetime

TARGET = "mutation_config.py"

OLD = '''INDICATORS = """
RSI
MACD
ADX
Bollinger Bands
ATR
VWAP
Stochastic
EMA
Volume Profile
OBV
CCI
Williams %R
Ichimoku Cloud
Keltner Channels
Donchian Channels
"""'''

NEW = '''INDICATORS = """
RSI
MACD
ADX
Bollinger Bands
ATR
Stochastic
EMA
CCI
Williams %R
Ichimoku Cloud
"""
# NOTE: OBV, VWAP, Volume Profile, Keltner Channels, and Donchian Channels were
# removed -- they do not exist in this Backtrader version and any variant using
# them crashes at backtest time. Re-add only after adding custom implementations.'''

MARKER = "do not exist in this Backtrader version and any variant using"


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
    backups = sorted(glob.glob(f"{TARGET}.indfix_*.bak"))
    if not backups:
        print("  [INFO] no backup found to revert.")
        return 1
    latest = backups[-1]
    shutil.copy2(latest, TARGET)
    print(f"  [OK] reverted {TARGET} from {latest}")
    return 0


def patch(dry_run=False):
    if not os.path.exists(TARGET):
        print(f"  [FAIL] {TARGET} not found. Run from the project root.")
        return 2

    with open(TARGET, "r", encoding="utf-8") as f:
        content = f.read()

    if MARKER in content:
        print("  [INFO] already patched (idempotent no-op).")
        return 0

    if OLD not in content:
        print("  [FAIL] could not find the exact INDICATORS block to replace.")
        print("         The file may have been edited. No changes made.")
        return 3

    new_content = content.replace(OLD, NEW, 1)

    if dry_run:
        print("  [DRY RUN] would remove: OBV, VWAP, Volume Profile, "
              "Keltner Channels, Donchian Channels")
        print("  [DRY RUN] keeping: RSI, MACD, ADX, Bollinger Bands, ATR, "
              "Stochastic, EMA, CCI, Williams %R, Ichimoku Cloud")
        print("  [DRY RUN] no file written.")
        return 0

    backup = f"{TARGET}.indfix_{_timestamp()}.bak"
    shutil.copy2(TARGET, backup)
    print(f"  [OK] backup written: {backup}")

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)

    if not _syntax_ok(TARGET):
        shutil.copy2(backup, TARGET)
        print("  [OK] restored from backup due to syntax failure.")
        return 4

    print(f"  [OK] patched {TARGET}: removed 5 indicators missing from this "
          f"Backtrader version.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Fix mutation_config indicator list (remove missing ones)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print(" PATCH: mutation_config indicator list (remove non-existent)")
    print("=" * 64)

    if args.revert:
        return revert()
    return patch(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
