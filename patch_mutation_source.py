#!/usr/bin/env python
# ==============================================================================
# patch_mutation_source.py -- Fix the mutation SOURCE so bad variants stop
#                             being generated (not the individual files)
# ==============================================================================
# Two source-level defects in mutate_strategy.py let the OBV crash keep happening:
#
#   1. THE PROMPT'S OWN EXAMPLE IS WRONG.
#      Rule 1 tells the agent:  bt.indicators.OBV(self.data)   [OK]
#      and warns against:       bt.indicators.OnBalanceVolume() [WRONG]
#      But THIS Backtrader build has NEITHER -- there is no OBV and no
#      OnBalanceVolume. So the agent follows the rule and still crashes.
#
#   2. THE BUG-CHECKER REDIRECTS TO THE BROKEN NAME AND ONLY WARNS.
#      validate/check flags 'OnBalanceVolume' and says "should be 'OBV'"
#      (also broken), and it is a warning, not a block -- so a variant using a
#      non-existent indicator is still written to disk and still crashes.
#
# THIS PATCH:
#   * Rewrites Rule 1 to state there is no built-in OBV in this build, and to
#     tell the agent to AVOID volume indicators (or paste a self-contained
#     custom class) rather than referencing a missing one.
#   * Rewrites the checker so ANY OBV / OnBalanceVolume reference is flagged as
#     a HARD problem, and adds a helper the caller can use to reject such a
#     variant instead of only warning.
#
# Result: future batches are generated against rules that match the real engine,
# so this whole class of "indicator does not exist" crash stops at the source.
# The existing legacy variant files predate these rules and are not touched.
#
# SAFETY: --dry-run, timestamped .bak, ast.parse check + auto-rollback,
#         --revert, idempotent.
#
# USAGE:
#   python patch_mutation_source.py --dry-run
#   python patch_mutation_source.py
#   python patch_mutation_source.py --revert
# ==============================================================================

import ast
import shutil
import argparse
import glob
import os
from datetime import datetime

TARGET = "mutate_strategy.py"

# --- Fix 1: the prompt's Rule 1 ------------------------------------------------
OLD_RULE1 = """### Rule 1: Indicator Names
Use the CORRECT Backtrader indicator names:
- [OK] CORRECT: `bt.indicators.OBV(self.data)` 
- [FAIL] WRONG: `bt.indicators.OnBalanceVolume()` (does not exist!)
- [OK] CORRECT: `bt.indicators.RSI(self.data.close, period=14)`
- [OK] CORRECT: `bt.indicators.ATR(self.data, period=14)`
- [OK] CORRECT: `bt.indicators.ADX(self.data, period=14)`
- [OK] CORRECT: `bt.indicators.BollingerBands(self.data.close)`
- [OK] CORRECT: `bt.indicators.MACD(self.data.close)`
- [OK] CORRECT: `bt.indicators.Stochastic(self.data)`"""

NEW_RULE1 = """### Rule 1: Indicator Names
Use ONLY indicators that exist in this Backtrader build. Verified available:
- [OK] `bt.indicators.RSI(self.data.close, period=14)`
- [OK] `bt.indicators.ATR(self.data, period=14)`
- [OK] `bt.indicators.ADX(self.data, period=14)`
- [OK] `bt.indicators.BollingerBands(self.data.close)`
- [OK] `bt.indicators.MACD(self.data.close)`
- [OK] `bt.indicators.Stochastic(self.data)`
- [OK] `bt.indicators.CCI(self.data)`, `bt.indicators.WilliamsR(self.data)`,
      `bt.indicators.Ichimoku(self.data)`, `bt.indicators.EMA(...)`,
      `bt.indicators.SMA(...)`

DO NOT USE these -- they DO NOT EXIST in this build and will crash:
- [FAIL] `bt.indicators.OBV(...)` and `bt.indicators.OnBalanceVolume(...)`
- [FAIL] `bt.indicators.VWAP(...)`
- [FAIL] `bt.indicators.KeltnerChannel(...)`
- [FAIL] `bt.indicators.DonchianChannel(...)`
- [FAIL] any "Volume Profile" indicator

If a volume-based idea is needed, DO NOT reference a missing indicator. Either
skip it, or include a fully self-contained custom indicator class in the file
(subclassing bt.Indicator) so nothing undefined is referenced."""

# --- Fix 2: the bug-checker ----------------------------------------------------
OLD_CHECK = """    # Check for wrong OBV indicator name
    if 'OnBalanceVolume' in code:
        warnings.append("[WARN]  Uses 'OnBalanceVolume' - should be 'OBV'")"""

NEW_CHECK = """    # HARD problem: this build has no OBV/OnBalanceVolume indicator at all.
    # Redirecting one missing name to another (the old behaviour) still crashes,
    # so flag ANY reference unless the file defines its own custom class.
    if ('OnBalanceVolume' in code or 'indicators.OBV' in code
            or 'ind.OBV' in code):
        if 'bt.Indicator' not in code:
            warnings.append("[BLOCK] References OBV/OnBalanceVolume, which does "
                            "not exist in this Backtrader build and has no "
                            "custom class defined -- variant will crash")
    for _missing in ('VWAP', 'KeltnerChannel', 'DonchianChannel'):
        if _missing in code and 'bt.Indicator' not in code:
            warnings.append(f"[BLOCK] References {_missing}, missing in this "
                            f"Backtrader build with no custom class")"""

MARKER = "DO NOT USE these -- they DO NOT EXIST in this build"


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
    backups = sorted(glob.glob(f"{TARGET}.srcfix_*.bak"))
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

    problems = []
    new_content = content
    if OLD_RULE1 in new_content:
        new_content = new_content.replace(OLD_RULE1, NEW_RULE1, 1)
    else:
        problems.append("could not find the Rule 1 block")
    if OLD_CHECK in new_content:
        new_content = new_content.replace(OLD_CHECK, NEW_CHECK, 1)
    else:
        problems.append("could not find the OBV checker block")

    if problems:
        print("  [FAIL] " + "; ".join(problems))
        print("         The file may have been edited. No changes made.")
        return 3

    if dry_run:
        print("  [DRY RUN] would rewrite Rule 1 (list only real indicators; ban")
        print("            OBV/VWAP/Keltner/Donchian/Volume Profile) and harden")
        print("            the bug-checker to BLOCK (not warn) missing-indicator")
        print("            references. No file written.")
        return 0

    backup = f"{TARGET}.srcfix_{_timestamp()}.bak"
    shutil.copy2(TARGET, backup)
    print(f"  [OK] backup written: {backup}")

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)

    if not _syntax_ok(TARGET):
        shutil.copy2(backup, TARGET)
        print("  [OK] restored from backup due to syntax failure.")
        return 4

    print(f"  [OK] patched {TARGET}: prompt Rule 1 corrected and bug-checker "
          f"hardened. Future variants won't be told to use missing indicators.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Fix the mutation source (prompt + checker) for missing "
                    "indicators")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print(" PATCH: mutation source -- prompt Rule 1 + bug-checker (OBV fix)")
    print("=" * 64)

    if args.revert:
        return revert()
    return patch(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
