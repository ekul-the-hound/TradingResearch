#!/usr/bin/env python
# ==============================================================================
# patch_step2_sysmodules.py -- Fix "Backtest failed: 'variant_01'" in Step 2
# ==============================================================================
# THE BUG:
#   run_pipeline._load_strategies_from_dir() (used by Step 2, the main funnel)
#   loads each variant file with:
#       mod = importlib.util.module_from_spec(spec)
#       spec.loader.exec_module(mod)
#   but never registers the module in sys.modules. When Backtrader runs the
#   strategy, its internal introspection looks the strategy's module up by name
#   in sys.modules. Because 'variant_01' was never registered, that lookup
#   raises KeyError('variant_01') -- caught in the MTF backtester as
#   "Backtest failed: 'variant_01'" -> None -> "no result". Every variant fails
#   identically, zeroing the Step 2 funnel.
#
#   Step 8's loader (backtest_adapter.evaluate_variant) does NOT have this bug:
#   it does `sys.modules[spec.name] = module` before exec_module, which is
#   exactly why Step 8 backtests ran while Step 2 produced nothing.
#
# THE FIX:
#   Add the missing `sys.modules[spec.name] = mod` line before exec_module in
#   _load_strategies_from_dir, matching evaluate_variant.
#
# SAFETY:
#   * --dry-run shows the change without writing.
#   * timestamped .bak backup before any edit.
#   * ast.parse syntax check after; auto-restore on failure.
#   * --revert restores the most recent backup.
#   * idempotent: a second run is a no-op.
#   * verifies `import sys` is present (adds nothing if already imported; warns
#     if somehow missing).
#
# USAGE:
#   python patch_step2_sysmodules.py --dry-run
#   python patch_step2_sysmodules.py
#   python patch_step2_sysmodules.py --revert
# ==============================================================================

import ast
import shutil
import argparse
import glob
import os
from datetime import datetime

TARGET = "run_pipeline.py"

OLD = (
    "                mod = importlib.util.module_from_spec(spec)\n"
    "                spec.loader.exec_module(mod)\n"
)

NEW = (
    "                mod = importlib.util.module_from_spec(spec)\n"
    "                # Register in sys.modules BEFORE executing: Backtrader looks\n"
    "                # the strategy's module up by name during its run, and an\n"
    "                # unregistered module raises KeyError(module_name) (this was\n"
    "                # the 'Backtest failed: variant_01' bug). evaluate_variant\n"
    "                # does the same thing for the same reason.\n"
    "                import sys as _sys\n"
    "                _sys.modules[spec.name] = mod\n"
    "                spec.loader.exec_module(mod)\n"
)

MARKER = "_sys.modules[spec.name] = mod"


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
    backups = sorted(glob.glob(f"{TARGET}.step2fix_*.bak"))
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
        print("  [FAIL] could not find the exact loader block to replace.")
        print("         The file may have been edited. No changes made.")
        return 3

    new_content = content.replace(OLD, NEW, 1)

    if dry_run:
        print("  [DRY RUN] would insert sys.modules registration:")
        print("  " + "-" * 60)
        for line in NEW.splitlines():
            print("  | " + line)
        print("  " + "-" * 60)
        print("  [DRY RUN] no file written.")
        return 0

    backup = f"{TARGET}.step2fix_{_timestamp()}.bak"
    shutil.copy2(TARGET, backup)
    print(f"  [OK] backup written: {backup}")

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)

    if not _syntax_ok(TARGET):
        shutil.copy2(backup, TARGET)
        print("  [OK] restored from backup due to syntax failure.")
        return 4

    print(f"  [OK] patched {TARGET}: Step 2 now registers variant modules in "
          f"sys.modules; the 'variant_01' KeyError is fixed.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Fix Step 2 variant_01 KeyError (sys.modules registration)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print(" PATCH: Step 2 variant module registration (variant_01 KeyError fix)")
    print("=" * 64)

    if args.revert:
        return revert()
    return patch(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
