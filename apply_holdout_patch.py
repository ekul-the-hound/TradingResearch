# ==============================================================================
# apply_holdout_patch.py
# ==============================================================================
# Phase 2, Item 10 -- wire the holdout guard into the single data choke point.
#
# WHY data_manager.get_data
# -------------------------
# Every asset path -- forex, crypto, indices, cached, live -- dispatches through
# get_data(). Guarding there means protection is inherited by everything that
# reads market data, including code not yet written. Guarding at each call site
# would mean the protection is only as good as the next person's memory, and the
# whole point is that memory is what fails over ten thousand automated runs.
#
# HOW
# ---
# get_data is renamed to _get_data_unguarded and a thin wrapper takes its place.
# One anchor, no touching of the dispatch logic, and the wrapper is the only
# thing that ever needs to change.
#
# The wrapper accepts an optional holdout_token. Without one, data is truncated
# at the pinned cutoff. This is deliberately the DEFAULT rather than an opt-in:
# a protection you have to remember to enable is not a protection.
#
# INERT UNTIL YOU PIN A CUTOFF
# ----------------------------
# With no ledger, the guard passes everything through unchanged and says so in
# its report. Applying this patch changes nothing about existing behaviour until
# you run:
#
#     python -c "import holdout_guard as h, data_manager as d; \
#                dm=d.DataManager(); df=dm.get_data('EUR-USD','1day'); \
#                print(h.HoldoutGuard.initialise(h.HoldoutGuard.suggest_cutoff(df.index)).report())"
#
# Pin it BEFORE the next research run, not after. A cutoff chosen once results
# already exist is a cutoff chosen with knowledge of the results.
#
# USAGE
#   python apply_holdout_patch.py --dry-run
#   python apply_holdout_patch.py
#   python apply_holdout_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

TARGET = 'data_manager.py'
DEP = 'holdout_guard.py'
BACKUP_SUFFIX = '.holdout_bak'

PATCHES = [
    {
        'name': 'Rename get_data -> _get_data_unguarded and insert the guard wrapper',
        'marker': 'HOLDOUT-GUARD-WRAPPER',
        'old': '''    def get_data(self, symbol, timeframe='1hour', max_bars=None, use_cache=True, **kwargs):
        """
        Get OHLCV data for a symbol''',
        'new': '''    def get_data(self, symbol, timeframe='1hour', max_bars=None, use_cache=True,
                 holdout_token=None, **kwargs):
        """
        HOLDOUT-GUARD-WRAPPER

        Get OHLCV data, truncated at the protected holdout cutoff.

        Every asset path in this class dispatches through here, so this is the
        one place the holdout can be enforced for all of them at once -- and for
        code that does not exist yet. Guarding at call sites instead would make
        the protection only as reliable as the next person's memory, which is
        exactly what fails across thousands of automated runs.

        Truncation is the DEFAULT. To read holdout data, obtain a token:

            token = guard.request_access('final validation', 'variant_07')
            df = dm.get_data('EUR-USD', '1hour', holdout_token=token)

        Inert until a cutoff is pinned: with no ledger the guard passes
        everything through unchanged.
        """
        df = self._get_data_unguarded(symbol, timeframe, max_bars, use_cache, **kwargs)
        try:
            import holdout_guard
            guard = holdout_guard.HoldoutGuard.load()
            return guard.enforce(df, symbol=symbol, timeframe=timeframe,
                                 token=holdout_token)
        except ImportError:
            return df
        except Exception as e:
            # A guard that fails open silently is worse than no guard, because
            # it looks like protection. Say so loudly and continue -- refusing
            # to return data at all would take the whole pipeline down over a
            # ledger problem.
            print(f"[HOLDOUT] [WARN] Guard did not run ({type(e).__name__}: {e}). "
                  f"Data returned UNPROTECTED.")
            return df

    def _get_data_unguarded(self, symbol, timeframe='1hour', max_bars=None,
                            use_cache=True, **kwargs):
        """
        Get OHLCV data for a symbol''',
    },
]

POST_CONDITIONS = [
    ('HOLDOUT-GUARD-WRAPPER', 'wrapper not inserted'),
    ('def _get_data_unguarded', 'original method not renamed'),
    ('token=holdout_token', 'token not forwarded to the guard'),
    ('holdout_token=None', 'wrapper does not accept a token'),
    ('import holdout_guard', 'guard not imported'),
]


def read_text(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    return raw.replace('\r\n', '\n'), ('\r\n' in raw)


def write_text(path, text, crlf):
    out = text.replace('\n', '\r\n') if crlf else text
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)


def verify_syntax(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def apply_patches(project_dir, dry_run=False):
    path = os.path.join(project_dir, TARGET)
    dep = os.path.join(project_dir, DEP)

    print(f"\n{'=' * 70}\nFILE: {TARGET}\n{'=' * 70}")
    if not os.path.exists(path):
        print(f"  [FAIL] Not found: {path}")
        return False
    if not os.path.exists(dep):
        print(f"  [FAIL] Missing dependency: {DEP}")
        return False
    print(f"  [DEP]   {DEP} present")

    text, crlf = read_text(path)
    applied, skipped, failed = [], [], []

    for p in PATCHES:
        if p['marker'] in text:
            skipped.append(p['name'])
            continue
        c = text.count(p['old'])
        if c == 0:
            failed.append((p['name'], 'anchor not found - file differs from the snapshot'))
            continue
        if c > 1:
            failed.append((p['name'], f'anchor matched {c} times - ambiguous, refusing'))
            continue
        text = text.replace(p['old'], p['new'], 1)
        applied.append(p['name'])

    for n in applied:
        print(f"  [APPLY] {n}")
    for n in skipped:
        print(f"  [SKIP]  {n} (already patched)")
    for n, w in failed:
        print(f"  [FAIL]  {n}\n          {w}")

    if failed:
        print("\n  Refusing to write a partial patch. File unchanged.")
        return False
    if not applied:
        print("  Nothing to write.")
        return True
    if dry_run:
        print(f"  [DRY-RUN] Would write {len(applied)} change(s). No file modified.")
        return True

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = f"{path}{BACKUP_SUFFIX}.{stamp}"
    shutil.copy2(path, backup)
    print(f"  [BACKUP] {os.path.basename(backup)}")
    write_text(path, text, crlf)

    ok, err = verify_syntax(path)
    if not ok:
        print(f"  [VERIFY] SYNTAX ERROR - {err}")
        shutil.copy2(backup, path)
        print("  [ROLLBACK] Restored")
        return False
    print("  [VERIFY] Syntax OK")

    raw, _ = read_text(path)
    problems = [msg for needle, msg in POST_CONDITIONS if needle not in raw]
    if problems:
        print("  [VERIFY] POST-CONDITIONS FAILED:")
        for p in problems:
            print(f"           - {p}")
        shutil.copy2(backup, path)
        print("  [ROLLBACK] Restored")
        return False

    print(f"  [VERIFY] Post-conditions OK ({len(POST_CONDITIONS)} checked)")
    return True


def revert(project_dir):
    path = os.path.join(project_dir, TARGET)
    bks = sorted(glob.glob(f"{path}{BACKUP_SUFFIX}.*"))
    print("\nREVERT\n" + "=" * 70)
    if not bks:
        print(f"  [SKIP] No backup for {TARGET}")
        return False
    shutil.copy2(bks[-1], path)
    print(f"  [OK] {TARGET}  <-  {os.path.basename(bks[-1])}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Wire the holdout guard into data_manager")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)
    print("=" * 70)
    print("HOLDOUT GUARD - PATCHER")
    print("=" * 70)
    print(f"Project: {project_dir}")
    if args.dry_run:
        print("Mode:    DRY RUN (no files will be modified)")

    if args.revert:
        revert(project_dir)
        return 0

    ok = apply_patches(project_dir, dry_run=args.dry_run)

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print("DRY RUN COMPLETE - re-run without --dry-run to apply")
    elif ok:
        print("PATCH COMPLETE")
        print("=" * 70)
        print("\nINERT until you pin a cutoff -- nothing changes yet.")
        print("\nNEXT:")
        print("  python test_holdout_guard.py")
        print("  python test_system.py")
        print("\nThen pin the cutoff BEFORE your next research run. Choosing it")
        print("after results exist is choosing it with knowledge of the results.")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
