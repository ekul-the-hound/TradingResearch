# ==============================================================================
# apply_tz_fix_patch.py
# ==============================================================================
# Phase 0, Item 1 -- HistData timezone audit.
#
# Applies the downstream half of the timezone fix. The upstream half is the
# drop-in forex_data_processor.py, which now converts HistData's EST-fixed
# stamps to UTC at ingest and writes {ticker}_1min_utc.csv.
#
# This patcher makes data_manager.py read that file, refuse to silently fall
# back to the pre-fix {ticker}_1min_merged.csv, and documents the naive-UTC
# invariant that ftmo_compliance.to_prague_time() has always relied on but
# never stated.
#
# PATCHES
#   1. data_manager.py    -- base file -> _1min_utc.csv, with a hard legacy guard
#   2. data_manager.py    -- document the naive-UTC contract at the read site
#   3. ftmo_compliance.py -- replace "Assume UTC" comment with the real invariant
#
# USAGE
#   python apply_tz_fix_patch.py --dry-run     # show what would change
#   python apply_tz_fix_patch.py               # apply (timestamped backups)
#   python apply_tz_fix_patch.py --revert      # restore most recent backups
#
# Safe to re-run: already-patched files are detected and skipped.
# CRLF line endings are preserved.
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

# ==============================================================================
# PATCH DEFINITIONS
# ==============================================================================

PATCHES = [
    {
        'file': 'data_manager.py',
        'name': 'Forex base file -> _1min_utc.csv (with legacy guard)',
        'marker': '_1min_utc.csv',
        'old': '''        # Load base 1-minute merged data
        base_file = os.path.join(config.CACHE_SUBDIRS['forex'], f"{ticker}_1min_merged.csv")
        
        if not os.path.exists(base_file):
            print(f"[FAIL] Missing data: {symbol} (merged file not found)")
            print(f"   Expected: {base_file}")
            print(f"   Run: python forex_data_processor.py first")
            return None
''',
        'new': '''        # Load base 1-minute merged data
        #
        # TIMEZONE CONTRACT: this file must be naive UTC. HistData.com publishes
        # in EST-fixed (UTC-5, no DST); forex_data_processor.py converts at
        # ingest and writes the _1min_utc.csv marker to prove it did.
        #
        # The legacy _1min_merged.csv was written WITHOUT that conversion. Its
        # timestamps are 5 hours behind the real instant, which moves the
        # Prague-midnight daily reset used by ftmo_compliance into the middle
        # of the Tokyo session. We refuse it rather than fall back to it -- a
        # loud failure here is far cheaper than a quietly wrong pass rate.
        base_file = os.path.join(config.CACHE_SUBDIRS['forex'], f"{ticker}_1min_utc.csv")
        legacy_file = os.path.join(config.CACHE_SUBDIRS['forex'], f"{ticker}_1min_merged.csv")
        
        if not os.path.exists(base_file):
            if os.path.exists(legacy_file):
                print(f"[FAIL] Stale pre-timezone-fix data for {symbol}")
                print(f"   Found:    {legacy_file}")
                print(f"   Expected: {base_file}")
                print(f"   The legacy file is on HistData's EST clock, not UTC.")
                print(f"   Every FTMO daily-loss number built from it is 5h off.")
                print(f"   Rebuild: python forex_data_processor.py")
                return None
            print(f"[FAIL] Missing data: {symbol} (merged file not found)")
            print(f"   Expected: {base_file}")
            print(f"   Run: python forex_data_processor.py first")
            return None
''',
    },
    {
        'file': 'data_manager.py',
        'name': 'Document naive-UTC contract at the forex read site',
        'marker': 'already naive UTC on disk',
        'old': '''            # Load base data
            df = pd.read_csv(base_file, index_col=0, parse_dates=True)
            
            # Normalize timezone
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
''',
        'new': '''            # Load base data
            df = pd.read_csv(base_file, index_col=0, parse_dates=True)
            
            # Normalize timezone.
            # Post-fix this is a no-op: the file is already naive UTC on disk.
            # Kept as a guard in case a tz-aware CSV is ever hand-placed here.
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
''',
    },
    {
        'file': 'ftmo_compliance.py',
        'name': 'State the naive-UTC invariant instead of assuming it',
        'marker': 'INVARIANT: naive timestamps are UTC',
        'old': '''def to_prague_time(dt: datetime) -> datetime:
    """Convert datetime to Prague timezone"""
    if dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = UTC_TZ.localize(dt)
    return dt.astimezone(PRAGUE_TZ)
''',
        'new': '''def to_prague_time(dt: datetime) -> datetime:
    """
    Convert datetime to Prague timezone.

    INVARIANT: naive timestamps are UTC.

    This is enforced upstream, not assumed here. Each data source is
    responsible for normalising to naive UTC before its trades reach this
    module:
      - Forex  : forex_data_processor.py converts HistData EST-fixed (UTC-5,
                 no DST) -> UTC at ingest and marks the file _1min_utc.csv.
      - Crypto : CCXT returns epoch milliseconds, which pandas parses as UTC.
      - Indices: NOT YET AUDITED. Kaggle equity files are typically US market
                 local time WITH daylight saving, which would be a different
                 bug from the forex one. Do not assume these are UTC.

    If that invariant breaks, the Prague-midnight daily reset lands in the
    wrong place and every max-daily-loss check silently measures the wrong
    24-hour window.
    """
    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt)
    return dt.astimezone(PRAGUE_TZ)
''',
    },
]

BACKUP_SUFFIX = '.tzfix_bak'


# ==============================================================================
# IO HELPERS (CRLF-preserving)
# ==============================================================================

def read_text(path):
    """Read file, return (normalized_text, used_crlf)."""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    used_crlf = '\r\n' in raw
    return raw.replace('\r\n', '\n'), used_crlf


def write_text(path, text, used_crlf):
    """Write file, restoring CRLF if the original used it."""
    out = text.replace('\n', '\r\n') if used_crlf else text
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)


def verify_syntax(path):
    """Parse the file to confirm the patch did not break it."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


# ==============================================================================
# CORE
# ==============================================================================

def apply_patches(project_dir, dry_run=False):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Group by file so each file is read/written once.
    by_file = {}
    for p in PATCHES:
        by_file.setdefault(p['file'], []).append(p)

    overall_ok = True

    for filename, patches in by_file.items():
        path = os.path.join(project_dir, filename)
        print(f"\n{'=' * 70}")
        print(f"FILE: {filename}")
        print('=' * 70)

        if not os.path.exists(path):
            print(f"  [FAIL] Not found: {path}")
            overall_ok = False
            continue

        text, used_crlf = read_text(path)
        original = text
        applied, skipped, failed = [], [], []

        for p in patches:
            # Idempotency: marker already present means this patch is in.
            if p['marker'] in text:
                skipped.append(p['name'])
                continue

            count = text.count(p['old'])
            if count == 0:
                failed.append((p['name'], 'anchor not found - file may already differ from the snapshot'))
                continue
            if count > 1:
                failed.append((p['name'], f'anchor matched {count} times - ambiguous, refusing'))
                continue

            text = text.replace(p['old'], p['new'], 1)
            applied.append(p['name'])

        for name in applied:
            print(f"  [APPLY] {name}")
        for name in skipped:
            print(f"  [SKIP]  {name} (already patched)")
        for name, why in failed:
            print(f"  [FAIL]  {name}")
            print(f"          {why}")
            overall_ok = False

        if not applied:
            print("  Nothing to write.")
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would write {len(applied)} change(s). No file modified.")
            continue

        # Backup, write, verify, auto-rollback on syntax break.
        backup = f"{path}{BACKUP_SUFFIX}.{stamp}"
        shutil.copy2(path, backup)
        print(f"  [BACKUP] {os.path.basename(backup)}")

        write_text(path, text, used_crlf)

        ok, err = verify_syntax(path)
        if ok:
            print(f"  [VERIFY] Syntax OK")
        else:
            print(f"  [VERIFY] SYNTAX ERROR - {err}")
            print(f"  [ROLLBACK] Restoring from backup")
            shutil.copy2(backup, path)
            overall_ok = False

    return overall_ok


def revert(project_dir):
    print("\nREVERT - restoring most recent backups")
    print("=" * 70)

    files = {p['file'] for p in PATCHES}
    any_done = False

    for filename in sorted(files):
        path = os.path.join(project_dir, filename)
        backups = sorted(glob.glob(f"{path}{BACKUP_SUFFIX}.*"))
        if not backups:
            print(f"  [SKIP] No backup for {filename}")
            continue
        newest = backups[-1]
        shutil.copy2(newest, path)
        print(f"  [OK] {filename}  <-  {os.path.basename(newest)}")
        any_done = True

    if not any_done:
        print("\n  Nothing to revert.")
    return any_done


def main():
    ap = argparse.ArgumentParser(description="Apply the HistData timezone fix to data_manager / ftmo_compliance")
    ap.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    ap.add_argument('--revert', action='store_true', help='Restore most recent backups')
    ap.add_argument('--dir', default='.', help='Project directory (default: current)')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)

    print("=" * 70)
    print("HISTDATA TIMEZONE FIX - PATCHER")
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
        print("\nREQUIRED NEXT STEPS - the fix is not live until these run:")
        print("  1. Drop in the new forex_data_processor.py")
        print("  2. python forex_data_processor.py       (rebuild + clear stale caches)")
        print("  3. python verify_histdata_timezone.py   (confirm UTC)")
        print("  4. python test_system.py")
        print("\nAny FTMO result produced before step 2 used a 5h-shifted daily")
        print("boundary. Treat those pass rates as void, not as a baseline.")
    else:
        print("PATCH INCOMPLETE - see failures above. Nothing partially written;")
        print("any file that failed syntax verification was rolled back.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
