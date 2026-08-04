# ==============================================================================
# apply_daily_anchor_patch.py
# ==============================================================================
# Phase 0, Item 2 -- daily-loss anchor divergence.
#
# Rewires ftmo_compliance.py onto the corrected anchor implemented in
# ftmo_daily_anchor.py. Four surgical edits:
#
#   1. import ftmo_daily_anchor
#   2. _build_equity_curve          -- checkpoints at Prague midnight, not UTC
#   3. _build_intraday_equity_curve -- same
#   4. _calculate_daily_stats       -- delegate to the balance-anchored version
#
# ftmo_daily_anchor.py must sit beside ftmo_compliance.py before running this.
#
# USAGE
#   python apply_daily_anchor_patch.py --dry-run
#   python apply_daily_anchor_patch.py
#   python apply_daily_anchor_patch.py --revert
#
# Safe to re-run. CRLF preserved. Syntax verified with auto-rollback.
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

TARGET = 'ftmo_compliance.py'
DEP = 'ftmo_daily_anchor.py'
BACKUP_SUFFIX = '.anchor_bak'

PATCHES = [
    {
        'name': 'Import the corrected anchor module',
        'marker': 'import ftmo_daily_anchor',
        'old': '''PRAGUE_TZ = pytz.timezone('Europe/Prague')
UTC_TZ = pytz.UTC
''',
        'new': '''import ftmo_daily_anchor

PRAGUE_TZ = pytz.timezone('Europe/Prague')
UTC_TZ = pytz.UTC
''',
    },
    {
        'name': 'Event curve: checkpoints at Prague midnight (was 00:00 UTC)',
        'marker': 'ANCHOR FIX (event curve)',
        'old': '''        # Inject daily checkpoints so every calendar day with an open position
        # gets an equity observation (multi-day holds previously skipped days --
        # no daily-loss check exactly where weekend gaps bite)
        if events:
            day = pd.Timestamp(events[0]['timestamp']).normalize() + pd.Timedelta(days=1)
            end_ts = pd.Timestamp(events[-1]['timestamp'])
            checkpoints = []
            while day < end_ts:
                checkpoints.append({'timestamp': day, 'event': 'checkpoint'})
                day += pd.Timedelta(days=1)
            events = sorted(events + checkpoints,
                            key=lambda x: (x['timestamp'], x['event'] == 'exit'))
''',
        'new': '''        # Inject daily checkpoints so every calendar day with an open position
        # gets an equity observation (multi-day holds previously skipped days --
        # no daily-loss check exactly where weekend gaps bite)
        #
        # ANCHOR FIX (event curve): these used to be placed with
        # pd.Timestamp.normalize(), i.e. 00:00 UTC, while _calculate_daily_stats
        # groups by PRAGUE date. Prague midnight is 23:00 UTC under CET and
        # 22:00 UTC under CEST, so the "day start" observation actually sat 1-2
        # hours into the trading day and moved by an hour twice a year. FTMO
        # recalculates at midnight CE(S)T, so that is where the checkpoint goes.
        if events:
            checkpoints = [
                {'timestamp': ts, 'event': 'checkpoint'}
                for ts in ftmo_daily_anchor.prague_midnight_checkpoints(
                    events[0]['timestamp'], events[-1]['timestamp']
                )
            ]
            events = sorted(events + checkpoints,
                            key=lambda x: (x['timestamp'], x['event'] == 'exit'))
''',
    },
    {
        'name': 'Intraday curve: checkpoints at Prague midnight (was 00:00 UTC)',
        'marker': 'ANCHOR FIX (intraday curve)',
        'old': '''        # Inject daily checkpoints so every calendar day with an open position
        # gets an equity observation (multi-day holds previously skipped days)
        if events:
            day = pd.Timestamp(events[0]['timestamp']).normalize() + pd.Timedelta(days=1)
            end_ts = pd.Timestamp(events[-1]['timestamp'])
            cps = []
            while day < end_ts:
                cps.append({'timestamp': day, 'type': 'checkpoint'})
                day += pd.Timedelta(days=1)
            events = sorted(events + cps, key=lambda x: x['timestamp'])
''',
        'new': '''        # Inject daily checkpoints so every calendar day with an open position
        # gets an equity observation (multi-day holds previously skipped days)
        #
        # ANCHOR FIX (intraday curve): see the note in _build_equity_curve.
        # Checkpoints now land on true Prague midnight, DST-correct.
        if events:
            cps = [
                {'timestamp': ts, 'type': 'checkpoint'}
                for ts in ftmo_daily_anchor.prague_midnight_checkpoints(
                    events[0]['timestamp'], events[-1]['timestamp']
                )
            ]
            events = sorted(events + cps, key=lambda x: x['timestamp'])
''',
    },
    {
        'name': 'Daily stats: balance anchor at Prague midnight (was equity at first event)',
        'marker': 'ANCHOR FIX (daily stats)',
        'old': '''        df = equity_curve.copy()
        
        # Convert to Prague timezone and extract date
        df['prague_time'] = df['timestamp'].apply(to_prague_time)
        df['prague_date'] = df['prague_time'].apply(lambda x: x.date())
        
        # Group by Prague date
        daily_stats = []
        
        for date, group in df.groupby('prague_date'):
            start_equity = group['equity'].iloc[0]
            end_equity = group['equity'].iloc[-1]
            min_equity = group['equity'].min()
            max_equity = group['equity'].max()
            
            # Daily loss is the worst drawdown from the START of the day's equity
            # FTMO rule: 5% of INITIAL BALANCE, not current balance
            daily_low_from_start = start_equity - min_equity
            daily_loss_pct = daily_low_from_start / initial_balance * 100
            
            daily_stats.append({
                'date': date,
                'start_equity': start_equity,
                'end_equity': end_equity,
                'min_equity': min_equity,
                'max_equity': max_equity,
                'daily_pnl': end_equity - start_equity,
                'daily_loss_from_start': daily_low_from_start,
                'daily_loss_pct': daily_loss_pct
            })
        
        return pd.DataFrame(daily_stats)
''',
        'new': '''        # ANCHOR FIX (daily stats)
        #
        # Was: start_equity = group['equity'].iloc[0], i.e. the first EVENT
        # inside the Prague day, measured on equity.
        #
        # FTMO recalculates the limit at midnight CE(S)T from the account
        # BALANCE ("Intraday changes resulting from open positions do not
        # affect the Maximum Daily Loss Limit"), and compares EQUITY against
        # it. Using equity on both sides meant that carrying a floating loss
        # across midnight lowered the anchor, shrank the measured daily loss,
        # and hid real breaches -- wrong in the optimistic direction.
        #
        # Column contract is preserved; anchor_balance, daily_loss_limit,
        # breached and anchor_source are added for diagnostics.
        return ftmo_daily_anchor.calculate_daily_stats_anchored(
            equity_curve,
            initial_balance,
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
        )
''',
    },
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

    print(f"\n{'=' * 70}")
    print(f"FILE: {TARGET}")
    print('=' * 70)

    if not os.path.exists(path):
        print(f"  [FAIL] Not found: {path}")
        return False

    if not os.path.exists(dep):
        print(f"  [FAIL] Missing dependency: {DEP}")
        print(f"         Copy ftmo_daily_anchor.py into {project_dir} first.")
        return False
    print(f"  [DEP]   {DEP} present")

    text, crlf = read_text(path)
    applied, skipped, failed = [], [], []

    for p in PATCHES:
        if p['marker'] in text:
            skipped.append(p['name'])
            continue
        count = text.count(p['old'])
        if count == 0:
            failed.append((p['name'], 'anchor not found - file differs from the snapshot'))
            continue
        if count > 1:
            failed.append((p['name'], f'anchor matched {count} times - ambiguous, refusing'))
            continue
        text = text.replace(p['old'], p['new'], 1)
        applied.append(p['name'])

    for n in applied:
        print(f"  [APPLY] {n}")
    for n in skipped:
        print(f"  [SKIP]  {n} (already patched)")
    for n, why in failed:
        print(f"  [FAIL]  {n}\n          {why}")

    if failed:
        print("\n  Refusing to write a partial patch. No file modified.")
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
    if ok:
        print("  [VERIFY] Syntax OK")
        return True

    print(f"  [VERIFY] SYNTAX ERROR - {err}")
    print("  [ROLLBACK] Restoring from backup")
    shutil.copy2(backup, path)
    return False


def revert(project_dir):
    path = os.path.join(project_dir, TARGET)
    backups = sorted(glob.glob(f"{path}{BACKUP_SUFFIX}.*"))
    print("\nREVERT")
    print("=" * 70)
    if not backups:
        print(f"  [SKIP] No backup for {TARGET}")
        return False
    shutil.copy2(backups[-1], path)
    print(f"  [OK] {TARGET}  <-  {os.path.basename(backups[-1])}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Apply the FTMO daily-loss anchor fix")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)

    print("=" * 70)
    print("FTMO DAILY-LOSS ANCHOR FIX - PATCHER")
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
        print("\nNEXT:")
        print("  python test_daily_anchor.py       (proves the fix, no market data needed)")
        print("  python test_ftmo_compliance.py    (existing suite must still pass)")
        print("\nExpect some existing daily-loss numbers to MOVE. That is the point.")
        print("Days where a position was carried across midnight will change most.")
    else:
        print("PATCH FAILED - see above. File unchanged or rolled back.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
