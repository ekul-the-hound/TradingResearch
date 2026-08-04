# ==============================================================================
# apply_dashboard_ftmo_patch.py
# ==============================================================================
# Phase 0, Item 3 -- replace the dashboards' fabricated FTMO badges with real
# FTMOComplianceChecker output.
#
# Patches are INDEPENDENT. If your live react_dashboard2.py has drifted from
# the project snapshot, that file's patches will report [FAIL] and be skipped
# while the other file still gets fixed. Nothing is written partially.
#
# Requires dashboard_ftmo_panel.py beside the dashboards.
#
# USAGE
#   python apply_dashboard_ftmo_patch.py --dry-run
#   python apply_dashboard_ftmo_patch.py
#   python apply_dashboard_ftmo_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

DEP = 'dashboard_ftmo_panel.py'
BACKUP_SUFFIX = '.ftmopanel_bak'

PATCHES = [
    # -------------------------------------------------------------------------
    # react_dashboard2.py -- reads from backtest_results, which stores no trades
    # -------------------------------------------------------------------------
    {
        'file': 'react_dashboard2.py',
        'name': 'Import the real-checker panel adapter',
        'marker': 'import dashboard_ftmo_panel',
        'old': '''_load("FTMOComplianceChecker","ftmo_compliance",       "FTMOComplianceChecker")''',
        'new': '''_load("FTMOComplianceChecker","ftmo_compliance",       "FTMOComplianceChecker")
try:
    import dashboard_ftmo_panel as _ftmo_panel
except Exception as _e:
    _ftmo_panel = None
    print(f"[WARN] dashboard_ftmo_panel unavailable: {_e}")''',
    },
    {
        'file': 'react_dashboard2.py',
        'name': 'FTMO table: real checker output, no proxy badges',
        'marker': 'FTMO PROXY FIX',
        'old': '''    # FTMO
    sizes=[10000,25000,50000,100000,200000]
    ftmo_rows=[]
    if bt:
        best=max(bt,key=lambda r:r.get("total_return_pct")or 0)
        ret_pct=(best.get("total_return_pct")or 0)/100
        dd_pct=abs(best.get("max_drawdown_pct")or 0)/100
        for sz in sizes:
            final=sz*(1+ret_pct); d_ok=dd_pct<0.05; t_ok=dd_pct<0.10; tgt=ret_pct>=0.10; vrf=ret_pct>=0.05
            p=d_ok and t_ok and tgt
            ftmo_rows.append([f"${sz:,}",f"${final:,.0f}",
                _badge("PASS" if d_ok else "FAIL",T["green"] if d_ok else T["red"]),
                _badge("PASS" if t_ok else "FAIL",T["green"] if t_ok else T["red"]),
                _badge("PASS" if tgt else "FAIL",T["green"] if tgt else T["red"]),
                _badge("PASS" if p else "FAIL",T["green"] if p else T["red"])])
''',
        'new': '''    # FTMO PROXY FIX
    #
    # Was: d_ok = dd_pct < 0.05 and t_ok = dd_pct < 0.10 -- the SAME total
    # max-drawdown number tested against two thresholds, with the "Daily<5%"
    # column never looking at a daily boundary. Min-trading-days was missing
    # from the table entirely. FTMOComplianceChecker was imported but unused.
    #
    # Now: real checker output when trade-level data is reachable, and an
    # explicit unavailable state when it is not. backtest_results stores only
    # summary statistics, so the panel resolves trades from the decay DB.
    ftmo_rows = []
    ftmo_panel = None
    if bt and _ftmo_panel is not None:
        best = max(bt, key=lambda r: r.get("total_return_pct") or 0)
        ftmo_panel = _ftmo_panel.build_panel(
            decay_db_path=DB_DECAY,
            strategy_id=best.get("variant_id") or best.get("strategy_name") or "",
            symbol=best.get("symbol"),
            phase="challenge",
        )
        if ftmo_panel.available:
            for r in ftmo_panel.rows:
                cells = _ftmo_panel.row_cells(r)
                ftmo_rows.append(
                    [f"${r['account_size']:,}", f"${r['final_equity']:,.0f}"]
                    + [_badge(lbl, T["green"] if ok else T["red"]) for lbl, ok in cells]
                )
''',
    },
    {
        'file': 'react_dashboard2.py',
        'name': 'FTMO table: add Min Days column + honest empty state',
        'marker': 'Min Days>=4',
        'old': '''            html.p({"style":{"color":T["dim"],"fontSize":"12px","marginBottom":"12px"}},
                f"Based on best strategy: {bt[0].get('variant_id','?') if bt else 'N/A'}") if bt else html.div(),
            _tbl(["Account","Final Equity","Daily<5%","Total<10%","Target+10%","Overall"],ftmo_rows) if ftmo_rows else
                _empty("No backtest data","[BANK]")),''',
        'new': '''            html.p({"style":{"color":T["dim"],"fontSize":"12px","marginBottom":"12px"}},
                _ftmo_panel.caption(ftmo_panel) if (ftmo_panel is not None and _ftmo_panel is not None)
                else "FTMO compliance unavailable - panel adapter not loaded") if bt else html.div(),
            _tbl(["Account","Final Equity","Daily<5%","Total<10%","Min Days>=4","Target+10%","Overall"],ftmo_rows)
                if ftmo_rows else
                _empty(ftmo_panel.reason if ftmo_panel is not None else "No backtest data","[BANK]")),''',
    },

    # -------------------------------------------------------------------------
    # dashboard_react.py -- already builds trades_df and then ignores it
    # -------------------------------------------------------------------------
    {
        'file': 'dashboard_react.py',
        'name': 'Use the trades already in hand instead of proxies',
        'marker': 'FTMO PROXY FIX',
        'old': '''    for size in account_sizes:
        # Scale the return to the account size
        # If we made -30% on $10k, we'd make -30% on any size
        scaled_return = result.total_return_pct
        scaled_final = size * (1 + scaled_return / 100)
        scaled_dd = result.max_drawdown_pct
        
        # FTMO limits
        daily_ok = scaled_dd < 5  # 5% daily limit
        total_ok = scaled_dd < 10  # 10% total limit
        profit_ok = scaled_return >= 10  # 10% profit target
        min_days_ok = result.total_trades >= 4  # Min trading days
        
        passed = daily_ok and total_ok and profit_ok and min_days_ok
        
        results_rows.append({
            'account_size': size,
            'daily_loss_ok': daily_ok,
            'total_drawdown_ok': total_ok,
            'profit_target_ok': profit_ok,
            'min_days_ok': min_days_ok,
            'final_return_pct': scaled_return,
            'final_equity': scaled_final,
            'PASS': passed
        })
''',
        'new': '''    # FTMO PROXY FIX
    #
    # Was: daily_ok = scaled_dd < 5 (total drawdown, not daily loss) and
    # min_days_ok = result.total_trades >= 4 (trade COUNT standing in for
    # distinct trading DAYS -- four trades in one session passed a rule that
    # requires four days). trades_df was built on the line above and ignored.
    #
    # Now: the real checker runs on those trades. When it cannot run, the rows
    # list stays empty and the UI shows why rather than a fabricated verdict.
    import dashboard_ftmo_panel as _ftmo_panel

    ftmo_panel = _ftmo_panel.rows_from_trades(
        trades_df,
        phase='challenge',
        account_sizes=account_sizes,
        strategy_id=getattr(result, 'strategy_name', '') or getattr(result, 'variant_id', ''),
    )
    ftmo_unavailable_reason = '' if ftmo_panel.available else ftmo_panel.reason

    for r in ftmo_panel.rows:
        results_rows.append({
            'account_size': r['account_size'],
            'daily_loss_ok': r['daily_ok'],
            'total_drawdown_ok': r['total_ok'],
            'profit_target_ok': r['profit_ok'],
            'min_days_ok': r['min_days_ok'],
            'final_return_pct': r['final_return_pct'],
            'final_equity': r['final_equity'],
            'PASS': r['passed']
        })
''',
    },
]

# react_dashboard2.py needs a decay-DB path constant; added only if absent.
DECAY_DB_PATCH = {
    'file': 'react_dashboard2.py',
    'name': 'Define DB_DECAY path constant',
    'marker': 'DB_DECAY',
    'old': '''DB_LIN = str(BASE / "data" / "lineage.db")''',
    'new': '''DB_LIN = str(BASE / "data" / "lineage.db")
DB_DECAY = str(BASE / "data" / "decay.db")''',
}


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
    dep = os.path.join(project_dir, DEP)
    if not os.path.exists(dep):
        print(f"\n  [FAIL] Missing dependency: {DEP}")
        print(f"         Copy it into {project_dir} first.")
        return False
    print(f"\n  [DEP] {DEP} present")

    all_patches = [DECAY_DB_PATCH] + PATCHES

    by_file = {}
    for p in all_patches:
        by_file.setdefault(p['file'], []).append(p)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    any_failure = False

    for filename, patches in by_file.items():
        path = os.path.join(project_dir, filename)
        print(f"\n{'=' * 70}")
        print(f"FILE: {filename}")
        print('=' * 70)

        if not os.path.exists(path):
            print(f"  [SKIP] Not present in this project")
            continue

        text, crlf = read_text(path)
        applied, skipped, failed = [], [], []

        for p in patches:
            if p['marker'] in text:
                skipped.append(p['name'])
                continue
            count = text.count(p['old'])
            if count == 0:
                failed.append((p['name'], 'anchor not found - live file has drifted from the snapshot'))
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
            any_failure = True
            print(f"\n  Refusing to partially patch {filename}. File unchanged.")
            print(f"  Send me the current FTMO block from {filename} and I'll re-anchor.")
            continue

        if not applied:
            print("  Nothing to write.")
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would write {len(applied)} change(s). No file modified.")
            continue

        backup = f"{path}{BACKUP_SUFFIX}.{stamp}"
        shutil.copy2(path, backup)
        print(f"  [BACKUP] {os.path.basename(backup)}")

        write_text(path, text, crlf)

        ok, err = verify_syntax(path)
        if ok:
            print("  [VERIFY] Syntax OK")
        else:
            print(f"  [VERIFY] SYNTAX ERROR - {err}")
            print("  [ROLLBACK] Restoring from backup")
            shutil.copy2(backup, path)
            any_failure = True

    return not any_failure


def revert(project_dir):
    print("\nREVERT")
    print("=" * 70)
    done = False
    for filename in sorted({p['file'] for p in [DECAY_DB_PATCH] + PATCHES}):
        path = os.path.join(project_dir, filename)
        backups = sorted(glob.glob(f"{path}{BACKUP_SUFFIX}.*"))
        if not backups:
            print(f"  [SKIP] No backup for {filename}")
            continue
        shutil.copy2(backups[-1], path)
        print(f"  [OK] {filename}  <-  {os.path.basename(backups[-1])}")
        done = True
    if not done:
        print("\n  Nothing to revert.")
    return done


def main():
    ap = argparse.ArgumentParser(description="Replace dashboard FTMO proxies with real checker output")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)

    print("=" * 70)
    print("DASHBOARD FTMO PROXY FIX - PATCHER")
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
        print("  python test_dashboard_ftmo_panel.py")
        print("  then start the dashboard and open the FTMO tab")
        print("\nEXPECT: the table may now say 'unavailable' instead of showing")
        print("badges. That is correct. backtest_results stores no trade list,")
        print("so real compliance needs trades persisted via")
        print("DecayCalculator.save_trades(). An honest blank beats a fake PASS.")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
