# ==============================================================================
# apply_trade_persistence_patch.py
# ==============================================================================
# Closes the trade-persistence gap -- item 11.4 in the session report, and the
# shared root cause behind two separate subsystems degrading to "unavailable".
#
# THE GAP
# -------
# backtest_results stores summary statistics only: total_trades is a COUNT, not
# a list. Two consequences followed from that single omission:
#
#   1. The dashboard's FTMO panel could not run FTMOComplianceChecker, because
#      the checker needs entry/exit dates and prices. That is why the proxy
#      badges existed in the first place -- they were a workaround for missing
#      data, not carelessness.
#
#   2. canonical_result had a synthetic-returns fallback for exactly the case
#      "summary statistics present, trade list absent". Removing the fabrication
#      (Phase 0 item 5) was correct, but it left those results with returns=None
#      rather than with real returns.
#
# Both are symptoms. This is the disease.
#
# WHAT MAKES THIS CHEAP
# ---------------------
# TradeTracker in backtester_multi_timeframe.py already produces exactly the
# fields the compliance checker needs:
#
#   entry_date, exit_date, entry_price, exit_price, size, pnl, return_pct,
#   duration_bars, is_long
#
# They are computed on every backtest and then discarded at the database
# boundary. Persisting them requires no new computation and makes compliance
# checking EXACT -- no price reconstruction, no approximation flag, no
# asset-class caveat for notional-based fees.
#
# CHANGES
#   database.py  1. backtest_trades table + indices
#                2. save_backtest() persists result['trades']
#                3. get_trades() / get_latest_trades() readers
#   react_dashboard2.py
#                4. FTMO panel reads the results DB first (exact), decay DB second
#
# Storage cost: roughly 120 bytes per trade. A 1,000-trade backtest adds ~120 KB.
#
# USAGE
#   python apply_trade_persistence_patch.py --dry-run
#   python apply_trade_persistence_patch.py
#   python apply_trade_persistence_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

BACKUP_SUFFIX = '.tradepersist_bak'

PATCHES = [
    {
        'file': 'database.py',
        'name': 'backtest_trades table + indices',
        'marker': 'TRADE-PERSISTENCE-SCHEMA',
        'old': '''        # Also create the old table name for backwards compatibility''',
        'new': '''        # TRADE-PERSISTENCE-SCHEMA
        # backtest_results holds summary statistics only -- total_trades is a
        # count. Without the underlying trades, FTMOComplianceChecker cannot
        # run at all (it needs entry/exit dates and prices), which is why the
        # dashboard fell back to proxy badges and why canonical_result had a
        # synthetic-returns branch. The trades are already computed by
        # TradeTracker on every backtest and thrown away here; this keeps them.
        cursor.execute(\'\'\'
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                strategy_name TEXT,
                variant_id TEXT,
                symbol TEXT,
                timeframe TEXT,
                entry_date TEXT,
                exit_date TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                return_pct REAL,
                duration_bars INTEGER,
                is_long INTEGER,
                FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
            )
        \'\'\')
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_bt_trades_backtest "
            "ON backtest_trades(backtest_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_bt_trades_lookup "
            "ON backtest_trades(variant_id, symbol, timeframe)")

        # Also create the old table name for backwards compatibility''',
    },
    {
        'file': 'database.py',
        'name': 'save_backtest persists the trade list',
        'marker': 'TRADE-PERSISTENCE-SAVE',
        'old': '''        backtest_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return backtest_id''',
        'new': '''        backtest_id = cursor.lastrowid

        # TRADE-PERSISTENCE-SAVE
        # Persist the trade list alongside the summary. Wrapped so that a
        # malformed trade record can never lose the backtest row itself --
        # a partial save is much better than a rolled-back result.
        trades = result.get('trades') or []
        if trades:
            try:
                rows = []
                for t in trades:
                    if not isinstance(t, dict):
                        continue
                    rows.append((
                        backtest_id,
                        result.get('strategy_name'),
                        result.get('variant_id'),
                        result.get('symbol'),
                        result.get('timeframe'),
                        str(t.get('entry_date')) if t.get('entry_date') is not None else None,
                        str(t.get('exit_date')) if t.get('exit_date') is not None else None,
                        t.get('entry_price'),
                        t.get('exit_price'),
                        t.get('size'),
                        t.get('pnl'),
                        t.get('return_pct'),
                        t.get('duration_bars'),
                        1 if t.get('is_long') else 0,
                    ))
                if rows:
                    cursor.executemany(\'\'\'
                        INSERT INTO backtest_trades (
                            backtest_id, strategy_name, variant_id, symbol, timeframe,
                            entry_date, exit_date, entry_price, exit_price, size,
                            pnl, return_pct, duration_bars, is_long
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    \'\'\', rows)
            except Exception as e:
                print(f"[WARN] Could not persist {len(trades)} trades for "
                      f"backtest {backtest_id}: {type(e).__name__}: {e}")

        conn.commit()
        conn.close()
        
        return backtest_id

    def get_trades(self, backtest_id):
        """
        Trade list for one backtest, as a list of dicts matching the shape
        FTMOComplianceChecker expects. Empty list if none were persisted
        (results saved before this table existed have none).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM backtest_trades WHERE backtest_id = ? "
                "ORDER BY exit_date ASC", (backtest_id,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            conn.close()

    def get_latest_trades(self, variant_id=None, strategy_name=None,
                          symbol=None, timeframe=None):
        """
        Trades from the most recent matching backtest.

        Returns (trades, backtest_id). Scoped to a single backtest deliberately:
        concatenating trades across runs would splice unrelated equity paths
        together and produce a meaningless daily-loss sequence.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            q = "SELECT id FROM backtest_results WHERE 1=1"
            p = []
            if variant_id:
                q += " AND variant_id = ?"; p.append(variant_id)
            if strategy_name:
                q += " AND strategy_name = ?"; p.append(strategy_name)
            if symbol:
                q += " AND symbol = ?"; p.append(symbol)
            if timeframe:
                q += " AND timeframe = ?"; p.append(timeframe)
            q += " ORDER BY id DESC LIMIT 1"

            row = conn.execute(q, p).fetchone()
            if not row:
                return [], None
            bid = row['id']
            trades = conn.execute(
                "SELECT * FROM backtest_trades WHERE backtest_id = ? "
                "ORDER BY exit_date ASC", (bid,)).fetchall()
            return [dict(t) for t in trades], bid
        except Exception:
            return [], None
        finally:
            conn.close()''',
    },
    {
        'file': 'react_dashboard2.py',
        'name': 'FTMO panel: prefer the results DB (exact) over the decay DB',
        'marker': 'results_db_path=DB_BT',
        'old': '''        ftmo_panel = _ftmo_panel.build_panel(
            decay_db_path=DB_DECAY,
            strategy_id=best.get("variant_id") or best.get("strategy_name") or "",
            symbol=best.get("symbol"),
            phase="challenge",
        )''',
        'new': '''        ftmo_panel = _ftmo_panel.build_panel(
            # Results DB first: it stores entry AND exit prices, so compliance
            # is exact. The decay DB has no prices, so fees for notional-fee
            # asset classes are only approximate there.
            results_db_path=DB_BT,
            decay_db_path=DB_DECAY,
            strategy_id=best.get("variant_id") or best.get("strategy_name") or "",
            strategy_name=best.get("strategy_name"),
            symbol=best.get("symbol"),
            timeframe=best.get("timeframe"),
            phase="challenge",
        )''',
    },
]

POST_CONDITIONS = [
    ('database.py', 'TRADE-PERSISTENCE-SCHEMA', 'schema not added'),
    ('database.py', 'TRADE-PERSISTENCE-SAVE', 'save path not patched'),
    ('database.py', 'def get_trades', 'reader not added'),
    ('database.py', 'def get_latest_trades', 'lookup reader not added'),
    ('react_dashboard2.py', 'results_db_path=DB_BT', 'dashboard not rewired'),
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
    by_file = {}
    for p in PATCHES:
        by_file.setdefault(p['file'], []).append(p)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backups, failed_any = {}, False

    for filename, patches in by_file.items():
        path = os.path.join(project_dir, filename)
        print(f"\n{'=' * 70}\nFILE: {filename}\n{'=' * 70}")

        if not os.path.exists(path):
            print(f"  [FAIL] Not found: {path}")
            failed_any = True
            continue

        text, crlf = read_text(path)
        applied, skipped, failed = [], [], []

        for p in patches:
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
            print(f"\n  Refusing to partially patch {filename}. File unchanged.")
            failed_any = True
            continue
        if not applied:
            print("  Nothing to write.")
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would write {len(applied)} change(s). No file modified.")
            continue

        backup = f"{path}{BACKUP_SUFFIX}.{stamp}"
        shutil.copy2(path, backup)
        backups[filename] = backup
        print(f"  [BACKUP] {os.path.basename(backup)}")
        write_text(path, text, crlf)

        ok, err = verify_syntax(path)
        if ok:
            print("  [VERIFY] Syntax OK")
        else:
            print(f"  [VERIFY] SYNTAX ERROR - {err}")
            shutil.copy2(backup, path)
            print("  [ROLLBACK] Restored")
            failed_any = True

    if dry_run or failed_any:
        return not failed_any

    problems = []
    for filename, needle, msg in POST_CONDITIONS:
        path = os.path.join(project_dir, filename)
        if os.path.exists(path):
            raw, _ = read_text(path)
            if needle not in raw:
                problems.append(f"{filename}: {msg}")

    print(f"\n{'=' * 70}")
    if problems:
        print("  [VERIFY] POST-CONDITIONS FAILED:")
        for p in problems:
            print(f"           - {p}")
        for f, b in backups.items():
            shutil.copy2(b, os.path.join(project_dir, f))
            print(f"  [ROLLBACK] {f}")
        return False

    print(f"  [VERIFY] Post-conditions OK ({len(POST_CONDITIONS)} checked)")
    return True


def revert(project_dir):
    print("\nREVERT\n" + "=" * 70)
    done = False
    for filename in sorted({p['file'] for p in PATCHES}):
        path = os.path.join(project_dir, filename)
        bks = sorted(glob.glob(f"{path}{BACKUP_SUFFIX}.*"))
        if not bks:
            print(f"  [SKIP] No backup for {filename}")
            continue
        shutil.copy2(bks[-1], path)
        print(f"  [OK] {filename}  <-  {os.path.basename(bks[-1])}")
        done = True
    if not done:
        print("\n  Nothing to revert.")
    return done


def main():
    ap = argparse.ArgumentParser(description="Persist trade lists in the results database")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)
    print("=" * 70)
    print("TRADE PERSISTENCE - PATCHER")
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
        print("\nThe table is created on next DatabaseManager init. Existing rows")
        print("have no trades, so the FTMO panel stays unavailable for them --")
        print("re-run any backtest you want real compliance numbers for.")
        print("\nNEXT:")
        print("  python test_trade_persistence.py")
        print("  python run_backtests.py        (populates the new table)")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
