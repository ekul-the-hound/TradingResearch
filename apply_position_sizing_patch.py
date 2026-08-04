# ==============================================================================
# apply_position_sizing_patch.py
# ==============================================================================
# Phase 0, Item 4 -- replace the inert 2%/2% sizing in execution_engine.py with
# risk sizing driven by the strategy's actual stop.
#
# Five edits:
#   1. import position_sizing
#   2. ExecutionConfig: add risk_per_trade_pct + max_leverage_pct
#   3. ExecutionEngine.__init__: attach a VolatilityTracker
#   4. process_signal: accept and forward stop_price / stop_distance
#   5. _calculate_size: delegate to position_sizing.size_position
#
# Backward compatible: process_signal's new stop arguments are keyword-only
# with None defaults, so existing callers keep working (and get the
# volatility-based fallback plus a warning instead of a silent wrong number).
#
# Requires position_sizing.py beside execution_engine.py.
#
# USAGE
#   python apply_position_sizing_patch.py --dry-run
#   python apply_position_sizing_patch.py
#   python apply_position_sizing_patch.py --revert
# ==============================================================================

import argparse
import ast
import glob
import os
import shutil
import sys
from datetime import datetime

TARGET = 'execution_engine.py'
DEP = 'position_sizing.py'
BACKUP_SUFFIX = '.sizing_bak'

PATCHES = [
    {
        'name': 'Import the sizing module',
        'marker': 'import position_sizing',
        'old': '''@dataclass
class ExecutionConfig:
    """Execution configuration"""''',
        'new': '''import position_sizing


@dataclass
class ExecutionConfig:
    """Execution configuration"""''',
    },
    {
        'name': 'ExecutionConfig: real risk knobs',
        'marker': 'risk_per_trade_pct',
        'old': '''    # Risk limits
    max_position_size: float = 100000
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0''',
        'new': '''    # Risk limits
    max_position_size: float = 100000
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0

    # Position sizing.
    # risk_per_trade_pct is the fraction of equity lost if price reaches the
    # stop. Default 0.5%: FTMO's binding constraint is a 5% daily loss, so at
    # 2% per trade three consecutive losers end a challenge. Set explicitly.
    risk_per_trade_pct: float = 0.005
    max_leverage_pct: float = 0.20''',
    },
    {
        'name': 'Engine: attach volatility tracker for fallback stops',
        'marker': 'self.vol_tracker',
        'old': '''        self.is_running = False
        self.signal_queue = queue.Queue()''',
        'new': '''        self.is_running = False
        self.signal_queue = queue.Queue()

        # Feeds the fallback stop when a strategy supplies none. Built from
        # the prices process_signal already receives -- no new data source.
        self.vol_tracker = position_sizing.VolatilityTracker()
        self.last_sizing = None  # SizingResult of the most recent sizing call''',
    },
    {
        'name': 'process_signal: accept and forward the strategy stop',
        'marker': 'SIZING FIX',
        'old': '''        # Update price
        self.trader.update_price(symbol, price, timestamp)
        
        # Get current position
        current_pos = self.trader.positions.get(symbol)
        current_side = current_pos.side if current_pos else PositionSide.FLAT''',
        'new': '''        # SIZING FIX: track realised volatility so a missing strategy stop
        # falls back to something volatility-aware instead of a flat 2%.
        self.vol_tracker.update(symbol, price)

        # Update price
        self.trader.update_price(symbol, price, timestamp)
        
        # Get current position
        current_pos = self.trader.positions.get(symbol)
        current_side = current_pos.side if current_pos else PositionSide.FLAT''',
    },
    {
        'name': 'Long entry: pass stop through',
        'marker': 'stop_price=stop_price, stop_distance=stop_distance)\n                self.trader.submit_order(symbol, \'BUY\'',
        'old': '''                order_size = size or self._calculate_size(symbol, price)
                self.trader.submit_order(symbol, 'BUY', order_size)''',
        'new': '''                order_size = size or self._calculate_size(
                    symbol, price, stop_price=stop_price, stop_distance=stop_distance)
                self.trader.submit_order(symbol, 'BUY', order_size)''',
    },
    {
        'name': 'Short entry: pass stop through',
        'marker': 'stop_price=stop_price, stop_distance=stop_distance)\n                self.trader.submit_order(symbol, \'SELL\'',
        'old': '''                order_size = size or self._calculate_size(symbol, price)
                self.trader.submit_order(symbol, 'SELL', order_size)''',
        'new': '''                order_size = size or self._calculate_size(
                    symbol, price, stop_price=stop_price, stop_distance=stop_distance)
                self.trader.submit_order(symbol, 'SELL', order_size)''',
    },
    {
        'name': '_calculate_size: real risk sizing (was algebraically inert)',
        'marker': 'SIZING FIX: the old implementation',
        'old': '''    def _calculate_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk parameters"""
        # Simple fixed fractional sizing
        risk_pct = 0.02  # 2% risk per trade
        risk_amount = self.trader.equity * risk_pct
        
        # Assume 2% stop loss
        stop_distance = price * 0.02
        
        size = risk_amount / stop_distance
        
        # Apply limits
        size = min(size, self.config.max_position_size)
        
        # Ensure position value doesn't exceed 20% of equity (leverage limit)
        max_position_value = self.trader.equity * 0.20
        max_size_by_value = max_position_value / price if price > 0 else 0
        size = min(size, max_size_by_value)
        ''',
        'new': '''    def _calculate_size(self, symbol: str, price: float,
                        stop_price: float = None,
                        stop_distance: float = None) -> float:
        """
        Calculate position size from the distance to the actual stop.

        SIZING FIX: the old implementation was
            risk_amount   = equity * 0.02
            stop_distance = price  * 0.02
            size          = risk_amount / stop_distance
        in which the two 2%s cancel to size = equity / price. It carried no
        risk parameter at all, and the 20% leverage cap (always 5x smaller)
        bound every single call -- so every position was exactly 20% of equity
        notional on every instrument. Changing risk_pct did nothing.

        Now: size = (equity * risk_per_trade_pct) / distance_to_stop, with the
        stop taken from the strategy when supplied and a volatility-scaled
        estimate when not. The full SizingResult is kept on self.last_sizing
        so callers can tell real risk sizing from a capped or fallback number.
        """
        result = position_sizing.size_position(
            symbol=symbol,
            price=price,
            equity=self.trader.equity,
            risk_per_trade=getattr(self.config, 'risk_per_trade_pct', 0.005),
            stop_distance=stop_distance,
            stop_price=stop_price,
            max_position_size=self.config.max_position_size,
            max_leverage_pct=getattr(self.config, 'max_leverage_pct', 0.20),
            vol_tracker=self.vol_tracker,
        )
        self.last_sizing = result

        for w in result.warnings:
            print(f"[SIZING] {symbol}: {w}")

        size = result.size
        ''',
    },
]

# process_signal's signature gains keyword-only stop arguments.
# NOTE: the marker must be unique to THIS edit. An earlier version used
# "stop_price: float = None", which already appears in submit_order() at the
# top of the file -- the patch was silently skipped as "already applied" and
# process_signal never gained the argument. Markers are now tagged comments.
SIGNATURE_PATCH = {
    'name': 'process_signal signature: keyword-only stop arguments',
    'marker': 'SIZING-FIX-SIGNATURE',
    'old': '''        size: float = None,
        timestamp: datetime = None
    ):''',
    'new': '''        size: float = None,
        timestamp: datetime = None,
        # SIZING-FIX-SIGNATURE: keyword-only so existing positional callers
        # are unaffected. Pass stop_price for real risk-based sizing.
        *,
        stop_price: float = None,
        stop_distance: float = None
    ):''',
}

# Strings that MUST be present after a successful patch. Checked structurally
# after writing, so a marker collision cannot silently skip an edit again.
POST_CONDITIONS = [
    ('SIZING-FIX-SIGNATURE', 'process_signal did not gain the stop arguments'),
    ('import position_sizing', 'sizing module was not imported'),
    ('risk_per_trade_pct', 'ExecutionConfig did not gain the risk knob'),
    ('self.vol_tracker', 'volatility tracker was not attached'),
    ('position_sizing.size_position', '_calculate_size still uses the old formula'),
    ('stop_price=stop_price', 'entries do not forward the stop'),
]

ABSENT_CONDITIONS = [
    ('stop_distance = price * 0.02', 'the inert 2% stop is still present'),
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
        print(f"         Copy it into {project_dir} first.")
        return False
    print(f"  [DEP]   {DEP} present")

    text, crlf = read_text(path)
    applied, skipped, failed = [], [], []

    for p in [SIGNATURE_PATCH] + PATCHES:
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
        print("  [ROLLBACK] Restoring from backup")
        shutil.copy2(backup, path)
        return False
    print("  [VERIFY] Syntax OK")

    # Structural check: confirm every edit actually landed. Guards against a
    # marker colliding with pre-existing code and skipping a patch silently.
    final, _ = read_text(path)
    problems = []
    for needle, msg in POST_CONDITIONS:
        if needle not in final:
            problems.append(msg)
    for needle, msg in ABSENT_CONDITIONS:
        if needle in final:
            problems.append(msg)

    if problems:
        print("  [VERIFY] POST-CONDITIONS FAILED:")
        for p in problems:
            print(f"           - {p}")
        print("  [ROLLBACK] Restoring from backup")
        shutil.copy2(backup, path)
        return False

    print(f"  [VERIFY] Post-conditions OK ({len(POST_CONDITIONS)} checked)")
    return True


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
    ap = argparse.ArgumentParser(description="Apply risk-based position sizing")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--revert', action='store_true')
    ap.add_argument('--dir', default='.')
    args = ap.parse_args()

    project_dir = os.path.abspath(args.dir)

    print("=" * 70)
    print("POSITION SIZING FIX - PATCHER")
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
        print("  python test_position_sizing.py")
        print("  python test_system.py")
        print("\nEXPECT: position sizes to CHANGE, and [SIZING] warnings whenever a")
        print("strategy enters without a stop. Those warnings are the point --")
        print("they mark every place risk is being estimated rather than measured.")
        print("\nTo silence them properly, pass stop_price= to process_signal.")
    else:
        print("PATCH INCOMPLETE - see failures above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
