# ==============================================================================
# verify_histdata_timezone.py
# ==============================================================================
# EVIDENCE SCRIPT for Phase 0, Item 1: HistData timezone audit.
#
# Does NOT trust documentation. Proves — from your actual files — which clock
# the timestamps are on, using the weekly forex session gap as a fingerprint.
#
# THE FINGERPRINT
# ---------------
# The spot FX week closes Friday 17:00 New York and reopens Sunday 17:00 NY.
# New York observes DST; HistData's clock does not. So if the data really is
# fixed-EST (UTC-5), the last bar of the week lands at:
#
#     Friday 17:00   during US winter (NY on EST, UTC-5)
#     Friday 16:00   during US summer (NY on EDT, UTC-4)
#
# If the data were already UTC, the same bar would land at:
#
#     Friday 22:00   winter
#     Friday 21:00   summer
#
# Those two hypotheses are 5 hours apart and cannot be confused. This script
# measures which one your files actually match.
#
# USAGE
#     python verify_histdata_timezone.py
#     python verify_histdata_timezone.py --ticker EURUSD
#     python verify_histdata_timezone.py --file "E:/TradingData/cache/forex/EURUSD_1min_utc.csv"
#
# Run it BEFORE the fix (expect: EST) and AFTER the fix (expect: UTC).
# ==============================================================================

import argparse
import glob
import os
import sys

import pandas as pd

try:
    import config
except ImportError:
    config = None


# Expected hour of the final bar of the trading week, by hypothesis and season.
# (US summer = NY on EDT, US winter = NY on EST)
HYPOTHESES = {
    'EST_FIXED (UTC-5, HistData raw)': {'winter': 17, 'summer': 16},
    'UTC (already converted)':         {'winter': 22, 'summer': 21},
}

# Tolerance in hours. Fridays can end a few minutes early on thin weeks, and
# some brokers stop a minute or two before the hour, so we allow +/- 1 hour.
TOLERANCE_HOURS = 1


def _is_us_summer(ts):
    """
    Rough US DST window: second Sunday of March -> first Sunday of November.
    Approximated by month, which is accurate except for a handful of edge weeks.
    Those edge weeks are excluded from scoring rather than guessed at.
    """
    m = ts.month
    if m in (4, 5, 6, 7, 8, 9, 10):
        return True
    if m in (12, 1, 2):
        return False
    return None  # March / November -> ambiguous, skip


def find_week_close_hours(df, max_weeks=400):
    """
    For each trading week in the frame, return the hour-of-day of the last bar
    before the weekend gap, tagged with season.
    """
    idx = df.index
    if idx.tz is not None:
        print("  [NOTE] Index is timezone-aware; comparing in its own local time.")

    # Friday = weekday 4. Take the last bar of each Friday.
    fridays = df[idx.weekday == 4]
    if fridays.empty:
        return []

    last_per_friday = fridays.groupby(fridays.index.date).apply(
        lambda g: g.index.max()
    )

    out = []
    for ts in list(last_per_friday)[-max_weeks:]:
        season = _is_us_summer(ts)
        if season is None:
            continue
        out.append((ts, 'summer' if season else 'winter'))
    return out


def score_hypotheses(week_closes):
    """Score each timezone hypothesis by how many week-closes it explains."""
    scores = {}
    for name, expected in HYPOTHESES.items():
        hits = 0
        for ts, season in week_closes:
            target = expected[season]
            if abs(ts.hour - target) <= TOLERANCE_HOURS:
                hits += 1
        scores[name] = hits
    return scores


def analyze(df, label):
    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {label}")
    print(f"{'=' * 70}")
    print(f"  Rows:  {len(df):,}")
    print(f"  Range: {df.index.min()}  ->  {df.index.max()}")

    week_closes = find_week_close_hours(df)
    if not week_closes:
        print("  [FAIL] No Friday bars found - cannot fingerprint.")
        return None

    print(f"  Weeks sampled: {len(week_closes)}")

    # Show the raw distribution so the conclusion is auditable, not just asserted.
    print("\n  Distribution of final-bar hour on Friday:")
    for season in ('winter', 'summer'):
        hours = [ts.hour for ts, s in week_closes if s == season]
        if not hours:
            continue
        counts = pd.Series(hours).value_counts()
        top = ', '.join(f"{h:02d}:00 x{c}" for h, c in counts.sort_values(ascending=False).head(4).items())
        print(f"    {season:7} (n={len(hours):4}): {top}")

    scores = score_hypotheses(week_closes)
    total = len(week_closes)

    print("\n  Hypothesis fit:")
    best_name, best_hits = None, -1
    for name, hits in scores.items():
        pct = hits / total * 100 if total else 0
        marker = ''
        print(f"    {name:34} {hits:5}/{total} ({pct:5.1f}%){marker}")
        if hits > best_hits:
            best_name, best_hits = name, hits

    best_pct = best_hits / total * 100 if total else 0

    print(f"\n  {'-' * 66}")
    if best_pct >= 70:
        print(f"  VERDICT: {best_name}")
        print(f"           Confidence: {best_pct:.1f}% of weeks match.")
    else:
        print(f"  VERDICT: INCONCLUSIVE (best fit {best_name} at {best_pct:.1f}%)")
        print("           Data may be from a different provider, already shifted,")
        print("           or contain a broker-specific session schedule.")
        print("           Do NOT apply a blind -5h shift. Inspect manually.")
    print(f"  {'-' * 66}")

    return best_name if best_pct >= 70 else None


def load_raw_xlsx(base_path, ticker, n_files=2):
    """Load a couple of raw HistData xlsx files (pre-conversion ground truth)."""
    patterns = [
        os.path.join(base_path, f"DAT_XLSX_{ticker}_M1_*.xlsx"),
        os.path.join(base_path, f"*{ticker}*.xlsx"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(f for f in files if not f.endswith('.txt')))
    if not files:
        return None

    dfs = []
    for fp in files[-n_files:]:
        df = pd.read_excel(
            fp, engine='openpyxl', header=None,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        )
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime']).set_index('datetime')
        dfs.append(df)
        print(f"  Loaded raw: {os.path.basename(fp)} ({len(df):,} rows)")

    return pd.concat(dfs).sort_index() if dfs else None


def main():
    ap = argparse.ArgumentParser(description="Fingerprint the timezone of HistData forex files")
    ap.add_argument('--ticker', default='EURUSD', help='Ticker to test (default EURUSD)')
    ap.add_argument('--file', default=None, help='Analyze one specific CSV instead')
    args = ap.parse_args()

    print("=" * 70)
    print("HISTDATA TIMEZONE VERIFICATION")
    print("=" * 70)
    print("Method: locate the weekly FX session gap and check which clock it")
    print("        falls on. EST-fixed and UTC differ by 5h - unmistakable.")

    if args.file:
        if not os.path.exists(args.file):
            print(f"\n[FAIL] File not found: {args.file}")
            return 1
        df = pd.read_csv(args.file, index_col=0, parse_dates=True)
        analyze(df, os.path.basename(args.file))
        return 0

    if config is None:
        print("\n[FAIL] Could not import config.py. Use --file to point at a CSV directly.")
        return 1

    ticker = args.ticker.upper()

    # 1. Raw source files -- the ground truth, untouched by any pipeline.
    print(f"\n[1/2] Raw HistData source files for {ticker}")
    print("-" * 70)
    try:
        raw = load_raw_xlsx(config.FOREX_BASE_PATH, ticker)
        if raw is not None:
            analyze(raw, f"RAW xlsx: {ticker}")
        else:
            print(f"  [WARN] No raw .xlsx found at {config.FOREX_BASE_PATH}")
    except Exception as e:
        print(f"  [WARN] Could not read raw files: {e}")

    # 2. Whatever the pipeline currently produces.
    print(f"\n[2/2] Pipeline output for {ticker}")
    print("-" * 70)
    cache_dir = config.CACHE_SUBDIRS['forex']
    for suffix, tag in (('_1min_utc.csv', 'POST-FIX'), ('_1min_merged.csv', 'LEGACY / PRE-FIX')):
        path = os.path.join(cache_dir, f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            analyze(df, f"{tag}: {os.path.basename(path)}")
        else:
            print(f"  (not present: {os.path.basename(path)})")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("  RAW files should read EST_FIXED. That is the expected input.")
    print("  If a LEGACY _1min_merged.csv also reads EST_FIXED, it was never")
    print("  converted -- every FTMO daily-loss number built on it is 5h off.")
    print("  After running the fixed forex_data_processor.py, the _1min_utc.csv")
    print("  must read UTC. If it does not, stop and investigate before")
    print("  trusting any compliance output.")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())