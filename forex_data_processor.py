# ==============================================================================
# forex_data_processor.py
# ==============================================================================
# Merges yearly Forex XLSX files from HistData.com into continuous datasets
# Run this ONCE to prepare Forex data, then data_manager.py uses the merged files
#
# IMPORTANT: For HistData.com .xlsx files, you need openpyxl:
#   pip install openpyxl
#
# HistData.com Format (NO HEADER ROW):
#   Column 0: DateTime (as Excel datetime)
#   Column 1: Open
#   Column 2: High
#   Column 3: Low
#   Column 4: Close
#   Column 5: Volume (usually 0 for forex)
#
# ==============================================================================
# TIMEZONE CONTRACT  (Phase 0 fix -- read this before touching anything below)
# ==============================================================================
# HistData.com states in its file specification:
#
#     "Eastern Standard Time (EST) time-zone WITHOUT Day Light Savings
#      adjustments"
#
# That is a FIXED UTC-5 offset, all year, every year.
#
# Previously this module parsed those stamps as naive datetimes and wrote them
# out unchanged. data_manager.py passed them through untouched (its tz
# normalisation only fires when the index is already tz-aware, which forex
# never was), and ftmo_compliance.to_prague_time() then treated any naive
# timestamp as UTC. Net effect: every forex bar was believed to have occurred
# 5 hours earlier than it really did.
#
# Consequence for FTMO: the Prague-midnight daily reset was drawn in the wrong
# place. Prague midnight is 18:00 in HistData's clock during CET and 17:00
# during CEST -- roughly the 17:00 New York rollover. The old code drew it at
# 23:00/22:00 HistData-clock instead, i.e. the middle of the Tokyo session.
# Every trade between roughly 18:00 and 23:00 EST was booked to the previous
# Prague trading day, so the 5% max-daily-loss window was measured over the
# wrong 24 hours. That can both hide a real breach and invent a fake one.
#
# THE FIX: convert EST-fixed -> UTC once, here, at ingest. Downstream stays
# naive-UTC, which is what ftmo_compliance already assumes, so its assumption
# becomes true instead of merely stated.
#
# DO NOT "improve" this by switching to America/New_York. That zone observes
# US daylight saving; HistData does not. Using it would be correct in winter
# and one hour wrong from March to November -- a subtler bug than the one this
# replaces. The offset here is deliberately fixed.
#
# Output filename changed from {ticker}_1min_merged.csv to {ticker}_1min_utc.csv
# so that a stale pre-fix cache cannot be silently picked up.
# ==============================================================================

import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import config

# ==============================================================================
# TIMEZONE CONSTANTS
# ==============================================================================

# HistData.com publishes in Eastern Standard Time with NO daylight saving.
# Fixed offset, deliberately not an IANA zone that would apply DST rules.
HISTDATA_UTC_OFFSET_HOURS = -5
HISTDATA_TZ_LABEL = "EST-fixed (UTC-5, no DST)"

# Suffix marking a file as timezone-normalised. Legacy files lack it.
UTC_SUFFIX = "_1min_utc.csv"
LEGACY_SUFFIX = "_1min_merged.csv"

# Timeframes whose resampled caches derive from the base file and must be
# invalidated whenever the base is rebuilt.
DERIVED_TIMEFRAMES = ['1min', '5min', '15min', '30min', '1hour', '4hour', '1day']


def histdata_to_utc(index):
    """
    Convert a naive DatetimeIndex of HistData (EST-fixed) stamps to naive UTC.

    EST is UTC-5, so the UTC instant is 5 hours LATER than the printed stamp.
    Returns a naive index (no tzinfo) because the rest of the pipeline -- and
    ftmo_compliance.to_prague_time() in particular -- expects naive-UTC.
    """
    if index.tz is not None:
        # Already tz-aware: trust it, just normalise to naive UTC.
        return index.tz_convert("UTC").tz_localize(None)
    return index + timedelta(hours=-HISTDATA_UTC_OFFSET_HOURS)


class ForexDataProcessor:
    """
    Processes raw yearly Forex XLSX files into continuous datasets.

    All output is naive UTC. See the TIMEZONE CONTRACT block above.
    """

    def __init__(self):
        self.base_path = config.FOREX_BASE_PATH
        self.cache_path = config.CACHE_SUBDIRS['forex']
        os.makedirs(self.cache_path, exist_ok=True)

    def find_yearly_files(self, ticker):
        """
        Find all yearly files for a specific ticker

        Args:
            ticker: Clean ticker name (e.g., 'EURUSD')

        Returns:
            List of file paths sorted chronologically
        """
        patterns = [
            os.path.join(self.base_path, f"DAT_XLSX_{ticker}_M1_*.xlsx"),
            os.path.join(self.base_path, f"*{ticker}_M1_*.xlsx"),
            os.path.join(self.base_path, f"*{ticker}*.xlsx"),
        ]

        files = []
        for pattern in patterns:
            found = glob.glob(pattern)
            files.extend(found)

        # Remove duplicates and filter out .txt files
        files = list(set(files))
        files = [f for f in files if not f.endswith('.txt')]

        # Sort by year extracted from filename
        def extract_year(filename):
            import re
            match = re.search(r'(\d{4})', os.path.basename(filename))
            return int(match.group(1)) if match else 0

        files.sort(key=extract_year)

        return files

    def load_and_merge_ticker(self, ticker, verbose=True):
        """
        Load all yearly files for a ticker and merge into one dataset.

        Timestamps are converted from HistData's EST-fixed clock to naive UTC
        per-file, before merging, so the dedup and sort operate on real
        instants rather than on printed local stamps.

        Args:
            ticker: Clean ticker name (e.g., 'EURUSD')
            verbose: Print progress messages

        Returns:
            Merged DataFrame indexed by naive UTC
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Processing {ticker}")
            print(f"{'=' * 70}")
            print(f"Source timezone: {HISTDATA_TZ_LABEL}  ->  converting to UTC")

        files = self.find_yearly_files(ticker)

        if not files:
            print(f"[FAIL] No files found for {ticker} at {self.base_path}")
            return None

        if verbose:
            print(f"Found {len(files)} yearly files")

        # Load and concatenate all files
        dfs = []
        sample_before = None
        sample_after = None

        for file_path in files:
            try:
                # HistData.com files have NO HEADER
                # Columns are: datetime, open, high, low, close, volume
                df = pd.read_excel(
                    file_path,
                    engine='openpyxl',
                    header=None,  # NO HEADER ROW
                    names=['datetime', 'open', 'high', 'low', 'close', 'volume']
                )

                # Convert datetime column
                # HistData stores as Excel datetime objects
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

                # Remove invalid rows
                df = df.dropna(subset=['datetime'])

                # Set datetime as index
                df.set_index('datetime', inplace=True)

                # ---- TIMEZONE NORMALISATION (EST-fixed -> naive UTC) --------
                if sample_before is None and len(df) > 0:
                    sample_before = df.index[0]
                df.index = histdata_to_utc(df.index)
                if sample_after is None and len(df) > 0:
                    sample_after = df.index[0]
                # -------------------------------------------------------------

                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['open', 'high', 'low', 'close'])

                dfs.append(df)

                if verbose:
                    year = os.path.basename(file_path).split('_')[-1].replace('.xlsx', '')
                    print(f"  [OK] {year}: {len(df):,} rows")

            except Exception as e:
                print(f"  [FAIL] Failed to load {os.path.basename(file_path)}: {e}")
                continue

        if not dfs:
            print(f"[FAIL] No valid data loaded for {ticker}")
            return None

        # Concatenate all years
        if verbose:
            print(f"\n  Merging {len(dfs)} files...")

        merged = pd.concat(dfs, axis=0)

        # Remove duplicates (keep first occurrence)
        merged = merged[~merged.index.duplicated(keep='first')]

        # Sort chronologically
        merged = merged.sort_index()

        # Keep only OHLCV columns
        merged = merged[['open', 'high', 'low', 'close', 'volume']]

        if verbose:
            print(f"\n  [OK] Merged {ticker}:")
            print(f"     Total rows: {len(merged):,}")
            print(f"     Date range: {merged.index.min()} to {merged.index.max()}  (UTC)")
            if sample_before is not None and sample_after is not None:
                print(f"     TZ shift:   {sample_before}  ->  {sample_after}  (+5h)")
            self._report_session_gap(merged)

        return merged

    def _report_session_gap(self, df):
        """
        Sanity check: after conversion, the weekly FX session gap should close
        around 21:00-22:00 UTC on Friday. If it still closes near 16:00-17:00,
        the conversion did not happen and the file is still on EST.
        """
        try:
            fridays = df[df.index.weekday == 4]
            if fridays.empty:
                return
            last_bars = fridays.groupby(fridays.index.date).apply(lambda g: g.index.max())
            hours = pd.Series([t.hour for t in last_bars])
            modal = hours.mode()
            if modal.empty:
                return
            modal_hour = int(modal.iloc[0])
            share = (hours == modal_hour).mean() * 100

            print(f"     Week close: most Fridays end at {modal_hour:02d}:00 UTC ({share:.0f}% of weeks)")
            if modal_hour in (20, 21, 22, 23):
                print(f"     [OK] Consistent with UTC. Conversion looks correct.")
            elif modal_hour in (15, 16, 17, 18):
                print(f"     [WARN] Still looks like EST. Conversion may not have applied.")
                print(f"            Run: python verify_histdata_timezone.py")
            else:
                print(f"     [WARN] Unexpected week-close hour. Verify manually.")
        except Exception:
            pass  # Diagnostic only; never block processing.

    def save_merged_data(self, ticker, data):
        """
        Save merged data to cache as CSV.

        Filename carries a _utc marker so a pre-fix file cannot be mistaken
        for a converted one.
        """
        filename = os.path.join(self.cache_path, f"{ticker}{UTC_SUFFIX}")
        data.to_csv(filename)
        print(f"  [SAVE] Saved to: {filename}")
        return filename

    def invalidate_stale_caches(self, ticker, verbose=True):
        """
        Remove caches derived from the pre-fix base file.

        Two families need clearing:
          1. The legacy {ticker}_1min_merged.csv base itself.
          2. Every resampled cache {ticker}_{timeframe}.csv, which was built
             from that base and therefore inherits the 5-hour error.

        Skipping this is the most likely way for the fix to appear applied
        while the system keeps serving wrong data.
        """
        removed = []

        legacy = os.path.join(self.cache_path, f"{ticker}{LEGACY_SUFFIX}")
        if os.path.exists(legacy):
            try:
                os.remove(legacy)
                removed.append(os.path.basename(legacy))
            except OSError as e:
                print(f"  [WARN] Could not remove {legacy}: {e}")

        for tf in DERIVED_TIMEFRAMES:
            path = os.path.join(self.cache_path, f"{ticker}_{tf}.csv")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed.append(os.path.basename(path))
                except OSError as e:
                    print(f"  [WARN] Could not remove {path}: {e}")

        if verbose and removed:
            print(f"  [CLEAN] Invalidated {len(removed)} stale cache file(s):")
            for name in removed:
                print(f"          - {name}")
        elif verbose:
            print(f"  [CLEAN] No stale caches found for {ticker}")

        return removed

    def process_all_tickers(self):
        """
        Process all Forex tickers defined in config
        """
        print("\n" + "=" * 70)
        print("FOREX DATA PROCESSOR")
        print("=" * 70)
        print(f"Base path:  {self.base_path}")
        print(f"Cache path: {self.cache_path}")
        print(f"Tickers:    {list(config.FOREX_TICKERS.values())}")
        print(f"Timezone:   {HISTDATA_TZ_LABEL} -> UTC (naive)")
        print("=" * 70)

        results = {}

        for symbol, ticker in config.FOREX_TICKERS.items():
            merged_data = self.load_and_merge_ticker(ticker)

            if merged_data is not None:
                # Clear stale derivatives BEFORE writing the new base, so a
                # crash mid-write cannot leave new base + old resampled caches.
                self.invalidate_stale_caches(ticker)
                self.save_merged_data(ticker, merged_data)
                results[ticker] = {
                    'success': True,
                    'rows': len(merged_data),
                    'start': merged_data.index.min(),
                    'end': merged_data.index.max()
                }
            else:
                results[ticker] = {'success': False}

        # Summary
        print(f"\n{'=' * 70}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 70}")

        successful = 0
        for ticker, result in results.items():
            if result['success']:
                successful += 1
                print(f"  [OK] {ticker:8} | {result['rows']:>10,} rows | {result['start'].date()} to {result['end'].date()}")
            else:
                print(f"  [FAIL] {ticker:8} | FAILED")

        print(f"\n  Total: {successful}/{len(results)} tickers processed successfully")
        print(f"  All timestamps are naive UTC.")
        print("=" * 70 + "\n")

        return results


def main():
    print("\n" + "=" * 70)
    print("FOREX DATA PROCESSOR - HistData.com XLSX Merger")
    print("=" * 70)
    print("This merges yearly HistData.com files into continuous datasets")
    print(f"and converts timestamps from {HISTDATA_TZ_LABEL} to UTC.")
    print("=" * 70)

    if not os.path.exists(config.FOREX_BASE_PATH):
        print(f"\n[FAIL] Forex data directory not found: {config.FOREX_BASE_PATH}")
        print("   Check config.FOREX_BASE_PATH")
        return

    processor = ForexDataProcessor()
    results = processor.process_all_tickers()

    if any(r.get('success') for r in results.values()):
        print("NEXT STEPS:")
        print("  1. Run: python verify_histdata_timezone.py    (confirm UTC)")
        print("  2. Run: python test_data_download.py")
        print("  3. Run: python run_backtests.py")
        print("")
        print("  NOTE: any FTMO compliance results produced before this run")
        print("        were computed on a 5-hour-shifted daily boundary and")
        print("        should be regenerated.")
        print("")


if __name__ == '__main__':
    main()
