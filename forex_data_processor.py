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
# ==============================================================================

import pandas as pd
import os
import glob
from datetime import datetime
import config

class ForexDataProcessor:
    """
    Processes raw yearly Forex XLSX files into continuous datasets
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
        Load all yearly files for a ticker and merge into one dataset
        
        Args:
            ticker: Clean ticker name (e.g., 'EURUSD')
            verbose: Print progress messages
        
        Returns:
            Merged DataFrame with all years of data
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing {ticker}")
            print(f"{'='*70}")
        
        files = self.find_yearly_files(ticker)
        
        if not files:
            print(f"[FAIL] No files found for {ticker} at {self.base_path}")
            return None
        
        if verbose:
            print(f"Found {len(files)} yearly files")
        
        # Load and concatenate all files
        dfs = []
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
                
                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
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
            print(f"     Date range: {merged.index.min()} to {merged.index.max()}")
        
        return merged
    
    def save_merged_data(self, ticker, data):
        """
        Save merged data to cache as CSV
        """
        filename = os.path.join(self.cache_path, f"{ticker}_1min_merged.csv")
        data.to_csv(filename)
        print(f"  [SAVE] Saved to: {filename}")
        return filename
    
    def process_all_tickers(self):
        """
        Process all Forex tickers defined in config
        """
        print("\n" + "="*70)
        print("FOREX DATA PROCESSOR")
        print("="*70)
        print(f"Base path:  {self.base_path}")
        print(f"Cache path: {self.cache_path}")
        print(f"Tickers:    {list(config.FOREX_TICKERS.values())}")
        print("="*70)
        
        results = {}
        
        for symbol, ticker in config.FOREX_TICKERS.items():
            merged_data = self.load_and_merge_ticker(ticker)
            
            if merged_data is not None:
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
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        successful = 0
        for ticker, result in results.items():
            if result['success']:
                successful += 1
                print(f"  [OK] {ticker:8} | {result['rows']:>10,} rows | {result['start'].date()} to {result['end'].date()}")
            else:
                print(f"  [FAIL] {ticker:8} | FAILED")
        
        print(f"\n  Total: {successful}/{len(results)} tickers processed successfully")
        print("="*70 + "\n")
        
        return results


def main():
    """
    Run the Forex data processor
    """
    processor = ForexDataProcessor()
    results = processor.process_all_tickers()
    
    # Check results
    successful = sum(1 for r in results.values() if r.get('success'))
    
    if successful > 0:
        print("[OK] Forex data processing complete!")
        print("\nNext steps:")
        print("  1. Run: python test_data_download.py")
        print("  2. Run: python run_backtests.py")
    else:
        print("[FAIL] No data was processed successfully.")
        print("\nTroubleshooting:")
        print(f"  1. Check that .xlsx files exist in: {config.FOREX_BASE_PATH}")
        print("  2. Run: python diagnose_histdata_files.py")


if __name__ == "__main__":
    main()