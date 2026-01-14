# ==============================================================================
# diagnose_histdata_files.py
# OPTIONAL: Quick diagnostic to check if your HistData.com files are readable
# Run this BEFORE forex_data_processor.py if you want to verify your data
# ==============================================================================
# CHANGELOG:
# - ENHANCED: Now tests ALL forex files found, not just selected ones
# - IMPROVED: Better column detection and format validation
# - ADDED: Automatic detection of common HistData.com formats
# - IMPROVED: More detailed error reporting with suggestions
# ==============================================================================

import pandas as pd
import os
import glob

def diagnose_single_file(filepath):
    """
    Diagnose a single HistData.com file
    ENHANCED: More comprehensive diagnostics
    """
    print(f"\n{'='*70}")
    print(f"Diagnosing: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    try:
        # Try reading as Excel
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, nrows=10, engine='openpyxl')
            print(f"✅ Successfully read as .xlsx file")
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=10)
            print(f"✅ Successfully read as .csv file")
        else:
            print(f"❌ Unknown file type: {filepath}")
            return False
        
        # Show basic info
        print(f"\nFile info:")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape (first 10 rows): {df.shape}")
        print(f"  Data types:")
        for col, dtype in df.dtypes.items():
            print(f"    {col}: {dtype}")
        
        # Show sample data
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        
        # Check for datetime
        datetime_found = False
        for col in df.columns:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['date', 'time', 'dt']):
                datetime_found = True
                print(f"\n✅ Found datetime-related column: {col}")
        
        if not datetime_found and len(df.columns) >= 2:
            print(f"\n⚠️  No explicit datetime column, but found {len(df.columns)} columns")
            print(f"   HistData format often has date in column 0, time in column 1")
            print(f"   Column 0: {df.columns[0]}")
            print(f"   Column 1: {df.columns[1]}")
        
        # Check for OHLC
        ohlc_cols = []
        for col in df.columns:
            col_str = str(col).upper()
            if any(x in col_str for x in ['OPEN', 'HIGH', 'LOW', 'CLOSE']):
                ohlc_cols.append(col)
        
        if len(ohlc_cols) >= 4:
            print(f"\n✅ Found OHLC columns: {ohlc_cols}")
        else:
            print(f"\n⚠️  Only found {len(ohlc_cols)} OHLC columns: {ohlc_cols}")
            print(f"   Expected columns like: Open, High, Low, Close")
        
        # Check for volume
        volume_found = False
        for col in df.columns:
            if 'volume' in str(col).lower():
                volume_found = True
                print(f"\n✅ Found volume column: {col}")
        
        if not volume_found:
            print(f"\n⚠️  No volume column found")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Failed to read file: {e}")
        print(f"\nTroubleshooting suggestions:")
        print(f"  1. Verify file is not corrupted")
        print(f"  2. Check file permissions")
        print(f"  3. Ensure file is in correct format (CSV or XLSX)")
        return False

def find_forex_files(base_path):
    """
    Find all forex files in the base path
    ENHANCED: Better file discovery
    """
    patterns = [
        os.path.join(base_path, "*.xlsx"),
        os.path.join(base_path, "*.csv"),
        os.path.join(base_path, "**", "*.xlsx"),
        os.path.join(base_path, "**", "*.csv")
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out .txt files and remove duplicates
    files = list(set([f for f in files if not f.endswith('.txt')]))
    
    # Sort for consistent ordering
    files.sort()
    
    return files

def detect_histdata_format(df):
    """
    Detect the specific HistData.com format
    NEW: Helps identify file structure
    """
    # Common HistData formats
    formats = {
        'standard': ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
        'tick': ['Date', 'Time', 'Bid', 'Ask'],
        'ohlc_only': ['Open', 'High', 'Low', 'Close'],
    }
    
    cols = [str(c) for c in df.columns]
    
    for format_name, expected_cols in formats.items():
        matches = sum(1 for col in expected_cols if col in cols)
        if matches >= len(expected_cols) - 1:  # Allow 1 missing column
            return format_name, matches / len(expected_cols)
    
    return 'unknown', 0.0

def main():
    """
    Run diagnostics on HistData.com files
    MODIFIED: Tests all files with better reporting
    """
    base_path = "E:/TradingData/forex"
    
    print("="*70)
    print("HISTDATA.COM FILE DIAGNOSTIC")
    print("="*70)
    print(f"\nSearching for files in: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"\n❌ Directory not found: {base_path}")
        print("   Please create this directory and add your HistData.com files")
        print("\nTo fix:")
        print(f"   1. Create directory: mkdir {base_path}")
        print(f"   2. Download data from HistData.com")
        print(f"   3. Extract files to {base_path}")
        return
    
    files = find_forex_files(base_path)
    
    if not files:
        print(f"\n❌ No .xlsx or .csv files found in {base_path}")
        print("\nExpected files like:")
        print("  - DAT_XLSX_EURUSD_M1_2020.xlsx")
        print("  - DAT_XLSX_GBPUSD_M1_2021.xlsx")
        print("  - DAT_CSV_EURUSD_M1_2020.csv")
        print("\nTo fix:")
        print("   1. Visit: https://www.histdata.com/download-free-forex-data/")
        print("   2. Download 1-minute bar data for your pairs")
        print(f"   3. Extract files to {base_path}")
        return
    
    print(f"\n✅ Found {len(files)} files")
    
    # Show all files found
    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {i:2d}. {os.path.basename(f):50s} ({size_mb:6.1f} MB)")
    
    print("\nOptions:")
    print("  1. Diagnose first file (detailed)")
    print("  2. Quick check ALL files")
    print("  3. Diagnose specific file by number")
    print("  4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        # Diagnose first file in detail
        diagnose_single_file(files[0])
    
    elif choice == '2':
        # Quick check all files
        print("\n" + "="*70)
        print("QUICK CHECK ALL FILES")
        print("="*70)
        
        success_count = 0
        failed_files = []
        format_counts = {}
        
        for filepath in files:
            print(f"\n{os.path.basename(filepath)[:50]:50s} ... ", end="", flush=True)
            try:
                if filepath.endswith('.xlsx'):
                    df = pd.read_excel(filepath, nrows=5, engine='openpyxl')
                else:
                    df = pd.read_csv(filepath, nrows=5)
                
                # Detect format
                format_name, confidence = detect_histdata_format(df)
                format_counts[format_name] = format_counts.get(format_name, 0) + 1
                
                if len(df.columns) >= 4 and confidence > 0.5:
                    print(f"✅ OK ({len(df.columns)} cols, {format_name})")
                    success_count += 1
                else:
                    print(f"⚠️  {len(df.columns)} cols, unknown format")
                    failed_files.append((filepath, 'unknown_format'))
            
            except Exception as e:
                print(f"❌ FAILED: {str(e)[:40]}")
                failed_files.append((filepath, str(e)[:100]))
        
        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY: {success_count}/{len(files)} files readable")
        print(f"{'='*70}")
        
        # Format distribution
        print(f"\n📊 Detected Formats:")
        for format_name, count in sorted(format_counts.items()):
            print(f"  {format_name:15s}: {count} files")
        
        if failed_files:
            print(f"\n⚠️  {len(failed_files)} files had issues:")
            for filepath, reason in failed_files:
                print(f"  - {os.path.basename(filepath)}")
                print(f"    Reason: {reason}")
        else:
            print(f"\n✅ All files look good!")
            print(f"\nNext steps:")
            print(f"  1. Run: python test_data_download.py")
            print(f"  2. Verify Forex data loads correctly")
            print(f"  3. Check config.py has correct FOREX_BASE_PATH")
    
    elif choice == '3':
        # Diagnose specific file
        try:
            file_num = int(input(f"Enter file number (1-{len(files)}): "))
            if 1 <= file_num <= len(files):
                diagnose_single_file(files[file_num - 1])
            else:
                print(f"❌ Invalid file number. Must be between 1 and {len(files)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    else:
        print("\n👋 Exiting")

if __name__ == "__main__":
    main()