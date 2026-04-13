# ==============================================================================
# run_variant_backtests.py
# ==============================================================================
# VARIANT BACKTESTER
# 
# This script:
# 1. Scans strategies/variants/ folder for variant files
# 2. Dynamically imports each variant strategy class
# 3. Runs backtests across all assets and timeframes
# 4. Saves results to database with variant_id for comparison
#
# Cost: FREE (no API calls)
# ==============================================================================

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime

import config
from backtester_multi_timeframe import MultiTimeframeBacktester
from database import ResultsDatabase

# ==============================================================================
# CONFIGURATION
# ==============================================================================

VARIANTS_DIR = Path(__file__).parent / 'strategies' / 'variants'

# Which timeframes to test (can be reduced for faster testing)
TEST_TIMEFRAMES = ['1hour', '4hour', '1day']

# Which assets to test (can be reduced for faster testing)
# Set to None to use all assets from config
TEST_ASSETS = None  # Uses all enabled assets


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def discover_variants():
    """Find all variant files in the variants directory"""
    
    if not VARIANTS_DIR.exists():
        print(f"[FAIL] Variants directory not found: {VARIANTS_DIR}")
        return []
    
    variant_files = sorted(VARIANTS_DIR.glob('variant_*.py'))
    
    print(f"[FOLDER] Found {len(variant_files)} variant files")
    
    return variant_files


def load_variant_class(filepath):
    """Dynamically import a strategy class from a variant file"""
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[filepath.stem] = module
        spec.loader.exec_module(module)
        
        # Find the strategy class (inherits from bt.Strategy)
        import backtrader as bt
        
        strategy_class = None
        class_name = None
        
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj != bt.Strategy:
                strategy_class = obj
                class_name = name
                break
        
        if strategy_class:
            return strategy_class, class_name, None
        else:
            return None, None, "No Strategy class found in file"
    
    except Exception as e:
        return None, None, str(e)


def get_test_assets():
    """Get list of assets to test"""
    
    if TEST_ASSETS:
        return TEST_ASSETS
    
    assets = []
    
    if config.FOREX_ENABLED:
        assets.extend(config.FOREX_WATCHLIST)
    
    if config.CRYPTO_ENABLED:
        assets.extend(config.CRYPTO_WATCHLIST)
    
    return assets


# ==============================================================================
# MAIN BACKTESTING FUNCTION
# ==============================================================================

def run_variant_backtests(quick_mode=False):
    """Run backtests for all variants"""
    
    print("\n" + "="*70)
    print("[LAB] VARIANT BACKTESTER")
    print("="*70)
    
    # Discover variants
    variant_files = discover_variants()
    
    if not variant_files:
        print("No variants to test. Run mutate_strategy.py first.")
        return
    
    # Get assets and timeframes
    assets = get_test_assets()
    timeframes = ['1hour', '4hour', '1day'] if quick_mode else TEST_TIMEFRAMES
    
    print(f"\n[LIST] Test Configuration:")
    print(f"   Variants:   {len(variant_files)}")
    print(f"   Assets:     {len(assets)}")
    print(f"   Timeframes: {len(timeframes)}")
    print(f"   Total tests per variant: {len(assets) * len(timeframes)}")
    print(f"   Total tests overall: {len(variant_files) * len(assets) * len(timeframes)}")
    
    if quick_mode:
        print(f"\n   [ZAP] QUICK MODE: Testing limited timeframes only")
    
    print("="*70)
    
    # Confirm
    confirm = input("\nProceed with variant backtests? (Y/N): ").strip().upper()
    if confirm != 'Y':
        print("Cancelled.")
        return
    
    # Initialize
    backtester = MultiTimeframeBacktester()
    all_results = []
    variant_summary = []
    
    start_time = datetime.now()
    
    # Test each variant
    for i, filepath in enumerate(variant_files):
        variant_id = filepath.stem  # e.g., "variant_01"
        
        print(f"\n{'-'*70}")
        print(f"[STATS] [{i+1}/{len(variant_files)}] Testing {variant_id}")
        print(f"{'-'*70}")
        
        # Load the variant class
        strategy_class, class_name, error = load_variant_class(filepath)
        
        if error:
            print(f"   [FAIL] Failed to load: {error}")
            variant_summary.append({
                'variant_id': variant_id,
                'status': 'load_failed',
                'error': error
            })
            continue
        
        print(f"   [OK] Loaded class: {class_name}")
        
        # Run backtests
        try:
            results = []
            test_count = 0
            success_count = 0
            
            for asset in assets:
                for timeframe in timeframes:
                    test_count += 1
                    print(f"   [{test_count}/{len(assets)*len(timeframes)}] {asset} {timeframe}...", end=" ")
                    
                    try:
                        result = backtester.run_single_backtest(
                            strategy_class=strategy_class,
                            symbol=asset,
                            timeframe=timeframe,
                            initial_cash=config.DEFAULT_INITIAL_CASH,
                            commission=config.DEFAULT_COMMISSION
                        )
                        
                        if result:
                            # Add variant tracking
                            result['variant_id'] = variant_id
                            result['strategy_name'] = class_name
                            results.append(result)
                            
                            # Save to database
                            backtester.db.save_backtest(result)
                            success_count += 1
                    
                    except Exception as e:
                        print(f"Error: {str(e)[:30]}")
            
            # Summarize this variant
            if results:
                avg_return = sum(r['total_return_pct'] for r in results) / len(results)
                best_return = max(r['total_return_pct'] for r in results)
                
                variant_summary.append({
                    'variant_id': variant_id,
                    'class_name': class_name,
                    'status': 'success',
                    'tests_run': success_count,
                    'avg_return': avg_return,
                    'best_return': best_return
                })
                
                all_results.extend(results)
                
                print(f"\n   [UP] {class_name}: Avg Return {avg_return:+.2f}%, Best {best_return:+.2f}%")
            else:
                variant_summary.append({
                    'variant_id': variant_id,
                    'class_name': class_name,
                    'status': 'no_results',
                    'tests_run': 0
                })
        
        except Exception as e:
            print(f"   [FAIL] Backtest error: {e}")
            traceback.print_exc()
            variant_summary.append({
                'variant_id': variant_id,
                'status': 'backtest_failed',
                'error': str(e)
            })
    
    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print("VARIANT BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Total results: {len(all_results)}")
    
    # Sort variants by average return
    successful_variants = [v for v in variant_summary if v.get('status') == 'success']
    successful_variants.sort(key=lambda x: x.get('avg_return', -999), reverse=True)
    
    if successful_variants:
        print(f"\n  [STATS] Variants Ranked by Average Return:")
        print(f"  {'-'*60}")
        for v in successful_variants[:10]:  # Top 10
            print(f"    {v['variant_id']:12} | {v['class_name']:30} | Avg: {v['avg_return']:+6.2f}%")
    
    # Show failures
    failed_variants = [v for v in variant_summary if v.get('status') != 'success']
    if failed_variants:
        print(f"\n  [WARN]  Failed Variants: {len(failed_variants)}")
        for v in failed_variants:
            print(f"    {v['variant_id']}: {v.get('status')} - {v.get('error', 'unknown')[:40]}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("  1. Run: python compare_variants.py   (detailed comparison)")
    print("  2. Run: streamlit run dashboard.py   (visual dashboard)")
    print(f"{'='*70}\n")
    
    return all_results, variant_summary


# ==============================================================================
# QUICK TEST MODE
# ==============================================================================

def quick_test():
    """Run a quick test on just 3 assets and 3 timeframes"""
    
    global TEST_ASSETS, TEST_TIMEFRAMES
    
    TEST_ASSETS = ['EUR-USD', 'BTC-USD', 'GBP-USD']
    TEST_TIMEFRAMES = ['1hour', '4hour', '1day']
    
    return run_variant_backtests(quick_mode=True)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtests on strategy variants')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (fewer assets/timeframes)')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        run_variant_backtests()
