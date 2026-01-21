# ==============================================================================
# test_system.py
# ==============================================================================
# FULL SYSTEM TEST RUNNER
#
# Runs through your entire trading research system step by step:
# 1. Config validation
# 2. Data manager (Forex, Crypto, Indices)
# 3. Database connection
# 4. Base strategy syntax check
# 5. Single backtest
# 6. Multi-asset backtest (small sample)
# 7. Mutation agent (DRY RUN - no API call)
# 8. Variant loading test
#
# After each step, tells you if it passed/failed and asks to continue.
#
# Usage: python test_system.py
# ==============================================================================

import sys
import os
import traceback
from pathlib import Path
from datetime import datetime

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_header(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_result(success, message=""):
    """Print pass/fail result"""
    if success:
        print(f"\n  ✅ PASSED {message}")
    else:
        print(f"\n  ❌ FAILED {message}")

def ask_continue():
    """Ask user if they want to continue to next test"""
    print(f"\n{'-'*70}")
    response = input("  Continue to next test? (Y/N): ").strip().upper()
    return response == 'Y'

def run_test(test_name, test_func):
    """Run a test function with error handling"""
    print_header(test_name)
    
    try:
        success, details = test_func()
        print_result(success, details if not success else "")
        return success
    except Exception as e:
        print(f"\n  ❌ EXCEPTION: {e}")
        traceback.print_exc()
        return False


# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

def test_config():
    """Test 1: Config validation"""
    print("  Loading config.py...")
    
    import config
    
    errors = []
    warnings = []
    
    # Check required attributes exist
    required_attrs = [
        'FOREX_ENABLED', 'CRYPTO_ENABLED', 'INDICES_ENABLED',
        'FOREX_WATCHLIST', 'CRYPTO_WATCHLIST', 'INDEX_WATCHLIST',
        'DATABASE_PATH', 'DATA_CACHE_PATH', 'CLAUDE_API_KEY'
    ]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            errors.append(f"Missing: {attr}")
        else:
            value = getattr(config, attr)
            print(f"    {attr}: {value if not 'KEY' in attr else '***configured***' if value else 'NOT SET'}")
    
    # Check directories
    print(f"\n  Checking directories...")
    
    if config.FOREX_ENABLED:
        forex_path = Path(config.FOREX_BASE_PATH)
        if forex_path.exists():
            files = list(forex_path.glob('*.csv')) + list(forex_path.glob('*.xlsx'))
            print(f"    Forex dir: ✅ Found ({len(files)} files)")
        else:
            warnings.append(f"Forex dir not found: {forex_path}")
            print(f"    Forex dir: ⚠️  Not found")
    
    if config.CRYPTO_ENABLED:
        crypto_path = Path(config.CACHE_SUBDIRS.get('crypto', ''))
        if crypto_path.exists():
            files = list(crypto_path.glob('**/*.csv'))
            print(f"    Crypto dir: ✅ Found ({len(files)} files)")
        else:
            warnings.append(f"Crypto dir not found: {crypto_path}")
            print(f"    Crypto dir: ⚠️  Not found")
    
    if config.INDICES_ENABLED:
        indices_path = Path(config.CACHE_SUBDIRS.get('indices', ''))
        if indices_path.exists():
            files = list(indices_path.glob('**/*.csv'))
            print(f"    Indices dir: ✅ Found ({len(files)} files)")
        else:
            warnings.append(f"Indices dir not found: {indices_path}")
            print(f"    Indices dir: ⚠️  Not found")
    
    # Check API key
    if not config.CLAUDE_API_KEY:
        warnings.append("Claude API key not set")
    
    if errors:
        return False, f"Errors: {', '.join(errors)}"
    elif warnings:
        print(f"\n  ⚠️  Warnings: {', '.join(warnings)}")
        return True, "(with warnings)"
    else:
        return True, ""


def test_data_manager():
    """Test 2: Data Manager"""
    print("  Initializing DataManager...")
    
    from data_manager import DataManager
    import config
    
    manager = DataManager()
    
    results = {'forex': None, 'crypto': None, 'indices': None}
    
    # Test Forex
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        print(f"\n  Testing Forex ({config.FOREX_WATCHLIST[0]})...")
        try:
            data = manager.get_data(config.FOREX_WATCHLIST[0], '1hour', 100)
            if data is not None and len(data) > 0:
                results['forex'] = len(data)
                print(f"    ✅ Got {len(data)} bars")
            else:
                results['forex'] = 0
                print(f"    ❌ No data returned")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results['forex'] = -1
    else:
        print(f"\n  Forex: Skipped (disabled)")
    
    # Test Crypto
    if config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        print(f"\n  Testing Crypto ({config.CRYPTO_WATCHLIST[0]})...")
        try:
            data = manager.get_data(config.CRYPTO_WATCHLIST[0], '1hour', 100)
            if data is not None and len(data) > 0:
                results['crypto'] = len(data)
                print(f"    ✅ Got {len(data)} bars")
            else:
                results['crypto'] = 0
                print(f"    ⚠️  No data (may need local files or CCXT)")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results['crypto'] = -1
    else:
        print(f"\n  Crypto: Skipped (disabled)")
    
    # Test Indices
    if config.INDICES_ENABLED and config.INDEX_WATCHLIST:
        print(f"\n  Testing Indices ({config.INDEX_WATCHLIST[0]})...")
        try:
            data = manager.get_data(config.INDEX_WATCHLIST[0], '1day', 100)
            if data is not None and len(data) > 0:
                results['indices'] = len(data)
                print(f"    ✅ Got {len(data)} bars")
            else:
                results['indices'] = 0
                print(f"    ⚠️  No data (check local files)")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results['indices'] = -1
    else:
        print(f"\n  Indices: Skipped (disabled)")
    
    # Determine overall result
    active_results = [v for v in results.values() if v is not None]
    if not active_results:
        return False, "No data sources enabled"
    
    successful = sum(1 for v in active_results if v and v > 0)
    total = len(active_results)
    
    if successful == total:
        return True, ""
    elif successful > 0:
        return True, f"({successful}/{total} sources working)"
    else:
        return False, "No data sources returned data"


def test_database():
    """Test 3: Database connection"""
    print("  Testing database connection...")
    
    from database import ResultsDatabase
    import config
    
    db = ResultsDatabase()
    
    # Check if database file exists/was created
    if Path(config.DATABASE_PATH).exists():
        print(f"    Database file: ✅ {config.DATABASE_PATH}")
    else:
        print(f"    Database file: ❌ Not found")
        return False, "Database file not created"
    
    # Try to get summary
    try:
        summary = db.get_backtest_summary()
        if summary:
            print(f"    Total backtests in DB: {summary.get('total_backtests', 0)}")
            print(f"    Unique strategies: {summary.get('unique_strategies', 0)}")
            print(f"    Unique variants: {summary.get('unique_variants', 0)}")
        else:
            print(f"    Database is empty (no backtests yet)")
        return True, ""
    except Exception as e:
        return False, f"Query failed: {e}"


def test_base_strategy():
    """Test 4: Base strategy syntax check"""
    print("  Loading base strategy...")
    
    try:
        from strategies.simple_strategy import SimpleMovingAverageCrossover
        import backtrader as bt
        
        # Check it's a valid strategy class
        if issubclass(SimpleMovingAverageCrossover, bt.Strategy):
            print(f"    Class: ✅ SimpleMovingAverageCrossover")
            print(f"    Inherits from: bt.Strategy")
            
            # Check params
            if hasattr(SimpleMovingAverageCrossover, 'params'):
                params = SimpleMovingAverageCrossover.params
                print(f"    Parameters: {dict(params._getitems()) if hasattr(params, '_getitems') else 'default'}")
            
            return True, ""
        else:
            return False, "Not a valid bt.Strategy subclass"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_single_backtest():
    """Test 5: Single backtest"""
    print("  Running single backtest (EUR-USD, 1hour, 500 bars)...")
    
    import config
    from backtester_multi_timeframe import MultiTimeframeBacktester
    from strategies.simple_strategy import SimpleMovingAverageCrossover
    
    # Find a working asset
    test_symbol = None
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        test_symbol = config.FOREX_WATCHLIST[0]
    elif config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        test_symbol = config.CRYPTO_WATCHLIST[0]
    
    if not test_symbol:
        return False, "No assets available for testing"
    
    print(f"    Testing with: {test_symbol}")
    
    backtester = MultiTimeframeBacktester()
    
    result = backtester.run_single_backtest(
        strategy_class=SimpleMovingAverageCrossover,
        symbol=test_symbol,
        timeframe='1hour',
        initial_cash=10000,
        commission=0.001
    )
    
    if result:
        print(f"\n    Results:")
        print(f"      Return:     {result.get('total_return_pct', 0):+.2f}%")
        print(f"      Trades:     {result.get('total_trades', 0)}")
        print(f"      Sharpe:     {result.get('sharpe_ratio', 'N/A')}")
        print(f"      Bars:       {result.get('bars_tested', 0)}")
        return True, ""
    else:
        return False, "Backtest returned no results"


def test_multi_backtest():
    """Test 6: Multi-asset backtest (small sample)"""
    print("  Running multi-asset backtest (2 assets, 2 timeframes)...")
    
    import config
    from backtester_multi_timeframe import MultiTimeframeBacktester
    from strategies.simple_strategy import SimpleMovingAverageCrossover
    
    # Get 2 assets
    test_assets = []
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        test_assets.extend(config.FOREX_WATCHLIST[:1])
    if config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        test_assets.extend(config.CRYPTO_WATCHLIST[:1])
    
    if len(test_assets) == 0:
        return False, "No assets available"
    
    test_timeframes = ['1hour', '4hour']
    
    print(f"    Assets: {test_assets}")
    print(f"    Timeframes: {test_timeframes}")
    print(f"    Total tests: {len(test_assets) * len(test_timeframes)}")
    
    backtester = MultiTimeframeBacktester()
    
    results = backtester.run_multi_asset_multi_timeframe(
        strategy_class=SimpleMovingAverageCrossover,
        assets=test_assets,
        timeframes=test_timeframes,
        initial_cash=10000,
        commission=0.001,
        save_to_db=False  # Don't save test results
    )
    
    if results and len(results) > 0:
        print(f"\n    Completed: {len(results)} backtests")
        avg_return = sum(r['total_return_pct'] for r in results) / len(results)
        print(f"    Avg return: {avg_return:+.2f}%")
        return True, ""
    else:
        return False, "No results returned"


def test_mutation_agent_dry():
    """Test 7: Mutation agent (DRY RUN - no API call)"""
    print("  Testing mutation agent components (no API call)...")
    
    # Test loading base strategy
    from mutate_strategy import load_base_strategy, get_performance_summary
    from mutation_config import get_all_ideas, get_ideas_list
    
    print(f"\n    Loading base strategy...")
    base_code = load_base_strategy()
    if base_code:
        print(f"      ✅ Loaded ({len(base_code)} chars)")
    else:
        return False, "Could not load base strategy"
    
    print(f"\n    Loading mutation ideas...")
    ideas = get_all_ideas()
    ideas_list = get_ideas_list()
    print(f"      ✅ Loaded {len(ideas_list)} ideas")
    
    print(f"\n    Loading performance summary...")
    performance = get_performance_summary()
    print(f"      ✅ Summary loaded ({len(performance)} chars)")
    
    # Check MUTATION_PROMPT has the coding rules
    from mutate_strategy import MUTATION_PROMPT
    
    print(f"\n    Checking prompt for Backtrader rules...")
    rules_present = []
    if 'bt.indicators.OBV' in MUTATION_PROMPT:
        rules_present.append('OBV')
    if 'position.price' in MUTATION_PROMPT:
        rules_present.append('position.price')
    if 'len(self)' in MUTATION_PROMPT:
        rules_present.append('min bars')
    
    if len(rules_present) >= 3:
        print(f"      ✅ Coding rules present: {', '.join(rules_present)}")
    else:
        print(f"      ⚠️  Some coding rules may be missing")
        return True, "(prompt may need Backtrader rules)"
    
    return True, ""


def test_variant_loading():
    """Test 8: Variant loading test"""
    print("  Testing variant file loading...")
    
    from pathlib import Path
    import importlib.util
    import sys
    
    variants_dir = Path(__file__).parent / 'strategies' / 'variants'
    
    if not variants_dir.exists():
        print(f"    Variants directory: ⚠️  Not found (will be created by mutation agent)")
        return True, "(no variants generated yet)"
    
    variant_files = list(variants_dir.glob('variant_*.py'))
    
    if not variant_files:
        print(f"    Variant files: ⚠️  None found (run mutation agent first)")
        return True, "(no variants generated yet)"
    
    print(f"    Found {len(variant_files)} variant files")
    
    # Try loading each
    loaded = 0
    failed = 0
    
    for filepath in variant_files[:5]:  # Test first 5 only
        try:
            spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded += 1
            print(f"      ✅ {filepath.name}")
        except Exception as e:
            failed += 1
            print(f"      ❌ {filepath.name}: {str(e)[:50]}")
    
    if failed == 0:
        return True, f"({loaded} variants loaded)"
    elif loaded > 0:
        return True, f"({loaded} loaded, {failed} failed)"
    else:
        return False, "All variants failed to load"


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all system tests"""
    
    print("\n" + "="*70)
    print("  🔧 TRADING RESEARCH SYSTEM - FULL TEST")
    print("="*70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print("="*70)
    
    tests = [
        ("1. Config Validation", test_config),
        ("2. Data Manager", test_data_manager),
        ("3. Database Connection", test_database),
        ("4. Base Strategy", test_base_strategy),
        ("5. Single Backtest", test_single_backtest),
        ("6. Multi-Asset Backtest", test_multi_backtest),
        ("7. Mutation Agent (Dry Run)", test_mutation_agent_dry),
        ("8. Variant Loading", test_variant_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        success = run_test(test_name, test_func)
        results.append((test_name, success))
        
        if not ask_continue():
            print("\n  🛑 Testing stopped by user")
            break
    
    # Final Summary
    print_header("FINAL SUMMARY")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n  Tests Run: {total}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {total - passed}")
    
    print(f"\n  Results:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"    {status}  {test_name}")
    
    if passed == total:
        print(f"\n  🎉 ALL TESTS PASSED!")
        print(f"\n  Your system is ready. Next steps:")
        print(f"    1. Run: python run_backtests.py")
        print(f"    2. Run: python mutate_strategy.py")
        print(f"    3. Run: python run_variant_backtests.py")
    elif passed > total // 2:
        print(f"\n  ⚠️  MOSTLY PASSED - Check failed tests above")
    else:
        print(f"\n  ❌ MULTIPLE FAILURES - Fix issues before proceeding")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()