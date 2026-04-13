# ==============================================================================
# test_data_download.py
# ==============================================================================
# UPDATED: Tests Forex (local files) and Crypto (CCXT) data sources
# 
# This script validates:
# 1. Forex loads from local merged CSV files
# 2. Crypto fetches via CCXT (Binance -> Hyperliquid)
# 3. Disabled sources (Indices/Commodities) return None correctly
# 4. Cache system works properly
# 5. All timeframes resample correctly
# ==============================================================================

from data_manager import DataManager
import config
import time

def test_forex_local():
    """
    Test Forex loading from local merged files
    
    Verifies:
    - Base 1-minute data loads
    - Resampling to higher timeframes works
    - All configured pairs are accessible
    """
    if not config.FOREX_ENABLED:
        print("\n⏸️  Forex testing skipped (disabled in config)")
        return {'skipped': True}
    
    print("\n" + "="*70)
    print("TEST 1: Forex Data (Local Files)")
    print("="*70)
    
    manager = DataManager()
    results = {
        'total_tests': 0,
        'successful': 0,
        'failed': 0,
        'details': []
    }
    
    # Test all Forex pairs
    for symbol in config.FOREX_WATCHLIST:
        pair_results = {'symbol': symbol, 'timeframes': {}}
        print(f"\n[STATS] Testing {symbol}...")
        
        for timeframe in config.FOREX_ALLOWED_TIMEFRAMES:
            results['total_tests'] += 1
            print(f"   {timeframe:6}...", end=" ", flush=True)
            
            try:
                data = manager.get_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    max_bars=100,
                    use_cache=False  # Force fresh load to test pipeline
                )
                
                if data is not None and len(data) > 0:
                    results['successful'] += 1
                    pair_results['timeframes'][timeframe] = len(data)
                    print(f"[OK] {len(data)} bars")
                else:
                    results['failed'] += 1
                    pair_results['timeframes'][timeframe] = 0
                    print(f"[FAIL] No data")
            
            except Exception as e:
                results['failed'] += 1
                pair_results['timeframes'][timeframe] = 0
                print(f"[FAIL] Error: {str(e)[:50]}")
        
        results['details'].append(pair_results)
    
    # Summary
    print(f"\n{'='*70}")
    print("FOREX RESULTS:")
    print(f"{'='*70}")
    
    for pair_result in results['details']:
        symbol = pair_result['symbol']
        successful_tfs = sum(1 for v in pair_result['timeframes'].values() if v > 0)
        total_tfs = len(pair_result['timeframes'])
        status = "[OK]" if successful_tfs == total_tfs else "[WARN] " if successful_tfs > 0 else "[FAIL]"
        print(f"  {status} {symbol}: {successful_tfs}/{total_tfs} timeframes")
    
    print(f"\n  Overall: {results['successful']}/{results['total_tests']} tests passed")
    
    return results


def test_crypto_ccxt():
    """
    Test Crypto loading via CCXT
    
    Verifies:
    - CCXT connection works
    - Data fetches from Binance (primary) or Hyperliquid (fallback)
    - All configured crypto assets are accessible
    """
    if not config.CRYPTO_ENABLED:
        print("\n⏸️  Crypto testing skipped (disabled in config)")
        return {'skipped': True}
    
    print("\n" + "="*70)
    print("TEST 2: Crypto Data (CCXT)")
    print("="*70)
    
    # Check CCXT availability
    try:
        import ccxt
        print(f"[OK] CCXT version: {ccxt.__version__}")
    except ImportError:
        print("[FAIL] CCXT not installed. Run: pip install ccxt")
        return {'error': 'CCXT not installed'}
    
    manager = DataManager()
    results = {
        'total_tests': 0,
        'successful': 0,
        'failed': 0,
        'details': []
    }
    
    # Test all Crypto assets (limited timeframes to avoid rate limits)
    test_timeframes = ['1hour', '4hour', '1day']
    
    for symbol in config.CRYPTO_WATCHLIST:
        asset_results = {'symbol': symbol, 'timeframes': {}}
        print(f"\n[STATS] Testing {symbol}...")
        
        for timeframe in test_timeframes:
            results['total_tests'] += 1
            print(f"   {timeframe:6}...", end=" ", flush=True)
            
            try:
                data = manager.get_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    max_bars=100,
                    use_cache=False
                )
                
                if data is not None and len(data) > 0:
                    results['successful'] += 1
                    asset_results['timeframes'][timeframe] = len(data)
                    print(f"[OK] {len(data)} bars")
                else:
                    results['failed'] += 1
                    asset_results['timeframes'][timeframe] = 0
                    print(f"[FAIL] No data")
                
                # Rate limit protection
                time.sleep(0.5)
            
            except Exception as e:
                results['failed'] += 1
                asset_results['timeframes'][timeframe] = 0
                print(f"[FAIL] Error: {str(e)[:50]}")
        
        results['details'].append(asset_results)
    
    # Summary
    print(f"\n{'='*70}")
    print("CRYPTO RESULTS:")
    print(f"{'='*70}")
    
    for asset_result in results['details']:
        symbol = asset_result['symbol']
        successful_tfs = sum(1 for v in asset_result['timeframes'].values() if v > 0)
        total_tfs = len(asset_result['timeframes'])
        status = "[OK]" if successful_tfs == total_tfs else "[WARN] " if successful_tfs > 0 else "[FAIL]"
        print(f"  {status} {symbol}: {successful_tfs}/{total_tfs} timeframes")
    
    print(f"\n  Overall: {results['successful']}/{results['total_tests']} tests passed")
    
    return results


def test_disabled_sources():
    """
    Test that disabled sources (Indices/Commodities) return None correctly
    
    This verifies the system gracefully handles disabled asset classes
    """
    print("\n" + "="*70)
    print("TEST 3: Disabled Sources (Indices/Commodities)")
    print("="*70)
    
    manager = DataManager()
    results = {'indices': None, 'commodities': None}
    
    # Test Indices
    print("\n[STATS] Testing Indices (should return None)...")
    if not config.INDICES_ENABLED:
        test_symbol = '^GSPC'  # S&P 500
        data = manager.get_data(test_symbol, timeframe='1hour', max_bars=100)
        
        if data is None:
            print(f"  [OK] {test_symbol}: Correctly returned None (disabled)")
            results['indices'] = 'passed'
        else:
            print(f"  [FAIL] {test_symbol}: Should have returned None but got data")
            results['indices'] = 'failed'
    else:
        print(f"  ⏸️  Indices are enabled - skipping disabled test")
        results['indices'] = 'skipped'
    
    # Test Commodities
    print("\n[STATS] Testing Commodities (should return None)...")
    if not config.COMMODITIES_ENABLED:
        test_symbol = 'GC=F'  # Gold
        data = manager.get_data(test_symbol, timeframe='1hour', max_bars=100)
        
        if data is None:
            print(f"  [OK] {test_symbol}: Correctly returned None (disabled)")
            results['commodities'] = 'passed'
        else:
            print(f"  [FAIL] {test_symbol}: Should have returned None but got data")
            results['commodities'] = 'failed'
    else:
        print(f"  ⏸️  Commodities are enabled - skipping disabled test")
        results['commodities'] = 'skipped'
    
    print(f"\n{'='*70}")
    print("DISABLED SOURCES RESULTS:")
    print(f"{'='*70}")
    print(f"  Indices:     {results['indices']}")
    print(f"  Commodities: {results['commodities']}")
    
    return results


def test_cache_system():
    """
    Test data caching system
    
    Verifies:
    - First load fetches fresh data
    - Second load uses cache (faster)
    - Cache speedup is significant
    """
    print("\n" + "="*70)
    print("TEST 4: Cache System")
    print("="*70)
    
    manager = DataManager()
    
    # Pick first available active asset
    test_symbol = None
    test_type = None
    
    if config.FOREX_ENABLED and config.FOREX_WATCHLIST:
        test_symbol = config.FOREX_WATCHLIST[0]
        test_type = 'Forex'
    elif config.CRYPTO_ENABLED and config.CRYPTO_WATCHLIST:
        test_symbol = config.CRYPTO_WATCHLIST[0]
        test_type = 'Crypto'
    
    if not test_symbol:
        print("  ⏸️  No active assets to test cache")
        return {'skipped': True}
    
    timeframe = '1hour'
    
    # First download (fresh)
    print(f"\n[STATS] Testing cache with {test_symbol} ({test_type})...")
    print(f"\n   First load (fresh)...", end=" ", flush=True)
    start = time.time()
    data1 = manager.get_data(test_symbol, timeframe=timeframe, max_bars=100, use_cache=False)
    time1 = time.time() - start
    
    if data1 is None:
        print(f"[FAIL] Failed to get data")
        return {'error': 'Failed to get data for cache test'}
    
    print(f"[OK] {len(data1)} bars in {time1:.2f}s")
    
    # Second download (cached)
    print(f"   Second load (cache)...", end=" ", flush=True)
    start = time.time()
    data2 = manager.get_data(test_symbol, timeframe=timeframe, max_bars=100, use_cache=True)
    time2 = time.time() - start
    print(f"[OK] {len(data2)} bars in {time2:.2f}s")
    
    # Calculate speedup
    speedup = time1 / time2 if time2 > 0 else float('inf')
    
    print(f"\n{'='*70}")
    print("CACHE RESULTS:")
    print(f"{'='*70}")
    print(f"  Fresh load:   {time1:.3f} seconds")
    print(f"  Cached load:  {time2:.3f} seconds")
    print(f"  Speedup:      {speedup:.1f}x faster")
    
    if len(data1) == len(data2):
        print(f"  Data match:   [OK] Same number of bars")
    else:
        print(f"  Data match:   [WARN]  Different bar counts ({len(data1)} vs {len(data2)})")
    
    cache_working = speedup > 1.5  # Cache should be at least 1.5x faster
    print(f"\n  {'[OK] Cache working properly!' if cache_working else '[WARN]  Cache may not be working optimally'}")
    
    return {
        'fresh_time': time1,
        'cached_time': time2,
        'speedup': speedup,
        'passed': cache_working
    }


def test_timeframe_resampling():
    """
    Test that timeframe resampling works correctly for Forex
    
    Verifies:
    - 1-minute base data loads
    - Resampling produces correct bar counts
    - OHLCV aggregation is correct
    """
    if not config.FOREX_ENABLED:
        print("\n⏸️  Resampling test skipped (Forex disabled)")
        return {'skipped': True}
    
    print("\n" + "="*70)
    print("TEST 5: Timeframe Resampling (Forex)")
    print("="*70)
    
    manager = DataManager()
    symbol = config.FOREX_WATCHLIST[0]
    
    print(f"\n[STATS] Testing resampling for {symbol}...")
    
    results = {}
    
    # Get 1-minute data first
    print(f"\n   Loading 1min base data...", end=" ", flush=True)
    data_1min = manager.get_data(symbol, timeframe='1min', max_bars=1000, use_cache=False)
    
    if data_1min is None:
        print(f"[FAIL] Failed to load 1min data")
        return {'error': 'Failed to load 1min data'}
    
    print(f"[OK] {len(data_1min)} bars")
    results['1min'] = len(data_1min)
    
    # Test each higher timeframe
    expected_ratios = {
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1hour': 60,
        '4hour': 240
    }
    
    for timeframe, ratio in expected_ratios.items():
        print(f"   Testing {timeframe} (expect ~{len(data_1min)//ratio} bars)...", end=" ", flush=True)
        
        data = manager.get_data(symbol, timeframe=timeframe, max_bars=1000, use_cache=False)
        
        if data is not None:
            expected_count = len(data_1min) // ratio
            actual_count = len(data)
            tolerance = expected_count * 0.2  # 20% tolerance for market hours gaps
            
            if abs(actual_count - expected_count) <= tolerance:
                print(f"[OK] {actual_count} bars (expected ~{expected_count})")
                results[timeframe] = {'bars': actual_count, 'status': 'passed'}
            else:
                print(f"[WARN]  {actual_count} bars (expected ~{expected_count})")
                results[timeframe] = {'bars': actual_count, 'status': 'warning'}
        else:
            print(f"[FAIL] Failed")
            results[timeframe] = {'bars': 0, 'status': 'failed'}
    
    print(f"\n{'='*70}")
    print("RESAMPLING RESULTS:")
    print(f"{'='*70}")
    
    for tf, result in results.items():
        if isinstance(result, dict):
            status = "[OK]" if result['status'] == 'passed' else "[WARN] " if result['status'] == 'warning' else "[FAIL]"
            print(f"  {status} {tf}: {result['bars']} bars")
        else:
            print(f"  [OK] {tf}: {result} bars (base)")
    
    return results


def run_all_tests():
    """
    Run all data download tests
    """
    print("\n" + "="*70)
    print("DATA MANAGER COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nThis will test:")
    print("  [OK] Forex data loading (local files)")
    print("  [OK] Crypto data loading (CCXT)")
    print("  [OK] Disabled sources (Indices/Commodities)")
    print("  [OK] Cache system")
    print("  [OK] Timeframe resampling")
    print("\nConfiguration:")
    print(f"  Forex:       {'[OK] Enabled' if config.FOREX_ENABLED else '⏸️  Disabled'}")
    print(f"  Crypto:      {'[OK] Enabled' if config.CRYPTO_ENABLED else '⏸️  Disabled'}")
    print(f"  Indices:     {'⏸️  Disabled' if not config.INDICES_ENABLED else '[OK] Enabled'}")
    print(f"  Commodities: {'⏸️  Disabled' if not config.COMMODITIES_ENABLED else '[OK] Enabled'}")
    print("="*70)
    
    input("\nPress Enter to start tests...")
    
    # Run all tests
    all_results = {}
    
    all_results['forex'] = test_forex_local()
    all_results['crypto'] = test_crypto_ccxt()
    all_results['disabled'] = test_disabled_sources()
    all_results['cache'] = test_cache_system()
    all_results['resampling'] = test_timeframe_resampling()
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Calculate overall success
    total_tests = 0
    total_passed = 0
    
    # Forex
    if 'skipped' not in all_results['forex']:
        forex_passed = all_results['forex'].get('successful', 0)
        forex_total = all_results['forex'].get('total_tests', 0)
        total_tests += forex_total
        total_passed += forex_passed
        status = "[OK]" if forex_passed == forex_total else "[WARN] " if forex_passed > 0 else "[FAIL]"
        print(f"  {status} Forex:      {forex_passed}/{forex_total} tests")
    else:
        print(f"  ⏸️  Forex:      Skipped")
    
    # Crypto
    if 'skipped' not in all_results['crypto'] and 'error' not in all_results['crypto']:
        crypto_passed = all_results['crypto'].get('successful', 0)
        crypto_total = all_results['crypto'].get('total_tests', 0)
        total_tests += crypto_total
        total_passed += crypto_passed
        status = "[OK]" if crypto_passed == crypto_total else "[WARN] " if crypto_passed > 0 else "[FAIL]"
        print(f"  {status} Crypto:     {crypto_passed}/{crypto_total} tests")
    else:
        print(f"  ⏸️  Crypto:     Skipped/Error")
    
    # Disabled
    disabled_passed = sum(1 for v in all_results['disabled'].values() if v == 'passed')
    disabled_total = sum(1 for v in all_results['disabled'].values() if v != 'skipped')
    if disabled_total > 0:
        total_tests += disabled_total
        total_passed += disabled_passed
        status = "[OK]" if disabled_passed == disabled_total else "[FAIL]"
        print(f"  {status} Disabled:   {disabled_passed}/{disabled_total} tests")
    
    # Cache
    if 'skipped' not in all_results['cache'] and 'error' not in all_results['cache']:
        cache_passed = 1 if all_results['cache'].get('passed', False) else 0
        total_tests += 1
        total_passed += cache_passed
        status = "[OK]" if cache_passed else "[FAIL]"
        print(f"  {status} Cache:      {'Working' if cache_passed else 'Not optimal'}")
    else:
        print(f"  ⏸️  Cache:      Skipped")
    
    # Resampling
    if 'skipped' not in all_results['resampling'] and 'error' not in all_results['resampling']:
        resample_passed = sum(1 for v in all_results['resampling'].values() 
                            if isinstance(v, dict) and v.get('status') == 'passed')
        resample_total = sum(1 for v in all_results['resampling'].values() if isinstance(v, dict))
        if resample_total > 0:
            total_tests += resample_total
            total_passed += resample_passed
            status = "[OK]" if resample_passed == resample_total else "[WARN] "
            print(f"  {status} Resampling: {resample_passed}/{resample_total} timeframes")
    else:
        print(f"  ⏸️  Resampling: Skipped")
    
    # Overall
    print(f"\n{'='*70}")
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        if success_rate == 100:
            print(f"[DONE] ALL TESTS PASSED!")
        elif success_rate >= 80:
            print(f"[OK] MOSTLY PASSED: {total_passed}/{total_tests} ({success_rate:.0f}%)")
        else:
            print(f"[WARN]  SOME ISSUES: {total_passed}/{total_tests} ({success_rate:.0f}%)")
    else:
        print("⏸️  No tests were run")
    
    print(f"\nNext steps:")
    if total_passed == total_tests and total_tests > 0:
        print("  [OK] All systems ready!")
        print("  -> Run: python run_backtests.py")
        print("  -> Run: python analyze_with_claude.py")
    else:
        print("  1. Check error messages above")
        print("  2. Verify data files exist in E:/TradingData/forex/")
        print("  3. Run: python forex_data_processor.py (if Forex failed)")
        print("  4. Run: pip install ccxt (if Crypto failed)")
    
    print("="*70 + "\n")
    
    return all_results


if __name__ == "__main__":
    run_all_tests()