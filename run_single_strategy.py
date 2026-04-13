# ==============================================================================
# run_single_strategy.py -- Backtest ONE strategy across all assets/timeframes
# ==============================================================================
# Usage:
#   python run_single_strategy.py strategies/discovered/my_strategy.py
#   python run_single_strategy.py strategies/variants/variant_01.py
#   python run_single_strategy.py strategies/variants/variant_01.py --quick
#   python run_single_strategy.py strategies/variants/variant_01.py --asset EUR-USD --tf 1hour
#
# This is the fastest way to test a single strategy without running 
# all 117 variants. Results go into the same backtest_results.db 
# that the dashboard reads from.
# ==============================================================================

import os
import sys
import argparse
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import config
from backtester_multi_timeframe import MultiTimeframeBacktester
from database import ResultsDatabase


def load_strategy(filepath):
    """Dynamically import a strategy class from a .py file."""
    import backtrader as bt
    
    path = Path(filepath)
    if not path.exists():
        print(f"[FAIL] File not found: {filepath}")
        return None, None
    
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        
        # Find the first class that inherits from bt.Strategy
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj != bt.Strategy:
                return obj, name
        
        print(f"[FAIL] No Strategy class found in {filepath}")
        return None, None
    except Exception as e:
        print(f"[FAIL] Could not load {filepath}: {e}")
        traceback.print_exc()
        return None, None


def get_assets(asset_filter=None):
    """Get list of assets to test."""
    if asset_filter:
        return [asset_filter]
    
    assets = []
    if config.FOREX_ENABLED:
        assets.extend(config.FOREX_WATCHLIST)
    if config.CRYPTO_ENABLED:
        assets.extend(config.CRYPTO_WATCHLIST)
    if config.INDICES_ENABLED:
        assets.extend(config.INDEX_WATCHLIST)
    return assets


def main():
    parser = argparse.ArgumentParser(description="Backtest a single strategy file")
    parser.add_argument("strategy_file", help="Path to the .py strategy file")
    parser.add_argument("--asset", help="Test only this asset (e.g. EUR-USD)", default=None)
    parser.add_argument("--tf", help="Test only this timeframe (e.g. 1hour)", default=None)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1hour + 4hour + daily only")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    # Load strategy
    strategy_class, class_name = load_strategy(args.strategy_file)
    if not strategy_class:
        sys.exit(1)
    
    variant_id = Path(args.strategy_file).stem
    
    # Determine assets and timeframes
    assets = get_assets(args.asset)
    
    if args.tf:
        timeframes = [args.tf]
    elif args.quick:
        timeframes = ["1hour", "4hour", "1day"]
    else:
        timeframes = ["1hour", "4hour", "1day"]  # sensible default
    
    total_tests = len(assets) * len(timeframes)
    
    print()
    print("=" * 60)
    print("  SINGLE STRATEGY BACKTESTER")
    print("=" * 60)
    print(f"  File:       {args.strategy_file}")
    print(f"  Class:      {class_name}")
    print(f"  Variant ID: {variant_id}")
    print(f"  Assets:     {len(assets)} ({', '.join(assets[:5])}{'...' if len(assets) > 5 else ''})")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Total runs: {total_tests}")
    print("=" * 60)
    
    if not args.yes:
        confirm = input("\nProceed? (Y/N): ").strip().upper()
        if confirm != "Y":
            print("Cancelled.")
            return
    
    # Run backtests
    backtester = MultiTimeframeBacktester()
    results = []
    errors = 0
    start = datetime.now()
    
    for i, asset in enumerate(assets):
        for j, timeframe in enumerate(timeframes):
            test_num = i * len(timeframes) + j + 1
            print(f"  [{test_num}/{total_tests}] {asset} {timeframe}...", end=" ")
            
            try:
                result = backtester.run_single_backtest(
                    strategy_class=strategy_class,
                    symbol=asset,
                    timeframe=timeframe,
                    initial_cash=config.DEFAULT_INITIAL_CASH,
                    commission=config.DEFAULT_COMMISSION,
                )
                
                if result:
                    result["variant_id"] = variant_id
                    result["strategy_name"] = class_name
                    results.append(result)
                    backtester.db.save_backtest(result)
                    
                    ret = result.get("total_return_pct", 0)
                    sr = result.get("sharpe_ratio", 0)
                    print(f"Return: {ret:+.2f}%  Sharpe: {sr:.2f}")
                else:
                    print("No result")
                    errors += 1
            except Exception as e:
                print(f"Error: {str(e)[:40]}")
                errors += 1
    
    elapsed = (datetime.now() - start).total_seconds()
    
    # Summary
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    
    if results:
        rets = [r.get("total_return_pct", 0) for r in results]
        srs = [r.get("sharpe_ratio", 0) for r in results if r.get("sharpe_ratio") is not None]
        dds = [r.get("max_drawdown_pct", 0) for r in results]
        wrs = [r.get("win_rate", 0) for r in results if r.get("win_rate") is not None]
        
        print(f"  Strategy:     {class_name}")
        print(f"  Tests run:    {len(results)} ({errors} errors)")
        print(f"  Time:         {elapsed:.1f}s")
        print(f"  Avg Return:   {sum(rets)/len(rets):+.2f}%")
        print(f"  Best Return:  {max(rets):+.2f}%")
        print(f"  Worst Return: {min(rets):+.2f}%")
        print(f"  Avg Sharpe:   {sum(srs)/len(srs):.2f}" if srs else "  Avg Sharpe:   N/A")
        print(f"  Avg DD:       {sum(dds)/len(dds):.1f}%")
        print(f"  Avg WR:       {sum(wrs)/len(wrs):.0f}%" if wrs else "  Avg WR:       N/A")
    else:
        print("  No results. Check errors above.")
    
    print("=" * 60)
    print(f"  Results saved to: {config.DATABASE_PATH}")
    print(f"  View in dashboard: python react_dashboard2.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()