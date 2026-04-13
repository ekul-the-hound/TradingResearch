# ==============================================================================
# run_backtests.py
# ==============================================================================
# Main orchestrator script for running comprehensive backtests
#
# Features:
# - Tests only ACTIVE asset classes (Forex + Crypto by default)
# - Runs across all configured timeframes
# - Saves results to SQLite database
# - Provides summary statistics
# - Does NOT call Claude API (use analyze_with_claude.py separately)
# ==============================================================================

from backtester_multi_timeframe import MultiTimeframeBacktester
from strategies.simple_strategy import SimpleMovingAverageCrossover
import config
from datetime import datetime

def main():
    """
    Run backtests on ACTIVE assets only (Forex + Crypto)
    
    This script:
    1. Builds list of active assets based on config flags
    2. Runs backtests across all timeframes
    3. Saves results to database
    4. Prints summary statistics
    
    Does NOT use Claude API - run analyze_with_claude.py separately for AI analysis
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST SESSION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStrategy: {SimpleMovingAverageCrossover.__name__}")
    
    # ========================================================================
    # BUILD ACTIVE ASSET LIST
    # ========================================================================
    active_assets = []
    
    print(f"\nAssets to test:")
    
    # Forex
    if config.FOREX_ENABLED:
        active_assets.extend(config.FOREX_WATCHLIST)
        print(f"  [OK] Forex:       {len(config.FOREX_WATCHLIST)} pairs (local files)")
    else:
        print(f"  ⏸️  Forex:       DISABLED")
    
    # Crypto
    if config.CRYPTO_ENABLED:
        active_assets.extend(config.CRYPTO_WATCHLIST)
        print(f"  [OK] Crypto:      {len(config.CRYPTO_WATCHLIST)} currencies (CCXT)")
    else:
        print(f"  ⏸️  Crypto:      DISABLED")
    
    # Indices (disabled)
    if config.INDICES_ENABLED:
        active_assets.extend(config.INDEX_WATCHLIST)
        print(f"  [OK] Indices:     {len(config.INDEX_WATCHLIST)} indices")
    else:
        print(f"  ⏸️  Indices:     DISABLED (awaiting IBKR)")
    
    # Commodities (disabled)
    if config.COMMODITIES_ENABLED:
        active_assets.extend(config.COMMODITY_WATCHLIST)
        print(f"  [OK] Commodities: {len(config.COMMODITY_WATCHLIST)} commodities")
    else:
        print(f"  ⏸️  Commodities: DISABLED (awaiting futures source)")
    
    print(f"\n  TOTAL ACTIVE:  {len(active_assets)} assets")
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    if not active_assets:
        print("\n[FAIL] No active assets to test!")
        print("   Enable Forex or Crypto in config.py:")
        print("   - Set FOREX_ENABLED = True")
        print("   - Set CRYPTO_ENABLED = True")
        return
    
    # ========================================================================
    # CALCULATE TEST SCOPE
    # ========================================================================
    timeframes = list(config.TIMEFRAMES.keys())
    total_tests = len(active_assets) * len(timeframes)
    
    print(f"\nTimeframes: {', '.join(timeframes)}")
    print(f"\nTotal backtests: {total_tests}")
    print(f"Estimated time:  {total_tests * 0.5 / 60:.0f}-{total_tests * 1.0 / 60:.0f} minutes")
    print("="*80)
    
    # ========================================================================
    # CONFIRMATION
    # ========================================================================
    response = input("\nProceed with backtests? (Y/N): ").strip().upper()
    
    if response != 'Y':
        print("[FAIL] Backtests cancelled.")
        return
    
    print("\n[LAUNCH] Starting backtests...\n")
    
    # ========================================================================
    # RUN BACKTESTS
    # ========================================================================
    backtester = MultiTimeframeBacktester()
    
    results = backtester.run_multi_asset_multi_timeframe(
        strategy_class=SimpleMovingAverageCrossover,
        assets=active_assets,
        timeframes=timeframes,
        initial_cash=config.DEFAULT_INITIAL_CASH,
        commission=config.DEFAULT_COMMISSION,
        save_to_db=True
    )
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    stats = backtester.get_summary_stats(results)
    
    if stats:
        # Overall Performance
        print(f"\nOverall Performance:")
        print(f"  Tests Completed:    {stats['total_tests']}")
        print(f"  Average Return:     {stats['avg_return']:+.2f}%")
        if stats['avg_sharpe']:
            print(f"  Average Sharpe:     {stats['avg_sharpe']:.2f}")
        print(f"  Average Drawdown:   {stats['avg_drawdown']:.2f}%")
        print(f"  Average Trades:     {stats['avg_trades']:.1f}")
        
        # Best Performer
        print(f"\nBest Performer:")
        best = stats['best_performer']
        print(f"  {best['symbol']:12} {best['timeframe']:6} | {best['total_return_pct']:+.2f}%")
        
        # Worst Performer
        print(f"\nWorst Performer:")
        worst = stats['worst_performer']
        print(f"  {worst['symbol']:12} {worst['timeframe']:6} | {worst['total_return_pct']:+.2f}%")
        
        # Performance by Timeframe
        print(f"\nPerformance by Timeframe:")
        for tf, tf_stats in sorted(stats['timeframe_stats'].items()):
            positive_pct = (tf_stats['positive_count'] / tf_stats['count']) * 100 if tf_stats['count'] > 0 else 0
            print(f"  {tf:6} | Avg Return: {tf_stats['avg_return']:+6.2f}% | "
                  f"Positive: {tf_stats['positive_count']}/{tf_stats['count']} ({positive_pct:.0f}%)")
        
        # Strong Performers (Sharpe > 0.5)
        strong_performers = [r for r in results if r.get('sharpe_ratio') and r['sharpe_ratio'] > 0.5]
        print(f"\n✨ Strong Performers (Sharpe > 0.5): {len(strong_performers)}/{len(results)}")
        
        if strong_performers:
            print("\nTop 5 by Sharpe Ratio:")
            sorted_by_sharpe = sorted(strong_performers, key=lambda x: x['sharpe_ratio'], reverse=True)[:5]
            for r in sorted_by_sharpe:
                print(f"  {r['symbol']:12} {r['timeframe']:6} | "
                      f"Sharpe: {r['sharpe_ratio']:5.2f} | Return: {r['total_return_pct']:+6.2f}%")
    else:
        print("\n[WARN]  No results to summarize")
    
    # ========================================================================
    # COMPLETION MESSAGE
    # ========================================================================
    print("\n" + "="*80)
    print("[OK] BACKTESTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to database: {config.DATABASE_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Review the summary above")
    print(f"  2. Run: python analyze_with_claude.py  (to get AI analysis)")
    print(f"  3. Run: python results_analyzer.py     (to query specific results)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()