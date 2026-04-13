# analyze_with_claude.py
# Send backtest results to Claude for AI analysis
# Asks for Y/N confirmation before using API credits

from anthropic import Anthropic
import json
import sqlite3
import config
from datetime import datetime

def get_recent_backtests(limit=None):
    """
    Retrieve recent backtest results from database
    """
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    query = """
        SELECT 
            strategy_name, symbol, timeframe,
            start_date, end_date, 
            total_return_pct, sharpe_ratio, max_drawdown_pct,
            total_trades, win_rate, profit_factor
        FROM backtests
        ORDER BY timestamp DESC
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return results

def count_by_category(results):
    """
    Count results by asset type and timeframe
    """
    by_timeframe = {}
    by_asset_type = {
        'Forex': [],
        'Indices': [],
        'Commodities': [],
        'Crypto': []
    }
    
    for r in results:
        # By timeframe
        tf = r.get('timeframe', 'unknown')
        if tf not in by_timeframe:
            by_timeframe[tf] = []
        by_timeframe[tf].append(r)
        
        # By asset type
        symbol = r['symbol']
        if symbol in config.FOREX_WATCHLIST:
            by_asset_type['Forex'].append(r)
        elif symbol in config.INDEX_WATCHLIST:
            by_asset_type['Indices'].append(r)
        elif symbol in config.COMMODITY_WATCHLIST:
            by_asset_type['Commodities'].append(r)
        elif symbol in config.CRYPTO_WATCHLIST:
            by_asset_type['Crypto'].append(r)
    
    return by_timeframe, by_asset_type

def estimate_cost(num_results):
    """
    Estimate API cost for analyzing results
    """
    # Rough estimate: ~200 tokens per result input + ~5000 tokens output
    input_tokens = num_results * 200 + 1000  # results + prompt
    output_tokens = 5000  # Claude's analysis
    
    input_cost = (input_tokens / 1_000_000) * 3  # $3 per million
    output_cost = (output_tokens / 1_000_000) * 15  # $15 per million
    
    total_cost = input_cost + output_cost
    return total_cost

def analyze_with_claude(results):
    """
    Send results to Claude for analysis
    """
    
    if not config.CLAUDE_API_KEY:
        print("[FAIL] ERROR: No Claude API key found!")
        print("   Check BacktestingAgent_API_KEY.txt")
        return None
    
    # Prepare data for Claude
    by_timeframe, by_asset_type = count_by_category(results)
    
    # Calculate summary statistics
    avg_return = sum(r['total_return_pct'] for r in results) / len(results)
    sharpe_results = [r for r in results if r['sharpe_ratio'] is not None]
    avg_sharpe = sum(r['sharpe_ratio'] for r in sharpe_results) / len(sharpe_results) if sharpe_results else None
    
    positive_returns = len([r for r in results if r['total_return_pct'] > 0])
    strong_performers = len([r for r in sharpe_results if r['sharpe_ratio'] > 0.5])
    
    # Build prompt
    prompt = f"""I just completed a comprehensive backtest of my trading strategy across multiple assets and timeframes.

**Strategy:** Simple Moving Average Crossover (10/30 period)

**Test Scope:**
- Total backtests: {len(results)}
- Asset classes: Forex ({len(by_asset_type['Forex'])}), Indices ({len(by_asset_type['Indices'])}), Commodities ({len(by_asset_type['Commodities'])}), Crypto ({len(by_asset_type['Crypto'])})
- Timeframes: {', '.join(by_timeframe.keys())}

**Aggregate Results:**
- Average Return: {avg_return:.2f}%
- Average Sharpe: {avg_sharpe:.2f if avg_sharpe else 'N/A'}
- Positive Returns: {positive_returns}/{len(results)} ({positive_returns/len(results)*100:.0f}%)
- Strong Performers (Sharpe > 0.5): {strong_performers}/{len(results)} ({strong_performers/len(results)*100:.0f}%)

**Top 10 Performers:**
{json.dumps(sorted(results, key=lambda x: x['total_return_pct'], reverse=True)[:10], indent=2)}

**Bottom 10 Performers:**
{json.dumps(sorted(results, key=lambda x: x['total_return_pct'])[:10], indent=2)}

**Performance by Timeframe:**
"""
    
    for tf, tf_results in sorted(by_timeframe.items()):
        tf_avg = sum(r['total_return_pct'] for r in tf_results) / len(tf_results)
        tf_positive = len([r for r in tf_results if r['total_return_pct'] > 0])
        prompt += f"\n- {tf}: Avg {tf_avg:+.2f}%, Positive {tf_positive}/{len(tf_results)}"
    
    prompt += "\n\n**Performance by Asset Class:**\n"
    
    for asset_type, type_results in by_asset_type.items():
        if type_results:
            type_avg = sum(r['total_return_pct'] for r in type_results) / len(type_results)
            type_positive = len([r for r in type_results if r['total_return_pct'] > 0])
            prompt += f"\n- {asset_type}: Avg {type_avg:+.2f}%, Positive {type_positive}/{len(type_results)}"
    
    prompt += """

Please provide a comprehensive analysis:

1. **Overall Assessment**: Is this MA crossover strategy viable, or does it show clear signs of failure?

2. **Asset Class Insights**: Which asset classes show promise? Which should I avoid?

3. **Timeframe Analysis**: Are there clear "winner" timeframes? Or is performance inconsistent?

4. **Red Flags**: What are the biggest concerns in these results? (e.g., overfitting, insufficient trades, poor risk-adjusted returns)

5. **Patterns & Correlations**: Do you see any meaningful patterns? (e.g., "crypto performs well on 5min but poorly on 4hour")

6. **Next Steps**: Given these results, what should I do next?
   - Should I abandon this strategy entirely?
   - Should I focus on specific asset/timeframe combinations?
   - What modifications would you suggest? (filters, different MA periods, risk management)

7. **Research Direction**: What should I prioritize when building the component library for the Mutation Agent?

Be brutally honest. If the strategy is fundamentally flawed, say so clearly."""

    # Call Claude API
    print("\n[AI] Sending results to Claude for analysis...")
    print("   This may take 10-30 seconds...\n")
    
    try:
        client = Anthropic(api_key=config.CLAUDE_API_KEY)
        
        message = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        analysis = message.content[0].text
        return analysis
        
    except Exception as e:
        print(f"[FAIL] Error calling Claude API: {e}")
        return None

def main():
    """
    Main analysis workflow with Y/N confirmation
    """
    
    print("\n" + "="*80)
    print("CLAUDE AI ANALYSIS")
    print("="*80)
    
    # Get recent results from database
    print("\n[STATS] Loading backtest results from database...")
    results = get_recent_backtests(limit=200)  # Get last 200 tests
    
    if not results:
        print("[FAIL] No backtest results found in database!")
        print("   Run: python run_backtests.py first")
        return
    
    print(f"[OK] Found {len(results)} backtest results")
    
    # Show what will be analyzed
    by_timeframe, by_asset_type = count_by_category(results)
    
    print(f"\nResults breakdown:")
    print(f"  - Forex:       {len(by_asset_type['Forex'])} tests")
    print(f"  - Indices:     {len(by_asset_type['Indices'])} tests")
    print(f"  - Commodities: {len(by_asset_type['Commodities'])} tests")
    print(f"  - Crypto:      {len(by_asset_type['Crypto'])} tests")
    print(f"\n  - Timeframes:  {', '.join(f'{tf}({len(r)})' for tf, r in by_timeframe.items())}")
    
    # Estimate cost
    estimated_cost = estimate_cost(len(results))
    print(f"\n[COST] Estimated API cost: ${estimated_cost:.3f}")
    print(f"   (Your $5 credit remaining: ~${5 - estimated_cost:.2f} after this)")
    
    # Quick preview of performance
    avg_return = sum(r['total_return_pct'] for r in results) / len(results)
    positive = len([r for r in results if r['total_return_pct'] > 0])
    strong = len([r for r in results if r.get('sharpe_ratio') and r['sharpe_ratio'] > 0.5])
    
    print(f"\n[UP] Quick Preview:")
    print(f"   Average Return:       {avg_return:+.2f}%")
    print(f"   Positive Returns:     {positive}/{len(results)} ({positive/len(results)*100:.0f}%)")
    print(f"   Strong (Sharpe>0.5):  {strong}/{len(results)} ({strong/len(results)*100:.0f}%)")
    
    print("\n" + "="*80)
    
    # Ask for confirmation
    response = input("\nSend these results to Claude for analysis? (Y/N): ").strip().upper()
    
    if response != 'Y':
        print("\n[FAIL] Analysis cancelled. No API credits used.")
        print("   You can run this script again anytime to analyze results.")
        return
    
    # Send to Claude
    analysis = analyze_with_claude(results)
    
    if analysis:
        print("\n" + "="*80)
        print("[AI] CLAUDE'S ANALYSIS")
        print("="*80)
        print(analysis)
        print("="*80 + "\n")
        
        # Offer to save analysis
        save = input("Save this analysis to a file? (Y/N): ").strip().upper()
        if save == 'Y':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/claude_analysis_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("="*80 + "\n")
                f.write("CLAUDE AI ANALYSIS\n")
                f.write("="*80 + "\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Results Analyzed: {len(results)}\n")
                f.write("="*80 + "\n\n")
                f.write(analysis)
            
            print(f"\n[OK] Analysis saved to: {filename}")
    
    else:
        print("\n[FAIL] Analysis failed. See error messages above.")
        print("   No API credits were used (failed before API call).")

if __name__ == "__main__":
    main()