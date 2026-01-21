# ==============================================================================
# compare_variants.py
# ==============================================================================
# VARIANT COMPARISON TOOL
# 
# This script:
# 1. Loads all backtest results from database
# 2. Compares variants against the base strategy
# 3. Ranks variants by multiple metrics
# 4. Identifies the best performers
# 5. Optionally sends to Claude for deeper analysis
#
# Cost: FREE (unless you use Claude analysis option)
# ==============================================================================

import sqlite3
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

import config

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_all_results():
    """Load all backtest results from database"""
    
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM backtest_results
        ORDER BY timestamp DESC
    """)
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results


def group_by_variant(results):
    """Group results by variant_id/strategy_name"""
    
    groups = {}
    
    for r in results:
        # Use variant_id if available, otherwise strategy_name
        key = r.get('variant_id') or r.get('strategy_name', 'Unknown')
        
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    return groups


def calculate_variant_stats(results):
    """Calculate aggregate statistics for a variant"""
    
    if not results:
        return None
    
    returns = [r['total_return_pct'] for r in results if r['total_return_pct'] is not None]
    sharpes = [r['sharpe_ratio'] for r in results if r['sharpe_ratio'] is not None]
    win_rates = [r['win_rate'] for r in results if r['win_rate'] is not None]
    drawdowns = [r['max_drawdown_pct'] for r in results if r['max_drawdown_pct'] is not None]
    trades = [r['total_trades'] for r in results if r['total_trades'] is not None]
    
    stats = {
        'strategy_name': results[0].get('strategy_name', 'Unknown'),
        'tests_count': len(results),
        'avg_return': sum(returns) / len(returns) if returns else 0,
        'best_return': max(returns) if returns else 0,
        'worst_return': min(returns) if returns else 0,
        'avg_sharpe': sum(sharpes) / len(sharpes) if sharpes else None,
        'avg_win_rate': sum(win_rates) / len(win_rates) if win_rates else None,
        'avg_drawdown': sum(drawdowns) / len(drawdowns) if drawdowns else 0,
        'avg_trades': sum(trades) / len(trades) if trades else 0,
        'positive_tests': len([r for r in returns if r > 0]),
        'positive_rate': len([r for r in returns if r > 0]) / len(returns) * 100 if returns else 0
    }
    
    return stats


def rank_variants(variant_stats, sort_by='avg_return'):
    """Rank variants by a specific metric"""
    
    # Filter out None values for the sort key
    valid_stats = [v for v in variant_stats if v.get(sort_by) is not None]
    
    # Sort descending (higher is better for most metrics)
    reverse = True
    if sort_by in ['avg_drawdown', 'worst_return']:
        reverse = False  # Lower is better for these
    
    return sorted(valid_stats, key=lambda x: x.get(sort_by, -999), reverse=reverse)


def get_best_individual_results(results, top_n=10):
    """Get the top N individual backtest results"""
    
    sorted_results = sorted(
        [r for r in results if r['total_return_pct'] is not None],
        key=lambda x: x['total_return_pct'],
        reverse=True
    )
    
    return sorted_results[:top_n]


def display_comparison_table(variant_stats, base_key='SimpleMovingAverageCrossover'):
    """Display a formatted comparison table"""
    
    # Find base strategy stats
    base_stats = None
    for key, stats in variant_stats.items():
        if base_key in key or key == base_key:
            base_stats = stats
            break
    
    base_return = base_stats['avg_return'] if base_stats else 0
    
    # Prepare table data
    table_data = []
    
    for key, stats in sorted(variant_stats.items(), key=lambda x: x[1]['avg_return'], reverse=True):
        improvement = stats['avg_return'] - base_return
        
        table_data.append([
            key[:25],  # Truncate long names
            stats['strategy_name'][:30] if stats['strategy_name'] else 'N/A',
            f"{stats['avg_return']:+.2f}%",
            f"{stats['avg_sharpe']:.2f}" if stats['avg_sharpe'] else 'N/A',
            f"{stats['avg_win_rate']:.1f}%" if stats['avg_win_rate'] else 'N/A',
            f"{stats['avg_drawdown']:.2f}%",
            f"{stats['positive_rate']:.0f}%",
            f"{improvement:+.2f}%" if base_stats else 'N/A'
        ])
    
    headers = ['Variant', 'Class Name', 'Avg Return', 'Sharpe', 'Win Rate', 'Drawdown', '% Positive', 'vs Base']
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


# ==============================================================================
# MAIN COMPARISON FUNCTION
# ==============================================================================

def compare_variants(show_individual=True):
    """Main comparison function"""
    
    print("\n" + "="*80)
    print("📊 VARIANT COMPARISON REPORT")
    print("="*80)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load results
    results = load_all_results()
    
    if not results:
        print("❌ No backtest results found in database.")
        print("   Run: python run_backtests.py")
        print("   Then: python run_variant_backtests.py")
        return
    
    print(f"\n📂 Loaded {len(results)} backtest results from database")
    
    # Group by variant
    grouped = group_by_variant(results)
    print(f"📋 Found {len(grouped)} unique strategies/variants")
    
    # Calculate stats for each variant
    variant_stats = {}
    for key, group_results in grouped.items():
        stats = calculate_variant_stats(group_results)
        if stats:
            variant_stats[key] = stats
    
    # Display comparison table
    print(f"\n{'─'*80}")
    print("VARIANT RANKINGS (by Average Return)")
    print(f"{'─'*80}\n")
    
    display_comparison_table(variant_stats)
    
    # Find the winner
    ranked = rank_variants(list(variant_stats.values()), sort_by='avg_return')
    
    if ranked:
        winner = ranked[0]
        print(f"\n{'─'*80}")
        print(f"🏆 BEST PERFORMER: {winner['strategy_name']}")
        print(f"{'─'*80}")
        print(f"   Average Return:  {winner['avg_return']:+.2f}%")
        print(f"   Average Sharpe:  {winner['avg_sharpe']:.2f}" if winner['avg_sharpe'] else "   Average Sharpe:  N/A")
        print(f"   Win Rate:        {winner['avg_win_rate']:.1f}%" if winner['avg_win_rate'] else "   Win Rate:        N/A")
        print(f"   Max Drawdown:    {winner['avg_drawdown']:.2f}%")
        print(f"   Positive Tests:  {winner['positive_rate']:.0f}% ({winner['positive_tests']}/{winner['tests_count']})")
    
    # Show top individual results
    if show_individual:
        print(f"\n{'─'*80}")
        print("TOP 10 INDIVIDUAL BACKTEST RESULTS")
        print(f"{'─'*80}\n")
        
        top_results = get_best_individual_results(results, top_n=10)
        
        table_data = []
        for r in top_results:
            variant = r.get('variant_id') or r.get('strategy_name', 'Unknown')
            table_data.append([
                variant[:20],
                r['symbol'],
                r['timeframe'],
                f"{r['total_return_pct']:+.2f}%",
                f"{r['sharpe_ratio']:.2f}" if r['sharpe_ratio'] else 'N/A',
                r['total_trades']
            ])
        
        headers = ['Variant', 'Asset', 'Timeframe', 'Return', 'Sharpe', 'Trades']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Recommendation
    print(f"\n{'─'*80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'─'*80}")
    
    if ranked and ranked[0]['avg_return'] > 0:
        print(f"   ✅ {ranked[0]['strategy_name']} shows positive average returns")
        print(f"   → Consider testing on out-of-sample data")
        print(f"   → Review individual results on best-performing assets")
    else:
        print(f"   ⚠️  No variants showing positive average returns")
        print(f"   → Consider adding more modifications to mutation_config.py")
        print(f"   → Try different base strategy approaches")
        print(f"   → Run mutation agent again with new ideas")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("  1. Run: streamlit run dashboard.py   (visual charts)")
    print("  2. Edit: mutation_config.py          (add new ideas)")
    print("  3. Run: python mutate_strategy.py    (generate more variants)")
    print(f"{'='*80}\n")
    
    return variant_stats, ranked


# ==============================================================================
# CLAUDE ANALYSIS (Optional)
# ==============================================================================

def analyze_with_claude(variant_stats, ranked):
    """Send results to Claude for deeper analysis"""
    
    print("\n" + "-"*70)
    print("🤖 CLAUDE AI ANALYSIS")
    print("-"*70)
    print("This will send your results to Claude for deeper analysis.")
    print("Estimated cost: ~$0.15-0.25")
    print("-"*70)
    
    confirm = input("\nProceed with Claude analysis? (Y/N): ").strip().upper()
    if confirm != 'Y':
        print("Skipped.")
        return
    
    from anthropic import Anthropic
    
    # Build summary for Claude
    summary = "## VARIANT COMPARISON RESULTS:\n\n"
    
    for stats in ranked[:10]:  # Top 10
        summary += f"**{stats['strategy_name']}**\n"
        summary += f"- Avg Return: {stats['avg_return']:+.2f}%\n"
        summary += f"- Avg Sharpe: {stats['avg_sharpe']:.2f}\n" if stats['avg_sharpe'] else ""
        summary += f"- Win Rate: {stats['avg_win_rate']:.1f}%\n" if stats['avg_win_rate'] else ""
        summary += f"- Positive Tests: {stats['positive_rate']:.0f}%\n\n"
    
    prompt = f"""Analyze these trading strategy backtest results and provide insights:

{summary}

Please provide:
1. Why you think the top performer outperformed
2. Patterns you notice across variants
3. Suggestions for further improvements
4. Risks or concerns to watch for
5. Recommended next steps

Be specific and actionable in your analysis."""
    
    client = Anthropic(api_key=config.CLAUDE_API_KEY)
    
    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        print("\n" + "="*70)
        print("CLAUDE'S ANALYSIS:")
        print("="*70)
        print(response.content[0].text)
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ API error: {e}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare strategy variants')
    parser.add_argument('--ai', action='store_true', help='Include Claude AI analysis')
    parser.add_argument('--no-individual', action='store_true', help='Skip individual results table')
    
    args = parser.parse_args()
    
    # Install tabulate if not present
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tabulate'])
        from tabulate import tabulate
    
    variant_stats, ranked = compare_variants(show_individual=not args.no_individual)
    
    if args.ai and ranked:
        analyze_with_claude(variant_stats, ranked)
