# ==============================================================================
# mutate_strategy.py
# ==============================================================================
# MUTATION AGENT
# 
# This script:
# 1. Reads your base strategy code
# 2. Reads your mutation ideas from mutation_config.py
# 3. Loads recent backtest results
# 4. Asks Claude to generate 15 variant strategies
# 5. Saves each variant as a separate .py file
#
# UPDATED: Added Backtrader-specific coding rules to prevent common bugs:
# - OBV indicator name (bt.indicators.OBV, not OnBalanceVolume)
# - Position price checks (avoid NoneType errors)
# - Minimum bar checks for indicators
#
# Cost: ~$0.15-0.30 per run
# ==============================================================================

import os
import re
import json
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic

import config
from mutation_config import get_all_ideas
from database import ResultsDatabase

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# How many variants to generate
NUM_VARIANTS = 15

# Where to save variants
VARIANTS_DIR = Path(__file__).parent / 'strategies' / 'variants'

# Create variants directory if it doesn't exist
VARIANTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# PROMPT TEMPLATE (UPDATED WITH BACKTRADER CODING RULES)
# ==============================================================================

MUTATION_PROMPT = """You are an expert quantitative trading strategy developer specializing in Backtrader.

## YOUR TASK:
Generate {num_variants} variant strategies based on the base strategy below. Each variant should make 1-3 modifications using the available ideas.

## BASE STRATEGY CODE:
```python
{base_strategy_code}
```

## BASE STRATEGY PERFORMANCE:
{performance_summary}

## AVAILABLE MODIFICATION IDEAS:
{mutation_ideas}

## REQUIREMENTS FOR EACH VARIANT:

1. **Must be a complete, working Backtrader strategy class**
2. **Must inherit from bt.Strategy**
3. **Must include all necessary imports at the top**
4. **Must have a unique class name: Variant01_Description, Variant02_Description, etc.**
5. **Must include a docstring explaining what was changed and why**
6. **Must be syntactically correct Python**

## CRITICAL BACKTRADER CODING RULES (MUST FOLLOW):

### Rule 1: Indicator Names
Use the CORRECT Backtrader indicator names:
- [OK] CORRECT: `bt.indicators.OBV(self.data)` 
- [FAIL] WRONG: `bt.indicators.OnBalanceVolume()` (does not exist!)
- [OK] CORRECT: `bt.indicators.RSI(self.data.close, period=14)`
- [OK] CORRECT: `bt.indicators.ATR(self.data, period=14)`
- [OK] CORRECT: `bt.indicators.ADX(self.data, period=14)`
- [OK] CORRECT: `bt.indicators.BollingerBands(self.data.close)`
- [OK] CORRECT: `bt.indicators.MACD(self.data.close)`
- [OK] CORRECT: `bt.indicators.Stochastic(self.data)`

### Rule 2: Position Price Access (CRITICAL - Prevents NoneType errors)
ALWAYS check position exists before accessing position.price:
```python
# [OK] CORRECT - Safe position price access
if self.position and self.position.size != 0:
    entry_price = self.position.price
    stop_price = entry_price - (2 * self.atr[0])

# [FAIL] WRONG - Will cause NoneType error when no position
stop_price = self.position.price - (2 * self.atr[0])
```

For partial exits, track entry price yourself:
```python
def __init__(self):
    self.entry_price = None  # Track entry price manually
    
def next(self):
    if not self.position:
        if self.crossover > 0:
            self.buy()
            self.entry_price = self.data.close[0]  # Store entry price
    else:
        if self.entry_price is not None:  # Safe to use
            stop_price = self.entry_price - (2 * self.atr[0])
```

### Rule 3: Minimum Bar Checks (CRITICAL - Prevents array index errors)
ALWAYS add minimum bar check at start of next():
```python
def next(self):
    # Wait for indicators to have enough data
    if len(self) < self.params.slow_period:
        return
    
    # Also check if indicators have valid values
    if self.atr[0] <= 0:
        return
        
    # Now safe to proceed with trading logic
```

### Rule 4: Multi-Timeframe Safety
When using multiple data feeds, check data availability:
```python
def next(self):
    # Check if higher timeframe has enough data
    if len(self.data1) < self.params.htf_period:
        return
    
    # Use try/except for safety
    try:
        htf_trend = self.data1.close[0] > self.htf_ma[0]
    except (IndexError, KeyError):
        return
```

### Rule 5: Partial Exits
For partial position exits, use explicit size:
```python
# [OK] CORRECT - Exit half position
if self.position.size > 0:
    half_size = self.position.size // 2
    if half_size > 0:
        self.sell(size=half_size)

# [FAIL] WRONG - Closes entire position
self.close()  # This closes EVERYTHING
```

## OUTPUT FORMAT:

For each variant, output exactly this format:

### VARIANT_01 ###
```python
# Full strategy code here
import backtrader as bt

class Variant01_RSI_Filter(bt.Strategy):
    '''
    Modification: Added RSI filter
    Reasoning: Avoid entries when market is overbought/oversold
    '''
    # ... rest of code
```

### VARIANT_02 ###
```python
# Full strategy code here
```

... and so on for all {num_variants} variants.

## GUIDELINES:

- Make each variant meaningfully different
- Mix different types of modifications (indicators, stops, filters, etc.)
- Include some conservative changes and some aggressive ones
- Try combinations of ideas, not just single changes
- Use realistic parameter values
- Ensure all indicator periods are reasonable (not too short, not too long)

Generate exactly {num_variants} complete strategy variants now:
"""


# ==============================================================================
# MAIN FUNCTIONS
# ==============================================================================

def load_base_strategy(strategy_path=None):
    """Load the base strategy code from file"""
    
    if strategy_path is None:
        strategy_path = Path(__file__).parent / 'strategies' / 'simple_strategy.py'
    
    if not strategy_path.exists():
        print(f"[FAIL] Base strategy not found: {strategy_path}")
        return None
    
    with open(strategy_path, 'r') as f:
        code = f.read()
    
    print(f"[OK] Loaded base strategy: {strategy_path.name}")
    return code


def get_performance_summary():
    """Get recent backtest results for the base strategy"""
    
    db = ResultsDatabase()
    
    try:
        # Query recent results for the base strategy
        import sqlite3
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(total_return_pct) as avg_return,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(win_rate) as avg_win_rate,
                AVG(max_drawdown_pct) as avg_drawdown,
                AVG(total_trades) as avg_trades,
                COUNT(*) as total_tests,
                MAX(total_return_pct) as best_return,
                MIN(total_return_pct) as worst_return
            FROM backtest_results
            WHERE strategy_name = 'SimpleMovingAverageCrossover'
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[5] > 0:  # total_tests > 0
            summary = f"""
- Tests Run: {row[5]}
- Average Return: {row[0]:.2f}%
- Average Sharpe Ratio: {row[1]:.2f if row[1] else 'N/A'}
- Average Win Rate: {row[2]:.1f}% if row[2] else 'N/A'
- Average Max Drawdown: {row[3]:.2f}%
- Average Trades per Test: {row[4]:.0f}
- Best Single Test Return: {row[6]:.2f}%
- Worst Single Test Return: {row[7]:.2f}%

The base strategy is currently unprofitable on average. Variants should aim to:
1. Improve win rate through better entry filters
2. Reduce drawdown through better stop losses
3. Increase profit factor through better exits
"""
            print(f"[OK] Loaded performance data from {row[5]} backtest results")
            return summary
        else:
            return "No backtest results available yet. Generate variants based on general best practices."
    
    except Exception as e:
        print(f"[WARN]  Could not load performance data: {e}")
        return "No backtest results available yet. Generate variants based on general best practices."


def call_mutation_agent(base_code, performance, ideas):
    """Call Claude to generate variants"""
    
    print(f"\n[AI] Calling Claude to generate {NUM_VARIANTS} variants...")
    print(f"   This may take 30-60 seconds...\n")
    
    client = Anthropic(api_key=config.CLAUDE_API_KEY)
    
    prompt = MUTATION_PROMPT.format(
        num_variants=NUM_VARIANTS,
        base_strategy_code=base_code,
        performance_summary=performance,
        mutation_ideas=ideas
    )
    
    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,  # Need lots of tokens for 15 strategies
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Calculate approximate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        # Claude Sonnet pricing (approximate)
        input_cost = (input_tokens / 1_000_000) * 3.00
        output_cost = (output_tokens / 1_000_000) * 15.00
        total_cost = input_cost + output_cost
        
        print(f"[OK] Response received!")
        print(f"   Input tokens:  {input_tokens:,}")
        print(f"   Output tokens: {output_tokens:,}")
        print(f"   Estimated cost: ${total_cost:.4f}")
        
        return response.content[0].text
    
    except Exception as e:
        print(f"[FAIL] API call failed: {e}")
        return None


def parse_variants(response_text):
    """Parse Claude's response to extract individual variant code blocks"""
    
    variants = []
    
    # Pattern to match each variant section
    # Looks for ### VARIANT_XX ### followed by ```python ... ```
    pattern = r'### VARIANT_(\d+) ###\s*```python\s*(.*?)```'
    
    matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for variant_num, code in matches:
        code = code.strip()
        
        # Basic validation - check if it has a class definition
        if 'class Variant' in code or 'class variant' in code:
            variants.append({
                'number': int(variant_num),
                'code': code
            })
        elif 'bt.Strategy' in code:
            # Try to fix if class name is different
            variants.append({
                'number': int(variant_num),
                'code': code
            })
    
    print(f"\n[SEARCH] Parsed {len(variants)} variants from response")
    
    return variants


def validate_variant(code):
    """Basic syntax validation for a variant"""
    
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_common_bugs(code):
    """
    Check for common Backtrader bugs that the rules should prevent.
    Returns list of warnings.
    """
    warnings = []
    
    # Check for wrong OBV indicator name
    if 'OnBalanceVolume' in code:
        warnings.append("[WARN]  Uses 'OnBalanceVolume' - should be 'OBV'")
    
    # Check for unsafe position.price access
    if 'self.position.price' in code:
        # Check if there's a guard
        if 'if self.position' not in code and 'if not self.position' not in code:
            warnings.append("[WARN]  Accesses position.price without checking if position exists")
    
    # Check for missing minimum bar check
    if 'def next(self):' in code:
        next_start = code.find('def next(self):')
        next_body = code[next_start:next_start+500]  # Check first 500 chars of next()
        if 'if len(self)' not in next_body and 'if len(self.data)' not in next_body:
            warnings.append("[WARN]  next() may be missing minimum bar check")
    
    return warnings


def save_variants(variants):
    """Save each variant to a separate file"""
    
    # Clear old variants
    for old_file in VARIANTS_DIR.glob('variant_*.py'):
        old_file.unlink()
    
    saved = []
    failed = []
    
    for variant in variants:
        num = variant['number']
        code = variant['code']
        
        # Validate syntax
        is_valid, error = validate_variant(code)
        
        if is_valid:
            # Check for common bugs (warning only, still save)
            bug_warnings = check_common_bugs(code)
            
            filename = VARIANTS_DIR / f"variant_{num:02d}.py"
            
            # Add header comment
            header = f"""# ==============================================================================
# variant_{num:02d}.py
# ==============================================================================
# Auto-generated by Mutation Agent
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# ==============================================================================

"""
            with open(filename, 'w') as f:
                f.write(header + code)
            
            saved.append(num)
            
            if bug_warnings:
                print(f"   [WARN]  Saved variant_{num:02d}.py (with warnings)")
                for warn in bug_warnings:
                    print(f"      {warn}")
            else:
                print(f"   [OK] Saved variant_{num:02d}.py")
        else:
            failed.append({'number': num, 'error': error})
            print(f"   [FAIL] variant_{num:02d} failed validation: {error[:50]}...")
    
    return saved, failed


def extract_variant_info(code):
    """Extract the class name and docstring from variant code"""
    
    # Find class name
    class_match = re.search(r'class\s+(\w+)', code)
    class_name = class_match.group(1) if class_match else "Unknown"
    
    # Find docstring
    doc_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if not doc_match:
        doc_match = re.search(r"'''(.*?)'''", code, re.DOTALL)
    
    docstring = doc_match.group(1).strip() if doc_match else "No description"
    
    return class_name, docstring


def generate_variants_summary(variants, saved, failed):
    """Generate a summary of what was created"""
    
    print(f"\n{'='*70}")
    print("MUTATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Variants generated: {len(variants)}")
    print(f"  Successfully saved: {len(saved)}")
    print(f"  Failed validation:  {len(failed)}")
    print(f"\n  Saved to: {VARIANTS_DIR}")
    
    if saved:
        print(f"\n  Variants ready for backtesting:")
        for num in saved:
            # Load the code to get class name
            filepath = VARIANTS_DIR / f"variant_{num:02d}.py"
            with open(filepath, 'r') as f:
                code = f.read()
            class_name, docstring = extract_variant_info(code)
            
            # Truncate docstring for display
            short_doc = docstring.split('\n')[0][:50]
            print(f"    - variant_{num:02d}.py: {class_name}")
            print(f"      {short_doc}...")
    
    if failed:
        print(f"\n  [WARN]  Failed variants (syntax errors):")
        for f in failed:
            print(f"    - variant_{f['number']:02d}: {f['error'][:40]}...")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("  1. Review variants in strategies/variants/")
    print("  2. Run: python run_variant_backtests.py")
    print("  3. Run: python compare_variants.py")
    print("  4. Run: streamlit run dashboard.py")
    print(f"{'='*70}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main mutation agent workflow"""
    
    print("\n" + "="*70)
    print("[DNA] MUTATION AGENT")
    print("="*70)
    print(f"  Variants to generate: {NUM_VARIANTS}")
    print(f"  Output directory: {VARIANTS_DIR}")
    print("="*70 + "\n")
    
    # Step 1: Load base strategy
    print("[FOLDER] Step 1: Loading base strategy...")
    base_code = load_base_strategy()
    if not base_code:
        return
    
    # Step 2: Get performance data
    print("\n[STATS] Step 2: Loading performance data...")
    performance = get_performance_summary()
    
    # Step 3: Get mutation ideas
    print("\n[TIP] Step 3: Loading mutation ideas...")
    ideas = get_all_ideas()
    idea_count = len([l for l in ideas.split('\n') if l.strip().startswith('-')])
    print(f"[OK] Loaded {idea_count} mutation ideas")
    
    # Step 4: Confirm with user
    print("\n" + "-"*70)
    print("Ready to generate variants.")
    print(f"Estimated cost: ~$0.15-0.30")
    print("-"*70)
    
    confirm = input("\nProceed? (Y/N): ").strip().upper()
    if confirm != 'Y':
        print("Cancelled.")
        return
    
    # Step 5: Call Claude
    print("\n" + "-"*70)
    response = call_mutation_agent(base_code, performance, ideas)
    if not response:
        return
    
    # Step 6: Parse variants
    print("\n[SEARCH] Step 4: Parsing variants...")
    variants = parse_variants(response)
    
    if not variants:
        print("[FAIL] No valid variants found in response")
        print("\nRaw response (first 500 chars):")
        print(response[:500])
        return
    
    # Step 7: Save variants
    print("\n[SAVE] Step 5: Saving variants...")
    saved, failed = save_variants(variants)
    
    # Step 8: Summary
    generate_variants_summary(variants, saved, failed)
    
    return saved


if __name__ == "__main__":
    main()