# ==============================================================================
# mutation_config.py
# ==============================================================================
# 
# HOW TO USE THIS FILE:
# Just type your ideas below each section. One idea per line.
# The AI will read these and use them to create strategy variants.
# 
# You don't need to write any code - just jot down ideas like notes.
# 
# ==============================================================================


# ------------------------------------------------------------------------------
# INDICATORS
# ------------------------------------------------------------------------------
# Add any indicators you want the AI to try adding to your strategy.
# Examples: RSI, MACD, ADX, Bollinger Bands, ATR, VWAP, Stochastic, etc.
# ------------------------------------------------------------------------------

INDICATORS = """
RSI
MACD
ADX
Bollinger Bands
ATR
VWAP
Stochastic
EMA
Volume Profile
OBV
CCI
Williams %R
Ichimoku Cloud
Keltner Channels
Donchian Channels
"""


# ------------------------------------------------------------------------------
# STOP LOSS IDEAS
# ------------------------------------------------------------------------------
# How should the strategy exit losing trades?
# Examples: ATR stop, percentage stop, trailing stop, time stop, etc.
# ------------------------------------------------------------------------------

STOP_LOSSES = """
ATR-based stop loss
Fixed percentage stop (2%, 3%, 5%)
Trailing stop
Break-even stop after X profit
Time-based stop (exit after N bars)
Support/resistance stop
Volatility-adjusted stop
"""


# ------------------------------------------------------------------------------
# TAKE PROFIT IDEAS
# ------------------------------------------------------------------------------
# How should the strategy exit winning trades?
# Examples: Fixed target, trailing, partial profits, etc.
# ------------------------------------------------------------------------------

TAKE_PROFITS = """
Fixed risk/reward ratio (1:2, 1:3)
Trailing take profit
Partial profits at levels (50% at 1R, 50% at 2R)
ATR-based profit target
Resistance-based exit
Time-based exit
"""


# ------------------------------------------------------------------------------
# ENTRY FILTERS
# ------------------------------------------------------------------------------
# What conditions should be true before entering a trade?
# Examples: Trend filters, volatility filters, time filters, etc.
# ------------------------------------------------------------------------------

ENTRY_FILTERS = """
ADX trend strength filter (only trade when ADX > 25)
Higher timeframe trend alignment
Volatility filter (skip low volatility)
Volume confirmation
RSI not overbought/oversold
Time of day filter (avoid Asian session, etc.)
Avoid news events
Only trade with 200 EMA trend
"""


# ------------------------------------------------------------------------------
# POSITION SIZING / MONEY MANAGEMENT
# ------------------------------------------------------------------------------
# How should the strategy size positions?
# Examples: Fixed size, percentage risk, Kelly criterion, etc.
# ------------------------------------------------------------------------------

POSITION_SIZING = """
Fixed percentage risk per trade (1%, 2%)
Volatility-adjusted position size
Martingale (increase after loss)
Anti-martingale (increase after win)
Kelly criterion
Scale in (multiple entries)
DCA (dollar cost averaging)
Pyramid (add to winners)
"""


# ------------------------------------------------------------------------------
# EXIT MODIFICATIONS
# ------------------------------------------------------------------------------
# Other ways to modify how/when to exit trades
# Examples: Trailing, partial exits, re-entry, etc.
# ------------------------------------------------------------------------------

EXIT_MODS = """
Trail stop after reaching 1R profit
Scale out (partial exits)
Re-entry after pullback
Exit on opposite signal
Exit on momentum divergence
Exit before weekend/holidays
"""


# ------------------------------------------------------------------------------
# STRATEGY VARIATIONS
# ------------------------------------------------------------------------------
# Different overall approaches to try
# Examples: Mean reversion, breakout, trend following, etc.
# ------------------------------------------------------------------------------

STRATEGY_TYPES = """
Trend following
Mean reversion
Breakout
Pullback entry
Range trading
Momentum
Counter-trend
"""


# ------------------------------------------------------------------------------
# PARAMETER VARIATIONS
# ------------------------------------------------------------------------------
# Specific parameters you want tested
# Examples: Different MA periods, RSI levels, etc.
# ------------------------------------------------------------------------------

PARAMETERS = """
Fast MA: 5, 10, 15, 20
Slow MA: 20, 30, 50, 100, 200
RSI period: 7, 14, 21
RSI overbought: 70, 75, 80
RSI oversold: 20, 25, 30
ATR period: 7, 14, 21
ATR multiplier: 1.5, 2, 2.5, 3
ADX threshold: 20, 25, 30
"""


# ------------------------------------------------------------------------------
# YOUR CUSTOM IDEAS
# ------------------------------------------------------------------------------
# Anything else you want to try! Just write it down.
# The AI will interpret it and try to implement it.
# ------------------------------------------------------------------------------

CUSTOM_IDEAS = """
Add your own ideas here
Combine RSI oversold with MACD crossover
Only enter on pullback to 20 EMA
Wait for confirmation candle before entry
Use higher timeframe for trend, lower for entry

# Advanced Entry Ideas
Enter only after price rejects from support/resistance
Enter on bullish/bearish engulfing candles
Enter on pin bar / hammer / shooting star patterns
Enter after consolidation breakout (low ADX then high ADX)
Enter on volume spike with trend
Use limit orders at EMA for better entries

# Advanced Exit Ideas
Exit on RSI divergence
Exit on volume climax (spike against position)
Exit when ADX drops below 20 (trend weakening)
Exit at Fibonacci extension levels (1.618, 2.618)
Move stop to break-even after 1 ATR profit
Use chandelier exit (highest high - ATR multiple)

# Risk Management
Maximum 3 trades per day
No trading on Fridays (avoid weekend risk)
Reduce size during high volatility (VIX proxy)
Correlation filter (don't trade correlated pairs same time)
Maximum drawdown circuit breaker

# Regime Adaptations
Different parameters for trending vs ranging markets
Increase trade frequency in high ADX environments
Use mean reversion in low ADX, trend following in high ADX
Tighter stops in high volatility
Wider stops and targets in low volatility

# Multi-Timeframe
4H trend + 1H entry signals
Daily bias + 4H execution
Weekly support/resistance + daily entries
Confirm signal on 2 timeframes before entry

# Pattern-Based
Trade only at round numbers (psychological levels)
Trade only at previous day high/low
Trade only after false breakout (failure test)
Enter after 3 consecutive same-color candles
"""


# ==============================================================================
# CODE BELOW - DO NOT EDIT
# ==============================================================================
# This reads your notes above and makes them available to the mutation agent.
# ==============================================================================

def get_all_ideas():
    """Returns all mutation ideas as a formatted string for the AI"""
    
    sections = {
        'Indicators': INDICATORS,
        'Stop Losses': STOP_LOSSES,
        'Take Profits': TAKE_PROFITS,
        'Entry Filters': ENTRY_FILTERS,
        'Position Sizing': POSITION_SIZING,
        'Exit Modifications': EXIT_MODS,
        'Strategy Types': STRATEGY_TYPES,
        'Parameter Variations': PARAMETERS,
        'Custom Ideas': CUSTOM_IDEAS,
    }
    
    output = []
    for section_name, content in sections.items():
        # Clean up the content - remove empty lines, strip whitespace
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        if lines:
            output.append(f"\n## {section_name}:")
            for line in lines:
                output.append(f"  - {line}")
    
    return '\n'.join(output)


def get_ideas_list():
    """Returns all ideas as a flat list"""
    
    all_content = [
        INDICATORS, STOP_LOSSES, TAKE_PROFITS, ENTRY_FILTERS,
        POSITION_SIZING, EXIT_MODS, STRATEGY_TYPES, PARAMETERS, CUSTOM_IDEAS
    ]
    
    ideas = []
    for content in all_content:
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        ideas.extend(lines)
    
    return ideas


# Quick test - run this file directly to see your ideas
if __name__ == "__main__":
    print("=" * 70)
    print("YOUR MUTATION IDEAS")
    print("=" * 70)
    print(get_all_ideas())
    print("\n" + "=" * 70)
    print(f"Total ideas: {len(get_ideas_list())}")
    print("=" * 70)