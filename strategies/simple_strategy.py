# simple_strategy.py
# This is a BASE STRATEGY - the AI will generate variants from this

import backtrader as bt

class SimpleMovingAverageCrossover(bt.Strategy):
    """
    BASE STRATEGY: Moving Average Crossover
    
    This is your starting point. The Mutation Agent will:
    - Test different MA periods (5/20, 10/30, 20/50, etc.)
    - Add filters (ADX, volatility, volume)
    - Add stops (ATR, percentage, trailing)
    - Add confirmations (RSI, MACD, etc.)
    
    You write this ONCE, AI generates 100s of variants
    """
    
    params = (
        ('fast_period', 10),   # Fast moving average period
        ('slow_period', 30),   # Slow moving average period
    )
    
    def __init__(self):
        # Calculate the two moving averages
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, 
            period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, 
            period=self.params.slow_period
        )
        
        # Crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        # This runs every day/candle
        
        if not self.position:  # Not in a trade
            if self.crossover > 0:  # Fast MA crossed above slow MA
                self.buy()
                
        else:  # In a trade
            if self.crossover < 0:  # Fast MA crossed below slow MA
                self.sell()