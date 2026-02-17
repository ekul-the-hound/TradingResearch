# Auto-discovered strategy: TestMomentum
# Source: https://example.com
# Source type: github (retail)
# Quality score: 0.800
# Strategy ID: strat_20260217_003535_055576_d34d4c23
# Extracted: 2026-02-17T00:35:35.072957
#
import backtrader as bt

class TestMomentumCrossover(bt.Strategy):
    """Test momentum crossover strategy."""
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if len(self) < self.params.slow_period:
            return
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.sell()
