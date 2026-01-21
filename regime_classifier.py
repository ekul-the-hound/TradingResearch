# ==============================================================================
# regime_classifier.py
# ==============================================================================
# Market Regime Classifier
#
# Classifies each bar into one of 6 market regimes:
# - BULL: Strong uptrend (price above MAs, positive momentum)
# - BEAR: Strong downtrend (price below MAs, negative momentum)
# - RANGING: Sideways/consolidation (low ADX, tight Bollinger Bands)
# - HIGH_VOL: High volatility period (ATR spike, wide BBands)
# - CRASH: Rapid decline (large negative returns, VIX spike proxy)
# - RECOVERY: Bounce from lows (oversold reversal)
#
# Usage:
#     from regime_classifier import RegimeClassifier
#     classifier = RegimeClassifier()
#     df = classifier.classify(df)  # Adds 'regime' column
#
# ==============================================================================

import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional, Dict, List, Tuple


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "BULL"
    BEAR = "BEAR"
    RANGING = "RANGING"
    HIGH_VOL = "HIGH_VOL"
    CRASH = "CRASH"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"


class RegimeClassifier:
    """
    Classifies market data into distinct regimes for segmented backtesting.
    
    This allows you to see how your strategy performs in different market
    conditions, revealing if it only works in bull markets, struggles in
    high volatility, etc.
    """
    
    def __init__(
        self,
        # Trend detection
        sma_fast: int = 20,
        sma_slow: int = 50,
        trend_threshold: float = 0.02,  # 2% above/below MA for trend
        
        # Volatility detection
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        vol_spike_threshold: float = 1.5,  # ATR 1.5x above average = high vol
        
        # Trend strength (ADX)
        adx_period: int = 14,
        adx_trend_threshold: float = 25,  # ADX > 25 = trending
        adx_range_threshold: float = 20,  # ADX < 20 = ranging
        
        # Crash/Recovery detection
        crash_threshold: float = -0.03,  # -3% single bar = crash signal
        crash_lookback: int = 5,  # Look for crashes in last N bars
        recovery_rsi_threshold: float = 30,  # RSI < 30 then reversal
        rsi_period: int = 14,
        
        # Smoothing
        regime_smoothing: int = 3,  # Bars to confirm regime change
    ):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.trend_threshold = trend_threshold
        
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.vol_spike_threshold = vol_spike_threshold
        
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        
        self.crash_threshold = crash_threshold
        self.crash_lookback = crash_lookback
        self.recovery_rsi_threshold = recovery_rsi_threshold
        self.rsi_period = rsi_period
        
        self.regime_smoothing = regime_smoothing
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for regime classification"""
        
        df = df.copy()
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # =====================================================================
        # TREND INDICATORS
        # =====================================================================
        
        # Simple Moving Averages
        df['sma_fast'] = df['close'].rolling(window=self.sma_fast).mean()
        df['sma_slow'] = df['close'].rolling(window=self.sma_slow).mean()
        
        # Price position relative to MAs
        df['above_fast_ma'] = df['close'] > df['sma_fast']
        df['above_slow_ma'] = df['close'] > df['sma_slow']
        
        # Trend strength (distance from slow MA)
        df['ma_distance'] = (df['close'] - df['sma_slow']) / df['sma_slow']
        
        # =====================================================================
        # VOLATILITY INDICATORS
        # =====================================================================
        
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # ATR
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        # Average ATR for comparison (longer lookback)
        df['atr_avg'] = df['atr'].rolling(window=self.atr_period * 3).mean()
        
        # ATR ratio (current vs average)
        df['atr_ratio'] = df['atr'] / df['atr_avg']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (self.bb_std * df['bb_std'])
        
        # Bollinger Band Width (volatility proxy)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_width_avg'] = df['bb_width'].rolling(window=self.bb_period * 2).mean()
        df['bb_width_ratio'] = df['bb_width'] / df['bb_width_avg']
        
        # =====================================================================
        # ADX (Trend Strength)
        # =====================================================================
        
        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Smoothed DM and TR
        df['plus_dm_smooth'] = df['plus_dm'].rolling(window=self.adx_period).mean()
        df['minus_dm_smooth'] = df['minus_dm'].rolling(window=self.adx_period).mean()
        df['tr_smooth'] = df['tr'].rolling(window=self.adx_period).mean()
        
        # DI+ and DI-
        df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
        df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])
        
        # DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=self.adx_period).mean()
        
        # =====================================================================
        # MOMENTUM / RSI
        # =====================================================================
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_5bar'] = df['close'].pct_change(5)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # =====================================================================
        # CRASH/RECOVERY DETECTION
        # =====================================================================
        
        # Rolling min return (detect recent crash)
        df['min_return_lookback'] = df['returns'].rolling(window=self.crash_lookback).min()
        
        # Was recently oversold
        df['was_oversold'] = df['rsi'].rolling(window=self.crash_lookback).min() < self.recovery_rsi_threshold
        
        # RSI rising from oversold
        df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
        
        return df
    
    def _classify_bar(self, row: pd.Series) -> str:
        """Classify a single bar into a regime"""
        
        # Handle NaN values (early bars without enough data)
        if pd.isna(row.get('adx')) or pd.isna(row.get('atr_ratio')):
            return MarketRegime.UNKNOWN.value
        
        # =====================================================================
        # PRIORITY 1: CRASH (overrides everything)
        # =====================================================================
        if row['returns'] < self.crash_threshold:
            return MarketRegime.CRASH.value
        
        if row['min_return_lookback'] < self.crash_threshold * 1.5:
            # Recent severe crash still in effect
            if row['returns'] < 0:
                return MarketRegime.CRASH.value
        
        # =====================================================================
        # PRIORITY 2: RECOVERY (after crash/oversold)
        # =====================================================================
        if row['was_oversold'] and row['rsi_rising'] and row['returns'] > 0:
            if row['rsi'] < 50:  # Still in recovery zone
                return MarketRegime.RECOVERY.value
        
        # =====================================================================
        # PRIORITY 3: HIGH VOLATILITY
        # =====================================================================
        if row['atr_ratio'] > self.vol_spike_threshold:
            return MarketRegime.HIGH_VOL.value
        
        if row['bb_width_ratio'] > self.vol_spike_threshold:
            return MarketRegime.HIGH_VOL.value
        
        # =====================================================================
        # PRIORITY 4: RANGING (low ADX)
        # =====================================================================
        if row['adx'] < self.adx_range_threshold:
            return MarketRegime.RANGING.value
        
        # =====================================================================
        # PRIORITY 5: TRENDING (BULL or BEAR)
        # =====================================================================
        if row['adx'] > self.adx_trend_threshold:
            if row['above_slow_ma'] and row['ma_distance'] > self.trend_threshold:
                return MarketRegime.BULL.value
            elif not row['above_slow_ma'] and row['ma_distance'] < -self.trend_threshold:
                return MarketRegime.BEAR.value
        
        # =====================================================================
        # DEFAULT: RANGING
        # =====================================================================
        return MarketRegime.RANGING.value
    
    def _smooth_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth regime changes to avoid whipsaws"""
        
        df = df.copy()
        
        if self.regime_smoothing <= 1:
            return df
        
        # Count consecutive bars of same regime
        df['regime_count'] = 1
        
        for i in range(1, len(df)):
            if df.iloc[i]['regime_raw'] == df.iloc[i-1]['regime_raw']:
                df.iloc[i, df.columns.get_loc('regime_count')] = df.iloc[i-1]['regime_count'] + 1
        
        # Only confirm regime change after N consecutive bars
        df['regime'] = df['regime_raw']
        
        confirmed_regime = MarketRegime.UNKNOWN.value
        
        for i in range(len(df)):
            if df.iloc[i]['regime_count'] >= self.regime_smoothing:
                confirmed_regime = df.iloc[i]['regime_raw']
            df.iloc[i, df.columns.get_loc('regime')] = confirmed_regime
        
        return df
    
    def classify(self, df: pd.DataFrame, smooth: bool = True) -> pd.DataFrame:
        """
        Classify all bars in a DataFrame into market regimes.
        
        Args:
            df: DataFrame with OHLCV data
            smooth: Whether to apply regime smoothing
        
        Returns:
            DataFrame with 'regime' column added
        """
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Classify each bar
        df['regime_raw'] = df.apply(self._classify_bar, axis=1)
        
        # Smooth regimes
        if smooth:
            df = self._smooth_regimes(df)
        else:
            df['regime'] = df['regime_raw']
        
        # Clean up intermediate columns (optional - keep for debugging)
        # cleanup_cols = ['regime_raw', 'regime_count', ...]
        # df = df.drop(columns=cleanup_cols, errors='ignore')
        
        return df
    
    def get_regime_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for each regime in the data.
        
        Args:
            df: DataFrame with 'regime' column
        
        Returns:
            Dictionary with regime statistics
        """
        
        if 'regime' not in df.columns:
            raise ValueError("DataFrame must have 'regime' column. Run classify() first.")
        
        summary = {}
        
        for regime in MarketRegime:
            regime_df = df[df['regime'] == regime.value]
            
            if len(regime_df) == 0:
                continue
            
            summary[regime.value] = {
                'bar_count': len(regime_df),
                'pct_of_total': len(regime_df) / len(df) * 100,
                'avg_return': regime_df['returns'].mean() * 100 if 'returns' in regime_df.columns else None,
                'volatility': regime_df['returns'].std() * 100 if 'returns' in regime_df.columns else None,
                'avg_atr_pct': regime_df['atr_pct'].mean() * 100 if 'atr_pct' in regime_df.columns else None,
            }
        
        return summary
    
    def print_regime_summary(self, df: pd.DataFrame):
        """Print a formatted regime summary"""
        
        summary = self.get_regime_summary(df)
        
        print("\n" + "="*70)
        print("MARKET REGIME SUMMARY")
        print("="*70)
        print(f"{'Regime':<12} {'Bars':>8} {'% Total':>10} {'Avg Ret':>10} {'Volatility':>12}")
        print("-"*70)
        
        for regime, stats in summary.items():
            avg_ret = f"{stats['avg_return']:.3f}%" if stats['avg_return'] else "N/A"
            vol = f"{stats['volatility']:.3f}%" if stats['volatility'] else "N/A"
            
            print(f"{regime:<12} {stats['bar_count']:>8} {stats['pct_of_total']:>9.1f}% {avg_ret:>10} {vol:>12}")
        
        print("="*70 + "\n")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def classify_regimes(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Quick function to classify regimes.
    
    Usage:
        from regime_classifier import classify_regimes
        df = classify_regimes(df)
    """
    classifier = RegimeClassifier(**kwargs)
    return classifier.classify(df)


def get_regime_stats(df: pd.DataFrame, **kwargs) -> Dict:
    """
    Quick function to get regime statistics.
    
    Usage:
        from regime_classifier import get_regime_stats
        stats = get_regime_stats(df)
    """
    classifier = RegimeClassifier(**kwargs)
    df = classifier.classify(df)
    return classifier.get_regime_summary(df)


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("REGIME CLASSIFIER TEST")
    print("="*70)
    
    # Create sample data for testing
    np.random.seed(42)
    n_bars = 1000
    
    # Simulate price data with different regimes
    dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='1H')
    
    # Create trending and ranging periods
    returns = np.random.normal(0.0001, 0.01, n_bars)
    
    # Add trend periods
    returns[100:200] += 0.002  # Bull period
    returns[300:400] -= 0.002  # Bear period
    returns[500:520] -= 0.02   # Crash
    returns[520:550] += 0.01   # Recovery
    returns[700:750] *= 2      # High volatility
    
    prices = 100 * np.cumprod(1 + returns)
    
    # Create OHLC from close
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_bars)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n_bars)),
        'low': prices * (1 - np.random.uniform(0, 0.01, n_bars)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)
    
    # Classify
    classifier = RegimeClassifier()
    df = classifier.classify(df)
    
    # Print summary
    classifier.print_regime_summary(df)
    
    # Show sample of each regime
    print("Sample bars from each regime:")
    print("-"*70)
    
    for regime in MarketRegime:
        regime_df = df[df['regime'] == regime.value]
        if len(regime_df) > 0:
            print(f"\n{regime.value}: {len(regime_df)} bars")
            print(regime_df[['close', 'returns', 'adx', 'atr_ratio', 'regime']].head(3))
    
    print("\n" + "="*70)
    print("✅ Regime classifier working!")
    print("="*70)