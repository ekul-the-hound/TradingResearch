#!/usr/bin/env python3
"""
ULTIMATE TEMA/LSMA + IDEAL + REVERSAL + INSTITUTIONAL + DAVIN MA [DYNAMIC]
==========================================================================
Python conversion of Ahtisham Qureshi's TradingView Pine Script strategy.

Combines 5 strategy layers with priority hierarchy:
  1. INSTITUTIONAL (Dynamic zone accumulation/distribution)
  2. IDEAL CONDITIONS (Liquidity sweeps, BoS retests, oversold confluence)
  3. REVERSAL (KAMA + multi-indicator consensus)
  4. DAVIN MA (200/10 SMA crossover + dip buying)
  5. ORIGINAL (TEMA/LSMA crossover with trend filters)

Requirements:
    pip install pandas numpy yfinance backtrader matplotlib tabulate

Usage:
    python ultimate_tema_lsma_strategy.py                          # default BTC-USD
    python ultimate_tema_lsma_strategy.py --symbol XAUUSD=X        # gold
    python ultimate_tema_lsma_strategy.py --symbol ETH-USD --days 365
    python ultimate_tema_lsma_strategy.py --no-plot                # skip chart
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Install yfinance: pip install yfinance")

try:
    import backtrader as bt
except ImportError:
    sys.exit("Install backtrader: pip install backtrader")


# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION (mirrors Pine Script inputs)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class StrategyConfig:
    # TEMA / LSMA core
    tema_length: int = 9
    lsma_length: int = 25
    ema100_length: int = 100
    tp_atr_mult: float = 3.0
    sl_atr_mult: float = 1.5
    trail_atr_mult: float = 1.0
    max_bars_in_trade: int = 25
    vol_length: int = 20
    min_vol_multiplier: float = 1.2

    # Trend filters
    use_trend_exhaustion: bool = True
    min_trend_bars: int = 10
    use_momentum_divergence: bool = True
    require_trend_alignment: bool = True
    use_trend_strength: bool = True
    trend_strength_threshold: float = 0.5

    # IDEAL conditions
    use_ideal: bool = True
    pivot_lookback: int = 5
    min_sweep_atr: float = 0.3
    retest_tolerance: float = 0.001
    max_test_count: int = 3
    atr_test_pct: float = 0.2

    # Reversal
    enable_reversals: bool = True
    kama_length: int = 10
    adx_length: int = 14
    adx_strong: float = 20.0
    rsi_length: int = 14
    rsi_ob: float = 70.0
    rsi_os: float = 30.0
    williams_length: int = 14
    williams_ob: float = -20.0
    williams_os: float = -80.0
    liquidity_zone_bars: int = 8
    min_vol_mult_rev: float = 1.0

    # Institutional
    use_institutional: bool = True
    inst_swing_lookback: int = 20
    inst_zone_atr_width: float = 0.5
    inst_mid_atr_buffer: float = 0.25
    inst_base_size: float = 10.0
    inst_size_scaling: bool = True
    use_vwap_mid: bool = True

    # Davin MA
    use_davin: bool = True
    davin_ma_long: int = 200
    davin_ma_short: int = 10
    davin_buy_dip: bool = True
    davin_dip_trigger: int = 14
    davin_lower_close: bool = True


# ──────────────────────────────────────────────────────────────────────
# INDICATOR FUNCTIONS (pure numpy/pandas — no Pine built-ins)
# ──────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple Exponential Moving Average."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def lsma(series: pd.Series, period: int) -> pd.Series:
    """Least Squares Moving Average (linear regression endpoint)."""
    out = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(period - 1, len(vals)):
        y = vals[i - period + 1: i + 1]
        x = np.arange(period, dtype=float)
        if np.any(np.isnan(y)):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        out.iloc[i] = intercept + slope * (period - 1)
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (RMA)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA / SMMA)."""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def stochastic(close: pd.Series, high: pd.Series, low: pd.Series, k_period: int = 9) -> pd.Series:
    """Raw stochastic %K."""
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    denom = highest - lowest
    return ((close - lowest) / denom.replace(0, np.nan)) * 100


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal, histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    denom = hh - ll
    return -100 * (hh - close) / denom.replace(0, np.nan)


def cci(close: pd.Series, period: int = 14) -> pd.Series:
    """Commodity Channel Index."""
    tp = close  # using close only (original uses close for CCI calc)
    ma = sma(tp, period)
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def vwap_cumulative(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP (simplified — resets are not modeled)."""
    tp = (high + low + close) / 3
    cum_tpv = (tp * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """Detect pivot highs. Returns the pivot value at pivot bar (offset by `right`)."""
    out = pd.Series(np.nan, index=high.index)
    vals = high.values
    for i in range(left + right, len(vals)):
        pivot_idx = i - right
        window = vals[pivot_idx - left: pivot_idx + right + 1]
        if vals[pivot_idx] == np.nanmax(window):
            out.iloc[i] = vals[pivot_idx]
    return out


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    """Detect pivot lows. Returns the pivot value at pivot bar (offset by `right`)."""
    out = pd.Series(np.nan, index=low.index)
    vals = low.values
    for i in range(left + right, len(vals)):
        pivot_idx = i - right
        window = vals[pivot_idx - left: pivot_idx + right + 1]
        if vals[pivot_idx] == np.nanmin(window):
            out.iloc[i] = vals[pivot_idx]
    return out


# ──────────────────────────────────────────────────────────────────────
# INSTRUMENT DETECTION
# ──────────────────────────────────────────────────────────────────────
def detect_instrument(symbol: str) -> str:
    s = symbol.upper()
    if any(k in s for k in ("XAU", "GOLD", "GC=F", "MGC")):
        return "GOLD"
    if any(k in s for k in ("BTC", "BITCOIN")):
        return "BITCOIN"
    if any(k in s for k in ("ETH", "ETHEREUM")):
        return "ETHEREUM"
    return "OTHER"


def adaptive_params(instrument: str, cfg: StrategyConfig):
    """Return instrument-adapted parameters."""
    mapping = {
        "GOLD":     (5,  0.3, 0.001, 0.2),
        "BITCOIN":  (10, 0.5, 0.005, 0.5),
        "ETHEREUM": (8,  0.4, 0.006, 0.6),
    }
    if instrument in mapping:
        return mapping[instrument]
    return (cfg.pivot_lookback, cfg.min_sweep_atr, cfg.retest_tolerance, cfg.atr_test_pct)


def zone_thresholds(instrument: str):
    mapping = {
        "GOLD":     (2.0, 1.0),
        "BITCOIN":  (5.0, 2.5),
        "ETHEREUM": (6.0, 3.0),
    }
    return mapping.get(instrument, (2.0, 1.0))


# ──────────────────────────────────────────────────────────────────────
# MAIN SIGNAL ENGINE — computes all signals on a DataFrame
# ──────────────────────────────────────────────────────────────────────
def compute_signals(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Takes OHLCV DataFrame and returns it enriched with all indicator
    columns plus final 'signal' and 'signal_source' columns.
    """
    instrument = detect_instrument(symbol)
    a_pivot, a_sweep, a_retest, a_atr_test = adaptive_params(instrument, cfg)
    zt1, zt2 = zone_thresholds(instrument)

    # ---------- core indicators ----------
    df["tema"]   = tema(df["Close"], cfg.tema_length)
    df["lsma"]   = lsma(df["Close"], cfg.lsma_length)
    df["ema100"] = ema(df["Close"], cfg.ema100_length)
    df["atr"]    = atr(df["High"], df["Low"], df["Close"], 14)
    df["rsi"]    = rsi(df["Close"], 14)
    df["vol_ma"] = sma(df["Volume"], cfg.vol_length)
    df["vwap"]   = vwap_cumulative(df["High"], df["Low"], df["Close"], df["Volume"])

    # ---------- Davin MAs ----------
    df["davin_ma_long"]  = sma(df["Close"], cfg.davin_ma_long)
    df["davin_ma_short"] = sma(df["Close"], cfg.davin_ma_short)
    df["davin_highest52"] = df["High"].rolling(52).max()
    df["davin_overall_change"] = ((df["davin_highest52"] - df["Close"]) / df["davin_highest52"]) * 100

    # ---------- reversal indicators ----------
    df["rsi_rev"]   = rsi(df["Close"], cfg.rsi_length)
    raw_k = stochastic(df["Close"], df["High"], df["Low"], 9)
    df["stoch_k"]   = sma(raw_k, 6)
    df["stoch_d"]   = sma(df["stoch_k"], 3)
    rsi_14 = rsi(df["Close"], 14)
    rsi_lo = rsi_14.rolling(14).min()
    rsi_hi = rsi_14.rolling(14).max()
    df["stoch_rsi"] = ((rsi_14 - rsi_lo) / (rsi_hi - rsi_lo).replace(0, np.nan)) * 100
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(df["Close"])
    df["williams"]  = williams_r(df["High"], df["Low"], df["Close"], cfg.williams_length)
    df["cci"]       = cci(df["Close"], 14)

    # DMI / ADX
    delta_high = df["High"].diff()
    delta_low  = -df["Low"].diff()
    plus_dm  = pd.Series(np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0), index=df.index)
    minus_dm = pd.Series(np.where((delta_low > delta_high) & (delta_low > 0), delta_low, 0), index=df.index)
    tr_series = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_adx    = rma(tr_series, cfg.adx_length)
    plus_di    = 100 * rma(plus_dm, cfg.adx_length) / atr_adx.replace(0, np.nan)
    minus_di   = 100 * rma(minus_dm, cfg.adx_length) / atr_adx.replace(0, np.nan)
    dx_series  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"]     = rma(dx_series, cfg.adx_length)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # KAMA (approximated as EMA of kama_length)
    df["kama"] = ema(df["Close"], cfg.kama_length)
    df["kama_rising"]  = df["kama"] > df["kama"].shift(1)
    df["kama_falling"] = df["kama"] < df["kama"].shift(1)

    # ---------- institutional zones ----------
    df["swing_high"] = df["High"].rolling(cfg.inst_swing_lookback).max()
    df["swing_low"]  = df["Low"].rolling(cfg.inst_swing_lookback).min()
    df["buy_zone_lo"]  = df["swing_low"]
    df["buy_zone_hi"]  = df["swing_low"] + df["atr"] * cfg.inst_zone_atr_width
    df["sell_zone_hi"] = df["swing_high"]
    df["sell_zone_lo"] = df["swing_high"] - df["atr"] * cfg.inst_zone_atr_width
    if cfg.use_vwap_mid:
        df["mid_price"] = df["vwap"]
    else:
        df["mid_price"] = (df["swing_high"] + df["swing_low"]) / 2
    df["mid_zone_lo"] = df["mid_price"] - df["atr"] * cfg.inst_mid_atr_buffer
    df["mid_zone_hi"] = df["mid_price"] + df["atr"] * cfg.inst_mid_atr_buffer
    df["range_width"] = df["atr"] * cfg.inst_zone_atr_width

    # ---------- pivots ----------
    df["pivot_high"] = pivot_high(df["High"], a_pivot, a_pivot)
    df["pivot_low"]  = pivot_low(df["Low"],  a_pivot, a_pivot)
    df["last_pivot_high"] = df["pivot_high"].ffill()
    df["last_pivot_low"]  = df["pivot_low"].ffill()

    # ---------- choch swing highs/lows for liquidity zones ----------
    ph3 = pivot_high(df["High"], 3, 3)
    pl3 = pivot_low(df["Low"], 3, 3)
    choch_hi = ph3.ffill()
    choch_lo = pl3.ffill()
    df["liq_zone_high"] = df["High"].rolling(cfg.liquidity_zone_bars).max()
    df["liq_zone_low"]  = df["Low"].rolling(cfg.liquidity_zone_bars).min()

    # ---------- volume OK ----------
    df["vol_ok"]     = df["Volume"] > df["vol_ma"] * cfg.min_vol_multiplier
    df["vol_ok_rev"] = df["Volume"] > df["vol_ma"] * cfg.min_vol_mult_rev

    # ---------- trend direction ----------
    df["tema_bull"]   = df["tema"] > df["tema"].shift(1)
    df["lsma_bull"]   = df["lsma"] > df["lsma"].shift(1)
    df["ema100_bull"] = df["ema100"] > df["ema100"].shift(1)
    df["all_bull"] = df["tema_bull"] & df["lsma_bull"] & df["ema100_bull"]
    df["all_bear"] = ~df["tema_bull"] & ~df["lsma_bull"] & ~df["ema100_bull"]

    if cfg.require_trend_alignment:
        df["long_allowed"]  = df["all_bull"]
        df["short_allowed"] = df["all_bear"]
    else:
        df["long_allowed"]  = (df["tema"] > df["lsma"]) & (df["Close"] > df["ema100"])
        df["short_allowed"] = (df["tema"] < df["lsma"]) & (df["Close"] < df["ema100"])

    # ---------- TEMA/LSMA crossovers ----------
    df["tema_cross_above"] = (df["tema"] > df["lsma"]) & (df["tema"].shift(1) <= df["lsma"].shift(1))
    df["tema_cross_below"] = (df["tema"] < df["lsma"]) & (df["tema"].shift(1) >= df["lsma"].shift(1))

    # ---------- trend strength ----------
    df["tema_slope"]  = df["tema"] - df["tema"].shift(5)
    df["lsma_slope"]  = df["lsma"] - df["lsma"].shift(10)
    df["ema100_slope"] = df["ema100"] - df["ema100"].shift(20)
    df["trend_strength"] = (df["tema_slope"] + df["lsma_slope"] + df["ema100_slope"]) / 3
    df["norm_strength"]  = df["trend_strength"].abs() / df["atr"].replace(0, np.nan)

    # ---------- zone scoring ----------
    df["pct_tema"]   = (df["Close"] - df["tema"])   / df["tema"]   * 100
    df["pct_lsma"]   = (df["Close"] - df["lsma"])   / df["lsma"]   * 100
    df["pct_ema100"] = (df["Close"] - df["ema100"]) / df["ema100"] * 100
    df["zone_score"] = (df["pct_tema"] + df["pct_lsma"] + df["pct_ema100"]) / 3

    def classify_zone(z):
        if pd.isna(z):
            return "NEUTRAL_LOW"
        if z <= -zt1:
            return "EXTREME_OVERSOLD"
        if z <= -zt2:
            return "OVERSOLD"
        if z <= 0.5:
            return "NEUTRAL_LOW"
        if z <= 1.5:
            return "NEUTRAL_HIGH"
        if z <= zt2:
            return "OVERBOUGHT"
        return "EXTREME_OVERBOUGHT"

    df["zone"] = df["zone_score"].apply(classify_zone)

    # ---------- market sentiment scoring ----------
    df["bear_score"] = (
        (df["rsi_rev"] < 50).astype(int) +
        ((df["stoch_k"] < 50) | (df["stoch_k"] < df["stoch_d"])).astype(int) +
        (df["stoch_rsi"] < 50).astype(int) +
        (df["macd_line"] < df["macd_signal"]).astype(int) +
        (df["williams"] < -50).astype(int) +
        (df["cci"] < 0).astype(int)
    )
    df["bull_score"] = (
        (df["rsi_rev"] > 50).astype(int) +
        ((df["stoch_k"] > 50) | (df["stoch_k"] > df["stoch_d"])).astype(int) +
        (df["stoch_rsi"] > 50).astype(int) +
        (df["macd_line"] > df["macd_signal"]).astype(int) +
        (df["williams"] > -50).astype(int) +
        (df["cci"] > 0).astype(int)
    )
    df["strong_bull"] = df["bull_score"] >= 3
    df["strong_bear"] = df["bear_score"] >= 3

    # ---------- reversal trend logic ----------
    df["long_trend_rev"] = (
        (df["Close"] > df["kama"]) & df["kama_rising"] &
        ((df["adx"] > cfg.adx_strong) | (df["plus_di"] > df["minus_di"]))
    )
    df["short_trend_rev"] = (
        (df["Close"] < df["kama"]) & df["kama_falling"] &
        ((df["adx"] > cfg.adx_strong) | (df["minus_di"] > df["plus_di"]))
    )

    # ---------- liquidity sweep detection ----------
    min_sweep = df["atr"] * a_sweep
    df["liq_sweep_down"] = (
        (df["Low"] < df["last_pivot_low"]) &
        (df["Close"] > df["High"].shift(1)) &
        ((df["High"].shift(1) - df["Low"]) > min_sweep)
    )
    df["liq_sweep_up"] = (
        (df["High"] > df["last_pivot_high"]) &
        (df["Close"] < df["Low"].shift(1)) &
        ((df["High"] - df["Low"].shift(1)) > min_sweep)
    )

    # ---------- BoS (Break of Structure) ----------
    df["bullish_bos"] = (
        (df["Close"] > df["ema100"]) &
        (df["Close"].shift(1) <= df["ema100"].shift(1))
    )
    df["bearish_bos"] = (
        (df["Close"] < df["ema100"]) &
        (df["Close"].shift(1) >= df["ema100"].shift(1))
    )

    # BoS level tracking
    n = len(df)
    last_bull_bos_arr = np.full(n, np.nan)
    last_bear_bos_arr = np.full(n, np.nan)
    last_bull = np.nan
    last_bear = np.nan
    for i in range(len(df)):
        if df["bullish_bos"].iat[i]:
            last_bull = df["ema100"].iat[i]
        if df["bearish_bos"].iat[i]:
            last_bear = df["ema100"].iat[i]
        last_bull_bos_arr[i] = last_bull
        last_bear_bos_arr[i] = last_bear
    df["last_bull_bos"] = last_bull_bos_arr
    df["last_bear_bos"] = last_bear_bos_arr

    # BoS retest
    tol = a_retest
    df["bull_bos_retest"] = (
        df["last_bull_bos"].notna() &
        (df["Close"] < df["last_bull_bos"] * (1 + tol)) &
        (df["Close"] > df["last_bull_bos"] * (1 - tol)) &
        (df["Close"] > df["Open"])
    )
    df["bear_bos_retest"] = (
        df["last_bear_bos"].notna() &
        (df["Close"] > df["last_bear_bos"] * (1 - tol)) &
        (df["Close"] < df["last_bear_bos"] * (1 + tol)) &
        (df["Close"] < df["Open"])
    )

    # ---------- EMA100 test counter (stateful) ----------
    ema100_test_arr = np.zeros(n, dtype=int)
    test_count = 0
    last_test_price = np.nan
    for i in range(len(df)):
        c = df["Close"].iat[i]
        e = df["ema100"].iat[i]
        a = df["atr"].iat[i]
        if pd.isna(a) or pd.isna(e):
            ema100_test_arr[i] = 0
            continue
        if abs(c - e) < a * a_atr_test:
            if pd.isna(last_test_price) or last_test_price != c:
                test_count += 1
                last_test_price = c
        if abs(c - e) > a * 0.5:
            test_count = 0
        ema100_test_arr[i] = test_count
    df["ema100_test_count"] = ema100_test_arr

    # lower high / higher low
    df["lower_high"] = (df["High"] < df["last_pivot_high"]) & (df["High"] > df["High"].shift(1))
    df["higher_low"] = (df["Low"]  > df["last_pivot_low"])  & (df["Low"]  < df["Low"].shift(1))

    # ──────────────────────────────────────────────────────────────
    # TREND DURATION & EXHAUSTION (stateful loop)
    # ──────────────────────────────────────────────────────────────
    n = len(df)
    trend_bars_arr   = np.zeros(n, dtype=int)
    trend_bull_arr   = np.zeros(n, dtype=bool)
    trend_start_arr  = np.zeros(n, dtype=float)
    exhausted_arr    = np.zeros(n, dtype=bool)

    curr_trend_bars   = 0
    curr_trend_bull   = False
    curr_trend_start  = 0.0

    for i in range(n):
        ab = df["all_bull"].iat[i]
        ar = df["all_bear"].iat[i]
        c  = df["Close"].iat[i]

        if ab and (not curr_trend_bull or curr_trend_start == 0.0):
            curr_trend_bars  = 1
            curr_trend_bull  = True
            curr_trend_start = c
        elif ar and (curr_trend_bull or curr_trend_start == 0.0):
            curr_trend_bars  = 1
            curr_trend_bull  = False
            curr_trend_start = c
        elif (curr_trend_bull and ab) or (not curr_trend_bull and ar):
            curr_trend_bars += 1
        else:
            curr_trend_bars  = 0
            curr_trend_bull  = False
            curr_trend_start = 0.0

        trend_bars_arr[i]  = curr_trend_bars
        trend_bull_arr[i]  = curr_trend_bull
        trend_start_arr[i] = curr_trend_start

        # exhaustion check
        if curr_trend_bars > 0 and curr_trend_start > 0:
            c1 = curr_trend_bars > cfg.min_trend_bars * 2
            ns = df["norm_strength"].iat[i] if not pd.isna(df["norm_strength"].iat[i]) else 999
            c2 = ns < cfg.trend_strength_threshold
            c3 = (curr_trend_bull and not ab) or (not curr_trend_bull and not ar)
            rsi_div = False
            if cfg.use_momentum_divergence and i >= 10:
                r_now = df["rsi"].iat[i]
                r_10  = df["rsi"].iat[i - 10]
                c_now = df["Close"].iat[i]
                c_10  = df["Close"].iat[i - 10]
                if not (pd.isna(r_now) or pd.isna(r_10)):
                    if curr_trend_bull:
                        rsi_div = c_now > c_10 and r_now < r_10
                    else:
                        rsi_div = c_now < c_10 and r_now > r_10
            exhausted_arr[i] = c1 and (c2 or c3 or rsi_div)

    df["trend_bars"]     = trend_bars_arr
    df["trend_bull"]     = trend_bull_arr
    df["trend_start"]    = trend_start_arr
    df["trend_exhausted"] = exhausted_arr

    # ──────────────────────────────────────────────────────────────
    # SIGNAL PRIORITY ENGINE (stateful)
    # ──────────────────────────────────────────────────────────────
    signals        = np.zeros(n, dtype=int)   # +1 = long, -1 = short, 0 = none
    signal_sources = [""] * n
    position       = 0  # simulated position tracking for position_size == 0 checks

    for i in range(n):
        sig = 0
        src = ""

        c   = df["Close"].iat[i]
        vo  = df["vol_ok"].iat[i]
        vor = df["vol_ok_rev"].iat[i]

        # --- 1. INSTITUTIONAL ---
        if cfg.use_institutional:
            bz_lo = df["buy_zone_lo"].iat[i]
            bz_hi = df["buy_zone_hi"].iat[i]
            sz_lo = df["sell_zone_lo"].iat[i]
            sz_hi = df["sell_zone_hi"].iat[i]
            if not pd.isna(bz_lo) and bz_lo <= c <= bz_hi:
                sig = 1; src = "INSTITUTIONAL"
            elif not pd.isna(sz_lo) and sz_lo <= c <= sz_hi:
                sig = -1; src = "INSTITUTIONAL"

        # --- 2. IDEAL ---
        if sig == 0 and cfg.use_ideal and position == 0:
            # Buy conditions
            if df["liq_sweep_down"].iat[i] and c > df["ema100"].iat[i] and vo:
                sig = 1; src = "IDEAL"
            elif df["bull_bos_retest"].iat[i] and df["tema"].iat[i] > df["lsma"].iat[i] and vo:
                sig = 1; src = "IDEAL"
            elif df["zone"].iat[i] in ("EXTREME_OVERSOLD", "OVERSOLD") and c > df["ema100"].iat[i] and df["tema"].iat[i] > df["lsma"].iat[i] and vo:
                sig = 1; src = "IDEAL"
            # Sell conditions
            if df["bear_bos_retest"].iat[i] and df["tema"].iat[i] < df["lsma"].iat[i] and vo:
                sig = -1; src = "IDEAL"
            elif df["lower_high"].iat[i] and df["zone"].iat[i] in ("OVERBOUGHT", "EXTREME_OVERBOUGHT") and df["tema"].iat[i] < df["lsma"].iat[i]:
                sig = -1; src = "IDEAL"
            elif (df["ema100_test_count"].iat[i] >= cfg.max_test_count and
                  c < df["ema100"].iat[i] and
                  df["tema"].iat[i] < df["lsma"].iat[i] and
                  df["tema_cross_below"].iat[i]):
                sig = -1; src = "IDEAL"

        # --- 3. REVERSAL ---
        if sig == 0 and cfg.enable_reversals and position == 0:
            if df["long_trend_rev"].iat[i] and df["strong_bull"].iat[i] and vor:
                sig = 1; src = "REVERSAL"
            elif df["short_trend_rev"].iat[i] and df["strong_bear"].iat[i] and vor:
                sig = -1; src = "REVERSAL"

        # --- 4. DAVIN MA ---
        if sig == 0 and cfg.use_davin:
            dma_l = df["davin_ma_long"].iat[i]
            dma_s = df["davin_ma_short"].iat[i]
            if not (pd.isna(dma_l) or pd.isna(dma_s)):
                davin_buy = (c > dma_l and c < dma_s and position == 0) or \
                            (position == 0 and cfg.davin_buy_dip and df["davin_overall_change"].iat[i] > cfg.davin_dip_trigger)
                davin_sell = (c > dma_s and position > 0 and
                             (not cfg.davin_lower_close or (i > 0 and c < df["Low"].iat[i - 1])))
                if davin_buy:
                    sig = 1; src = "DAVIN MA"
                elif davin_sell:
                    sig = -1; src = "DAVIN MA"

        # --- 5. ORIGINAL TEMA/LSMA ---
        if sig == 0 and position == 0:
            if cfg.use_trend_exhaustion:
                long_orig = (df["long_allowed"].iat[i] and df["tema_cross_above"].iat[i] and vo) or \
                            (exhausted_arr[i] and not trend_bull_arr[i] and df["tema_cross_above"].iat[i] and vo)
                short_orig = (df["short_allowed"].iat[i] and df["tema_cross_below"].iat[i] and vo) or \
                             (exhausted_arr[i] and trend_bull_arr[i] and df["tema_cross_below"].iat[i] and vo)
            else:
                long_orig  = df["long_allowed"].iat[i] and df["tema_cross_above"].iat[i] and vo
                short_orig = df["short_allowed"].iat[i] and df["tema_cross_below"].iat[i] and vo
            if long_orig:
                sig = 1; src = "ORIGINAL"
            elif short_orig:
                sig = -1; src = "ORIGINAL"

        signals[i]        = sig
        signal_sources[i] = src

        # simple position tracking
        if sig == 1:
            position = 1
        elif sig == -1:
            position = -1
        # auto-close after max_bars_in_trade
        if position != 0:
            # find entry bar (rough)
            bars_held = 0
            for j in range(i, max(i - cfg.max_bars_in_trade - 1, -1), -1):
                if signals[j] != 0:
                    bars_held = i - j
                    break
            if bars_held >= cfg.max_bars_in_trade:
                position = 0

    df["signal"]        = signals
    df["signal_source"] = signal_sources
    return df


# ──────────────────────────────────────────────────────────────────────
# BACKTRADER STRATEGY WRAPPER
# ──────────────────────────────────────────────────────────────────────
class UltimateTEMALSMA(bt.Strategy):
    """Backtrader strategy that reads pre-computed signals from data feed."""

    params = dict(
        sl_atr_mult=1.5,
        tp_atr_mult=3.0,
        trail_atr_mult=1.0,
        max_bars=25,
        inst_base_pct=10.0,
    )

    def __init__(self):
        self.signal = self.data.signal
        self.signal_source = self.data.signal_source
        self.atr_val = self.data.atr
        self.entry_bar = None

    def next(self):
        sig = int(self.signal[0])
        a   = self.atr_val[0]

        # time-based exit
        if self.position and self.entry_bar is not None:
            if len(self) - self.entry_bar >= self.p.max_bars:
                self.close()
                self.entry_bar = None
                return

        if not self.position:
            if sig == 1:
                self.buy()
                self.entry_bar = len(self)
            elif sig == -1:
                self.sell()
                self.entry_bar = len(self)
        else:
            # exit on opposite signal
            if self.position.size > 0 and sig == -1:
                self.close()
                self.sell()
                self.entry_bar = len(self)
            elif self.position.size < 0 and sig == 1:
                self.close()
                self.buy()
                self.entry_bar = len(self)

            # ATR-based SL/TP check
            if self.position.size > 0:
                sl = self.position.price - self.p.sl_atr_mult * a
                tp = self.position.price + self.p.tp_atr_mult * a
                if self.data.close[0] <= sl or self.data.close[0] >= tp:
                    self.close()
                    self.entry_bar = None
            elif self.position.size < 0:
                sl = self.position.price + self.p.sl_atr_mult * a
                tp = self.position.price - self.p.tp_atr_mult * a
                if self.data.close[0] >= sl or self.data.close[0] <= tp:
                    self.close()
                    self.entry_bar = None


class PandasSignalData(bt.feeds.PandasData):
    """Extended Pandas data feed with signal columns."""
    lines = ("signal", "signal_source", "atr",)
    params = (
        ("signal",        -1),
        ("signal_source", -1),
        ("atr",           -1),
    )


# ──────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────
def print_signal_summary(df: pd.DataFrame):
    """Print summary of generated signals."""
    sigs = df[df["signal"] != 0].copy()
    if sigs.empty:
        print("\n  No signals generated in this period.")
        return

    print(f"\n{'='*70}")
    print(f"  SIGNAL SUMMARY — {len(sigs)} total signals")
    print(f"{'='*70}")

    for src in ["INSTITUTIONAL", "IDEAL", "REVERSAL", "DAVIN MA", "ORIGINAL"]:
        subset = sigs[sigs["signal_source"] == src]
        if not subset.empty:
            longs  = (subset["signal"] == 1).sum()
            shorts = (subset["signal"] == -1).sum()
            print(f"  {src:20s}  Longs: {longs:4d}  |  Shorts: {shorts:4d}  |  Total: {len(subset):4d}")

    print(f"{'='*70}")

    # last 10 signals
    tail = sigs.tail(10)
    print(f"\n  LAST {len(tail)} SIGNALS:")
    print(f"  {'Date':20s} {'Direction':10s} {'Source':15s} {'Close':>12s}")
    print(f"  {'-'*60}")
    for idx, row in tail.iterrows():
        d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        direction = "LONG" if row["signal"] == 1 else "SHORT"
        print(f"  {d:20s} {direction:10s} {row['signal_source']:15s} {row['Close']:12.2f}")


def print_indicator_snapshot(df: pd.DataFrame, symbol: str):
    """Print latest indicator values."""
    last = df.iloc[-1]
    inst = detect_instrument(symbol)
    print(f"\n{'='*70}")
    print(f"  INDICATOR SNAPSHOT — {symbol} ({inst})")
    print(f"{'='*70}")
    fields = [
        ("Close",     last["Close"]),
        ("TEMA(9)",   last.get("tema")),
        ("LSMA(25)",  last.get("lsma")),
        ("EMA(100)",  last.get("ema100")),
        ("ATR(14)",   last.get("atr")),
        ("RSI(14)",   last.get("rsi")),
        ("ADX",       last.get("adx")),
        ("VWAP",      last.get("vwap")),
        ("Zone",      last.get("zone")),
        ("Zone Score", last.get("zone_score")),
        ("Bull Score", last.get("bull_score")),
        ("Bear Score", last.get("bear_score")),
        ("Trend Bars", last.get("trend_bars")),
        ("Trend Exhausted", last.get("trend_exhausted")),
        ("Signal",    "LONG" if last.get("signal") == 1 else "SHORT" if last.get("signal") == -1 else "NONE"),
        ("Source",    last.get("signal_source", "")),
    ]
    for name, val in fields:
        if isinstance(val, float) and not pd.isna(val):
            print(f"  {name:20s}: {val:>14.4f}")
        else:
            print(f"  {name:20s}: {str(val):>14s}")
    print(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ULTIMATE TEMA/LSMA Strategy — Python Edition")
    parser.add_argument("--symbol",  default="BTC-USD",  help="Yahoo Finance ticker symbol")
    parser.add_argument("--days",    type=int, default=730, help="Days of historical data")
    parser.add_argument("--cash",    type=float, default=100_000, help="Starting capital")
    parser.add_argument("--no-plot", action="store_true",   help="Skip matplotlib chart")
    parser.add_argument("--csv",     default=None,          help="Export signals to CSV path")
    args = parser.parse_args()

    cfg = StrategyConfig()

    # ── fetch data ──
    print(f"\n  Fetching {args.days} days of {args.symbol} data...")
    end = datetime.now()
    start = end - timedelta(days=args.days)
    df = yf.download(args.symbol, start=start, end=end, progress=False)

    if df.empty:
        sys.exit(f"No data returned for {args.symbol}. Check the ticker.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"  Loaded {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    # ── compute signals ──
    print("  Computing indicators & signals...")
    df = compute_signals(df, args.symbol, cfg)

    # ── reporting ──
    print_indicator_snapshot(df, args.symbol)
    print_signal_summary(df)

    # ── export ──
    if args.csv:
        export_cols = ["Open", "High", "Low", "Close", "Volume",
                       "tema", "lsma", "ema100", "atr", "rsi", "adx",
                       "zone", "zone_score", "signal", "signal_source"]
        df[export_cols].to_csv(args.csv)
        print(f"\n  Signals exported to {args.csv}")

    # ── backtest ──
    print("\n  Running backtest...")
    # Prepare data for backtrader
    bt_df = df[["Open", "High", "Low", "Close", "Volume", "signal", "atr"]].copy()
    bt_df["signal_source"] = 0  # backtrader needs numeric; source used for logging only

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%

    data = PandasSignalData(
        dataname=bt_df,
        signal="signal",
        signal_source="signal_source",
        atr="atr",
    )
    cerebro.adddata(data)
    cerebro.addstrategy(UltimateTEMALSMA,
                        sl_atr_mult=cfg.sl_atr_mult,
                        tp_atr_mult=cfg.tp_atr_mult,
                        trail_atr_mult=cfg.trail_atr_mult,
                        max_bars=cfg.max_bars_in_trade)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    start_val = cerebro.broker.getvalue()
    cerebro.run()
    end_val = cerebro.broker.getvalue()

    pnl = end_val - start_val
    pct = (pnl / start_val) * 100

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Start Capital:  ${start_val:>14,.2f}")
    print(f"  End Capital:    ${end_val:>14,.2f}")
    print(f"  P&L:            ${pnl:>14,.2f}  ({pct:+.2f}%)")
    print(f"{'='*70}")

    # ── plot ──
    if not args.no_plot:
        try:
            cerebro.plot(style="candle", volume=True, barup="green", bardown="red")
        except Exception as e:
            print(f"  Plot failed ({e}) — install matplotlib or use --no-plot")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()