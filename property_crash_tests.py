# ==============================================================================
# property_crash_tests.py
# ==============================================================================
# Phase 1, Item 7 -- property-based crash testing.
#
# WHY THIS EXISTS
# ---------------
# The pipeline runs unattended over many generated strategies. A strategy that
# raises on some market condition it never saw in the sample data does not
# announce itself politely -- it takes down whatever step is running, and in
# this codebase that has repeatedly meant a whole stage silently producing
# nothing (steps 5, 6 and 7 all did exactly that).
#
# Hand-written test data cannot cover this, because the crashes come from
# conditions nobody thought to write down: a flat bar where high == low so an
# ATR is zero and something divides by it, a gap so large a stop is jumped
# entirely, a series where volume is zero throughout, a single-bar dataset.
#
# `hypothesis` generates these adversarially and shrinks any failure to the
# smallest input that still reproduces it -- so a report says "fails on 3 bars
# with high==low" rather than "fails somewhere in 50,000 rows".
#
# WHAT IS ASSERTED
# ----------------
# Deliberately weak properties. The strategy is NOT required to trade, or to be
# profitable, or to behave sensibly. It is required only to:
#
#   1. not raise
#   2. not produce NaN or infinite equity
#   3. not end with negative equity from a long-only, cash-bounded run
#   4. terminate
#
# Anything stronger would fail for reasons that are not bugs.
#
# OHLC VALIDITY
# -------------
# Generated bars always satisfy low <= min(open, close) <= max(open, close) <=
# high, and prices stay positive. Invalid bars would produce crashes that say
# nothing about the strategy -- a detector that reports failures the real world
# cannot produce trains people to ignore it.
# ==============================================================================

import io
import math
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# Below this many bars, Backtrader itself raises IndexError for ANY strategy
# whose indicator period exceeds the series length -- verified with a bare
# SMA(period=10) on a 1-bar and 3-bar frame. That is a framework constraint,
# not a strategy defect, and counting it as one would reject essentially every
# real strategy. Such cases are recorded separately as insufficient_data.
#
# The correct guard for tiny datasets lives in the runner that feeds the
# strategy, not in the strategy: never dispatch fewer bars than the warmup.
MIN_VIABLE_BARS = 50


# ==============================================================================
# PATHOLOGICAL DATA
# ==============================================================================

def make_frame(closes, volumes=None, start='2024-01-02', freq='D') -> pd.DataFrame:
    """Build a valid OHLCV frame from a close series. Always OHLC-consistent."""
    closes = np.asarray(closes, dtype=float)
    closes = np.maximum(closes, 1e-6)
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)
    lows = np.maximum(lows, 1e-9)
    vols = np.ones(n) * 1000 if volumes is None else np.asarray(volumes, dtype=float)
    return pd.DataFrame(
        {'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': vols},
        index=pd.date_range(start, periods=n, freq=freq))


# Named edge cases. These are the ones worth naming because each corresponds to
# a real market condition that has broken real backtests.
def pathological_cases(n: int = 80) -> Dict[str, pd.DataFrame]:
    rng = np.random.RandomState(0)
    base = 100.0
    cases = {
        # Every bar identical: ATR and stdev are zero. Anything dividing by a
        # volatility measure blows up here.
        'flat_line': make_frame(np.full(n, base)),

        # Zero volume throughout -- common in thin FX hours and in HistData.
        'zero_volume': make_frame(base + np.arange(n) * 0.01, volumes=np.zeros(n)),

        # A single enormous gap: stops are jumped, not filled.
        'huge_gap': make_frame(np.concatenate([
            np.full(n // 2, base), np.full(n - n // 2, base * 4)])),

        # Monotonic ramp: no mean reversion, indicators saturate.
        'monotonic_up': make_frame(base * np.exp(np.arange(n) * 0.02)),
        'monotonic_down': make_frame(base * np.exp(-np.arange(n) * 0.02)),

        # Price approaching zero: percentage maths gets unstable.
        'near_zero': make_frame(np.maximum(base * np.exp(-np.arange(n) * 0.3), 1e-5)),

        # Alternating spikes: every bar reverses.
        'sawtooth': make_frame(base + (np.arange(n) % 2) * base * 0.5),

        # Extreme volatility: 20% daily moves.
        'extreme_vol': make_frame(base * np.exp(np.cumsum(rng.normal(0, 0.20, n)))),

        # Barely enough bars to compute anything.
        'minimal': make_frame(np.full(3, base)),

        # One bar.
        'single_bar': make_frame(np.array([base])),
    }
    return cases


if HAS_HYPOTHESIS:
    @st.composite
    def ohlc_frames(draw, min_bars=MIN_VIABLE_BARS, max_bars=160):
        """
        Adversarial but VALID OHLCV frames.

        Widely-spread magnitudes on purpose: a strategy tuned on EUR-USD at 1.10
        can behave very differently at 60,000 (BTC) or 0.0001, and the pipeline
        runs across all of them.
        """
        n = draw(st.integers(min_value=min_bars, max_value=max_bars))
        magnitude = draw(st.sampled_from([1e-4, 1e-2, 1.0, 100.0, 1e4, 1e6]))
        closes = draw(st.lists(
            st.floats(min_value=0.01, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n))
        vols = draw(st.one_of(
            st.just(None),
            st.lists(st.floats(min_value=0.0, max_value=1e7,
                               allow_nan=False, allow_infinity=False),
                     min_size=n, max_size=n)))
        return make_frame(np.array(closes) * magnitude, volumes=vols)


# ==============================================================================
# RUNNER
# ==============================================================================

@dataclass
class CrashResult:
    name: str
    total: int = 0
    passed: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)
    insufficient_data: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        L = [f"\n{'=' * 68}", f"  PROPERTY CRASH TEST: {self.name}", '=' * 68]
        L.append(f"  Cases run: {self.total}   Passed: {self.passed}")
        if self.insufficient_data:
            names = ', '.join(f['case'] for f in self.insufficient_data)
            L.append(f"  Below {MIN_VIABLE_BARS} bars ({names}): raised, as any")
            L.append("  strategy with an indicator would. Not counted as a defect --")
            L.append("  guard this in the runner by not dispatching short series.")
        if self.clean:
            L.append("  No crashes, no NaN/inf equity on viable-length data.")
            L.append("  VERDICT: PASS")
        else:
            L.append(f"  {len(self.failures)} failure(s):")
            for f in self.failures[:10]:
                L.append(f"    {f['case']:16} {f['error']}")
            L.append("")
            L.append("  VERDICT: FAIL - the strategy raises or produces invalid")
            L.append("  equity on market conditions it will eventually meet.")
        L.append('=' * 68)
        return '\n'.join(L)


def _run_once(strategy_class, df, cash=100_000, params=None):
    """
    Run one backtest and return (ok, error_string).

    Interpretation of results is deliberately narrow: only crashes and invalid
    equity count. Not trading is fine; losing money is fine.
    """
    import backtrader as bt

    if df is None or df.empty:
        return True, None
    try:
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(cash)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(strategy_class, **(params or {}))
        buf = io.StringIO()
        with redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cerebro.run()
        final = cerebro.broker.getvalue()
    except ZeroDivisionError as e:
        return False, f"ZeroDivisionError: {e}"
    except IndexError as e:
        return False, f"IndexError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if final is None or (isinstance(final, float) and (math.isnan(final) or math.isinf(final))):
        return False, f"invalid final equity: {final}"
    if final < 0:
        return False, f"negative equity: {final:.2f}"
    return True, None


def run_pathological(strategy_class, cash=100_000, params=None,
                     name: str = '', n: int = 80) -> CrashResult:
    """Run the named edge cases. Fast, deterministic, no hypothesis needed."""
    name = name or getattr(strategy_class, '__name__', 'strategy')
    res = CrashResult(name=name)
    for case_name, df in pathological_cases(n).items():
        res.total += 1
        ok, err = _run_once(strategy_class, df, cash, params)
        if ok:
            res.passed += 1
        elif len(df) < MIN_VIABLE_BARS:
            # Framework constraint, not a strategy defect. Surfaced, not counted.
            res.insufficient_data.append({'case': case_name, 'error': err, 'bars': len(df)})
        else:
            res.failures.append({'case': case_name, 'error': err, 'bars': len(df)})
    return res


def run_fuzz(strategy_class, max_examples: int = 50, cash=100_000,
             params=None, name: str = '') -> CrashResult:
    """
    Hypothesis-driven fuzzing. Shrinks failures to a minimal reproducer.

    Slower than run_pathological and non-deterministic in what it explores, so
    it belongs in a nightly run rather than in the per-strategy gate.
    """
    name = name or getattr(strategy_class, '__name__', 'strategy')
    res = CrashResult(name=f"{name} (fuzz)")

    if not HAS_HYPOTHESIS:
        res.failures.append({'case': 'setup', 'error': 'hypothesis not installed'})
        return res

    found = []

    @settings(max_examples=max_examples, deadline=None,
              suppress_health_check=[HealthCheck.too_slow,
                                     HealthCheck.function_scoped_fixture])
    @given(df=ohlc_frames())
    def _prop(df):
        ok, err = _run_once(strategy_class, df, cash, params)
        if not ok:
            found.append({'case': f'fuzz[{len(df)} bars]', 'error': err, 'bars': len(df)})
        assert ok, err

    try:
        _prop()
        res.total = max_examples
        res.passed = max_examples
    except AssertionError as e:
        res.total = max_examples
        res.passed = 0
        res.failures.append(found[-1] if found else {'case': 'fuzz', 'error': str(e)})
    except Exception as e:
        res.failures.append({'case': 'fuzz', 'error': f"{type(e).__name__}: {e}"})
    return res


def gate(strategy_class, cash=100_000, params=None, verbose=False) -> bool:
    """
    Pathological cases only -- cheap enough for the per-strategy gate.
    Fuzzing is deliberately excluded here; run it separately.
    """
    r = run_pathological(strategy_class, cash=cash, params=params)
    if verbose:
        print(r.summary())
    return r.clean
