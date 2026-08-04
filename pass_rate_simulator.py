# ==============================================================================
# pass_rate_simulator.py
# ==============================================================================
# Replaces FTMOComplianceChecker.simulate_pass_rate -- item 11.3.
#
# WHAT WAS WRONG (worse than "resamples incorrectly")
# ---------------------------------------------------
#     shuffled = trades.sample(frac=1, random_state=random_seed + i)
#     result   = self.validate(shuffled, ...)
#
# Three problems, compounding:
#
#   1. THE SHUFFLE IS A NO-OP. Each trade row carries its own entry_date and
#      exit_date, and _build_equity_curve sorts events by timestamp before
#      doing anything. Reordering rows therefore changes nothing at all. Run
#      the original 1,000 times and you get 1,000 IDENTICAL results: pass_rate
#      is exactly 0.0 or 1.0, never anything between, and the reported
#      percentiles of max drawdown all collapse to the same number. It spends
#      1,000 backtests computing a value that one backtest already gave.
#
#   2. frac=1 WITHOUT replace=True is a permutation, not a bootstrap. Even if
#      the dates had been regenerated, every simulation would contain exactly
#      the same trades, so total return would be constant and the profit-target
#      rule would have no variance by construction.
#
#   3. FIXED SAMPLE SIZE. A challenge runs for a fixed WINDOW. How many trades
#      fall inside it is a random variable. Resampling the full historical set
#      assumes you take exactly N trades every time.
#
# THE CORRECTION
# --------------
# Sample trades WITH REPLACEMENT and lay them onto a fresh synthetic calendar
# spanning the challenge window. Both the composition of the trade set and its
# path through time then vary between simulations, which is what a pass-rate
# distribution is supposed to measure.
#
# Two resampling modes:
#
#   'iid'    trades drawn independently. Simple, but destroys serial structure.
#   'block'  contiguous runs of trades drawn together (stationary bootstrap
#            with geometric block lengths). Preserves streaks -- and streaks are
#            precisely what breaches a daily-loss rule. This is the default,
#            and it anticipates roadmap Phase 2 item 15.
#
# Trades keep their intraday time-of-day so that session structure survives the
# re-dating, and clustering is preserved by assigning per-day trade counts drawn
# from the observed distribution rather than spreading trades evenly.
#
# THE NUMBERS WILL GET WORSE. Real resampling produces breaches the no-op never
# could. That is the point.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MODE_IID = 'iid'
MODE_BLOCK = 'block'
VALID_MODES = (MODE_IID, MODE_BLOCK)

DEFAULT_WINDOW_DAYS = 30          # typical evaluation window
DEFAULT_MEAN_BLOCK = 5.0          # mean geometric block length, in trades


@dataclass
class PassRateResult:
    pass_rate: float
    n_simulations: int
    account_size: int
    phase: str
    n_trades: int
    mode: str
    window_days: int
    fail_reasons: Dict[str, int] = field(default_factory=dict)
    return_pct: Dict[str, float] = field(default_factory=dict)
    max_dd_pct: Dict[str, float] = field(default_factory=dict)
    degenerate: bool = False
    error: Optional[str] = None

    def summary(self) -> str:
        L = [f"\n{'=' * 64}", "  FTMO PASS RATE SIMULATION", '=' * 64]
        if self.error:
            L += [f"  [ERROR] {self.error}", '=' * 64]
            return '\n'.join(L)
        L.append(f"  Account:      ${self.account_size:,}   Phase: {self.phase}")
        L.append(f"  Trades:       {self.n_trades}   Window: {self.window_days}d   "
                 f"Mode: {self.mode}")
        L.append(f"  Simulations:  {self.n_simulations}")
        L.append("")
        L.append(f"  PASS RATE:    {self.pass_rate * 100:.1f}%")
        L.append("")
        L.append(f"  Return %      p5 {self.return_pct.get('p5', 0):>7.2f}   "
                 f"median {self.return_pct.get('p50', 0):>7.2f}   "
                 f"p95 {self.return_pct.get('p95', 0):>7.2f}")
        L.append(f"  Max DD %      p5 {self.max_dd_pct.get('p5', 0):>7.2f}   "
                 f"median {self.max_dd_pct.get('p50', 0):>7.2f}   "
                 f"p95 {self.max_dd_pct.get('p95', 0):>7.2f}")
        if self.fail_reasons:
            L.append("")
            L.append("  Failure causes (may overlap):")
            for k, v in sorted(self.fail_reasons.items(), key=lambda x: -x[1]):
                L.append(f"    {k:18} {v:5} ({v / self.n_simulations * 100:5.1f}%)")
        if self.degenerate:
            L.append("")
            L.append("  [WARN] Every simulation produced an identical outcome.")
            L.append("         With real resampling that should be impossible;")
            L.append("         it means the resampler is not varying anything.")
        L.append('=' * 64)
        return '\n'.join(L)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pass_rate': self.pass_rate,
            'n_simulations': self.n_simulations,
            'account_size': self.account_size,
            'phase': self.phase,
            'n_trades': self.n_trades,
            'mode': self.mode,
            'window_days': self.window_days,
            'fail_reasons': self.fail_reasons,
            'return_pct': self.return_pct,
            'max_dd_pct': self.max_dd_pct,
            'degenerate': self.degenerate,
            'error': self.error,
        }


# ==============================================================================
# RESAMPLING
# ==============================================================================

def _draw_indices(n_available: int, n_wanted: int, mode: str,
                  rng: np.random.RandomState, mean_block: float) -> np.ndarray:
    """Indices into the historical trade set, drawn WITH replacement."""
    if n_wanted <= 0:
        return np.array([], dtype=int)

    if mode == MODE_IID:
        return rng.randint(0, n_available, size=n_wanted)

    # Stationary bootstrap: geometric block lengths, wrapping at the end.
    # Preserves streaks, which is what actually breaches a daily-loss rule.
    p = 1.0 / max(mean_block, 1.0)
    out = []
    while len(out) < n_wanted:
        start = rng.randint(0, n_available)
        length = max(1, rng.geometric(p))
        for k in range(length):
            out.append((start + k) % n_available)
            if len(out) >= n_wanted:
                break
    return np.array(out[:n_wanted], dtype=int)


def _observed_daily_counts(trades: pd.DataFrame) -> np.ndarray:
    """
    Trades per active day in the source history. Resampling from this preserves
    clustering -- spreading trades evenly across the window would understate
    daily loss by construction, because the rule is breached by concentration.
    """
    try:
        d = pd.to_datetime(trades['entry_date']).dt.date
        counts = d.value_counts().values
        return counts if len(counts) else np.array([1])
    except Exception:
        return np.array([1])


def _times_of_day(trades: pd.DataFrame):
    """(entry_time, holding_duration) pairs, so session structure survives."""
    try:
        e = pd.to_datetime(trades['entry_date'])
        x = pd.to_datetime(trades['exit_date'])
        return list(zip(e.dt.time, (x - e))), True
    except Exception:
        return [], False


def build_synthetic_window(
    trades: pd.DataFrame,
    window_days: int,
    rng: np.random.RandomState,
    mode: str = MODE_BLOCK,
    mean_block: float = DEFAULT_MEAN_BLOCK,
    start: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    One synthetic challenge: trades drawn with replacement and re-dated onto a
    fresh calendar of `window_days` weekdays.

    Re-dating is the part the original was missing. Without it the sampled
    trades keep their original timestamps, the equity curve sorts them back into
    their original order, and the simulation reproduces history exactly.
    """
    n_avail = len(trades)
    daily_counts = _observed_daily_counts(trades)
    tod, has_tod = _times_of_day(trades)

    start = start or pd.Timestamp('2024-01-01')
    days, cursor = [], start
    while len(days) < window_days:
        if cursor.weekday() < 5:          # weekdays only; FX is closed weekends
            days.append(cursor)
        cursor += pd.Timedelta(days=1)

    rows = []
    for day in days:
        k = int(daily_counts[rng.randint(0, len(daily_counts))])
        if k <= 0:
            continue
        idx = _draw_indices(n_avail, k, mode, rng, mean_block)
        for j, i in enumerate(idx):
            src = trades.iloc[int(i)]
            if has_tod:
                t, dur = tod[int(i) % len(tod)]
                entry = pd.Timestamp.combine(day.date(), t)
                exit_ = entry + (dur if pd.notna(dur) else pd.Timedelta(hours=1))
            else:
                entry = day + pd.Timedelta(hours=9 + (j % 8))
                exit_ = entry + pd.Timedelta(hours=1)

            rows.append({
                'entry_date': entry,
                'exit_date': exit_,
                'entry_price': src.get('entry_price', 1.0),
                'exit_price': src.get('exit_price', 1.0),
                'size': src.get('size', 0),
                'symbol': src.get('symbol', 'EUR-USD'),
            })

    return pd.DataFrame(rows)


# ==============================================================================
# SIMULATION
# ==============================================================================

def simulate_pass_rate(
    checker,
    trades_df: pd.DataFrame,
    account_size: int = 100_000,
    phase: str = 'challenge',
    n_simulations: int = 1000,
    window_days: int = DEFAULT_WINDOW_DAYS,
    mode: str = MODE_BLOCK,
    mean_block: float = DEFAULT_MEAN_BLOCK,
    random_seed: int = 42,
    verbose: bool = True,
) -> PassRateResult:
    """
    Bootstrap estimate of P(pass) over a challenge window.

    Args:
        checker: an FTMOComplianceChecker (injected so this module does not
                 import ftmo_compliance and create a cycle).
        window_days: length of the simulated evaluation, in weekdays.
        mode: 'block' (default, preserves streaks) or 'iid'.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    n_trades = len(trades_df) if trades_df is not None else 0
    base = PassRateResult(
        pass_rate=0.0, n_simulations=n_simulations, account_size=account_size,
        phase=phase, n_trades=n_trades, mode=mode, window_days=window_days,
    )

    if n_trades < 4:
        base.error = ("Insufficient trades (need at least 4 to satisfy the "
                      "minimum trading days rule)")
        return base

    rng = np.random.RandomState(random_seed)
    passes = 0
    fails = {'daily_loss': 0, 'total_drawdown': 0, 'min_days': 0, 'profit_target': 0}
    returns, dds = [], []

    if verbose:
        print(f"\n{'=' * 60}")
        print("FTMO PASS RATE SIMULATION (bootstrap)")
        print(f"{'=' * 60}")
        print(f"Account: ${account_size:,}  Phase: {phase}  Mode: {mode}")
        print(f"Source trades: {n_trades}  Window: {window_days} weekdays")
        print(f"Simulations: {n_simulations}")
        print(f"{'=' * 60}")

    for i in range(n_simulations):
        if verbose and (i + 1) % 200 == 0:
            print(f"  Progress: {i + 1}/{n_simulations} "
                  f"({(i + 1) / n_simulations * 100:.0f}%)")

        sim = build_synthetic_window(
            trades_df, window_days, rng, mode=mode, mean_block=mean_block)
        if sim.empty:
            continue

        try:
            r = checker.validate(sim, account_size=account_size, phase=phase)
        except Exception:
            continue

        returns.append(r.final_return_pct)
        dds.append(r.max_total_drawdown_pct)

        if r.passed:
            passes += 1
        else:
            if not r.daily_loss_ok:
                fails['daily_loss'] += 1
            if not r.total_drawdown_ok:
                fails['total_drawdown'] += 1
            if not r.min_days_ok:
                fails['min_days'] += 1
            if not r.profit_target_ok:
                fails['profit_target'] += 1

    n_done = max(len(returns), 1)
    ra, da = np.array(returns), np.array(dds)

    base.pass_rate = passes / n_done
    base.fail_reasons = fails
    base.return_pct = {
        'p5': float(np.percentile(ra, 5)), 'p50': float(np.percentile(ra, 50)),
        'p95': float(np.percentile(ra, 95)), 'mean': float(ra.mean()),
    } if len(ra) else {}
    base.max_dd_pct = {
        'p5': float(np.percentile(da, 5)), 'p50': float(np.percentile(da, 50)),
        'p95': float(np.percentile(da, 95)), 'mean': float(da.mean()),
    } if len(da) else {}

    # Self-check: real resampling cannot produce an identical outcome every
    # time. If it does, the resampler is broken -- which is exactly the failure
    # the previous implementation had and never reported.
    base.degenerate = bool(len(ra) > 1 and np.std(ra) < 1e-12 and np.std(da) < 1e-12)

    if verbose:
        print(base.summary())

    return base
