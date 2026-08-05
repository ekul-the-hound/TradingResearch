# ==============================================================================
# portfolio_merge.py
# ==============================================================================
# Phase 3. Combines N validated strategies into ONE CanonicalResult that flows
# through the identical downstream pipeline -- same compliance checker, same
# filters, same dashboard panel.
#
# WHY NOT portfolio_engine.py
# ---------------------------
# portfolio_engine.py consumes equity curves and emits allocation weights (HRP,
# risk parity, min-variance). Useful, and unchanged by this module. But it
# cannot answer the question a prop challenge asks, for three reasons:
#
#   1. ALIGNMENT. Strategy A on M15 EURUSD and Strategy B on H1 GBPUSD have
#      different equity-curve index. Adding them requires first mapping every
#      trade onto a shared wall-clock calendar anchored to the firm's reset.
#
#   2. FLOATING P&L. The daily-loss rule is equity-based including unrealized.
#      Two strategies simultaneously holding losers produce a combined floating
#      drawdown that appears in neither closed-trade equity curve.
#
#   3. DILUTION. Weighted returns answer "what if I split capital." A challenge
#      usually runs strategies concurrently at full size on one account, where
#      losses ADD.
#
# The worked case, $100k account, 5% daily limit:
#
#       Day    Strat A    Strat B    Combined    vs -5%
#       Mon      +1.2%      -0.4%       +0.8%    ok
#       Tue      -3.0%      -2.8%       -5.8%    BREACH
#       Wed      +0.5%      +1.1%       +1.6%    ok
#
# Neither strategy breaches alone. The portfolio blows the account on Tuesday.
#
# WHAT THIS MODULE REFUSES TO DO
# ------------------------------
# - Merge a strategy with no real trade ledger. Summary statistics cannot be
#   placed on a calendar, and a fabricated placement would put invented trades
#   into a compliance check. Raises rather than approximating.
# - Merge over the union of date ranges. If A ran 2020-2023 and B ran 2022-2023,
#   the union reports A-only years as portfolio performance. Default is the
#   INTERSECTION, and the truncation is reported loudly.
# - Bootstrap strategies independently. See joint_block_bootstrap.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pytz
except ImportError:  # pragma: no cover
    pytz = None

from canonical_result import CanonicalResult
from firm_rules import FirmRules, UnsupportedRule, ftmo
import consistency_rule
# challenge_simulator imports firm_rules and consistency_rule but not this
# module, so this does not create a cycle.
import challenge_simulator

OVERLAP_INTERSECTION = 'intersection'
OVERLAP_UNION = 'union'
VALID_OVERLAP = (OVERLAP_INTERSECTION, OVERLAP_UNION)

DEFAULT_MEAN_BLOCK_DAYS = 5.0


class PortfolioMergeError(RuntimeError):
    """
    Raised when a merge cannot be performed honestly.

    Deliberately not caught-and-defaulted anywhere in this module. A portfolio
    that cannot be built is a result the caller needs to see.
    """


# ==============================================================================
# RESULT OBJECTS
# ==============================================================================

@dataclass
class MergeDiagnostics:
    """Everything the caller needs to judge whether the merge is trustworthy."""
    n_strategies: int = 0
    strategy_ids: List[str] = field(default_factory=list)
    overlap_mode: str = OVERLAP_INTERSECTION
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    trades_before_truncation: int = 0
    trades_after_truncation: int = 0
    trades_dropped_pct: float = 0.0
    per_strategy_native_window: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    same_day_loss_days: int = 0
    worst_combined_day_pct: float = 0.0
    worst_combined_day_date: Optional[str] = None
    unsupported_rules: List[UnsupportedRule] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        L = ['', '=' * 68, '  PORTFOLIO MERGE', '=' * 68]
        L.append(f"  Strategies:   {self.n_strategies}  "
                 f"({', '.join(self.strategy_ids)})")
        L.append(f"  Window:       {self.window_start} -> {self.window_end}  "
                 f"[{self.overlap_mode}]")
        L.append(f"  Trades:       {self.trades_after_truncation} kept of "
                 f"{self.trades_before_truncation} "
                 f"({self.trades_dropped_pct:.1f}% dropped)")
        L.append('')
        L.append(f"  Days where >1 strategy lost:  {self.same_day_loss_days}")
        L.append(f"  Worst combined day:           "
                 f"{self.worst_combined_day_pct:+.2f}%  "
                 f"({self.worst_combined_day_date})")
        if self.unsupported_rules:
            L.append('')
            L.append(f"  [PARTIAL] {len(self.unsupported_rules)} firm rule(s) "
                     f"NOT modelled:")
            for u in self.unsupported_rules:
                L.append(f"    - {u.capability.value}")
        if self.warnings:
            L.append('')
            L.append('  Warnings:')
            for w in self.warnings:
                L.append(f"    [!] {w}")
        L.append('=' * 68)
        return '\n'.join(L)


@dataclass
class PortfolioMergeResult:
    """
    The merge output.

    `canonical` is a normal CanonicalResult -- hand it to FTMOComplianceChecker,
    the filtering pipeline, or the dashboard exactly as you would a single
    strategy's result. That interchangeability is the point of Phase 3.
    """
    canonical: CanonicalResult
    merged_ledger: pd.DataFrame
    daily_pnl: pd.DataFrame          # index = firm-local date, cols = strategy
    diagnostics: MergeDiagnostics

    @property
    def is_fully_modelled(self) -> bool:
        return not self.diagnostics.unsupported_rules


# ==============================================================================
# LEDGER EXTRACTION
# ==============================================================================

_ENTRY_KEYS = ('entry_date', 'entry_time', 'datetime_in', 'open_datetime')
_EXIT_KEYS = ('exit_date', 'exit_time', 'datetime_out', 'close_datetime')
_PNL_KEYS = ('pnl', 'profit', 'pnl_net', 'net_profit')


def _first_key(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def extract_ledger(result: CanonicalResult) -> pd.DataFrame:
    """
    Pull a trade ledger out of a CanonicalResult, or refuse.

    The refusal is the important part. A CanonicalResult whose returns_source
    is 'synthetic' or 'none' has no executed trades to place on a calendar.
    Merging it would mean inventing timestamps for trades that never happened
    and then running a compliance check against them.
    """
    sid = result.strategy_id or result.strategy_name or '<unnamed>'

    if not result.trade_list:
        raise PortfolioMergeError(
            f"Strategy '{sid}' has an empty trade_list and cannot enter a "
            f"portfolio merge. Portfolio compliance is computed from trade "
            f"timestamps; summary statistics cannot be placed on a calendar. "
            f"Re-run the backtest with trade extraction enabled."
        )

    if result.returns_source in ('synthetic', 'mixed') or result.returns_synthetic:
        raise PortfolioMergeError(
            f"Strategy '{sid}' has returns_source='{result.returns_source}'. "
            f"Refusing to merge fabricated data into a portfolio that will be "
            f"compliance-checked."
        )

    rows = []
    for i, t in enumerate(result.trade_list):
        entry = _first_key(t, _ENTRY_KEYS)
        exit_ = _first_key(t, _EXIT_KEYS)
        pnl = _first_key(t, _PNL_KEYS)

        if exit_ is None:
            raise PortfolioMergeError(
                f"Strategy '{sid}' trade #{i} has no exit timestamp "
                f"(looked for {list(_EXIT_KEYS)}). Every trade needs a "
                f"timestamp to be assigned to a trading day."
            )
        if pnl is None:
            raise PortfolioMergeError(
                f"Strategy '{sid}' trade #{i} has no P&L field "
                f"(looked for {list(_PNL_KEYS)})."
            )

        rows.append({
            'strategy_id': sid,
            'entry_date': pd.to_datetime(entry) if entry is not None
                          else pd.to_datetime(exit_),
            'exit_date': pd.to_datetime(exit_),
            'entry_price': t.get('entry_price', np.nan),
            'exit_price': t.get('exit_price', np.nan),
            'size': t.get('size', t.get('quantity', 0.0)),
            'symbol': t.get('symbol', result.symbol or 'UNKNOWN'),
            'pnl': float(pnl),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise PortfolioMergeError(f"Strategy '{sid}' produced an empty ledger.")
    return df.sort_values(by='exit_date').reset_index(drop=True)


# ==============================================================================
# CALENDAR ALIGNMENT
# ==============================================================================

def _firm_local_dates(ts: Any, tz_name: str) -> Any:
    # Delegates to challenge_simulator.firm_local_dates, which is the one
    # implementation. Kept as a thin alias so existing callers and tests
    # in this module do not have to change.
    #
    # Annotated Any because df[col] is typed Series | DataFrame by the
    # pandas stubs even though a single-label lookup always yields a
    # Series. pd.to_datetime accepts either.
    return challenge_simulator.firm_local_dates(ts, tz_name)


def daily_pnl_matrix(ledger: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """
    Per-strategy realised P&L per firm-local trading day.

    Rows are days, columns are strategies. This is the artifact the joint
    bootstrap resamples, because a day is the unit the daily-loss rule
    operates on.

    Note the limitation, stated rather than hidden: this attributes P&L at
    EXIT. Floating P&L on positions still open across a day boundary is not
    reflected here. The compliance check does not use this matrix -- it runs
    on the merged ledger through the real equity-curve builder, which does
    handle floating. This matrix is for resampling and diagnostics only.
    """
    df = ledger.copy()
    df['trade_date'] = _firm_local_dates(df['exit_date'], tz_name)
    pivot = df.pivot_table(
        index='trade_date', columns='strategy_id', values='pnl',
        aggfunc='sum', fill_value=0.0,
    )
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()


# ==============================================================================
# THE MERGE
# ==============================================================================

def merge_strategies(
    results: Sequence[CanonicalResult],
    rules: Optional[FirmRules] = None,
    account_size: float = 100_000.0,
    weights: Optional[Dict[str, float]] = None,
    overlap: str = OVERLAP_INTERSECTION,
    portfolio_id: str = 'portfolio',
) -> PortfolioMergeResult:
    """
    Combine N strategies into one CanonicalResult at trade level.

    Args:
        results:      strategies to combine. Each needs a real trade_list.
        rules:        FirmRules; defaults to the FTMO profile.
        account_size: starting balance for the combined account.
        weights:      per-strategy capital multiplier, keyed by strategy_id.
                      Default 1.0 each -- i.e. every strategy runs at full
                      size concurrently, which is the challenge case. Pass
                      fractional weights to model capital splitting.
        overlap:      'intersection' (default, honest) or 'union' (reports
                      single-strategy periods as portfolio performance --
                      allowed, but warned about).
    """
    if rules is None:
        rules = ftmo()
    if overlap not in VALID_OVERLAP:
        raise ValueError(f"overlap must be one of {VALID_OVERLAP}, got {overlap!r}")
    if len(results) < 2:
        raise PortfolioMergeError(
            f"A portfolio needs at least 2 strategies, got {len(results)}. "
            f"For a single strategy use its CanonicalResult directly."
        )

    diag = MergeDiagnostics(overlap_mode=overlap)
    diag.unsupported_rules = rules.unsupported()

    # -- 1. extract, refusing anything without a real ledger ---------------
    ledgers = []
    for r in results:
        led = extract_ledger(r)
        sid = led['strategy_id'].iloc[0]
        if sid in diag.strategy_ids:
            raise PortfolioMergeError(
                f"Duplicate strategy_id '{sid}'. Portfolio members must be "
                f"distinguishable; otherwise their P&L is silently pooled."
            )
        diag.strategy_ids.append(sid)
        diag.per_strategy_native_window[sid] = (
            str(led['exit_date'].min()), str(led['exit_date'].max())
        )
        ledgers.append(led)

    diag.n_strategies = len(ledgers)

    # -- 2. apply capital weights ------------------------------------------
    weights = weights or {}
    for led in ledgers:
        sid = led['strategy_id'].iloc[0]
        w = float(weights.get(sid, 1.0))
        if w <= 0:
            raise PortfolioMergeError(
                f"Weight for '{sid}' is {w}; weights must be positive."
            )
        led['pnl'] = led['pnl'] * w
        led['size'] = led['size'] * w
        led['weight'] = w

    combined = pd.concat(ledgers, ignore_index=True)
    diag.trades_before_truncation = len(combined)

    # -- 3. window resolution ----------------------------------------------
    starts = [led['exit_date'].min() for led in ledgers]
    ends = [led['exit_date'].max() for led in ledgers]

    if overlap == OVERLAP_INTERSECTION:
        win_start, win_end = max(starts), min(ends)
        if win_start >= win_end:
            raise PortfolioMergeError(
                "Strategies have no overlapping test period, so there is no "
                "window in which they traded together. Native windows: "
                + '; '.join(f"{k}: {v[0]}..{v[1]}"
                            for k, v in diag.per_strategy_native_window.items())
            )
        combined = combined[
            (combined['exit_date'] >= win_start) &
            (combined['exit_date'] <= win_end)
        ].copy()
    else:
        win_start, win_end = min(starts), max(ends)
        diag.warnings.append(
            "overlap='union': periods where only one strategy was live are "
            "being reported as portfolio performance. Diversification will "
            "look better than it is."
        )

    # pd.DataFrame(...) re-pins the type: boolean masking above widens
    # `combined` to DataFrame | Series under the stubs, and Series has no
    # `by` parameter on sort_values. Runtime behaviour is unchanged.
    combined = pd.DataFrame(combined).sort_values(
        by='exit_date').reset_index(drop=True)
    diag.trades_after_truncation = len(combined)
    diag.window_start, diag.window_end = str(win_start), str(win_end)
    if diag.trades_before_truncation:
        diag.trades_dropped_pct = 100.0 * (
            1 - diag.trades_after_truncation / diag.trades_before_truncation
        )
    if diag.trades_dropped_pct > 50:
        diag.warnings.append(
            f"{diag.trades_dropped_pct:.0f}% of trades fell outside the "
            f"overlap window. The portfolio is being judged on a small slice "
            f"of the evidence."
        )
    if combined.empty:
        raise PortfolioMergeError("No trades survived window truncation.")

    # -- 4. daily matrix + clustering diagnostics --------------------------
    daily = daily_pnl_matrix(combined, rules.reset_timezone)

    losing = (daily < 0)
    diag.same_day_loss_days = int((losing.sum(axis=1) > 1).sum())

    combined_daily = daily.sum(axis=1)
    if len(combined_daily):
        worst_idx = combined_daily.idxmin()
        diag.worst_combined_day_pct = float(
            combined_daily.loc[worst_idx] / account_size * 100.0
        )
        diag.worst_combined_day_date = str(pd.Timestamp(str(worst_idx)).date())

        limit_pct = -rules.max_daily_loss_pct * 100.0
        if diag.worst_combined_day_pct <= limit_pct:
            diag.warnings.append(
                f"Worst combined day {diag.worst_combined_day_pct:.2f}% "
                f"breaches the {limit_pct:.2f}% daily limit. Check whether "
                f"any constituent breaches alone -- if not, this loss is "
                f"created by combining them."
            )

    # -- 5. build the CanonicalResult --------------------------------------
    canonical = _to_canonical(
        combined, daily, account_size, rules, portfolio_id, diag
    )

    return PortfolioMergeResult(
        canonical=canonical,
        merged_ledger=combined,
        daily_pnl=daily,
        diagnostics=diag,
    )


def _to_canonical(
    ledger: pd.DataFrame,
    daily: pd.DataFrame,
    account_size: float,
    rules: FirmRules,
    portfolio_id: str,
    diag: MergeDiagnostics,
) -> CanonicalResult:
    """
    Wrap the merged ledger as a CanonicalResult.

    trade_list is populated with real merged trades, so CanonicalResult's own
    _compute_arrays derives returns with returns_source='trade_list'. No
    special-casing, no synthetic branch -- the portfolio earns real provenance
    the same way a single strategy does.
    """
    trade_list = ledger.to_dict('records')
    for t in trade_list:
        for k in ('entry_date', 'exit_date'):
            if isinstance(t.get(k), pd.Timestamp):
                t[k] = t[k].isoformat()

    # np.asarray rather than staying in pandas: Series arithmetic is typed
    # loosely enough that every scalar conversion below needs a cast, and
    # pd.to_numeric is declared as returning a union that includes
    # DatetimeIndex. The ndarray is well-typed and faster.
    pnl_arr = np.asarray(ledger['pnl'], dtype=float)
    total_pnl = float(pnl_arr.sum())
    ending = account_size + total_pnl
    wins = int((pnl_arr > 0).sum())
    n = len(ledger)

    gross_win = float(pnl_arr[pnl_arr > 0].sum())
    gross_loss = float(-pnl_arr[pnl_arr < 0].sum())

    n_days = max(len(daily), 1)

    cr = CanonicalResult(
        strategy_id=portfolio_id,
        strategy_name=f"Portfolio[{'+'.join(diag.strategy_ids)}]",
        symbol='MULTI',
        timeframe='MULTI',
        strategy_params={
            'members': diag.strategy_ids,
            'firm': rules.firm_name,
            'overlap_mode': diag.overlap_mode,
        },
        total_return_pct=total_pnl / account_size * 100.0,
        total_trades=n,
        win_rate=(wins / n * 100.0) if n else None,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else None,
        starting_value=account_size,
        ending_value=ending,
        start_date=diag.window_start or '',
        end_date=diag.window_end or '',
        trades_per_day=n / n_days,
        avg_trade_return_pct=(total_pnl / n / account_size * 100.0) if n else 0.0,
        trade_list=trade_list,
    )

    # CanonicalResult has NO __post_init__ -- _compute_arrays() is invoked
    # explicitly by from_backtest() and nowhere else. A directly constructed
    # result therefore carries returns_source='none' no matter how complete its
    # trade_list is, and require_returns() would refuse a perfectly real
    # portfolio. Trigger it here.
    cr._compute_arrays()

    daily_ret = daily.sum(axis=1) / account_size
    if len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0:
        cr.sharpe_ratio = float(
            daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252)
        )

    eq = account_size + daily.sum(axis=1).cumsum()
    if len(eq):
        peak = eq.cummax()
        cr.max_drawdown_pct = float(((peak - eq) / peak).max() * 100.0)

    return cr


# ==============================================================================
# JOINT BLOCK BOOTSTRAP
# ==============================================================================

def joint_block_bootstrap(
    daily: pd.DataFrame,
    n_simulations: int = 1000,
    window_days: int = 30,
    mean_block_days: float = DEFAULT_MEAN_BLOCK_DAYS,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Resample whole trading days across ALL strategies together.

    THE WORD DOING THE WORK IS 'JOINT'.

    A day is drawn as an entire ROW of the daily matrix, so every strategy's
    P&L for that day travels together. If A and B both lost on 12 March, they
    lose together in every simulation that draws 12 March.

    Bootstrapping each strategy independently would break exactly that link.
    Independent draws implicitly assume the strategies never lose on the same
    day, which deletes the only scenario that actually ends a challenge, and
    returns a P(pass) that is too high in the specific direction you would
    most like to believe.

    BLOCKS, not single days: contiguous runs are drawn with geometric lengths
    (stationary bootstrap), because losing streaks are what breach daily and
    total limits. Independent day draws would smooth streaks away.

    Returns:
        (n_simulations, window_days) array of combined daily P&L in currency.
    """
    if daily.empty:
        raise PortfolioMergeError("Cannot bootstrap an empty daily matrix.")
    if window_days <= 0:
        raise ValueError("window_days must be positive.")
    if mean_block_days <= 0:
        raise ValueError("mean_block_days must be positive.")

    values = daily.values                      # (n_days, n_strategies)
    n_days = values.shape[0]

    if n_days < 2:
        raise PortfolioMergeError(
            f"Only {n_days} trading day(s) available; a bootstrap over a "
            f"{window_days}-day window would be resampling a single point."
        )
    if n_days < window_days:
        # Allowed, but the caller should know the window is being assembled
        # from heavy reuse of a short history.
        pass

    rng = np.random.RandomState(random_seed)
    p = 1.0 / mean_block_days                  # geometric continuation prob
    out = np.empty((n_simulations, window_days), dtype=np.float64)

    for s in range(n_simulations):
        filled = 0
        row = np.empty(window_days, dtype=np.float64)
        while filled < window_days:
            start = rng.randint(0, n_days)
            block_len = rng.geometric(p)
            take = min(block_len, window_days - filled)
            for j in range(take):
                # wrap so blocks near the end of history stay contiguous
                day = values[(start + j) % n_days]
                row[filled + j] = day.sum()
            filled += take
        out[s] = row

    return out


def bootstrap_summary(sims: np.ndarray, account_size: float,
                      rules: Optional[FirmRules] = None) -> Dict[str, Any]:
    """
    Descriptive stats over bootstrap paths, for ONE stage.

    THE PASS RATE COMES FROM challenge_simulator.walk_stage.

    An earlier version of this function computed its own, by scanning the full
    window for any breach and comparing final equity to the target. That is
    systematically pessimistic, because a trader who reaches the target STOPS.
    Days after the win never happen, so a breach among them is not a breach --
    but the fixed-window scan counted it, and final equity missed wins that
    were later given back. On a t-distributed test path the two approaches
    disagreed by more than a factor of two (5.5% vs 13.7%).

    Two functions in one codebase disagreeing about P(pass) is the kind of
    thing that gets believed selectively later, so the rule mechanics now live
    in exactly one place and this delegates to them.

    The breach RATES below remain honest descriptions of the raw path
    distribution over the whole window -- they answer "how often does this
    strategy have a day that big", which is a different and still useful
    question from "would it pass". They are named accordingly.
    """
    if rules is None:
        rules = ftmo()

    cum = np.cumsum(sims, axis=1)
    equity = account_size + cum

    daily_limit = -rules.max_daily_loss_pct * account_size
    dd_floor = rules.drawdown_floor(account_size)

    # Whole-window descriptive statistics. NOT pass/fail verdicts.
    daily_breach = (sims <= daily_limit).any(axis=1)
    dd_breach = (equity <= dd_floor).any(axis=1)
    final = equity[:, -1]
    target = rules.profit_target_value(account_size, 'challenge')
    survived = ~(daily_breach | dd_breach)
    hit_target = final >= target

    # The verdict, walked day by day with early stopping.
    stage = challenge_simulator.StageSpec.from_rules(rules, 'challenge')
    outcomes = [
        challenge_simulator.walk_stage(row, account_size, stage, rules)['outcome']
        for row in sims
    ]
    n = len(outcomes) or 1
    n_passed = sum(1 for o in outcomes if o == challenge_simulator.PASSED)
    # walk_stage only returns FAIL_CONSISTENCY at the moment a path would
    # otherwise have passed, so passed + consistency-failures is exactly the
    # pass rate the same paths would score with the rule switched off. No
    # second pass over the data needed.
    n_cons_fail = sum(1 for o in outcomes
                      if o == challenge_simulator.FAIL_CONSISTENCY)

    consistency = consistency_rule.consistency_stats(
        sims, rules.consistency_max_day_pct)

    return {
        'n_simulations': int(sims.shape[0]),
        'window_days': int(sims.shape[1]),
        'daily_breach_rate': float(daily_breach.mean()),
        'drawdown_breach_rate': float(dd_breach.mean()),
        'survived_rate': float(survived.mean()),
        'reached_target_rate': float(hit_target.mean()),
        'consistency_breach_rate': consistency['breach_rate'],
        'consistency': consistency,
        'consistency_stage_failures': n_cons_fail,
        # From the day-by-day walk, so early stopping is respected.
        'modelled_pass_rate': float(n_passed / n),
        'pass_rate_ignoring_consistency': float((n_passed + n_cons_fail) / n),
        # Kept for continuity, clearly labelled: this is the old fixed-window
        # figure, which understates the pass rate. Do not display it as P(pass).
        'fixed_window_pass_rate': float((survived & hit_target).mean()),
        'final_equity_p5': float(np.percentile(final, 5)),
        'final_equity_p50': float(np.percentile(final, 50)),
        'final_equity_p95': float(np.percentile(final, 95)),
        'unsupported_rules': [u.capability.value for u in rules.unsupported()],
        'is_complete': rules.is_fully_modelled,
        'caveat': rules.caveat_line(),
    }