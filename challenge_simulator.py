# ==============================================================================
# challenge_simulator.py
# ==============================================================================
# Phase 4. Simulates the WHOLE path to a funded account, not one window of it.
#
# WHAT WAS MISSING
# ----------------
# pass_rate_simulator.py evaluates a single phase (`phase='challenge'`) and has
# no concept of verification. bootstrap_summary in portfolio_merge does the
# same for a portfolio. Both answer "would this pass stage one", and a funded
# account requires passing every stage in sequence.
#
# Two corrections fall out of walking the stages properly.
#
# 1. EARLY STOPPING. A trader who hits +10% on day 8 STOPS. They do not keep
#    trading to the end of the window. Evaluating a fixed 30-day window and
#    checking final equity therefore fails paths that had already won and then
#    gave it back -- a pessimism the fixed-window approach cannot express.
#    This module walks day by day and halts the moment the target is reached
#    (subject to the minimum-trading-days rule).
#
# 2. EARLY STOPPING FIGHTS THE CONSISTENCY RULE. Stopping early concentrates
#    the profit into fewer days, which makes the best day a LARGER share of the
#    total. A strategy that reaches the target in four days has almost
#    certainly violated a 30% consistency cap in doing so. These two rules pull
#    in opposite directions and only a sequential walk shows the interaction.
#
# WHAT IT REFUSES TO DO
# ---------------------
# The conditional P(stage 2 | passed stage 1) is estimated on the subset of
# paths that actually reached stage 2. When that subset is small the estimate
# is noise, so the result carries the subset size and a warning rather than a
# clean-looking percentage. Same principle as everywhere else here: an answer
# that cannot be supported says so.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import consistency_rule
from firm_rules import FirmRules, ftmo

try:
    import pytz
except ImportError:  # pragma: no cover
    pytz = None


def firm_local_dates(ts, tz_name: str):
    """
    Map timestamps to the firm's trading date.

    The daily rules reset at midnight in the firm's timezone, so 'which
    day did this land on' is a timezone question, not a naive-date one.
    A 23:30 UTC trade belongs to the NEXT Prague day in winter.

    Canonical implementation. portfolio_merge delegates here rather than
    keeping its own copy -- two versions of this drifting apart would
    silently reassign trades to different days in different modules.
    Lives in challenge_simulator because portfolio_merge imports this
    module and not the reverse.
    """
    import pandas as _pd
    s = _pd.to_datetime(ts)
    if pytz is None:
        return s.dt.date
    tz = pytz.timezone(tz_name)
    if s.dt.tz is None:
        s = s.dt.tz_localize('UTC')
    return s.dt.tz_convert(tz).dt.date

# Failure reasons -- exhaustive and mutually exclusive.
FAIL_DAILY_LOSS = 'daily_loss'
FAIL_DRAWDOWN = 'total_drawdown'
FAIL_TIME_LIMIT = 'ran_out_of_days'
FAIL_CONSISTENCY = 'consistency'
FAIL_NOT_REACHED = 'target_not_reached'
PASSED = 'passed'

# Below this many surviving paths, a conditional rate is not reportable.
MIN_CONDITIONAL_SAMPLE = 30


def _reached(equity: float, target: float) -> bool:
    """
    Has equity reached the target, counted to the cent?

    A plain >= is wrong at the boundary: initial * (1 + pct) carries a
    float artifact, so 100_000 * 1.10 is 110000.00000000001 and a path
    landing on exactly +10.00% is judged short by one hundred-billionth
    of a cent. Firms count money to the cent; so does this.
    """
    return round(equity, 2) >= round(target, 2)


@dataclass
class StageSpec:
    """One evaluation stage."""
    name: str
    profit_target_pct: float
    max_days: Optional[int] = None
    min_trading_days: int = 4

    @classmethod
    def from_rules(cls, rules: FirmRules, phase: str,
                   max_days: Optional[int] = None) -> "StageSpec":
        if phase not in rules.profit_targets:
            raise ValueError(
                f"{rules.firm_name} has no phase {phase!r}. "
                f"Known: {sorted(rules.profit_targets)}")
        return cls(
            name=phase,
            profit_target_pct=rules.profit_targets[phase],
            max_days=max_days if max_days is not None else rules.max_calendar_days,
            min_trading_days=rules.min_trading_days,
        )


@dataclass
class StageStats:
    name: str
    n_entered: int = 0
    n_passed: int = 0
    outcomes: Dict[str, int] = field(default_factory=dict)
    days_used: List[int] = field(default_factory=list)

    @property
    def pass_rate(self) -> Optional[float]:
        """None when nobody reached this stage -- not zero."""
        if self.n_entered == 0:
            return None
        return self.n_passed / self.n_entered

    @property
    def reliable(self) -> bool:
        return self.n_entered >= MIN_CONDITIONAL_SAMPLE

    @property
    def median_days(self) -> Optional[float]:
        return float(np.median(self.days_used)) if self.days_used else None


@dataclass
class ChallengeResult:
    stages: List[StageStats] = field(default_factory=list)
    n_simulations: int = 0
    n_funded: int = 0
    rules_firm: str = ''
    unchecked_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Total days taken by each path that cleared EVERY stage. Needed
    # because the median of per-path totals is not the sum of per-stage
    # medians -- different paths sit at the median of each stage.
    funded_path_days: List[int] = field(default_factory=list)

    @property
    def p_funded(self) -> float:
        """Probability of clearing every stage. The number that matters."""
        return self.n_funded / self.n_simulations if self.n_simulations else 0.0

    @property
    def is_complete(self) -> bool:
        return not self.unchecked_rules

    def expected_attempts(self) -> Optional[float]:
        """
        Mean attempts to get funded once, assuming independent retries.

        None when p_funded is zero: the expectation is infinite, and printing
        a large finite number would misrepresent 'never' as 'expensive'.
        """
        p = self.p_funded
        return (1.0 / p) if p > 0 else None

    def expected_fee(self, fee_per_attempt: float) -> Optional[float]:
        att = self.expected_attempts()
        return att * fee_per_attempt if att is not None else None

    def median_days_to_funded(self) -> Optional[float]:
        """
        True median of end-to-end days over paths that got funded.

        Not the sum of per-stage medians: the path sitting at the median
        of stage one is generally not the one at the median of stage two,
        so adding them describes a path that may not exist. None when no
        path was funded, because there is nothing to take a median of.
        """
        if not self.funded_path_days:
            return None
        return float(np.median(self.funded_path_days))

    def days_to_funded_percentile(self, q: float) -> Optional[float]:
        """q in [0, 100]. Useful for the 90-day window question."""
        if not self.funded_path_days:
            return None
        return float(np.percentile(self.funded_path_days, q))

    def p_funded_within(self, days: int) -> float:
        """
        Share of ALL simulated attempts funded within a day budget.

        Denominator is every path, not just the funded ones: the question
        is "if I start today, what is the chance I am funded in N days",
        and failing does not stop the clock.
        """
        if not self.n_simulations:
            return 0.0
        return sum(1 for d in self.funded_path_days
                   if d <= days) / self.n_simulations

    def summary(self) -> str:
        L = ['', '=' * 70, f'  CHALLENGE SIMULATION -- {self.rules_firm}', '=' * 70]
        L.append(f"  Simulations: {self.n_simulations}")
        L.append('')
        for s in self.stages:
            pr = s.pass_rate
            rate = 'n/a' if pr is None else f"{pr * 100:5.1f}%"
            flag = '' if s.reliable else '   [SMALL SAMPLE]'
            L.append(f"  {s.name:<14} entered {s.n_entered:>6}  "
                     f"passed {s.n_passed:>6}  rate {rate}{flag}")
            for reason, n in sorted(s.outcomes.items(),
                                    key=lambda kv: -kv[1]):
                if reason == PASSED:
                    continue
                pct = 100.0 * n / s.n_entered if s.n_entered else 0.0
                L.append(f"      {reason:<22} {n:>6}  ({pct:4.1f}%)")
        L.append('')
        L.append(f"  P(funded) = {self.p_funded * 100:.2f}%")
        att = self.expected_attempts()
        L.append(f"  Expected attempts: "
                 + ('never (0 successes)' if att is None else f"{att:.1f}"))
        med = self.median_days_to_funded()
        if med is not None:
            p90 = self.days_to_funded_percentile(90)
            L.append(f"  Days to funded: median {med:.0f}, "
                     f"p90 {p90:.0f}" if p90 is not None else
                     f"  Days to funded: median {med:.0f}")
            L.append(f"  P(funded within 90 days) = "
                     f"{self.p_funded_within(90) * 100:.2f}%")
        if self.unchecked_rules:
            L.append('')
            L.append(f"  [PARTIAL] not checked: {', '.join(self.unchecked_rules)}")
        for w in self.warnings:
            L.append(f"  [!] {w}")
        L.append('=' * 70)
        return '\n'.join(L)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'firm': self.rules_firm,
            'n_simulations': self.n_simulations,
            'p_funded': self.p_funded,
            'expected_attempts': self.expected_attempts(),
            'median_days_to_funded': self.median_days_to_funded(),
            'p90_days_to_funded': self.days_to_funded_percentile(90),
            'p_funded_within_90d': self.p_funded_within(90),
            'is_complete': self.is_complete,
            'unchecked_rules': list(self.unchecked_rules),
            'warnings': list(self.warnings),
            'stages': [{
                'name': s.name,
                'n_entered': s.n_entered,
                'n_passed': s.n_passed,
                'pass_rate': s.pass_rate,
                'reliable': s.reliable,
                'median_days': s.median_days,
                'outcomes': dict(s.outcomes),
            } for s in self.stages],
        }


# ==============================================================================
# THE WALK
# ==============================================================================

def walk_stage(
    daily: np.ndarray,
    starting_balance: float,
    stage: StageSpec,
    rules: FirmRules,
) -> Dict[str, Any]:
    """
    Run one stage day by day over a single path of daily P&L.

    Returns a dict with 'outcome', 'days', 'equity', 'realised' (the P&L
    actually experienced before stopping).

    Stops at the first of: rule breach, target reached with the minimum days
    satisfied, or running out of days. The early stop is the point -- a trader
    who reaches the target does not keep trading, so the remainder of the path
    never happens and must not be evaluated.
    """
    daily_limit = rules.max_daily_loss_pct * starting_balance
    dd_floor = rules.drawdown_floor(starting_balance)
    target_equity = starting_balance * (1.0 + stage.profit_target_pct)

    equity = starting_balance
    traded_days = 0
    realised: List[float] = []

    horizon = len(daily)
    if stage.max_days is not None:
        horizon = min(horizon, stage.max_days)

    for i in range(horizon):
        pnl = float(daily[i])
        equity += pnl
        realised.append(pnl)
        if pnl != 0.0:
            traded_days += 1

        # Rule order matters: a day that both breaches the daily limit and
        # the floor is attributed to the daily limit, because that is the one
        # that triggers first intraday.
        if pnl <= -daily_limit:
            return {'outcome': FAIL_DAILY_LOSS, 'days': i + 1,
                    'equity': equity, 'realised': realised}
        if equity <= dd_floor:
            return {'outcome': FAIL_DRAWDOWN, 'days': i + 1,
                    'equity': equity, 'realised': realised}

        if (_reached(equity, target_equity)
                and traded_days >= stage.min_trading_days):
            # Consistency is evaluated on the days actually traded, which is
            # why it belongs here and not at the end of a fixed window.
            cons = consistency_rule.check_consistency(
                realised, rules.consistency_max_day_pct)
            if cons.is_fail:
                return {'outcome': FAIL_CONSISTENCY, 'days': i + 1,
                        'equity': equity, 'realised': realised}
            return {'outcome': PASSED, 'days': i + 1,
                    'equity': equity, 'realised': realised}

    if stage.max_days is not None and horizon >= stage.max_days:
        return {'outcome': FAIL_TIME_LIMIT, 'days': horizon,
                'equity': equity, 'realised': realised}
    return {'outcome': FAIL_NOT_REACHED, 'days': horizon,
            'equity': equity, 'realised': realised}


def simulate_challenge(
    sims: np.ndarray,
    account_size: float = 100_000.0,
    rules: Optional[FirmRules] = None,
    stages: Optional[Sequence[StageSpec]] = None,
    random_seed: int = 42,
) -> ChallengeResult:
    """
    Run the full multi-stage evaluation over bootstrap paths.

    Args:
        sims: (n_paths, n_days) combined daily P&L, e.g. from
              portfolio_merge.joint_block_bootstrap.
        stages: defaults to every phase in rules.profit_targets, ordered with
                'challenge' first.

    A path that clears stage one is re-drawn for stage two rather than
    continuing the same rows. Reusing the tail would make stage two
    conditional on the same lucky draw that won stage one, inflating the
    joint estimate.
    """
    if rules is None:
        rules = ftmo()
    if sims.ndim != 2 or sims.size == 0:
        raise ValueError("sims must be a non-empty 2-D array of daily P&L.")

    if stages is None:
        order = sorted(rules.profit_targets,
                       key=lambda p: (p != 'challenge', p))
        stages = [StageSpec.from_rules(rules, p) for p in order]
    if not stages:
        raise ValueError("No stages to simulate.")

    rng = np.random.RandomState(random_seed)
    n_paths = sims.shape[0]
    days_so_far: Dict[int, int] = {i: 0 for i in range(n_paths)}

    result = ChallengeResult(
        n_simulations=n_paths,
        rules_firm=rules.firm_name,
        unchecked_rules=[u.capability.value for u in rules.unsupported()],
    )
    stats = [StageStats(name=s.name) for s in stages]
    result.stages = stats

    alive = np.arange(n_paths)

    for si, stage in enumerate(stages):
        st = stats[si]
        st.n_entered = int(alive.size)
        if st.n_entered == 0:
            break

        survivors = []
        for path_idx in alive:
            # Independent redraw per stage; see docstring.
            row = sims[path_idx] if si == 0 else sims[rng.randint(0, n_paths)]
            out = walk_stage(row, account_size, stage, rules)
            st.outcomes[out['outcome']] = st.outcomes.get(out['outcome'], 0) + 1
            if out['outcome'] == PASSED:
                st.n_passed += 1
                st.days_used.append(out['days'])
                days_so_far[int(path_idx)] += int(out['days'])
                survivors.append(path_idx)

        alive = np.array(survivors, dtype=int)

        if not st.reliable and si > 0:
            result.warnings.append(
                f"Only {st.n_entered} path(s) reached '{stage.name}'; its "
                f"pass rate is estimated from too small a sample to trust.")

    result.n_funded = int(alive.size)
    result.funded_path_days = [days_so_far[int(i)] for i in alive]

    if result.n_funded == 0:
        result.warnings.append(
            "No simulated path reached a funded account. P(funded) is 0 for "
            "this configuration, so expected attempts is undefined rather "
            "than merely large.")

    return result


# ==============================================================================
# SINGLE-STRATEGY SIMULATION WITH EARLY STOPPING
# ==============================================================================
# pass_rate_simulator.simulate_pass_rate builds a synthetic window and calls
# checker.validate() on the whole thing. validate() is a fixed-window
# evaluator: it scans every day for breaches and compares FINAL equity to the
# target. That is the same assumption bootstrap_summary had, and it fails paths
# that reached the target and then gave it back -- days that, in reality, never
# happened because the trader had stopped.
#
# The fix keeps the real checker rather than reimplementing it. Fees, spreads,
# the Prague daily anchor and the intrabar equity curve are all things
# validate() gets right and this module has no business duplicating. Instead:
#
#   1. validate the full window
#   2. if it passed, done
#   3. if it failed, ask whether equity ever CLOSED above the target with the
#      minimum trading days already satisfied
#   4. if it did, truncate the trades at that day and validate again
#
# Only failed-but-touched paths get a second validate, so the cost is small.
#
# WHY end_equity AND NOT max_equity
# ---------------------------------
# daily_stats carries both. max_equity would treat a fleeting intraday spike
# through the target as a win, which is right for a human watching a screen and
# wrong for an algo. This platform builds algos, so the realistic stop is "at
# the daily close, if equity is above target, switch it off". end_equity is
# therefore the conservative and appropriate choice; a screen-watching
# discretionary trader would score somewhat better than this reports.


def find_early_stop_date(
    daily_stats,
    initial_balance: float,
    target_pct: float,
    min_trading_days: int,
):
    """
    First Prague date whose CLOSING equity satisfies the profit target with
    the minimum trading days already met, or None if that never happens.
    """
    if daily_stats is None or len(daily_stats) == 0:
        return None
    if 'end_equity' not in daily_stats.columns:
        return None

    target_equity = initial_balance * (1.0 + target_pct)
    for i, (_, row) in enumerate(daily_stats.iterrows()):
        days_elapsed = i + 1
        if days_elapsed < min_trading_days:
            continue
        if _reached(float(row['end_equity']), target_equity):
            return row['date']
    return None


@dataclass
class EarlyStopResult:
    """
    Single-strategy pass rate, with the fixed-window figure kept for contrast.

    `n_rescued` is the interesting number: paths the fixed-window evaluator
    failed that actually would have been stopped out as winners. It is a direct
    measure of how much that assumption was costing.
    """
    pass_rate: float = 0.0
    pass_rate_fixed_window: float = 0.0
    n_simulations: int = 0
    n_evaluated: int = 0
    n_rescued: int = 0
    account_size: float = 0.0
    phase: str = 'challenge'
    window_days: int = 0
    fail_reasons: Dict[str, int] = field(default_factory=dict)
    unchecked_rules: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def rescue_share(self) -> Optional[float]:
        """Rescued paths as a share of those evaluated. None if none ran."""
        if not self.n_evaluated:
            return None
        return self.n_rescued / self.n_evaluated

    def summary(self) -> str:
        if self.error:
            return f"Simulation unavailable: {self.error}"
        L = ['', '=' * 66, '  SINGLE-STRATEGY PASS RATE (early stopping)', '=' * 66]
        L.append(f"  Simulations evaluated : {self.n_evaluated}/{self.n_simulations}")
        L.append(f"  Pass rate             : {self.pass_rate * 100:.2f}%")
        L.append(f"  Fixed-window figure   : "
                 f"{self.pass_rate_fixed_window * 100:.2f}%  (understates)")
        L.append(f"  Rescued by early stop : {self.n_rescued}"
                 + (f"  ({self.rescue_share * 100:.1f}% of paths)"
                    if self.rescue_share is not None else ''))
        if self.fail_reasons:
            L.append('')
            for k, v in sorted(self.fail_reasons.items(), key=lambda kv: -kv[1]):
                L.append(f"    {k:<20} {v}")
        if self.unchecked_rules:
            L.append('')
            L.append(f"  [PARTIAL] not checked: {', '.join(self.unchecked_rules)}")
        L.append('=' * 66)
        return '\n'.join(L)


def simulate_pass_rate_early_stop(
    checker,
    trades_df,
    account_size: float = 100_000.0,
    phase: str = 'challenge',
    n_simulations: int = 1000,
    window_days: int = 30,
    mode: str = 'block',
    mean_block: float = 5.0,
    random_seed: int = 42,
    rules: Optional[FirmRules] = None,
) -> EarlyStopResult:
    """
    Bootstrap P(pass) for one strategy, honouring early stopping.

    `checker` is injected exactly as pass_rate_simulator does it, so this
    module never imports ftmo_compliance and no cycle forms.
    """
    import numpy as _np

    if rules is None:
        rules = ftmo()

    out = EarlyStopResult(
        n_simulations=n_simulations, account_size=account_size, phase=phase,
        window_days=window_days,
        unchecked_rules=[u.capability.value for u in rules.unsupported()],
    )

    # Argument checks first: they do not depend on the optional import, and
    # reporting a missing module when the real problem is three trades sends
    # the caller looking in the wrong place.
    if trades_df is None or len(trades_df) < 4:
        out.error = ("Insufficient trades (need at least 4 to satisfy the "
                     "minimum trading days rule)")
        return out

    if phase not in rules.profit_targets:
        out.error = f"{rules.firm_name} has no phase {phase!r}"
        return out
    target_pct = rules.profit_targets[phase]

    if checker is None:
        out.error = "No compliance checker supplied."
        return out

    try:
        import pass_rate_simulator as _prs
    except Exception as e:
        out.error = f"pass_rate_simulator unavailable: {e}"
        return out

    rng = _np.random.RandomState(random_seed)
    passes = 0
    fixed_passes = 0
    evaluated = 0
    rescued = 0
    fails: Dict[str, int] = {}

    for _ in range(n_simulations):
        sim = _prs.build_synthetic_window(
            trades_df, window_days, rng, mode=mode, mean_block=mean_block)
        if sim is None or sim.empty:
            continue

        try:
            r = checker.validate(sim, account_size=account_size, phase=phase,
                                 include_daily_equity=True)
        except Exception:
            continue

        evaluated += 1

        if r.passed:
            fixed_passes += 1
            if _consistency_ok(r.daily_equity, account_size, rules):
                passes += 1
            else:
                fails[FAIL_CONSISTENCY] = fails.get(FAIL_CONSISTENCY, 0) + 1
            continue

        # Did it ever close above the target before things went wrong?
        stop_date = find_early_stop_date(
            r.daily_equity, account_size, target_pct, rules.min_trading_days)
        if stop_date is None:
            _record_failure(fails, r)
            continue

        # Both sides mapped to firm-local DATES. Comparing a Prague date
        # object directly against a datetime64 column raises
        # 'Invalid comparison between dtype=datetime64[us] and date'.
        local = firm_local_dates(sim['exit_date'], rules.reset_timezone)
        truncated = sim[local <= stop_date]
        if truncated.empty:
            _record_failure(fails, r)
            continue

        try:
            r2 = checker.validate(truncated, account_size=account_size,
                                  phase=phase, include_daily_equity=True)
        except Exception:
            _record_failure(fails, r)
            continue

        if not r2.passed:
            _record_failure(fails, r2)
        elif _consistency_ok(r2.daily_equity, account_size, rules):
            passes += 1
            rescued += 1
        else:
            # Early stopping concentrates profit into fewer days, so a
            # rescued path is exactly the shape most likely to breach.
            fails[FAIL_CONSISTENCY] = fails.get(FAIL_CONSISTENCY, 0) + 1

    n = evaluated or 1
    out.n_evaluated = evaluated
    out.pass_rate = passes / n
    out.pass_rate_fixed_window = fixed_passes / n
    out.n_rescued = rescued
    out.fail_reasons = fails

    if evaluated == 0:
        out.error = ("No simulation produced a usable window; the pass rate "
                     "is not an estimate of anything.")
    return out


def daily_pnl_from_stats(daily_stats, initial_balance: float):
    """
    Per-day P&L implied by a daily_stats frame.

    First day is measured from the opening balance; every later day from
    the previous close. Returns an empty list when the frame cannot supply
    it, so callers can tell 'no data' from 'no profit'.
    """
    if daily_stats is None or len(daily_stats) == 0:
        return []
    if 'end_equity' not in daily_stats.columns:
        return []
    closes = [float(v) for v in daily_stats['end_equity']]
    prev = float(initial_balance)
    out = []
    for c in closes:
        out.append(c - prev)
        prev = c
    return out


def _consistency_ok(daily_stats, initial_balance: float,
                    rules: FirmRules) -> bool:
    """
    Apply the consistency rule to a path the checker has already passed.

    WHY THIS IS NEEDED SEPARATELY: checker.validate() implements the daily
    loss, drawdown, minimum-days and profit-target rules. It knows nothing
    about consistency. Without this step a caller could pass a FirmRules
    carrying consistency_max_day_pct, be told unchecked_rules is empty,
    and receive a pass rate in which the rule was silently ignored --
    a claim of full coverage over a check that never ran.

    An unevaluable result (no net profit) is NOT a failure here: the path
    only reaches this function by having met the profit target, so an
    unevaluable verdict means the day series could not be reconstructed,
    and failing the path for that would invent a breach.
    """
    if rules.consistency_max_day_pct is None:
        return True
    pnl = daily_pnl_from_stats(daily_stats, initial_balance)
    if not pnl:
        return True
    return not consistency_rule.check_consistency(
        pnl, rules.consistency_max_day_pct).is_fail


def _record_failure(fails: Dict[str, int], result) -> None:
    """Attribute a failure to the first rule it broke."""
    if not getattr(result, 'daily_loss_ok', True):
        key = FAIL_DAILY_LOSS
    elif not getattr(result, 'total_drawdown_ok', True):
        key = FAIL_DRAWDOWN
    elif not getattr(result, 'min_days_ok', True):
        key = 'min_trading_days'
    elif not getattr(result, 'profit_target_ok', True):
        key = FAIL_NOT_REACHED
    else:
        key = 'unattributed'
    fails[key] = fails.get(key, 0) + 1