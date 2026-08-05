# ==============================================================================
# consistency_rule.py
# ==============================================================================
# The rule most prop firms use to stop people gambling their way to the target:
# no single day's profit may exceed some share of total profit.
#
# WHY THIS MATTERS HERE SPECIFICALLY
# ----------------------------------
# A strategy tuned to reach +10% quickly tends to make its money in a handful
# of large days. That is precisely the shape the rule forbids. Until this
# module existed, every pass-rate number the platform produced was optimistic
# for exactly the System A strategies the 90-day plan depends on -- a strategy
# could show a clean PASS while being structurally disqualified.
#
# THE SEMANTIC IMPLEMENTED HERE
# -----------------------------
#     largest single-day PROFIT / total NET profit  <=  threshold
#
# Profitable days only in the numerator; net profit (wins minus losses) in the
# denominator. This is the most common formulation, and it is the one this
# module claims. It is NOT the only one in use:
#
#   - some firms measure against the PROFIT TARGET rather than achieved profit
#   - some apply the rule only at payout, not during the evaluation
#   - some use gross profit in the denominator, ignoring losing days
#   - some cap the largest TRADE rather than the largest DAY
#
# Those are different computations, not different thresholds, so they are not
# covered by setting a different number. VARIANTS_NOT_MODELLED below names them
# so the gap is visible at the point of use rather than discovered later.
#
# THE UNDEFINED CASE IS THE IMPORTANT ONE
# ---------------------------------------
# When total profit is zero or negative, "what share came from the best day" has
# no meaningful answer. The tempting shortcuts are both wrong:
#
#   passed=True   -- claims compliance with a rule that was never evaluated
#   passed=False  -- fails an account for a rule that does not apply to it
#
# So the result carries evaluated=False and a reason, and the caller must decide.
# Same principle as the rest of the codebase: absence of an answer is
# representable, propagating, and loud.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


VARIANTS_NOT_MODELLED: List[str] = [
    "measured against the profit target instead of achieved profit",
    "applied only at payout rather than during evaluation",
    "gross profit in the denominator (losing days excluded)",
    "largest single TRADE capped instead of largest DAY",
]

NOT_EVALUATED_NO_PROFIT = 'no_net_profit'
NOT_EVALUATED_NO_DAYS = 'no_trading_days'
NOT_EVALUATED_NO_THRESHOLD = 'no_threshold_configured'


@dataclass
class ConsistencyResult:
    """
    Outcome of the consistency check.

    `passed` is Optional on purpose. None means the rule could not be
    evaluated, which is a third state distinct from pass and fail. Callers
    that flatten it to a boolean are reintroducing the bug this guards.
    """
    evaluated: bool
    passed: Optional[bool] = None
    threshold: Optional[float] = None
    best_day_profit: float = 0.0
    best_day_date: Optional[str] = None
    total_profit: float = 0.0
    best_day_share: Optional[float] = None
    n_profitable_days: int = 0
    n_days: int = 0
    reason: str = ''

    @property
    def is_pass(self) -> bool:
        """
        Strict reading: only an evaluated pass counts as a pass.

        Deliberately does NOT treat 'could not evaluate' as success.
        """
        return self.evaluated and self.passed is True

    @property
    def is_fail(self) -> bool:
        return self.evaluated and self.passed is False

    def summary(self) -> str:
        if not self.evaluated:
            return f"Consistency rule NOT evaluated: {self.reason}"
        pct = (self.best_day_share or 0.0) * 100.0
        cap = (self.threshold or 0.0) * 100.0
        verdict = 'PASS' if self.passed else 'FAIL'
        return (f"Consistency {verdict}: best day "
                f"{self.best_day_profit:,.2f} is {pct:.1f}% of total profit "
                f"{self.total_profit:,.2f} (cap {cap:.1f}%)"
                + (f" on {self.best_day_date}" if self.best_day_date else ''))


def check_consistency(
    daily_pnl: Sequence[float],
    threshold: Optional[float],
    dates: Optional[Sequence[Any]] = None,
) -> ConsistencyResult:
    """
    Evaluate the consistency rule over a series of daily P&L values.

    Args:
        daily_pnl: net P&L per trading day, in account currency.
        threshold: max share of total profit one day may contribute, as a
                   fraction (0.30 for 30%). None means the firm has no such
                   rule, which yields evaluated=False rather than a pass.
        dates:     optional labels aligned to daily_pnl, for reporting.
    """
    values = np.asarray(list(daily_pnl), dtype=float)

    if threshold is None:
        return ConsistencyResult(
            evaluated=False, reason=NOT_EVALUATED_NO_THRESHOLD,
            n_days=int(values.size))

    if values.size == 0:
        return ConsistencyResult(
            evaluated=False, threshold=threshold,
            reason=NOT_EVALUATED_NO_DAYS)

    total = float(values.sum())
    profitable = values[values > 0]

    # The undefined case. A flat or losing account has no profit for any day
    # to be a share OF. Returning a pass here would be a fabricated verdict.
    if total <= 0 or profitable.size == 0:
        return ConsistencyResult(
            evaluated=False, threshold=threshold, total_profit=total,
            n_days=int(values.size), n_profitable_days=int(profitable.size),
            reason=NOT_EVALUATED_NO_PROFIT)

    best_idx = int(np.argmax(values))
    best = float(values[best_idx])
    share = best / total

    label = None
    if dates is not None:
        seq = list(dates)
        if 0 <= best_idx < len(seq):
            label = str(seq[best_idx])

    return ConsistencyResult(
        evaluated=True,
        passed=bool(share <= threshold),
        threshold=threshold,
        best_day_profit=best,
        best_day_date=label,
        total_profit=total,
        best_day_share=share,
        n_profitable_days=int(profitable.size),
        n_days=int(values.size),
    )


def check_consistency_frame(daily_df, threshold: Optional[float]) -> ConsistencyResult:
    """
    Convenience wrapper for portfolio_merge's daily matrix.

    Sums across strategies first: the rule applies to the ACCOUNT's daily
    profit, not to any one strategy's contribution.
    """
    if daily_df is None or len(daily_df) == 0:
        return ConsistencyResult(
            evaluated=False, threshold=threshold, reason=NOT_EVALUATED_NO_DAYS)
    combined = daily_df.sum(axis=1)
    return check_consistency(
        [float(v) for v in combined.values],
        threshold,
        dates=[str(getattr(i, 'date', lambda: i)()) for i in combined.index],
    )


# ==============================================================================
# BOOTSTRAP SUPPORT
# ==============================================================================

def consistency_breach_mask(sims: np.ndarray,
                            threshold: Optional[float]) -> np.ndarray:
    """
    Per-simulation consistency verdicts over bootstrap paths.

    Returns a boolean array where True means the path BREACHED the rule --
    i.e. it reached profit but concentrated too much of it in one day.

    Paths that end flat or down are False, not because they complied but
    because they never got far enough for the rule to apply. Those paths fail
    on the profit target instead, so they are already counted as failures
    elsewhere; marking them as consistency breaches too would double-count.
    """
    if threshold is None:
        return np.zeros(sims.shape[0], dtype=bool)

    totals = sims.sum(axis=1)
    best = sims.max(axis=1)

    breached = np.zeros(sims.shape[0], dtype=bool)
    evaluable = totals > 0
    if not np.any(evaluable):
        return breached

    with np.errstate(divide='ignore', invalid='ignore'):
        shares = np.where(evaluable, best / np.where(evaluable, totals, 1.0), 0.0)
    breached[evaluable] = shares[evaluable] > threshold
    return breached


def consistency_stats(sims: np.ndarray,
                      threshold: Optional[float]) -> Dict[str, Any]:
    """Descriptive stats for the dashboard."""
    if threshold is None:
        return {
            'evaluated': False,
            'reason': NOT_EVALUATED_NO_THRESHOLD,
            'breach_rate': None,
            'evaluable_rate': None,
            'variants_not_modelled': list(VARIANTS_NOT_MODELLED),
        }

    totals = sims.sum(axis=1)
    evaluable = totals > 0
    breached = consistency_breach_mask(sims, threshold)

    n_eval = int(evaluable.sum())
    return {
        'evaluated': True,
        'threshold': threshold,
        'n_simulations': int(sims.shape[0]),
        'n_evaluable': n_eval,
        'evaluable_rate': float(evaluable.mean()),
        # Denominator is the evaluable subset: "of the paths that made money,
        # how many made it too unevenly". Dividing by all paths would dilute
        # the number with paths the rule never applied to.
        'breach_rate': float(breached[evaluable].mean()) if n_eval else None,
        'breach_rate_all_paths': float(breached.mean()),
        'variants_not_modelled': list(VARIANTS_NOT_MODELLED),
    }
