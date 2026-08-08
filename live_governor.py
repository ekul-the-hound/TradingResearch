# ==============================================================================
# live_governor.py
# ==============================================================================
# Phase 6. The runtime enforcer.
#
# WHAT WAS MISSING
# ----------------
# ftmo_compliance.py is a BACKTEST VALIDATOR. It answers "did this historical
# trade sequence comply", after the fact, with the whole series in hand. That
# is the wrong shape for live trading, where the only useful question is "may
# I place this order right now", asked before the order exists.
#
# kill_switch.py is closer but generic. Its check() takes daily_loss_pct as a
# float the CALLER computes, so it never knows when the trading day reset, what
# the balance was at that reset, or whether floating P&L counts. Hand it a
# daily loss measured from the wrong anchor and it fires at the wrong moment or
# not at all. It also carries its own ftmo_daily_limit_pct = 5.0 alongside
# firm_rules.FirmRules, which is a second set of numbers to keep in sync.
#
# THE ASYMMETRY THAT SHAPES THIS MODULE
# -------------------------------------
# In a backtest a breach is a data point. In a live challenge a breach is
# terminal: the account is failed, the fee is gone, and no subsequent good
# trading undoes it. Stopping early costs you some upside. Stopping late costs
# you everything.
#
# So this governor does not wait for the limit. It halts at a FRACTION of it,
# and every uncertain case resolves toward halting:
#
#   - account state older than max_state_age_seconds  -> HALT
#   - no anchor recorded for today                    -> HALT
#   - initial balance unknown or non-positive         -> FLATTEN
#   - an unexpected exception inside the check        -> FLATTEN
#
# A governor that fails open is worse than no governor, because it is trusted.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import consistency_rule
from firm_rules import FirmRules, ftmo

try:
    import pytz
except ImportError:  # pragma: no cover
    pytz = None


class Decision(str, Enum):
    """
    Ordered by severity. Higher ordinal wins when several rules fire.

    ALLOW      trade normally
    REDUCE     open positions may stay, new ones must be smaller
    HALT_NEW   no new positions; existing ones may run
    FLATTEN    close everything now
    """
    ALLOW = 'allow'
    REDUCE = 'reduce'
    HALT_NEW = 'halt_new'
    FLATTEN = 'flatten'


_SEVERITY = {
    Decision.ALLOW: 0,
    Decision.REDUCE: 1,
    Decision.HALT_NEW: 2,
    Decision.FLATTEN: 3,
}

# Reasons, so callers can branch on a stable string rather than prose.
R_OK = 'ok'
R_STALE_STATE = 'stale_account_state'
R_NO_ANCHOR = 'no_daily_anchor'
R_BAD_BALANCE = 'invalid_initial_balance'
R_DAILY_APPROACH = 'approaching_daily_loss'
R_DAILY_BREACH = 'daily_loss_breached'
R_DD_APPROACH = 'approaching_total_drawdown'
R_DD_BREACH = 'total_drawdown_breached'
R_CONSISTENCY_RISK = 'consistency_at_risk'
R_INTERNAL_ERROR = 'internal_error'


@dataclass
class AccountState:
    """
    One snapshot from the broker. Filling this is the adapter's whole job.

    `equity` must include floating P&L when the firm's daily rule does --
    which is the usual case. `balance` is closed-only. Supplying balance in
    both fields silently disables intraday protection, so they are separate
    fields with no default.
    """
    timestamp: datetime
    balance: float
    equity: float
    initial_balance: float
    open_positions: int = 0
    symbol_exposure: Dict[str, float] = field(default_factory=dict)


@dataclass
class GovernorConfig:
    rules: FirmRules = field(default_factory=ftmo)

    # Act at a fraction of each limit, never at the limit itself. 0.80 means a
    # 5% daily limit halts at 4%. The remaining 1% is room for slippage on the
    # closing fills, a gapping market, and the delay between this decision and
    # the broker acting on it.
    halt_at_fraction: float = 0.80
    reduce_at_fraction: float = 0.60

    # Beyond this age a snapshot is not evidence about the present.
    max_state_age_seconds: float = 30.0

    # Warn when today's profit is on track to breach the consistency cap.
    consistency_warn_fraction: float = 0.90

    def __post_init__(self):
        if not 0 < self.halt_at_fraction <= 1:
            raise ValueError("halt_at_fraction must be in (0, 1].")
        if not 0 < self.reduce_at_fraction <= 1:
            raise ValueError("reduce_at_fraction must be in (0, 1].")
        if self.reduce_at_fraction > self.halt_at_fraction:
            raise ValueError(
                f"reduce_at_fraction ({self.reduce_at_fraction}) is above "
                f"halt_at_fraction ({self.halt_at_fraction}); the governor "
                f"would halt before it ever reduced.")
        if self.max_state_age_seconds <= 0:
            raise ValueError("max_state_age_seconds must be positive.")


@dataclass
class Verdict:
    decision: Decision
    reason: str
    detail: str = ''
    trading_date: Optional[date] = None
    anchor_equity: Optional[float] = None
    daily_loss: Optional[float] = None
    daily_loss_pct: Optional[float] = None
    daily_limit: Optional[float] = None
    drawdown_floor: Optional[float] = None
    headroom: Optional[float] = None
    unchecked_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def may_open(self) -> bool:
        return self.decision in (Decision.ALLOW, Decision.REDUCE)

    @property
    def must_flatten(self) -> bool:
        return self.decision is Decision.FLATTEN

    def __str__(self) -> str:
        s = f"[{self.decision.value.upper()}] {self.reason}"
        if self.detail:
            s += f" -- {self.detail}"
        return s


class LiveGovernor:
    """
    Stateful across observations: it remembers each trading day's opening
    equity, which is the thing a stateless checker cannot know.

    Not thread-safe. One governor per account, called from the trading loop.
    """

    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig()
        self.anchors: Dict[date, float] = {}
        self.daily_close: Dict[date, float] = {}
        self._last_date: Optional[date] = None
        self._halted_dates: set = set()

    # ------------------------------------------------------------------
    # CALENDAR
    # ------------------------------------------------------------------
    def trading_date(self, ts: datetime) -> date:
        """
        The firm's trading date for a timestamp.

        Naive timestamps are treated as UTC, matching the rest of the
        codebase. Getting this wrong shifts every daily limit by hours.
        """
        tz_name = self.config.rules.reset_timezone
        if pytz is None:                                  # pragma: no cover
            return ts.date()
        tz = pytz.timezone(tz_name)
        aware = pytz.UTC.localize(ts) if ts.tzinfo is None else ts
        return aware.astimezone(tz).date()

    def seed_anchor(self, trading_day: date, equity: float) -> None:
        """
        Supply a day's opening equity explicitly.

        Needed when the governor starts mid-session: it has no record of this
        morning's balance, and inferring it from the current equity would
        silently forgive whatever has already been lost today. Persist this
        alongside the rest of your live state so a restart does not reset the
        day's loss budget to zero.
        """
        self.anchors[trading_day] = float(equity)

    def save_state(self) -> Dict[str, Any]:
        return {
            'anchors': {d.isoformat(): v for d, v in self.anchors.items()},
            'daily_close': {d.isoformat(): v
                            for d, v in self.daily_close.items()},
            'halted_dates': sorted(d.isoformat() for d in self._halted_dates),
        }

    def load_state(self, blob: Dict[str, Any]) -> None:
        self.anchors = {date.fromisoformat(k): float(v)
                        for k, v in (blob.get('anchors') or {}).items()}
        self.daily_close = {date.fromisoformat(k): float(v)
                            for k, v in (blob.get('daily_close') or {}).items()}
        self._halted_dates = {date.fromisoformat(d)
                              for d in (blob.get('halted_dates') or [])}

    # ------------------------------------------------------------------
    # THE CHECK
    # ------------------------------------------------------------------
    def observe(self, state: AccountState,
                now: Optional[datetime] = None) -> Verdict:
        """
        Judge one snapshot. Call before every order and on every heartbeat.

        Wrapped so that an unexpected failure inside the governor produces
        FLATTEN rather than an exception the trading loop might swallow into
        a permissive default.
        """
        try:
            return self._observe(state, now)
        except Exception as e:                            # pragma: no cover
            return Verdict(
                decision=Decision.FLATTEN,
                reason=R_INTERNAL_ERROR,
                detail=(f"{type(e).__name__}: {e}. Flattening because the "
                        f"governor cannot vouch for the account."),
            )

    def _observe(self, state: AccountState,
                 now: Optional[datetime] = None) -> Verdict:
        cfg = self.config
        rules = cfg.rules
        unchecked = [u.capability.value for u in rules.unsupported()]

        if state.initial_balance is None or state.initial_balance <= 0:
            return Verdict(
                decision=Decision.FLATTEN, reason=R_BAD_BALANCE,
                detail=(f"initial_balance is {state.initial_balance!r}. Every "
                        f"limit is a fraction of it, so none can be computed."),
                unchecked_rules=unchecked)

        # -- staleness -------------------------------------------------
        now = now or datetime.utcnow()
        age = (now - state.timestamp).total_seconds()
        if age > cfg.max_state_age_seconds:
            return Verdict(
                decision=Decision.HALT_NEW, reason=R_STALE_STATE,
                detail=(f"Snapshot is {age:.0f}s old, limit is "
                        f"{cfg.max_state_age_seconds:.0f}s. A stale equity "
                        f"reading is not evidence about the present."),
                unchecked_rules=unchecked)

        # -- day rollover ----------------------------------------------
        today = self.trading_date(state.timestamp)
        if today != self._last_date:
            if self._last_date is not None:
                self.daily_close[self._last_date] = state.equity
            self._last_date = today

        if today not in self.anchors:
            if not self.daily_close and not self.anchors:
                # First ever observation: this IS the day's opening equity.
                self.anchors[today] = state.equity
            elif self.daily_close:
                # Carry the most recent close forward as this day's opening
                # equity. Correct for an overnight gap-free account and the
                # best available answer when the governor was running
                # yesterday but not at this morning's reset.
                self.anchors[today] = self.daily_close[max(self.daily_close)]
            else:
                return Verdict(
                    decision=Decision.HALT_NEW, reason=R_NO_ANCHOR,
                    trading_date=today,
                    detail=(f"No opening equity recorded for {today}. "
                            f"Using the current equity would forgive any "
                            f"loss already taken today. Call "
                            f"seed_anchor({today!r}, <opening equity>)."),
                    unchecked_rules=unchecked)

        anchor = self.anchors[today]

        # -- the numbers ------------------------------------------------
        equity = state.equity if rules.includes_floating_pnl else state.balance
        daily_loss = anchor - equity
        daily_limit = rules.daily_loss_limit(anchor)
        dd_floor = rules.drawdown_floor(state.initial_balance)

        halt_at = daily_limit * cfg.halt_at_fraction
        reduce_at = daily_limit * cfg.reduce_at_fraction

        v = Verdict(
            decision=Decision.ALLOW, reason=R_OK, trading_date=today,
            anchor_equity=anchor, daily_loss=daily_loss,
            daily_loss_pct=(daily_loss / anchor * 100.0) if anchor else None,
            daily_limit=daily_limit, drawdown_floor=dd_floor,
            headroom=daily_limit - daily_loss,
            unchecked_rules=unchecked)

        candidates: List[Verdict] = []

        # -- total drawdown (terminal; check first) ---------------------
        if equity <= dd_floor:
            candidates.append(_worse(v, Decision.FLATTEN, R_DD_BREACH,
                f"Equity {equity:,.2f} is at or below the drawdown floor "
                f"{dd_floor:,.2f}. The account is already failed; closing "
                f"positions prevents further loss but does not undo it."))
        else:
            dd_used = state.initial_balance - equity
            dd_budget = state.initial_balance - dd_floor
            if dd_budget > 0 and dd_used >= dd_budget * cfg.halt_at_fraction:
                candidates.append(_worse(v, Decision.FLATTEN, R_DD_APPROACH,
                    f"Drawdown {dd_used:,.2f} of {dd_budget:,.2f} budget "
                    f"({dd_used / dd_budget * 100:.0f}%). Flattening while "
                    f"there is still room to close at a bad price."))
            elif dd_budget > 0 and dd_used >= dd_budget * cfg.reduce_at_fraction:
                candidates.append(_worse(v, Decision.REDUCE, R_DD_APPROACH,
                    f"Drawdown at {dd_used / dd_budget * 100:.0f}% of budget."))

        # -- daily loss -------------------------------------------------
        if daily_loss >= daily_limit:
            candidates.append(_worse(v, Decision.FLATTEN, R_DAILY_BREACH,
                f"Daily loss {daily_loss:,.2f} has reached the limit "
                f"{daily_limit:,.2f}."))
        elif daily_loss >= halt_at:
            candidates.append(_worse(v, Decision.HALT_NEW, R_DAILY_APPROACH,
                f"Daily loss {daily_loss:,.2f} is {daily_loss / daily_limit * 100:.0f}% "
                f"of the {daily_limit:,.2f} limit. "
                f"{daily_limit - daily_loss:,.2f} of headroom left."))
            self._halted_dates.add(today)
        elif daily_loss >= reduce_at:
            candidates.append(_worse(v, Decision.REDUCE, R_DAILY_APPROACH,
                f"Daily loss at {daily_loss / daily_limit * 100:.0f}% of the "
                f"daily limit."))

        # -- once halted for the day, stay halted -----------------------
        # Equity recovering after a halt does not mean the day is safe again:
        # the recovery is usually an open position moving back, and re-arming
        # on it is how an account round-trips through the limit.
        if today in self._halted_dates and daily_loss < halt_at:
            candidates.append(_worse(v, Decision.HALT_NEW, R_DAILY_APPROACH,
                f"Already halted today at {halt_at:,.2f}. Not re-arming until "
                f"the next daily reset, even though loss has recovered to "
                f"{daily_loss:,.2f}."))

        # -- consistency ------------------------------------------------
        cons = self._consistency_warning(today, anchor, equity)
        if cons:
            v.warnings.append(cons)

        if not candidates:
            return v
        return max(candidates, key=lambda c: _SEVERITY[c.decision])

    # ------------------------------------------------------------------
    def _consistency_warning(self, today: date, anchor: float,
                             equity: float) -> Optional[str]:
        """
        Warn when today's profit is approaching the consistency cap.

        Advisory only, never a halt. Making too much money in one day is not
        a rule breach at the moment it happens -- the rule is evaluated on the
        finished account -- so the correct response is to tell the operator,
        not to close a winning position.
        """
        cap = self.config.rules.consistency_max_day_pct
        if cap is None:
            return None

        today_profit = equity - anchor
        if today_profit <= 0:
            return None

        prior = [c - self.anchors.get(d, c)
                 for d, c in self.daily_close.items() if d != today]
        total = sum(p for p in prior) + today_profit
        if total <= 0:
            return None

        share = today_profit / total
        if share >= cap * self.config.consistency_warn_fraction:
            return (f"Today is {share * 100:.0f}% of total profit; the "
                    f"consistency cap is {cap * 100:.0f}%. Further gains today "
                    f"make this worse, not better -- the cap is a ratio, and "
                    f"only trading on OTHER days lowers it.")
        return None

    # ------------------------------------------------------------------
    def consistency_now(self) -> "consistency_rule.ConsistencyResult":
        """Current consistency verdict over completed days."""
        daily = [c - self.anchors.get(d, c)
                 for d, c in sorted(self.daily_close.items())]
        return consistency_rule.check_consistency(
            daily, self.config.rules.consistency_max_day_pct,
            dates=sorted(self.daily_close))


def _worse(base: Verdict, decision: Decision, reason: str,
           detail: str) -> Verdict:
    """Copy a verdict with a more severe decision attached."""
    return Verdict(
        decision=decision, reason=reason, detail=detail,
        trading_date=base.trading_date, anchor_equity=base.anchor_equity,
        daily_loss=base.daily_loss, daily_loss_pct=base.daily_loss_pct,
        daily_limit=base.daily_limit, drawdown_floor=base.drawdown_floor,
        headroom=base.headroom, unchecked_rules=list(base.unchecked_rules),
        warnings=list(base.warnings))
