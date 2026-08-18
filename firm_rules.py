# ==============================================================================
# firm_rules.py
# ==============================================================================
# Phase 3 foundation. Replaces the module-level constants in ftmo_compliance.py
# (MAX_DAILY_LOSS_PCT, MAX_TOTAL_DRAWDOWN_PCT, MIN_TRADING_DAYS,
# PROFIT_TARGETS) with a value object the dashboard can edit.
#
# WHY THIS EXISTS, AND WHY IT IS NOT A "RULE ENGINE"
# -------------------------------------------------
# A config-driven prop-firm rule engine was attempted once and abandoned, for a
# good reason that has not gone away: firm rules are a moving target full of
# undocumented edge cases, and a rule engine manufactures false confidence. A
# strategy that "passes" your model of Firm X can still fail Firm X, because
# your model was subtly wrong -- and the green badge tells you nothing about
# which parts of Firm X you actually modelled.
#
# So this module draws a hard line that the abandoned design did not:
#
#   NUMBERS are configurable.      A threshold is a threshold. 5% or 4% or 6%
#                                  is the same comparison against a different
#                                  float. Editing it cannot be wrong.
#
#   SEMANTICS are capabilities.    Static vs trailing drawdown is not a number,
#                                  it is a different algorithm. Whether floating
#                                  P&L counts is a different equity curve. These
#                                  cannot be dropdowns unless code backs them.
#
# Every semantic is declared as a Capability with an explicit implementation
# status. A FirmRules profile that requires an unimplemented capability does not
# silently ignore it and it does not fail closed either -- it reports the gap
# through unsupported() so the caller can surface "PASS, 2 rules unchecked"
# instead of a bare "PASS".
#
# This is the same principle as the rest of the Phase 0/1 work: the absence of
# an answer is representable, propagating, and loud.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ==============================================================================
# CAPABILITIES -- the semantics half
# ==============================================================================

class Capability(str, Enum):
    """
    A rule *semantic*. Each value names a distinct computation, not a number.

    Adding a member here does NOT make it work. It has to be added to
    IMPLEMENTED as well, and that addition must be backed by real code in the
    compliance checker. The two-step is deliberate: it makes shipping an
    unimplemented rule a conscious act rather than an oversight.
    """

    # -- drawdown shape ----------------------------------------------------
    STATIC_DRAWDOWN = 'static_drawdown'
    TRAILING_DRAWDOWN_INTRADAY = 'trailing_drawdown_intraday'
    TRAILING_DRAWDOWN_EOD = 'trailing_drawdown_eod'

    # -- what counts toward the daily limit --------------------------------
    DAILY_LOSS_INCLUDES_FLOATING = 'daily_loss_includes_floating'
    DAILY_LOSS_CLOSED_ONLY = 'daily_loss_closed_only'

    # -- time --------------------------------------------------------------
    MIN_TRADING_DAYS = 'min_trading_days'
    MAX_CALENDAR_DAYS = 'max_calendar_days'

    # -- not yet backed by code -------------------------------------------
    CONSISTENCY_RULE = 'consistency_rule'
    WEEKEND_HOLDING_BAN = 'weekend_holding_ban'
    NEWS_TRADING_BAN = 'news_trading_ban'
    MAX_LOT_SIZE = 'max_lot_size'
    STOP_LOSS_MANDATORY = 'stop_loss_mandatory'


# The whitelist. A capability absent from this set is NOT checked by anything,
# no matter how a FirmRules profile is configured.
#
# Keep this honest. Moving a member up here without writing the checker is the
# single easiest way to reintroduce exactly the false confidence that got the
# original rule engine killed.
IMPLEMENTED: frozenset = frozenset({
    Capability.STATIC_DRAWDOWN,
    Capability.DAILY_LOSS_INCLUDES_FLOATING,
    Capability.DAILY_LOSS_CLOSED_ONLY,
    Capability.MIN_TRADING_DAYS,
    Capability.MAX_CALENDAR_DAYS,
    # Backed by consistency_rule.check_consistency. Only the
    # best-day-share-of-net-profit formulation; see
    # consistency_rule.VARIANTS_NOT_MODELLED for the ones that are not
    # reachable by changing the threshold.
    Capability.CONSISTENCY_RULE,
})


# Human-readable explanation of what each unimplemented capability would need.
# Surfaced in the dashboard next to the greyed-out control so the reason is
# visible at the point of confusion rather than buried in a docstring.
CAPABILITY_NOTES: Dict[Capability, str] = {
    Capability.TRAILING_DRAWDOWN_INTRADAY:
        "Needs a high-water-mark tracker evaluated on every equity point, not "
        "a fixed floor from initial balance.",
    Capability.TRAILING_DRAWDOWN_EOD:
        "Needs a high-water mark that ratchets only at the daily close, and "
        "usually stops ratcheting once the account is up by the profit target.",
    Capability.CONSISTENCY_RULE:
        "Implemented as best-day profit / total net profit. Other "
        "formulations (vs profit target, payout-only, gross profit, "
        "per-trade) are different computations, not different numbers.",
    Capability.WEEKEND_HOLDING_BAN:
        "Needs a session/holiday calendar to know when the weekend starts for "
        "each symbol.",
    Capability.NEWS_TRADING_BAN:
        "Needs a scheduled-news feed aligned to trade timestamps.",
    Capability.MAX_LOT_SIZE:
        "Needs per-symbol lot normalisation; 'size' in the trade ledger is not "
        "currently in lots for every asset class.",
    Capability.STOP_LOSS_MANDATORY:
        "Needs stop distance recorded on every trade; the ledger does not "
        "carry it today.",
}


@dataclass(frozen=True)
class UnsupportedRule:
    """
    A rule the profile asked for that the engine will not evaluate.

    This travels with the compliance result. A PASS carrying a non-empty list
    of these is not a PASS against the firm -- it is a PASS against the subset
    of the firm's rules that are modelled, and it must be displayed that way.
    """
    capability: Capability
    reason: str

    def __str__(self) -> str:
        return f"{self.capability.value}: {self.reason}"


# ==============================================================================
# FIRM RULES -- the numbers half
# ==============================================================================

@dataclass
class FirmRules:
    """
    A prop firm's evaluation rules.

    Everything on this object is either a number you may freely edit, or a
    capability flag whose truth is enforced against IMPLEMENTED.
    """

    # -- identity ----------------------------------------------------------
    firm_name: str = 'FTMO'
    profile_note: str = ''

    # -- numeric thresholds (freely editable) ------------------------------
    max_daily_loss_pct: float = 0.05        # fraction of anchor balance
    max_total_drawdown_pct: float = 0.10    # fraction of initial balance
    min_trading_days: int = 4
    max_calendar_days: Optional[int] = None  # None = no time limit
    profit_targets: Dict[str, float] = field(
        default_factory=lambda: {'challenge': 0.10, 'verification': 0.05}
    )
    account_sizes: List[int] = field(
        default_factory=lambda: [10_000, 25_000, 50_000, 100_000, 200_000]
    )
    reset_timezone: str = 'Europe/Prague'

    # Consistency: largest share of total profit any single day may contribute.
    # None = firm has no such rule. A number here is currently ASPIRATIONAL --
    # CONSISTENCY_RULE is not in IMPLEMENTED, so setting it produces an
    # UnsupportedRule rather than a check. That is intentional and visible.
    consistency_max_day_pct: Optional[float] = None

    # -- semantics (validated against IMPLEMENTED) -------------------------
    required_capabilities: List[Capability] = field(
        default_factory=lambda: [
            Capability.STATIC_DRAWDOWN,
            Capability.DAILY_LOSS_INCLUDES_FLOATING,
            Capability.MIN_TRADING_DAYS,
        ]
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    def __post_init__(self):
        # Coerce strings coming back from the dashboard's JSON round-trip.
        self.required_capabilities = [
            c if isinstance(c, Capability) else Capability(c)
            for c in self.required_capabilities
        ]
        self.validate_numbers()

    def validate_numbers(self) -> None:
        """
        Reject numerically impossible profiles at construction.

        These are the errors a form can produce by typo -- a 50% daily loss
        limit, a negative profit target -- and catching them here means the
        dashboard cannot hand the checker a profile that would produce a
        meaningless result.
        """
        if not 0 < self.max_daily_loss_pct < 1:
            raise ValueError(
                f"max_daily_loss_pct must be a fraction in (0, 1), "
                f"got {self.max_daily_loss_pct!r}. 5% is 0.05, not 5."
            )
        if not 0 < self.max_total_drawdown_pct < 1:
            raise ValueError(
                f"max_total_drawdown_pct must be a fraction in (0, 1), "
                f"got {self.max_total_drawdown_pct!r}."
            )
        if self.max_daily_loss_pct > self.max_total_drawdown_pct:
            raise ValueError(
                f"max_daily_loss_pct ({self.max_daily_loss_pct}) exceeds "
                f"max_total_drawdown_pct ({self.max_total_drawdown_pct}). "
                f"The daily limit would be unreachable -- total drawdown "
                f"always breaches first. Check the profile."
            )
        if self.min_trading_days < 0:
            raise ValueError("min_trading_days cannot be negative.")
        if self.max_calendar_days is not None and self.max_calendar_days <= 0:
            raise ValueError(
                "max_calendar_days must be positive, or None for no limit."
            )
        if not self.profit_targets:
            raise ValueError("profit_targets cannot be empty.")
        for phase, target in self.profit_targets.items():
            if target <= 0:
                raise ValueError(
                    f"profit_targets[{phase!r}] must be positive, got {target!r}."
                )
        if self.consistency_max_day_pct is not None:
            if not 0 < self.consistency_max_day_pct <= 1:
                raise ValueError(
                    f"consistency_max_day_pct must be a fraction in (0, 1], "
                    f"got {self.consistency_max_day_pct!r}. 30% is 0.30."
                )
        if not self.account_sizes:
            raise ValueError("account_sizes cannot be empty.")

        # Mutually exclusive semantics -- catching this is the whole point of
        # keeping capabilities explicit rather than inferring them.
        dd_modes = {
            Capability.STATIC_DRAWDOWN,
            Capability.TRAILING_DRAWDOWN_INTRADAY,
            Capability.TRAILING_DRAWDOWN_EOD,
        } & set(self.required_capabilities)
        if len(dd_modes) > 1:
            raise ValueError(
                f"Profile requires more than one drawdown mode: "
                f"{sorted(c.value for c in dd_modes)}. Pick one."
            )
        daily_modes = {
            Capability.DAILY_LOSS_INCLUDES_FLOATING,
            Capability.DAILY_LOSS_CLOSED_ONLY,
        } & set(self.required_capabilities)
        if len(daily_modes) > 1:
            raise ValueError(
                "Profile requires both floating-inclusive and closed-only "
                "daily loss. Pick one."
            )

    # ------------------------------------------------------------------
    # THE HONEST-ABSENCE SURFACE
    # ------------------------------------------------------------------
    def unsupported(self) -> List[UnsupportedRule]:
        """
        Every rule this profile asks for that will NOT be evaluated.

        Callers must propagate this into their result object. A compliance
        PASS with a non-empty unsupported() list is a partial answer, and
        rendering it as an unqualified pass is the bug this module exists to
        prevent.
        """
        gaps: List[UnsupportedRule] = []

        for cap in self.required_capabilities:
            if cap not in IMPLEMENTED:
                gaps.append(UnsupportedRule(
                    capability=cap,
                    reason=CAPABILITY_NOTES.get(
                        cap, "No implementation registered."
                    ),
                ))

        # A threshold set without the matching capability flag still means
        # the firm HAS that rule. Whether it counts as a gap depends only on
        # whether code backs it -- which is why this is gated on IMPLEMENTED
        # rather than on how the flags happened to be filled in.
        #
        # consistency_max_day_pct is honoured by the checker whenever it is
        # set, independent of required_capabilities, so with CONSISTENCY_RULE
        # implemented this correctly reports no gap.
        if (self.consistency_max_day_pct is not None
                and Capability.CONSISTENCY_RULE not in IMPLEMENTED
                and Capability.CONSISTENCY_RULE not in self.required_capabilities):
            gaps.append(UnsupportedRule(
                capability=Capability.CONSISTENCY_RULE,
                reason=CAPABILITY_NOTES[Capability.CONSISTENCY_RULE],
            ))

        return gaps

    @property
    def is_fully_modelled(self) -> bool:
        """True only if every rule this firm has is actually checked."""
        return not self.unsupported()

    def caveat_line(self) -> str:
        """One-line summary for display next to any PASS/FAIL badge."""
        gaps = self.unsupported()
        if not gaps:
            return f"All configured {self.firm_name} rules are modelled."
        names = ', '.join(g.capability.value for g in gaps)
        return (
            f"PARTIAL: {len(gaps)} {self.firm_name} rule(s) NOT checked "
            f"({names}). Result is a pass against the modelled subset only."
        )

    # ------------------------------------------------------------------
    # DERIVED VALUES used by the checker
    # ------------------------------------------------------------------
    def daily_loss_limit(self, anchor_balance: float) -> float:
        """Absolute currency loss that breaches the daily rule."""
        return float(anchor_balance) * self.max_daily_loss_pct

    def drawdown_floor(self, initial_balance: float) -> float:
        """Equity level at which total drawdown breaches (static mode)."""
        return float(initial_balance) * (1.0 - self.max_total_drawdown_pct)

    def profit_target_value(self, initial_balance: float, phase: str) -> float:
        """Absolute equity level that satisfies the profit target."""
        if phase not in self.profit_targets:
            raise ValueError(
                f"Unknown phase {phase!r} for {self.firm_name}. "
                f"Known phases: {sorted(self.profit_targets)}"
            )
        return float(initial_balance) * (1.0 + self.profit_targets[phase])

    @property
    def includes_floating_pnl(self) -> bool:
        return Capability.DAILY_LOSS_CLOSED_ONLY not in self.required_capabilities

    # ------------------------------------------------------------------
    # SERIALISATION -- dashboard round-trip
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['required_capabilities'] = [c.value for c in self.required_capabilities]
        d['unsupported'] = [
            {'capability': u.capability.value, 'reason': u.reason}
            for u in self.unsupported()
        ]
        d['is_fully_modelled'] = self.is_fully_modelled()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FirmRules":
        """
        Rebuild from a dashboard form payload.

        Unknown keys are rejected rather than ignored: a renamed field that
        silently reverts to its default is precisely the kind of quiet wrong
        number this codebase keeps getting bitten by.
        """
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in d.items()
                   if k not in ('unsupported', 'is_fully_modelled')}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"Unknown FirmRules fields: {sorted(unknown)}. "
                f"Known fields: {sorted(known)}"
            )
        return cls(**payload)


# ==============================================================================
# BUILT-IN PROFILES
# ==============================================================================
# Starting points, not gospel. Every one of these needs verifying against the
# firm's current rulebook before you trade against it -- that is the moving-
# target problem, and no amount of config solves it. Verify, then edit in the
# dashboard.

def ftmo() -> FirmRules:
    """FTMO-style: static drawdown, floating P&L counts, Prague reset."""
    return FirmRules(
        firm_name='FTMO',
        profile_note='Verify against current FTMO rulebook before relying on.',
        max_daily_loss_pct=0.05,
        max_total_drawdown_pct=0.10,
        min_trading_days=4,
        max_calendar_days=None,
        profit_targets={'challenge': 0.10, 'verification': 0.05},
        reset_timezone='Europe/Prague',
        required_capabilities=[
            Capability.STATIC_DRAWDOWN,
            Capability.DAILY_LOSS_INCLUDES_FLOATING,
            Capability.MIN_TRADING_DAYS,
        ],
    )


def generic_static(name: str = 'Generic (static DD)') -> FirmRules:
    """Blank-slate static-drawdown profile to edit in the dashboard."""
    return FirmRules(
        firm_name=name,
        profile_note='Template. Fill in from the firm rulebook.',
        required_capabilities=[
            Capability.STATIC_DRAWDOWN,
            Capability.DAILY_LOSS_INCLUDES_FLOATING,
            Capability.MIN_TRADING_DAYS,
        ],
    )


def generic_trailing(name: str = 'Generic (trailing DD)') -> FirmRules:
    """
    Trailing-drawdown template.

    Deliberately constructible and deliberately NOT evaluable: it reports
    TRAILING_DRAWDOWN_EOD through unsupported(). Available so you can model
    which firms you'd need code for before writing that code.
    """
    return FirmRules(
        firm_name=name,
        profile_note='Trailing drawdown is NOT implemented -- results partial.',
        required_capabilities=[
            Capability.TRAILING_DRAWDOWN_EOD,
            Capability.DAILY_LOSS_INCLUDES_FLOATING,
            Capability.MIN_TRADING_DAYS,
        ],
    )


BUILTIN_PROFILES = {
    'ftmo': ftmo,
    'generic_static': generic_static,
    'generic_trailing': generic_trailing,
}


def load_profile(key: str) -> FirmRules:
    if key not in BUILTIN_PROFILES:
        raise ValueError(
            f"Unknown profile {key!r}. Available: {sorted(BUILTIN_PROFILES)}"
        )
    return BUILTIN_PROFILES[key]()


# Backwards compatibility with the module-level constants in ftmo_compliance.py.
# The patcher rewires the checker to read from a FirmRules instance; these keep
# any straggling import working and resolving to the same numbers.
DEFAULT_RULES = ftmo()
MAX_DAILY_LOSS_PCT = DEFAULT_RULES.max_daily_loss_pct
MAX_TOTAL_DRAWDOWN_PCT = DEFAULT_RULES.max_total_drawdown_pct
MIN_TRADING_DAYS = DEFAULT_RULES.min_trading_days
PROFIT_TARGETS = dict(DEFAULT_RULES.profit_targets)
ACCOUNT_SIZES = list(DEFAULT_RULES.account_sizes)