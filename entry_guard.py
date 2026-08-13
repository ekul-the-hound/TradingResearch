# ==============================================================================
# entry_guard.py -- Pre-Order Entry Feasibility Guard
# ==============================================================================
# The last gate before an order is sent. It answers: "given the current quote,
# account state, and daily budget, is it safe and feasible to open THIS size in
# THIS symbol right now?" -- so the engine never fires an order that the broker
# will reject, or that would blow a rule the instant it fills.
#
# WHAT IT CHECKS (only things the system genuinely knows):
#   1. Spread     -- quote is clean and not too wide (reuses live_tick_guard).
#   2. Notional   -- position notional does not exceed the leverage-cap rule the
#                    sizing layer already enforces (default 20% of equity).
#   3. Free margin-- notional fits available free margin with a safety buffer.
#   4. Daily risk -- the new position's stop-distance risk fits the REMAINING
#                    daily-loss budget (open risk + this trade <= budget).
#
# WHAT IT DELIBERATELY DOES NOT FAKE:
#   Broker-style margin (e.g. 1:30 leverage x contract size) cannot be computed
#   accurately because the system has no per-symbol contract/lot normalisation
#   (see the note in position_sizing.py). Rather than invent a contract size and
#   stamp an order "feasible" on fabricated inputs -- the exact "confident wrong
#   numbers" failure mode -- this guard reports that dimension as UNVERIFIABLE.
#   Under strict=True an unverifiable margin dimension blocks the entry.
#
# SEVERITY -> DECISION:
#   PASS  every check satisfied                       -> ALLOW
#   WARN  a soft concern or an unverifiable dimension -> ALLOW (unless strict)
#   BLOCK a hard failure                              -> DENY
#
# USAGE:
#   from entry_guard import EntryGuard, EntryGuardConfig, EntryRequest
#   guard = EntryGuard(EntryGuardConfig(max_spread_bps=8.0))
#   decision = guard.check(EntryRequest(
#       symbol="EURUSD", side="long", size=0.5, price=1.10,
#       stop_distance=0.0020, tick=tick, balance=balance,
#       remaining_daily_budget=4200.0, current_open_risk=800.0))
#   if decision.allowed:
#       broker.submit_order(...)
#   else:
#       log.warning(decision.reasons)
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

# ── Severities ────────────────────────────────────────────────────────────────
PASS = "PASS"
WARN = "WARN"
BLOCK = "BLOCK"


@dataclass
class EntryGuardConfig:
    """Thresholds for the entry guard."""
    # Reject quotes wider than this (basis points of mid). Mirrors tick guard.
    max_spread_bps: float = 10.0
    # Notional leverage cap as a fraction of equity, matching position_sizing's
    # default max_leverage_pct. A position notional above this is over-levered.
    max_notional_pct_of_equity: float = 0.20
    # Require free margin to cover notional times this buffer (headroom for
    # adverse moves / other positions). 1.0 = exactly cover; 1.2 = 20% cushion.
    free_margin_buffer: float = 1.10
    # If True, any UNVERIFIABLE dimension (e.g. broker margin) blocks the entry.
    strict: bool = False


@dataclass
class EntryRequest:
    """Everything needed to judge one proposed entry."""
    symbol: str
    side: str                       # 'long' | 'short'
    size: float                     # absolute position size (units)
    price: float                    # intended entry price (mid or signal price)
    stop_distance: float = 0.0      # price distance to stop (per unit)
    tick: Any = None                # broker_base.BrokerTick, if available
    balance: Any = None             # broker_base.BrokerBalance, if available
    remaining_daily_budget: Optional[float] = None   # currency, remaining today
    current_open_risk: float = 0.0  # currency, stop-distance risk already open


@dataclass
class GuardCheck:
    name: str
    severity: str
    detail: str = ""


@dataclass
class EntryDecision:
    allowed: bool
    checks: List[GuardCheck] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        return [f"[{c.severity}] {c.name}: {c.detail}"
                for c in self.checks if c.severity != PASS]

    def __bool__(self) -> bool:
        return self.allowed


class EntryGuard:
    """Stateless pre-order feasibility checker."""

    def __init__(self, config: Optional[EntryGuardConfig] = None):
        self.config = config or EntryGuardConfig()

    def check(self, req: EntryRequest) -> EntryDecision:
        checks: List[GuardCheck] = [
            self._check_size_valid(req),
            self._check_spread(req),
            self._check_notional_cap(req),
            self._check_free_margin(req),
            self._check_daily_risk(req),
            self._check_broker_margin(req),
        ]
        # Decision: any BLOCK denies. A WARN denies only under strict.
        has_block = any(c.severity == BLOCK for c in checks)
        has_warn = any(c.severity == WARN for c in checks)
        allowed = not has_block and not (self.config.strict and has_warn)
        return EntryDecision(allowed=allowed, checks=checks)

    # -- Individual checks -----------------------------------------------------
    def _check_size_valid(self, req: EntryRequest) -> GuardCheck:
        name = "size_valid"
        if req.size is None or req.size <= 0:
            return GuardCheck(name, BLOCK, f"non-positive size {req.size}")
        if req.price is None or req.price <= 0:
            return GuardCheck(name, BLOCK, f"non-positive price {req.price}")
        if req.side not in ("long", "short"):
            return GuardCheck(name, BLOCK, f"invalid side {req.side!r}")
        return GuardCheck(name, PASS, f"{req.side} {req.size} @ {req.price}")

    def _check_spread(self, req: EntryRequest) -> GuardCheck:
        name = "spread"
        if req.tick is None:
            return GuardCheck(name, WARN, "no tick supplied; spread unverified")
        bid = _f(getattr(req.tick, "bid", None))
        ask = _f(getattr(req.tick, "ask", None))
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return GuardCheck(name, BLOCK, f"invalid quote bid={bid} ask={ask}")
        if bid > ask:
            return GuardCheck(name, BLOCK, f"crossed market bid {bid} > ask {ask}")
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10_000.0
        if self.config.max_spread_bps > 0 and spread_bps > self.config.max_spread_bps:
            return GuardCheck(name, BLOCK,
                              f"spread {spread_bps:.2f} bps > "
                              f"{self.config.max_spread_bps:.2f} bps limit")
        return GuardCheck(name, PASS, f"spread {spread_bps:.2f} bps")

    def _check_notional_cap(self, req: EntryRequest) -> GuardCheck:
        name = "notional_cap"
        equity = self._equity(req)
        if equity is None:
            return GuardCheck(name, WARN, "no balance supplied; notional cap unverified")
        if equity <= 0:
            return GuardCheck(name, BLOCK, f"non-positive equity {equity}")
        notional = abs(req.size) * req.price
        pct = notional / equity
        cap = self.config.max_notional_pct_of_equity
        if cap > 0 and pct > cap + 1e-9:
            return GuardCheck(name, BLOCK,
                              f"notional {notional:.0f} is {pct*100:.1f}% of equity, "
                              f"over the {cap*100:.0f}% cap")
        return GuardCheck(name, PASS,
                          f"notional {notional:.0f} = {pct*100:.1f}% of equity")

    def _check_free_margin(self, req: EntryRequest) -> GuardCheck:
        name = "free_margin"
        bal = req.balance
        if bal is None:
            return GuardCheck(name, WARN, "no balance supplied; free margin unverified")
        free = _f(getattr(bal, "free_margin", None))
        if free is None:
            return GuardCheck(name, WARN, "balance has no free_margin field")
        notional = abs(req.size) * req.price
        needed = notional * self.config.free_margin_buffer
        # NOTE: this compares notional-with-buffer against free margin. It is a
        # conservative proxy, NOT true broker margin (see _check_broker_margin).
        if needed > free + 1e-9:
            return GuardCheck(name, BLOCK,
                              f"needs ~{needed:.0f} (notional x buffer) but only "
                              f"{free:.0f} free margin available")
        return GuardCheck(name, PASS,
                          f"{free:.0f} free covers ~{needed:.0f} needed")

    def _check_daily_risk(self, req: EntryRequest) -> GuardCheck:
        name = "daily_risk"
        budget = req.remaining_daily_budget
        if budget is None:
            return GuardCheck(name, WARN,
                              "no remaining_daily_budget supplied; risk unverified")
        if req.stop_distance is None or req.stop_distance <= 0:
            # Without a stop we cannot bound the trade's contribution to daily
            # loss. That is a real hazard for a daily-loss rule, so WARN loudly.
            return GuardCheck(name, WARN,
                              "no stop distance; trade's daily-loss contribution "
                              "is unbounded")
        trade_risk = abs(req.size) * req.stop_distance
        projected = req.current_open_risk + trade_risk
        if projected > budget + 1e-9:
            return GuardCheck(name, BLOCK,
                              f"open risk {req.current_open_risk:.0f} + this trade "
                              f"{trade_risk:.0f} = {projected:.0f} exceeds remaining "
                              f"daily budget {budget:.0f}")
        return GuardCheck(name, PASS,
                          f"projected open risk {projected:.0f} within budget "
                          f"{budget:.0f}")

    def _check_broker_margin(self, req: EntryRequest) -> GuardCheck:
        name = "broker_margin"
        # Honest limitation: no per-symbol contract size / lot normalisation
        # exists in the system, so true broker margin (leverage x contract size)
        # cannot be computed without fabricating inputs. Report it as such.
        return GuardCheck(
            name, WARN,
            "broker-leverage margin UNVERIFIABLE: no per-symbol contract/lot "
            "data in the system. Notional and free-margin proxies were checked "
            "instead. Validate real margin against the broker on a demo account.",
        )

    def _equity(self, req: EntryRequest) -> Optional[float]:
        """Equity from the balance object, tolerant of field naming."""
        return _equity_from_balance(req.balance)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _equity_from_balance(bal: Any) -> Optional[float]:
    if bal is None:
        return None
    for attr in ("total_equity", "total"):
        v = _f(getattr(bal, attr, None))
        if v is not None:
            return v
    return None


if __name__ == "__main__":
    # Tiny self-demo.
    from dataclasses import dataclass as _dc

    @_dc
    class _Tick:
        symbol: str
        bid: float
        ask: float
        last: float = 0.0

    @_dc
    class _Bal:
        total_equity: float
        free_margin: float
        used_margin: float = 0.0

    g = EntryGuard(EntryGuardConfig(max_spread_bps=10.0))
    req = EntryRequest(
        symbol="EURUSD", side="long", size=10_000, price=1.10,
        stop_distance=0.0020,
        tick=_Tick("EURUSD", 1.09998, 1.10002),
        balance=_Bal(total_equity=100_000, free_margin=95_000),
        remaining_daily_budget=4000.0, current_open_risk=500.0,
    )
    d = g.check(req)
    print("ALLOWED" if d.allowed else "DENIED")
    for c in d.checks:
        print(f"  [{c.severity}] {c.name}: {c.detail}")
