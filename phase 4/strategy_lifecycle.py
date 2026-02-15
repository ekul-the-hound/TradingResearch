# ==============================================================================
# strategy_lifecycle.py
# ==============================================================================
# Phase 4, Module 4 (Week 18): Strategy Lifecycle Management
#
# State machine that manages strategy lifecycle from research through
# retirement. Automates promotion/demotion decisions based on objective
# performance criteria.
#
# States:
#   RESEARCH → PAPER → LIVE → RETIRED
#        ↓         ↓        ↓
#     REJECTED   REVIEW   PAUSED → RETIRED
#
# Promotion (PAPER → LIVE) requires:
#   - 30+ days paper trading
#   - Live Sharpe within 20% of backtest
#   - No kill switch triggers
#   - Drift tests passing
#   - Max drawdown < 20%
#
# Demotion (LIVE → REVIEW) triggers:
#   - Sharpe degradation > 50%
#   - Drawdown > 20%
#   - Critical drift detected
#   - Kill switch triggered
#
# Consumed by:
#   - optimization_pipeline.py (register new strategies)
#   - live_monitor.py (automated health-based transitions)
#   - dashboard (display lifecycle state)
#
# Usage:
#     from strategy_lifecycle import StrategyLifecycle, LifecycleConfig
#     lc = StrategyLifecycle()
#     lc.register("S_001", backtest_sharpe=1.8)
#     lc.promote("S_001")  # RESEARCH → PAPER
#     lc.check_auto_transitions("S_001", metrics)
# ==============================================================================

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ==============================================================================
# STATES
# ==============================================================================

class LifecycleState(Enum):
    RESEARCH = "research"
    PAPER = "paper"
    LIVE = "live"
    PAUSED = "paused"
    REVIEW = "review"
    RETIRED = "retired"
    REJECTED = "rejected"


# Allowed transitions
VALID_TRANSITIONS = {
    LifecycleState.RESEARCH: [LifecycleState.PAPER, LifecycleState.REJECTED],
    LifecycleState.PAPER: [LifecycleState.LIVE, LifecycleState.REVIEW, LifecycleState.REJECTED],
    LifecycleState.LIVE: [LifecycleState.PAUSED, LifecycleState.REVIEW, LifecycleState.RETIRED],
    LifecycleState.PAUSED: [LifecycleState.LIVE, LifecycleState.REVIEW, LifecycleState.RETIRED],
    LifecycleState.REVIEW: [LifecycleState.PAPER, LifecycleState.RETIRED, LifecycleState.LIVE],
    LifecycleState.RETIRED: [],
    LifecycleState.REJECTED: [],
}


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class LifecycleConfig:
    """Promotion/demotion thresholds."""
    # Promotion: PAPER → LIVE
    min_paper_days: int = 30
    max_sharpe_degradation_pct: float = 20.0
    max_drawdown_for_promotion: float = 0.20
    min_trades_for_promotion: int = 20
    require_no_drift: bool = True

    # Demotion: LIVE → REVIEW
    sharpe_demotion_pct: float = 50.0
    drawdown_demotion: float = 0.20
    max_consecutive_losses_demotion: int = 10

    # Retirement
    max_review_days: int = 30
    max_days_without_trade: int = 60


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class TransitionRecord:
    """Audit record of a state transition."""
    strategy_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    reason: str
    timestamp: str
    auto: bool = False  # True if automated, False if manual


@dataclass
class StrategyRecord:
    """Full lifecycle record for one strategy."""
    strategy_id: str
    state: LifecycleState
    created_at: str
    backtest_sharpe: float = 0.0
    paper_start: Optional[str] = None
    live_start: Optional[str] = None
    retired_at: Optional[str] = None
    paper_days: int = 0
    live_days: int = 0
    transitions: List[TransitionRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# LIFECYCLE MANAGER
# ==============================================================================

class StrategyLifecycle:
    """
    Manages strategy state transitions with audit trail.
    """

    def __init__(self, config: Optional[LifecycleConfig] = None):
        self.config = config or LifecycleConfig()
        self._strategies: Dict[str, StrategyRecord] = {}

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------
    def register(
        self,
        strategy_id: str,
        backtest_sharpe: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> StrategyRecord:
        """Register a new strategy in RESEARCH state."""
        rec = StrategyRecord(
            strategy_id=strategy_id,
            state=LifecycleState.RESEARCH,
            created_at=datetime.now().isoformat(),
            backtest_sharpe=backtest_sharpe,
            metadata=metadata or {},
        )
        self._strategies[strategy_id] = rec
        return rec

    # ------------------------------------------------------------------
    # MANUAL TRANSITIONS
    # ------------------------------------------------------------------
    def transition(
        self,
        strategy_id: str,
        to_state: LifecycleState,
        reason: str = "manual",
    ) -> TransitionRecord:
        """Manually transition a strategy to a new state."""
        rec = self._get(strategy_id)
        if to_state not in VALID_TRANSITIONS.get(rec.state, []):
            raise ValueError(
                f"Invalid transition: {rec.state.value} → {to_state.value}. "
                f"Valid: {[s.value for s in VALID_TRANSITIONS[rec.state]]}"
            )
        return self._do_transition(rec, to_state, reason, auto=False)

    def promote(self, strategy_id: str, reason: str = "manual promotion") -> TransitionRecord:
        """Promote to next state (RESEARCH→PAPER, PAPER→LIVE)."""
        rec = self._get(strategy_id)
        if rec.state == LifecycleState.RESEARCH:
            return self.transition(strategy_id, LifecycleState.PAPER, reason)
        elif rec.state == LifecycleState.PAPER:
            return self.transition(strategy_id, LifecycleState.LIVE, reason)
        elif rec.state == LifecycleState.PAUSED:
            return self.transition(strategy_id, LifecycleState.LIVE, reason)
        elif rec.state == LifecycleState.REVIEW:
            return self.transition(strategy_id, LifecycleState.PAPER, reason)
        raise ValueError(f"Cannot promote from {rec.state.value}")

    def demote(self, strategy_id: str, reason: str = "manual demotion") -> TransitionRecord:
        """Demote (LIVE→REVIEW, PAPER→REJECTED, etc)."""
        rec = self._get(strategy_id)
        if rec.state == LifecycleState.LIVE:
            return self.transition(strategy_id, LifecycleState.REVIEW, reason)
        elif rec.state == LifecycleState.PAPER:
            return self.transition(strategy_id, LifecycleState.REVIEW, reason)
        raise ValueError(f"Cannot demote from {rec.state.value}")

    def retire(self, strategy_id: str, reason: str = "retired") -> TransitionRecord:
        """Retire a strategy."""
        rec = self._get(strategy_id)
        if LifecycleState.RETIRED in VALID_TRANSITIONS.get(rec.state, []):
            return self.transition(strategy_id, LifecycleState.RETIRED, reason)
        raise ValueError(f"Cannot retire from {rec.state.value}")

    # ------------------------------------------------------------------
    # AUTO TRANSITIONS
    # ------------------------------------------------------------------
    def check_auto_transitions(
        self,
        strategy_id: str,
        live_sharpe: float = 0.0,
        max_drawdown: float = 0.0,
        days_active: int = 0,
        total_trades: int = 0,
        drift_detected: bool = False,
        consecutive_losses: int = 0,
        kill_switch_triggered: bool = False,
    ) -> Optional[TransitionRecord]:
        """
        Check if automated transition should occur based on metrics.
        Returns TransitionRecord if transition happened, None otherwise.
        """
        rec = self._get(strategy_id)
        cfg = self.config

        # PAPER → LIVE (auto-promote)
        if rec.state == LifecycleState.PAPER:
            rec.paper_days = days_active
            bt_sharpe = max(rec.backtest_sharpe, 0.01)
            degradation = (1 - live_sharpe / bt_sharpe) * 100 if bt_sharpe > 0 else 100

            can_promote = (
                days_active >= cfg.min_paper_days and
                degradation <= cfg.max_sharpe_degradation_pct and
                max_drawdown < cfg.max_drawdown_for_promotion and
                total_trades >= cfg.min_trades_for_promotion and
                (not cfg.require_no_drift or not drift_detected)
            )
            if can_promote:
                return self._do_transition(
                    rec, LifecycleState.LIVE,
                    f"Auto-promote: {days_active}d, Sharpe={live_sharpe:.2f}, "
                    f"DD={max_drawdown:.1%}, degrad={degradation:.0f}%",
                    auto=True,
                )

        # LIVE → REVIEW (auto-demote)
        elif rec.state == LifecycleState.LIVE:
            rec.live_days = days_active
            bt_sharpe = max(rec.backtest_sharpe, 0.01)
            degradation = (1 - live_sharpe / bt_sharpe) * 100 if bt_sharpe > 0 else 100

            should_demote = (
                degradation > cfg.sharpe_demotion_pct or
                max_drawdown > cfg.drawdown_demotion or
                consecutive_losses >= cfg.max_consecutive_losses_demotion or
                kill_switch_triggered
            )
            if should_demote:
                reasons = []
                if degradation > cfg.sharpe_demotion_pct:
                    reasons.append(f"Sharpe degrad {degradation:.0f}%")
                if max_drawdown > cfg.drawdown_demotion:
                    reasons.append(f"DD {max_drawdown:.1%}")
                if consecutive_losses >= cfg.max_consecutive_losses_demotion:
                    reasons.append(f"{consecutive_losses} consecutive losses")
                if kill_switch_triggered:
                    reasons.append("kill switch")
                return self._do_transition(
                    rec, LifecycleState.REVIEW,
                    f"Auto-demote: {', '.join(reasons)}", auto=True,
                )

        # LIVE → PAUSED (drift)
        elif rec.state == LifecycleState.LIVE and drift_detected:
            return self._do_transition(
                rec, LifecycleState.PAUSED,
                "Auto-pause: drift detected", auto=True,
            )

        # REVIEW → RETIRED (timeout)
        elif rec.state == LifecycleState.REVIEW:
            if days_active > cfg.max_review_days:
                return self._do_transition(
                    rec, LifecycleState.RETIRED,
                    f"Auto-retire: review timeout ({days_active}d)", auto=True,
                )

        return None

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------
    def get_state(self, strategy_id: str) -> LifecycleState:
        return self._get(strategy_id).state

    def get_record(self, strategy_id: str) -> StrategyRecord:
        return self._get(strategy_id)

    def get_all_by_state(self, state: LifecycleState) -> List[StrategyRecord]:
        return [r for r in self._strategies.values() if r.state == state]

    def get_audit_trail(self, strategy_id: str) -> List[TransitionRecord]:
        return self._get(strategy_id).transitions

    def summary(self) -> Dict[str, int]:
        counts = {}
        for state in LifecycleState:
            counts[state.value] = sum(1 for r in self._strategies.values() if r.state == state)
        return counts

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------
    def _get(self, strategy_id: str) -> StrategyRecord:
        if strategy_id not in self._strategies:
            raise KeyError(f"Strategy not registered: {strategy_id}")
        return self._strategies[strategy_id]

    def _do_transition(
        self, rec: StrategyRecord, to: LifecycleState, reason: str, auto: bool,
    ) -> TransitionRecord:
        tr = TransitionRecord(
            strategy_id=rec.strategy_id,
            from_state=rec.state,
            to_state=to,
            reason=reason,
            timestamp=datetime.now().isoformat(),
            auto=auto,
        )
        rec.transitions.append(tr)
        # Update timestamps
        if to == LifecycleState.PAPER:
            rec.paper_start = tr.timestamp
        elif to == LifecycleState.LIVE:
            rec.live_start = tr.timestamp
        elif to == LifecycleState.RETIRED:
            rec.retired_at = tr.timestamp
        rec.state = to
        return tr
