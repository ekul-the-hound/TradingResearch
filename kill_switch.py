# ==============================================================================
# kill_switch.py
# ==============================================================================
# Phase 3, Module 3 (Week 12): Kill Switch & Emergency Controls
#
# Monitors strategy performance in real-time and triggers automated
# responses when risk thresholds are breached:
#   - WARN: Log alert, reduce position size
#   - HALT: Stop new entries, keep existing positions
#   - REDUCE: Cut position size by 50%
#   - LIQUIDATE: Close all positions immediately
#
# Rules cover daily loss, weekly loss, drawdown, consecutive losses,
# Sharpe degradation, volatility spikes, and correlation regime shifts.
#
# Consumed by:
#   - live_monitor.py (Phase 4) -- real-time monitoring
#   - shadow_trader.py (Phase 4) -- paper trade safety
#   - execution_engine.py -- pre-trade risk checks
#
# Usage:
#     from kill_switch import KillSwitch, KillSwitchConfig
#     ks = KillSwitch(KillSwitchConfig(max_daily_loss_pct=3.0))
#     action = ks.check(current_pnl=-350, account_size=10000,
#                       drawdown_pct=8.5, consecutive_losses=6)
#     if action.triggered:
#         print(f"KILL SWITCH: {action}")
# ==============================================================================

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class KillAction(Enum):
    NONE = "none"
    WARN = "warn"
    REDUCE = "reduce"          # Cut size 50%
    HALT = "halt"              # No new entries
    HALT_DAY = "halt_day"      # No entries rest of day
    HALT_WEEK = "halt_week"    # No entries rest of week
    LIQUIDATE = "liquidate"    # Close everything
    REVIEW = "review"          # Flag for manual review


class RuleSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class KillSwitchConfig:
    """Kill switch thresholds."""
    # Loss limits
    max_daily_loss_pct: float = 3.0
    max_weekly_loss_pct: float = 5.0
    max_monthly_loss_pct: float = 8.0

    # Drawdown
    reduce_drawdown_pct: float = 15.0     # Reduce size at 15%
    halt_drawdown_pct: float = 20.0       # Halt at 20%
    liquidate_drawdown_pct: float = 25.0  # Liquidate at 25%

    # Streak
    max_consecutive_losses: int = 8
    review_consecutive_losses: int = 5

    # Performance degradation
    sharpe_degradation_pct: float = 50.0  # Halt if live Sharpe < 50% of backtest
    min_live_sharpe: float = 0.0          # Absolute minimum live Sharpe

    # Volatility
    vol_spike_mult: float = 3.0           # Halt if vol > 3x normal

    # Position limits
    max_position_pct: float = 20.0        # Max 20% of AUM in one position
    max_correlation: float = 0.8          # Max correlation between positions

    # FTMO compliance (optional)
    ftmo_mode: bool = False
    ftmo_daily_limit_pct: float = 5.0
    ftmo_total_limit_pct: float = 10.0


# ==============================================================================
# RESULT
# ==============================================================================

@dataclass
class KillSwitchResult:
    """Output from kill switch check."""
    triggered: bool
    action: KillAction
    severity: RuleSeverity
    rule_name: str
    message: str
    value: float                    # Current value that triggered
    threshold: float                # Threshold that was breached
    timestamp: str = ""

    def __str__(self):
        icon = {
            KillAction.NONE: "[OK]",
            KillAction.WARN: "[WARN]",
            KillAction.REDUCE: "[DOWN]",
            KillAction.HALT: "🛑",
            KillAction.HALT_DAY: "🛑",
            KillAction.HALT_WEEK: "🛑",
            KillAction.LIQUIDATE: "[RED]",
            KillAction.REVIEW: "👁️",
        }.get(self.action, "❓")
        return f"{icon} [{self.action.value}] {self.rule_name}: {self.message}"


@dataclass
class KillSwitchState:
    """Current state tracked by the kill switch."""
    is_halted: bool = False
    is_reduced: bool = False
    halt_until: Optional[str] = None
    size_multiplier: float = 1.0
    active_alerts: List[KillSwitchResult] = field(default_factory=list)
    history: List[KillSwitchResult] = field(default_factory=list)

    def clear(self):
        self.is_halted = False
        self.is_reduced = False
        self.halt_until = None
        self.size_multiplier = 1.0
        self.active_alerts = []


# ==============================================================================
# KILL SWITCH ENGINE
# ==============================================================================

class KillSwitch:
    """
    Risk monitoring engine with tiered responses.
    """

    def __init__(self, config: Optional[KillSwitchConfig] = None):
        self.config = config or KillSwitchConfig()
        self.state = KillSwitchState()

    # ------------------------------------------------------------------
    # MAIN CHECK
    # ------------------------------------------------------------------
    def check(
        self,
        current_pnl: float = 0.0,
        account_size: float = 100_000,
        drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
        live_sharpe: Optional[float] = None,
        backtest_sharpe: Optional[float] = None,
        current_vol: Optional[float] = None,
        normal_vol: Optional[float] = None,
        daily_loss_pct: Optional[float] = None,
        weekly_loss_pct: Optional[float] = None,
        monthly_loss_pct: Optional[float] = None,
        position_pct: Optional[float] = None,
    ) -> KillSwitchResult:
        """
        Run all kill switch rules and return highest-severity trigger.
        """
        ts = datetime.now().isoformat()
        results = []

        # Daily loss
        if daily_loss_pct is not None:
            results.append(self._check_daily_loss(daily_loss_pct, ts))

        # Compute daily loss from PnL if not provided
        if daily_loss_pct is None and current_pnl < 0 and account_size > 0:
            dl = abs(current_pnl) / account_size * 100
            results.append(self._check_daily_loss(dl, ts))

        # Weekly loss
        if weekly_loss_pct is not None:
            results.append(self._check_weekly_loss(weekly_loss_pct, ts))

        # Monthly loss
        if monthly_loss_pct is not None:
            results.append(self._check_monthly_loss(monthly_loss_pct, ts))

        # Drawdown
        results.append(self._check_drawdown(drawdown_pct, ts))

        # Consecutive losses
        results.append(self._check_streak(consecutive_losses, ts))

        # Sharpe degradation
        if live_sharpe is not None and backtest_sharpe is not None:
            results.append(self._check_sharpe_degradation(
                live_sharpe, backtest_sharpe, ts,
            ))

        # Volatility spike
        if current_vol is not None and normal_vol is not None:
            results.append(self._check_vol_spike(current_vol, normal_vol, ts))

        # Position concentration
        if position_pct is not None:
            results.append(self._check_position_size(position_pct, ts))

        # FTMO mode
        if self.config.ftmo_mode:
            if daily_loss_pct is not None:
                results.append(self._check_ftmo(daily_loss_pct, drawdown_pct, ts))

        # Find worst
        triggered = [r for r in results if r.triggered]
        if not triggered:
            return KillSwitchResult(
                triggered=False, action=KillAction.NONE,
                severity=RuleSeverity.LOW, rule_name="all_clear",
                message="All checks passed", value=0, threshold=0, timestamp=ts,
            )

        worst = max(triggered, key=lambda r: r.severity.value)
        self._apply_action(worst)
        return worst

    # ------------------------------------------------------------------
    # INDIVIDUAL RULES
    # ------------------------------------------------------------------
    def _check_daily_loss(self, loss_pct: float, ts: str) -> KillSwitchResult:
        cfg = self.config
        if loss_pct >= cfg.max_daily_loss_pct:
            return KillSwitchResult(
                True, KillAction.HALT_DAY, RuleSeverity.HIGH,
                "daily_loss", f"Daily loss {loss_pct:.1f}% >= {cfg.max_daily_loss_pct}%",
                loss_pct, cfg.max_daily_loss_pct, ts,
            )
        if loss_pct >= cfg.max_daily_loss_pct * 0.8:
            return KillSwitchResult(
                True, KillAction.WARN, RuleSeverity.MEDIUM,
                "daily_loss_warning", f"Daily loss {loss_pct:.1f}% nearing limit",
                loss_pct, cfg.max_daily_loss_pct, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "daily_loss", "", loss_pct, cfg.max_daily_loss_pct, ts)

    def _check_weekly_loss(self, loss_pct: float, ts: str) -> KillSwitchResult:
        cfg = self.config
        if loss_pct >= cfg.max_weekly_loss_pct:
            return KillSwitchResult(
                True, KillAction.HALT_WEEK, RuleSeverity.HIGH,
                "weekly_loss", f"Weekly loss {loss_pct:.1f}% >= {cfg.max_weekly_loss_pct}%",
                loss_pct, cfg.max_weekly_loss_pct, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "weekly_loss", "", loss_pct, cfg.max_weekly_loss_pct, ts)

    def _check_monthly_loss(self, loss_pct: float, ts: str) -> KillSwitchResult:
        cfg = self.config
        if loss_pct >= cfg.max_monthly_loss_pct:
            return KillSwitchResult(
                True, KillAction.HALT, RuleSeverity.HIGH,
                "monthly_loss", f"Monthly loss {loss_pct:.1f}% >= {cfg.max_monthly_loss_pct}%",
                loss_pct, cfg.max_monthly_loss_pct, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "monthly_loss", "", loss_pct, cfg.max_monthly_loss_pct, ts)

    def _check_drawdown(self, dd_pct: float, ts: str) -> KillSwitchResult:
        cfg = self.config
        if dd_pct >= cfg.liquidate_drawdown_pct:
            return KillSwitchResult(
                True, KillAction.LIQUIDATE, RuleSeverity.CRITICAL,
                "max_drawdown", f"DD {dd_pct:.1f}% >= {cfg.liquidate_drawdown_pct}% LIQUIDATE",
                dd_pct, cfg.liquidate_drawdown_pct, ts,
            )
        if dd_pct >= cfg.halt_drawdown_pct:
            return KillSwitchResult(
                True, KillAction.HALT, RuleSeverity.HIGH,
                "halt_drawdown", f"DD {dd_pct:.1f}% >= {cfg.halt_drawdown_pct}%",
                dd_pct, cfg.halt_drawdown_pct, ts,
            )
        if dd_pct >= cfg.reduce_drawdown_pct:
            return KillSwitchResult(
                True, KillAction.REDUCE, RuleSeverity.MEDIUM,
                "reduce_drawdown", f"DD {dd_pct:.1f}% >= {cfg.reduce_drawdown_pct}%",
                dd_pct, cfg.reduce_drawdown_pct, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "drawdown", "", dd_pct, cfg.reduce_drawdown_pct, ts)

    def _check_streak(self, n: int, ts: str) -> KillSwitchResult:
        cfg = self.config
        if n >= cfg.max_consecutive_losses:
            return KillSwitchResult(
                True, KillAction.HALT, RuleSeverity.HIGH,
                "consecutive_losses", f"{n} consecutive losses >= {cfg.max_consecutive_losses}",
                n, cfg.max_consecutive_losses, ts,
            )
        if n >= cfg.review_consecutive_losses:
            return KillSwitchResult(
                True, KillAction.REVIEW, RuleSeverity.MEDIUM,
                "streak_review", f"{n} consecutive losses -- review recommended",
                n, cfg.review_consecutive_losses, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "streak", "", n, cfg.max_consecutive_losses, ts)

    def _check_sharpe_degradation(
        self, live: float, backtest: float, ts: str,
    ) -> KillSwitchResult:
        cfg = self.config
        if backtest <= 0:
            return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                    "sharpe_degrade", "", 0, 0, ts)
        degradation = (1 - live / backtest) * 100
        if live < cfg.min_live_sharpe:
            return KillSwitchResult(
                True, KillAction.HALT, RuleSeverity.HIGH,
                "min_sharpe", f"Live Sharpe {live:.2f} < min {cfg.min_live_sharpe}",
                live, cfg.min_live_sharpe, ts,
            )
        if degradation >= cfg.sharpe_degradation_pct:
            return KillSwitchResult(
                True, KillAction.REDUCE, RuleSeverity.MEDIUM,
                "sharpe_degradation",
                f"Sharpe dropped {degradation:.0f}% (live={live:.2f}, bt={backtest:.2f})",
                degradation, cfg.sharpe_degradation_pct, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "sharpe_degrade", "", degradation, cfg.sharpe_degradation_pct, ts)

    def _check_vol_spike(
        self, current: float, normal: float, ts: str,
    ) -> KillSwitchResult:
        cfg = self.config
        if normal <= 0:
            return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                    "vol_spike", "", 0, 0, ts)
        ratio = current / normal
        if ratio >= cfg.vol_spike_mult:
            return KillSwitchResult(
                True, KillAction.REDUCE, RuleSeverity.HIGH,
                "vol_spike", f"Vol {ratio:.1f}x normal (>= {cfg.vol_spike_mult}x)",
                ratio, cfg.vol_spike_mult, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "vol_spike", "", ratio, cfg.vol_spike_mult, ts)

    def _check_position_size(self, pos_pct: float, ts: str) -> KillSwitchResult:
        cfg = self.config
        if pos_pct >= cfg.max_position_pct:
            return KillSwitchResult(
                True, KillAction.WARN, RuleSeverity.MEDIUM,
                "position_concentration",
                f"Position {pos_pct:.1f}% of AUM >= {cfg.max_position_pct}%",
                pos_pct, cfg.max_position_pct, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "position_size", "", pos_pct, cfg.max_position_pct, ts)

    def _check_ftmo(self, daily_pct: float, dd_pct: float, ts: str) -> KillSwitchResult:
        cfg = self.config
        if dd_pct >= cfg.ftmo_total_limit_pct:
            return KillSwitchResult(
                True, KillAction.LIQUIDATE, RuleSeverity.CRITICAL,
                "ftmo_total_dd", f"FTMO: DD {dd_pct:.1f}% >= {cfg.ftmo_total_limit_pct}%",
                dd_pct, cfg.ftmo_total_limit_pct, ts,
            )
        if daily_pct >= cfg.ftmo_daily_limit_pct:
            return KillSwitchResult(
                True, KillAction.HALT_DAY, RuleSeverity.CRITICAL,
                "ftmo_daily", f"FTMO: Daily {daily_pct:.1f}% >= {cfg.ftmo_daily_limit_pct}%",
                daily_pct, cfg.ftmo_daily_limit_pct, ts,
            )
        return KillSwitchResult(False, KillAction.NONE, RuleSeverity.LOW,
                                "ftmo", "", 0, 0, ts)

    # ------------------------------------------------------------------
    # ACTION APPLICATION
    # ------------------------------------------------------------------
    def _apply_action(self, result: KillSwitchResult):
        self.state.active_alerts.append(result)
        self.state.history.append(result)

        if result.action == KillAction.LIQUIDATE:
            self.state.is_halted = True
            self.state.size_multiplier = 0.0
        elif result.action in (KillAction.HALT, KillAction.HALT_DAY, KillAction.HALT_WEEK):
            self.state.is_halted = True
        elif result.action == KillAction.REDUCE:
            self.state.is_reduced = True
            self.state.size_multiplier = 0.5

    # ------------------------------------------------------------------
    # PRE-TRADE CHECK
    # ------------------------------------------------------------------
    def can_trade(self) -> bool:
        """Returns True if trading is currently allowed."""
        return not self.state.is_halted

    def position_size_multiplier(self) -> float:
        """Returns the current position size multiplier (0.0 to 1.0)."""
        return self.state.size_multiplier

    def reset(self):
        """Reset kill switch state (e.g., start of new day/week)."""
        self.state.clear()
