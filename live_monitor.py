# ==============================================================================
# live_monitor.py
# ==============================================================================
# Phase 4, Module 3 (Week 17): Live Performance Monitor
#
# Central monitoring hub that aggregates real-time data from all active
# strategies (shadow + live), tracks portfolio-level metrics, and
# dispatches alerts when thresholds are breached.
#
# Tracks:
#   - Per-strategy: PnL, Sharpe, DD, trade count, drift status
#   - Portfolio: total exposure, correlation, net PnL
#   - System: uptime, data freshness, error counts
#
# Consumed by:
#   - Dashboard (ReactPy pages)
#   - kill_switch.py (triggers)
#   - strategy_lifecycle.py (health checks)
#
# Usage:
#     from live_monitor import LiveMonitor
#     monitor = LiveMonitor()
#     monitor.register_strategy("S_001", shadow_trader=trader1)
#     monitor.update("S_001", price=1.105)
#     snapshot = monitor.get_portfolio_snapshot()
# ==============================================================================

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ==============================================================================
# ENUMS
# ==============================================================================

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class StrategyHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Alert:
    """System alert."""
    level: AlertLevel
    strategy_id: str
    message: str
    timestamp: str
    acknowledged: bool = False

    def __str__(self):
        icon = {"info": "ℹ️", "warning": "[WARN]", "critical": "[RED]"}[self.level.value]
        return f"{icon} [{self.strategy_id}] {self.message}"


@dataclass
class StrategySnapshot:
    """Point-in-time snapshot of one strategy."""
    strategy_id: str
    health: StrategyHealth
    mode: str                   # "shadow", "live", "paused"
    pnl: float
    pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    total_trades: int
    win_rate: float
    position_size: float
    position_side: str
    days_active: int
    drift_detected: bool
    last_update: str


@dataclass
class PortfolioSnapshot:
    """Portfolio-level aggregation."""
    timestamp: str
    total_capital: float
    total_pnl: float
    total_pnl_pct: float
    portfolio_sharpe: float
    portfolio_drawdown: float
    n_strategies_active: int
    n_strategies_healthy: int
    n_strategies_degraded: int
    n_strategies_critical: int
    total_exposure: float
    strategies: List[StrategySnapshot]
    active_alerts: List[Alert]

    def __str__(self):
        return (
            f"\n{'='*60}\n"
            f"  PORTFOLIO MONITOR -- {self.timestamp[:19]}\n"
            f"{'='*60}\n"
            f"  Capital:    ${self.total_capital:,.0f}\n"
            f"  PnL:        ${self.total_pnl:,.2f} ({self.total_pnl_pct:+.2f}%)\n"
            f"  Sharpe:     {self.portfolio_sharpe:.3f}\n"
            f"  Drawdown:   {self.portfolio_drawdown:.2%}\n"
            f"  Strategies: {self.n_strategies_active} "
            f"([OK]{self.n_strategies_healthy} [WARN]{self.n_strategies_degraded} "
            f"[RED]{self.n_strategies_critical})\n"
            f"  Alerts:     {len(self.active_alerts)}\n"
        )


# ==============================================================================
# ALERT RULES
# ==============================================================================

@dataclass
class AlertConfig:
    """Alert thresholds."""
    drawdown_warning: float = 0.10
    drawdown_critical: float = 0.20
    sharpe_degraded: float = 0.3
    sharpe_critical: float = 0.0
    max_daily_loss_pct: float = 3.0
    stale_data_minutes: int = 30
    max_consecutive_losses: int = 8


# ==============================================================================
# LIVE MONITOR
# ==============================================================================

class LiveMonitor:
    """Central monitoring hub for all strategies."""

    def __init__(self, alert_config: Optional[AlertConfig] = None):
        self.alert_config = alert_config or AlertConfig()
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[Alert] = []
        self._portfolio_returns: List[float] = []
        self._total_initial_capital = 0.0

    # ------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------
    def register_strategy(
        self,
        strategy_id: str,
        shadow_trader: Optional[Any] = None,
        mode: str = "shadow",
        capital: float = 100_000,
        backtest_sharpe: float = 0.0,
    ):
        """Register a strategy for monitoring."""
        self._strategies[strategy_id] = {
            "shadow_trader": shadow_trader,
            "mode": mode,
            "capital": capital,
            "backtest_sharpe": backtest_sharpe,
            "health": StrategyHealth.HEALTHY,
            "drift_detected": False,
            "last_update": datetime.now().isoformat(),
            "daily_returns": [],
            "pnl": 0.0,
            "consecutive_losses": 0,
        }
        self._total_initial_capital += capital

    def unregister_strategy(self, strategy_id: str):
        if strategy_id in self._strategies:
            self._total_initial_capital -= self._strategies[strategy_id]["capital"]
            del self._strategies[strategy_id]

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------
    def update(
        self,
        strategy_id: str,
        price: Optional[float] = None,
        pnl_update: Optional[float] = None,
        drift_result: Optional[Any] = None,
        daily_return: Optional[float] = None,
    ):
        """Push new data for a strategy."""
        if strategy_id not in self._strategies:
            return

        s = self._strategies[strategy_id]
        s["last_update"] = datetime.now().isoformat()

        # Update shadow trader
        if s["shadow_trader"] and price is not None:
            s["shadow_trader"].mark_to_market(price)

        # PnL update
        if pnl_update is not None:
            s["pnl"] += pnl_update
            if pnl_update < 0:
                s["consecutive_losses"] = s.get("consecutive_losses", 0) + 1
            else:
                s["consecutive_losses"] = 0

        # Daily return
        if daily_return is not None:
            s["daily_returns"].append(daily_return)

        # Drift
        if drift_result is not None:
            s["drift_detected"] = getattr(drift_result, "drift_detected", False)

        # Health check
        self._check_health(strategy_id)

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------
    def _check_health(self, strategy_id: str):
        s = self._strategies[strategy_id]
        cfg = self.alert_config
        health = StrategyHealth.HEALTHY
        ts = datetime.now().isoformat()

        trader = s.get("shadow_trader")
        if trader:
            metrics = trader.metrics
            # Drawdown
            if metrics.max_drawdown > cfg.drawdown_critical:
                health = StrategyHealth.CRITICAL
                self._add_alert(AlertLevel.CRITICAL, strategy_id,
                                f"DD {metrics.max_drawdown:.1%} > {cfg.drawdown_critical:.0%}", ts)
            elif metrics.max_drawdown > cfg.drawdown_warning:
                health = StrategyHealth.DEGRADED
                self._add_alert(AlertLevel.WARNING, strategy_id,
                                f"DD {metrics.max_drawdown:.1%} warning", ts)

            # Sharpe
            if metrics.sharpe_ratio < cfg.sharpe_critical and metrics.days_active >= 10:
                health = StrategyHealth.CRITICAL
                self._add_alert(AlertLevel.CRITICAL, strategy_id,
                                f"Sharpe {metrics.sharpe_ratio:.2f} < {cfg.sharpe_critical}", ts)
            elif metrics.sharpe_ratio < cfg.sharpe_degraded and metrics.days_active >= 10:
                health = max(health, StrategyHealth.DEGRADED, key=lambda h: h.value)

        # Consecutive losses
        if s.get("consecutive_losses", 0) >= cfg.max_consecutive_losses:
            health = StrategyHealth.CRITICAL
            self._add_alert(AlertLevel.CRITICAL, strategy_id,
                            f"{s['consecutive_losses']} consecutive losses", ts)

        # Drift
        if s.get("drift_detected"):
            health = max(health, StrategyHealth.DEGRADED, key=lambda h: h.value)
            self._add_alert(AlertLevel.WARNING, strategy_id, "Drift detected", ts)

        s["health"] = health

    def _add_alert(self, level: AlertLevel, sid: str, msg: str, ts: str):
        alert = Alert(level, sid, msg, ts)
        self._alerts.append(alert)

    # ------------------------------------------------------------------
    # SNAPSHOTS
    # ------------------------------------------------------------------
    def get_strategy_snapshot(self, strategy_id: str) -> Optional[StrategySnapshot]:
        if strategy_id not in self._strategies:
            return None
        s = self._strategies[strategy_id]
        trader = s.get("shadow_trader")
        if trader:
            m = trader.metrics
            pos = trader.position
            return StrategySnapshot(
                strategy_id=strategy_id,
                health=s["health"],
                mode=s["mode"],
                pnl=m.total_pnl,
                pnl_pct=m.total_pnl / max(s["capital"], 1) * 100,
                sharpe_ratio=m.sharpe_ratio,
                max_drawdown=m.max_drawdown,
                current_drawdown=m.current_drawdown,
                total_trades=m.total_trades,
                win_rate=m.win_rate,
                position_size=pos.size,
                position_side=pos.side.value,
                days_active=m.days_active,
                drift_detected=s.get("drift_detected", False),
                last_update=s["last_update"],
            )
        return StrategySnapshot(
            strategy_id=strategy_id, health=s["health"], mode=s["mode"],
            pnl=s["pnl"], pnl_pct=s["pnl"]/max(s["capital"],1)*100,
            sharpe_ratio=0, max_drawdown=0, current_drawdown=0,
            total_trades=0, win_rate=0, position_size=0, position_side="flat",
            days_active=len(s.get("daily_returns", [])),
            drift_detected=s.get("drift_detected", False),
            last_update=s["last_update"],
        )

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        ts = datetime.now().isoformat()
        snaps = [self.get_strategy_snapshot(sid) for sid in self._strategies]
        snaps = [s for s in snaps if s is not None]

        total_pnl = sum(s.pnl for s in snaps)
        total_cap = self._total_initial_capital or 1
        n_h = sum(1 for s in snaps if s.health == StrategyHealth.HEALTHY)
        n_d = sum(1 for s in snaps if s.health == StrategyHealth.DEGRADED)
        n_c = sum(1 for s in snaps if s.health == StrategyHealth.CRITICAL)

        # Portfolio Sharpe from strategy returns
        all_rets = []
        for s in self._strategies.values():
            all_rets.extend(s.get("daily_returns", []))
        if len(all_rets) > 5:
            r = np.array(all_rets)
            p_sharpe = float(np.mean(r) / max(np.std(r, ddof=1), 1e-10) * np.sqrt(252))
        else:
            p_sharpe = 0.0

        max_dd = max((s.max_drawdown for s in snaps), default=0)
        active_alerts = [a for a in self._alerts[-50:] if not a.acknowledged]

        return PortfolioSnapshot(
            timestamp=ts, total_capital=total_cap,
            total_pnl=total_pnl, total_pnl_pct=total_pnl / total_cap * 100,
            portfolio_sharpe=p_sharpe, portfolio_drawdown=max_dd,
            n_strategies_active=len(snaps), n_strategies_healthy=n_h,
            n_strategies_degraded=n_d, n_strategies_critical=n_c,
            total_exposure=sum(s.position_size for s in snaps),
            strategies=snaps, active_alerts=active_alerts,
        )

    def acknowledge_alerts(self):
        for a in self._alerts:
            a.acknowledged = True
