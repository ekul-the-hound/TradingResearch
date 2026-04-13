# ==============================================================================
# live_engine.py
# ==============================================================================
# Connects broker_adapter.py to all Phase 4 modules.
#
# This is the runtime loop that:
#   1. Polls prices from the broker
#   2. Feeds them to shadow traders / live strategies
#   3. Updates live_monitor with PnL/drawdown
#   4. Runs drift_detector on live returns
#   5. Checks kill_switch rules after every update
#   6. Executes kill_switch actions (flatten) through the broker
#   7. Triggers strategy_lifecycle transitions
#
# Usage:
#     from live_engine import LiveEngine
#     from broker_adapter import create_broker
#
#     broker = create_broker("ccxt", exchange="binance",
#                            api_key="...", api_secret="...")
#     broker.connect()
#
#     engine = LiveEngine(broker)
#     engine.add_strategy("S_001", symbol="BTC/USDT",
#                         backtest_sharpe=1.8, mode="shadow")
#     engine.run_loop(interval_seconds=60)
# ==============================================================================

import time
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from broker_adapter import BaseBroker, BrokerTick, BrokerOrder, PaperBroker

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class LiveEngineConfig:
    """Configuration for the live engine loop."""
    poll_interval_seconds: float = 60.0
    eod_hour_utc: int = 21          # 5pm EST = 21:00 UTC
    max_consecutive_errors: int = 10
    enable_kill_switch: bool = True
    enable_drift_detection: bool = True
    enable_auto_lifecycle: bool = True
    verbose: bool = True


# ==============================================================================
# STRATEGY SLOT
# ==============================================================================

@dataclass
class StrategySlot:
    """Tracks one strategy's live state."""
    strategy_id: str
    symbol: str
    mode: str                         # "shadow" or "live"
    backtest_sharpe: float = 0.0
    backtest_returns: Optional[np.ndarray] = None

    # Components (lazy-loaded)
    shadow_trader: Any = None
    drift_detector: Any = None
    kill_switch: Any = None

    # Signal callback: takes a tick, returns ("BUY"/"SELL"/None, size)
    signal_fn: Optional[Callable] = None

    # State
    daily_returns: List[float] = field(default_factory=list)
    last_equity: float = 0.0
    last_tick: Optional[BrokerTick] = None
    is_halted: bool = False
    error_count: int = 0


# ==============================================================================
# LIVE ENGINE
# ==============================================================================

class LiveEngine:
    """
    Runtime loop connecting the broker to all monitoring/risk modules.
    """

    def __init__(
        self,
        broker: BaseBroker,
        config: Optional[LiveEngineConfig] = None,
    ):
        self.broker = broker
        self.config = config or LiveEngineConfig()
        self._slots: Dict[str, StrategySlot] = {}
        self._monitor = None
        self._lifecycle = None
        self._running = False
        self._tick_count = 0
        self._log = logger.info if self.config.verbose else lambda *a, **k: None

        # Try to import Phase 4 modules
        self._init_monitor()
        self._init_lifecycle()

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    def _init_monitor(self):
        try:
            from live_monitor import LiveMonitor
            self._monitor = LiveMonitor()
            self._log("LiveEngine: LiveMonitor loaded")
        except ImportError:
            self._log("LiveEngine: live_monitor.py not available")

    def _init_lifecycle(self):
        try:
            from strategy_lifecycle import StrategyLifecycle
            self._lifecycle = StrategyLifecycle()
            self._log("LiveEngine: StrategyLifecycle loaded")
        except ImportError:
            self._log("LiveEngine: strategy_lifecycle.py not available")

    # ------------------------------------------------------------------
    # REGISTER STRATEGIES
    # ------------------------------------------------------------------
    def add_strategy(
        self,
        strategy_id: str,
        symbol: str,
        mode: str = "shadow",
        backtest_sharpe: float = 0.0,
        backtest_returns: Optional[np.ndarray] = None,
        signal_fn: Optional[Callable] = None,
        initial_capital: float = 100_000,
    ) -> StrategySlot:
        """
        Register a strategy for live monitoring.

        Args:
            strategy_id: Unique identifier
            symbol: Trading symbol (e.g. "BTC/USDT", "ES", "EUR/USD")
            mode: "shadow" (paper) or "live" (real orders)
            backtest_sharpe: Expected Sharpe from backtesting
            backtest_returns: Array of backtest daily returns (for drift detection)
            signal_fn: Callback that receives BrokerTick, returns (side, size) or None
            initial_capital: Capital allocated to this strategy
        """
        slot = StrategySlot(
            strategy_id=strategy_id,
            symbol=symbol,
            mode=mode,
            backtest_sharpe=backtest_sharpe,
            backtest_returns=backtest_returns,
            signal_fn=signal_fn,
            last_equity=initial_capital,
        )

        # Shadow trader
        try:
            from shadow_trader import ShadowTrader
            slot.shadow_trader = ShadowTrader(
                strategy_id=strategy_id,
                initial_capital=initial_capital,
            )
        except ImportError:
            pass

        # Drift detector (needs backtest returns as reference)
        if backtest_returns is not None and len(backtest_returns) > 20:
            try:
                from drift_detector import DriftDetector, DriftConfig
                slot.drift_detector = DriftDetector(
                    reference_returns=backtest_returns,
                    config=DriftConfig(),
                )
            except ImportError:
                pass

        # Kill switch
        try:
            from kill_switch import KillSwitch, KillSwitchConfig
            slot.kill_switch = KillSwitch(KillSwitchConfig())
        except ImportError:
            pass

        # Register with monitor
        if self._monitor:
            self._monitor.register_strategy(
                strategy_id,
                shadow_trader=slot.shadow_trader,
                mode=mode,
                capital=initial_capital,
                backtest_sharpe=backtest_sharpe,
            )

        # Register with lifecycle
        if self._lifecycle:
            self._lifecycle.register(strategy_id, backtest_sharpe=backtest_sharpe)
            if mode == "shadow":
                self._lifecycle.promote(strategy_id)  # RESEARCH -> PAPER

        self._slots[strategy_id] = slot
        self._log(f"Registered {strategy_id} on {symbol} (mode={mode})")
        return slot

    def remove_strategy(self, strategy_id: str):
        if strategy_id in self._slots:
            del self._slots[strategy_id]
            if self._monitor:
                self._monitor.unregister_strategy(strategy_id)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run_loop(self, interval_seconds: Optional[float] = None, max_ticks: int = 0):
        """
        Main execution loop. Polls broker, updates everything.

        Args:
            interval_seconds: Override poll interval
            max_ticks: Stop after N ticks (0 = run forever)
        """
        interval = interval_seconds or self.config.poll_interval_seconds
        self._running = True
        self._log(f"LiveEngine started -- polling every {interval}s, "
                  f"{len(self._slots)} strategies")

        consecutive_errors = 0

        while self._running:
            try:
                self._tick()
                self._tick_count += 1
                consecutive_errors = 0

                if max_ticks > 0 and self._tick_count >= max_ticks:
                    self._log(f"Reached max_ticks ({max_ticks}), stopping")
                    break

                time.sleep(interval)

            except KeyboardInterrupt:
                self._log("KeyboardInterrupt -- shutting down")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Tick error ({consecutive_errors}): {e}")
                if consecutive_errors >= self.config.max_consecutive_errors:
                    logger.critical(f"Too many errors ({consecutive_errors}), stopping")
                    self._emergency_shutdown()
                    break
                time.sleep(min(interval * 2, 30))

        self._running = False
        self._log("LiveEngine stopped")

    def stop(self):
        """Signal the loop to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # SINGLE TICK
    # ------------------------------------------------------------------
    def _tick(self):
        """Process one poll cycle for all strategies."""
        for sid, slot in self._slots.items():
            if slot.is_halted:
                continue

            # 1. Get price
            tick = self.broker.get_tick(slot.symbol)
            if tick is None:
                slot.error_count += 1
                continue
            slot.last_tick = tick
            slot.error_count = 0

            # 2. Run signal (if provided)
            if slot.signal_fn and not slot.is_halted:
                signal = slot.signal_fn(tick)
                if signal is not None:
                    side, size = signal
                    self._execute_signal(slot, side, size, tick)

            # 3. Mark to market
            if slot.shadow_trader:
                slot.shadow_trader.mark_to_market(tick.last)

            # 4. Update monitor
            if self._monitor:
                self._monitor.update(sid, price=tick.last)

            # 5. Drift detection
            if slot.drift_detector and self.config.enable_drift_detection:
                self._check_drift(slot)

            # 6. Kill switch
            if slot.kill_switch and self.config.enable_kill_switch:
                self._check_kill_switch(slot, tick)

        # End-of-day processing
        self._check_eod()

    def _execute_signal(self, slot: StrategySlot, side: str, size: float,
                        tick: BrokerTick):
        """Execute a trade signal through the appropriate channel."""
        if slot.mode == "shadow":
            # Paper trade only
            if slot.shadow_trader:
                slot.shadow_trader.submit_order(side, tick.last, size)
                self._log(f"[SHADOW] {slot.strategy_id}: {side} {size} @ {tick.last}")
        elif slot.mode == "live":
            # Real order through broker
            order = self.broker.submit_order(
                side=side, symbol=slot.symbol, size=size, order_type="market",
            )
            # Mirror in shadow trader for tracking
            if slot.shadow_trader:
                fill = order.fill_price or tick.last
                slot.shadow_trader.submit_order(side, fill, size)
            self._log(f"[LIVE] {slot.strategy_id}: {side} {size} @ "
                      f"{order.fill_price} [{order.status.value}]")

    # ------------------------------------------------------------------
    # DRIFT
    # ------------------------------------------------------------------
    def _check_drift(self, slot: StrategySlot):
        if not slot.daily_returns or len(slot.daily_returns) < 10:
            return
        try:
            result = slot.drift_detector.detect(np.array(slot.daily_returns[-60:]))
            if self._monitor:
                self._monitor.update(slot.strategy_id, drift_result=result)
            if hasattr(result, "drift_detected") and result.drift_detected:
                self._log(f"[WARN]  DRIFT detected for {slot.strategy_id}")
        except Exception as e:
            logger.debug(f"Drift check failed for {slot.strategy_id}: {e}")

    # ------------------------------------------------------------------
    # KILL SWITCH
    # ------------------------------------------------------------------
    def _check_kill_switch(self, slot: StrategySlot, tick: BrokerTick):
        if not slot.shadow_trader:
            return

        metrics = slot.shadow_trader.metrics
        result = slot.kill_switch.check(
            current_pnl=metrics.total_pnl,
            account_size=slot.shadow_trader.initial_capital,
            drawdown_pct=metrics.max_drawdown * 100,
            consecutive_losses=metrics.losing_trades,  # Simplified
            live_sharpe=metrics.sharpe_ratio,
            backtest_sharpe=slot.backtest_sharpe,
        )

        if result.triggered:
            self._log(f"[ALERT] KILL SWITCH [{slot.strategy_id}]: {result}")
            self._handle_kill_action(slot, result)

    def _handle_kill_action(self, slot: StrategySlot, result: Any):
        """Execute kill switch action through the broker."""
        from kill_switch import KillAction

        action = result.action

        if action == KillAction.LIQUIDATE:
            self._log(f"[RED] LIQUIDATING {slot.strategy_id}")
            if slot.mode == "live":
                self.broker.flatten(slot.symbol)
            if slot.shadow_trader:
                slot.shadow_trader.stop()
            slot.is_halted = True

            # Lifecycle transition
            if self._lifecycle:
                try:
                    self._lifecycle.demote(slot.strategy_id)
                except Exception:
                    pass

        elif action in (KillAction.HALT, KillAction.HALT_DAY, KillAction.HALT_WEEK):
            self._log(f"🛑 HALTING {slot.strategy_id}")
            slot.is_halted = True

        elif action == KillAction.REDUCE:
            self._log(f"[DOWN] REDUCING {slot.strategy_id}")
            # Close half the position
            pos = self.broker.get_position(slot.symbol)
            if pos and pos.size > 0 and slot.mode == "live":
                close_side = "sell" if pos.side == "long" else "buy"
                self.broker.submit_order(close_side, slot.symbol,
                                         pos.size / 2, "market")

    # ------------------------------------------------------------------
    # END OF DAY
    # ------------------------------------------------------------------
    def _check_eod(self):
        """Record daily returns at end of day."""
        now = datetime.utcnow()
        if now.hour != self.config.eod_hour_utc or now.minute > 1:
            return

        for sid, slot in self._slots.items():
            if slot.shadow_trader:
                tick = slot.last_tick
                if tick:
                    slot.shadow_trader.end_of_day(tick.last)
                    equity = slot.shadow_trader.capital + slot.shadow_trader.position.unrealized_pnl
                    if slot.last_equity > 0:
                        daily_ret = (equity - slot.last_equity) / slot.last_equity
                        slot.daily_returns.append(daily_ret)
                        if self._monitor:
                            self._monitor.update(sid, daily_return=daily_ret)
                    slot.last_equity = equity

        # Auto lifecycle checks
        if self._lifecycle and self.config.enable_auto_lifecycle:
            self._check_promotions()

    def _check_promotions(self):
        """Check if any shadow strategies qualify for promotion."""
        for sid, slot in self._slots.items():
            if slot.mode != "shadow" or not slot.shadow_trader:
                continue
            comparison = slot.shadow_trader.get_comparison(slot.backtest_sharpe)
            if comparison.get("meets_promotion_criteria", False):
                self._log(f"[DONE] {sid} meets promotion criteria!")
                if self._lifecycle:
                    try:
                        self._lifecycle.promote(sid)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # EMERGENCY
    # ------------------------------------------------------------------
    def _emergency_shutdown(self):
        """Flatten everything and stop."""
        self._log("[RED] EMERGENCY SHUTDOWN -- flattening all positions")
        try:
            orders = self.broker.flatten_all()
            self._log(f"   Flattened {len(orders)} positions")
        except Exception as e:
            logger.critical(f"   Flatten failed: {e}")
        for slot in self._slots.values():
            slot.is_halted = True
            if slot.shadow_trader:
                slot.shadow_trader.stop()
        self._running = False

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Get engine status summary."""
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "broker": self.broker.name,
            "broker_connected": self.broker.is_connected,
            "strategies": {
                sid: {
                    "symbol": s.symbol,
                    "mode": s.mode,
                    "halted": s.is_halted,
                    "errors": s.error_count,
                    "daily_returns": len(s.daily_returns),
                    "last_price": s.last_tick.last if s.last_tick else None,
                }
                for sid, s in self._slots.items()
            },
            "portfolio": str(self._monitor.get_portfolio_snapshot())
            if self._monitor else "monitor not loaded",
        }

    def get_portfolio_snapshot(self):
        """Delegate to live_monitor."""
        if self._monitor:
            return self._monitor.get_portfolio_snapshot()
        return None
