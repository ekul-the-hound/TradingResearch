# ==============================================================================
# governed_broker.py
# ==============================================================================
# Connects live_governor to the order path.
#
# THE GAP THIS CLOSES
# -------------------
# The governor knows when trading must stop. Nothing was asking it.
# live_engine._execute_signal calls broker.submit_order directly, so a
# strategy could keep opening positions straight through a daily-loss limit
# while a perfectly correct governor sat unconsulted. A safety component that
# is not on the path it protects is decoration.
#
# WHY A WRAPPER AND NOT AN EDIT TO live_engine
# --------------------------------------------
# Wrapping BaseBroker means the check cannot be bypassed by any caller,
# present or future. Editing live_engine would protect exactly the one call
# site I edited, and leave shadow_trader, backfill scripts, manual REPL
# sessions and whatever gets written next going straight to the broker.
#
#     broker = GovernedBroker(MT5Broker(...), governor, initial_balance=100_000)
#     engine = LiveEngine(broker=broker, ...)
#
# live_engine needs no changes at all.
#
# FAIL-CLOSED
# -----------
# Every uncertainty rejects the order:
#   - governor says HALT_NEW or FLATTEN     -> rejected
#   - account state cannot be read          -> rejected
#   - the governor itself raises            -> rejected
#
# A wrapper that falls back to the inner broker when something goes wrong is
# worse than no wrapper, because the operator believes orders are being
# checked. If this class cannot verify an order is safe, the order does not
# happen.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from broker_adapter import (
    BaseBroker, BrokerBalance, BrokerOrder, BrokerPosition, BrokerTick,
    OrderSide, OrderStatus, OrderType,
)
from live_governor import (
    AccountState, Decision, LiveGovernor, Verdict,
)


# Optional capabilities a broker MAY provide. Declared as protocols rather
# than probed with getattr so the calls below are typed, and so the optional
# surface an adapter can implement is written down somewhere.
@runtime_checkable
class SupportsAccountState(Protocol):
    def to_account_state(self, initial_balance: float) -> AccountState: ...


@runtime_checkable
class SupportsFlattenAll(Protocol):
    def flatten_all(self) -> List[BrokerOrder]: ...


@dataclass
class GateEvent:
    """One governor decision, for the audit trail."""
    timestamp: str
    decision: str
    reason: str
    detail: str
    symbol: str = ''
    side: str = ''
    requested_size: float = 0.0
    executed_size: float = 0.0


def account_state_from_broker(broker: BaseBroker,
                              initial_balance: float) -> AccountState:
    """
    Build an AccountState from any BaseBroker.

    BrokerBalance carries total_equity (marked to market) and unrealized_pnl,
    so closed balance is the difference. Keeping them apart matters: firms
    whose daily rule includes floating P&L need equity, and collapsing the two
    would quietly disable intraday protection for exactly those firms.

    MT5Broker.to_account_state() is preferred when available, since it reads
    balance and equity directly rather than deriving one from the other.
    """
    if isinstance(broker, SupportsAccountState):
        return broker.to_account_state(initial_balance)

    bal = broker.get_balance()
    equity = float(bal.total_equity)
    unrealized = float(getattr(bal, 'unrealized_pnl', 0.0) or 0.0)
    positions = broker.get_positions()
    return AccountState(
        timestamp=datetime.utcnow(),
        balance=equity - unrealized,
        equity=equity,
        initial_balance=float(initial_balance),
        open_positions=len(positions),
        symbol_exposure={p.symbol: p.size for p in positions},
    )


class GovernedBroker(BaseBroker):
    """
    A BaseBroker that consults a LiveGovernor before every order.

    Reads and cancellations pass straight through. Only order submission is
    gated, because only order submission can increase risk.
    """

    def __init__(
        self,
        broker: BaseBroker,
        governor: LiveGovernor,
        initial_balance: float,
        reduce_size_factor: float = 0.5,
        auto_flatten: bool = True,
        log_fn: Optional[Any] = None,
    ):
        super().__init__(name=f"governed({broker.name})")
        if initial_balance <= 0:
            raise ValueError(
                f"initial_balance must be positive, got {initial_balance!r}. "
                f"Every prop-firm limit is a fraction of it.")
        if not 0 < reduce_size_factor <= 1:
            raise ValueError("reduce_size_factor must be in (0, 1].")

        self.broker = broker
        self.governor = governor
        self.initial_balance = float(initial_balance)
        self.reduce_size_factor = reduce_size_factor
        self.auto_flatten = auto_flatten
        self._log_fn = log_fn
        self.events: List[GateEvent] = []
        self.last_verdict: Optional[Verdict] = None
        self.blocked_count = 0
        self.flatten_count = 0

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self._log_fn:
            try:
                self._log_fn(msg)
            except Exception:                             # pragma: no cover
                pass

    def _record(self, verdict: Verdict, symbol: str = '', side: str = '',
                requested: float = 0.0, executed: float = 0.0) -> None:
        self.events.append(GateEvent(
            timestamp=datetime.utcnow().isoformat(),
            decision=verdict.decision.value,
            reason=verdict.reason,
            detail=verdict.detail,
            symbol=symbol, side=side,
            requested_size=requested, executed_size=executed,
        ))

    # ------------------------------------------------------------------
    # PASS-THROUGH
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        ok = self.broker.connect()
        self.is_connected = self.broker.is_connected
        return ok

    def disconnect(self):
        self.broker.disconnect()
        self.is_connected = False

    def get_balance(self) -> BrokerBalance:
        return self.broker.get_balance()

    def get_positions(self) -> List[BrokerPosition]:
        return self.broker.get_positions()

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        return self.broker.get_position(symbol)

    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        return self.broker.get_tick(symbol)

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        return self.broker.get_order(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """Never gated. Cancelling can only reduce exposure."""
        return self.broker.cancel_order(order_id)

    # ------------------------------------------------------------------
    # THE GATE
    # ------------------------------------------------------------------
    def check(self) -> Verdict:
        """
        Ask the governor about the account right now.

        Any failure reading the account becomes a FLATTEN verdict rather than
        an exception: the caller is a trading loop, and an exception there is
        one `except` away from being treated as "carry on".
        """
        try:
            state = account_state_from_broker(self.broker, self.initial_balance)
        except Exception as e:
            v = Verdict(
                decision=Decision.FLATTEN,
                reason='account_state_unavailable',
                detail=(f"Could not read account state: {type(e).__name__}: "
                        f"{e}. Treating as unsafe rather than assuming it is "
                        f"fine."))
            self.last_verdict = v
            return v

        v = self.governor.observe(state)
        self.last_verdict = v
        return v

    def heartbeat(self) -> Verdict:
        """
        Call on every poll cycle, not just before orders.

        A breach can happen with no signal firing at all -- an open position
        drifting against you moves equity without anyone submitting anything.
        Checking only at order time means the account can fail during a quiet
        stretch and nothing notices.
        """
        v = self.check()
        if v.must_flatten and self.auto_flatten:
            self._flatten(v)
        return v

    def _flatten(self, verdict: Verdict) -> List[BrokerOrder]:
        self.flatten_count += 1
        self._log(f"[GOVERNOR] FLATTEN: {verdict.reason} -- {verdict.detail}")

        # Narrowed through a local: isinstance against a Protocol narrows the
        # attribute to Never in the negative branch, which makes the fallback
        # look unreachable.
        orders: List[BrokerOrder] = []
        inner: Any = self.broker
        if isinstance(inner, SupportsFlattenAll):
            orders = list(inner.flatten_all())
        else:
            for pos in self.broker.get_positions():
                if pos.size <= 0 or pos.side == 'flat':
                    continue
                orders.append(self.broker.submit_order(
                    side='sell' if pos.side == 'long' else 'buy',
                    symbol=pos.symbol, size=pos.size, order_type='market'))
        self._record(verdict, executed=float(len(orders)))
        return orders

    # ------------------------------------------------------------------
    def submit_order(
        self,
        side: str,
        symbol: str,
        size: float,
        order_type: str = 'market',
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> BrokerOrder:
        verdict = self.check()

        if verdict.must_flatten:
            if self.auto_flatten:
                self._flatten(verdict)
            self.blocked_count += 1
            self._record(verdict, symbol, side, size, 0.0)
            return self._blocked(symbol, side, size, verdict)

        if not verdict.may_open:
            self.blocked_count += 1
            self._record(verdict, symbol, side, size, 0.0)
            self._log(f"[GOVERNOR] blocked {side} {size} {symbol}: "
                      f"{verdict.reason}")
            return self._blocked(symbol, side, size, verdict)

        exec_size = size
        if verdict.decision is Decision.REDUCE:
            exec_size = size * self.reduce_size_factor
            self._log(f"[GOVERNOR] reduced {symbol} {size} -> {exec_size}: "
                      f"{verdict.reason}")

        order = self.broker.submit_order(
            side=side, symbol=symbol, size=exec_size,
            order_type=order_type, price=price, stop_price=stop_price)

        self._record(verdict, symbol, side, size, exec_size)
        self._order_history.append(order)
        return order

    def _blocked(self, symbol: str, side: str, size: float,
                 verdict: Verdict) -> BrokerOrder:
        """
        A rejected order, not an exception and not a silent no-op.

        Returning None or a filled-looking order would let the caller's
        position tracking drift away from reality, which is its own hazard on
        top of the one the governor was raising.
        """
        order = BrokerOrder(
            order_id='',
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=size,
            status=OrderStatus.REJECTED,
            timestamp=datetime.utcnow().isoformat(),
            raw={
                'error': f"Blocked by governor: {verdict.reason}",
                'governor_decision': verdict.decision.value,
                'governor_detail': verdict.detail,
                'daily_loss': verdict.daily_loss,
                'headroom': verdict.headroom,
                'unchecked_rules': list(verdict.unchecked_rules),
            },
        )
        self._order_history.append(order)
        return order

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        v = self.last_verdict
        return {
            'broker': self.broker.name,
            'initial_balance': self.initial_balance,
            'blocked_orders': self.blocked_count,
            'flatten_events': self.flatten_count,
            'events_recorded': len(self.events),
            'last_decision': v.decision.value if v else None,
            'last_reason': v.reason if v else None,
            'headroom': v.headroom if v else None,
            'unchecked_rules': list(v.unchecked_rules) if v else [],
        }
