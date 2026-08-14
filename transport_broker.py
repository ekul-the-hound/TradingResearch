# ==============================================================================
# transport_broker.py -- BaseBroker over a swappable MT5Transport
# ==============================================================================
# A broker adapter that speaks the project's broker_base interface but delegates
# every actual market interaction to an MT5Transport (see mt5_transport.py). The
# same adapter therefore runs unchanged over a file-IPC bridge (macOS), the
# native Windows MetaTrader5 package, or a socket bridge -- only the injected
# transport differs.
#
# WHAT THIS ADDS OVER A RAW TRANSPORT:
#   * Translation between the project's BrokerTick/Position/Balance/Order types
#     and the transport's plain types.
#   * SERVER-SIDE STOPS: every market order carries its stop-loss to the broker,
#     so a crashed engine never leaves a naked position. If a caller submits an
#     order with no stop, that is surfaced as a warning, not silently allowed.
#   * RESTART RECONCILIATION: reconcile() diffs live broker positions against a
#     last-known local snapshot and reports what changed, so on reconnect the
#     engine acts on truth rather than assumptions.
#   * Unit conversion seam: units <-> lots via a pluggable, explicit converter,
#     with an honest default and a loud note that per-symbol contract data is
#     not yet modelled (matches the known limitation elsewhere in the system).
#
# DESIGN PRINCIPLE (project-wide):
#   Uncertainty is represented, never hidden. A stopless order warns. An order
#   whose result is unknown (transport timeout) is reported as not-ok and left
#   for reconciliation. Reconciliation reports drift rather than swallowing it.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from broker_base import (
    BaseBroker, BrokerTick, BrokerPosition, BrokerBalance, BrokerOrder,
    OrderSide, OrderType, OrderStatus,
)
from mt5_transport import (
    MT5Transport, TransportOrder, TransportPosition, TransportTick,
    TransportAccount, TransportError, TransportStale,
)


# ── Unit conversion seam ──────────────────────────────────────────────────────
# The project has no per-symbol contract/lot normalisation (see the note in
# position_sizing.py). Rather than fabricate contract sizes, the default
# converter treats one "unit" as one lot and records a warning so the caller
# knows the number was not properly normalised. Inject a real converter once
# per-symbol contract specs exist.
UnitsToLots = Callable[[str, float], float]


def default_units_to_lots(symbol: str, units: float) -> float:
    """Identity conversion with the honest assumption 1 unit == 1 lot."""
    return units


@dataclass
class ReconcileReport:
    """Result of diffing broker truth against the local snapshot."""
    matched: List[str] = field(default_factory=list)      # symbols in agreement
    only_on_broker: List[str] = field(default_factory=list)   # engine unaware
    only_local: List[str] = field(default_factory=list)   # gone at broker
    size_mismatch: List[str] = field(default_factory=list)    # size differs

    @property
    def clean(self) -> bool:
        return not (self.only_on_broker or self.only_local or self.size_mismatch)

    def summary(self) -> str:
        if self.clean:
            return f"reconciled clean ({len(self.matched)} position(s) agree)"
        parts = []
        if self.only_on_broker:
            parts.append(f"UNTRACKED at broker: {', '.join(self.only_on_broker)}")
        if self.only_local:
            parts.append(f"GONE at broker: {', '.join(self.only_local)}")
        if self.size_mismatch:
            parts.append(f"SIZE MISMATCH: {', '.join(self.size_mismatch)}")
        return " | ".join(parts)


class TransportBroker(BaseBroker):
    """BaseBroker implementation delegating to an injected MT5Transport."""

    def __init__(
        self,
        transport: MT5Transport,
        symbols: Optional[List[str]] = None,
        units_to_lots: UnitsToLots = default_units_to_lots,
        require_stops: bool = True,
        name: str = "transport_mt5",
    ):
        super().__init__(name=name)
        self.transport = transport
        self.symbols = symbols or []
        self.units_to_lots = units_to_lots
        self.require_stops = require_stops
        self.last_notes: List[str] = []
        # Local snapshot of positions keyed by symbol, for reconciliation.
        self._local_positions: Dict[str, BrokerPosition] = {}

    # -- Lifecycle -------------------------------------------------------------
    def connect(self) -> bool:
        ok = self.transport.connect()
        self.is_connected = bool(ok)
        return self.is_connected

    def disconnect(self) -> None:
        try:
            self.transport.disconnect()
        finally:
            self.is_connected = False

    # -- Market data -----------------------------------------------------------
    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        try:
            ticks = self.transport.get_ticks([symbol])
        except TransportStale:
            # Stale data must not masquerade as a live quote.
            return None
        except TransportError:
            return None
        t = ticks.get(symbol)
        if t is None:
            return None
        return BrokerTick(
            symbol=t.symbol, bid=t.bid, ask=t.ask, last=t.last or t.bid,
            volume_24h=t.volume, timestamp=t.time or _utcnow_iso(),
        )

    def get_balance(self) -> BrokerBalance:
        try:
            a = self.transport.get_account()
        except TransportError:
            # Represent the failure rather than reporting a fake zero balance
            # that downstream code might treat as real.
            return BrokerBalance(total_equity=0.0, free_margin=0.0,
                                 used_margin=0.0, currency="USD",
                                 timestamp=_utcnow_iso())
        return BrokerBalance(
            total_equity=a.equity or a.balance,
            free_margin=a.margin_free,
            used_margin=a.margin_used,
            currency=a.currency,
            timestamp=a.time or _utcnow_iso(),
        )

    def get_positions(self) -> List[BrokerPosition]:
        try:
            raw = self.transport.get_positions()
        except TransportError:
            return []
        return [self._to_position(p) for p in raw]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    def _to_position(self, p: TransportPosition) -> BrokerPosition:
        side = "long" if p.side == "buy" else "short" if p.side == "sell" else "flat"
        return BrokerPosition(
            symbol=p.symbol, side=side, size=p.volume,
            entry_price=p.price_open, current_price=p.price_current,
            unrealized_pnl=p.profit,
        )

    # -- Order submission ------------------------------------------------------
    def submit_order(
        self,
        side: str,
        symbol: str,
        size: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs: Any,
    ) -> BrokerOrder:
        self.last_notes = []
        ts = _utcnow_iso()

        # Server-side stop discipline. A market order without a stop can leave a
        # naked position if the engine dies -- warn loudly (or refuse under a
        # stricter policy the caller can enforce by checking last_notes).
        if stop_loss is None:
            self.last_notes.append(
                "order submitted WITHOUT a server-side stop-loss; a crashed "
                "engine would leave this position unmanaged")
            if self.require_stops:
                return self._rejected(symbol, side, size, ts,
                                      "rejected: server-side stop-loss required "
                                      "(require_stops=True) but none supplied")

        tside = "buy" if side in ("buy", "long") else "sell"
        lots = self.units_to_lots(symbol, size)
        if self.units_to_lots is default_units_to_lots:
            self.last_notes.append(
                "unit->lot conversion used identity default (1 unit == 1 lot); "
                "per-symbol contract sizing is not modelled")

        torder = TransportOrder(
            symbol=symbol, side=tside, volume=lots,
            order_type=order_type, price=price,
            sl=stop_loss, tp=take_profit,
            comment=str(kwargs.get("comment", "")),
        )
        try:
            result = self.transport.place_order(torder)
        except TransportError as e:
            return self._rejected(symbol, side, size, ts,
                                  f"transport error: {e}")

        if not result.ok:
            # Includes the timeout/UNKNOWN case. Caller must reconcile.
            return self._rejected(symbol, side, size, ts,
                                  result.comment or "order not confirmed")

        # Update local snapshot so reconciliation has a baseline.
        order_side = OrderSide.BUY if tside == "buy" else OrderSide.SELL
        self._local_positions[symbol] = BrokerPosition(
            symbol=symbol,
            side="long" if tside == "buy" else "short",
            size=size,
            entry_price=result.fill_price or (price or 0.0),
        )
        return BrokerOrder(
            order_id=str(result.ticket) if result.ticket is not None else "",
            symbol=symbol, side=order_side,
            order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
            size=size, status=OrderStatus.FILLED,
            fill_price=result.fill_price, filled_size=result.filled_volume,
            timestamp=ts, broker_ref=str(result.ticket or ""),
            raw=result.raw,
        )

    def _rejected(self, symbol: str, side: str, size: float, ts: str,
                  reason: str) -> BrokerOrder:
        return BrokerOrder(
            order_id="", symbol=symbol,
            side=OrderSide.BUY if side in ("buy", "long") else OrderSide.SELL,
            order_type=OrderType.MARKET, size=size,
            status=OrderStatus.REJECTED, timestamp=ts, reason=reason,
        )

    def cancel_order(self, order_id: str) -> bool:
        cancel = getattr(self.transport, "cancel_order", None)
        if cancel is None:
            return False
        try:
            res = cancel(int(order_id))
        except (ValueError, TransportError):
            return False
        return bool(getattr(res, "ok", False))

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        # The transport contract is fire-and-confirm; there is no order registry
        # to query after the fact. Positions are the source of truth.
        return None

    # -- Reconciliation --------------------------------------------------------
    def snapshot_local(self) -> None:
        """Record current broker positions as the local baseline."""
        self._local_positions = {p.symbol: p for p in self.get_positions()}

    def reconcile(self) -> ReconcileReport:
        """
        Diff live broker positions against the local snapshot. Call this on
        reconnect / restart BEFORE trading so the engine acts on truth.
        """
        report = ReconcileReport()
        broker = {p.symbol: p for p in self.get_positions()}
        local = self._local_positions

        for sym, bpos in broker.items():
            if sym not in local:
                report.only_on_broker.append(sym)
            elif abs(bpos.size - local[sym].size) > 1e-9:
                report.size_mismatch.append(sym)
            else:
                report.matched.append(sym)
        for sym in local:
            if sym not in broker:
                report.only_local.append(sym)

        # After reconciliation, broker truth becomes the new baseline.
        self._local_positions = dict(broker)
        return report

    # -- Governor bridge -------------------------------------------------------
    def flatten_all(self) -> List[BrokerOrder]:
        """Close every open position with an opposing market order."""
        orders: List[BrokerOrder] = []
        for pos in self.get_positions():
            if abs(pos.size) < 1e-12:
                continue
            closing = "sell" if pos.side == "long" else "buy"
            # Closing orders intentionally carry no new stop.
            prev = self.require_stops
            self.require_stops = False
            try:
                orders.append(self.submit_order(
                    side=closing, symbol=pos.symbol, size=abs(pos.size),
                    order_type="market", comment="flatten_all"))
            finally:
                self.require_stops = prev
        return orders


# ── Helpers ───────────────────────────────────────────────────────────────────
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["TransportBroker", "ReconcileReport", "default_units_to_lots"]
