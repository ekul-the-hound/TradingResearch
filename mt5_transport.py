# ==============================================================================
# mt5_transport.py -- Swappable Transport Layer for MT5 (and MT5-like) Brokers
# ==============================================================================
# The trading engine must not care HOW it talks to MetaTrader 5. This module
# defines a thin internal transport interface -- five methods -- and ships two
# implementations plus a fake for tests. The broker adapter (transport_broker.py)
# is written against this interface ONLY, so the same adapter runs unchanged
# whether the actual bridge is:
#
#   * a file-based IPC bridge (an MQL5 Expert Advisor writing/reading JSON),
#     which is macOS-native and needs no Wine/Docker/emulation, or
#   * the native Windows `MetaTrader5` package on a VPS, or
#   * a socket/RPC bridge over emulation.
#
# Only the transport changes between those; the adapter and everything above it
# stay identical. This is the "locked" design constraint from the roadmap.
#
# ------------------------------------------------------------------------------
# THE FIVE METHODS
# ------------------------------------------------------------------------------
#   get_ticks(symbols)     -> {symbol: TransportTick}
#   get_positions()        -> [TransportPosition]
#   place_order(order)     -> TransportOrderResult
#   get_account()          -> TransportAccount
#   get_rates(symbol, tf, n) -> [TransportBar]      (optional; may be empty)
#
# Plus lifecycle: connect() / disconnect() / is_alive().
#
# ------------------------------------------------------------------------------
# FILE-IPC CONTRACT (what the EA and this side agree on)
# ------------------------------------------------------------------------------
# The EA (MQL5, running inside MT5) and FileIPCTransport (Python) communicate
# through a shared directory. Files are written atomically (temp + rename) by
# whichever side owns them, and every payload carries a UTC "timestamp" and a
# monotonically increasing "seq" so the reader can detect staleness.
#
#   <dir>/state.json        EA -> Python. The EA rewrites this on each tick with
#                           the full snapshot:
#       {
#         "seq": 12345,
#         "timestamp": "2026-08-13T21:00:00.123Z",
#         "account": {"balance":..., "equity":..., "margin_free":...,
#                     "margin_used":..., "currency":"USD"},
#         "ticks": {"EURUSD": {"bid":..., "ask":..., "last":...,
#                              "time":"...", "volume":...}, ...},
#         "positions": [{"ticket":123, "symbol":"EURUSD", "type":"buy",
#                        "volume":0.10, "price_open":..., "price_current":...,
#                        "sl":..., "tp":..., "profit":...}, ...]
#       }
#
#   <dir>/commands.jsonl    Python -> EA. Append-only. Each line is one command:
#       {"id":"uuid", "action":"order", "seq":N, "order":{...}}
#       {"id":"uuid", "action":"cancel", "seq":N, "ticket":123}
#     The EA consumes lines it has not seen (by "id"), acts, and writes results.
#
#   <dir>/results.jsonl     EA -> Python. Append-only. One line per handled
#     command:
#       {"id":"uuid", "ok":true, "ticket":123, "retcode":10009,
#        "fill_price":..., "filled_volume":..., "comment":"..."}
#
# This module implements the PYTHON side of that contract. The MQL5 EA that
# writes state.json / reads commands.jsonl / writes results.jsonl must be built
# separately and validated on a demo account -- no amount of Python testing can
# substitute for that, so the contract above is the spec the EA is written to.
#
# DESIGN PRINCIPLE (project-wide):
#   Missing/stale data is represented and raised, never papered over. A stale
#   state file raises TransportStale rather than returning old prices as if live.
# ==============================================================================

from __future__ import annotations

import os
import io
import abc
import json
import uuid
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Errors ────────────────────────────────────────────────────────────────────
class TransportError(RuntimeError):
    """Base for all transport failures."""


class TransportNotConnected(TransportError):
    """Operation attempted before connect() / after disconnect()."""


class TransportStale(TransportError):
    """The freshest data available is older than the allowed staleness."""


# ── Data types (transport-level, deliberately plain dicts-of-floats) ──────────
@dataclass
class TransportTick:
    symbol: str
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    time: str = ""


@dataclass
class TransportPosition:
    ticket: int
    symbol: str
    side: str            # 'buy' | 'sell'
    volume: float        # in lots
    price_open: float = 0.0
    price_current: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0


@dataclass
class TransportAccount:
    balance: float = 0.0
    equity: float = 0.0
    margin_free: float = 0.0
    margin_used: float = 0.0
    currency: str = "USD"
    time: str = ""


@dataclass
class TransportBar:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class TransportOrder:
    """A request to place an order, transport-agnostic."""
    symbol: str
    side: str                       # 'buy' | 'sell'
    volume: float                   # lots
    order_type: str = "market"      # 'market' | 'limit' | 'stop'
    price: Optional[float] = None   # for limit/stop
    sl: Optional[float] = None      # server-side stop-loss price
    tp: Optional[float] = None      # server-side take-profit price
    deviation: int = 20
    comment: str = ""


@dataclass
class TransportOrderResult:
    ok: bool
    ticket: Optional[int] = None
    retcode: Optional[int] = None
    fill_price: Optional[float] = None
    filled_volume: float = 0.0
    comment: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


# ── The interface ─────────────────────────────────────────────────────────────
class MT5Transport(abc.ABC):
    """Abstract five-method transport. Implementations must be swappable."""

    @abc.abstractmethod
    def connect(self) -> bool: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def is_alive(self) -> bool: ...

    @abc.abstractmethod
    def get_ticks(self, symbols: List[str]) -> Dict[str, TransportTick]: ...

    @abc.abstractmethod
    def get_positions(self) -> List[TransportPosition]: ...

    @abc.abstractmethod
    def place_order(self, order: TransportOrder) -> TransportOrderResult: ...

    @abc.abstractmethod
    def get_account(self) -> TransportAccount: ...

    def get_rates(self, symbol: str, timeframe: str,
                  count: int) -> List[TransportBar]:
        """Optional historical bars. Default: not supported -> empty."""
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ==============================================================================
# FILE-IPC TRANSPORT (Python side of the EA contract)
# ==============================================================================
@dataclass
class FileIPCConfig:
    directory: str
    max_state_age_seconds: float = 5.0   # state.json older than this -> stale
    result_wait_seconds: float = 10.0    # how long to wait for an order result
    result_poll_seconds: float = 0.1


class FileIPCTransport(MT5Transport):
    """
    Talks to an MQL5 EA through files in a shared directory. See the module
    docstring for the exact file contract.

    This side never blocks on the EA except when waiting for an order result,
    and even then it times out rather than hanging forever.
    """

    STATE_FILE = "state.json"
    COMMANDS_FILE = "commands.jsonl"
    RESULTS_FILE = "results.jsonl"

    def __init__(self, config: FileIPCConfig):
        self.config = config
        self.dir = Path(config.directory)
        self._connected = False
        self._seq = 0
        self._seen_result_ids: set = set()

    # -- Lifecycle -------------------------------------------------------------
    def connect(self) -> bool:
        self.dir.mkdir(parents=True, exist_ok=True)
        # Ensure the command/results files exist so appends/reads never race a
        # missing file.
        for fname in (self.COMMANDS_FILE, self.RESULTS_FILE):
            p = self.dir / fname
            if not p.exists():
                p.write_text("", encoding="utf-8")
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_alive(self) -> bool:
        """Connected AND a fresh state file is present."""
        if not self._connected:
            return False
        try:
            self._read_state(require_fresh=True)
            return True
        except TransportError:
            return False

    # -- State reads -----------------------------------------------------------
    def _read_state(self, require_fresh: bool = True) -> Dict[str, Any]:
        if not self._connected:
            raise TransportNotConnected("transport not connected")
        p = self.dir / self.STATE_FILE
        if not p.exists():
            raise TransportStale(f"no state file at {p}")
        try:
            raw = p.read_text(encoding="utf-8")
            state = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            raise TransportError(f"unreadable state file: {e}") from e

        if require_fresh:
            ts = _parse_iso(state.get("timestamp", ""))
            if ts is None:
                raise TransportStale("state file has no parseable timestamp")
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > self.config.max_state_age_seconds:
                raise TransportStale(
                    f"state file is {age:.1f}s old "
                    f"(max {self.config.max_state_age_seconds}s)")
        return state

    def get_ticks(self, symbols: List[str]) -> Dict[str, TransportTick]:
        state = self._read_state(require_fresh=True)
        ticks_raw = state.get("ticks", {}) or {}
        out: Dict[str, TransportTick] = {}
        for sym in symbols:
            t = ticks_raw.get(sym)
            if not t:
                continue
            out[sym] = TransportTick(
                symbol=sym,
                bid=float(t.get("bid", 0) or 0),
                ask=float(t.get("ask", 0) or 0),
                last=float(t.get("last", 0) or 0),
                volume=float(t.get("volume", 0) or 0),
                time=str(t.get("time", "")),
            )
        return out

    def get_positions(self) -> List[TransportPosition]:
        state = self._read_state(require_fresh=True)
        out: List[TransportPosition] = []
        for p in state.get("positions", []) or []:
            out.append(TransportPosition(
                ticket=int(p.get("ticket", 0) or 0),
                symbol=str(p.get("symbol", "")),
                side=str(p.get("type", p.get("side", ""))).lower(),
                volume=float(p.get("volume", 0) or 0),
                price_open=float(p.get("price_open", 0) or 0),
                price_current=float(p.get("price_current", 0) or 0),
                sl=float(p.get("sl", 0) or 0),
                tp=float(p.get("tp", 0) or 0),
                profit=float(p.get("profit", 0) or 0),
            ))
        return out

    def get_account(self) -> TransportAccount:
        state = self._read_state(require_fresh=True)
        a = state.get("account", {}) or {}
        return TransportAccount(
            balance=float(a.get("balance", 0) or 0),
            equity=float(a.get("equity", 0) or 0),
            margin_free=float(a.get("margin_free", 0) or 0),
            margin_used=float(a.get("margin_used", 0) or 0),
            currency=str(a.get("currency", "USD")),
            time=str(state.get("timestamp", "")),
        )

    # -- Order placement (write command, await result) ------------------------
    def place_order(self, order: TransportOrder) -> TransportOrderResult:
        if not self._connected:
            raise TransportNotConnected("transport not connected")
        self._seq += 1
        cmd_id = str(uuid.uuid4())
        command = {
            "id": cmd_id,
            "action": "order",
            "seq": self._seq,
            "timestamp": _utcnow_iso(),
            "order": asdict(order),
        }
        self._append_line(self.COMMANDS_FILE, json.dumps(command))
        return self._await_result(cmd_id)

    def cancel_order(self, ticket: int) -> TransportOrderResult:
        if not self._connected:
            raise TransportNotConnected("transport not connected")
        self._seq += 1
        cmd_id = str(uuid.uuid4())
        command = {
            "id": cmd_id,
            "action": "cancel",
            "seq": self._seq,
            "timestamp": _utcnow_iso(),
            "ticket": ticket,
        }
        self._append_line(self.COMMANDS_FILE, json.dumps(command))
        return self._await_result(cmd_id)

    def _append_line(self, fname: str, line: str) -> None:
        p = self.dir / fname
        # Append is atomic enough for single-writer; the EA is the only reader.
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _await_result(self, cmd_id: str) -> TransportOrderResult:
        deadline = time.monotonic() + self.config.result_wait_seconds
        while time.monotonic() < deadline:
            res = self._find_result(cmd_id)
            if res is not None:
                return res
            time.sleep(self.config.result_poll_seconds)
        # Timeout: we do NOT know if the order filled. Report ok=False loudly;
        # the reconciliation pass on the broker side will discover any position
        # that actually opened. Never assume success on timeout.
        return TransportOrderResult(
            ok=False, retcode=None,
            comment=f"no result for command {cmd_id} within "
                    f"{self.config.result_wait_seconds}s (order status UNKNOWN; "
                    f"reconcile positions before retrying)",
        )

    def _find_result(self, cmd_id: str) -> Optional[TransportOrderResult]:
        p = self.dir / self.RESULTS_FILE
        if not p.exists():
            return None
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") != cmd_id:
                continue
            return TransportOrderResult(
                ok=bool(r.get("ok", False)),
                ticket=r.get("ticket"),
                retcode=r.get("retcode"),
                fill_price=r.get("fill_price"),
                filled_volume=float(r.get("filled_volume", 0) or 0),
                comment=str(r.get("comment", "")),
                raw=r,
            )
        return None


# ==============================================================================
# FAKE TRANSPORT (for tests and offline development)
# ==============================================================================
class FakeTransport(MT5Transport):
    """
    An in-memory transport with programmable state. Lets the broker adapter and
    reconciliation logic be tested with zero external dependencies.
    """

    def __init__(self):
        self._connected = False
        self._alive = True
        self.ticks: Dict[str, TransportTick] = {}
        self.positions: List[TransportPosition] = []
        self.account = TransportAccount(balance=100_000, equity=100_000,
                                        margin_free=100_000)
        self.bars: Dict[str, List[TransportBar]] = {}
        # Programmable order behaviour.
        self.next_result: Optional[TransportOrderResult] = None
        self.placed_orders: List[TransportOrder] = []
        self._next_ticket = 1000

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_alive(self) -> bool:
        return self._connected and self._alive

    def _require(self):
        if not self._connected:
            raise TransportNotConnected("fake transport not connected")

    def get_ticks(self, symbols: List[str]) -> Dict[str, TransportTick]:
        self._require()
        return {s: self.ticks[s] for s in symbols if s in self.ticks}

    def get_positions(self) -> List[TransportPosition]:
        self._require()
        return list(self.positions)

    def get_account(self) -> TransportAccount:
        self._require()
        return self.account

    def get_rates(self, symbol: str, timeframe: str,
                  count: int) -> List[TransportBar]:
        self._require()
        return list(self.bars.get(symbol, []))[-count:]

    def place_order(self, order: TransportOrder) -> TransportOrderResult:
        self._require()
        self.placed_orders.append(order)
        if self.next_result is not None:
            res = self.next_result
            self.next_result = None
            return res
        # Default: fill at the ask (buy) / bid (sell) if we have a tick.
        tick = self.ticks.get(order.symbol)
        if tick is not None:
            fill = tick.ask if order.side == "buy" else tick.bid
        else:
            fill = order.price or 0.0
        ticket = self._next_ticket
        self._next_ticket += 1
        # Reflect the new position into state so reconciliation sees it.
        self.positions.append(TransportPosition(
            ticket=ticket, symbol=order.symbol, side=order.side,
            volume=order.volume, price_open=fill, price_current=fill,
            sl=order.sl or 0.0, tp=order.tp or 0.0,
        ))
        return TransportOrderResult(ok=True, ticket=ticket, retcode=10009,
                                    fill_price=fill, filled_volume=order.volume)


__all__ = [
    "MT5Transport", "FileIPCTransport", "FileIPCConfig", "FakeTransport",
    "TransportTick", "TransportPosition", "TransportAccount", "TransportBar",
    "TransportOrder", "TransportOrderResult",
    "TransportError", "TransportNotConnected", "TransportStale",
]
