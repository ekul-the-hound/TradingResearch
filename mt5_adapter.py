# ==============================================================================
# mt5_adapter.py
# ==============================================================================
# Phase 5. MetaTrader 5 implementation of BaseBroker.
#
# WHY THIS EXISTS: broker_adapter.py ships CCXT (crypto exchanges) and IBKR.
# Prop firms run MT5 or cTrader. Neither existing adapter can place a single
# order on a funded account, so this is a prerequisite for a live attempt, not
# an optimisation.
#
# VERIFICATION STATUS -- READ THIS
# --------------------------------
# The MetaTrader5 package is Windows-only and requires a running, logged-in
# terminal. It could not be exercised where this was written. What that means
# concretely:
#
#   TESTED HERE    volume normalisation, filling-mode negotiation, symbol
#                  resolution, retcode interpretation, side/type translation,
#                  the governor bridge, and every failure path -- all against
#                  an injected fake terminal.
#
#   NOT TESTED     that the real terminal behaves as the docs describe. Field
#                  names on the real objects, the exact filling modes YOUR
#                  broker accepts, and netting-vs-hedging behaviour must be
#                  confirmed against your account before any live use.
#
# Run selftest_against_terminal() on the Windows box to check the second group.
#
# THE FOUR THINGS THAT ACTUALLY BREAK MT5 ORDERS
# ----------------------------------------------
# 1. VOLUME. MT5 trades LOTS, not units, and rejects any volume that is not a
#    multiple of volume_step between volume_min and volume_max. 100_000 units
#    of EURUSD is 1.0 lot; sending 100_000 asks for ten billion currency units.
# 2. FILLING MODE. Brokers support different subsets of FOK / IOC / RETURN.
#    The wrong one returns retcode 10030 with no position opened.
# 3. SYMBOL SELECTION. A symbol absent from Market Watch cannot be traded, and
#    brokers suffix names (EURUSD.raw, EURUSDm, EURUSD_i).
# 4. RETCODES. order_send returns an object even on failure. Only
#    TRADE_RETCODE_DONE means the order actually went through; treating a
#    non-None result as success reports fills that never happened.
# ==============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from broker_adapter import (
    BaseBroker, BrokerBalance, BrokerOrder, BrokerPosition, BrokerTick,
    OrderSide, OrderStatus, OrderType,
)

# MT5 constants, mirrored so this module imports without the package present.
# Values are from the MetaTrader5 Python docs; selftest_against_terminal()
# checks them against the live module rather than trusting them.
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_DONE_PARTIAL = 10010
TRADE_RETCODE_PLACED = 10008

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1

TRADE_ACTION_DEAL = 1

ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

SUCCESS_RETCODES = frozenset({
    TRADE_RETCODE_DONE, TRADE_RETCODE_DONE_PARTIAL, TRADE_RETCODE_PLACED})

# The subset worth naming; anything else is reported with its raw code.
RETCODE_MEANINGS: Dict[int, str] = {
    10004: 'Requote',
    10006: 'Request rejected',
    10007: 'Cancelled by trader',
    10013: 'Invalid request',
    10014: 'Invalid volume',
    10015: 'Invalid price',
    10016: 'Invalid stops',
    10018: 'Market is closed',
    10019: 'Insufficient funds',
    10027: 'Autotrading disabled in the terminal',
    10030: 'Unsupported filling mode',
    10031: 'No connection to the trade server',
}


class MT5Error(RuntimeError):
    """Raised for terminal-level failures that are not order rejections."""


# ==============================================================================
# PURE LOGIC -- fully testable without a terminal
# ==============================================================================

@dataclass
class SymbolSpec:
    """
    The parts of MT5's symbol_info this adapter depends on.

    Extracted into a plain object so the volume and filling logic can be
    tested without constructing a terminal response.
    """
    name: str
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    trade_contract_size: float = 100_000.0
    filling_mode: int = 0
    digits: int = 5
    visible: bool = True

    @classmethod
    def from_mt5(cls, info: Any) -> "SymbolSpec":
        return cls(
            name=getattr(info, 'name', ''),
            volume_min=float(getattr(info, 'volume_min', 0.01)),
            volume_max=float(getattr(info, 'volume_max', 100.0)),
            volume_step=float(getattr(info, 'volume_step', 0.01)),
            trade_contract_size=float(
                getattr(info, 'trade_contract_size', 100_000.0)),
            filling_mode=int(getattr(info, 'filling_mode', 0)),
            digits=int(getattr(info, 'digits', 5)),
            visible=bool(getattr(info, 'visible', True)),
        )


def units_to_lots(units: float, spec: SymbolSpec) -> float:
    """Convert a size in currency units to lots for this symbol."""
    if spec.trade_contract_size <= 0:
        raise MT5Error(
            f"{spec.name} reports trade_contract_size="
            f"{spec.trade_contract_size}; cannot convert units to lots.")
    return units / spec.trade_contract_size


def normalize_volume(lots: float, spec: SymbolSpec) -> Tuple[float, List[str]]:
    """
    Snap a lot size onto the broker's grid.

    Returns (volume, notes). MT5 rejects any volume that is not a whole
    multiple of volume_step within [volume_min, volume_max], so this is not
    cosmetic rounding -- an unsnapped value is retcode 10014 and no trade.

    Rounds DOWN to the step. Rounding up would silently place a larger
    position than the risk calculation asked for, and the whole point of the
    position sizer is that the number it produces is the one that gets traded.
    A size below volume_min becomes 0.0 with a note, never volume_min: opening
    the smallest allowed trade when the model asked for less is the sizer
    being overridden by the plumbing.
    """
    notes: List[str] = []

    if spec.volume_step <= 0:
        raise MT5Error(f"{spec.name} reports volume_step={spec.volume_step}.")
    if lots <= 0:
        return 0.0, ['Requested volume is zero or negative.']

    steps = math.floor(lots / spec.volume_step + 1e-9)
    vol = steps * spec.volume_step
    # Re-round to kill binary representation dust (0.1 + 0.2 territory).
    decimals = max(0, -int(math.floor(math.log10(spec.volume_step))) + 1)
    vol = round(vol, decimals)

    if vol < spec.volume_min:
        notes.append(
            f"Requested {lots:.4f} lots rounds to {vol:.4f}, below "
            f"{spec.name} minimum {spec.volume_min}. Returning 0 rather than "
            f"rounding up to the minimum, which would trade more than asked.")
        return 0.0, notes

    if vol > spec.volume_max:
        notes.append(
            f"Requested {lots:.4f} lots exceeds {spec.name} maximum "
            f"{spec.volume_max}; capped. The remainder was NOT split into a "
            f"second order.")
        vol = spec.volume_max

    if abs(vol - lots) > spec.volume_step * 0.01:
        notes.append(
            f"Volume snapped from {lots:.4f} to {vol:.4f} "
            f"(step {spec.volume_step}).")

    return vol, notes


def choose_filling_mode(spec: SymbolSpec) -> int:
    """
    Pick a filling mode the symbol actually supports.

    filling_mode is a BITMASK of allowed modes, not a single value, and the
    bit positions do not equal the ORDER_FILLING_* constants -- a detail that
    produces retcode 10030 for anyone who assumes they match.

    Preference order is IOC, then FOK, then RETURN. IOC first because a
    partial fill on a market order beats no fill: the governor can size the
    next order down, but it cannot act on a position that was never opened.
    """
    mask = spec.filling_mode
    if mask & 2:        # SYMBOL_FILLING_IOC
        return ORDER_FILLING_IOC
    if mask & 1:        # SYMBOL_FILLING_FOK
        return ORDER_FILLING_FOK
    return ORDER_FILLING_RETURN


def resolve_symbol(requested: str, available: List[str]) -> Optional[str]:
    """
    Match a plain symbol name against a broker's decorated list.

    Brokers append suffixes (EURUSD.raw, EURUSDm, EURUSD_i) and some strip
    separators. An exact match always wins; otherwise the shortest name whose
    alphanumeric core matches, so EURUSD prefers EURUSD.raw over EURUSDCHF.
    """
    if requested in available:
        return requested

    def core(s: str) -> str:
        return ''.join(c for c in s if c.isalnum()).upper()

    want = core(requested)
    hits = [a for a in available if core(a).startswith(want)]
    exact_core = [a for a in hits if core(a) == want]
    pool = exact_core or hits
    return min(pool, key=len) if pool else None


def describe_retcode(retcode: Optional[int]) -> str:
    if retcode is None:
        return 'no retcode returned'
    name = RETCODE_MEANINGS.get(retcode)
    return f"{retcode} ({name})" if name else f"{retcode}"


def order_succeeded(result: Any) -> bool:
    """
    order_send returns an object on failure too.

    Only specific retcodes mean the order reached the market. Truthiness of
    the result is not the test, and using it reports fills that never
    happened -- which then propagates into the governor as a position that
    does not exist.
    """
    if result is None:
        return False
    return getattr(result, 'retcode', None) in SUCCESS_RETCODES


# ==============================================================================
# ADAPTER
# ==============================================================================

class MT5Broker(BaseBroker):
    """
    MetaTrader 5 adapter.

    The terminal module is injected rather than imported at module scope, so
    the translation logic is testable with a fake and this file imports fine
    on a machine with no MetaTrader5 package.
    """

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        terminal_path: Optional[str] = None,
        deviation_points: int = 20,
        magic: int = 20260805,
        mt5_module: Any = None,
    ):
        super().__init__(name='mt5')
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        self.deviation_points = deviation_points
        self.magic = magic
        self._mt5 = mt5_module
        self._specs: Dict[str, SymbolSpec] = {}
        self._symbol_map: Dict[str, str] = {}
        self.last_notes: List[str] = []

    # ------------------------------------------------------------------
    @property
    def mt5(self) -> Any:
        if self._mt5 is None:
            try:
                import MetaTrader5 as _mt5      # type: ignore[import]
            except ImportError as e:
                raise MT5Error(
                    "MetaTrader5 package not available. It is Windows-only "
                    "and requires a running terminal. "
                    f"({e})") from e
            self._mt5 = _mt5
        return self._mt5

    def connect(self) -> bool:
        kwargs: Dict[str, Any] = {}
        if self.terminal_path:
            kwargs['path'] = self.terminal_path
        if self.login is not None:
            kwargs.update(login=self.login, password=self.password,
                          server=self.server)
        ok = bool(self.mt5.initialize(**kwargs))
        if not ok:
            err = self._last_error()
            self.is_connected = False
            raise MT5Error(f"mt5.initialize failed: {err}")
        self.is_connected = True
        return True

    def disconnect(self):
        try:
            self.mt5.shutdown()
        finally:
            self.is_connected = False

    def _last_error(self) -> str:
        try:
            return str(self.mt5.last_error())
        except Exception:                                 # pragma: no cover
            return 'unknown'

    # ------------------------------------------------------------------
    # SYMBOLS
    # ------------------------------------------------------------------
    def _available_symbols(self) -> List[str]:
        syms = self.mt5.symbols_get()
        return [getattr(s, 'name', '') for s in (syms or [])]

    def broker_symbol(self, symbol: str) -> str:
        """Map a plain name to this broker's decorated one, once."""
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]
        resolved = resolve_symbol(symbol, self._available_symbols())
        if resolved is None:
            raise MT5Error(
                f"Symbol {symbol!r} not offered by this broker. Check the "
                f"suffix convention in Market Watch.")
        self._symbol_map[symbol] = resolved
        return resolved

    def spec(self, symbol: str) -> SymbolSpec:
        name = self.broker_symbol(symbol)
        if name in self._specs:
            return self._specs[name]
        info = self.mt5.symbol_info(name)
        if info is None:
            raise MT5Error(f"symbol_info({name!r}) returned None.")
        spec = SymbolSpec.from_mt5(info)
        if not spec.visible:
            # Not in Market Watch means not tradeable, and the failure would
            # otherwise surface later as an opaque order rejection.
            if not self.mt5.symbol_select(name, True):
                raise MT5Error(
                    f"symbol_select({name!r}) failed; it cannot be traded "
                    f"while hidden from Market Watch.")
            spec.visible = True
        self._specs[name] = spec
        return spec

    # ------------------------------------------------------------------
    # READS
    # ------------------------------------------------------------------
    def get_tick(self, symbol: str) -> Optional[BrokerTick]:
        name = self.broker_symbol(symbol)
        t = self.mt5.symbol_info_tick(name)
        if t is None:
            return None
        bid = float(getattr(t, 'bid', 0.0))
        ask = float(getattr(t, 'ask', 0.0))
        # MT5 reports last=0 for FX -- there is no centralised last trade in
        # a decentralised market. Falling back to the mid is more useful
        # downstream than a hard zero, which reads as a real price of nothing.
        last = float(getattr(t, 'last', 0.0)) or (bid + ask) / 2.0
        return BrokerTick(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            volume_24h=float(getattr(t, 'volume', 0.0)),
            timestamp=datetime.utcnow().isoformat(),
        )

    def get_balance(self) -> BrokerBalance:
        info = self.mt5.account_info()
        if info is None:
            raise MT5Error("account_info() returned None; not logged in?")
        return BrokerBalance(
            total_equity=float(getattr(info, 'equity', 0.0)),
            free_margin=float(getattr(info, 'margin_free', 0.0)),
            used_margin=float(getattr(info, 'margin', 0.0)),
            unrealized_pnl=float(getattr(info, 'profit', 0.0)),
            currency=str(getattr(info, 'currency', 'USD')),
            timestamp=datetime.utcnow().isoformat(),
        )

    def get_positions(self) -> List[BrokerPosition]:
        raw = self.mt5.positions_get()
        return [self._to_position(p) for p in (raw or [])]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        name = self.broker_symbol(symbol)
        raw = self.mt5.positions_get(symbol=name)
        if not raw:
            return None
        # A hedging account can hold several positions per symbol; the net is
        # what the risk rules care about.
        return self._net_position(symbol, [self._to_position(p) for p in raw])

    def _to_position(self, p: Any) -> BrokerPosition:
        size = float(getattr(p, 'volume', 0.0))
        is_long = int(getattr(p, 'type', POSITION_TYPE_BUY)) == POSITION_TYPE_BUY
        return BrokerPosition(
            symbol=str(getattr(p, 'symbol', '')),
            side='long' if is_long else 'short',
            size=size,
            entry_price=float(getattr(p, 'price_open', 0.0)),
            current_price=float(getattr(p, 'price_current', 0.0)),
            unrealized_pnl=float(getattr(p, 'profit', 0.0)),
            realized_pnl=0.0,
        )

    @staticmethod
    def _net_position(symbol: str,
                      parts: List[BrokerPosition]) -> Optional[BrokerPosition]:
        if not parts:
            return None
        net = sum(p.size if p.side == 'long' else -p.size for p in parts)
        gross = sum(p.size for p in parts) or 1.0
        vwap = sum(p.entry_price * p.size for p in parts) / gross
        return BrokerPosition(
            symbol=symbol,
            side='long' if net > 0 else 'short' if net < 0 else 'flat',
            size=abs(net),
            entry_price=vwap,
            current_price=parts[-1].current_price,
            unrealized_pnl=sum(p.unrealized_pnl for p in parts),
            realized_pnl=0.0,
        )

    # ------------------------------------------------------------------
    # ORDERS
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
        """
        Place a market order. `size` is in UNITS, converted to lots here.

        Only market orders are implemented. A pending-order request returns a
        REJECTED BrokerOrder rather than quietly placing a market order in its
        place -- substituting an order type the caller did not ask for is how
        a limit strategy ends up chasing price.
        """
        self.last_notes = []
        ts = datetime.utcnow().isoformat()

        if order_type != 'market':
            return self._rejected(
                symbol, side, size, ts,
                f"order_type={order_type!r} is not implemented by the MT5 "
                f"adapter; only market orders are supported.")

        if side not in ('buy', 'sell'):
            return self._rejected(symbol, side, size, ts,
                                  f"Unknown side {side!r}.")

        try:
            spec = self.spec(symbol)
            name = self.broker_symbol(symbol)
        except MT5Error as e:
            return self._rejected(symbol, side, size, ts, str(e))

        lots, notes = normalize_volume(units_to_lots(size, spec), spec)
        self.last_notes = notes
        if lots <= 0:
            return self._rejected(
                symbol, side, size, ts,
                'Normalised volume is zero. ' + ' '.join(notes))

        tick = self.mt5.symbol_info_tick(name)
        if tick is None:
            return self._rejected(symbol, side, size, ts,
                                  f"No tick for {name}; market may be closed.")

        request = {
            'action': TRADE_ACTION_DEAL,
            'symbol': name,
            'volume': lots,
            'type': ORDER_TYPE_BUY if side == 'buy' else ORDER_TYPE_SELL,
            'price': float(getattr(tick, 'ask' if side == 'buy' else 'bid', 0.0)),
            'deviation': self.deviation_points,
            'magic': self.magic,
            'comment': 'tradinglab',
            'type_filling': choose_filling_mode(spec),
        }

        result = self.mt5.order_send(request)

        if not order_succeeded(result):
            code = getattr(result, 'retcode', None) if result else None
            comment = getattr(result, 'comment', '') if result else ''
            return self._rejected(
                symbol, side, size, ts,
                f"order_send failed: retcode {describe_retcode(code)}"
                + (f" -- {comment}" if comment else ''),
                raw=self._raw(result))

        filled = float(getattr(result, 'volume', lots))
        order = BrokerOrder(
            order_id=str(getattr(result, 'order', '')),
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=filled,
            status=OrderStatus.FILLED,
            fill_price=float(getattr(result, 'price', 0.0)),
            filled_size=filled,
            timestamp=ts,
            broker_ref=str(getattr(result, 'deal', '')),
            raw=self._raw(result),
        )
        self._order_history.append(order)
        return order

    def _rejected(self, symbol: str, side: str, size: float, ts: str,
                  why: str, raw: Optional[Dict] = None) -> BrokerOrder:
        order = BrokerOrder(
            order_id='',
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=size,
            status=OrderStatus.REJECTED,
            timestamp=ts,
            raw={'error': why, **(raw or {})},
        )
        self._order_history.append(order)
        return order

    @staticmethod
    def _raw(result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        try:
            return dict(result._asdict())
        except Exception:
            return {k: getattr(result, k) for k in
                    ('retcode', 'order', 'deal', 'volume', 'price', 'comment')
                    if hasattr(result, k)}

    def cancel_order(self, order_id: str) -> bool:
        """
        Market orders fill or reject immediately; there is nothing pending to
        cancel. Returns False rather than True so a caller cannot read a
        no-op as a successful cancellation.
        """
        return False

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        for o in reversed(self._order_history):
            if o.order_id == order_id:
                return o
        return None

    # ------------------------------------------------------------------
    # GOVERNOR BRIDGE
    # ------------------------------------------------------------------
    def to_account_state(self, initial_balance: float):
        """
        Snapshot for live_governor.LiveGovernor.observe().

        equity carries floating P&L and balance does not, which is the
        distinction the firm's daily rule turns on.
        """
        from live_governor import AccountState

        info = self.mt5.account_info()
        if info is None:
            raise MT5Error("account_info() returned None; cannot build state.")
        positions = self.get_positions()
        return AccountState(
            timestamp=datetime.utcnow(),
            balance=float(getattr(info, 'balance', 0.0)),
            equity=float(getattr(info, 'equity', 0.0)),
            initial_balance=float(initial_balance),
            open_positions=len(positions),
            symbol_exposure={p.symbol: p.size for p in positions},
        )

    def flatten_all(self) -> List[BrokerOrder]:
        """Close every open position. What FLATTEN verdicts act on."""
        out = []
        for pos in self.get_positions():
            if pos.size <= 0 or pos.side == 'flat':
                continue
            spec = self.spec(pos.symbol)
            out.append(self.submit_order(
                side='sell' if pos.side == 'long' else 'buy',
                symbol=pos.symbol,
                size=pos.size * spec.trade_contract_size))
        return out


# ==============================================================================
# TERMINAL SELF-TEST -- run this on the Windows box
# ==============================================================================

def selftest_against_terminal(symbol: str = 'EURUSD',
                              terminal_path: Optional[str] = None,
                              login: Optional[int] = None,
                              password: Optional[str] = None,
                              server: Optional[str] = None) -> Dict[str, Any]:
    """
    Check the assumptions this module makes against a real terminal.

    Read-only: places no orders. Every item here is something that was
    ASSUMED from documentation and could not be verified where this was
    written. Run it before trusting the adapter with an account.
    """
    report: Dict[str, Any] = {'checks': [], 'ok': True}

    def check(label: str, passed: bool, detail: str = ''):
        report['checks'].append(
            {'check': label, 'passed': bool(passed), 'detail': detail})
        if not passed:
            report['ok'] = False

    try:
        # Bound to Any deliberately. The MetaTrader5 package ships no type
        # stubs, so a static checker sees a module with almost no known
        # attributes and flags every real call -- terminal_info, account_info,
        # symbol_info -- as unknown. Any is the accurate type for an
        # unstubbed C extension; the alternative is a `type: ignore` on each
        # line, which suppresses genuine typos alongside the false positives.
        import MetaTrader5                    # type: ignore[import]
        mt5: Any = MetaTrader5
    except ImportError as e:
        import platform
        import sys as _sys
        hint = (
            'pip install MetaTrader5 (inside the conda env you run from). '
            'If pip reports "no matching distribution", the cause is the '
            'PYTHON VERSION, not the package name -- MetaTrader5 ships only '
            'for the versions MetaQuotes builds wheels for. '
            f'You are on Python {_sys.version_info.major}.'
            f'{_sys.version_info.minor} ({platform.machine()}) on '
            f'{platform.system()}.')
        if platform.system() != 'Windows':
            hint += (' MetaTrader5 is Windows-only; there is no Linux or '
                     'macOS build.')
        check('MetaTrader5 importable', False, f'{e}. {hint}')
        return report
    check('MetaTrader5 importable', True)

    for const, expected in (('TRADE_RETCODE_DONE', TRADE_RETCODE_DONE),
                            ('ORDER_TYPE_BUY', ORDER_TYPE_BUY),
                            ('ORDER_TYPE_SELL', ORDER_TYPE_SELL),
                            ('TRADE_ACTION_DEAL', TRADE_ACTION_DEAL),
                            ('ORDER_FILLING_IOC', ORDER_FILLING_IOC),
                            ('ORDER_FILLING_FOK', ORDER_FILLING_FOK)):
        actual = getattr(mt5, const, None)
        check(f'constant {const}', actual == expected,
              f'mirrored {expected}, terminal says {actual}')

    # terminal_path matters when several MT5 builds are installed, which is
    # normal once a prop firm ships its own. Auto-discovery finds one of them,
    # not necessarily the one holding the account you mean to trade.
    broker = MT5Broker(mt5_module=mt5, terminal_path=terminal_path,
                       login=login, password=password, server=server)
    try:
        broker.connect()
        check('initialize', True)
    except MT5Error as e:
        detail = str(e)
        if 'not found' in detail or '-10003' in detail:
            detail += (' -- the MetaTrader5 PACKAGE is installed but the'
                       ' desktop TERMINAL is not, or is somewhere'
                       ' auto-discovery does not look. Install MT5 and log'
                       ' in once, then rerun; if it is installed under a'
                       ' broker-specific folder, pass'
                       ' terminal_path=r"C:\\...\\terminal64.exe".')
        check('initialize', False, detail)
        return report

    try:
        term = mt5.terminal_info()
        check('terminal_info returns data', term is not None)
        if term is not None:
            # Two separate switches, both of which reject orders with
            # retcode 10027 and neither of which is visible from a price
            # feed working correctly.
            check('Algo Trading enabled in terminal',
                  bool(getattr(term, 'trade_allowed', False)),
                  'Toolbar Algo Trading button, or Tools > Options > '
                  'Expert Advisors > Allow algorithmic trading.')
            check('Python API not blocked',
                  not bool(getattr(term, 'trade_api_disabled', False)),
                  'Tools > Options > Expert Advisors > "Disable automatic '
                  'trading via external Python API" must be UNCHECKED. This '
                  'is separate from the Algo Trading button and blocks '
                  'Python specifically while MQL programs keep trading.')

        info = mt5.account_info()
        check('account_info returns data', info is not None)
        if info is not None:
            demo = getattr(info, 'trade_mode', None)
            check('account is a demo/contest account', demo != 0,
                  f'trade_mode={demo} (0 = REAL money). Verify the adapter '
                  f'on a demo before pointing it at a funded account.')
            for field_name in ('balance', 'equity', 'margin_free', 'currency'):
                check(f'account_info.{field_name}', hasattr(info, field_name))

        resolved = resolve_symbol(symbol, broker._available_symbols())
        check(f'resolve {symbol}', resolved is not None, str(resolved))

        if resolved:
            raw = mt5.symbol_info(resolved)
            check('symbol_info returns data', raw is not None)
            if raw is not None:
                spec = SymbolSpec.from_mt5(raw)
                check('volume_step positive', spec.volume_step > 0,
                      f'{spec.volume_step}')
                check('contract size positive',
                      spec.trade_contract_size > 0,
                      f'{spec.trade_contract_size}')
                check('filling_mode readable', spec.filling_mode is not None,
                      f'mask={spec.filling_mode} -> '
                      f'{choose_filling_mode(spec)}')
                lots, notes = normalize_volume(
                    units_to_lots(100_000.0, spec), spec)
                check('100k units normalises', lots > 0,
                      f'{lots} lots; {notes}')
    finally:
        broker.disconnect()

    return report