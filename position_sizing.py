# ==============================================================================
# position_sizing.py
# ==============================================================================
# Phase 0, Item 4 -- hardcoded 2% stop in execution_engine._calculate_size.
#
# WHAT WAS WRONG (worse than the roadmap entry said)
# --------------------------------------------------
#     risk_pct      = 0.02
#     risk_amount   = self.trader.equity * risk_pct
#     stop_distance = price * 0.02          # "Assume 2% stop loss"
#     size          = risk_amount / stop_distance
#
# The two 2%s cancel:
#
#     size = (equity * 0.02) / (price * 0.02) = equity / price
#
# So the "risk-based" sizing is algebraically inert -- it reduces to 1x equity
# notional and carries no risk parameter at all. Then two caps are applied, and
# the 20%-of-equity leverage cap is always 5x smaller than equity/price, so it
# always wins. Net result: every position was exactly 20% of equity notional,
# on every instrument, forever. Changing risk_pct would have done nothing.
#
# Worked example at $100k equity:
#     EUR-USD  px 1.10       risk-size 90,909      final 18,181.8   (20% cap)
#     USD-JPY  px 150.00     risk-size 667         final 133.3      (20% cap)
#     BTC-USD  px 60,000     risk-size 1.67        final 0.333      (20% cap)
#
# WHY THAT MATTERS FOR A PROP CHALLENGE
# -------------------------------------
# Notional-equalised sizing is volatility-blind. It gives EUR-USD and BTC-USD
# the same notional exposure even though a 2% move is a once-a-year event in
# one and a normal Tuesday in the other. Since FTMO's binding constraint is a
# 5% daily loss, you cannot reason about how many losing trades fit inside the
# daily budget unless size is tied to the distance to the actual stop.
#
# THE FIX
# -------
# Size from real risk: risk_amount / distance_to_stop, where the stop comes
# from the strategy. Resolution order, most trustworthy first:
#
#   1. explicit stop_distance passed by the caller
#   2. explicit stop_price passed by the caller  -> abs(price - stop_price)
#   3. volatility-scaled fallback from observed returns (ATR-like)
#   4. asset-class default -- flagged, never silent
#
# A flat "2% of price" default is deliberately NOT used at any level: 2% of
# price means completely different risk across asset classes, which is the
# defect being removed.
#
# Every result reports which input determined the stop and which constraint
# bound the final size, so the caller can tell real risk sizing from a cap.
# ==============================================================================

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

# Stop source, in descending order of trustworthiness
STOP_EXPLICIT_DISTANCE = 'explicit_distance'
STOP_EXPLICIT_PRICE = 'explicit_stop_price'
STOP_VOLATILITY = 'volatility_estimate'
STOP_ASSET_DEFAULT = 'asset_class_default'

# Which constraint produced the final size
BOUND_RISK = 'risk_sizing'
BOUND_MAX_SIZE = 'max_position_size'
BOUND_LEVERAGE = 'leverage_cap'
BOUND_ZERO = 'zero'

# Fallback stop distances as a fraction of price, by asset class. These are
# rough volatility-aware defaults, used only when the strategy supplies no stop
# and there is not yet enough price history. They exist so the number is not
# silently wrong; they are not a substitute for a real stop.
ASSET_DEFAULT_STOP_FRAC = {
    'fx': 0.0035,          # ~35 pips on a 1.10 pair
    'crypto': 0.025,       # crypto routinely moves 2-3% intraday
    'indices': 0.008,
    'commodities': 0.012,
}

DEFAULT_RISK_PER_TRADE = 0.005   # 0.5% of equity -- see note in size_position()
VOL_LOOKBACK = 100               # returns retained per symbol
VOL_MIN_SAMPLES = 20             # below this, volatility estimate is untrusted
VOL_STOP_MULTIPLE = 2.0          # stop at 2 sigma of per-bar return


@dataclass
class SizingResult:
    """A position size plus the full story of how it was reached."""
    size: float
    stop_distance: float
    stop_source: str
    bound_by: str
    risk_amount: float
    risk_pct_of_equity: float
    notional: float
    notional_pct_of_equity: float
    warnings: list = field(default_factory=list)

    @property
    def is_real_risk_sizing(self) -> bool:
        """True only if a genuine stop drove the size and no cap overrode it."""
        return (self.bound_by == BOUND_RISK
                and self.stop_source in (STOP_EXPLICIT_DISTANCE, STOP_EXPLICIT_PRICE))

    def describe(self) -> str:
        return (f"size={self.size:,.4f} stop={self.stop_distance:.6f} "
                f"({self.stop_source}) risk={self.risk_pct_of_equity * 100:.2f}% "
                f"notional={self.notional_pct_of_equity * 100:.1f}% "
                f"bound_by={self.bound_by}")


# ==============================================================================
# VOLATILITY TRACKER
# ==============================================================================

class VolatilityTracker:
    """
    Rolling per-bar return volatility per symbol.

    Fed from the same price updates the engine already receives, so it needs no
    new data source. Used only to build a fallback stop when the strategy does
    not supply one.
    """

    def __init__(self, lookback: int = VOL_LOOKBACK):
        self.lookback = lookback
        self._returns: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=lookback))
        self._last_price: Dict[str, float] = {}

    def update(self, symbol: str, price: Optional[float]) -> None:
        if price is None or price <= 0:
            return
        prev = self._last_price.get(symbol)
        if prev and prev > 0:
            self._returns[symbol].append((price - prev) / prev)
        self._last_price[symbol] = price

    def sigma(self, symbol: str) -> Optional[float]:
        """Stdev of recent returns, or None if there is not enough history."""
        r = self._returns.get(symbol)
        if not r or len(r) < VOL_MIN_SAMPLES:
            return None
        n = len(r)
        mean = sum(r) / n
        var = sum((x - mean) ** 2 for x in r) / (n - 1)
        s = math.sqrt(var)
        return s if s > 0 else None

    def reset(self, symbol: Optional[str] = None) -> None:
        if symbol is None:
            self._returns.clear()
            self._last_price.clear()
        else:
            self._returns.pop(symbol, None)
            self._last_price.pop(symbol, None)


# ==============================================================================
# ASSET CLASS
# ==============================================================================

def detect_asset_class(symbol: str) -> str:
    """Reuses ftmo_compliance's classifier when importable, else a local rule."""
    try:
        from ftmo_compliance import detect_asset_class as _d
        return _d(symbol or '').value
    except Exception:
        pass
    s = (symbol or '').upper()
    if any(x in s for x in ('BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'LTC', 'USDT')):
        return 'crypto'
    if any(x in s for x in ('SPX', 'NDX', 'DJI', 'GSPC', 'IXIC')):
        return 'indices'
    if any(x in s for x in ('GOLD', 'XAU', 'OIL', 'WTI', 'XAG', 'SILVER')):
        return 'commodities'
    return 'fx'


# ==============================================================================
# SIZING
# ==============================================================================

def resolve_stop_distance(
    symbol: str,
    price: float,
    stop_distance: Optional[float] = None,
    stop_price: Optional[float] = None,
    vol_tracker: Optional[VolatilityTracker] = None,
) -> tuple:
    """
    Returns (stop_distance, source, warnings).

    Never returns zero or a negative distance -- those would produce infinite
    size, which is exactly the class of bug this module exists to prevent.
    """
    warnings = []

    if stop_distance is not None and stop_distance > 0:
        return float(stop_distance), STOP_EXPLICIT_DISTANCE, warnings

    if stop_price is not None and stop_price > 0:
        d = abs(price - stop_price)
        if d > 0:
            return d, STOP_EXPLICIT_PRICE, warnings
        warnings.append("stop_price equals entry price; ignoring it")

    if vol_tracker is not None:
        sigma = vol_tracker.sigma(symbol)
        if sigma:
            d = price * sigma * VOL_STOP_MULTIPLE
            if d > 0:
                warnings.append(
                    f"No stop supplied by strategy; using {VOL_STOP_MULTIPLE}-sigma "
                    f"volatility estimate ({sigma * 100:.3f}% per bar). "
                    f"Pass stop_price for accurate risk sizing."
                )
                return d, STOP_VOLATILITY, warnings

    frac = ASSET_DEFAULT_STOP_FRAC.get(detect_asset_class(symbol),
                                       ASSET_DEFAULT_STOP_FRAC['fx'])
    warnings.append(
        f"No stop and insufficient price history for {symbol}; falling back to "
        f"the {detect_asset_class(symbol)} default of {frac * 100:.2f}% of price. "
        f"This is a placeholder, not a measured risk."
    )
    return price * frac, STOP_ASSET_DEFAULT, warnings


def size_position(
    symbol: str,
    price: float,
    equity: float,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    stop_distance: Optional[float] = None,
    stop_price: Optional[float] = None,
    max_position_size: Optional[float] = None,
    max_leverage_pct: Optional[float] = 0.20,
    vol_tracker: Optional[VolatilityTracker] = None,
) -> SizingResult:
    """
    Risk-based position size.

        size = (equity * risk_per_trade) / distance_to_stop

    On risk_per_trade: the old code nominally said 2%, but never applied it.
    The default here is 0.5% because FTMO's binding constraint is a 5% daily
    loss -- at 2% per trade, three consecutive losers end a challenge. Set it
    explicitly rather than inheriting this default.

    Returns a SizingResult. Check .is_real_risk_sizing before treating the
    number as a genuine risk-derived size; a cap or a fallback stop means it
    is not.
    """
    warnings = []

    if price is None or price <= 0:
        return SizingResult(0.0, 0.0, STOP_ASSET_DEFAULT, BOUND_ZERO, 0.0, 0.0, 0.0, 0.0,
                            [f"Non-positive price for {symbol}; refusing to size"])
    if equity is None or equity <= 0:
        return SizingResult(0.0, 0.0, STOP_ASSET_DEFAULT, BOUND_ZERO, 0.0, 0.0, 0.0, 0.0,
                            ["Non-positive equity; refusing to size"])
    if risk_per_trade <= 0:
        return SizingResult(0.0, 0.0, STOP_ASSET_DEFAULT, BOUND_ZERO, 0.0, 0.0, 0.0, 0.0,
                            ["risk_per_trade must be > 0"])

    dist, source, w = resolve_stop_distance(
        symbol, price, stop_distance, stop_price, vol_tracker)
    warnings.extend(w)

    risk_amount = equity * risk_per_trade
    raw_size = risk_amount / dist
    bound_by = BOUND_RISK
    size = raw_size

    if max_position_size is not None and size > max_position_size:
        size = float(max_position_size)
        bound_by = BOUND_MAX_SIZE

    if max_leverage_pct and max_leverage_pct > 0:
        max_by_value = (equity * max_leverage_pct) / price
        if size > max_by_value:
            size = max_by_value
            bound_by = BOUND_LEVERAGE

    if size <= 0:
        bound_by = BOUND_ZERO

    if bound_by in (BOUND_MAX_SIZE, BOUND_LEVERAGE):
        warnings.append(
            f"Size capped by {bound_by}: risk sizing wanted {raw_size:,.4f}, "
            f"capped to {size:,.4f}. Actual risk at stop is now "
            f"{(size * dist) / equity * 100:.3f}% of equity, not "
            f"{risk_per_trade * 100:.2f}%."
        )

    notional = size * price
    return SizingResult(
        size=size,
        stop_distance=dist,
        stop_source=source,
        bound_by=bound_by,
        risk_amount=size * dist,
        risk_pct_of_equity=(size * dist) / equity,
        notional=notional,
        notional_pct_of_equity=notional / equity,
        warnings=warnings,
    )