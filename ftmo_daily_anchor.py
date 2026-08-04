# ==============================================================================
# ftmo_daily_anchor.py
# ==============================================================================
# Phase 0, Item 2 -- daily-loss anchor divergence.
#
# WHAT WAS WRONG
# --------------
# ftmo_compliance._calculate_daily_stats() measured the daily loss as:
#
#     start_equity = group['equity'].iloc[0]        # first EVENT of the day
#     daily_loss   = start_equity - min_equity
#
# FTMO's published rule is different in two ways that both matter:
#
#   1. ANCHOR TIME. FTMO recalculates the limit at midnight CE(S)T. The old
#      code anchored on the first *event* inside the Prague day. Checkpoints
#      were injected to cover event-less days, but at pd.Timestamp.normalize(),
#      which is 00:00 UTC -- while the grouping was by Prague date. Prague
#      midnight is 23:00 UTC in CET and 22:00 UTC in CEST, so the anchor sat
#      1-2 hours inside the day and the offset changed twice a year with
#      European DST. Everything happening in the first hours of the Prague day
#      (the Sydney/Tokyo session) was invisible to the daily-loss check.
#
#   2. ANCHOR QUANTITY. FTMO anchors on BALANCE at midnight, not equity:
#         "The Maximum Daily Loss Limit is recalculated every midnight CE(S)T:
#          Account balance at midnight CE(S)T of the previous day
#          - 5% of the Initial Simulated Capital"
#         "Intraday changes resulting from open positions do not affect the
#          Maximum Daily Loss Limit."
#      Only the measured side is equity (balance + floating P/L). The old code
#      used equity on BOTH sides. Carrying a floating LOSS across midnight
#      therefore lowered the anchor, shrinking the measured daily loss and
#      hiding real breaches -- an error in the optimistic direction.
#
# WHAT THIS MODULE DOES
# ---------------------
#   prague_midnight_checkpoints()  -- checkpoint timestamps at true Prague
#                                     midnight, DST-correct, as naive UTC.
#   calculate_daily_stats_anchored() -- drop-in replacement for the daily stats
#                                     computation, using the balance anchor.
#
# INPUT CONTRACT: equity-curve timestamps are naive UTC. That is guaranteed
# upstream by forex_data_processor.py (Phase 0 item 1). If item 1 has not been
# applied, this module will faithfully compute the daily loss against the wrong
# instants -- the two fixes are complementary, not alternatives.
#
# FIRST DAY: FTMO uses Initial Simulated Capital as the day-1 anchor.
#
# NOTE ON SOURCES: FTMO's own documentation (academy.ftmo.com, the Trading
# Objectives page, and the FTMO/OANDA FAQ) all say the limit is anchored on
# BALANCE at midnight. Several third-party guides claim "balance or equity,
# whichever is higher". Those disagree with the primary source. The default
# here follows FTMO. Set anchor_mode='max_balance_equity' to model the
# third-party reading if your firm turns out to work that way.
# ==============================================================================

from datetime import datetime, timedelta

import pandas as pd
import pytz

PRAGUE_TZ = pytz.timezone('Europe/Prague')
UTC_TZ = pytz.UTC

ANCHOR_BALANCE = 'balance'                    # FTMO documented behaviour
ANCHOR_MAX = 'max_balance_equity'             # third-party reading
VALID_ANCHOR_MODES = (ANCHOR_BALANCE, ANCHOR_MAX)


# ==============================================================================
# TIMESTAMP HELPERS
# ==============================================================================

def _to_naive_utc(ts) -> datetime:
    """Coerce a timestamp to a naive-UTC datetime."""
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts.to_pydatetime()


def prague_date_of(ts) -> 'datetime.date':
    """Prague calendar date for a naive-UTC timestamp."""
    naive = _to_naive_utc(ts)
    return UTC_TZ.localize(naive).astimezone(PRAGUE_TZ).date()


def prague_midnight_utc(d) -> datetime:
    """
    Naive-UTC instant of 00:00 Prague on calendar date d.

    Prague midnight always exists -- European DST switches at 02:00/03:00 local
    -- so there is no nonexistent-time case to handle here.
    """
    local = PRAGUE_TZ.localize(datetime(d.year, d.month, d.day, 0, 0, 0))
    return local.astimezone(UTC_TZ).replace(tzinfo=None)


def prague_midnight_checkpoints(first_ts, last_ts):
    """
    Every Prague midnight strictly inside (first_ts, last_ts], as naive UTC.

    Replaces the old pd.Timestamp.normalize() walk, which produced 00:00 UTC
    and therefore landed 1-2 hours into the Prague day depending on season.
    """
    first = _to_naive_utc(first_ts)
    last = _to_naive_utc(last_ts)
    if last <= first:
        return []

    d = prague_date_of(first)
    out = []
    # Walk forward a day at a time; cheap even over multi-year backtests.
    for _ in range(20000):  # ~54 years, hard stop against pathological input
        d = d + timedelta(days=1)
        m = prague_midnight_utc(d)
        if m > last:
            break
        if m > first:
            out.append(m)
    return out


# ==============================================================================
# DAILY STATS WITH THE CORRECT ANCHOR
# ==============================================================================

def calculate_daily_stats_anchored(
    equity_curve: pd.DataFrame,
    initial_balance: float,
    max_daily_loss_pct: float = 0.05,
    anchor_mode: str = ANCHOR_BALANCE,
) -> pd.DataFrame:
    """
    Per-Prague-day loss statistics anchored the way FTMO anchors them.

    For each Prague trading day D:
        anchor      = balance as of 00:00 Prague on D   (day 1: initial capital)
        limit       = anchor - max_daily_loss_pct * initial_balance
        daily_loss  = anchor - min(equity during D)
        breached    = min(equity during D) < limit

    Args:
        equity_curve: DataFrame with 'timestamp' (naive UTC), 'equity',
                      and 'balance' columns.
        initial_balance: Initial Simulated Capital.
        max_daily_loss_pct: fraction of initial capital, default 0.05.
        anchor_mode: 'balance' (FTMO) or 'max_balance_equity' (third-party).

    Returns:
        DataFrame preserving the original column contract
        (date, start_equity, end_equity, min_equity, max_equity, daily_pnl,
         daily_loss_from_start, daily_loss_pct) plus the new anchor columns
        (anchor_balance, daily_loss_limit, breached, anchor_source).
    """
    if anchor_mode not in VALID_ANCHOR_MODES:
        raise ValueError(f"anchor_mode must be one of {VALID_ANCHOR_MODES}, got {anchor_mode!r}")

    if equity_curve is None or equity_curve.empty:
        return pd.DataFrame()

    df = equity_curve.copy()

    if 'balance' not in df.columns:
        # Without a balance column we cannot separate closed from floating.
        # Fail loudly rather than silently reverting to the equity anchor.
        raise KeyError(
            "equity_curve has no 'balance' column; the FTMO daily anchor is "
            "balance-based and cannot be computed from equity alone."
        )

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if getattr(df['timestamp'].dt, 'tz', None) is not None:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['prague_date'] = df['timestamp'].apply(prague_date_of)

    daily_stats = []
    max_loss_amount = max_daily_loss_pct * initial_balance
    first_date = df['prague_date'].iloc[0]

    for date, group in df.groupby('prague_date', sort=True):
        start_equity = group['equity'].iloc[0]
        end_equity = group['equity'].iloc[-1]
        min_equity = group['equity'].min()
        max_equity = group['equity'].max()

        # ---- ANCHOR ---------------------------------------------------------
        if date == first_date:
            # FTMO: day 1 uses Initial Simulated Capital, not the equity after
            # the first entry's fees.
            anchor = float(initial_balance)
            anchor_source = 'initial_capital'
        else:
            midnight = prague_midnight_utc(date)
            prior = df[df['timestamp'] <= midnight]
            if prior.empty:
                anchor = float(initial_balance)
                anchor_source = 'initial_capital'
            else:
                row = prior.iloc[-1]
                if anchor_mode == ANCHOR_MAX:
                    anchor = float(max(row['balance'], row['equity']))
                    anchor_source = 'max(balance,equity)@prague_midnight'
                else:
                    anchor = float(row['balance'])
                    anchor_source = 'balance@prague_midnight'
        # ---------------------------------------------------------------------

        limit = anchor - max_loss_amount
        daily_loss = anchor - min_equity
        daily_loss_pct = daily_loss / initial_balance * 100

        daily_stats.append({
            'date': date,
            'start_equity': start_equity,
            'end_equity': end_equity,
            'min_equity': min_equity,
            'max_equity': max_equity,
            'daily_pnl': end_equity - start_equity,
            'daily_loss_from_start': daily_loss,
            'daily_loss_pct': daily_loss_pct,
            # New diagnostic columns
            'anchor_balance': anchor,
            'daily_loss_limit': limit,
            'breached': bool(min_equity < limit),
            'anchor_source': anchor_source,
        })

    return pd.DataFrame(daily_stats)


# ==============================================================================
# DIAGNOSTIC: how much does the anchor choice change the answer?
# ==============================================================================

def compare_anchors(equity_curve: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    """
    Run the old (equity-at-first-event) and new (balance-at-Prague-midnight)
    anchors side by side. Use this to see whether a given strategy's pass/fail
    verdict actually moved, rather than assuming it did.
    """
    new = calculate_daily_stats_anchored(equity_curve, initial_balance)

    df = equity_curve.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['prague_date'] = df['timestamp'].apply(prague_date_of)

    old_rows = []
    for date, group in df.groupby('prague_date', sort=True):
        se = group['equity'].iloc[0]
        me = group['equity'].min()
        old_rows.append({
            'date': date,
            'old_daily_loss_pct': (se - me) / initial_balance * 100,
        })
    old = pd.DataFrame(old_rows)

    merged = new.merge(old, on='date', how='outer')
    merged['delta_pct'] = merged['daily_loss_pct'] - merged['old_daily_loss_pct']
    return merged[['date', 'anchor_source', 'anchor_balance',
                   'old_daily_loss_pct', 'daily_loss_pct', 'delta_pct', 'breached']]
