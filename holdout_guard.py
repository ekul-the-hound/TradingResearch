# ==============================================================================
# holdout_guard.py
# ==============================================================================
# Phase 2, Item 10 -- protected chronological holdout, structurally enforced.
#
# THE PROBLEM THIS SOLVES
# -----------------------
# A research loop that generates and evaluates thousands of strategies destroys
# its own out-of-sample set. Not through any single mistake -- through use. Every
# time an OOS number informs a decision (keep this, kill that, retune the other),
# information flows from the OOS set into the strategy population. After enough
# iterations the OOS set is as overfit as the training set, and it stops being
# evidence of anything while still looking exactly like evidence.
#
# This is the failure this codebase is most exposed to, because the whole point
# of the pipeline is running many candidates automatically. A discipline of
# "don't look at the holdout" cannot survive that. It has to be structural.
#
# WHAT "STRUCTURALLY ENFORCED" MEANS HERE
# ---------------------------------------
#   1. DEFAULT DENY. Every data request is truncated at the cutoff. Getting
#      holdout data requires an explicit token; there is no flag to forget.
#
#   2. THE CUTOFF IS PINNED, NOT COMPUTED. Written to the ledger the first time
#      it is set, and never recomputed. Deriving "the last 20% of data" on each
#      call would silently move the boundary as new data arrives, quietly
#      promoting yesterday's holdout into today's training set. That is a
#      failure mode nobody notices, because every individual run looks correct.
#
#   3. A FINITE BUDGET. Each look costs a peek, and peeks are limited. This is
#      the part that makes it real. An unlimited holdout you are trusted not to
#      overuse is just a test set with extra steps.
#
#   4. AN APPEND-ONLY LEDGER ON DISK. Every peek is recorded permanently:
#      when, which strategy, why, what the result was. Restarting the process
#      does not reset it, and there is no method to erase an entry.
#
#   5. DEFLATION. After N looks you have run N trials, and the best of N is
#      biased upward whether or not you intended a search. deflate_sharpe()
#      applies the correction so the number you report accounts for the looking
#      you already did.
#
# WHAT THIS DOES NOT DO
# ---------------------
# It cannot stop someone reading the parquet files directly, and it is not meant
# to. It stops the *pipeline* from consuming the holdout by accident, which is
# how the holdout actually gets destroyed -- not by deliberate cheating but by a
# loop doing exactly what it was told, ten thousand times.
#
# USAGE
#     guard = HoldoutGuard.load()                  # or .initialise(...)
#     df = guard.enforce(df, symbol='EUR-USD')     # truncated, always
#
#     token = guard.request_access('final validation of variant_07', 'variant_07')
#     df = guard.enforce(df, symbol='EUR-USD', token=token)   # full series
#     guard.record_outcome(token, sharpe=1.4, notes='passed')
# ==============================================================================

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

DEFAULT_LEDGER = os.path.join('data', 'holdout_ledger.json')
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_MAX_PEEKS = 5

_LOCK = threading.Lock()


class HoldoutViolation(RuntimeError):
    """Raised when holdout data is requested without a valid token."""


class HoldoutExhausted(RuntimeError):
    """Raised when the peek budget is spent. The holdout is burned."""


@dataclass
class HoldoutToken:
    """
    Single-use permission to read holdout data.

    Scoped to one strategy and one reason so the ledger records what the look
    was actually for. Consumed on first use -- a token that could be reused
    would make the budget meaningless.
    """
    token_id: str
    strategy_id: str
    reason: str
    issued_at: str
    consumed: bool = False


@dataclass
class PeekRecord:
    token_id: str
    strategy_id: str
    reason: str
    issued_at: str
    consumed_at: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)


class HoldoutGuard:
    """Default-deny access control over the chronological holdout period."""

    def __init__(self, ledger_path: str = DEFAULT_LEDGER):
        self.ledger_path = ledger_path
        self._state: Dict[str, Any] = {
            'cutoff_date': None,
            'max_peeks': DEFAULT_MAX_PEEKS,
            'created_at': None,
            'peeks': [],
        }
        self._tokens: Dict[str, HoldoutToken] = {}
        self._warned = set()
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self):
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, 'r', encoding='utf-8') as f:
                    self._state = json.load(f)
            except Exception as e:
                # A corrupt ledger must not silently become an empty one --
                # that would reset the budget, which is the whole protection.
                raise RuntimeError(
                    f"Holdout ledger at {self.ledger_path} is unreadable ({e}). "
                    f"Refusing to continue: treating it as empty would reset the "
                    f"peek budget and silently un-burn a spent holdout."
                )

    def _save(self):
        d = os.path.dirname(self.ledger_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = f"{self.ledger_path}.tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, indent=2)
        os.replace(tmp, self.ledger_path)          # atomic

    # -- setup ------------------------------------------------------------
    @classmethod
    def initialise(cls, cutoff_date, max_peeks: int = DEFAULT_MAX_PEEKS,
                   ledger_path: str = DEFAULT_LEDGER, force: bool = False):
        """
        Pin the cutoff. Idempotent unless force=True.

        Re-pinning after peeks have been spent is refused: moving the boundary
        after looking is indistinguishable from choosing the boundary that
        flatters the result.
        """
        g = cls(ledger_path)
        if g._state.get('cutoff_date') and not force:
            if str(pd.Timestamp(cutoff_date)) != g._state['cutoff_date']:
                raise HoldoutViolation(
                    f"Cutoff already pinned at {g._state['cutoff_date']}; refusing "
                    f"to move it to {cutoff_date}. A moving boundary silently "
                    f"promotes holdout data into the training set. Pass force=True "
                    f"only when starting a genuinely new research programme."
                )
            return g

        if g._state.get('peeks') and not force:
            raise HoldoutViolation(
                f"{len(g._state['peeks'])} peek(s) already spent against the "
                f"existing cutoff. Re-pinning now would invalidate them."
            )

        g._state['cutoff_date'] = str(pd.Timestamp(cutoff_date))
        g._state['max_peeks'] = max_peeks
        g._state['created_at'] = datetime.now().isoformat()
        g._state.setdefault('peeks', [])
        g._save()
        return g

    @classmethod
    def load(cls, ledger_path: str = DEFAULT_LEDGER):
        return cls(ledger_path)

    @staticmethod
    def suggest_cutoff(index, fraction: float = DEFAULT_HOLDOUT_FRACTION):
        """
        Date that leaves `fraction` of the series as holdout.

        Only a suggestion -- call initialise() to pin it. Deliberately separate
        so the pinning is an explicit decision rather than a side effect of
        looking at the data.
        """
        idx = pd.DatetimeIndex(index).sort_values()
        if len(idx) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(idx)}")
        pos = int(len(idx) * (1.0 - fraction))
        return idx[max(pos, 1)]

    # -- state ------------------------------------------------------------
    @property
    def cutoff(self) -> Optional[pd.Timestamp]:
        c = self._state.get('cutoff_date')
        if not c:
            return None
        ts = pd.Timestamp(c)
        # pd.Timestamp can yield NaT on unparseable input. A NaT cutoff would
        # make every comparison False and silently disable the guard, so treat
        # it as "not configured" rather than letting it through. isna() rather
        # than `is pd.NaT` so the checker sees the narrowing.
        if bool(pd.isna(ts)):
            return None
        # cast: pd.isna is not a TypeGuard, so the narrowing above is invisible
        # to the checker even though it is real at runtime.
        return cast(pd.Timestamp, ts)

    @property
    def is_configured(self) -> bool:
        return self.cutoff is not None

    @property
    def peeks_used(self) -> int:
        return len(self._state.get('peeks', []))

    @property
    def max_peeks(self) -> int:
        return int(self._state.get('max_peeks', DEFAULT_MAX_PEEKS))

    @property
    def peeks_remaining(self) -> int:
        return max(0, self.max_peeks - self.peeks_used)

    @property
    def is_burned(self) -> bool:
        return self.peeks_remaining <= 0

    # -- the choke point --------------------------------------------------
    def enforce(self, df: pd.DataFrame, symbol: str = '', timeframe: str = '',
                token: Optional[HoldoutToken] = None) -> pd.DataFrame:
        """
        Truncate a frame at the cutoff unless a valid token is supplied.

        This is the whole mechanism. Every data request routes through here,
        so the default outcome is protection and the exception requires effort.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        if not self.is_configured:
            return df                                # nothing pinned yet
        if not isinstance(df.index, pd.DatetimeIndex):
            return df                                # cannot reason about dates

        cutoff = self.cutoff
        if cutoff is None:
            return df

        if token is not None:
            self._consume(token, symbol)
            return df                                # full series, logged

        before = len(df)
        # Wrap the mask result: pandas types boolean indexing loosely, and an
        # unannotated return here propagates a Series|None union to every caller.
        out = pd.DataFrame(df[df.index < cutoff])
        if len(out) < before:
            key = (symbol, timeframe)
            if key not in self._warned:
                self._warned.add(key)
                print(f"[HOLDOUT] {symbol or 'data'} {timeframe} truncated at "
                      f"{cutoff.date()} ({before - len(out):,} bars withheld). "
                      f"{self.peeks_remaining}/{self.max_peeks} peeks remaining.")
        return out

    # -- access -----------------------------------------------------------
    def request_access(self, reason: str, strategy_id: str) -> HoldoutToken:
        """
        Spend one peek. Refused when the budget is gone.

        Reason and strategy_id are required and recorded. A peek you cannot
        justify in writing is a peek you should not be taking.
        """
        if not reason or not str(reason).strip():
            raise ValueError("A written reason is required for holdout access")
        if not strategy_id or not str(strategy_id).strip():
            raise ValueError("strategy_id is required for holdout access")
        if not self.is_configured:
            raise HoldoutViolation(
                "No cutoff pinned. Call HoldoutGuard.initialise() first.")

        with _LOCK:
            if self.is_burned:
                raise HoldoutExhausted(
                    f"Holdout budget exhausted: {self.peeks_used}/{self.max_peeks} "
                    f"peeks spent. This holdout no longer provides an unbiased "
                    f"estimate and cannot be un-spent. Collect new out-of-sample "
                    f"data, or run forward on demo, or accept that the remaining "
                    f"evidence is in-sample.\n"
                    f"Spent on: " + ', '.join(
                        p['strategy_id'] for p in self._state['peeks']))

            token = HoldoutToken(
                token_id=f"peek_{self.peeks_used + 1}_{int(datetime.now().timestamp())}",
                strategy_id=str(strategy_id),
                reason=str(reason).strip(),
                issued_at=datetime.now().isoformat(),
            )
            self._tokens[token.token_id] = token
            self._state['peeks'].append({
                'token_id': token.token_id,
                'strategy_id': token.strategy_id,
                'reason': token.reason,
                'issued_at': token.issued_at,
                'consumed_at': None,
                'symbols': [],
                'outcome': {},
            })
            self._save()

        print(f"[HOLDOUT] Peek {self.peeks_used}/{self.max_peeks} issued for "
              f"{strategy_id}: {reason}")
        if self.is_burned:
            print("[HOLDOUT] This was the LAST peek. The holdout is now burned.")
        return token

    def _consume(self, token: HoldoutToken, symbol: str = ''):
        rec = self._find(token.token_id)
        if rec is None:
            raise HoldoutViolation(
                f"Token {token.token_id} is not in the ledger. Tokens must come "
                f"from request_access(); a fabricated one defeats the budget.")
        if token.consumed and symbol in rec['symbols']:
            raise HoldoutViolation(
                f"Token {token.token_id} was already consumed for {symbol}. "
                f"Request a new peek.")
        token.consumed = True
        rec['consumed_at'] = rec['consumed_at'] or datetime.now().isoformat()
        if symbol and symbol not in rec['symbols']:
            rec['symbols'].append(symbol)
        self._save()

    def record_outcome(self, token: HoldoutToken, **outcome):
        """
        Attach what the peek actually showed.

        Recording the result matters as much as recording the access: a ledger
        of looks without outcomes cannot tell you whether you have been
        selecting on the holdout.
        """
        rec = self._find(token.token_id)
        if rec is None:
            raise HoldoutViolation(f"Unknown token {token.token_id}")
        rec['outcome'].update({k: _jsonable(v) for k, v in outcome.items()})
        self._save()

    def _find(self, token_id):
        for p in self._state.get('peeks', []):
            if p['token_id'] == token_id:
                return p
        return None

    # -- multiple-testing correction --------------------------------------
    def deflate_sharpe(self, sharpe: float, n_obs: int,
                       extra_trials: int = 0) -> Dict[str, float]:
        """
        Adjust a holdout Sharpe for the number of times the holdout was used.

        Every peek is a trial. The best result across N trials is biased upward
        even when no explicit search was run, because the decision to stop
        looking is itself a selection. This applies the standard expected-maximum
        correction so the reported figure reflects the looking already done.

        Returns the raw and deflated Sharpe plus the haircut applied.
        """
        trials = max(1, self.peeks_used + extra_trials)
        if trials <= 1 or n_obs <= 1:
            return {'sharpe': sharpe, 'deflated_sharpe': sharpe,
                    'trials': trials, 'haircut': 0.0}

        # Expected maximum of `trials` draws from a standard normal.
        gamma = 0.5772156649
        e_max = ((1 - gamma) * _ppf(1 - 1.0 / trials)
                 + gamma * _ppf(1 - 1.0 / (trials * np.e)))
        haircut = float(e_max / np.sqrt(max(n_obs - 1, 1)))
        return {
            'sharpe': float(sharpe),
            'deflated_sharpe': float(sharpe - haircut),
            'trials': trials,
            'haircut': haircut,
        }

    # -- reporting --------------------------------------------------------
    def report(self) -> str:
        L = [f"\n{'=' * 64}", "  HOLDOUT STATUS", '=' * 64]
        if not self.is_configured:
            L += ["  NOT CONFIGURED - no cutoff pinned, no protection active.",
                  "  Call HoldoutGuard.initialise(cutoff_date) to enable.", '=' * 64]
            return '\n'.join(L)
        cutoff = self.cutoff
        assert cutoff is not None      # is_configured already established this
        L.append(f"  Cutoff:  {cutoff.date()} (pinned {self._state['created_at'][:10]})")
        L.append(f"  Peeks:   {self.peeks_used}/{self.max_peeks} spent, "
                 f"{self.peeks_remaining} remaining")
        if self._state['peeks']:
            L.append("")
            L.append("  Access history:")
            for p in self._state['peeks']:
                out = p.get('outcome') or {}
                tail = f" -> {out}" if out else " (no outcome recorded)"
                L.append(f"    {p['issued_at'][:16]}  {p['strategy_id']:20} "
                         f"{p['reason'][:38]}{tail}")
        if self.is_burned:
            L.append("")
            L.append("  [BURNED] No unbiased out-of-sample estimate remains.")
        L.append('=' * 64)
        return '\n'.join(L)


def _ppf(p: float) -> float:
    """Standard-normal inverse CDF. Uses scipy when present, else an approximation."""
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        # Acklam's approximation; accurate to ~1e-9 over the useful range.
        if p <= 0 or p >= 1:
            return 0.0
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    return v