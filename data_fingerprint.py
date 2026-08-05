# ==============================================================================
# data_fingerprint.py
# ==============================================================================
# Phase 2, Item 18 -- data fingerprint on every result.
#
# WHY THIS IS URGENT RIGHT NOW
# ----------------------------
# The Phase 0 timezone fix shifted every forex timestamp by +5 hours and the
# forex cache was rebuilt from scratch. The results database still holds rows
# computed on the OLD data, and there is no field anywhere that distinguishes
# them. A result computed on a 5-hour-shifted daily boundary sits in the same
# table, in the same format, next to a correct one.
#
# That is not a hypothetical. It is the current state of backtest_results.
#
# More generally: a research platform that cannot say WHICH data produced a
# number cannot reproduce it, cannot invalidate it when the data changes, and
# cannot tell you whether two results are comparable. Every data correction
# from here -- and there will be more -- silently poisons the archive unless
# results carry provenance.
#
# WHAT IS FINGERPRINTED
# ---------------------
#   data   content hash of the actual OHLCV served: row count, first and last
#          timestamp, and a checksum over the close series. Two frames with the
#          same hash are the same data; different hashes mean do not compare
#          the results.
#   code   git commit if available, plus versions of the libraries whose
#          behaviour can move a number (pandas, numpy, backtrader).
#
# HOW IT GETS RECORDED
# --------------------
# data_manager.get_data is the single choke point every backtester dispatches
# through -- the same one the holdout guard uses. Frames are fingerprinted
# there and stashed in a registry keyed by (symbol, timeframe); save_backtest
# looks the entry up when writing the result. No backtester needs to know this
# exists, including ones not written yet.
#
# The registry is process-local and deliberately so: a fingerprint that
# outlived the process could be attached to a result from different data.
# ==============================================================================

import hashlib
import os
import subprocess
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
_CODE_FP: Optional[Dict[str, str]] = None

# Timestamps at or after this hour in the raw series indicate post-timezone-fix
# forex data. Pre-fix EURUSD began 2000-05-30 17:27; post-fix, 22:27.
TZ_FIX_SHIFT_HOURS = 5


@dataclass
class DataFingerprint:
    hash: str
    rows: int
    first: str
    last: str
    symbol: str
    timeframe: str
    columns: str
    computed_at: str

    def short(self) -> str:
        return self.hash[:12]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fingerprint_frame(df: pd.DataFrame, symbol: str = '',
                      timeframe: str = '') -> Optional[DataFingerprint]:
    """
    Content hash of an OHLCV frame.

    Hashes the close series rather than the whole frame: it is the column every
    strategy actually consumes, and hashing all five columns of a multi-million
    row frame is slow enough that people would turn it off. Row count and
    endpoints are included separately, so a truncation or a re-dating changes
    the hash even if the close values are untouched -- which is exactly what the
    timezone fix did.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    try:
        h = hashlib.sha256()
        h.update(f"{len(df)}".encode())
        h.update(str(df.index[0]).encode())
        h.update(str(df.index[-1]).encode())
        h.update(','.join(map(str, df.columns)).encode())

        if 'close' in df.columns:
            arr = np.ascontiguousarray(df['close'].to_numpy(dtype='float64'))
            h.update(arr.tobytes())

        return DataFingerprint(
            hash=h.hexdigest(),
            rows=len(df),
            first=str(df.index[0]),
            last=str(df.index[-1]),
            symbol=symbol,
            timeframe=timeframe,
            columns=','.join(map(str, df.columns)),
            computed_at=datetime.now().isoformat(timespec='seconds'),
        )
    except Exception:
        # Fingerprinting must never break a backtest. A missing fingerprint is
        # recoverable; a crashed run is not.
        return None


def code_fingerprint() -> Dict[str, str]:
    """
    Git commit plus versions of libraries that can move a number.

    Cached: shelling out to git on every backtest would be noticeable across
    thousands of runs.
    """
    global _CODE_FP
    if _CODE_FP is not None:
        return _CODE_FP

    fp: Dict[str, str] = {}
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)))
        if commit.returncode == 0:
            fp['git'] = commit.stdout.strip()
            dirty = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__)))
            if dirty.returncode == 0 and dirty.stdout.strip():
                # An uncommitted tree means the commit hash does not identify
                # the code that ran. Say so rather than implying it does.
                fp['git'] += '-dirty'
    except Exception:
        pass

    for mod in ('pandas', 'numpy', 'backtrader'):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, '__version__', '?')
        except Exception:
            pass

    _CODE_FP = fp
    return fp


def code_fingerprint_str() -> str:
    fp = code_fingerprint()
    return ' '.join(f"{k}={v}" for k, v in sorted(fp.items()))


# ==============================================================================
# REGISTRY
# ==============================================================================

def _key(symbol: str, timeframe: str) -> str:
    return f"{symbol}|{timeframe}"


def record(symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[DataFingerprint]:
    """Fingerprint a frame and stash it for whatever result comes from it."""
    fp = fingerprint_frame(df, symbol, timeframe)
    if fp is not None:
        with _LOCK:
            _REGISTRY[_key(symbol, timeframe)] = fp.to_dict()
    return fp


def lookup(symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        return _REGISTRY.get(_key(symbol, timeframe))


def clear():
    with _LOCK:
        _REGISTRY.clear()


def registry_size() -> int:
    with _LOCK:
        return len(_REGISTRY)


# ==============================================================================
# COMPARISON AND AUDIT
# ==============================================================================

def compare(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    What differs between two fingerprints.

    `comparable` is the field that matters: False means results from these two
    datasets should not be ranked against each other, whatever their metrics say.
    """
    if not a or not b:
        return {'comparable': False,
                'reason': 'one or both fingerprints missing'}
    if a.get('hash') == b.get('hash'):
        return {'comparable': True, 'reason': 'identical data'}

    diffs = []
    for f in ('rows', 'first', 'last', 'columns'):
        if a.get(f) != b.get(f):
            diffs.append(f"{f}: {a.get(f)} vs {b.get(f)}")
    return {
        'comparable': False,
        'reason': '; '.join(diffs) if diffs
                  else 'same shape and range but different values',
    }


def looks_pre_timezone_fix(first_timestamp: Any, symbol: str = '') -> Optional[bool]:
    """
    Heuristic: does this forex series predate the EST->UTC conversion?

    The fix shifted every timestamp +5h. HistData weeks open at 17:00 in its
    own EST clock, which becomes 22:00 UTC. So a forex series whose first bar
    sits in the 17:00-19:00 band was almost certainly never converted.

    Returns None when it cannot tell -- an unknown must not read as a clean
    bill of health.
    """
    try:
        ts = pd.Timestamp(first_timestamp)
    except Exception:
        return None
    if ts is pd.NaT:
        return None

    sym = (symbol or '').upper().replace('-', '').replace('/', '')
    majors = ('EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD')
    is_fx = len(sym) == 6 and sym[:3] in majors and sym[3:] in majors
    if not is_fx:
        return None

    if 16 <= ts.hour <= 19:
        return True
    if 21 <= ts.hour <= 23:
        return False
    return None


def describe(fp: Optional[Dict[str, Any]]) -> str:
    if not fp:
        return "no fingerprint (provenance unknown)"
    return (f"{fp['hash'][:12]} | {fp['rows']:,} rows | "
            f"{str(fp['first'])[:19]} -> {str(fp['last'])[:19]}")