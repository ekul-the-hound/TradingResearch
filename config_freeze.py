# ==============================================================================
# config_freeze.py -- Challenge Config Freeze / Tamper Guard
# ==============================================================================
# Locks the exact code + parameters of every strategy at challenge start and
# refuses to run anything whose bytes have changed since -- unless a human
# supplies an explicit, recorded override.
#
# WHY IT EXISTS:
#   The single most common way a funded challenge is lost is not a bad strategy
#   -- it is the trader tinkering mid-challenge: nudging a parameter after a
#   losing day, "just tightening" a stop, swapping a filter. Each change quietly
#   invalidates all the validation the strategy passed and turns a tested edge
#   into an untested guess at the worst possible moment. This guard makes such a
#   change impossible to do by accident: the engine simply will not start a
#   frozen strategy that no longer matches its locked hash.
#
# HOW IT WORKS:
#   freeze()  -- at challenge start, hash each strategy's code + params (SHA256
#                over a canonical, whitespace-normalized representation) and
#                store the manifest to disk.
#   verify()  -- before trading, re-hash the current code + params and compare.
#                Any mismatch is a BLOCK. A strategy missing from the manifest is
#                also a BLOCK (nothing runs that was not frozen).
#   override()-- records an explicit, timestamped, reason-bearing exception for
#                one strategy, so a deliberate change is possible but never
#                silent -- it leaves an audit trail.
#
# DESIGN PRINCIPLE (project-wide):
#   The safe default is REFUSE. An unknown strategy, an unreadable manifest, or
#   a hash that does not match all block. Overrides must be explicit and are
#   recorded; the guard never silently accepts a change it cannot account for.
# ==============================================================================

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from discovery_config import DATA_DIR
    _DEFAULT_MANIFEST = str(DATA_DIR / "config_freeze.json")
except Exception:
    _D = Path(__file__).parent / "data"
    _D.mkdir(parents=True, exist_ok=True)
    _DEFAULT_MANIFEST = str(_D / "config_freeze.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Canonicalization + hashing ────────────────────────────────────────────────
def _canonical_code(code: str) -> str:
    """
    Normalize code so that only MEANINGFUL changes alter the hash.

    We strip trailing whitespace per line and normalize line endings, so a
    CRLF<->LF conversion or an editor adding trailing spaces does not trip the
    guard. We deliberately do NOT strip comments or blank lines -- a change to a
    comment is still a change the trader made, and during a challenge that is
    worth surfacing rather than hiding.
    """
    if code is None:
        code = ""
    # Normalize line endings and strip trailing whitespace on each line.
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    # Drop trailing blank lines (a file gaining/losing a final newline is noise).
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _canonical_params(params: Any) -> str:
    """
    Normalize params to a canonical JSON string: sorted keys, fixed separators.
    Accepts a dict or a JSON string. Non-JSON strings are hashed as-is.
    """
    if params is None:
        return "{}"
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, ValueError):
            return params.strip()
    try:
        return json.dumps(params, sort_keys=True, separators=(",", ":"),
                          default=str)
    except (TypeError, ValueError):
        return str(params)


def compute_hash(code: str, params: Any) -> str:
    """SHA256 over canonical code + canonical params."""
    payload = _canonical_code(code) + "\x00" + _canonical_params(params)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class FrozenEntry:
    strategy_id: str
    code_hash: str
    frozen_at: str
    params_summary: str = ""   # short human hint, not authoritative


@dataclass
class VerifyResult:
    strategy_id: str
    ok: bool
    reason: str = ""
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    overridden: bool = False

    def __bool__(self) -> bool:
        return self.ok


# ── The guard ─────────────────────────────────────────────────────────────────
class ConfigFreeze:
    """Freeze-and-verify manifest for challenge strategy integrity."""

    def __init__(self, manifest_path: str = _DEFAULT_MANIFEST):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Freeze ----------------------------------------------------------------
    def freeze(self, strategies: List[Dict[str, Any]],
               challenge_id: str = "") -> Dict[str, Any]:
        """
        Freeze a set of strategies. Each dict needs 'strategy_id', 'code', and
        optionally 'params'. Overwrites any existing manifest -- freezing is a
        deliberate start-of-challenge act.
        """
        entries: Dict[str, Any] = {}
        for s in strategies:
            sid = str(s.get("strategy_id", "")).strip()
            if not sid:
                raise ValueError("every strategy needs a non-empty strategy_id")
            code = s.get("code", s.get("generated_code", ""))
            params = s.get("params", s.get("strategy_params", {}))
            h = compute_hash(code, params)
            entries[sid] = asdict(FrozenEntry(
                strategy_id=sid, code_hash=h, frozen_at=_utcnow(),
                params_summary=_canonical_params(params)[:120]))

        manifest = {
            "challenge_id": challenge_id,
            "frozen_at": _utcnow(),
            "entries": entries,
            "overrides": {},
        }
        self._write(manifest)
        return manifest

    def is_frozen(self) -> bool:
        return self.manifest_path.exists()

    # -- Verify ----------------------------------------------------------------
    def verify(self, strategy_id: str, code: str,
               params: Any = None) -> VerifyResult:
        """
        Verify a strategy's current code+params against the frozen hash.

        Safe default is REFUSE: no manifest, unknown strategy, or hash mismatch
        all return ok=False. An active override for this strategy flips a
        mismatch to ok=True but records overridden=True.
        """
        manifest = self._read()
        if manifest is None:
            return VerifyResult(strategy_id, False,
                                "no freeze manifest exists; nothing is frozen "
                                "so nothing may run under freeze policy")
        entries = manifest.get("entries", {})
        entry = entries.get(strategy_id)
        if entry is None:
            return VerifyResult(strategy_id, False,
                                "strategy is not in the frozen manifest; refusing "
                                "to run an unfrozen strategy during a challenge")

        expected = entry.get("code_hash")
        actual = compute_hash(code, params)
        if actual == expected:
            return VerifyResult(strategy_id, True, "hash matches frozen config",
                                expected_hash=expected, actual_hash=actual)

        # Mismatch. Check for an explicit override.
        override = manifest.get("overrides", {}).get(strategy_id)
        if override and override.get("active"):
            return VerifyResult(
                strategy_id, True,
                f"hash MISMATCH but overridden: {override.get('reason', '')}",
                expected_hash=expected, actual_hash=actual, overridden=True)

        return VerifyResult(
            strategy_id, False,
            "code/params changed since freeze (hash mismatch); refusing to run. "
            "Use override() with a reason to run a deliberate change.",
            expected_hash=expected, actual_hash=actual)

    def verify_all(self, strategies: List[Dict[str, Any]]) -> List[VerifyResult]:
        out = []
        for s in strategies:
            sid = str(s.get("strategy_id", ""))
            code = s.get("code", s.get("generated_code", ""))
            params = s.get("params", s.get("strategy_params", {}))
            out.append(self.verify(sid, code, params))
        return out

    # -- Override --------------------------------------------------------------
    def override(self, strategy_id: str, reason: str,
                 active: bool = True) -> None:
        """
        Record an explicit override for a strategy, so a deliberate mid-challenge
        change is permitted but leaves an audit trail. A reason is required.
        """
        if not reason or not reason.strip():
            raise ValueError("an override requires a non-empty reason")
        manifest = self._read()
        if manifest is None:
            raise RuntimeError("cannot override before any freeze exists")
        manifest.setdefault("overrides", {})[strategy_id] = {
            "active": bool(active),
            "reason": reason.strip(),
            "recorded_at": _utcnow(),
        }
        self._write(manifest)

    def clear_override(self, strategy_id: str) -> None:
        manifest = self._read()
        if manifest is None:
            return
        overrides = manifest.get("overrides", {})
        if strategy_id in overrides:
            overrides[strategy_id]["active"] = False
            self._write(manifest)

    # -- Manifest IO -----------------------------------------------------------
    def _read(self) -> Optional[Dict[str, Any]]:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # An unreadable manifest is treated as "no valid freeze" -> refuse.
            return None

    def _write(self, manifest: Dict[str, Any]) -> None:
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def get_manifest(self) -> Optional[Dict[str, Any]]:
        return self._read()


__all__ = ["ConfigFreeze", "compute_hash", "FrozenEntry", "VerifyResult"]


if __name__ == "__main__":
    import tempfile, os
    mp = tempfile.mktemp(suffix=".json")
    cf = ConfigFreeze(manifest_path=mp)
    strat = {"strategy_id": "s1", "code": "def next(self):\n    pass\n",
             "params": {"period": 20, "mult": 2.0}}
    cf.freeze([strat])
    print("unchanged:", cf.verify("s1", strat["code"], strat["params"]).ok)
    print("tweaked param:", cf.verify("s1", strat["code"], {"period": 21, "mult": 2.0}).reason[:50])
    print("unknown strat:", cf.verify("s2", "x", {}).ok)
    cf.override("s1", "intentional retune after week 1 review")
    print("after override:", cf.verify("s1", strat["code"], {"period": 21}).ok)
    os.remove(mp)
