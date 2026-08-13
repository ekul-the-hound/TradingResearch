# ==============================================================================
# preflight.py -- Pre-Flight Go / No-Go Checklist for Live Trading
# ==============================================================================
# One command to run before starting a live/challenge session. It answers a
# single question: "is it safe to start trading right now?" -- and prints a
# green (GO) or red (NO-GO) verdict with a per-check breakdown.
#
# WHY THIS EXISTS:
#   The #1 way a live challenge dies is not a bad strategy -- it is an
#   operational mistake at start-up: a dead feed, a clock that disagrees with
#   the broker, unreconciled positions left from a crash, a full disk that
#   silently drops the trade journal, or a rule budget that was never computed.
#   This script makes those failures LOUD and BLOCKING before any order is sent.
#
# DESIGN PRINCIPLE (project-wide):
#   A check that cannot actually run must report that it could not run -- it may
#   never fabricate a PASS. "Unknown" is a WARN or FAIL, never a silent green.
#   That is the same discipline as the synthetic-returns fix: make the absence
#   of a valid answer representable and loud.
#
# SEVERITY MODEL:
#   PASS  -- check ran and is satisfied.
#   WARN  -- check ran, non-blocking concern (session may proceed with caution).
#   FAIL  -- check ran and is NOT satisfied -> NO-GO.
#   SKIP  -- check could not run (missing dependency/arg). Treated as WARN for
#            the verdict unless --strict, where any SKIP becomes NO-GO.
#
# EXIT CODES:
#   0  -- GO         (no FAILs; SKIPs allowed unless --strict)
#   1  -- NO-GO      (at least one FAIL, or a SKIP under --strict)
#   2  -- usage/internal error
#
# USAGE:
#   python preflight.py --firm ftmo --symbols EURUSD,GBPUSD
#   python preflight.py --firm ftmo --symbols EURUSD --broker paper --strict
#   python preflight.py --firm ftmo --symbols EURUSD --json
#
# The broker defaults to 'paper' so this is safe to run without a live terminal;
# pass --broker mt5 (once that adapter exists) for a real pre-flight.
# ==============================================================================

from __future__ import annotations

import sys
import json
import shutil
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


# ── Severity + result types ───────────────────────────────────────────────────
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_ICON = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}


@dataclass
class CheckResult:
    name: str
    severity: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class PreflightConfig:
    firm: str = "ftmo"
    symbols: List[str] = field(default_factory=list)
    broker_kind: str = "paper"
    min_free_disk_mb: float = 500.0
    max_clock_skew_seconds: float = 2.0
    db_paths_to_back_up: List[str] = field(default_factory=list)
    strict: bool = False


# ── The runner ────────────────────────────────────────────────────────────────
class Preflight:
    """
    Runs an ordered list of independent checks and aggregates a verdict.

    Each check is a method returning a CheckResult. A check must catch its own
    expected failure modes and translate them into FAIL/SKIP -- an uncaught
    exception is itself a FAIL (the check is not trustworthy, so neither is GO).
    """

    def __init__(self, config: PreflightConfig):
        self.config = config
        self.results: List[CheckResult] = []
        self._broker = None  # lazily created, shared across checks

    # -- Orchestration ---------------------------------------------------------
    def run(self) -> List[CheckResult]:
        checks: List[Callable[[], CheckResult]] = [
            self.check_firm_rules,
            self.check_broker_connect,
            self.check_feed_alive,
            self.check_clock_sync,
            self.check_positions_reconciled,
            self.check_rule_budgets,
            self.check_disk_space,
            self.check_db_backup,
            self.check_governor_importable,
        ]
        self.results = []
        for chk in checks:
            try:
                self.results.append(chk())
            except Exception as e:  # a check that crashes is not a pass
                self.results.append(CheckResult(
                    name=getattr(chk, "__name__", "unknown_check"),
                    severity=FAIL,
                    detail=f"check raised {type(e).__name__}: {e}",
                ))
        return self.results

    def verdict(self) -> str:
        """GO or NO-GO based on results and strictness."""
        severities = [r.severity for r in self.results]
        if FAIL in severities:
            return "NO-GO"
        if self.config.strict and SKIP in severities:
            return "NO-GO"
        return "GO"

    def exit_code(self) -> int:
        return 0 if self.verdict() == "GO" else 1

    # -- Broker helper ---------------------------------------------------------
    def _get_broker(self):
        if self._broker is not None:
            return self._broker
        from broker_base import create_broker
        self._broker = create_broker(self.config.broker_kind)
        return self._broker

    # ======================================================================
    # CHECKS
    # ======================================================================
    def check_firm_rules(self) -> CheckResult:
        """Firm profile loads and its numbers validate."""
        name = "firm_rules"
        try:
            from firm_rules import load_profile
        except ImportError as e:
            return CheckResult(name, SKIP, f"firm_rules not importable: {e}")
        try:
            rules = load_profile(self.config.firm)
        except Exception as e:
            return CheckResult(name, FAIL, f"could not load firm '{self.config.firm}': {e}")

        # If the firm has unmodelled rules, that is a real caveat for a live run.
        detail = f"loaded '{self.config.firm}'"
        data: Dict[str, Any] = {}
        try:
            data["fully_modelled"] = bool(rules.is_fully_modelled())
            if not rules.is_fully_modelled():
                return CheckResult(
                    name, WARN,
                    f"{detail}, but rules are not fully modelled: "
                    f"{rules.caveat_line()}",
                    data,
                )
        except Exception:
            # Older FirmRules without these helpers -- not fatal.
            pass
        return CheckResult(name, PASS, detail, data)

    def check_broker_connect(self) -> CheckResult:
        """Broker constructs and connects."""
        name = "broker_connect"
        try:
            broker = self._get_broker()
        except Exception as e:
            return CheckResult(name, FAIL, f"could not create broker "
                                           f"'{self.config.broker_kind}': {e}")
        try:
            ok = broker.connect()
        except Exception as e:
            return CheckResult(name, FAIL, f"connect() raised: {e}")
        if not ok or not getattr(broker, "is_connected", False):
            return CheckResult(name, FAIL, "broker did not report connected")
        return CheckResult(name, PASS, f"{self.config.broker_kind} connected")

    def check_feed_alive(self) -> CheckResult:
        """Every requested symbol returns a usable two-sided quote."""
        name = "feed_alive"
        if not self.config.symbols:
            return CheckResult(name, SKIP, "no symbols provided (--symbols)")
        try:
            broker = self._get_broker()
        except Exception as e:
            return CheckResult(name, FAIL, f"broker unavailable: {e}")

        # Reuse the tick guard if present -- same definition of a good quote.
        guard = None
        try:
            from live_tick_guard import LiveTickGuard, TickGuardConfig
            # Staleness/outlier need history; disable them for a one-shot check.
            guard = LiveTickGuard(TickGuardConfig(max_frozen_ticks=0,
                                                  outlier_sigma=0))
        except ImportError:
            guard = None

        bad: List[str] = []
        details: Dict[str, Any] = {}
        for sym in self.config.symbols:
            try:
                tick = broker.get_tick(sym)
            except Exception as e:
                bad.append(f"{sym} (get_tick raised: {e})")
                continue
            if tick is None:
                bad.append(f"{sym} (no tick)")
                continue
            if guard is not None:
                v = guard.check(tick)
                if not v.ok:
                    bad.append(f"{sym} ({'; '.join(v.reasons)})")
                else:
                    details[sym] = {"mid": v.mid, "spread_bps": round(v.spread_bps, 3)}
            else:
                # No guard: at least require a positive two-sided quote.
                bid = getattr(tick, "bid", 0) or 0
                ask = getattr(tick, "ask", 0) or 0
                if bid <= 0 or ask <= 0 or bid > ask:
                    bad.append(f"{sym} (bad quote bid={bid} ask={ask})")
                else:
                    details[sym] = {"bid": bid, "ask": ask}

        if bad:
            return CheckResult(name, FAIL, f"unusable feed: {', '.join(bad)}", details)
        return CheckResult(name, PASS,
                           f"{len(self.config.symbols)} symbol(s) quoting", details)

    def check_clock_sync(self) -> CheckResult:
        """
        Local clock vs broker server time within tolerance.

        Many brokers expose server time via the latest tick timestamp. If we
        cannot obtain a broker time we SKIP rather than pretend the clock is
        fine -- an unverified clock is exactly what corrupts daily-loss anchors.
        """
        name = "clock_sync"
        if not self.config.symbols:
            return CheckResult(name, SKIP, "no symbols to read a server timestamp from")
        try:
            broker = self._get_broker()
            tick = broker.get_tick(self.config.symbols[0])
        except Exception as e:
            return CheckResult(name, SKIP, f"could not read broker time: {e}")
        ts = getattr(tick, "timestamp", "") if tick is not None else ""
        if not ts:
            return CheckResult(name, SKIP,
                               "broker tick has no timestamp; cannot verify skew")
        server_dt = _parse_iso(ts)
        if server_dt is None:
            return CheckResult(name, SKIP, f"unparseable broker timestamp {ts!r}")
        now = datetime.now(timezone.utc)
        if server_dt.tzinfo is None:
            server_dt = server_dt.replace(tzinfo=timezone.utc)
        skew = abs((now - server_dt).total_seconds())
        data = {"skew_seconds": round(skew, 3),
                "tolerance": self.config.max_clock_skew_seconds}
        if skew > self.config.max_clock_skew_seconds:
            return CheckResult(name, FAIL,
                               f"clock skew {skew:.2f}s exceeds "
                               f"{self.config.max_clock_skew_seconds:.2f}s", data)
        return CheckResult(name, PASS, f"skew {skew:.2f}s within tolerance", data)

    def check_positions_reconciled(self) -> CheckResult:
        """
        Report open broker positions so a human confirms they are expected.

        We cannot know your intended state, so any pre-existing position is a
        WARN (needs eyes), never an auto-PASS. Flat is a clean PASS.
        """
        name = "positions_reconciled"
        try:
            broker = self._get_broker()
            positions = broker.get_positions()
        except Exception as e:
            return CheckResult(name, FAIL, f"get_positions raised: {e}")
        open_pos = [p for p in (positions or [])
                    if abs(getattr(p, "size", 0) or 0) > 0]
        if not open_pos:
            return CheckResult(name, PASS, "account is flat")
        summary = ", ".join(
            f"{getattr(p, 'symbol', '?')} {getattr(p, 'side', '?')} "
            f"{getattr(p, 'size', 0)}"
            for p in open_pos
        )
        return CheckResult(
            name, WARN,
            f"{len(open_pos)} open position(s) present -- confirm expected: {summary}",
            {"open_positions": summary},
        )

    def check_rule_budgets(self) -> CheckResult:
        """Compute today's daily-loss budget and drawdown floor from live equity."""
        name = "rule_budgets"
        try:
            from firm_rules import load_profile
            rules = load_profile(self.config.firm)
        except Exception as e:
            return CheckResult(name, SKIP, f"firm rules unavailable: {e}")
        try:
            broker = self._get_broker()
            bal = broker.get_balance()
            equity = float(getattr(bal, "total_equity", 0) or getattr(bal, "total", 0))
        except Exception as e:
            return CheckResult(name, FAIL, f"could not read balance: {e}")
        if equity <= 0:
            return CheckResult(name, FAIL, f"non-positive equity {equity}")
        try:
            daily_limit = rules.daily_loss_limit(equity)
            dd_floor = rules.drawdown_floor(equity)
        except Exception as e:
            return CheckResult(name, FAIL, f"budget computation failed: {e}")
        data = {
            "equity": round(equity, 2),
            "daily_loss_limit": round(daily_limit, 2),
            "drawdown_floor": round(dd_floor, 2),
        }
        return CheckResult(
            name, PASS,
            f"equity {equity:.0f} -> daily-loss budget {daily_limit:.0f}, "
            f"DD floor {dd_floor:.0f}", data,
        )

    def check_disk_space(self) -> CheckResult:
        """Enough free disk for logs/journal/DB writes."""
        name = "disk_space"
        try:
            usage = shutil.disk_usage(str(Path(__file__).parent))
            free_mb = usage.free / (1024 * 1024)
        except Exception as e:
            return CheckResult(name, FAIL, f"disk_usage failed: {e}")
        data = {"free_mb": round(free_mb, 1),
                "required_mb": self.config.min_free_disk_mb}
        if free_mb < self.config.min_free_disk_mb:
            return CheckResult(name, FAIL,
                               f"only {free_mb:.0f} MB free, need "
                               f"{self.config.min_free_disk_mb:.0f} MB", data)
        return CheckResult(name, PASS, f"{free_mb:.0f} MB free", data)

    def check_db_backup(self) -> CheckResult:
        """
        Back up any specified DBs to a timestamped copy so a mid-session crash
        is recoverable. If no DBs specified, SKIP (nothing to protect).
        """
        name = "db_backup"
        paths = self.config.db_paths_to_back_up
        if not paths:
            return CheckResult(name, SKIP, "no --backup-db paths given")
        backed_up: List[str] = []
        missing: List[str] = []
        for p in paths:
            src = Path(p)
            if not src.exists():
                missing.append(p)
                continue
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = src.with_suffix(src.suffix + f".preflight_{stamp}.bak")
            try:
                shutil.copy2(src, dst)
                backed_up.append(str(dst.name))
            except Exception as e:
                return CheckResult(name, FAIL, f"backup of {p} failed: {e}")
        if missing:
            return CheckResult(name, FAIL,
                               f"DB(s) not found: {', '.join(missing)}",
                               {"backed_up": backed_up})
        return CheckResult(name, PASS, f"backed up {len(backed_up)} DB(s)",
                           {"backed_up": backed_up})

    def check_governor_importable(self) -> CheckResult:
        """The live risk governor must at least import and construct."""
        name = "governor"
        try:
            from live_governor import LiveGovernor, GovernorConfig
        except ImportError as e:
            return CheckResult(name, FAIL, f"live_governor not importable: {e}")
        try:
            LiveGovernor(GovernorConfig())
        except Exception as e:
            return CheckResult(name, FAIL, f"governor construction failed: {e}")
        return CheckResult(name, PASS, "governor constructs")

    # -- Reporting -------------------------------------------------------------
    def render_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f" PRE-FLIGHT  firm={self.config.firm}  "
                     f"broker={self.config.broker_kind}  "
                     f"strict={self.config.strict}")
        lines.append("=" * 60)
        for r in self.results:
            lines.append(f"{_ICON.get(r.severity, '[????]')} {r.name:24} {r.detail}")
        lines.append("-" * 60)
        counts = {sev: sum(1 for r in self.results if r.severity == sev)
                  for sev in (PASS, WARN, FAIL, SKIP)}
        lines.append(f" {counts[PASS]} pass, {counts[WARN]} warn, "
                     f"{counts[FAIL]} fail, {counts[SKIP]} skip")
        verdict = self.verdict()
        banner = ">>> GO <<<" if verdict == "GO" else ">>> NO-GO <<<"
        lines.append("")
        lines.append(banner)
        return "\n".join(lines)

    def render_json(self) -> str:
        return json.dumps({
            "firm": self.config.firm,
            "broker": self.config.broker_kind,
            "strict": self.config.strict,
            "verdict": self.verdict(),
            "checks": [r.to_dict() for r in self.results],
        }, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight go/no-go checklist for live trading")
    parser.add_argument("--firm", default="ftmo", help="Firm profile key")
    parser.add_argument("--symbols", default="",
                        help="Comma-separated symbols to verify feeds for")
    parser.add_argument("--broker", default="paper",
                        help="Broker kind (paper/ccxt/mt5/...)")
    parser.add_argument("--min-disk-mb", type=float, default=500.0)
    parser.add_argument("--max-skew", type=float, default=2.0,
                        help="Max allowed clock skew (seconds)")
    parser.add_argument("--backup-db", default="",
                        help="Comma-separated DB paths to back up")
    parser.add_argument("--strict", action="store_true",
                        help="Treat SKIP as NO-GO")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    dbs = [d.strip() for d in args.backup_db.split(",") if d.strip()]

    cfg = PreflightConfig(
        firm=args.firm,
        symbols=symbols,
        broker_kind=args.broker,
        min_free_disk_mb=args.min_disk_mb,
        max_clock_skew_seconds=args.max_skew,
        db_paths_to_back_up=dbs,
        strict=args.strict,
    )
    pf = Preflight(cfg)
    pf.run()
    print(pf.render_json() if args.json else pf.render_text())
    return pf.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
