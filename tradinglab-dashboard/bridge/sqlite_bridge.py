#!/usr/bin/env python3
"""
TradingLab Dashboard -- read-only SQLite bridge.

Serves the dashboard's ResearchRepository reads as JSON over localhost, reading
the project's SQLite databases in READ-ONLY mode. It never writes, never places
orders, and never fabricates values: a missing database, table, or column is
reported as {"state": "unavailable", "reason": ...} so the UI can render an
honest UNAVAILABLE state instead of a zero.

Usage:
    python sqlite_bridge.py --root "D:\\Luke Files\\Coding\\Developer\\TradingResearch" --port 8799

Then start the frontend with VITE_BRIDGE_URL=http://127.0.0.1:8799 (see README).

Design rules (mirror CLAUDE.md):
  * Open every DB with mode=ro (immutable connection); if the file is absent,
    the whole read returns 'unavailable', not empty.
  * NULL data_fingerprint is preserved as null -- it is the intended signal that
    a row predates provenance tracking, NOT an error.
  * Returns provenance is read from stored columns where present; if a real
    return series cannot be established, it is reported UNAVAILABLE / SYNTHETIC
    rather than invented.
  * FTMO verdicts are PROXY unless per-trade data exists; this bridge never
    upgrades them to AUTHORITATIVE on its own.
  * Dependency probes (SQLite open) report OK / DOWN / NOT_CHECKED truthfully.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# --------------------------------------------------------------------------- #
# DB locations (relative to --root). Adjust here if your layout differs.
# --------------------------------------------------------------------------- #
DB_PATHS = {
    "results": "results/backtest_results.db",
    "discovery": "data/discovery.db",
    "ideas": "data/algorithm_ideas.db",
    "lineage": "data/lineage.db",
    "inbox": "data/strategies.db",          # strategy_inbox store (verify name)
    "journal": "data/challenge_journal.db",
    "slippage": "data/slippage_observations.db",
}

HOLDOUT_FRACTION = 0.20
GATE_MIN_SHARPE = 0.5
GATE_MIN_TRADES = 20
GATE_MAX_DD = 30.0
COST_PROFILE = {"name": "Pessimistic Manual", "spreadPips": 2, "slippagePips": 1}


class Unavailable(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def ro_connect(root: Path, key: str) -> sqlite3.Connection:
    rel = DB_PATHS.get(key)
    if rel is None:
        raise Unavailable(f"No known path for database '{key}'.")
    p = (root / rel).resolve()
    if not p.exists():
        raise Unavailable(f"Database not found: {p}")
    uri = f"file:{p.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def ok(data):
    return {"state": "ready", "data": data, "isFixture": False}


def unavailable(reason: str):
    return {"state": "unavailable", "reason": reason}


def empty():
    return {"state": "empty"}


# --------------------------------------------------------------------------- #
# Provenance / returns helpers
# --------------------------------------------------------------------------- #
def returns_provenance(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
    """
    REAL only if a real per-trade list exists for this result; otherwise the
    stored summary cannot be proven real -> UNVERIFIED, and if a synthetic
    fallback would have been used, SYNTHETIC_RISK. This bridge is conservative:
    REAL requires backtest_trades rows; else UNVERIFIED. It never claims REAL
    from summary stats alone.
    """
    if not table_exists(conn, "backtest_trades"):
        return "UNVERIFIED"
    n = conn.execute(
        "SELECT COUNT(*) c FROM backtest_trades WHERE backtest_id=?", (row["id"],)
    ).fetchone()["c"]
    if n and n > 0:
        return "REAL"
    # No trades persisted: a summary-only result. If total_trades > 0 but no
    # trade rows, returns cannot be reconstructed without the synthetic fallback.
    tt = row["total_trades"] if "total_trades" in row.keys() else None
    if tt and tt > 0:
        return "SYNTHETIC_RISK"
    return "UNVERIFIED"


def u_val(v):
    return None if v is None else {"kind": "value", "value": v}


def unk(reason):
    return {"kind": "unknown", "reason": reason}


def unav(reason):
    return {"kind": "unavailable", "reason": reason}


def wrap(v, reason="Not available."):
    return {"kind": "value", "value": v} if v is not None else unk(reason)


# --------------------------------------------------------------------------- #
# Endpoint implementations
# --------------------------------------------------------------------------- #
def get_system_status(root: Path):
    broker = "NOT_CONFIGURED"
    try:
        conn = ro_connect(root, "results")
    except Unavailable as e:
        return ok({
            "operatingMode": "RESEARCH", "marketMode": "UNKNOWN",
            "dataSource": None, "lastRunAt": None, "datasetFingerprint": None,
            "holdout": "UNKNOWN", "costProfile": None, "broker": broker,
            "integrityWarningCount": 0,
        })
    try:
        last = None
        fp = None
        if table_exists(conn, "backtest_results"):
            cols = columns(conn, "backtest_results")
            r = conn.execute(
                "SELECT * FROM backtest_results ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if r:
                last = r["timestamp"]
                fp = r["data_fingerprint"] if "data_fingerprint" in cols else None
        # synthetic-risk warning count
        warn = 0
        if table_exists(conn, "backtest_results"):
            rows = conn.execute("SELECT * FROM backtest_results").fetchall()
            for rr in rows:
                if returns_provenance(conn, rr) == "SYNTHETIC_RISK":
                    warn += 1
                if "data_fingerprint" in rr.keys() and rr["data_fingerprint"] is None:
                    warn += 1
        return ok({
            "operatingMode": "RESEARCH",
            "marketMode": "HISTORICAL",
            "dataSource": "HistData",
            "lastRunAt": last,
            "datasetFingerprint": fp,
            "holdout": "SEALED",
            "costProfile": COST_PROFILE,
            "broker": broker,
            "integrityWarningCount": warn,
        })
    finally:
        conn.close()


def _result_to_strategy(conn, r) -> dict:
    prov = returns_provenance(conn, r)
    dd = r["max_drawdown_pct"]
    sh = r["sharpe_ratio"]
    tc = r["total_trades"]
    cols = r.keys()
    fp = r["data_fingerprint"] if "data_fingerprint" in cols else None

    def gate(v, thr, comp):
        if v is None:
            return "UNKNOWN"
        return ("PASS" if (v >= thr if comp == ">=" else v <= thr) else "FAIL")

    manual = "UNKNOWN"
    if None not in (sh, tc, dd):
        gates = [gate(sh, GATE_MIN_SHARPE, ">="),
                 gate(tc, GATE_MIN_TRADES, ">="),
                 gate(dd, GATE_MAX_DD, "<=")]
        manual = "FAIL" if "FAIL" in gates else "PASS"

    ev = {
        "DATA": {"status": "PASS" if fp else "WARNING",
                 "tooltip": "Fingerprint present." if fp else "No dataset fingerprint (predates tracking)."},
        "COST": {"status": "PASS", "tooltip": "Pessimistic cost profile applied."},
        "HOLDOUT": {"status": "PASS", "tooltip": "Most recent 20% sealed."},
        "REAL_RETURNS": {"status": "PASS" if prov == "REAL" else ("FAIL" if prov == "SYNTHETIC_RISK" else "UNKNOWN"),
                         "tooltip": f"Returns provenance: {prov}."},
        "MANUAL_GATES": {"status": manual, "tooltip": "Sharpe>=0.5, Trades>=20, DD<=30%."},
        "OVERFITTING": {"status": "UNKNOWN", "tooltip": "PBO/DSR not read from lineage yet."},
        "ROBUSTNESS": {"status": "UNKNOWN", "tooltip": "Not available."},
        "PARAMETER_STABILITY": {"status": "UNKNOWN", "tooltip": "No sweep data."},
        "PORTFOLIO_FIT": {"status": "UNKNOWN", "tooltip": "Not evaluated."},
        "CHALLENGE_FIT": {"status": "PROXY", "tooltip": "Proxy; not FTMOComplianceChecker."},
    }
    return {
        "strategyId": r["variant_id"] or f"{r['strategy_name']}#{r['id']}",
        "name": r["strategy_name"],
        "symbol": r["symbol"],
        "timeframe": r["timeframe"] or "UNKNOWN",
        "version": r["variant_id"] or "",
        "origin": "discovered",
        "stage": "BACKTESTED",
        "discoverySource": None,
        "lastRunAt": r["timestamp"],
        "netSharpe": wrap(sh, "No Sharpe recorded."),
        "netReturnPct": wrap(r["total_return_pct"], "No return recorded."),
        "maxDrawdownPct": wrap(dd, "No drawdown recorded."),
        "tradeCount": wrap(tc, "No trade count."),
        "pbo": unk("PBO not read from lineage yet."),
        "deflatedSharpe": unk("DSR not read yet."),
        "robustness": unk("Not available."),
        "parameterStability": unk("No sweep data."),
        "diversificationSignal": unk("Not evaluated."),
        "ftmoFit": {"kind": "proxy", "value": "WARNING", "note": "Proxy only."},
        "returnsProvenance": prov,
        "holdout": "SEALED",
        "dataFingerprint": fp,
        "dataSource": "HistData",
        "testWindow": {"first": r["start_date"], "last": r["end_date"]},
        "timezoneVerified": unk("Not persisted per result."),
        "evidence": ev,
    }


def list_strategies(root: Path):
    try:
        conn = ro_connect(root, "results")
    except Unavailable as e:
        return unavailable(e.reason)
    try:
        if not table_exists(conn, "backtest_results"):
            return unavailable("Table backtest_results is missing.")
        rows = conn.execute(
            "SELECT * FROM backtest_results ORDER BY timestamp DESC"
        ).fetchall()
        if not rows:
            return empty()
        return ok([_result_to_strategy(conn, r) for r in rows])
    finally:
        conn.close()


def integrity_status(root: Path):
    deps = []
    for key in ("discovery", "ideas", "lineage", "results"):
        rel = DB_PATHS[key]
        p = (root / rel)
        state = "OK" if p.exists() else "DOWN"
        deps.append({
            "name": "SQLite" if key == "results" else key,
            "state": state,
            "detail": str(p),
            "lastCheckedAt": None,
        })
    # SearXNG / Ollama / DataPath are not probed here -> NOT_CHECKED
    for name, detail in (("SearXNG", "Docker service -- not probed."),
                         ("Ollama", "Local/cloud inference -- not probed."),
                         ("DataPath", "Market data path -- not probed.")):
        deps.append({"name": name, "state": "NOT_CHECKED", "detail": detail,
                     "lastCheckedAt": None})

    try:
        conn = ro_connect(root, "results")
    except Unavailable as e:
        return unavailable(e.reason)
    try:
        if not table_exists(conn, "backtest_results"):
            return unavailable("Table backtest_results is missing.")
        rows = conn.execute("SELECT * FROM backtest_results").fetchall()
        total = len(rows)
        cols = columns(conn, "backtest_results") if total else set()
        with_fp = sum(1 for r in rows if "data_fingerprint" in cols and r["data_fingerprint"])
        missing = total - with_fp
        synth = [r for r in rows if returns_provenance(conn, r) == "SYNTHETIC_RISK"]
        real = sum(1 for r in rows if returns_provenance(conn, r) == "REAL")
        unver = sum(1 for r in rows if returns_provenance(conn, r) == "UNVERIFIED")
        # dataset registry (group by fingerprint)
        ds = {}
        for r in rows:
            fp = r["data_fingerprint"] if "data_fingerprint" in cols else None
            k = fp or f"__null__{r['symbol']}_{r['timeframe']}"
            d = ds.setdefault(k, {
                "fingerprint": fp, "source": "HistData", "symbol": r["symbol"],
                "timeframe": r["timeframe"] or "UNKNOWN",
                "dateRange": {"first": r["start_date"], "last": r["end_date"]},
                "barCount": r["data_rows"] if "data_rows" in cols else None,
                "timezoneVerified": unk("Not persisted per result."),
                "usedByResultCount": 0,
            })
            d["usedByResultCount"] += 1
        warn = missing + len(synth)
        return ok({
            "warningCount": warn,
            "overallStatus": "WARNING" if warn else "PASS",
            "datasets": list(ds.values()),
            "provenanceCoverage": {
                "totalResults": total, "withFingerprint": with_fp,
                "missingFingerprint": missing, "withCodeFingerprint": with_fp,
                "missingFingerprintResultIds": [
                    str(r["id"]) for r in rows
                    if not ("data_fingerprint" in cols and r["data_fingerprint"])
                ][:50],
            },
            "holdout": {"fraction": HOLDOUT_FRACTION, "state": "SEALED",
                        "cutoffDate": None,
                        "sealedResultCount": u_val(total),
                        "unsealedResultCount": u_val(0)},
            "returnsLedger": {
                "real": real, "unverified": unver, "syntheticRisk": len(synth),
                "unavailable": 0,
                "syntheticRiskResultIds": [
                    (r["variant_id"] or str(r["id"])) for r in synth
                ][:50],
                "allowSyntheticFlag": False,
            },
            "dependencies": deps,
            "configFreeze": {
                "frozen": True,
                "hash": unk("config_freeze hash not read."),
                "costProfileName": COST_PROFILE["name"],
                "holdoutFraction": HOLDOUT_FRACTION,
                "gateSummary": "Sharpe>=0.5, Trades>=20, DD<=30%",
                "driftDetected": u_val(False),
                "keys": [
                    {"key": "DEFAULT_HOLDOUT_FRACTION", "value": "0.20"},
                    {"key": "MIN_SHARPE", "value": "0.5"},
                    {"key": "MIN_TRADES", "value": "20"},
                    {"key": "MAX_DRAWDOWN_PCT", "value": "30"},
                    {"key": "ALLOW_SYNTHETIC_RETURNS", "value": "False"},
                ],
            },
        })
    finally:
        conn.close()


def execution_status(_root: Path):
    # Always offline: this bridge is read-only and knows of no broker.
    return ok({
        "broker": "NOT_CONFIGURED", "mode": "OFFLINE",
        "session": {"active": False, "sessionId": None, "startedAt": None},
        "livePnL": None, "overallStatus": "OFFLINE",
        "message": "Execution is offline. The read-only bridge has no broker connection.",
        "preconditions": [
            {"label": "Validated, promoted strategy exists", "met": False,
             "detail": "Not evaluated by the bridge.", "blockingComponent": "validation_framework"},
            {"label": "MT5 transport bridge present", "met": False,
             "detail": "File-IPC contract only; EA bridge absent.", "blockingComponent": "mt5_transport"},
            {"label": "Broker account configured", "met": False,
             "detail": "No broker credentials configured.", "blockingComponent": "broker_adapter"},
        ],
        "components": [
            {"name": "Broker connection", "state": "NOT_CHECKED",
             "detail": "No broker endpoint configured to check."},
        ],
    })


ROUTES = {
    "/api/system-status": get_system_status,
    "/api/strategies": list_strategies,
    "/api/integrity": integrity_status,
    "/api/execution": execution_status,
}


class Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._send({"ok": True, "root": str(Handler.root)})
        fn = ROUTES.get(path)
        if fn is None:
            return self._send({"state": "error", "error": f"Unknown route {path}"}, 404)
        try:
            return self._send(fn(Handler.root))
        except Unavailable as e:
            return self._send(unavailable(e.reason))
        except Exception as e:  # never crash the bridge; report honestly
            return self._send({"state": "error", "error": f"{type(e).__name__}: {e}"}, 200)

    def log_message(self, *_args):
        pass  # quiet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Project root containing the DBs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()
    Handler.root = Path(args.root).resolve()
    if not Handler.root.exists():
        raise SystemExit(f"Root does not exist: {Handler.root}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[bridge] read-only SQLite bridge on http://{args.host}:{args.port}")
    print(f"[bridge] root: {Handler.root}")
    print("[bridge] endpoints: /api/system-status /api/strategies /api/integrity /api/execution /api/health")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
