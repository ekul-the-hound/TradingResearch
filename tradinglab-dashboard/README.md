# TradingLab Dashboard

A read-only quantitative research terminal over the TradingLab pipeline. See
`CLAUDE.md` for the permanent product/engineering rules, `docs/AUDIT.md` for the
repository audit, and `docs/ARCHITECTURE.md` for the design.

## Run (development, DEV FIXTURE data)

```bash
npm install
npm run dev          # http://localhost:5173 — every panel shows a DEV FIXTURE badge
npm test             # vitest
npm run build        # typecheck + production build
```

With no `VITE_BRIDGE_URL` set, the app uses `MockResearchRepository` and marks
everything as **DEV FIXTURE**.

## Run against real data (read-only)

The dashboard is a static frontend; it reaches your SQLite databases through a
small **read-only Python sidecar**. The sidecar opens every database with
`mode=ro&immutable=1`, never writes, never places orders, and reports a missing
database/table/column as an honest `unavailable` state (never a fabricated zero).

### 1. Start the bridge (in your `quant2` env, from the project root)

```powershell
python bridge/sqlite_bridge.py --root "D:\Luke Files\Coding\Developer\TradingResearch" --port 8799
```

Check it: open `http://127.0.0.1:8799/api/health`.

If a database isn't at the expected path, edit `DB_PATHS` at the top of
`bridge/sqlite_bridge.py`. Defaults: results=results/backtest_results.db,
discovery=data/discovery.db, ideas=data/algorithm_ideas.db, lineage=data/lineage.db,
inbox=data/strategies.db, journal=data/challenge_journal.db,
slippage=data/slippage_observations.db.

### 2. Point the frontend at the bridge

```bash
VITE_BRIDGE_URL=http://127.0.0.1:8799 npm run dev
# or for a build:
VITE_BRIDGE_URL=http://127.0.0.1:8799 npm run build
```

Now `SqliteResearchRepository` is used, `isFixture` is false, and the DEV FIXTURE
badges disappear on the wired routes.

## What is wired to real data today

The bridge currently serves (read-only, from `database.py`'s schema):

- **System status bar** — last run timestamp, dataset fingerprint, synthetic-risk
  warning count (`/api/system-status`).
- **Strategy Lab** — one row per `backtest_results` result, with returns
  provenance derived honestly: `REAL` only when `backtest_trades` rows exist,
  `SYNTHETIC_RISK` when `total_trades > 0` but no trade rows were persisted,
  `UNVERIFIED` otherwise (`/api/strategies`).
- **Data & Integrity** — dataset registry grouped by fingerprint, provenance
  coverage (NULL fingerprints counted as "predates tracking"), returns ledger,
  SQLite dependency probes, config-freeze summary (`/api/integrity`).
- **Execution** — always offline; the bridge knows of no broker (`/api/execution`).

Everything else (funnel, research queue/health, strategy detail, portfolio,
challenge readiness, discovery inbox) returns an explicit UNAVAILABLE state
through the adapter until the corresponding joins are added to the bridge. It
never shows fabricated data for those routes.

## Honesty guarantees carried across the swap

- `REAL` returns are asserted from persisted trades, not summary stats.
- NULL `data_fingerprint` is preserved as the signal that a row predates
  provenance tracking.
- FTMO verdicts remain PROXY; the bridge never upgrades them to AUTHORITATIVE.
- SearXNG / Ollama / DataPath show NOT CHECKED (the bridge does not probe them);
  SQLite/lineage/ideas/discovery show OK/DOWN based on file existence.
- A down bridge yields a clear error state telling you how to start it.

## Extending the bridge (remaining routes)

Each unwired read has a one-line `notWired(...)` stub in
`src/data/sqlite/SqliteResearchRepository.ts`. To wire one: add an endpoint to
`bridge/sqlite_bridge.py` that reads the relevant DB/table read-only and returns
the same `Loadable` JSON shape, then replace the stub with `this.get<T>('/api/...')`.

Priority joins to add next (from `docs/AUDIT.md` §9): strategy detail/trades from
`backtest_trades`; PBO/DSR/robustness from `lineage.db backtest_metrics`; discovery
inbox from `discovery.db` documents + `strategies`; untestable ideas from
`algorithm_ideas.db ideas`; challenge journal from `challenge_journal.db`;
FTMO compliance by running `FTMOComplianceChecker` against `backtest_trades`.
