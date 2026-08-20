# TradingLab Dashboard — Permanent Product & Engineering Rules

> This file is the single source of truth for how the TradingLab dashboard is
> built. Every build task must read it first. If a later instruction conflicts
> with these rules, these rules win unless a human explicitly overrides them.

---

## 1. What this product is

TradingLab is a **desktop-first quantitative forex strategy research,
validation, portfolio-construction, and FTMO challenge-readiness dashboard.**

It is the analyst-facing surface over an existing research repository that
discovers, backtests, validates, and stress-tests systematic forex strategies.

### It is NOT

- A generic retail brokerage dashboard
- A crypto dashboard
- An "AI trader" product
- A live trading terminal (today)
- A marketing site
- A collection of decorative KPI cards

### The workflow it supports

```
raw strategy idea
  → deduplication and quality review
  → static checks for lookahead / prohibited patterns
  → historical backtest
  → pessimistic cost adjustment
  → hard manual gates
  → holdout and statistical validation
  → robustness / parameter-stability review
  → diversification and portfolio construction
  → FTMO challenge simulation and stress testing
  → future paper / demo / live lifecycle
```

The dashboard exists to make the **current state of that workflow**, and the
**evidence behind every promotion decision**, legible and trustworthy.

---

## 2. Product truth (confirmed facts about the backend)

The existing repository is **research-first and primarily on-demand.** Confirmed:

- **Data source:** local **HistData** forex data, typically 1-minute base data
  resampled to the requested timeframe.
- **Execution model:** the pipeline runs **manually** through commands such as
  `run_pipeline.py` and `run_discovery.py`. **Do not imply automatic refresh,
  scheduled jobs, or live updates** unless a real scheduler is verified in code.
- **Databases are SQLite**, including (subject to source verification):
  `discovery.db`, `results/backtest_results.db`, `lineage.db`,
  `algorithm_ideas.db`, `slippage_observations.db`, `challenge_journal.db`.
- **Live / MT5 broker path is NOT connected.** MT5 transport is only a file-IPC
  contract; the EA bridge is not present.
- **Claude and Ollama are build/research helpers, not runtime dashboard
  features.**
- The dashboard must **never** show fake live P&L, fake broker connectivity,
  artificial tickers, fabricated execution data, or inferred refresh times.

---

## 3. Required truth labels

Every data-backed surface must expose the appropriate state. Truth labels are
first-class UI, not footnotes.

| Dimension              | Allowed states                                            |
| ---------------------- | --------------------------------------------------------- |
| Operating mode         | `RESEARCH` · `BACKTEST` · `PAPER` · `DEMO` · `LIVE`       |
| Market / data mode     | `HISTORICAL` · `DELAYED` · `REAL-TIME` · `UNKNOWN`         |
| Last run / refresh     | actual timestamp when available, else `UNKNOWN`           |
| Data source & period   | named source + period, e.g. HistData + date range         |
| Dataset fingerprint    | shown when present, else absent (not faked)               |
| Holdout state          | `SEALED` · `UNSEALED` · `UNKNOWN`                          |
| Returns provenance     | `REAL` · `UNVERIFIED` · `SYNTHETIC RISK` · `UNAVAILABLE`   |
| Cost basis             | named profile + its values                                |
| Compliance basis       | `AUTHORITATIVE` · `PROXY` · `INCOMPLETE` · `UNKNOWN`       |
| Broker state           | `OFFLINE` · `NOT CONFIGURED` · `CONNECTED`                 |

**If a datum cannot be verified from code or database, label it `UNKNOWN` or
`NOT AVAILABLE`. Never silently invent a value.**

---

## 4. Known integrity constraints (do not violate)

- The **most recent 20% of history is sealed** by `holdout_guard.py` by default.
- **Backtest manual gates:** minimum Sharpe **0.5**, minimum **20** trades,
  maximum **30%** drawdown — unless inspection shows config differs.
- **Pessimistic manual cost profile:** **2-pip spread, 1-pip slippage,
  intraday / no swaps** — unless actual config differs.
- `canonical_result.py` has a **synthetic-returns fallback risk** if
  `require_returns()` is bypassed. Do not portray returns as real until verified
  at the result level.
- The existing **FTMO dashboard panel uses display proxies** and must not be
  presented as authoritative compliance until wired to `FTMOComplianceChecker`.
- The **consistency-rule threshold is unknown** until explicitly configured.
  Do not show "consistency headroom" as a real number without that threshold.
- **Capacity / liquidity** results require sufficient underlying data. Never make
  liquidity or impact charts look measured if they are extrapolated or absent.
- **`multi_objective_optimizer` is selection over an existing backtested pool**,
  not true parameter optimization. Label it **"Pareto Selection"** or
  **"Portfolio Selection"** — never "optimization."

---

## 5. Visual system

**Design direction: a modern institutional quantitative research terminal.**

The interface should feel like a proprietary research workstation used by a
systematic PM and quant researcher — calm, credible, dense, precise, and useful
during analysis. **Favor evidence over spectacle.**

### Layout (desktop-first)

- Primary target: **1440px+** width
- Fixed left navigation: **~232px**
- Fixed top system status bar: **~44px**
- **12-column** responsive content grid
- **8px** spacing scale
- Compact table row heights
- **6px** maximum panel corner radius
- Subtle **1px** borders
- Minimal / no shadows
- No large hero content

### Color tokens

| Token                | Hex       | Use                          |
| -------------------- | --------- | ---------------------------- |
| App background       | `#090E14` | page canvas                  |
| Base surface         | `#101821` | panels                       |
| Raised surface       | `#16212C` | nested / raised panels       |
| Subtle border        | `#263646` | 1px borders, dividers        |
| Primary text         | `#E6EDF4` | headings, key values         |
| Secondary text       | `#91A2B5` | labels, supporting copy      |
| Muted text           | `#617286` | metadata, timestamps         |
| Information / active  | `#28B8C8` | active nav, info state        |
| Pass / positive      | `#3ACB8F` | PASS, positive               |
| Failure / negative   | `#F26D6D` | FAIL, negative               |
| Warning              | `#F6B84B` | WARN, caution                |
| Research / selection | `#A78BFA` | selection, research accent    |

Use **tabular numerals** for all performance, risk, timestamps, prices,
percentages, bar counts, and financial quantities.

---

## 6. Anti-slop rules — never produce

- Purple-to-blue gradients
- Glassmorphism
- Giant "Welcome back" / "Good morning trader" hero messages
- Generic "AI confidence" gauges
- Crypto-exchange visual language
- Fake real-time tickers
- Decorative 3D objects, blobs, stock photos, or illustration art
- A page made entirely of identical rounded cards
- Chart-only panels without decision context
- Isolated metrics without a period, calculation basis, units, sample size, or
  status
- Green/red-only communication — **every state must also carry a label, icon, or
  pattern**
- Excessive animation, glowing elements, pills, or badges

---

## 7. Engineering rules

- **Inspect existing source, tables, and schemas before inventing an API
  contract.**
- Keep backend / data adapters **separate** from UI components.
- Use **TypeScript types** (or equivalent typed interfaces) for every dashboard
  entity.
- Create fixtures only behind a clearly named adapter such as
  `MockResearchRepository`.
- Fixtures must be **visibly marked** as `DEMO DATA` / `DEVELOPMENT FIXTURE`.
- **Never hardcode metrics** in production data adapters.
- Build **loading, empty, unavailable, stale, error, and partial-data** states
  for every important data component.
- Do **not** build write operations, order placement, broker controls,
  destructive DB actions, or automated trading actions unless explicitly
  requested.
- Provide concise implementation notes and list changed files after each task.
- Do **not** refactor unrelated code without asking.

---

## 8. Canonical status semantics

Status values used across the app, each with accessible text + iconography
(never color alone):

`PASS` · `FAIL` · `WARNING` · `INFO` · `UNKNOWN` · `PROXY` · `INCOMPLETE` ·
`OFFLINE`

---

## 9. Build sequence (for reference)

1. Rules (this file)
2. Repository & data audit (report only)
3. Implementation architecture (design only)
4. Design system (shell, tokens, primitives, preview)
5. System Status Bar
6. `/research-command`
7. `/strategy-lab`
8. `/strategies/:strategyId`
9. `/data-integrity`
10. `/portfolio-builder`
11. `/challenge-readiness`
12. `/discovery-inbox`
13. `/execution`
14. Replace fixtures with read-only data adapters (per route)
15. Product / accessibility refinement pass
16. Visual critique pass

Each step reads this file first.
