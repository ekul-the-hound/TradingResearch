# TradingLab Dashboard — Repository & Data-Access Audit

> Prompt 2 deliverable. Source-grounded, report only — no frontend implemented.
> Every claim below is traced to a file in the existing repository. Anything not
> verifiable from source is marked **UNKNOWN**. Nothing here is called "live"
> because no broker/feed connection exists in source.

---

## 1. Canonical application entry points (existing dashboards)

There are **five overlapping dashboard stacks** in the repo. None is
authoritative; they were built at different times and duplicate each other.

| Stack        | Entry file                    | Framework            | Run command                          | State |
| ------------ | ----------------------------- | -------------------- | ------------------------------------ | ----- |
| ReactPy #1   | `dashboard_react.py`          | ReactPy + Starlette + Plotly | `python dashboard_react.py`  | Most feature-complete; runs real backtests inline |
| ReactPy #2   | `react_dashboard2.py`         | ReactPy + FastAPI + Plotly | `python react_dashboard2.py` → `:8080` | 14–18 pages over all 6 phases; reads SQLite directly |
| Streamlit    | `dashboard_for_now.py`        | Streamlit + Plotly   | `streamlit run dashboard_for_now.py` | Broad analytics; header comment still says `dashboard_complete.py` |
| Vizro        | `dashboard_vizro.py`          | Vizro (Plotly/Dash)  | `python dashboard_vizro.py` → `:8050` | **Sample data only** (`np.random.seed(42)`) — not wired |
| Reflex       | `rxconfig.py` (+ `setup_reflex_dashboard.bat`) | Reflex + TailwindV4 | reflex run                | Config scaffold only; no substantial pages found |

Supporting (non-entry) panels, all designed to be import-safe and render-free:

- `dashboard_ftmo_panel.py` — FTMO data layer. **Explicitly documents that both
  dashboards rendered PASS/FAIL FTMO badges without ever calling
  `FTMOComplianceChecker`** (display proxy). Root cause stated in-file:
  `backtest_results` stores summary stats only, no trade list.
- `dashboard_portfolio_panel.py` — comparison + firm-rules data layer; returns
  plain dataclasses, locks capability toggles not backed by `firm_rules.IMPLEMENTED`.
- `dashboard_compare_page.py` — `make_page()` factory for the compare page,
  consumes `react_dashboard2`'s helpers to avoid an import cycle.

**Verdict:** the repo has drifted into duplicate dashboards. `react_dashboard2.py`
is the most complete *data-reading* surface; `dashboard_react.py` is the most
complete *interactive* one. Neither matches the CLAUDE.md design system.

---

## 2. Recommended canonical frontend (reason from source, not preference)

**None of the existing Python UI stacks should be made canonical.** Reasoning
grounded in what the source actually is:

- `dashboard_vizro.py` is disqualified: it renders **fabricated random data**
  (`np.random.seed(42)`, `strategies_df = pd.DataFrame({...})`), which violates
  CLAUDE.md §2/§6 (no invented metrics).
- The ReactPy stacks mix **data access, backtest execution, and rendering in one
  process** (`dashboard_react.py` imports Backtrader and runs backtests inside
  the UI). That violates CLAUDE.md §7 (adapters must be separate from UI) and
  makes typed entity contracts impossible.
- Reflex is a config scaffold with no substantial page source to preserve.
- CLAUDE.md §5–§7 mandate a **12-col grid, reusable typed primitives, hover
  metric definitions, dense sortable tables, and TypeScript types for every
  entity** — none of which the Python stacks provide, and which are impractical
  to retrofit onto ReactPy/Streamlit.

**Recommendation:** a dedicated **React + TypeScript (Vite)** frontend that talks
to the existing SQLite databases through a thin **read-only** adapter layer. The
Python side keeps ownership of all metric *computation*; the frontend only
*reads and displays*. This is the only path that satisfies the typed-entity,
adapter-separation, and truth-label requirements without rewriting the research
backend.

The existing Python dashboards are **frozen, not deleted**, during migration
(they still work for ad-hoc research) until the canonical app reaches parity.

---

## 3. SQLite databases referenced in source

Confirmed via `CREATE TABLE` / connection strings in source. Actual on-disk
presence is **UNKNOWN from here** (databases live on the user's `D:`/`E:` drives).

| Database                         | Owning module            | Purpose (from source) |
| -------------------------------- | ------------------------ | --------------------- |
| `results/backtest_results.db`    | `database.py`            | Backtest summary results + trades + fingerprints |
| `discovery.db`                   | `research_db.py`         | Scraped documents + extracted strategies |
| `algorithm_ideas.db`             | `algorithm_ideas.py`     | Untestable/blocked idea backlog |
| `lineage.db`                     | `lineage_tracker.py`     | Strategy lineage + `backtest_metrics` (PBO/DSR) |
| `challenge_journal.db`           | `challenge_journal.py`   | FTMO challenge day/event journal |
| `slippage_observations.db`       | `slippage_recorder.py`   | Recorded fills vs. signal (`fills` table) |
| decay store (`decay.db`?)        | `decay_calculator.py`    | `strategy_decay_snapshots` |
| source-extraction store          | `source_extractor.py`    | `source_strategies` (LLM-extracted raw ideas) |
| strategy inbox store             | `strategy_inbox.py`      | `strategies` (validated/generated) |

> Test/example paths like `/nope/missing.db`, `toy.db`, `t.db`, `path/to/results.db`
> appear in source but are fixtures/tests — **not production databases.**

---

## 4. Confirmed tables, key columns, relationships

### Backtest results — `database.py`

**`backtest_results`** (summary statistics only):
`id` PK, `strategy_name`, `variant_id`, `symbol`, `timeframe`,
`start_date`, `end_date`, `bars_tested`, `initial_cash`, `final_value`,
`total_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `total_trades`,
`win_rate`, `profit_factor`, `strategy_params`, `modifications`, `timestamp`,
`claude_analysis`
— plus ALTER-added provenance columns: `data_fingerprint`, `data_rows`,
`data_first`, `data_last`, `code_fingerprint` (**NULL on pre-provenance rows —
that NULL is the intended signal they predate the timezone fix**).

**`backtest_trades`** (trade blotter; `backtest_id` → `backtest_results.id`):
`entry_date`, `exit_date`, `entry_price`, `exit_price`, `size`, `pnl`,
`return_pct`, `duration_bars`, `is_long`.
> In-file comment confirms: **without these trades, `FTMOComplianceChecker`
> cannot run** — this table is the fix for the proxy-badge problem.

### Discovery — `research_db.py`

**`documents`**: `doc_id` PK, `url`, `title`, `content`, `content_hash` UNIQUE,
`source_type`, `source_bias`, `search_query`, `fetch_timestamp`,
`content_length`, `status`.
Plus a `strategies` table (extracted+validated) keyed off `doc_id`.

### Raw extraction — `source_extractor.py`

**`source_strategies`**: `id` PK, `job_id`, `source_title`, `model_used`,
`extracted_at`, `name`, `summary`, `hypothesis`, `entry_rules`, `exit_rules`,
`stop_loss`, `take_profit`, `indicators`, `parameters`, `asset_class`,
`timeframe`, `position_sizing`, …

### Strategy inbox — `strategy_inbox.py`

**`strategies`**: `strategy_id` PK, `doc_id`, `strategy_name`, `summary`,
`description`, `generated_code`, `code_file_path`, `origin_source` (default
`'manual'`), `source_url`, `source_type`, `source_bias`, `parent_docs`,
`quality_score`, `has_math`, `has_backtest`, `has_code`, `has_explicit_params`, …

### Ideas backlog — `algorithm_ideas.py`

**`ideas`**: `idea_id` PK, `title`, `description`, `why_untestable`,
`data_needed`, `category`, `tags`, `confidence`, `effort`, `generated_by`,
`source_context`, `asset_class`, `timeframe`, `status` (default `'open'`),
`promoted_strategy_id`, `notes`, `created_at`.
> Cleanly supports the Discovery-Inbox "untestable ideas" section (Prompt 12D):
> `why_untestable`, `data_needed`, `confidence`, `category` are all first-class.

### Challenge journal — `challenge_journal.py`

**`journal_days`**: `trading_date` PK, `opening_equity`, `closing_equity`,
`day_pnl`, `day_pnl_pct`, `trades`, `worst_decision`, `tightest_headroom`,
`min_daily_pct_to_limit`, `complete`, `notes`, `updated_at`.
Plus **`journal_events`** (autoincrement id, per-event log).

### Fills — `slippage_recorder.py`

**`fills`**: `id` PK, `symbol`, `side`, `signal_price`, `fill_price`,
`quoted_spread`, `slippage_pct`, `spread_pct`, `timestamp` (indexed by symbol).

### Decay — `decay_calculator.py`

**`strategy_decay_snapshots`**: `strategy_id`, `symbol`, `snapshot_date`,
baseline vs. rolling metric pairs (win rate, expectancy, frequency, profit
factor, consecutive losses, avg duration…).

### Lineage — `lineage_tracker.py`

Owns `lineage.db` and, per `overfitting_detector.py` header, a
**`backtest_metrics`** table where **PBO/DSR scores are stored**. Exact columns
**UNKNOWN** (not dumped in this pass) — flag for Prompt 14.

---

## 5. Existing data-access layers usable by the dashboard

- `database.py` — canonical results read/write for `backtest_results` /
  `backtest_trades`.
- `research_db.py` — discovery documents + extracted strategies.
- `canonical_result.py` — **the single most important contract for the UI.**
  `class CanonicalResult` with `returns_source` ∈ {real, synthetic, mixed},
  `returns_synthetic: bool`, `has_real_returns` property, and
  `require_returns(purpose)` which raises `SyntheticReturnsError` when returns
  are missing/synthetic/mixed. Global switch `ALLOW_SYNTHETIC_RETURNS = False`.
  **The dashboard's REAL / SYNTHETIC-RISK / UNAVAILABLE label must be driven by
  `returns_source`, never recomputed in the UI.**
- `ftmo_compliance.py` — `class FTMOComplianceChecker` (line 323): **the only
  authoritative FTMO verdict source.** Requires per-trade data.
- `firm_rules.py` — `class FirmRules` (defaults: `max_daily_loss_pct=0.05`,
  `max_total_drawdown_pct=0.10`, `min_trading_days=4`, `profit_targets` dict) and
  an `IMPLEMENTED` frozenset gating which rules are actually modeled.
  **`CONSISTENCY_RULE` is deliberately absent from `IMPLEMENTED`.**
- `holdout_guard.py` — `DEFAULT_HOLDOUT_FRACTION = 0.20`, `HoldoutGuard`,
  `HoldoutToken`, `suggest_cutoff()`. Authoritative holdout-seal source.
- `challenge_simulator.py` — `simulate_challenge()`, `ChallengeResult`,
  `simulate_pass_rate_early_stop()` → bootstrap P(pass). Authoritative sim source.
- `overfitting_detector.py` — `compute_pbo()` (CSCV), `PBOResult`
  (`probability`, `is_overfit` = PBO > 0.5), DSR. Authoritative overfitting source.
- `consistency_rule.py` — `ConsistencyResult` with `passed: Optional[bool]`
  (**None = could not evaluate**), sentinel
  `NOT_EVALUATED_NO_THRESHOLD = 'no_threshold_configured'`.
- `manual_gates.py` — gate helpers (`require_positive_sharpe`, threshold plumbing).

---

## 6. Metric authority classification

| Metric / signal              | Status         | Source of truth |
| ---------------------------- | -------------- | --------------- |
| Sharpe, return, max DD, trades, win rate, profit factor | **Authoritative** | `backtest_results` |
| Per-trade blotter            | **Authoritative (when present)** | `backtest_trades` |
| Data fingerprint / window    | **Authoritative when non-NULL; NULL = predates provenance** | `backtest_results` ALTER cols |
| Returns realness             | **Authoritative** | `canonical_result.returns_source` |
| PBO / DSR                    | **Authoritative when stored** | `overfitting_detector` → `backtest_metrics` |
| FTMO compliance verdict      | **Authoritative only via `FTMOComplianceChecker`; existing dashboards show PROXY** | `ftmo_compliance.py` |
| Challenge P(pass), P95 worst day | **Authoritative when simulated** | `challenge_simulator.py` |
| Holdout sealed state         | **Authoritative** | `holdout_guard.py` |
| Consistency headroom         | **ABSENT — threshold not configured** | `consistency_rule.py` (returns None) |
| Multi-objective "optimization" | **MISLABELED — it is Pareto/portfolio *selection*** over an existing pool | `multi_objective_optimizer.py` |
| Capacity / liquidity / impact | **Potentially extrapolated — treat as UNAVAILABLE unless data supports** | capacity/liquidity modules |
| Live P&L / broker connectivity | **ABSENT — no broker connection exists** | (none) |
| Vizro sample metrics         | **SYNTHETIC — fabricated random data** | `dashboard_vizro.py` |

---

## 7. Recommended single canonical frontend path

`/frontend` (React + TS + Vite) → typed `ResearchRepository` interface →
two implementations:
1. `MockResearchRepository` (DEV FIXTURE, clearly marked) for building UI now.
2. `SqliteResearchRepository` (read-only) wired per-route in Prompt 14, reading
   the databases in §3 and the authoritative sources in §5.

All FTMO/validation/sim verdicts are **read from the Python-owned results**, not
recomputed in TypeScript.

---

## 8. Phased migration plan (no duplicate dashboards, nothing breaks)

1. **Freeze** the five existing dashboards; add no features to them.
2. Build the canonical React/TS app against `MockResearchRepository` (Prompts 4–13).
3. Wire read-only SQLite adapters route-by-route (Prompt 14), starting with
   Research Command → Strategy Lab → Strategy Detail → Data & Integrity →
   Portfolio Builder → Challenge Readiness.
4. Reach parity, then retire the ReactPy/Streamlit/Vizro/Reflex stacks in a
   separate, explicitly-approved cleanup (never silently).

---

## 9. Missing fields / gaps to close before a trustworthy UI

1. **`lineage.db` / `backtest_metrics` schema** — not dumped; PBO/DSR/robustness
   column names UNKNOWN. Needed for Strategy Lab + Strategy Detail.
2. **Validation persistence** — walk-forward, permutation, CSCV, parameter-stability
   results: storage location per-strategy UNKNOWN. Strategy Lab columns (PBO, DSR,
   robustness, parameter stability) may be UNAVAILABLE until confirmed.
3. **Returns series at the result level** — `backtest_results` holds summaries;
   whether a real return *series* is retrievable per result (vs. reconstructed
   from `backtest_trades`) must be confirmed to avoid the synthetic-returns branch.
4. **Consistency threshold** — unset by design; UI must render
   "Configuration required" everywhere it would show headroom.
5. **Diversification / correlation** — cross-strategy overlap + correlation matrix
   source table UNKNOWN; Portfolio Builder correlation matrix may be UNAVAILABLE.
6. **FTMO wiring** — Challenge Readiness / Strategy Detail must call
   `FTMOComplianceChecker` against `backtest_trades`; until wired, label PROXY.
7. **Timezone-verification persistence per result** — verification state may not
   be stored per row (`verify_histdata_timezone.py` exists but per-result persistence
   UNKNOWN); Data & Integrity must show UNKNOWN where not persisted.
8. **Health checks** for SearXNG / Ollama / data path — whether real health probes
   exist is UNKNOWN; Data & Integrity must show "not checked", not fake green.

---

### One-line summary

The backend has **honest, well-designed truth contracts** (`canonical_result`,
`firm_rules.IMPLEMENTED`, `holdout_guard`, `consistency_rule` returning `None`),
but the **existing dashboards bypass them with proxies and, in one case,
fabricated data.** The canonical fix is a separate typed read-only frontend that
surfaces those contracts instead of re-deriving metrics.
