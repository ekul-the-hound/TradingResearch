# TradingLab / TradingResearch

A quantitative trading research platform: discover candidate strategies from
public sources, put them through statistical validation designed to reject
most of them, and — for the ones that survive — model whether they would pass
a prop-firm evaluation and then trade them under runtime rule enforcement.

**Goal:** pass a prop-firm funded-trading challenge, then trade the funded
account. Those are two different optimisation problems and the platform treats
them as such (see *System A vs System B* below).

**Scale:** ~100 modules, 31 test suites, ~880 tests.

---

## The pipeline

`run_pipeline.py` orchestrates eleven steps.

| # | Step | Module(s) | What it does |
|---|------|-----------|--------------|
| 1 | Discovery | `discovery_pipeline.py` | SearXNG search → LLM extraction → candidate strategies |
| 2 | Backtest & Filter | `backtester*.py`, `filtering_pipeline.py` | Backtrader run, hard filters |
| 3 | Optimize | `multi_objective_optimizer.py` | NSGA-II **selection over the existing pool** |
| 4 | Validate | `overfitting_detector.py`, `lookahead_detector.py`, `prohibited_patterns.py` | PBO / DSR / CSCV, lookahead, banned patterns |
| 5 | Risk Analysis | `tail_risk.py`, `liquidity_stress.py`, `capacity_model.py` | Tail behaviour, capacity limits |
| 6 | Diversification | `diversification_filter.py` | Correlation-based culling |
| 7 | Split & Mutate | `mutate_strategy.py`, `genetic_operators.py` | Variant generation (the "mutation agent") |
| 8 | Re-Validate Mutations | as step 4 | Mutants go back through the same gates |
| 9 | Drift Baselines | `drift_detector.py`, `decay_calculator.py` | Reference points for edge decay |
| 10 | Learning Loop | `learning_loop.py` | Outcomes feed back into discovery |
| 11 | Analytics | `lineage_analytics.py`, `performance_attribution.py` | Lineage and attribution reporting |

`--from-step` defaults to **2**, so discovery is opt-in. Use `--from-step 1` to
include it.

**Step 1 has six sub-steps** (`discovery_pipeline.py`): search → fetch →
extract → deduplicate (FAISS) → validate → save. `preflight_check()` verifies
SearXNG and the LLM endpoints are reachable before any of it starts.

### Prop-firm layer

Sits alongside steps 4–5 and downstream of them.

```
strategies (validated)
    │
    ├─ portfolio_merge.py ──── combine N strategies at TRADE level
    │                          into one CanonicalResult
    │
    ├─ ftmo_compliance.py ──── backtest validator: did this comply?
    │
    ├─ challenge_simulator.py ─ multi-stage walk with early stopping;
    │                          P(funded), not P(pass stage one)
    │
    ├─ consistency_rule.py ─── best-day share of net profit
    │
    └─ live_governor.py ────── runtime enforcer: may I trade right now?
           │
           └─ governed_broker.py ── wraps any BaseBroker so the
                                    governor cannot be bypassed
                    │
                    └─ mt5_adapter.py ── MetaTrader 5
```

`firm_rules.py` is the single source of truth for firm thresholds. Every
component above reads from it; a test fails the build if a second copy appears.

---

## Setup

### One-time

```powershell
conda env create -f environment.yml     # or: conda create -n quant2 python=3.12
conda activate quant2
pip install -r requirements.txt
```

MetaTrader 5 (only needed for live execution) requires **both** the Python
package and the desktop terminal:

```powershell
pip install MetaTrader5
# then install the MT5 terminal from metatrader5.com and log in once
python -c "import mt5_adapter as m; print(m.selftest_against_terminal())"
```

### Every session

```powershell
# 1. Docker Desktop must be running, then:
docker start searxng

# 2. Ollama (see below)
ollama serve

# 3. Environment
conda activate quant2
cd "D:\Luke Files\Coding\Developer\TradingResearch"
$env:PYTHONUTF8 = "1"
$env:DISCOVERY_MODE = "hybrid"
```

### Ollama

`DISCOVERY_MODE` selects a model preset in `discovery_config.py`:

| Mode | Summarizer | Code generator | Notes |
|------|-----------|----------------|-------|
| `cloud` | `qwen3.5:cloud` | `qwen3-coder:480b-cloud` | 60k context |
| `local` | `qwen2.5:7b-instruct` | `qwen2.5-coder:7b` | 12k context, slower |
| `hybrid` | `minimax-m3:cloud` | `minimax-m3:cloud` | default working mode |

```powershell
ollama serve            # start the server (skip if running as a service)
ollama list             # models available locally
ollama ps               # models currently loaded in memory
```

Endpoint is `http://localhost:11434/v1`, overridable via `OLLAMA_HOST`.

---

## Running it

```powershell
python run_discovery.py --status            # what is already in the DB
python run_discovery.py --max-runs 1        # one discovery batch
python run_discovery.py --continuous --interval 3600

python run_pipeline.py                      # steps 2-11 (no discovery)
python run_pipeline.py --from-step 1        # everything
python run_pipeline.py --from-step 4 --to-step 6
```

Dashboard (React + TypeScript, read-only):

```powershell
cd tradinglab-dashboard
npm install          # first time only
npm run dev          # http://localhost:5173 -- DEV FIXTURE data
```

To view real data instead of fixtures, run the read-only SQLite bridge from the
repo root in a second terminal, then point the frontend at it:

```powershell
conda activate quant2
python tradinglab-dashboard\bridge\sqlite_bridge.py --root . --port 8799

# in the dashboard terminal:
$env:VITE_BRIDGE_URL = 'http://127.0.0.1:8799'; npm run dev
```

The bridge opens every database with `mode=ro&immutable=1`. It never writes,
never places orders, and reports a missing database, table, or column as an
explicit UNAVAILABLE state rather than a fabricated zero. See
`tradinglab-dashboard/README.md` for the full frontend documentation and
`tradinglab-dashboard/CLAUDE.md` for the design and honesty rules it follows.

---

## Tests

```powershell
python run_all_tests.py             # everything
python run_all_tests.py -k merge    # filter by filename
python run_all_tests.py --list      # what runs, and what is excluded and why
```

Each suite runs in its own process, so a module that fails at import takes
down only its own file rather than aborting collection for everything.

**Two rules the runner enforces:**

- **Skips count as failures.** A skipped test is a test that did not run.
- **A suite that reports no test count is `SILENT`, not `PASS`.** "Exited 0
  with no summary" is indistinguishable from "ran nothing", and calling that
  green is how a suite quietly stops testing anything.

Exclusions are listed with a reason. An unexplained exclusion is the same
problem in a different place.

---

## Design principles

These were arrived at by hitting the corresponding bug, and most have a test
that fails if they are violated again.

### Confident wrong numbers are the failure mode

The recurring defect in this codebase has not been crashes; it has been
components producing plausible numbers with no way to signal uncertainty.
Every significant fix has taken the same shape: make the absence of an answer
representable, propagating, and loud.

- `CanonicalResult.returns_source` distinguishes real trade data from
  synthetic; `require_returns()` refuses the latter.
- `ConsistencyResult.passed` is `Optional`. When there is no profit, "what
  share came from the best day" has no answer, and both `True` and `False`
  would be fabrications.
- `StageStats.pass_rate` is `None` when nobody reached that stage — different
  from `0.0`, which claims everyone failed.
- `FirmRules.unsupported()` lists rules the engine does **not** check. A PASS
  carrying a non-empty list is a partial answer and renders as one.

### One source of truth for firm thresholds

Prop-firm limits once lived in four places. `test_single_source_of_truth.py`
scans the source and fails if any method reads a module constant instead of
the configured profile — behavioural tests only catch that if someone
remembers to write one for each new method.

### Numbers are configurable; semantics are not

In `firm_rules.py`, a threshold is a float you can edit freely. Static vs
trailing drawdown is a different *algorithm*, so it is a capability gated
behind an `IMPLEMENTED` whitelist. The dashboard greys out what has no code
behind it and says why.

### The live governor acts before the limit, not at it

In a backtest a breach is a data point. Live, it is terminal — account failed,
fee gone, no later trading undoes it. The governor halts at a fraction of each
limit, and every uncertainty resolves toward halting: stale account state,
missing daily anchor, or an exception inside the governor itself all stop
trading.

### Early stopping is not optional in challenge simulation

A trader who reaches the target stops. Evaluating a fixed window and checking
final equity fails paths that had already won and then gave it back. This
understated pass rates by up to 2.5×. It also interacts with the consistency
rule: stopping early concentrates profit into fewer days, making the best day a
larger share of the total.

### System A vs System B

Passing the evaluation and profitably trading the funded account are different
objectives with different fitness functions. Consistency rules may structurally
prohibit the burst-style strategies that optimise best for System A.

---

## Data

| Asset class | Source | Status |
|-------------|--------|--------|
| Forex | HistData.com yearly files, `E:\TradingData` | Usable |
| Crypto | — | **~100 bars, not usable** |
| Indices | — | **0 files** |

HistData timestamps are Prague-local; the daily-loss anchor depends on getting
that conversion right. `verify_histdata_timezone.py` checks it.

---

## Prop-firm target and live rollout

**Target firm: FTMO (US).** US accounts run MT5 only, routed via OANDA, under
netting / FIFO / single-position-per-symbol rules.

| Phase | Target | Min trading days | Daily loss | Max drawdown |
|-------|--------|------------------|-----------|--------------|
| Free Trial | 5% | 2 | 5% | 10% |
| Challenge | 10% | 4 | 5% | 10% |
| Verification | 5% | 4 | 5% | 10% |

Rollout order — **the Free Trial is a dress rehearsal, not a formality.**
Passing it does not grant funding; it exists to prove the loop before money is
spent. FTMO allows unlimited Free Trials, each running 14 days.

1. Install MT5 on the Mac mini; log into an FTMO US Free Trial account.
2. Build the MQL5 Expert Advisor that implements the `mt5_transport.py`
   file-IPC contract.
3. Arm `live_governor.py` and `kill_switch.py` against the account.
4. Run the trial for 2–4 weeks. **Go/no-go bar: live results within ±10% of
   backtest.** Divergence beyond ~20% means investigate before paying for a
   Challenge — broker speed, spreads, or the netting/FIFO constraints.
5. Only then buy the paid Challenge.

Note that a breached FTMO account has no reset option and must be repurchased
at full price, which is the entire reason for step 4.

---

## Known gaps

Live trading is blocked on:

- **MT5 EA bridge not built.** `mt5_transport.py` defines the file-IPC
  contract, but the MetaTrader Expert Advisor that reads and writes those files
  does not exist yet. The adapter is unverified past `initialize`; its 54 tests
  run against an injected fake.
- **MT5 terminal** not yet installed on the Mac mini / FTMO US account.
- **FTMO US netting / FIFO / single-position-per-symbol** behaviour is not
  reconciled against backtester assumptions. The Free Trial is the intended
  place to catch any divergence.
- **Free Trial rule variant not configured.** The FTMO Free Trial is one phase
  with a 5% target and 2 minimum trading days; the paid Challenge is 10% and 4.
  Loss limits (5% daily / 10% total) are identical. `FirmRules.profit_targets`
  currently has `challenge` and `verification` only.

Data and configuration:

- Crypto and indices datasets are effectively empty.
- `CLAUDE_API_KEY` unset, blocking the adversarial-review gate.
- ~117 stored results predate the timezone fix. Run
  `audit_result_provenance.py --tag`, then decide discard vs re-run.
- Holdout cutoff never pinned (10 / 15 / 20%).
- Parameter-stability gate built but not wired into promotion.

Known behaviour worth remembering:

- **The compliance checker recomputes P&L from prices** and ignores the
  ledger's `pnl` column. Invisible when prices imply the P&L; silently
  substitutes its own number when they do not.
- **`multi_objective_optimizer` is selection, not parameter optimisation.** It
  chooses among already-backtested strategies. There is no parameter sweep
  anywhere in the codebase.
- Merge totals are **gross**; compliance totals are **net of fees**. The
  difference is correct.

---

## Stack

Python 3.12 / conda (`quant2`) · Backtrader · pandas / numpy · pymoo (NSGA-II)
· SQLite · React + TypeScript + Vite (dashboard) · SearXNG (Docker) · Ollama ·
FAISS · hypothesis ·
MetaTrader5