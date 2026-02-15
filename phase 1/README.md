# Phase 1: Foundation Completion

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           phase1_pipeline.py                │
                    │         (Integration Orchestrator)          │
                    ├──────────┬──────────┬───────────┬───────────┤
                    │  Step 1  │  Step 2  │  Step 3   │  Step 4   │
                    │ Register │   PBO    │  Filter   │ Diversify │
                    │          │ Analysis │  & Rank   │ & Select  │
                    ├──────────┼──────────┼───────────┼───────────┤
                    │ Module 1 │ Module 2 │ Module 3  │ Module 4  │
                    │ lineage_ │ overfit_ │ filtering_│ diversif_ │
                    │ tracker  │ detector │ pipeline  │ filter    │
                    └──────┬───┴────┬─────┴─────┬─────┴─────┬─────┘
                           │        │           │           │
                    ┌──────┴───┐ ┌──┴────┐ ┌────┴───┐ ┌────┴───┐
                    │  SQLite  │ │ scipy │ │ numpy  │ │ scipy  │
                    │  MLflow  │ │joblib │ │        │ │cluster │
                    │          │ │quant- │ │        │ │        │
                    │          │ │ stats │ │        │ │        │
                    └──────────┘ └───────┘ └────────┘ └────────┘
```

## Data Flow

```
strategies + returns (from Phase 5 Discovery or existing backtests)
    │
    ▼
Module 1: Register in lineage tracker (SQLite + MLflow)
    │
    ▼
Module 2: PBO analysis (CSCV on returns matrix)
    │
    ▼
Module 3: Hard filters → composite score → rank → top N
    │    (Sharpe ≥ 0.3, DD ≤ 30%, trades ≥ 30, PBO ≤ 0.5)
    │    500 strategies → 50–100 survivors
    ▼
Module 4: Correlation filter → greedy diversification
    │    max pairwise |ρ| < 0.5, greedy by score
    │    50–100 → final pool (ready for Phase 2)
    ▼
Output: Diversified strategy pool + lineage DB + JSON reports
```

## Four Modules

### Module 1: `lineage_tracker.py` — Strategy Genealogy

| Item       | Detail                                                              |
|------------|---------------------------------------------------------------------|
| **Role**   | Tracks every strategy's ancestry, mutations, and performance        |
| **Inputs** | Strategy metadata, backtest metrics dicts                           |
| **Outputs**| StrategyRecord, FamilyNode trees, mutation analytics                |
| **GitHub** | [mlflow/mlflow](https://github.com/mlflow/mlflow) — experiment tracking, parent/child run linking |

Key methods:
- `register_strategy(name, origin, parent_id, mutation_type)` → strategy_id
- `log_backtest(strategy_id, metrics)` → row_id
- `get_family_tree(root_id)` → nested FamilyNode
- `get_mutation_success_rates()` → {mutation_type: {count, avg_sharpe, ...}}
- `update_status(strategy_id, status)` — pending→backtested→filtered→promoted→retired

### Module 2: `overfitting_detector.py` — PBO & Deflated Sharpe

| Item       | Detail                                                              |
|------------|---------------------------------------------------------------------|
| **Role**   | Detects backtest overfitting using statistical methods               |
| **Inputs** | Returns DataFrame (T×N), observed Sharpe, trial count               |
| **Outputs**| PBOResult, DSRResult, PSRResult, HTML tearsheets                    |
| **GitHub** | Algorithm from Bailey et al. (2015) paper. [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) for tearsheets |

Key methods:
- `compute_pbo(returns_df, n_partitions)` → PBOResult (probability, logits, degradation)
- `deflated_sharpe_ratio(sr, n_trials, T)` → DSRResult (deflated SR, p-value)
- `probabilistic_sharpe_ratio(returns)` → PSRResult
- `analyze_strategy(returns, n_trials)` → combined dict
- `generate_tearsheet(returns, output_path)` → HTML file

### Module 3: `filtering_pipeline.py` — Threshold Filters & Ranking

| Item       | Detail                                                              |
|------------|---------------------------------------------------------------------|
| **Role**   | Chains hard filters → composite scoring → top-N selection           |
| **Inputs** | List of strategy dicts with metrics, FilterConfig thresholds        |
| **Outputs**| PipelineResult with ranked survivors and rejection reasons          |
| **GitHub** | No external repos — pure orchestration of Modules 1 & 2            |

Key methods:
- `run(strategies, config, returns_matrix)` → PipelineResult
- `save_results(result, path)` → JSON file
- `load_from_database(db_path)` → list of strategy dicts

### Module 4: `diversification_filter.py` — Correlation & Redundancy Removal

| Item       | Detail                                                              |
|------------|---------------------------------------------------------------------|
| **Role**   | Removes redundant strategies via correlation + trade overlap        |
| **Inputs** | Survivor list from Module 3, returns dict, DiversityConfig          |
| **Outputs**| DiversificationResult with selected/removed, correlation matrix     |
| **GitHub** | No external repos — numpy/scipy correlation + hierarchical clustering |

Key methods:
- `run(strategies, returns_dict, trade_dates_dict, config)` → DiversificationResult
- `compute_trade_overlap(dates_a, dates_b)` → Jaccard similarity float

## Module Interconnections

```
Module 1 (LineageTracker)
    ↑ writes status         ↑ writes status
    │                       │
Module 3 (FilterPipeline)   Module 4 (DiversityFilter)
    ↑ calls PBO/DSR             ↑ receives survivors
    │                           │
Module 2 (OverfitDetector)  Module 3 (FilterPipeline)
```

- Module 3 injects Module 1 (lineage_tracker) and Module 2 (overfitting_detector)
- Module 4 injects Module 1 (lineage_tracker) for status updates
- Phase1Pipeline wires all four together

## Project Structure

```
phase1/
├── lineage_tracker.py           # Module 1: Strategy genealogy (747 lines)
├── overfitting_detector.py      # Module 2: PBO + DSR + PSR (280 lines)
├── filtering_pipeline.py        # Module 3: Filters + scoring (260 lines)
├── diversification_filter.py    # Module 4: Correlation filter (280 lines)
├── phase1_pipeline.py           # Integration orchestrator (210 lines)
├── test_phase1.py               # 32 tests across all modules (500 lines)
├── requirements_phase1.txt      # Dependencies
└── README.md                    # This file
```

Total: ~2,280 lines across 6 Python files.

## Setup Instructions

### Prerequisites

- Python 3.9+
- pip

### 1. Install Dependencies

```bash
pip install -r requirements_phase1.txt
```

Or manually:
```bash
pip install numpy pandas scipy scikit-learn mlflow quantstats statsmodels joblib matplotlib seaborn
```

### 2. Configuration

No configuration needed for standalone testing. The modules auto-detect paths:

- **LineageTracker** defaults to `./data/lineage.db` (SQLite)
- **MLflow** defaults to `./data/mlruns/` (local file store)
- When integrated with the main project, modules import `config.py` for `BASE_DIR`

### 3. Run Tests

```bash
# Full suite (32 tests, ~10 seconds)
python test_phase1.py

# Quick mode (8 critical tests, ~3 seconds)
python test_phase1.py --quick
```

### 4. Run the Pipeline

```python
from phase1_pipeline import Phase1Pipeline
from filtering_pipeline import FilterConfig
from diversification_filter import DiversityConfig

pipeline = Phase1Pipeline(
    db_path="data/lineage.db",
    enable_mlflow=True,       # Set False to skip MLflow logging
    filter_config=FilterConfig(
        min_sharpe=0.3,
        max_drawdown=30.0,
        min_trades=50,
        max_pbo=0.5,
        top_n=100,
    ),
    diversity_config=DiversityConfig(
        max_correlation=0.5,
        max_strategies=50,
    ),
)

result = pipeline.run(
    strategies=your_strategy_list,    # list of dicts with metrics
    returns_dict=your_returns_dict,   # {strategy_id: np.array}
)

print(result.summary())
```

### 5. Browse MLflow Dashboard (Optional)

```bash
mlflow ui --backend-store-uri file:///path/to/data/mlruns
# Open http://localhost:5000
```

### 6. CLI Tools

```bash
# Lineage tracker CLI
python lineage_tracker.py --summary --db data/lineage.db
python lineage_tracker.py --show-tree STRATEGY_ID
python lineage_tracker.py --mutation-stats
python lineage_tracker.py --generation-stats
```

## Integration with Existing TradingLab

Copy the 4 module files into the project root alongside `config.py`:

```bash
cp lineage_tracker.py overfitting_detector.py \
   filtering_pipeline.py diversification_filter.py \
   phase1_pipeline.py /path/to/TradingLab/
```

The modules auto-import `config.py` for `BASE_DIR` when available,
and fall back to `Path(__file__).parent` when running standalone.

**Connecting to existing backtests:**
```python
from filtering_pipeline import FilteringPipeline
pipe = FilteringPipeline()
strategies = pipe.load_from_database("results/backtest_results.db")
```

## Assumptions

1. **Returns are daily.** Sharpe annualization uses √252. Adjust if using
   intraday bars (√(252*bars_per_day)) or weekly (√52).

2. **PBO requires ≥ 4 strategies and ≥ 200 return observations.** Below
   these thresholds, PBO is skipped and strategies are filtered on
   hard metrics only.

3. **pypbo is not used as a dependency.** The CSCV/PBO algorithm is
   implemented directly from the Bailey et al. (2015) paper. This
   eliminates dependence on an unmaintained package (esvhd/pypbo is
   not on PyPI and has stale deps).

4. **SQLite is the sole database.** No PostgreSQL or external DB server
   needed. MLflow uses a local file-based tracking store.

5. **All filtering thresholds are configurable.** The defaults
   (Sharpe ≥ 0.3, DD ≤ 30%, trades ≥ 30, PBO ≤ 0.5) are conservative
   starting points. Tune via FilterConfig.

6. **Strategies are passed as plain dicts.** This matches the existing
   database.py and results_analyzer.py patterns. No class wrappers needed.

7. **Phase 2 consumes the output directly.** The diversified pool from
   `DiversificationResult.selected` is a list of dicts ready to serve
   as the initial population for NSGA-II.
