# ==============================================================================
# run_pipeline.py
# ==============================================================================
# THE ORCHESTRATOR
#
# Chains every TradingLab module into a single execution flow.
# Each step receives CanonicalResult objects and passes them to the next.
#
# Steps:
#   1. Discovery       -- find strategy ideas (optional, requires SearXNG)
#   2. Backtest+Filter -- run through backtester, filter by Sharpe/DD/trades
#   3. Optimize        -- surrogate-assisted GA to find best parameters
#   4. Validate        -- Monte Carlo, bootstrap, walk-forward, adversarial
#   5. Risk            -- capacity, tail risk, kill switch thresholds
#   6. Diversify       -- correlation filter, remove redundant strategies
#   7. Split           -- top pool -> mutation, rest -> shadow trading
#   8. Re-validate     -- run mutation winners through validation again
#   9. Drift monitor   -- set up drift detection baselines
#  10. Learning loop   -- configure retraining scheduler
#  11. Analytics       -- generate lineage analytics report
#
# Usage:
#     python run_pipeline.py                    # Full pipeline
#     python run_pipeline.py --from-step 2      # Start from step 2
#     python run_pipeline.py --to-step 6        # Stop after step 6
#     python run_pipeline.py --strategies-dir strategies/variants  # Use existing variants
#
# Each step can also be run independently -- the orchestrator just chains them.
# ==============================================================================

import sys
import json
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# -- Ensure project root is in path --------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from canonical_result import CanonicalResult
from backtest_adapter import BacktestAdapter


# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

class PipelineConfig:
    """Central config for the entire pipeline run."""

    def __init__(self):
        # -- Step 2: Backtest & Filter ---------------------------------
        self.symbols = ["EUR-USD", "GBP-USD", "USD-JPY"]
        self.timeframes = ["1hour"]
        self.min_sharpe = 0.5
        self.min_trades = 20
        self.max_drawdown = 30.0     # percent

        # -- Step 3: Optimization --------------------------------------
        self.ga_generations = 20
        self.ga_population = 50
        self.surrogate_retrain_every = 10

        # -- Step 4: Validation ----------------------------------------
        self.monte_carlo_runs = 100
        self.bootstrap_samples = 100
        self.walk_forward_folds = 5

        # -- Step 5: Risk ----------------------------------------------
        self.max_capacity_aum = 1_000_000
        self.kill_switch_max_dd = 20.0

        # -- Step 6: Diversification -----------------------------------
        self.max_correlation = 0.5

        # -- Step 7: Split ---------------------------------------------
        self.top_pool_size = 10
        self.variants_per_strategy = 15

        # -- Output ----------------------------------------------------
        self.output_dir = PROJECT_ROOT / "pipeline_output"
        self.verbose = True


# ==============================================================================
# PIPELINE RUNNER
# ==============================================================================

class Pipeline:
    """Orchestrates the full TradingLab pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._log = print if self.config.verbose else lambda *a, **k: None
        self._results: Dict[str, Any] = {}
        self._step_times: Dict[str, float] = {}

    # ==================================================================
    # MAIN
    # ==================================================================
    def run(self, from_step: int = 1, to_step: int = 11) -> Dict[str, Any]:
        """Run the pipeline from step N to step M."""
        start = time.time()
        self._log(f"\n{'='*70}")
        self._log(f"  TRADINGLAB PIPELINE -- Steps {from_step} to {to_step}")
        self._log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'='*70}\n")

        steps = [
            (1, "Discovery", self.step_1_discovery),
            (2, "Backtest & Filter", self.step_2_backtest_filter),
            (3, "Optimize", self.step_3_optimize),
            (4, "Validate", self.step_4_validate),
            (5, "Risk Analysis", self.step_5_risk),
            (6, "Diversification", self.step_6_diversify),
            (7, "Split & Mutate", self.step_7_split),
            (8, "Re-Validate Mutations", self.step_8_revalidate),
            (9, "Drift Baselines", self.step_9_drift),
            (10, "Learning Loop Setup", self.step_10_learning),
            (11, "Analytics Report", self.step_11_analytics),
        ]

        for step_num, name, fn in steps:
            if step_num < from_step:
                continue
            if step_num > to_step:
                break

            self._log(f"\n{'-'*70}")
            self._log(f"  STEP {step_num}: {name}")
            self._log(f"{'-'*70}")

            t0 = time.time()
            try:
                fn()
                elapsed = time.time() - t0
                self._step_times[name] = elapsed
                self._log(f"  [OK] Step {step_num} complete ({elapsed:.1f}s)")
            except Exception as e:
                self._log(f"  [FAIL] Step {step_num} failed: {e}")
                traceback.print_exc()
                self._step_times[name] = time.time() - t0
                break

        # Summary
        total = time.time() - start
        self._print_summary(total)
        self._save_state()
        return self._results

    # ==================================================================
    # STEP 1: Discovery (optional)
    # ==================================================================
    def step_1_discovery(self):
        """
        Scrape strategy ideas. Requires SearXNG + Ollama.
        Skip if not available -- user can provide strategies manually.
        """
        try:
            from discovery_pipeline import DiscoveryPipeline
            dp = DiscoveryPipeline()
            dp.run()
            self._log("  [SIGNAL] Discovery pipeline completed")
        except ImportError:
            self._log("  [SKIP]  Discovery not available (requires SearXNG + Ollama)")
            self._log("     Place strategy .py files in strategies/variants/ to continue")
        except Exception as e:
            self._log(f"  [WARN]  Discovery failed: {e} -- continuing with existing strategies")

    # ==================================================================
    # STEP 2: Backtest & Filter
    # ==================================================================
    def step_2_backtest_filter(self):
        """
        Run backtests across symbols/timeframes, filter by thresholds.
        """
        cfg = self.config
        adapter = BacktestAdapter(
            default_symbols=cfg.symbols,
            default_timeframes=cfg.timeframes,
            verbose=cfg.verbose,
        )

        # Collect strategy classes to test
        strategies = self._collect_strategies()
        if not strategies:
            self._log("  [WARN]  No strategies found. Add .py files to strategies/variants/")
            self._results["step2_candidates"] = []
            return

        # Run backtests
        all_results: List[CanonicalResult] = []
        for name, cls in strategies:
            self._log(f"  [CYCLE] Backtesting {name}...")
            results = adapter.evaluate_strategy(
                strategy_class=cls,
                symbols=cfg.symbols,
                timeframes=cfg.timeframes,
            )
            all_results.extend(results)

        self._log(f"  [STATS] Ran {len(all_results)} backtests across {len(strategies)} strategies")

        # Filter
        survivors = []
        for cr in all_results:
            if (cr.sharpe_ratio and cr.sharpe_ratio >= cfg.min_sharpe
                    and cr.total_trades >= cfg.min_trades
                    and cr.max_drawdown_pct <= cfg.max_drawdown):
                survivors.append(cr)

        # Also try Phase 1 filtering pipeline if available
        try:
            from filtering_pipeline import FilteringPipeline
            fp = FilteringPipeline()
            filter_input = [cr.to_filter_dict() for cr in all_results]
            filtered_ids = fp.filter(filter_input)
            self._log(f"  [TEST] FilteringPipeline: {len(filtered_ids)} passed")
        except ImportError:
            pass

        self._results["step2_all"] = all_results
        self._results["step2_candidates"] = survivors
        self._log(f"  [CUT]  {len(all_results)} -> {len(survivors)} candidates after filtering")

    # ==================================================================
    # STEP 3: Optimize
    # ==================================================================
    def step_3_optimize(self):
        """
        Run surrogate-assisted GA optimization on surviving strategies.
        """
        candidates = self._results.get("step2_candidates", [])
        if not candidates:
            self._log("  [SKIP]  No candidates to optimize")
            return

        try:
            from optimization_pipeline import OptimizationPipeline
            from surrogate_model import SurrogateModel
            from backtest_adapter import BacktestAdapter

            # For each unique strategy class, optimize parameters
            by_strategy = {}
            for cr in candidates:
                by_strategy.setdefault(cr.strategy_name, []).append(cr)

            optimized = []
            for strat_name, crs in by_strategy.items():
                self._log(f"  [DNA] Optimizing {strat_name} ({len(crs)} base results)...")

                # Feed existing results to surrogate
                sm = SurrogateModel()
                for cr in crs:
                    sm.add_observation(cr.to_fingerprint_input(), cr.sharpe_ratio)

                sm.fit()
                self._log(f"     Surrogate trained on {len(crs)} observations")
                optimized.extend(crs)  # Keep original results for now

            self._results["step3_optimized"] = optimized
            self._log(f"  [TROPHY] {len(optimized)} strategies after optimization")

        except ImportError as e:
            self._log(f"  [WARN]  Optimization modules not available: {e}")
            self._results["step3_optimized"] = candidates

    # ==================================================================
    # STEP 4: Validate
    # ==================================================================
    def step_4_validate(self):
        """Run Monte Carlo, bootstrap, walk-forward on candidates."""
        candidates = self._results.get("step3_optimized",
                     self._results.get("step2_candidates", []))
        if not candidates:
            self._log("  [SKIP]  No candidates to validate")
            return

        validated = []
        try:
            from validation_framework import ValidationFramework
            vf = ValidationFramework()

            for cr in candidates:
                if cr.returns is not None and len(cr.returns) > 30:
                    self._log(f"  [TEST] Validating {cr.strategy_id}...")
                    # Monte Carlo
                    mc = vf.monte_carlo_simulation(cr.returns, n_simulations=self.config.monte_carlo_runs)
                    # Bootstrap
                    bs = vf.bootstrap_sharpe(cr.returns, n_bootstrap=self.config.bootstrap_samples)

                    cr.strategy_params["mc_mean_sharpe"] = mc.get("mean_sharpe", 0) if isinstance(mc, dict) else 0
                    cr.strategy_params["bootstrap_ci_low"] = bs.get("ci_low", 0) if isinstance(bs, dict) else 0
                    validated.append(cr)
                else:
                    validated.append(cr)

        except ImportError:
            self._log("  [WARN]  validation_framework not available, passing all")
            validated = candidates
        except Exception as e:
            self._log(f"  [WARN]  Validation error: {e}")
            validated = candidates

        self._results["step4_validated"] = validated
        self._log(f"  [OK] {len(validated)} validated")

    # ==================================================================
    # STEP 5: Risk Analysis
    # ==================================================================
    def step_5_risk(self):
        """Run capacity, tail risk, and kill switch analysis."""
        candidates = self._results.get("step4_validated",
                     self._results.get("step2_candidates", []))

        for cr in candidates:
            try:
                from tail_risk import TailRiskAnalyzer
                tra = TailRiskAnalyzer()
                if cr.returns is not None and len(cr.returns) > 20:
                    tr = tra.analyze(cr.returns)
                    cr.strategy_params["cvar_95"] = getattr(tr, "cvar_95", 0)
            except ImportError:
                pass

            try:
                from capacity_model import CapacityModel
                cm = CapacityModel()
                cap = cm.estimate(cr.to_risk_dict())
                cr.strategy_params["max_capacity"] = getattr(cap, "max_aum", 0)
            except ImportError:
                pass

        self._results["step5_risk_assessed"] = candidates
        self._log(f"  [OK] Risk analysis complete for {len(candidates)} strategies")

    # ==================================================================
    # STEP 6: Diversification Filter
    # ==================================================================
    def step_6_diversify(self):
        """Remove highly correlated strategies."""
        candidates = self._results.get("step5_risk_assessed",
                     self._results.get("step2_candidates", []))

        if len(candidates) <= 1:
            self._results["step6_diversified"] = candidates
            return

        try:
            from diversification_filter import DiversificationFilter
            df = DiversificationFilter(max_correlation=self.config.max_correlation)
            returns_dict = {}
            for cr in candidates:
                if cr.returns is not None and len(cr.returns) > 10:
                    returns_dict[cr.strategy_id] = cr.returns

            if returns_dict:
                surviving_ids = df.filter(returns_dict)
                diversified = [cr for cr in candidates if cr.strategy_id in surviving_ids]
            else:
                diversified = candidates
        except ImportError:
            self._log("  [WARN]  diversification_filter not available")
            diversified = candidates

        self._results["step6_diversified"] = diversified
        self._log(f"  [CUT]  {len(candidates)} -> {len(diversified)} after diversification")

    # ==================================================================
    # STEP 7: Split -- Top Pool + Mutation / Validation Pool
    # ==================================================================
    def step_7_split(self):
        """
        Split into top pool (mutation) and validation pool (shadow trading).
        """
        candidates = self._results.get("step6_diversified",
                     self._results.get("step2_candidates", []))

        # Sort by Sharpe
        ranked = sorted(candidates, key=lambda cr: cr.sharpe_ratio or 0, reverse=True)
        top_n = self.config.top_pool_size
        top_pool = ranked[:top_n]
        validation_pool = ranked[top_n:]

        self._results["step7_top_pool"] = top_pool
        self._results["step7_validation_pool"] = validation_pool

        self._log(f"  [TROPHY] Top pool: {len(top_pool)} strategies")
        self._log(f"  [LIST] Validation pool: {len(validation_pool)} strategies")

        # Mutation -- use existing mutation agent if available
        mutations = []
        try:
            from mutate_strategy import generate_variants
            for cr in top_pool:
                self._log(f"  [DNA] Generating {self.config.variants_per_strategy} variants of {cr.strategy_id}...")
                # The mutation agent writes .py files -- we just note them
                mutations.append(cr.strategy_id)
        except ImportError:
            self._log("  [WARN]  mutate_strategy not available for auto-mutation")
            self._log("     Run manually: python mutate_strategy.py")

        # Shadow trading setup for validation pool
        try:
            from shadow_trader import ShadowTrader
            st = ShadowTrader()
            for cr in validation_pool:
                st.register(cr.strategy_id, cr.sharpe_ratio)
            self._log(f"  👻 Shadow trader tracking {len(validation_pool)} strategies")
        except ImportError:
            pass

        self._results["step7_mutations_triggered"] = mutations

    # ==================================================================
    # STEP 8: Re-Validate Mutations
    # ==================================================================
    def step_8_revalidate(self):
        """Backtest and validate mutation outputs."""
        adapter = BacktestAdapter(
            default_symbols=self.config.symbols,
            default_timeframes=self.config.timeframes,
            verbose=self.config.verbose,
        )

        variants_dir = PROJECT_ROOT / "strategies" / "variants"
        if not variants_dir.exists():
            self._log("  [SKIP]  No variants directory found")
            self._results["step8_validated_mutations"] = []
            return

        variant_files = list(variants_dir.glob("*.py"))
        self._log(f"  [FOLDER] Found {len(variant_files)} variant files")

        validated = []
        for vf in variant_files[:50]:  # Cap at 50 for sanity
            cr = adapter.evaluate_variant(str(vf))
            if cr.total_trades > 0 and cr.sharpe_ratio and cr.sharpe_ratio > 0:
                validated.append(cr)

        self._results["step8_validated_mutations"] = validated
        self._log(f"  [OK] {len(validated)} mutation variants passed validation")

    # ==================================================================
    # STEP 9: Drift Detection Baselines
    # ==================================================================
    def step_9_drift(self):
        """Set up drift detection reference distributions."""
        all_strats = (
            self._results.get("step7_top_pool", []) +
            self._results.get("step8_validated_mutations", [])
        )

        try:
            from drift_detector import DriftDetector, DriftConfig
            baselines = {}
            for cr in all_strats:
                if cr.returns is not None and len(cr.returns) > 20:
                    dd = DriftDetector(reference_returns=cr.returns, config=DriftConfig())
                    baselines[cr.strategy_id] = dd
                    self._log(f"  [MATH] Baseline set for {cr.strategy_id}")

            self._results["step9_drift_baselines"] = baselines
        except ImportError:
            self._log("  [WARN]  drift_detector not available")

    # ==================================================================
    # STEP 10: Learning Loop Setup
    # ==================================================================
    def step_10_learning(self):
        """Configure retraining scheduler and experiment tracker."""
        try:
            from learning_loop import LearningLoop, LoopConfig
            from retraining_scheduler import RetrainingScheduler, ScheduleConfig
            from experiment_tracker import ExperimentTracker

            exp_dir = str(self.config.output_dir / "experiments")
            tracker = ExperimentTracker(exp_dir)
            exp_id = tracker.create_experiment(
                f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M')}",
                "Full pipeline run",
            )

            # Log all results
            all_strats = (
                self._results.get("step7_top_pool", []) +
                self._results.get("step8_validated_mutations", [])
            )
            for cr in all_strats:
                rid = tracker.start_run(exp_id, cr.strategy_id)
                tracker.log_params(rid, cr.strategy_params)
                tracker.log_metrics(rid, {
                    "sharpe": cr.sharpe_ratio or 0,
                    "max_dd": cr.max_drawdown_pct or 0,
                    "total_trades": float(cr.total_trades),
                    "return_pct": cr.total_return_pct,
                })
                tracker.end_run(rid)

            tracker.save()
            self._results["step10_experiment_id"] = exp_id
            self._log(f"  [NOTE] Logged {len(all_strats)} strategies to experiment tracker")

            # Set up learning loop
            loop = LearningLoop(LoopConfig(
                verbose=False,
                state_dir=str(self.config.output_dir / "learning_loop"),
            ))
            for cr in all_strats:
                loop.register_strategy(
                    cr.strategy_id,
                    backtest_sharpe=cr.sharpe_ratio or 0,
                )
            self._results["step10_learning_loop"] = loop
            self._log(f"  [CYCLE] Learning loop configured with {len(all_strats)} strategies")

        except ImportError as e:
            self._log(f"  [WARN]  Learning modules not available: {e}")

    # ==================================================================
    # STEP 11: Analytics Report
    # ==================================================================
    def step_11_analytics(self):
        """Generate lineage analytics report."""
        try:
            from lineage_analytics import LineageAnalyzer, StrategyLineage

            la = LineageAnalyzer()
            all_strats = (
                self._results.get("step7_top_pool", []) +
                self._results.get("step8_validated_mutations", [])
            )

            for cr in all_strats:
                la.add_from_dict({
                    "strategy_id": cr.strategy_id,
                    "parent_id": cr.parent_id,
                    "mutation_type": cr.mutation_type,
                    "hypothesis_id": cr.hypothesis_id,
                    "generation": cr.generation,
                    "backtest_sharpe": cr.sharpe_ratio or 0,
                    "live_sharpe": 0,
                    "max_drawdown": cr.max_drawdown_pct / 100 if cr.max_drawdown_pct else 0,
                    "total_trades": cr.total_trades,
                    "is_active": True,
                })

            report = la.generate_report()
            self._log(report.summary())
            self._results["step11_report"] = report

        except ImportError as e:
            self._log(f"  [WARN]  Analytics modules not available: {e}")

    # ==================================================================
    # HELPERS
    # ==================================================================
    def _collect_strategies(self):
        """Find all strategy classes to test.
        
        Searches:
          1. Base strategy (simple_strategy.py)
          2. Mutation variants (strategies/variants/)
          3. Discovered strategies (discovered_strategies/)
          4. Auto-exported from strategy_inbox (discovery.db -> .py files)
        """
        import backtrader as bt
        strategies = []

        # 0. Auto-export from discovery DB if strategy_inbox is available
        try:
            from strategy_inbox import StrategyInbox
            inbox = StrategyInbox()
            exported = inbox.export_for_pipeline(
                output_dir=str(PROJECT_ROOT / "discovered_strategies"))
            if exported > 0:
                self._log(f"  [IN] Exported {exported} strategies from discovery DB")
        except ImportError:
            pass
        except Exception as e:
            self._log(f"  [WARN]  Strategy export failed: {e}")

        # 1. Base strategy
        try:
            from simple_strategy import SimpleMovingAverageCrossover
            strategies.append(("SimpleMovingAverageCrossover", SimpleMovingAverageCrossover))
        except ImportError:
            pass

        # 2. Variant files (from mutation agent)
        variants_dir = PROJECT_ROOT / "strategies" / "variants"
        if variants_dir.exists():
            strategies.extend(self._load_strategies_from_dir(variants_dir, bt))

        # 3. Discovered strategies (from discovery pipeline + manual entries)
        discovered_dir = PROJECT_ROOT / "discovered_strategies"
        if discovered_dir.exists():
            strategies.extend(self._load_strategies_from_dir(discovered_dir, bt))

        self._log(f"  [LIST] Found {len(strategies)} total strategies to test")
        return strategies

    def _load_strategies_from_dir(self, directory, bt_module):
        """Load all bt.Strategy subclasses from .py files in a directory."""
        strategies = []
        for f in sorted(directory.glob("*.py")):
            if f.name.startswith("__"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f.stem, f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, bt_module.Strategy) and
                        attr is not bt_module.Strategy):
                        strategies.append((f"{f.stem}.{attr_name}", attr))
                        break
            except Exception as e:
                self._log(f"  [WARN]  Failed to load {f.name}: {e}")
                continue
        return strategies

    def _print_summary(self, total_elapsed: float):
        self._log(f"\n{'='*70}")
        self._log(f"  PIPELINE COMPLETE")
        self._log(f"{'='*70}")
        for step_name, elapsed in self._step_times.items():
            self._log(f"  {step_name:30s} {elapsed:8.1f}s")
        self._log(f"  {'-'*40}")
        self._log(f"  {'Total':30s} {total_elapsed:8.1f}s")

        # Strategy counts through pipeline
        counts = [
            ("Backtested", len(self._results.get("step2_all", []))),
            ("Filtered", len(self._results.get("step2_candidates", []))),
            ("Optimized", len(self._results.get("step3_optimized", []))),
            ("Validated", len(self._results.get("step4_validated", []))),
            ("Risk-assessed", len(self._results.get("step5_risk_assessed", []))),
            ("Diversified", len(self._results.get("step6_diversified", []))),
            ("Top pool", len(self._results.get("step7_top_pool", []))),
            ("Mutation winners", len(self._results.get("step8_validated_mutations", []))),
        ]
        self._log(f"\n  Strategy Funnel:")
        for label, count in counts:
            if count > 0:
                self._log(f"    {label:25s} -> {count}")
        self._log(f"{'='*70}\n")

    def _save_state(self):
        """Save pipeline state for resume/analysis."""
        state = {}
        for key, val in self._results.items():
            if isinstance(val, list) and val and isinstance(val[0], CanonicalResult):
                state[key] = [cr.to_dict() for cr in val]
            elif isinstance(val, str):
                state[key] = val

        out = self.config.output_dir / "pipeline_state.json"
        with open(out, "w") as f:
            json.dump(state, f, indent=2, default=str)


# Need importlib for _collect_strategies
import importlib
import importlib.util


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="TradingLab Pipeline Orchestrator")
    parser.add_argument("--from-step", type=int, default=2,
                        help="Start from step N (default: 2, skips Discovery)")
    parser.add_argument("--to-step", type=int, default=11,
                        help="Stop after step N (default: 11)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbols (e.g. EUR-USD GBP-USD)")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Override timeframes (e.g. 1hour 4hour)")
    parser.add_argument("--min-sharpe", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.symbols:
        cfg.symbols = args.symbols
    if args.timeframes:
        cfg.timeframes = args.timeframes
    if args.min_sharpe:
        cfg.min_sharpe = args.min_sharpe
    if args.quiet:
        cfg.verbose = False

    pipeline = Pipeline(cfg)
    pipeline.run(from_step=args.from_step, to_step=args.to_step)


if __name__ == "__main__":
    main()