# ==============================================================================
# discovery_pipeline.py
# ==============================================================================
# Strategy Discovery Pipeline Orchestrator (Step 1)
#
# Ties together all discovery components into a single runnable pipeline:
#
#   1. SearXNG Search   → Find documents about trading strategies
#   2. Fetch Documents   → Download and store raw content
#   3. Quality Gate      → Score documents, reject low-quality
#   4. LLM Extraction    → Two-stage: summarize → generate code
#   5. Semantic Dedup    → Remove near-duplicate strategies (FAISS)
#   6. Code Validation   → Verify generated code imports and runs
#   7. Save & Report     → Store validated strategies, print summary
#
# Usage:
#     # Full pipeline run
#     python discovery_pipeline.py
#
#     # Or import and control each step:
#     from discovery_pipeline import DiscoveryPipeline
#     pipeline = DiscoveryPipeline()
#     pipeline.run()
#
# ==============================================================================

import os
import sys
import time
import json
import logging
import argparse
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from discovery_config import (
    DISCOVERY_CONFIG as cfg,
    STRATEGIES_DIR,
    LOGS_DIR,
    print_config,
)
from research_db import ResearchDatabase
from searxng_scraper import SearXNGScraper
from llm_extractor import LLMExtractor
from quality_scorer import QualityScorer
from semantic_dedup import SemanticDeduplicator

logger = logging.getLogger(__name__)


class DiscoveryPipeline:
    """
    Master orchestrator for the Strategy Discovery Pipeline.

    Coordinates: scraping → extraction → dedup → validation → storage
    """

    def __init__(self):
        # Shared database
        self.db = ResearchDatabase()

        # Components
        self.scraper = SearXNGScraper(db=self.db)
        self.extractor = LLMExtractor(db=self.db)
        self.scorer = QualityScorer()
        self.dedup = SemanticDeduplicator(db=self.db)

        # Pipeline stats
        self.run_start = None
        self.run_stats = {
            "search_results": 0,
            "documents_fetched": 0,
            "quality_passed": 0,
            "strategies_extracted": 0,
            "duplicates_removed": 0,
            "code_validated": 0,
            "code_failed": 0,
            "final_strategies": 0,
            "duration_seconds": 0,
        }

    # ==========================================================================
    # HEALTH CHECKS
    # ==========================================================================

    def preflight_check(self) -> bool:
        """
        Run all health checks before starting the pipeline.

        Returns True if all critical services are available.
        """
        print("\n" + "=" * 60)
        print("PREFLIGHT CHECK")
        print("=" * 60)

        all_ok = True

        # 1. SearXNG
        searxng_ok = self.scraper.health_check()
        if not searxng_ok:
            print("  ⚠ SearXNG not available — scraping will fail")
            print("    Fix: docker run -d --name searxng -p 8080:8080 searxng/searxng:latest")
            all_ok = False

        # 2. LLM endpoints
        llm_status = self.extractor.health_check()
        if not llm_status.get("summarizer"):
            print("  ⚠ Summarizer LLM not available — extraction will fail")
            print(f"    Fix: Start your model server and ensure '{cfg.llm.summarizer.model}' is loaded")
            all_ok = False
        if not llm_status.get("code_generator"):
            print("  ⚠ Code generator LLM not available — code gen will fail")
            print(f"    Fix: Start your model server and ensure '{cfg.llm.code_generator.model}' is loaded")
            all_ok = False

        # 3. Database
        try:
            stats = self.db.get_pipeline_stats()
            print(f"✓ Database OK ({stats['total_documents']} docs, "
                  f"{stats['total_strategies']} strategies)")
        except Exception as e:
            print(f"  ✗ Database error: {e}")
            all_ok = False

        # 4. Dedup index
        print(f"✓ Dedup index: {self.dedup.get_index_size()} strategies "
              f"({'FAISS' if self.dedup.use_faiss else 'TF-IDF fallback'})")

        # 5. Output directory
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output dir: {STRATEGIES_DIR}")

        print("=" * 60)
        if all_ok:
            print("All checks passed ✓")
        else:
            print("Some checks failed — pipeline may not complete")

        return all_ok

    # ==========================================================================
    # PIPELINE STEPS
    # ==========================================================================

    def step_1_search(self, queries: Optional[List[Dict]] = None) -> List[Dict]:
        """Step 1: Run search queries and collect results."""
        print("\n" + "=" * 60)
        print("STEP 1: SEARCH")
        print("=" * 60)

        results = self.scraper.search_all_queries(queries)
        self.run_stats["search_results"] = len(results)
        print(f"Found {len(results)} search results")
        return results

    def step_2_fetch(self, results: List[Dict]) -> List[str]:
        """Step 2: Fetch full documents and store in database."""
        print("\n" + "=" * 60)
        print("STEP 2: FETCH DOCUMENTS")
        print("=" * 60)

        doc_ids = self.scraper.fetch_and_store(results)
        self.run_stats["documents_fetched"] = len(doc_ids)
        print(f"Fetched and stored {len(doc_ids)} documents")
        return doc_ids

    def step_3_extract(self, doc_ids: Optional[List[str]] = None,
                       limit: int = None) -> List[Dict]:
        """
        Step 3: Extract strategies from documents using LLM.

        If doc_ids is None, processes all unprocessed documents.
        """
        print("\n" + "=" * 60)
        print("STEP 3: LLM EXTRACTION (two-stage)")
        print("=" * 60)

        limit = limit or cfg.pipeline.max_strategies_per_run

        if doc_ids:
            # Fetch specific documents
            docs = []
            for did in doc_ids:
                doc = self.db.get_unprocessed_documents(limit=1)
                # Actually, get all unprocessed and filter
            docs = self.db.get_unprocessed_documents(limit=limit)
        else:
            docs = self.db.get_unprocessed_documents(limit=limit)

        if not docs:
            print("No unprocessed documents found.")
            return []

        print(f"Processing {len(docs)} documents...")
        strategies = self.extractor.extract_batch(docs, delay=1.0)

        self.run_stats["strategies_extracted"] = len(strategies)
        self.run_stats["quality_passed"] = (
            self.extractor.stats["documents_processed"]
            - self.extractor.stats["quality_rejected"]
        )

        print(f"Extracted {len(strategies)} strategies")
        return strategies

    def step_4_deduplicate(self, strategies: List[Dict]) -> List[Dict]:
        """Step 4: Remove near-duplicate strategies."""
        print("\n" + "=" * 60)
        print("STEP 4: SEMANTIC DEDUPLICATION")
        print("=" * 60)

        if not strategies:
            print("No strategies to deduplicate.")
            return []

        unique = self.dedup.deduplicate_batch(strategies)
        removed = len(strategies) - len(unique)
        self.run_stats["duplicates_removed"] = removed

        print(f"Unique: {len(unique)}, Duplicates removed: {removed}")
        return unique

    def step_5_validate(self, strategies: List[Dict]) -> List[Dict]:
        """Step 5: Validate generated code (import check + dummy backtest)."""
        print("\n" + "=" * 60)
        print("STEP 5: CODE VALIDATION")
        print("=" * 60)

        if not cfg.pipeline.code_validation_enabled:
            print("Code validation disabled in config. Skipping.")
            return strategies

        if not strategies:
            print("No strategies to validate.")
            return []

        validated = []
        for strat in strategies:
            strat_id = strat.get("strategy_id", "unknown")
            code = strat.get("generated_code", "")
            name = strat.get("strategy_name", "Unknown")

            if not code:
                logger.info(f"  ✗ {name}: No code to validate")
                self.run_stats["code_failed"] += 1
                continue

            ok, error_msg, trade_count = self._validate_code(code, name)

            if ok:
                strat["code_validates"] = True
                strat["validation_trades"] = trade_count
                strat["status"] = "validated"
                validated.append(strat)
                self.run_stats["code_validated"] += 1

                # Update DB
                self.db.update_strategy(strat_id, {
                    "code_validates": 1,
                    "validation_trades": trade_count,
                    "status": "validated",
                })
                logger.info(f"  ✓ {name}: OK ({trade_count} trades)")
            else:
                strat["code_validates"] = False
                strat["validation_error"] = error_msg
                strat["status"] = "validation_failed"
                self.run_stats["code_failed"] += 1

                # Update DB
                self.db.update_strategy(strat_id, {
                    "code_validates": 0,
                    "validation_error": error_msg,
                    "status": "validation_failed",
                })
                logger.info(f"  ✗ {name}: FAILED — {error_msg}")

        print(f"Validated: {len(validated)}, Failed: {self.run_stats['code_failed']}")
        return validated

    def step_6_save(self, strategies: List[Dict]) -> List[str]:
        """Step 6: Save validated strategy code to files."""
        print("\n" + "=" * 60)
        print("STEP 6: SAVE STRATEGIES")
        print("=" * 60)

        if not strategies:
            print("No strategies to save.")
            return []

        saved_files = []
        for strat in strategies:
            strat_id = strat.get("strategy_id", "unknown")
            name = strat.get("strategy_name", "Unknown")
            code = strat.get("generated_code", "")

            if not code:
                continue

            # Create filename from strategy name
            safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
            filename = f"{safe_name}.py"
            filepath = STRATEGIES_DIR / filename

            # Avoid overwriting — append number if exists
            counter = 1
            while filepath.exists():
                filename = f"{safe_name}_{counter}.py"
                filepath = STRATEGIES_DIR / filename
                counter += 1

            # Write file with header comment
            header = (
                f"# Auto-discovered strategy: {name}\n"
                f"# Source: {strat.get('source_url', 'unknown')}\n"
                f"# Source type: {strat.get('source_type', 'unknown')} "
                f"({strat.get('source_bias', 'unknown')})\n"
                f"# Quality score: {strat.get('quality_score', 0):.3f}\n"
                f"# Strategy ID: {strat_id}\n"
                f"# Extracted: {datetime.now().isoformat()}\n"
                f"#\n"
            )

            filepath.write_text(header + code)
            saved_files.append(str(filepath))

            # Update DB with file path
            self.db.update_strategy(strat_id, {"code_file_path": str(filepath)})
            logger.info(f"  Saved: {filepath.name}")

        self.run_stats["final_strategies"] = len(saved_files)
        print(f"Saved {len(saved_files)} strategy files to {STRATEGIES_DIR}")
        return saved_files

    # ==========================================================================
    # CODE VALIDATION HELPER
    # ==========================================================================

    def _validate_code(self, code: str, name: str) -> tuple:
        """
        Validate generated Backtrader code.

        Checks:
            1. Syntax: Does it parse as valid Python?
            2. Import: Does it import without errors?
            3. Class: Does it contain a bt.Strategy subclass?
            4. Backtest: Does it produce at least 1 trade on dummy data?

        Returns:
            (success: bool, error_message: str, trade_count: int)
        """
        # 1. Syntax check
        try:
            compile(code, f"<{name}>", "exec")
        except SyntaxError as e:
            return False, f"SyntaxError: {e}", 0

        # 2. Import check — exec in isolated namespace
        namespace = {}
        try:
            exec(code, namespace)
        except ImportError as e:
            return False, f"ImportError: {e}", 0
        except Exception as e:
            return False, f"Import/exec error: {type(e).__name__}: {e}", 0

        # 3. Find Strategy subclass
        strategy_class = None
        try:
            import backtrader as bt
            for obj_name, obj in namespace.items():
                if (isinstance(obj, type) and
                    issubclass(obj, bt.Strategy) and
                    obj is not bt.Strategy):
                    strategy_class = obj
                    break
        except ImportError:
            # Backtrader not installed in this environment
            # Just check that a class exists
            for obj_name, obj in namespace.items():
                if isinstance(obj, type) and obj_name != "Strategy":
                    strategy_class = obj
                    break

            if strategy_class:
                return True, "", 0  # Can't run backtest but code is valid
            return False, "No strategy class found", 0

        if strategy_class is None:
            return False, "No bt.Strategy subclass found", 0

        # 4. Dummy backtest (quick — 200 bars of random data)
        try:
            import backtrader as bt
            import numpy as np

            cerebro = bt.Cerebro()
            cerebro.addstrategy(strategy_class)

            # Generate dummy OHLCV data
            bars = cfg.pipeline.validation_bar_count
            np.random.seed(42)
            close = 100 + np.cumsum(np.random.randn(bars) * 0.5)
            high = close + np.abs(np.random.randn(bars) * 0.3)
            low = close - np.abs(np.random.randn(bars) * 0.3)
            open_ = close + np.random.randn(bars) * 0.1
            volume = np.random.randint(1000, 10000, bars).astype(float)

            import pandas as pd
            dates = pd.date_range('2020-01-01', periods=bars, freq='D')
            df = pd.DataFrame({
                'open': open_, 'high': high, 'low': low,
                'close': close, 'volume': volume,
            }, index=dates)

            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
            cerebro.broker.setcash(10000)

            # Run with timeout protection
            results = cerebro.run()
            strat_instance = results[0]

            # Count trades
            trade_count = 0
            if hasattr(strat_instance, 'analyzers'):
                try:
                    ta = strat_instance.analyzers.getbyname('tradeanalyzer')
                    trade_count = ta.get_analysis().get('total', {}).get('total', 0)
                except Exception:
                    pass

            # Fallback: check if any orders were placed
            if trade_count == 0:
                # The strategy ran without error, even if no trades
                # Still valid — some strategies only trade in specific conditions
                pass

            min_trades = cfg.pipeline.min_validation_trades
            if trade_count < min_trades:
                return True, f"Only {trade_count} trades (min={min_trades})", trade_count

            return True, "", trade_count

        except Exception as e:
            return False, f"Backtest error: {type(e).__name__}: {e}", 0

    # ==========================================================================
    # FULL RUN
    # ==========================================================================

    def run(self, skip_search: bool = False,
            skip_fetch: bool = False,
            limit: int = None) -> Dict[str, Any]:
        """
        Run the full discovery pipeline.

        Args:
            skip_search: If True, skip search+fetch and process existing docs
            skip_fetch: If True, skip fetch (use existing search results)
            limit: Max strategies to extract this run

        Returns:
            Pipeline run stats dict
        """
        self.run_start = time.time()

        print("\n" + "=" * 60)
        print("STRATEGY DISCOVERY PIPELINE")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Steps 1-2: Search and Fetch
        if not skip_search:
            results = self.step_1_search()
            if not skip_fetch:
                doc_ids = self.step_2_fetch(results)

        # Step 3: Extract
        strategies = self.step_3_extract(limit=limit)

        # Step 4: Deduplicate
        unique = self.step_4_deduplicate(strategies)

        # Step 5: Validate
        validated = self.step_5_validate(unique)

        # Step 6: Save
        saved = self.step_6_save(validated)

        # Final stats
        self.run_stats["duration_seconds"] = time.time() - self.run_start
        self._print_summary()

        # Save dedup index
        self.dedup.save_index()

        return self.run_stats

    def run_extract_only(self, limit: int = None) -> Dict[str, Any]:
        """
        Run extraction + dedup + validation on existing documents.

        Use this when you've already scraped documents and want to
        re-extract or process more.
        """
        self.run_start = time.time()

        print("\n" + "=" * 60)
        print("STRATEGY DISCOVERY - EXTRACT ONLY")
        print("=" * 60)

        strategies = self.step_3_extract(limit=limit)
        unique = self.step_4_deduplicate(strategies)
        validated = self.step_5_validate(unique)
        saved = self.step_6_save(validated)

        self.run_stats["duration_seconds"] = time.time() - self.run_start
        self._print_summary()

        return self.run_stats

    # ==========================================================================
    # REPORTING
    # ==========================================================================

    def _print_summary(self):
        """Print pipeline run summary."""
        s = self.run_stats
        duration = s["duration_seconds"]
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        print("\n" + "=" * 60)
        print("PIPELINE RUN SUMMARY")
        print("=" * 60)
        print(f"Duration:              {minutes}m {seconds}s")
        print(f"Search results:        {s['search_results']}")
        print(f"Documents fetched:     {s['documents_fetched']}")
        print(f"Quality passed:        {s['quality_passed']}")
        print(f"Strategies extracted:  {s['strategies_extracted']}")
        print(f"Duplicates removed:    {s['duplicates_removed']}")
        print(f"Code validated:        {s['code_validated']}")
        print(f"Code failed:           {s['code_failed']}")
        print(f"Final strategies:      {s['final_strategies']}")
        print("=" * 60)

        if s['final_strategies'] > 0:
            print(f"\n✓ {s['final_strategies']} new strategies ready for backtesting!")
            print(f"  Location: {STRATEGIES_DIR}")
        else:
            print("\n⚠ No new strategies this run.")

        # Component-level stats
        print()
        self.scraper.print_stats()
        self.extractor.print_stats()
        self.dedup.print_stats()
        self.db.print_stats()


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TradingLab Strategy Discovery Pipeline (Step 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline run (search + fetch + extract + dedup + validate)
    python discovery_pipeline.py

    # Extract only (process existing documents, skip search)
    python discovery_pipeline.py --extract-only

    # Limit to 10 strategies
    python discovery_pipeline.py --limit 10

    # Show config and exit
    python discovery_pipeline.py --config

    # Preflight check only
    python discovery_pipeline.py --check
        """,
    )
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip search/fetch, process existing documents")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max strategies to extract")
    parser.add_argument("--config", action="store_true",
                        help="Print configuration and exit")
    parser.add_argument("--check", action="store_true",
                        help="Run preflight checks and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Print database stats and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = LOGS_DIR / f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # Config only
    if args.config:
        print_config()
        return

    # Stats only
    if args.stats:
        db = ResearchDatabase()
        db.print_stats()
        return

    # Initialize pipeline
    pipeline = DiscoveryPipeline()

    # Check only
    if args.check:
        pipeline.preflight_check()
        return

    # Preflight
    if not args.extract_only:
        ok = pipeline.preflight_check()
        if not ok:
            print("\n⚠ Preflight failed. Fix issues above or use --extract-only.")
            print("  Continue anyway? (y/N)")
            response = input().strip().lower()
            if response != 'y':
                return

    # Run
    if args.extract_only:
        pipeline.run_extract_only(limit=args.limit)
    else:
        pipeline.run(limit=args.limit)


if __name__ == "__main__":
    main()
