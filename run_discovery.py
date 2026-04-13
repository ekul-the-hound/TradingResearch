# ==============================================================================
# run_discovery.py -- Standalone Strategy Discovery Runner
# ==============================================================================
# Run this on your VPS independently from the rest of the system.
# It continuously scrapes, extracts, and deduplicates trading strategies,
# saving them to discovery.db. The main pipeline reads from this DB later.
#
# Usage:
#   python run_discovery.py                    # Single batch run
#   python run_discovery.py --continuous       # Run forever in loop
#   python run_discovery.py --interval 3600    # Custom interval (seconds)
#   python run_discovery.py --max-runs 10      # Stop after N batches
#   python run_discovery.py --queries-only     # Only use custom_queries.txt
#   python run_discovery.py --status           # Print DB stats and exit
#
# Requires:
#   - SearXNG running (docker, port 8080)
#   - Ollama running with llama3.1:8b and/or deepseek-coder-v2:16b
#
# This script does NOT run backtests -- it only discovers strategy ideas.
# ==============================================================================

import sys
import time
import json
import argparse
import signal
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from discovery_config import DISCOVERY_CONFIG as cfg, STRATEGIES_DIR
from research_db import ResearchDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("discovery_runner")

# Graceful shutdown
_RUNNING = True
def _handle_signal(sig, frame):
    global _RUNNING
    log.info("Shutdown signal received -- finishing current batch...")
    _RUNNING = False
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def print_status():
    """Print current discovery database statistics."""
    db = ResearchDatabase()
    conn = db._get_conn()

    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    strats = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    unique = conn.execute("SELECT COUNT(*) FROM strategies WHERE is_duplicate=0").fetchone()[0]
    validated = conn.execute("SELECT COUNT(*) FROM strategies WHERE code_validates=1").fetchone()[0]
    manual = conn.execute("SELECT COUNT(*) FROM strategies WHERE origin_source='manual'").fetchone()[0]

    avg_q = conn.execute("SELECT AVG(quality_score) FROM strategies WHERE quality_score > 0").fetchone()[0]
    avg_q = f"{avg_q:.1f}" if avg_q else "N/A"

    # Top strategies
    top = conn.execute(
        "SELECT strategy_name, quality_score, origin_source, status "
        "FROM strategies WHERE is_duplicate=0 ORDER BY quality_score DESC LIMIT 10"
    ).fetchall()

    conn.close()

    print()
    print("=" * 60)
    print("  DISCOVERY DATABASE STATUS")
    print("=" * 60)
    print(f"  Documents scraped:     {docs}")
    print(f"  Strategies extracted:  {strats}")
    print(f"  Unique (not dupes):    {unique}")
    print(f"  Code validated:        {validated}")
    print(f"  Manual entries:        {manual}")
    print(f"  Avg quality score:     {avg_q}")
    print("-" * 60)
    if top:
        print("  Top 10 Strategies:")
        for name, score, origin, status in top:
            tag = f"[{origin}]" if origin != "scraped" else ""
            print(f"    {score:5.0f}  {name[:40]}  {tag}  ({status})")
    print("=" * 60)
    print()


def run_single_batch(queries_only=False):
    """Run one batch of discovery."""
    try:
        from discovery_pipeline import DiscoveryPipeline
        pipeline = DiscoveryPipeline()

        if queries_only:
            # Only use custom queries
            from discovery_config import CUSTOM_QUERIES_FILE
            if CUSTOM_QUERIES_FILE.exists():
                with open(CUSTOM_QUERIES_FILE) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                queries = [{"query": q, "source_type": "general", "bias": "unknown"} for q in lines]
                if not queries:
                    log.warning("No custom queries found in custom_queries.txt")
                    return 0
                log.info(f"Running with {len(queries)} custom queries")
                pipeline.run(queries=queries)
            else:
                log.warning("custom_queries.txt not found")
                return 0
        else:
            pipeline.run()

        return pipeline.run_stats.get("strategies_extracted", 0)

    except Exception as e:
        log.error(f"Batch failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def run_continuous(interval=3600, max_runs=None, queries_only=False):
    """Run discovery in a loop."""
    run_count = 0
    total_found = 0

    log.info(f"Starting continuous discovery (interval={interval}s, max_runs={max_runs or 'unlimited'})")

    while _RUNNING:
        run_count += 1
        log.info(f"--- Batch {run_count} starting ---")

        found = run_single_batch(queries_only=queries_only)
        total_found += found
        log.info(f"Batch {run_count} complete: {found} strategies found ({total_found} total)")

        if max_runs and run_count >= max_runs:
            log.info(f"Reached max_runs={max_runs}, stopping.")
            break

        if _RUNNING:
            log.info(f"Sleeping {interval}s until next batch...")
            for _ in range(interval):
                if not _RUNNING:
                    break
                time.sleep(1)

    log.info(f"Discovery runner finished. {run_count} batches, {total_found} strategies found.")


def main():
    parser = argparse.ArgumentParser(description="TradingLab Strategy Discovery Runner")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between batches (default: 3600)")
    parser.add_argument("--max-runs", type=int, default=None, help="Max number of batches")
    parser.add_argument("--queries-only", action="store_true", help="Only use custom_queries.txt")
    parser.add_argument("--status", action="store_true", help="Print DB stats and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    print()
    print("=" * 60)
    print("  TradingLab Strategy Discovery Runner")
    print("=" * 60)
    print(f"  Mode:     {'Continuous' if args.continuous else 'Single batch'}")
    print(f"  Interval: {args.interval}s")
    print(f"  Queries:  {'Custom only' if args.queries_only else 'All (built-in + custom)'}")
    print(f"  DB:       {cfg.db_path}")
    print("=" * 60)
    print()

    if args.continuous:
        run_continuous(interval=args.interval, max_runs=args.max_runs,
                       queries_only=args.queries_only)
    else:
        found = run_single_batch(queries_only=args.queries_only)
        print(f"\nDone. {found} strategies found.")
        print_status()


if __name__ == "__main__":
    main()
