# ==============================================================================
# test_discovery.py
# ==============================================================================
# FULL TEST SUITE FOR STRATEGY DISCOVERY PIPELINE (STEP 1)
#
# Tests every component without needing live services (SearXNG, LLMs).
# Uses mocked responses and synthetic data so you can validate the entire
# pipeline before going live.
#
# Test Groups:
#   1-3:   Config & Database
#   4-6:   Quality Scorer
#   7-9:   SearXNG Scraper (offline tests)
#   10-13: LLM Extractor (parsing, cleaning, mocked extraction)
#   14-16: Semantic Deduplication
#   17-19: Code Validation
#   20-21: Full Pipeline Integration
#   22:    Live Service Checks (optional, skippable)
#
# Usage:
#     python test_discovery.py           # Run all offline tests
#     python test_discovery.py --live    # Also run live service checks
#     python test_discovery.py --quick   # Only critical tests (1-6, 17-18)
#
# ==============================================================================

import sys
import os
import json
import time
import shutil
import tempfile
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple

# ==============================================================================
# HELPERS
# ==============================================================================

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(test_num: int, name: str, success: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if success:
        PASS_COUNT += 1
        print(f"  ✓ Test {test_num}: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  ✗ Test {test_num}: {name}")
        if detail:
            print(f"    → {detail}")


def print_skip(test_num: int, name: str, reason: str = ""):
    global SKIP_COUNT
    SKIP_COUNT += 1
    print(f"  ⊘ Test {test_num}: {name} (SKIPPED{': ' + reason if reason else ''})")


def run_test(test_num: int, name: str, func) -> bool:
    try:
        success, detail = func()
        print_result(test_num, name, success, detail)
        return success
    except Exception as e:
        print_result(test_num, name, False, f"EXCEPTION: {e}")
        traceback.print_exc()
        return False


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

# Temporary directory for test databases and files
TEST_DIR = None


def setup_test_dir():
    """Create a temporary directory for test artifacts."""
    global TEST_DIR
    TEST_DIR = Path(tempfile.mkdtemp(prefix="tradinglab_test_"))
    return TEST_DIR


def cleanup_test_dir():
    """Remove temporary test directory."""
    global TEST_DIR
    if TEST_DIR and TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)


# Sample documents for testing
SAMPLE_DOCS = {
    "arxiv_good": {
        "url": "https://arxiv.org/abs/2401.12345",
        "title": "Momentum Strategy with Moving Average Crossover and ATR Stops",
        "content": """This paper presents a systematic momentum trading strategy using a fast/slow
            moving average crossover with ATR-based stop losses.

            Entry signal: when the 10-period SMA crosses above the 50-period SMA.
            Exit: reverse crossover or stop loss at 2x ATR below entry price.
            Parameters: fast_period=10, slow_period=50, atr_stop_mult=2.0
            Lookback period: 252 trading days.

            Backtest results (2010-2023):
            Sharpe ratio: 1.2
            Max drawdown: -15.3%
            Win rate: 54%
            Total trades: 847
            Annualized return: 12.4%

            The expected return follows: E[r_t] = μ + β * r_{t-1} + ε_t
            With p-value < 0.05 indicating statistical significance.
            The strategy performs well when ADX > 25 (trending regime).
        """,
        "source_type": "arxiv",
        "source_bias": "academic",
        "search_query": "momentum trading strategy",
    },
    "reddit_low": {
        "url": "https://reddit.com/r/algotrading/post123",
        "title": "RSI works great",
        "content": "I use RSI and it works. Buy low sell high. Made good money last week.",
        "source_type": "reddit",
        "source_bias": "retail",
        "search_query": "trading strategy",
    },
    "github_code": {
        "url": "https://github.com/user/repo/blob/main/strategy.py",
        "title": "Backtrader Bollinger Band Strategy",
        "content": """```python
import backtrader as bt

class BollingerBandStrategy(bt.Strategy):
    params = (('period', 20), ('devfactor', 2.0),)

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(self.data.close,
            period=self.params.period, devfactor=self.params.devfactor)
        self.crossover_up = bt.indicators.CrossOver(self.data.close, self.bb.top)

    def next(self):
        if len(self) < self.params.period:
            return
        if not self.position:
            if self.data.close[0] > self.bb.top[0]:
                self.buy()
        else:
            if self.data.close[0] < self.bb.mid[0]:
                self.sell()
```
        Entry: price closes above upper Bollinger Band
        Exit: price drops below middle band
        Parameters: period=20, devfactor=2.0
        Timeframe: daily
        """,
        "source_type": "github",
        "source_bias": "retail",
        "search_query": "backtrader strategy",
    },
}

# Sample valid Backtrader code
VALID_STRATEGY_CODE = '''import backtrader as bt

class TestMomentumCrossover(bt.Strategy):
    """Test momentum crossover strategy."""
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if len(self) < self.params.slow_period:
            return
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.sell()
'''

BROKEN_STRATEGY_CODE = '''import backtrader as bt

class BrokenStrategy(bt.Strategy):
    def __init__(self):
        self.indicator = bt.indicators.FakeIndicatorThatDoesNotExist()
    def next(self):
        pass
'''

SYNTAX_ERROR_CODE = '''def oops(:
    pass
'''

# Sample LLM JSON response (for mocked extraction)
SAMPLE_SUMMARY_JSON = {
    "strategy_name": "MomentumCrossoverATR",
    "strategy_type": "trend_following",
    "summary": "A momentum strategy using fast/slow SMA crossover with ATR stop loss. "
               "Enters long when 10-period SMA crosses above 50-period SMA.",
    "entry_rules": [
        "10-period SMA crosses above 50-period SMA",
        "ADX > 25 (optional trend filter)"
    ],
    "exit_rules": [
        "Reverse crossover (10 SMA crosses below 50 SMA)",
        "Stop loss at 2x ATR below entry price"
    ],
    "indicators": ["SMA(10)", "SMA(50)", "ATR(14)", "ADX(14)"],
    "parameters": {
        "fast_period": 10,
        "slow_period": 50,
        "atr_stop_mult": 2.0,
        "adx_threshold": 25
    },
    "timeframe": "1D",
    "asset_class": "forex",
    "risk_management": "Fixed 2x ATR stop loss, position size based on risk per trade",
    "confidence": "high"
}


# ==============================================================================
# TESTS 1-3: CONFIG & DATABASE
# ==============================================================================

def test_01_config_loads() -> Tuple[bool, str]:
    """Test that discovery_config.py loads without errors."""
    from discovery_config import DISCOVERY_CONFIG as cfg, print_config

    checks = [
        cfg.searxng.base_url != "",
        cfg.llm.summarizer.model != "",
        cfg.llm.code_generator.model != "",
        len(cfg.search_queries) > 0,
        cfg.quality.min_quality_threshold > 0,
        cfg.dedup.similarity_threshold > 0,
    ]
    if not all(checks):
        return False, "Some config values are empty or invalid"
    return True, ""


def test_02_database_init() -> Tuple[bool, str]:
    """Test database creation and table structure."""
    from research_db import ResearchDatabase

    db_path = TEST_DIR / "test_db.sqlite"
    db = ResearchDatabase(db_path=db_path)

    # Check tables exist
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    required = {"documents", "strategies", "extraction_log", "dedup_log"}
    missing = required - tables
    if missing:
        return False, f"Missing tables: {missing}"

    return True, ""


def test_03_database_crud() -> Tuple[bool, str]:
    """Test document and strategy CRUD operations."""
    from research_db import ResearchDatabase

    db = ResearchDatabase(db_path=TEST_DIR / "test_crud.sqlite")

    # Save document
    doc_id = db.save_document(SAMPLE_DOCS["arxiv_good"])
    if not doc_id:
        return False, "save_document returned None"

    # Check duplicate detection (URL)
    if not db.document_exists(url=SAMPLE_DOCS["arxiv_good"]["url"]):
        return False, "URL duplicate detection failed"

    # Check duplicate detection (content hash)
    if not db.document_exists(content=SAMPLE_DOCS["arxiv_good"]["content"]):
        return False, "Content hash duplicate detection failed"

    # Save document with same hash → should return None
    dup_id = db.save_document(SAMPLE_DOCS["arxiv_good"])
    if dup_id is not None:
        return False, "Duplicate document was not rejected"

    # Save strategy
    strat_id = db.save_strategy({
        "doc_id": doc_id,
        "strategy_name": "TestStrategy",
        "summary": "Test summary",
        "origin_source": "scraped",
        "quality_score": 0.75,
        "status": "extracted",
    })
    if not strat_id:
        return False, "save_strategy returned None"

    # Retrieve strategy
    strat = db.get_strategy_by_id(strat_id)
    if not strat or strat["strategy_name"] != "TestStrategy":
        return False, "Strategy retrieval failed"

    # Update strategy
    db.update_strategy(strat_id, {"status": "validated", "code_validates": 1})
    strat2 = db.get_strategy_by_id(strat_id)
    if strat2["status"] != "validated":
        return False, "Strategy update failed"

    # Get unprocessed documents
    unproc = db.get_unprocessed_documents()
    if len(unproc) != 1:
        return False, f"Expected 1 unprocessed doc, got {len(unproc)}"

    # Pipeline stats
    stats = db.get_pipeline_stats()
    if stats["total_documents"] != 1 or stats["total_strategies"] != 1:
        return False, f"Stats incorrect: {stats}"

    return True, ""


# ==============================================================================
# TESTS 4-6: QUALITY SCORER
# ==============================================================================

def test_04_quality_scorer_basics() -> Tuple[bool, str]:
    """Test quality scorer pattern detection."""
    from quality_scorer import QualityScorer

    scorer = QualityScorer()

    # Test math detection
    doc_math = {"content": "E[r_t] = μ + β * r_{t-1}, Sharpe ratio = 1.5", "source_type": "arxiv", "title": ""}
    result = scorer.score_document(doc_math)
    if not result["has_math"]:
        return False, "Failed to detect math patterns"

    # Test backtest detection
    doc_bt = {"content": "Sharpe ratio: 1.2, Max drawdown: -15%, Win rate: 54%", "source_type": "general", "title": ""}
    result = scorer.score_document(doc_bt)
    if not result["has_backtest"]:
        return False, "Failed to detect backtest patterns"

    # Test code detection
    doc_code = {"content": "import backtrader as bt\nclass MyStrategy(bt.Strategy):\n    def next(self):", "source_type": "github", "title": ""}
    result = scorer.score_document(doc_code)
    if not result["has_code"]:
        return False, "Failed to detect code patterns"

    return True, ""


def test_05_quality_scoring_ranking() -> Tuple[bool, str]:
    """Test that quality scores rank documents correctly."""
    from quality_scorer import QualityScorer

    scorer = QualityScorer()

    score_arxiv = scorer.score_document(SAMPLE_DOCS["arxiv_good"])["quality_score"]
    score_reddit = scorer.score_document(SAMPLE_DOCS["reddit_low"])["quality_score"]
    score_github = scorer.score_document(SAMPLE_DOCS["github_code"])["quality_score"]

    # arxiv paper should score highest, reddit lowest
    if not (score_arxiv > score_github > score_reddit):
        return False, f"Ranking wrong: arxiv={score_arxiv:.3f}, github={score_github:.3f}, reddit={score_reddit:.3f}"

    return True, ""


def test_06_quality_threshold() -> Tuple[bool, str]:
    """Test quality threshold gate."""
    from quality_scorer import QualityScorer

    scorer = QualityScorer()

    passes_arxiv, _ = scorer.passes_extraction_threshold(SAMPLE_DOCS["arxiv_good"])
    passes_reddit, _ = scorer.passes_extraction_threshold(SAMPLE_DOCS["reddit_low"])

    if not passes_arxiv:
        return False, "arXiv paper should pass threshold"
    if passes_reddit:
        return False, "Low-quality Reddit post should NOT pass threshold"

    return True, ""


# ==============================================================================
# TESTS 7-9: SEARXNG SCRAPER (offline)
# ==============================================================================

def test_07_source_type_detection() -> Tuple[bool, str]:
    """Test URL-based source type detection."""
    from searxng_scraper import SearXNGScraper

    scraper = SearXNGScraper.__new__(SearXNGScraper)

    tests = [
        ("https://arxiv.org/abs/2401.12345", "arxiv"),
        ("https://ssrn.com/abstract=1234567", "ssrn"),
        ("https://github.com/user/repo", "github"),
        ("https://reddit.com/r/algotrading/comments/abc", "reddit"),
        ("https://medium.com/@user/article", "blog"),
        ("https://example.com/strategy", "general"),
    ]

    for url, expected in tests:
        detected = scraper._detect_source_type(url)
        if detected != expected:
            return False, f"URL {url}: expected {expected}, got {detected}"

    return True, ""


def test_08_url_filter() -> Tuple[bool, str]:
    """Test URL skip filter."""
    from searxng_scraper import SearXNGScraper

    scraper = SearXNGScraper.__new__(SearXNGScraper)

    should_skip = [
        "https://youtube.com/watch?v=abc",
        "https://twitter.com/user/status/123",
        "https://example.com/paper.pdf",
        "https://example.com/archive.zip",
        "https://facebook.com/page",
    ]
    should_keep = [
        "https://arxiv.org/abs/2401.12345",
        "https://github.com/user/repo",
        "https://reddit.com/r/algotrading/post",
        "https://quantconnect.com/forum/discussion",
    ]

    for url in should_skip:
        if not scraper._should_skip_url(url):
            return False, f"Should skip but didn't: {url}"

    for url in should_keep:
        if scraper._should_skip_url(url):
            return False, f"Should keep but skipped: {url}"

    return True, ""


def test_09_custom_queries_loading() -> Tuple[bool, str]:
    """Test custom queries file loading."""
    import discovery_config
    from searxng_scraper import SearXNGScraper

    # Write test custom queries
    test_queries = "# Comment line\nturtle trading strategy\n\nKeltner channel breakout\n"
    custom_path = TEST_DIR / "custom_queries.txt"
    custom_path.write_text(test_queries)

    # Temporarily override the module-level path BEFORE constructing scraper
    original = discovery_config.CUSTOM_QUERIES_FILE
    discovery_config.CUSTOM_QUERIES_FILE = custom_path

    # Also patch the already-imported reference in searxng_scraper
    import searxng_scraper
    original_scraper_ref = searxng_scraper.CUSTOM_QUERIES_FILE
    searxng_scraper.CUSTOM_QUERIES_FILE = custom_path

    try:
        scraper = SearXNGScraper.__new__(SearXNGScraper)
        queries = scraper._load_custom_queries()

        if len(queries) != 2:
            return False, f"Expected 2 queries, got {len(queries)}"
        if queries[0]["query"] != "turtle trading strategy":
            return False, f"First query wrong: {queries[0]['query']}"
    finally:
        discovery_config.CUSTOM_QUERIES_FILE = original
        searxng_scraper.CUSTOM_QUERIES_FILE = original_scraper_ref

    return True, ""


# ==============================================================================
# TESTS 10-13: LLM EXTRACTOR
# ==============================================================================

def test_10_json_parsing() -> Tuple[bool, str]:
    """Test JSON response parsing from LLM output."""
    from llm_extractor import LLMExtractor

    ext = LLMExtractor.__new__(LLMExtractor)

    # Clean JSON
    r1 = ext._parse_json_response('{"strategy_name": "Test", "confidence": "high"}')
    if not r1 or r1["strategy_name"] != "Test":
        return False, "Failed to parse clean JSON"

    # Markdown-wrapped JSON
    r2 = ext._parse_json_response('```json\n{"strategy_name": "Test2"}\n```')
    if not r2 or r2["strategy_name"] != "Test2":
        return False, "Failed to parse markdown-wrapped JSON"

    # JSON with preamble
    r3 = ext._parse_json_response('Here is the result:\n{"strategy_name": "Test3"}')
    if not r3 or r3["strategy_name"] != "Test3":
        return False, "Failed to parse JSON with preamble"

    # No-strategy response
    r4 = ext._parse_json_response('{"strategy_name": "NONE", "confidence": "none"}')
    if not r4 or r4["strategy_name"] != "NONE":
        return False, "Failed to parse no-strategy response"

    # Invalid JSON
    r5 = ext._parse_json_response('This is not JSON at all.')
    if r5 is not None:
        return False, "Should return None for invalid JSON"

    return True, ""


def test_11_code_cleaning() -> Tuple[bool, str]:
    """Test code response cleaning."""
    from llm_extractor import LLMExtractor

    ext = LLMExtractor.__new__(LLMExtractor)

    # Code wrapped in markdown
    raw1 = '```python\nimport backtrader as bt\n\nclass Test(bt.Strategy):\n    pass\n```'
    clean1 = ext._clean_code_response(raw1)
    if not clean1.startswith("import backtrader"):
        return False, f"Markdown not stripped: starts with '{clean1[:30]}'"

    # Code with preamble text
    raw2 = 'Here is the code:\n\nimport backtrader as bt\n\nclass Test(bt.Strategy):\n    pass'
    clean2 = ext._clean_code_response(raw2)
    if "Here is" in clean2:
        return False, "Preamble not removed"

    # Code with trailing explanation
    raw3 = 'import backtrader as bt\n\nclass Test(bt.Strategy):\n    pass\n\nThis strategy uses RSI for entries.'
    clean3 = ext._clean_code_response(raw3)
    if "This strategy uses" in clean3:
        return False, "Trailing explanation not removed"

    return True, ""


def test_12_mocked_summarize() -> Tuple[bool, str]:
    """Test summarize flow with mocked LLM response."""
    from llm_extractor import LLMExtractor, LocalLLMClient
    from research_db import ResearchDatabase
    from quality_scorer import QualityScorer

    db = ResearchDatabase(db_path=TEST_DIR / "test_extract.sqlite")

    # Create extractor with mocked LLM client
    ext = LLMExtractor.__new__(LLMExtractor)
    ext.db = db
    ext.scorer = QualityScorer()
    ext.stats = {
        "documents_processed": 0, "summaries_extracted": 0,
        "code_generated": 0, "no_strategy_found": 0,
        "extraction_errors": 0, "quality_rejected": 0,
    }

    # Mock the summarizer
    class MockClient:
        model = "mock-model"
        def chat(self, system, user, temperature=None):
            return json.dumps(SAMPLE_SUMMARY_JSON), {"prompt_tokens": 100, "completion_tokens": 50}

    ext.summarizer = MockClient()

    # Save a test document first
    doc_id = db.save_document(SAMPLE_DOCS["arxiv_good"])
    doc = db.get_unprocessed_documents(limit=1)[0]

    # Run summarize
    summary = ext.summarize_document(doc)

    if summary is None:
        return False, "Summarize returned None"
    if summary["strategy_name"] != "MomentumCrossoverATR":
        return False, f"Wrong strategy name: {summary['strategy_name']}"
    if ext.stats["summaries_extracted"] != 1:
        return False, "Stats not updated"

    return True, ""


def test_13_mocked_full_extraction() -> Tuple[bool, str]:
    """Test full two-stage extraction with mocked LLMs."""
    from llm_extractor import LLMExtractor
    from research_db import ResearchDatabase
    from quality_scorer import QualityScorer
    from discovery_config import DISCOVERY_CONFIG as cfg

    db = ResearchDatabase(db_path=TEST_DIR / "test_full_extract.sqlite")

    ext = LLMExtractor.__new__(LLMExtractor)
    ext.db = db
    ext.scorer = QualityScorer()
    ext.stats = {
        "documents_processed": 0, "summaries_extracted": 0,
        "code_generated": 0, "no_strategy_found": 0,
        "extraction_errors": 0, "quality_rejected": 0,
    }

    class MockSummarizer:
        model = "mock-summarizer"
        def chat(self, system, user, temperature=None):
            return json.dumps(SAMPLE_SUMMARY_JSON), {"prompt_tokens": 100, "completion_tokens": 50}

    class MockCodeGen:
        model = "mock-codegen"
        def chat(self, system, user, temperature=None):
            return VALID_STRATEGY_CODE, {"prompt_tokens": 200, "completion_tokens": 150}

    ext.summarizer = MockSummarizer()
    ext.code_generator = MockCodeGen()

    # Save doc
    doc_id = db.save_document(SAMPLE_DOCS["arxiv_good"])
    doc = db.get_unprocessed_documents(limit=1)[0]

    # Run full extraction
    result = ext.extract_strategy(doc)

    if result is None:
        return False, "Full extraction returned None"
    if result["strategy_name"] != "MomentumCrossoverATR":
        return False, f"Wrong name: {result['strategy_name']}"
    if not result.get("generated_code"):
        return False, "No generated code"
    if "class TestMomentumCrossover" not in result["generated_code"]:
        return False, "Generated code doesn't contain expected class"
    if result["origin_source"] != "scraped":
        return False, f"Wrong origin: {result['origin_source']}"
    if result.get("quality_score", 0) <= 0:
        return False, "Quality score not set"

    return True, ""


# ==============================================================================
# TESTS 14-16: SEMANTIC DEDUPLICATION
# ==============================================================================

def test_14_dedup_basic() -> Tuple[bool, str]:
    """Test basic dedup: unique strategies should be added."""
    from semantic_dedup import SemanticDeduplicator
    from research_db import ResearchDatabase
    from discovery_config import DISCOVERY_CONFIG as cfg

    db = ResearchDatabase(db_path=TEST_DIR / "test_dedup.sqlite")

    # Isolate index paths for this test
    orig_idx, orig_meta = cfg.dedup.index_path, cfg.dedup.metadata_path
    cfg.dedup.index_path = TEST_DIR / "t14_index.faiss"
    cfg.dedup.metadata_path = TEST_DIR / "t14_meta.json"

    try:
        dedup = SemanticDeduplicator(db=db)

        is_dup, dup_of, score = dedup.check_and_add(
            "s1", "RSI mean reversion strategy. Buy when RSI drops below 30."
        )
        if is_dup:
            return False, "First strategy should not be a duplicate"
        if dedup.get_index_size() != 1:
            return False, f"Index size should be 1, got {dedup.get_index_size()}"
    finally:
        cfg.dedup.index_path, cfg.dedup.metadata_path = orig_idx, orig_meta

    return True, ""


def test_15_dedup_different_strategies() -> Tuple[bool, str]:
    """Test that genuinely different strategies are kept."""
    from semantic_dedup import SemanticDeduplicator
    from research_db import ResearchDatabase
    from discovery_config import DISCOVERY_CONFIG as cfg

    db = ResearchDatabase(db_path=TEST_DIR / "test_dedup_diff.sqlite")

    # Isolate index paths
    orig_idx, orig_meta = cfg.dedup.index_path, cfg.dedup.metadata_path
    cfg.dedup.index_path = TEST_DIR / "t15_index.faiss"
    cfg.dedup.metadata_path = TEST_DIR / "t15_meta.json"

    try:
        dedup = SemanticDeduplicator(db=db)

        strategies = [
            ("s1", "RSI mean reversion. Buy when RSI below 30, sell above 70."),
            ("s2", "Moving average crossover trend following. Long when fast SMA crosses above slow SMA."),
            ("s3", "Bollinger Band breakout. Buy above upper band, sell below middle."),
            ("s4", "MACD histogram divergence strategy for forex pairs on 4H timeframe."),
        ]

        for sid, summary in strategies:
            is_dup, _, _ = dedup.check_and_add(sid, summary)
            if is_dup:
                return False, f"Strategy {sid} incorrectly marked as duplicate"

        if dedup.get_index_size() != 4:
            return False, f"Expected 4 strategies in index, got {dedup.get_index_size()}"
    finally:
        cfg.dedup.index_path, cfg.dedup.metadata_path = orig_idx, orig_meta

    return True, ""


def test_16_dedup_persistence() -> Tuple[bool, str]:
    """Test that dedup index persists and reloads correctly."""
    from semantic_dedup import SemanticDeduplicator
    from research_db import ResearchDatabase
    from discovery_config import DISCOVERY_CONFIG as cfg

    db = ResearchDatabase(db_path=TEST_DIR / "test_dedup_persist.sqlite")

    # Override index paths to test dir
    original_index = cfg.dedup.index_path
    original_meta = cfg.dedup.metadata_path
    cfg.dedup.index_path = TEST_DIR / "test_index.faiss"
    cfg.dedup.metadata_path = TEST_DIR / "test_metadata.json"

    try:
        dedup1 = SemanticDeduplicator(db=db)
        dedup1.check_and_add("s1", "RSI mean reversion strategy")
        dedup1.check_and_add("s2", "Bollinger band breakout strategy")
        dedup1.save_index()

        size_before = dedup1.get_index_size()

        # Create new instance — should load from disk
        dedup2 = SemanticDeduplicator(db=db)
        size_after = dedup2.get_index_size()

        if size_after != size_before:
            return False, f"Index size changed after reload: {size_before} → {size_after}"

    finally:
        cfg.dedup.index_path = original_index
        cfg.dedup.metadata_path = original_meta

    return True, ""


# ==============================================================================
# TESTS 17-19: CODE VALIDATION
# ==============================================================================

def test_17_valid_code_passes() -> Tuple[bool, str]:
    """Test that valid Backtrader code passes validation."""
    from discovery_pipeline import DiscoveryPipeline

    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)

    # Minimal config for validation
    from discovery_config import DISCOVERY_CONFIG as cfg
    ok, err, trades = pipeline._validate_code(VALID_STRATEGY_CODE, "TestMomentum")

    if not ok:
        return False, f"Valid code should pass: {err}"

    return True, ""


def test_18_broken_code_fails() -> Tuple[bool, str]:
    """Test that broken code fails validation."""
    from discovery_pipeline import DiscoveryPipeline

    pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)

    # Fake indicator
    ok1, err1, _ = pipeline._validate_code(BROKEN_STRATEGY_CODE, "Broken")
    if ok1:
        return False, "Broken code (fake indicator) should fail"

    # Syntax error
    ok2, err2, _ = pipeline._validate_code(SYNTAX_ERROR_CODE, "SyntaxBad")
    if ok2:
        return False, "Syntax error code should fail"

    # Empty code
    ok3, err3, _ = pipeline._validate_code("", "Empty")
    if ok3:
        return False, "Empty code should fail"

    # No strategy class
    ok4, err4, _ = pipeline._validate_code("import backtrader as bt\nx = 42", "NoClass")
    if ok4:
        return False, "Code without strategy class should fail"

    return True, ""


def test_19_strategy_file_saving() -> Tuple[bool, str]:
    """Test strategy code file saving."""
    from discovery_pipeline import DiscoveryPipeline
    from research_db import ResearchDatabase
    from discovery_config import STRATEGIES_DIR

    # Use test output dir
    test_strat_dir = TEST_DIR / "discovered_strategies"
    test_strat_dir.mkdir(exist_ok=True)

    import discovery_config
    original_dir = discovery_config.STRATEGIES_DIR
    discovery_config.STRATEGIES_DIR = test_strat_dir

    try:
        db = ResearchDatabase(db_path=TEST_DIR / "test_save.sqlite")

        pipeline = DiscoveryPipeline.__new__(DiscoveryPipeline)
        pipeline.db = db
        pipeline.run_stats = {"final_strategies": 0}

        # Save a strategy
        strat_id = db.save_strategy({
            "strategy_name": "TestMomentum",
            "summary": "Test strategy",
            "generated_code": VALID_STRATEGY_CODE,
            "source_url": "https://example.com",
            "source_type": "github",
            "source_bias": "retail",
            "quality_score": 0.8,
            "status": "validated",
        })

        strat = db.get_strategy_by_id(strat_id)
        strat["strategy_id"] = strat_id

        saved = pipeline.step_6_save([strat])

        if len(saved) != 1:
            return False, f"Expected 1 saved file, got {len(saved)}"

        filepath = Path(saved[0])
        if not filepath.exists():
            return False, f"File not created: {filepath}"

        content = filepath.read_text()
        if "class TestMomentumCrossover" not in content:
            return False, "Strategy class not in saved file"
        if "# Auto-discovered strategy" not in content:
            return False, "Header comment not in saved file"

    finally:
        discovery_config.STRATEGIES_DIR = original_dir

    return True, ""


# ==============================================================================
# TESTS 20-21: INTEGRATION
# ==============================================================================

def test_20_pipeline_initialization() -> Tuple[bool, str]:
    """Test that the full pipeline initializes all components."""
    from discovery_pipeline import DiscoveryPipeline

    pipeline = DiscoveryPipeline()

    checks = [
        pipeline.db is not None,
        pipeline.scraper is not None,
        pipeline.extractor is not None,
        pipeline.scorer is not None,
        pipeline.dedup is not None,
    ]

    if not all(checks):
        return False, "Some pipeline components failed to initialize"

    return True, ""


def test_21_pipeline_dedup_integration() -> Tuple[bool, str]:
    """Test extraction → dedup flow with mocked data."""
    from semantic_dedup import SemanticDeduplicator
    from research_db import ResearchDatabase
    from discovery_config import DISCOVERY_CONFIG as cfg

    db = ResearchDatabase(db_path=TEST_DIR / "test_integration.sqlite")

    # Isolate index paths
    orig_idx, orig_meta = cfg.dedup.index_path, cfg.dedup.metadata_path
    cfg.dedup.index_path = TEST_DIR / "t21_index.faiss"
    cfg.dedup.metadata_path = TEST_DIR / "t21_meta.json"

    try:
        dedup = SemanticDeduplicator(db=db)

        # Simulate batch of extracted strategies
        batch = [
            {"strategy_id": "s1", "summary": "RSI mean reversion. Buy below 30, sell above 70."},
            {"strategy_id": "s2", "summary": "Moving average crossover trend following."},
            {"strategy_id": "s3", "summary": "RSI oversold strategy. Purchase under 30, exit over 70."},
            {"strategy_id": "s4", "summary": "Bollinger band breakout with volatility filter."},
        ]

        # Save strategies to DB first
        for s in batch:
            db.save_strategy({
                "strategy_id": s["strategy_id"],
                "strategy_name": s["strategy_id"],
                "summary": s["summary"],
                "status": "extracted",
            })

        unique = dedup.deduplicate_batch(batch)

        # In fallback mode, similarity detection may not catch near-dups
        # but the flow should complete without errors
        if len(unique) < 2:
            return False, f"Too few unique strategies: {len(unique)}"
        if len(unique) > len(batch):
            return False, f"More unique than input: {len(unique)} > {len(batch)}"
    finally:
        cfg.dedup.index_path, cfg.dedup.metadata_path = orig_idx, orig_meta

    return True, ""


# ==============================================================================
# TEST 22: LIVE SERVICE CHECKS (optional)
# ==============================================================================

def test_22_live_searxng() -> Tuple[bool, str]:
    """Test live SearXNG connection."""
    from searxng_scraper import SearXNGScraper
    from research_db import ResearchDatabase

    db = ResearchDatabase(db_path=TEST_DIR / "test_live.sqlite")
    scraper = SearXNGScraper(db=db)

    if not scraper.health_check():
        return False, "SearXNG not responding"

    # Try a single search
    results = scraper.search("python backtrader strategy", source_type="general")
    if len(results) == 0:
        return False, "Search returned no results"

    return True, ""


def test_23_live_llm() -> Tuple[bool, str]:
    """Test live LLM endpoint connection."""
    from llm_extractor import LLMExtractor
    from research_db import ResearchDatabase

    db = ResearchDatabase(db_path=TEST_DIR / "test_live_llm.sqlite")
    ext = LLMExtractor(db=db)

    status = ext.health_check()
    if not status.get("summarizer"):
        return False, "Summarizer LLM not responding"
    if not status.get("code_generator"):
        return False, "Code generator LLM not responding"

    return True, ""


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Suite for Strategy Discovery Pipeline")
    parser.add_argument("--live", action="store_true", help="Include live service tests")
    parser.add_argument("--quick", action="store_true", help="Only run critical tests")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  STRATEGY DISCOVERY PIPELINE — TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    setup_test_dir()
    print(f"\n  Test directory: {TEST_DIR}\n")

    try:
        # Group 1: Config & Database
        print_header("CONFIG & DATABASE")
        run_test(1, "Config loads", test_01_config_loads)
        run_test(2, "Database initialization", test_02_database_init)
        run_test(3, "Database CRUD operations", test_03_database_crud)

        # Group 2: Quality Scorer
        print_header("QUALITY SCORER")
        run_test(4, "Pattern detection (math, backtest, code)", test_04_quality_scorer_basics)
        run_test(5, "Quality ranking (arxiv > github > reddit)", test_05_quality_scoring_ranking)
        run_test(6, "Quality threshold gate", test_06_quality_threshold)

        if args.quick:
            # Skip to code validation for quick mode
            print_header("CODE VALIDATION")
            run_test(17, "Valid Backtrader code passes", test_17_valid_code_passes)
            run_test(18, "Broken/syntax error code fails", test_18_broken_code_fails)
        else:
            # Group 3: Scraper (offline)
            print_header("SEARXNG SCRAPER (offline)")
            run_test(7, "Source type detection from URLs", test_07_source_type_detection)
            run_test(8, "URL skip filter", test_08_url_filter)
            run_test(9, "Custom queries file loading", test_09_custom_queries_loading)

            # Group 4: LLM Extractor
            print_header("LLM EXTRACTOR (mocked)")
            run_test(10, "JSON response parsing", test_10_json_parsing)
            run_test(11, "Code response cleaning", test_11_code_cleaning)
            run_test(12, "Mocked summarization", test_12_mocked_summarize)
            run_test(13, "Mocked full extraction (2-stage)", test_13_mocked_full_extraction)

            # Group 5: Deduplication
            print_header("SEMANTIC DEDUPLICATION")
            run_test(14, "Basic dedup (first strategy is unique)", test_14_dedup_basic)
            run_test(15, "Different strategies stay unique", test_15_dedup_different_strategies)
            run_test(16, "Index persistence (save/reload)", test_16_dedup_persistence)

            # Group 6: Code Validation
            print_header("CODE VALIDATION")
            run_test(17, "Valid Backtrader code passes", test_17_valid_code_passes)
            run_test(18, "Broken/syntax error code fails", test_18_broken_code_fails)
            run_test(19, "Strategy file saving", test_19_strategy_file_saving)

            # Group 7: Integration
            print_header("INTEGRATION")
            run_test(20, "Pipeline initialization", test_20_pipeline_initialization)
            run_test(21, "Extract → dedup flow", test_21_pipeline_dedup_integration)

            # Group 8: Live (optional)
            if args.live:
                print_header("LIVE SERVICE CHECKS")
                run_test(22, "Live SearXNG connection", test_22_live_searxng)
                run_test(23, "Live LLM endpoints", test_23_live_llm)
            else:
                print_header("LIVE SERVICE CHECKS")
                print_skip(22, "Live SearXNG connection", "use --live flag")
                print_skip(23, "Live LLM endpoints", "use --live flag")

    finally:
        cleanup_test_dir()

    # Summary
    total = PASS_COUNT + FAIL_COUNT
    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, "
          f"{SKIP_COUNT} skipped (of {total + SKIP_COUNT} total)")
    print("=" * 70)

    if FAIL_COUNT == 0:
        print("\n  ✓ ALL TESTS PASSED\n")
    else:
        print(f"\n  ✗ {FAIL_COUNT} TEST(S) FAILED\n")

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
