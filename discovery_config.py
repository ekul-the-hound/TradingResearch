# ==============================================================================
# discovery_config.py
# ==============================================================================
# Central configuration for the Strategy Discovery Pipeline (Step 1)
#
# Uses Ollama for all LLM inference -- local or cloud.
#
# SETUP:
#   Local mode:  ollama pull qwen3.5:9b && ollama pull qwen2.5-coder:7b
#   Cloud mode:  ollama signin  (then models run on Ollama's cloud GPUs)
#
# Switch between modes by changing MODE below.
# ==============================================================================

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# ==============================================================================
# BASE PATHS
# ==============================================================================
RESEARCH_LAB_DIR = Path(__file__).parent
PROJECT_DIR = RESEARCH_LAB_DIR.parent
DATA_DIR = RESEARCH_LAB_DIR / "data"
STRATEGIES_DIR = RESEARCH_LAB_DIR / "discovered_strategies"
LOGS_DIR = RESEARCH_LAB_DIR / "logs"

for d in [DATA_DIR, STRATEGIES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MODE -- "cloud" or "local"
# ==============================================================================
# "cloud"  -> Uses Ollama Cloud (big models on Ollama's GPUs, needs ollama signin)
# "local"  -> Uses local Ollama (smaller models on your GPU, free, offline)

MODE = os.getenv("DISCOVERY_MODE", "cloud")


# ==============================================================================
# MODEL PRESETS
# ==============================================================================

MODES = {
    "cloud": {
        "summarizer_model": "qwen3.5:cloud",
        "code_model": "qwen3-coder:480b-cloud",
        "reviewer_model": "qwen3.5:cloud",
        "max_content_chars": 60000,   # cloud models handle big context
        "request_timeout": 180,
    },
    "local": {
        "summarizer_model": "qwen3.5:9b",
        "code_model": "qwen2.5-coder:7b",
        "reviewer_model": "qwen3.5:9b",
        "max_content_chars": 12000,   # smaller context for local models
        "request_timeout": 300,       # local models are slower
    },
}

_ACTIVE = MODES.get(MODE, MODES["cloud"])

# Ollama always runs on localhost
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")


# ==============================================================================
# LLM CONFIGURATION
# ==============================================================================

@dataclass
class LLMEndpoint:
    """Configuration for a single Ollama LLM endpoint."""
    base_url: str = OLLAMA_URL
    api_key: str = "not-needed"
    model: str = "default"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = _ACTIVE["request_timeout"]


@dataclass
class LLMConfig:
    """All LLM endpoints used in the discovery pipeline."""
    summarizer: LLMEndpoint = field(default_factory=lambda: LLMEndpoint(
        model=_ACTIVE["summarizer_model"],
        max_tokens=2048,
        temperature=0.2,
    ))
    code_generator: LLMEndpoint = field(default_factory=lambda: LLMEndpoint(
        model=_ACTIVE["code_model"],
        max_tokens=8192,
        temperature=0.1,
    ))
    reviewer: LLMEndpoint = field(default_factory=lambda: LLMEndpoint(
        model=_ACTIVE["reviewer_model"],
        max_tokens=2048,
        temperature=0.4,
    ))
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_content_chars: int = _ACTIVE["max_content_chars"]


# ==============================================================================
# SEARXNG CONFIGURATION
# ==============================================================================
@dataclass
class SearXNGConfig:
    """SearXNG metasearch engine settings."""
    base_url: str = "http://localhost:8080"
    max_results_per_query: int = 30
    request_delay: float = 2.0
    timeout: int = 30
    output_format: str = "json"
    categories: str = "general,science"
    language: str = "en"


# ==============================================================================
# SEARCH QUERIES
# ==============================================================================
SEARCH_QUERIES: List[Dict[str, str]] = [
    # --- Academic / Research Papers ---
    {"query": "systematic trading strategy backtest results site:arxiv.org", "source_type": "arxiv", "bias": "academic"},
    {"query": "quantitative trading strategy momentum mean reversion site:ssrn.com", "source_type": "ssrn", "bias": "academic"},
    {"query": "algorithmic trading strategy machine learning forex", "source_type": "arxiv", "bias": "academic"},
    {"query": "pairs trading cointegration backtest", "source_type": "arxiv", "bias": "academic"},
    {"query": "trend following strategy performance analysis", "source_type": "arxiv", "bias": "academic"},
    {"query": "volatility breakout trading strategy", "source_type": "general", "bias": "academic"},
    {"query": "statistical arbitrage strategy implementation", "source_type": "arxiv", "bias": "academic"},
    {"query": "market microstructure trading strategy", "source_type": "arxiv", "bias": "academic"},

    # --- GitHub / Code Repositories ---
    {"query": "backtrader strategy python github", "source_type": "github", "bias": "retail"},
    {"query": "algorithmic trading bot python backtest site:github.com", "source_type": "github", "bias": "retail"},
    {"query": "quantitative trading strategy python open source", "source_type": "github", "bias": "retail"},

    # --- Retail / Blog Sources ---
    {"query": "moving average crossover strategy backtest", "source_type": "blog", "bias": "retail"},
    {"query": "RSI trading strategy with backtest results", "source_type": "blog", "bias": "retail"},
    {"query": "Bollinger Band squeeze breakout strategy", "source_type": "blog", "bias": "retail"},
    {"query": "MACD divergence trading strategy backtested", "source_type": "blog", "bias": "retail"},
    {"query": "ichimoku cloud trading strategy results", "source_type": "blog", "bias": "retail"},
    {"query": "price action trading strategy systematic", "source_type": "blog", "bias": "retail"},
    {"query": "donchian channel breakout system backtest", "source_type": "blog", "bias": "retail"},
    {"query": "ADX trend strength trading strategy python", "source_type": "blog", "bias": "retail"},

    # --- Forum / Community Sources ---
    {"query": "profitable trading strategy reddit algotrading", "source_type": "forum", "bias": "retail"},
    {"query": "automated trading strategy elite trader forum", "source_type": "forum", "bias": "retail"},
    {"query": "backtest results shared trading strategy forum", "source_type": "forum", "bias": "retail"},
    {"query": "quantconnect algorithm shared strategy", "source_type": "forum", "bias": "retail"},
    {"query": "tradingview pine script strategy popular", "source_type": "forum", "bias": "retail"},

    # --- Institutional / Professional ---
    {"query": "order book imbalance trading strategy", "source_type": "general", "bias": "institutional"},
    {"query": "Kyle lambda market impact model trading", "source_type": "arxiv", "bias": "institutional"},
    {"query": "Almgren Chriss optimal execution strategy", "source_type": "arxiv", "bias": "institutional"},
    {"query": "Hawkes process event-driven trading", "source_type": "arxiv", "bias": "institutional"},
    {"query": "market making inventory management strategy", "source_type": "general", "bias": "institutional"},
    {"query": "cross-sectional momentum factor portfolio", "source_type": "arxiv", "bias": "institutional"},
    {"query": "regime switching model hidden markov trading", "source_type": "arxiv", "bias": "institutional"},
    {"query": "carry trade strategy currency forward premium", "source_type": "arxiv", "bias": "institutional"},
]

CUSTOM_QUERIES_FILE = RESEARCH_LAB_DIR / "custom_queries.txt"


# ==============================================================================
# QUALITY SCORING WEIGHTS
# ==============================================================================
@dataclass
class QualityConfig:
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "arxiv": 1.0, "ssrn": 0.9, "github": 0.7, "blog": 0.5,
        "forum": 0.4, "reddit": 0.3, "general": 0.5,
    })
    has_math_bonus: float = 1.2
    has_backtest_bonus: float = 1.3
    has_code_bonus: float = 1.15
    has_params_bonus: float = 1.1
    min_quality_threshold: float = 0.3
    min_store_threshold: float = 0.15


# ==============================================================================
# DEDUPLICATION SETTINGS
# ==============================================================================
@dataclass
class DedupConfig:
    similarity_threshold: float = 0.85
    index_path: str = str(DATA_DIR / "dedup_index.faiss")
    metadata_path: str = str(DATA_DIR / "dedup_metadata.json")
    batch_size: int = 32


# ==============================================================================
# PIPELINE SETTINGS
# ==============================================================================
@dataclass
class PipelineConfig:
    max_strategies_per_run: int = 50
    code_validation_enabled: bool = True
    min_validation_trades: int = 0
    two_stage_extraction: bool = True
    max_retries: int = 3
    retry_delay: float = 5.0
    retry_backoff: float = 2.0
    validation_bar_count: int = 200
    min_document_length: int = 200
    max_document_length: int = 50000
    max_docs_per_query: int = 10
    log_level: str = "INFO"


# ==============================================================================
# MASTER CONFIG
# ==============================================================================
@dataclass
class DiscoveryConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    searxng: SearXNGConfig = field(default_factory=SearXNGConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    db_path: Path = DATA_DIR / "discovery.db"
    mode: str = MODE


DISCOVERY_CONFIG = DiscoveryConfig()


def print_config():
    c = DISCOVERY_CONFIG
    print("\n" + "=" * 60)
    print("  DISCOVERY PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"  Mode:            {c.mode}")
    print(f"  Ollama:          {OLLAMA_URL}")
    print(f"  Summarizer:      {c.llm.summarizer.model}")
    print(f"  Code Generator:  {c.llm.code_generator.model}")
    print(f"  Embeddings:      {c.llm.embedding_model} (local)")
    print(f"  SearXNG:         {c.searxng.base_url}")
    print(f"  Max content:     {c.llm.max_content_chars} chars")
    print(f"  Database:        {c.db_path}")
    print(f"  Strategies dir:  {STRATEGIES_DIR}")
    print(f"  Retries:         {c.pipeline.max_retries}")
    print("=" * 60)
