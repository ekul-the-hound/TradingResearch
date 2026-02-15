# ==============================================================================
# discovery_config.py
# ==============================================================================
# Central configuration for the Strategy Discovery Pipeline (Step 1)
#
# All LLM calls use OpenAI-compatible API format, so you can point them at
# any local model server: Ollama, vLLM, llama.cpp, LM Studio, text-gen-webui
#
# Usage:
#     from discovery_config import DISCOVERY_CONFIG as cfg
#     print(cfg.SEARXNG_URL)
#
# ==============================================================================

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# ==============================================================================
# BASE PATHS
# ==============================================================================
RESEARCH_LAB_DIR = Path(__file__).parent
PROJECT_DIR = RESEARCH_LAB_DIR.parent  # main TradingResearch folder
DATA_DIR = RESEARCH_LAB_DIR / "data"
STRATEGIES_DIR = RESEARCH_LAB_DIR / "discovered_strategies"
LOGS_DIR = RESEARCH_LAB_DIR / "logs"

# Create directories
for d in [DATA_DIR, STRATEGIES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# LLM CONFIGURATION (OpenAI-compatible local endpoints)
# ==============================================================================
# You can run as many local models as needed. Each role can point to a
# different model/endpoint. For example:
#   - Summarizer: small fast model (Phi-3, Qwen2.5-7B)
#   - Code generator: large capable model (DeepSeek-Coder-33B, Codestral)
#   - Reviewer: medium reasoning model (Llama-3.1-70B, Mixtral)

@dataclass
class LLMEndpoint:
    """Configuration for a single LLM endpoint."""
    base_url: str = "http://localhost:11434/v1"  # Ollama default
    api_key: str = "not-needed"  # Local models don't need keys
    model: str = "llama3.1:8b"
    max_tokens: int = 4096
    temperature: float = 0.3


@dataclass
class LLMConfig:
    """All LLM endpoints used in the discovery pipeline."""
    # Stage 1: Summarize raw documents into strategy descriptions
    summarizer: LLMEndpoint = field(default_factory=lambda: LLMEndpoint(
        model="llama3.1:8b",
        max_tokens=1024,
        temperature=0.2,
    ))
    # Stage 2: Generate Backtrader code from strategy summaries
    code_generator: LLMEndpoint = field(default_factory=lambda: LLMEndpoint(
        model="deepseek-coder-v2:16b",
        max_tokens=8192,
        temperature=0.1,
    ))
    # Adversarial reviewer (optional, reuses existing adversarial_reviewer.py pattern)
    reviewer: LLMEndpoint = field(default_factory=lambda: LLMEndpoint(
        model="llama3.1:8b",
        max_tokens=2048,
        temperature=0.4,
    ))
    # Embedding model for semantic deduplication
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers
    embedding_dim: int = 384


# ==============================================================================
# SEARXNG CONFIGURATION
# ==============================================================================
@dataclass
class SearXNGConfig:
    """SearXNG metasearch engine settings."""
    base_url: str = "http://localhost:8080"
    # Max results per query
    max_results_per_query: int = 30
    # Rate limiting (seconds between requests)
    request_delay: float = 2.0
    # Timeout per request (seconds)
    timeout: int = 30
    # Output format
    output_format: str = "json"
    # Search categories
    categories: str = "general,science"
    # Language
    language: str = "en"


# ==============================================================================
# SEARCH QUERIES
# ==============================================================================
# Organized by source bias: academic, retail, institutional
# Each query is a dict with the query string, expected source type, and bias tag

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
    {"query": "forex trading strategy python implementation", "source_type": "github", "bias": "retail"},

    # --- Forums / Community ---
    {"query": "trading strategy backtest results profitable site:reddit.com/r/algotrading", "source_type": "reddit", "bias": "retail"},
    {"query": "systematic trading strategy that works site:reddit.com", "source_type": "reddit", "bias": "retail"},
    {"query": "quantitative trading strategy discussion forum", "source_type": "forum", "bias": "retail"},

    # --- Institutional / Advanced (underrepresented - bias correction) ---
    {"query": "order book imbalance trading strategy", "source_type": "general", "bias": "institutional"},
    {"query": "Kyle lambda market impact model trading", "source_type": "arxiv", "bias": "institutional"},
    {"query": "Almgren Chriss optimal execution strategy", "source_type": "arxiv", "bias": "institutional"},
    {"query": "Hawkes process event-driven trading", "source_type": "arxiv", "bias": "institutional"},
    {"query": "market making inventory management strategy", "source_type": "general", "bias": "institutional"},
    {"query": "cross-sectional momentum factor portfolio", "source_type": "arxiv", "bias": "institutional"},
    {"query": "regime switching model hidden markov trading", "source_type": "arxiv", "bias": "institutional"},
    {"query": "carry trade strategy currency forward premium", "source_type": "arxiv", "bias": "institutional"},
]

# Custom queries file - user can add more queries here
CUSTOM_QUERIES_FILE = RESEARCH_LAB_DIR / "custom_queries.txt"


# ==============================================================================
# QUALITY SCORING WEIGHTS
# ==============================================================================
@dataclass
class QualityConfig:
    """Weights for strategy quality scoring."""
    # Source type weights (higher = more trustworthy)
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "arxiv": 1.0,
        "ssrn": 0.9,
        "github": 0.7,
        "blog": 0.5,
        "forum": 0.4,
        "reddit": 0.3,
        "general": 0.5,
    })
    # Bonus multipliers for quality signals
    has_math_bonus: float = 1.2       # Document contains equations/formulas
    has_backtest_bonus: float = 1.3   # Document includes backtest results
    has_code_bonus: float = 1.15      # Document includes code snippets
    has_params_bonus: float = 1.1     # Strategy has explicit parameters
    # Minimum quality score to proceed to code generation (0.0 - 1.0 scale after normalization)
    min_quality_threshold: float = 0.3
    # Minimum quality to keep in database (below this = log failure and skip)
    min_store_threshold: float = 0.15


# ==============================================================================
# DEDUPLICATION SETTINGS
# ==============================================================================
@dataclass
class DedupConfig:
    """FAISS semantic deduplication settings."""
    # Cosine similarity threshold (0.0 = nothing alike, 1.0 = identical)
    similarity_threshold: float = 0.85
    # Path to FAISS index file
    index_path: Path = field(default_factory=lambda: DATA_DIR / "strategy_embeddings.faiss")
    # Path to metadata mapping (index position -> strategy_id)
    metadata_path: Path = field(default_factory=lambda: DATA_DIR / "embedding_metadata.json")


# ==============================================================================
# PIPELINE SETTINGS
# ==============================================================================
@dataclass
class PipelineConfig:
    """Overall pipeline behavior."""
    # Maximum strategies to discover per run
    max_strategies_per_run: int = 50
    # Maximum documents to fetch per query
    max_docs_per_query: int = 10
    # Skip documents shorter than this (characters)
    min_document_length: int = 200
    # Skip documents longer than this (characters) - prevents feeding entire textbooks
    max_document_length: int = 50000
    # Enable two-stage extraction (summarize -> code gen)
    two_stage_extraction: bool = True
    # Enable code execution validation gate
    code_validation_enabled: bool = True
    # Dummy backtest bar count for validation
    validation_bar_count: int = 200
    # Minimum trades required in validation backtest
    min_validation_trades: int = 1
    # Log level
    log_level: str = "INFO"


# ==============================================================================
# DATABASE
# ==============================================================================
RESEARCH_DB_PATH = DATA_DIR / "research_lab.db"


# ==============================================================================
# MASTER CONFIG OBJECT
# ==============================================================================
@dataclass
class DiscoveryConfig:
    """Master configuration combining all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    searxng: SearXNGConfig = field(default_factory=SearXNGConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    search_queries: List[Dict[str, str]] = field(default_factory=lambda: SEARCH_QUERIES)
    db_path: Path = field(default_factory=lambda: RESEARCH_DB_PATH)


# Singleton instance - import this
DISCOVERY_CONFIG = DiscoveryConfig()


# ==============================================================================
# CONVENIENCE: Print config summary
# ==============================================================================
def print_config():
    """Print current configuration for debugging."""
    cfg = DISCOVERY_CONFIG
    print("=" * 60)
    print("STRATEGY DISCOVERY PIPELINE - CONFIGURATION")
    print("=" * 60)
    print(f"\nSearXNG URL:        {cfg.searxng.base_url}")
    print(f"Search queries:     {len(cfg.search_queries)}")
    print(f"Max per run:        {cfg.pipeline.max_strategies_per_run}")
    print(f"\nLLM Summarizer:     {cfg.llm.summarizer.base_url} / {cfg.llm.summarizer.model}")
    print(f"LLM Code Gen:       {cfg.llm.code_generator.base_url} / {cfg.llm.code_generator.model}")
    print(f"Embedding Model:    {cfg.llm.embedding_model}")
    print(f"\nDedup threshold:    {cfg.dedup.similarity_threshold}")
    print(f"Quality threshold:  {cfg.quality.min_quality_threshold}")
    print(f"Two-stage extract:  {cfg.pipeline.two_stage_extraction}")
    print(f"Code validation:    {cfg.pipeline.code_validation_enabled}")
    print(f"\nDB path:            {cfg.db_path}")
    print(f"Strategies dir:     {STRATEGIES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
