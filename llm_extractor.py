# ==============================================================================
# llm_extractor.py
# ==============================================================================
# Two-Stage LLM Strategy Extractor (Ollama)
#
# Stage 1 (Summarizer): Reads raw document -> structured strategy summary
# Stage 2 (Code Generator): Reads summary -> Backtrader strategy code
#
# Uses Ollama's OpenAI-compatible API (local or cloud mode).
# ==============================================================================

import re
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import requests

from discovery_config import DISCOVERY_CONFIG as cfg, LLMEndpoint
from research_db import ResearchDatabase
from quality_scorer import QualityScorer

logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

SUMMARIZE_SYSTEM_PROMPT = """You are a quantitative trading researcher. Your job is to extract trading strategy ideas from documents.

You MUST respond with ONLY a JSON object (no markdown, no backticks, no preamble). The JSON must have these exact keys:

{
    "strategy_name": "short descriptive name using PascalCase (e.g., RSIMeanReversion, MomentumBreakout)",
    "strategy_type": "one of: trend_following, mean_reversion, breakout, momentum, volatility, statistical_arbitrage, market_making, event_driven, other",
    "summary": "2-4 sentence description of the core strategy logic",
    "entry_rules": ["list of specific entry conditions"],
    "exit_rules": ["list of specific exit conditions"],
    "indicators": ["list of technical indicators used"],
    "parameters": {"param_name": "default_value or description"},
    "timeframe": "recommended timeframe (e.g., 1H, 4H, 1D)",
    "asset_class": "forex, crypto, equities, futures, or multi",
    "risk_management": "description of position sizing / stop loss approach",
    "confidence": "high, medium, or low - how clearly the document describes the strategy"
}

If the document does NOT contain a clear trading strategy, respond with:
{"strategy_name": "NONE", "confidence": "none", "summary": "reason why no strategy found"}

Focus on EXTRACTING what is described, not inventing new ideas."""


SUMMARIZE_USER_TEMPLATE = """Extract the trading strategy from this document.

DOCUMENT TITLE: {title}
SOURCE: {source_type} ({source_bias})

DOCUMENT CONTENT:
{content}

Respond with ONLY the JSON object, nothing else."""


CODEGEN_SYSTEM_PROMPT = """You are an expert Backtrader developer. Your job is to convert a strategy description into a working Backtrader strategy class.

CRITICAL BACKTRADER CODING RULES:

### Rule 1: Indicator Names
Use CORRECT Backtrader indicator names:
- bt.indicators.OBV(self.data)
- bt.indicators.RSI(self.data.close, period=14)
- bt.indicators.ATR(self.data, period=14)
- bt.indicators.ADX(self.data, period=14)
- bt.indicators.BollingerBands(self.data.close)
- bt.indicators.MACD(self.data.close)
- bt.indicators.Stochastic(self.data)
- bt.indicators.SimpleMovingAverage / ExponentialMovingAverage
- bt.indicators.CrossOver(fast, slow)

### Rule 2: Position Price Access
ALWAYS check position exists before accessing .price:
```python
if self.position and self.position.size != 0:
    entry_price = self.position.price
```

### Rule 3: Minimum Bar Checks
ALWAYS add at start of next():
```python
def next(self):
    if len(self) < self.params.slow_period:
        return
```

### Rule 4: Multi-Timeframe Safety
Check data availability with try/except when using multiple data feeds.

### Rule 5: Partial Exits
Use explicit size: self.sell(size=half_size), NOT self.close()

OUTPUT REQUIREMENTS:
- Complete, working Python file
- Single class inheriting from bt.Strategy
- Include all imports (import backtrader as bt)
- Include params tuple with ALL parameters
- Include docstring explaining the strategy
- The class name MUST match the strategy_name from the summary
- NO markdown code blocks - just raw Python code
- NO explanation text before or after the code"""


CODEGEN_USER_TEMPLATE = """Convert this strategy description into a working Backtrader strategy class.

STRATEGY NAME: {strategy_name}
TYPE: {strategy_type}
SUMMARY: {summary}

ENTRY RULES:
{entry_rules}

EXIT RULES:
{exit_rules}

INDICATORS: {indicators}
PARAMETERS: {parameters}
TIMEFRAME: {timeframe}
RISK MANAGEMENT: {risk_management}

Generate the complete Backtrader strategy class. Output ONLY the Python code, nothing else."""


# ==============================================================================
# OLLAMA CLIENT (with retry logic for cloud mode)
# ==============================================================================

class OllamaClient:
    """
    Client for Ollama's OpenAI-compatible API.
    Works for both local and cloud models (cloud models end with :cloud).
    Includes retry logic with exponential backoff for cloud reliability.
    """

    def __init__(self, endpoint: LLMEndpoint):
        self.base_url = endpoint.base_url.rstrip("/")
        self.model = endpoint.model
        self.max_tokens = endpoint.max_tokens
        self.temperature = endpoint.temperature
        self.timeout = getattr(endpoint, 'timeout', 120)

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        self.max_retries = cfg.pipeline.max_retries
        self.retry_delay = cfg.pipeline.retry_delay
        self.retry_backoff = cfg.pipeline.retry_backoff

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: Optional[float] = None) -> Tuple[str, Dict]:
        """Send a chat completion request with automatic retry."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }

        last_error = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", delay))
                    logger.warning(f"Rate limited. Waiting {retry_after}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(retry_after)
                    delay *= self.retry_backoff
                    continue

                if response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code} (attempt {attempt}/{self.max_retries})")
                    time.sleep(delay)
                    delay *= self.retry_backoff
                    continue

                response.raise_for_status()
                data = response.json()

                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                # ── Empty-response retry (silent rate limiting: 200 OK + empty body) ──
                if not text or not text.strip():
                    logger.warning(
                        f"Empty response from {self.model} "
                        f"(attempt {attempt}/{self.max_retries}) – likely silent rate-limit"
                    )
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        delay *= self.retry_backoff
                        continue
                    else:
                        raise ValueError(
                            f"Empty response from {self.model} after "
                            f"{self.max_retries} attempts (silent rate-limit)"
                        )

                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)

                return text, usage

            except requests.exceptions.ConnectionError:
                last_error = ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Is Ollama running? (attempt {attempt}/{self.max_retries})"
                )
                if attempt < self.max_retries:
                    logger.warning(f"Connection failed, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff

            except requests.exceptions.Timeout:
                last_error = TimeoutError(
                    f"Ollama request timed out after {self.timeout}s "
                    f"(model: {self.model}, attempt {attempt}/{self.max_retries})"
                )
                if attempt < self.max_retries:
                    logger.warning(f"Timeout, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff

            except KeyError as e:
                raise ValueError(f"Unexpected Ollama response format: {e}\nResponse: {response.text[:500]}")

        raise last_error

    def health_check(self) -> bool:
        """Check if Ollama is responding and the model is available."""
        try:
            resp = self.session.get(f"{self.base_url}/models", timeout=10)
            if resp.status_code == 200:
                return True
        except Exception:
            pass

        try:
            text, _ = self.chat("You are a test.", "Say OK", temperature=0)
            return bool(text)
        except Exception:
            return False

    def get_token_stats(self) -> Dict:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }


# Backwards compatibility alias
LocalLLMClient = OllamaClient


# ==============================================================================
# EXTRACTOR
# ==============================================================================

class LLMExtractor:
    """
    Two-stage strategy extractor using Ollama.

    Stage 1 (Summarizer): Raw document -> Structured JSON summary
    Stage 2 (Code Generator): Strategy summary -> Working Backtrader code
    """

    def __init__(self, db: Optional[ResearchDatabase] = None):
        self.summarizer = OllamaClient(cfg.llm.summarizer)
        self.code_generator = OllamaClient(cfg.llm.code_generator)
        self.scorer = QualityScorer()
        self.db = db or ResearchDatabase()

        self.stats = {
            "documents_processed": 0,
            "summaries_extracted": 0,
            "code_generated": 0,
            "no_strategy_found": 0,
            "extraction_errors": 0,
            "quality_rejected": 0,
        }

    # ==========================================================================
    # STAGE 1: SUMMARIZE
    # ==========================================================================

    def summarize_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Stage 1: Extract a structured strategy summary from a raw document."""
        doc_id = doc.get("doc_id", "unknown")
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        source_type = doc.get("source_type", "general")
        source_bias = doc.get("source_bias", "unknown")

        max_content = cfg.llm.max_content_chars
        if len(content) > max_content:
            content = content[:max_content] + "\n\n[TRUNCATED]"

        user_prompt = SUMMARIZE_USER_TEMPLATE.format(
            title=title, source_type=source_type,
            source_bias=source_bias, content=content,
        )

        start_time = time.time()
        try:
            response_text, usage = self.summarizer.chat(
                SUMMARIZE_SYSTEM_PROMPT, user_prompt,
            )
            duration = time.time() - start_time

            self.db.log_extraction({
                "doc_id": doc_id, "stage": "summarize",
                "model_used": cfg.llm.summarizer.model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "success": True, "duration_seconds": duration,
            })

            summary = self._parse_json_response(response_text)
            if summary is None:
                logger.warning(f"Failed to parse summary JSON for doc {doc_id}")
                self.db.log_extraction({
                    "doc_id": doc_id, "stage": "summarize_parse",
                    "model_used": cfg.llm.summarizer.model,
                    "success": False, "error_message": "JSON parse failure",
                    "duration_seconds": duration,
                })
                self.stats["extraction_errors"] += 1
                return None

            if summary.get("strategy_name") == "NONE" or summary.get("confidence") == "none":
                logger.info(f"No strategy found in doc {doc_id}: {summary.get('summary', 'N/A')}")
                self.stats["no_strategy_found"] += 1
                self.db.update_document_status(doc_id, "no_strategy")
                return None

            self.stats["summaries_extracted"] += 1
            logger.info(f"Extracted summary: {summary.get('strategy_name')} from doc {doc_id}")
            return summary

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Ollama error during summarization: {e}")
            self.db.log_extraction({
                "doc_id": doc_id, "stage": "summarize",
                "model_used": cfg.llm.summarizer.model,
                "success": False, "error_message": str(e),
                "duration_seconds": time.time() - start_time,
            })
            self.stats["extraction_errors"] += 1
            return None

    # ==========================================================================
    # STAGE 2: GENERATE CODE
    # ==========================================================================

    def generate_code(self, summary: Dict[str, Any],
                      doc_id: str = "unknown") -> Optional[str]:
        """Stage 2: Generate Backtrader strategy code from a summary."""
        entry_rules = "\n".join(f"- {r}" for r in summary.get("entry_rules", ["Not specified"]))
        exit_rules = "\n".join(f"- {r}" for r in summary.get("exit_rules", ["Not specified"]))
        indicators = ", ".join(summary.get("indicators", ["None specified"]))
        parameters = json.dumps(summary.get("parameters", {}), indent=2)

        user_prompt = CODEGEN_USER_TEMPLATE.format(
            strategy_name=summary.get("strategy_name", "UnknownStrategy"),
            strategy_type=summary.get("strategy_type", "unknown"),
            summary=summary.get("summary", ""),
            entry_rules=entry_rules, exit_rules=exit_rules,
            indicators=indicators, parameters=parameters,
            timeframe=summary.get("timeframe", "1D"),
            risk_management=summary.get("risk_management", "Not specified"),
        )

        start_time = time.time()
        try:
            response_text, usage = self.code_generator.chat(
                CODEGEN_SYSTEM_PROMPT, user_prompt,
            )
            duration = time.time() - start_time

            self.db.log_extraction({
                "doc_id": doc_id, "stage": "code_generation",
                "model_used": cfg.llm.code_generator.model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "success": True, "duration_seconds": duration,
            })

            code = self._clean_code_response(response_text)

            if not code or "class " not in code:
                logger.warning(f"Code generation produced no valid class for doc {doc_id}")
                self.stats["extraction_errors"] += 1
                return None

            self.stats["code_generated"] += 1
            logger.info(f"Generated code for {summary.get('strategy_name')} ({len(code)} chars)")
            return code

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Ollama error during code generation: {e}")
            self.db.log_extraction({
                "doc_id": doc_id, "stage": "code_generation",
                "model_used": cfg.llm.code_generator.model,
                "success": False, "error_message": str(e),
                "duration_seconds": time.time() - start_time,
            })
            self.stats["extraction_errors"] += 1
            return None

    # ==========================================================================
    # FULL EXTRACTION PIPELINE
    # ==========================================================================

    def extract_strategy(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Full two-stage extraction: document -> summary -> code."""
        doc_id = doc.get("doc_id", "unknown")
        self.stats["documents_processed"] += 1

        passes, score_result = self.scorer.passes_extraction_threshold(doc)
        if not passes:
            logger.info(f"Doc {doc_id} rejected by quality gate (score={score_result['quality_score']:.3f})")
            self.stats["quality_rejected"] += 1
            self.db.update_document_status(doc_id, "quality_rejected")
            return None

        logger.info(f"Stage 1: Summarizing doc {doc_id}...")
        summary = self.summarize_document(doc)
        if summary is None:
            return None

        strategy_data = {
            "doc_id": doc_id,
            "strategy_name": summary.get("strategy_name", "Unknown"),
            "summary": summary.get("summary", ""),
            "description": json.dumps(summary),
            "origin_source": "scraped",
            "source_url": doc.get("url"),
            "source_type": doc.get("source_type"),
            "source_bias": doc.get("source_bias"),
            "parent_docs": [doc_id],
            "quality_score": score_result["quality_score"],
            "has_math": score_result["has_math"],
            "has_backtest": score_result["has_backtest"],
            "has_code": score_result["has_code"],
            "has_explicit_params": score_result["has_explicit_params"],
            "extraction_model": cfg.llm.summarizer.model,
        }

        if cfg.pipeline.two_stage_extraction:
            logger.info(f"Stage 2: Generating code for {summary.get('strategy_name')}...")
            code = self.generate_code(summary, doc_id=doc_id)

            if code:
                strategy_data["generated_code"] = code
                strategy_data["extraction_model"] = (
                    f"{cfg.llm.summarizer.model}+{cfg.llm.code_generator.model}"
                )
                strategy_data["status"] = "extracted"
            else:
                strategy_data["status"] = "code_gen_failed"
                strategy_data["validation_error"] = "Code generation failed"
        else:
            strategy_data["status"] = "summary_only"

        self.db.update_document_status(doc_id, "processed")
        return strategy_data

    def extract_batch(self, documents: list, delay: float = 1.0) -> list:
        """Extract strategies from a batch of documents."""
        strategies = []
        total = len(documents)

        for i, doc in enumerate(documents, 1):
            doc_id = doc.get("doc_id", "unknown")
            logger.info(f"[{i}/{total}] Processing doc {doc_id}...")

            try:
                result = self.extract_strategy(doc)
                if result:
                    strat_id = self.db.save_strategy(result)
                    result["strategy_id"] = strat_id
                    strategies.append(result)
                    logger.info(f"  -> Saved strategy: {result.get('strategy_name')} ({strat_id})")
            except Exception as e:
                logger.error(f"  -> Unexpected error processing doc {doc_id}: {e}")
                self.stats["extraction_errors"] += 1

            if i < total:
                time.sleep(delay)

        return strategies

    # ==========================================================================
    # HELPERS
    # ==========================================================================

    @staticmethod
    def _parse_json_response(text: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling common formatting issues."""
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(text.replace("'", '"'))
        except (json.JSONDecodeError, Exception):
            pass

        logger.warning(f"Could not parse JSON from response: {text[:200]}...")
        return None

    @staticmethod
    def _clean_code_response(text: str) -> str:
        """Clean up code from LLM response."""
        text = text.strip()
        text = re.sub(r'^```(?:python|py)?\s*\n?', '', text)
        text = re.sub(r'\n?\s*```$', '', text)

        lines = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ') or
                stripped.startswith('class ') or stripped.startswith('#')):
                start_idx = i
                break

        code = '\n'.join(lines[start_idx:])

        code_lines = code.split('\n')
        end_idx = len(code_lines)
        in_class = False
        for i, line in enumerate(code_lines):
            if line.strip().startswith('class '):
                in_class = True
            elif in_class and line.strip() and not line[0].isspace() and not line.startswith('#'):
                if not line.strip().startswith(('import ', 'from ', 'class ', 'def ')):
                    end_idx = i
                    break

        return '\n'.join(code_lines[:end_idx]).rstrip()

    # ==========================================================================
    # HEALTH CHECK & STATS
    # ==========================================================================

    def health_check(self) -> Dict[str, bool]:
        """Check if Ollama is responding for both models."""
        results = {}
        print(f"Checking Ollama endpoints ({cfg.mode} mode)...")
        for name, client in [("summarizer", self.summarizer),
                              ("code_generator", self.code_generator)]:
            ok = client.health_check()
            status = "OK" if ok else "FAIL"
            cloud_tag = " (cloud)" if ":cloud" in client.model else " (local)"
            print(f"  [{status}] {name}: {client.model}{cloud_tag}")
            results[name] = ok
        return results

    def print_stats(self):
        """Print extraction session statistics."""
        s = self.stats
        sum_t = self.summarizer.get_token_stats()
        code_t = self.code_generator.get_token_stats()
        print("\n" + "=" * 50)
        print("LLM EXTRACTOR - SESSION STATS")
        print("=" * 50)
        print(f"Mode:                  {cfg.mode}")
        print(f"Summarizer:            {cfg.llm.summarizer.model}")
        print(f"Code Generator:        {cfg.llm.code_generator.model}")
        print(f"Documents processed:   {s['documents_processed']}")
        print(f"Quality rejected:      {s['quality_rejected']}")
        print(f"No strategy found:     {s['no_strategy_found']}")
        print(f"Summaries extracted:   {s['summaries_extracted']}")
        print(f"Code generated:        {s['code_generated']}")
        print(f"Errors:                {s['extraction_errors']}")
        print(f"Summarizer tokens:     {sum_t['input_tokens']}in / {sum_t['output_tokens']}out")
        print(f"Code gen tokens:       {code_t['input_tokens']}in / {code_t['output_tokens']}out")
        print("=" * 50)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from discovery_config import print_config
    print_config()

    extractor = LLMExtractor()
    results = extractor.health_check()
    if not all(results.values()):
        print("\nOllama is not responding or models are not available.")
        if cfg.mode == "cloud":
            print("For cloud mode, make sure you ran: ollama signin")
            print("And pull the cloud models:")
            print("  ollama pull qwen3.5-flash:cloud")
            print("  ollama pull qwen3-coder:480b-cloud")
        else:
            print("For local mode, make sure Ollama is running: ollama serve")
            print("And pull the local models:")
            print("  ollama pull qwen3.5:9b")
            print("  ollama pull qwen2.5-coder:7b")
        exit(1)

    docs = extractor.db.get_unprocessed_documents(limit=10)
    if not docs:
        print("\nNo unprocessed documents found. Run searxng_scraper.py first.")
        exit(0)

    print(f"\nProcessing {len(docs)} documents...")
    strategies = extractor.extract_batch(docs)

    print(f"\nExtracted {len(strategies)} strategies:")
    for s in strategies:
        print(f"  - {s['strategy_name']} (quality={s['quality_score']:.3f})")

    extractor.print_stats()
    extractor.db.print_stats()