# ==============================================================================
# source_extractor.py -- Paste-a-Source Strategy Extraction
# ==============================================================================
# Paste YouTube transcripts / articles into the dashboard, extract EVERY
# strategy described (written out in plain English, no code), review them
# in an accordion, then Approve -> code generation -> validation -> the
# strategy lands in the main `strategies` table (origin_source='transcript').
#
# Design:
#   - Background worker thread + extraction_status.json (same pattern as
#     backtest_status.json). Dashboard polls the file every 2s.
#   - Source text is held ONLY in memory during the job and discarded after
#     extraction. Nothing persisted. Re-extracting = re-pasting.
#   - New table `source_strategies` in discovery.db -- separate from the
#     discovery feed until you approve.
#   - Model picker: SOURCE_MODELS below. Add strings to the list to add
#     options to the dashboard dropdown.
#
# CLI (for testing without the dashboard):
#   python source_extractor.py --status
#   python source_extractor.py --list
#   python source_extractor.py --extract-file transcript.txt --title "My Video"
# ==============================================================================

import os
import re
import sys
import json
import time
import uuid
import queue
import sqlite3
import argparse
import threading
import ast
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent))

# ------------------------------------------------------------------------
# Config -- reuse discovery_config where available, fall back gracefully
# ------------------------------------------------------------------------
try:
    from discovery_config import DISCOVERY_CONFIG as cfg, OLLAMA_URL
    DB_PATH = str(cfg.db_path)
    _MAX_CONTENT = getattr(cfg.llm, "max_content_chars", 30000)
    _MAX_RETRIES = getattr(cfg.pipeline, "max_retries", 3)
    _RETRY_DELAY = getattr(cfg.pipeline, "retry_delay", 5.0)
    _RETRY_BACKOFF = getattr(cfg.pipeline, "retry_backoff", 2.0)
    _VALIDATION_BARS = getattr(cfg.pipeline, "validation_bar_count", 600)
    _VALIDATION_TIMEOUT = getattr(cfg.pipeline, "validation_timeout_seconds", 60)
except ImportError:
    OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
    DB_PATH = str(Path(__file__).parent / "data" / "discovery.db")
    _MAX_CONTENT = 30000
    _MAX_RETRIES = 3
    _RETRY_DELAY = 5.0
    _RETRY_BACKOFF = 2.0
    _VALIDATION_BARS = 600
    _VALIDATION_TIMEOUT = 60

BASE_DIR = Path(__file__).parent
STATUS_FILE = BASE_DIR / "extraction_status.json"

# ------------------------------------------------------------------------
# MODEL SELECTION -- add strings here to add options to the dashboard
# ------------------------------------------------------------------------
SOURCE_MODELS = [
    "minimax-m3:cloud",
]
DEFAULT_MODEL = SOURCE_MODELS[0]

REQUEST_TIMEOUT = 300  # seconds per LLM call
EXTRACT_MAX_TOKENS = 8192
CODEGEN_MAX_TOKENS = 8192


# ==============================================================================
# PROMPTS
# ==============================================================================

EXTRACT_SYSTEM_PROMPT = """You are a quantitative trading researcher. You extract trading strategies from transcripts and articles.

A single source may describe SEVERAL strategies or SEVERAL distinct variations of one strategy. Extract EACH one as a SEPARATE entry. A variation counts as separate if it has different entry/exit rules, different indicators, or a materially different parameter regime.

You MUST respond with ONLY a JSON array (no markdown, no backticks, no preamble, no explanation). Each element must have these exact keys:

{
    "name": "short descriptive PascalCase name (e.g. RSIMeanReversion)",
    "summary": "2-4 sentence plain-English description of the core logic",
    "hypothesis": "why the author claims this works (the edge)",
    "entry_rules": ["specific entry conditions, one per element"],
    "exit_rules": ["specific exit conditions, one per element"],
    "stop_loss": "stop loss approach as described, or 'Not specified'",
    "take_profit": "take profit approach as described, or 'Not specified'",
    "indicators": ["indicator names with parameters, e.g. 'RSI(14)', 'EMA(50)'"],
    "parameters": {"param_name": "value or description"},
    "asset_class": "forex, crypto, equities, futures, indices, or multi",
    "timeframe": "timeframe mentioned (e.g. 1H, 4H, 1D) or 'Not specified'",
    "position_sizing": "position sizing / risk-per-trade as described, or 'Not specified'",
    "confidence": "high, medium, or low -- how clearly the source describes it",
    "source_quote": "one short verbatim quote (max 25 words) from the source that best captures this strategy"
}

Rules:
- EXTRACT what the source describes. Do NOT invent rules the source does not state.
- If a rule is vague, record it as stated and lower the confidence.
- If the source contains NO trading strategy at all, respond with exactly: []
"""

EXTRACT_USER_TEMPLATE = """Extract ALL trading strategies from this source.

SOURCE TITLE: {title}

SOURCE CONTENT:
{content}

Respond with ONLY the JSON array, nothing else."""


# Codegen prompt uses ONLY verified Backtrader indicators (Rule 1),
# manual computation for anything else (Rule 1b), and bans fixed-size
# bar-indexed buffers (Rule 6).
CODEGEN_SYSTEM_PROMPT = """You are an expert Backtrader developer. Convert a strategy description into a working Backtrader strategy class.

CRITICAL BACKTRADER CODING RULES:

### Rule 1: Use ONLY these verified Backtrader indicators
- bt.indicators.SimpleMovingAverage(data, period=N) / bt.indicators.SMA
- bt.indicators.ExponentialMovingAverage(data, period=N) / bt.indicators.EMA
- bt.indicators.WeightedMovingAverage(data, period=N)
- bt.indicators.RSI(data, period=N)
- bt.indicators.ATR(data, period=N)
- bt.indicators.ADX(data, period=N)
- bt.indicators.BollingerBands(data, period=N, devfactor=X)
- bt.indicators.MACD(data)
- bt.indicators.Stochastic(data)
- bt.indicators.CCI(data, period=N)
- bt.indicators.WilliamsR(data, period=N)
- bt.indicators.Momentum(data, period=N)
- bt.indicators.RateOfChange(data, period=N)
- bt.indicators.StdDev(data, period=N)
- bt.indicators.Highest(data, period=N)
- bt.indicators.Lowest(data, period=N)
- bt.indicators.CrossOver(a, b)
- bt.indicators.ParabolicSAR(data)
- bt.indicators.Ichimoku(data)
- bt.indicators.DirectionalMovementIndex(data)
- bt.indicators.AroonUpDown(data, period=N)
- bt.indicators.TrueRange(data)
Do NOT use any indicator not on this list. bt.indicators.OBV does NOT exist.

### Rule 1b: Unsupported indicators -> compute manually
If the strategy needs an indicator not on the list (OBV, VWAP, SuperTrend, Keltner, etc.), compute it manually inside next() using self.data lines and instance-variable accumulators. Example OBV:
```python
def __init__(self):
    self.obv = 0.0
def next(self):
    if len(self) > 1:
        if self.data.close[0] > self.data.close[-1]:
            self.obv += self.data.volume[0]
        elif self.data.close[0] < self.data.close[-1]:
            self.obv -= self.data.volume[0]
```

### Rule 2: Position price access
ALWAYS check the position exists before accessing .price:
```python
if self.position and self.position.size != 0:
    entry_price = self.position.price
```

### Rule 3: Minimum bar checks
ALWAYS add at the start of next():
```python
def next(self):
    if len(self) < self.params.slow_period:
        return
```

### Rule 4: Multi-timeframe / multi-data safety
Check data availability with try/except when using multiple data feeds.

### Rule 5: Partial exits
Use explicit size: self.sell(size=half_size), NOT self.close()

### Rule 6: No fixed-size bar-indexed buffers
NEVER pre-allocate arrays indexed by bar number (e.g. numpy arrays of length = total bars). Use rolling instance variables, collections.deque(maxlen=N), or Backtrader line access (self.data.close[-k]) instead.

OUTPUT REQUIREMENTS:
- Complete, working Python file
- Single class inheriting from bt.Strategy
- Include all imports (import backtrader as bt)
- Include a params tuple with ALL parameters
- Include a docstring explaining the strategy
- The class name MUST match the strategy name given
- NO markdown code blocks -- just raw Python code
- NO explanation text before or after the code"""

CODEGEN_USER_TEMPLATE = """Convert this strategy description into a working Backtrader strategy class.

STRATEGY NAME: {name}
SUMMARY: {summary}
HYPOTHESIS: {hypothesis}

ENTRY RULES:
{entry_rules}

EXIT RULES:
{exit_rules}

STOP LOSS: {stop_loss}
TAKE PROFIT: {take_profit}
INDICATORS: {indicators}
PARAMETERS: {parameters}
TIMEFRAME: {timeframe}
POSITION SIZING: {position_sizing}

Generate the complete Backtrader strategy class. Output ONLY the Python code, nothing else."""


# ==============================================================================
# DATABASE
# ==============================================================================

def _conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def ensure_source_tables():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS source_strategies (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            source_title TEXT,
            model_used TEXT,
            extracted_at TEXT,
            name TEXT,
            summary TEXT,
            hypothesis TEXT,
            entry_rules TEXT,
            exit_rules TEXT,
            stop_loss TEXT,
            take_profit TEXT,
            indicators TEXT,
            parameters TEXT,
            asset_class TEXT,
            timeframe TEXT,
            position_sizing TEXT,
            confidence TEXT,
            source_quote TEXT,
            status TEXT DEFAULT 'pending',
            codegen_status TEXT DEFAULT '',
            code_valid INTEGER DEFAULT 0,
            validation_error TEXT DEFAULT '',
            promoted_strategy_id TEXT DEFAULT ''
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_src_status ON source_strategies(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_src_job ON source_strategies(job_id)")
    c.commit()
    c.close()


def list_source_strategies(limit: int = 100) -> List[Dict]:
    ensure_source_tables()
    c = _conn()
    rows = [dict(r) for r in c.execute(
        "SELECT * FROM source_strategies ORDER BY extracted_at DESC, name ASC LIMIT ?",
        (limit,)).fetchall()]
    c.close()
    return rows


def get_source_strategy(row_id: str) -> Optional[Dict]:
    c = _conn()
    r = c.execute("SELECT * FROM source_strategies WHERE id=?", (row_id,)).fetchone()
    c.close()
    return dict(r) if r else None


def update_source_strategy(row_id: str, **fields):
    if not fields:
        return
    c = _conn()
    sets = ", ".join(f"{k}=?" for k in fields)
    c.execute(f"UPDATE source_strategies SET {sets} WHERE id=?",
              list(fields.values()) + [row_id])
    c.commit()
    c.close()


def reject_strategy(row_id: str):
    update_source_strategy(row_id, status="rejected")


def delete_source_strategy(row_id: str):
    c = _conn()
    c.execute("DELETE FROM source_strategies WHERE id=?", (row_id,))
    c.commit()
    c.close()


def source_stats() -> Dict:
    ensure_source_tables()
    c = _conn()
    def _n(sql):
        return c.execute(sql).fetchone()[0]
    out = {
        "total": _n("SELECT COUNT(*) FROM source_strategies"),
        "pending": _n("SELECT COUNT(*) FROM source_strategies WHERE status='pending'"),
        "approved": _n("SELECT COUNT(*) FROM source_strategies WHERE status='approved'"),
        "rejected": _n("SELECT COUNT(*) FROM source_strategies WHERE status='rejected'"),
        "valid_code": _n("SELECT COUNT(*) FROM source_strategies WHERE code_valid=1"),
    }
    c.close()
    return out


# ==============================================================================
# STATUS FILE (atomic write, dashboard polls this)
# ==============================================================================

_status_lock = threading.Lock()


def _write_status(**kw):
    with _status_lock:
        data = {}
        if STATUS_FILE.exists():
            try:
                data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data.update(kw)
        data["updated_at"] = datetime.now().isoformat()
        data["queued"] = _job_queue.qsize()
        tmp = STATUS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, STATUS_FILE)


def read_status() -> Dict:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "idle", "queued": 0}


# ==============================================================================
# OLLAMA CLIENT (self-contained: think disabled, think-strip, retries)
# ==============================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def ollama_chat(model: str, system_prompt: str, user_prompt: str,
                max_tokens: int = EXTRACT_MAX_TOKENS,
                temperature: float = 0.2) -> str:
    """One chat completion with retry/backoff. Raises on final failure."""
    url = f"{OLLAMA_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "think": False,
    }
    delay = _RETRY_DELAY
    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("Retry-After", delay))
                except (TypeError, ValueError):
                    wait = delay
                last_error = ConnectionError("Rate limited (429)")
                time.sleep(wait)
                delay *= _RETRY_BACKOFF
                continue
            if resp.status_code >= 500:
                last_error = ConnectionError(f"Server error {resp.status_code}")
                time.sleep(delay)
                delay *= _RETRY_BACKOFF
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if not text or not text.strip():
                last_error = ValueError("Empty response (silent rate limit?)")
                time.sleep(delay)
                delay *= _RETRY_BACKOFF
                continue
            return _strip_think(text)
        except requests.exceptions.ConnectionError:
            last_error = ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_URL}. Is Ollama running?")
            time.sleep(delay)
            delay *= _RETRY_BACKOFF
        except requests.exceptions.Timeout:
            last_error = TimeoutError(
                f"Ollama timed out after {REQUEST_TIMEOUT}s (model: {model})")
            time.sleep(delay)
            delay *= _RETRY_BACKOFF

    raise last_error if last_error else RuntimeError("LLM call failed")


# ==============================================================================
# JSON ARRAY PARSING (robust)
# ==============================================================================

def _parse_strategy_array(text: str) -> Optional[List[Dict]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # First [ ... last ]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    # Last resort: pull individual objects
    objs = []
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            objs.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    return objs if objs else None


def _as_text(v) -> str:
    """Normalize a field that may arrive as list/dict/str."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, indent=2)
    return str(v)


# ==============================================================================
# EXTRACTION WORKER (background thread + queue)
# ==============================================================================

_job_queue: "queue.Queue" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _worker_loop():
    while True:
        job = _job_queue.get()  # blocks
        job_id = job["job_id"]
        title = job["title"]
        model = job["model"]
        text = job["text"]  # held in memory only; discarded after this job
        try:
            _write_status(state="extracting", job_id=job_id, title=title,
                          model=model, started_at=datetime.now().isoformat(),
                          strategies_found=0, error="")

            content = text
            if len(content) > _MAX_CONTENT:
                content = content[:_MAX_CONTENT] + "\n\n[TRUNCATED]"

            user_prompt = EXTRACT_USER_TEMPLATE.format(title=title, content=content)
            response = ollama_chat(model, EXTRACT_SYSTEM_PROMPT, user_prompt,
                                   max_tokens=EXTRACT_MAX_TOKENS, temperature=0.2)

            strategies = _parse_strategy_array(response)
            if strategies is None:
                _write_status(state="error", job_id=job_id, title=title, model=model,
                              error="Could not parse JSON from model response")
                continue

            ensure_source_tables()
            now = datetime.now().isoformat()
            n = 0
            c = _conn()
            for s in strategies:
                name = str(s.get("name", "")).strip()
                if not name or name.upper() == "NONE":
                    continue
                row_id = uuid.uuid4().hex[:16]
                c.execute("""
                    INSERT INTO source_strategies (
                        id, job_id, source_title, model_used, extracted_at,
                        name, summary, hypothesis, entry_rules, exit_rules,
                        stop_loss, take_profit, indicators, parameters,
                        asset_class, timeframe, position_sizing,
                        confidence, source_quote, status
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending')
                """, (row_id, job_id, title, model, now,
                      name,
                      _as_text(s.get("summary")),
                      _as_text(s.get("hypothesis")),
                      _as_text(s.get("entry_rules")),
                      _as_text(s.get("exit_rules")),
                      _as_text(s.get("stop_loss")),
                      _as_text(s.get("take_profit")),
                      _as_text(s.get("indicators")),
                      _as_text(s.get("parameters")),
                      _as_text(s.get("asset_class")) or "forex",
                      _as_text(s.get("timeframe")) or "Not specified",
                      _as_text(s.get("position_sizing")),
                      str(s.get("confidence", "medium")).lower(),
                      _as_text(s.get("source_quote"))))
                n += 1
            c.commit()
            c.close()

            _write_status(state="complete", job_id=job_id, title=title,
                          model=model, strategies_found=n, error="")
        except Exception as e:
            _write_status(state="error", job_id=job_id, title=title,
                          model=model, error=f"{type(e).__name__}: {e}")
        finally:
            # Explicitly drop the source text -- nothing persisted
            job["text"] = None
            del text
            _job_queue.task_done()


def _ensure_worker():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True,
                                 name="source-extraction-worker")
            t.start()
            _worker_started = True


def submit_extraction(text: str, title: str = "", model: str = DEFAULT_MODEL) -> str:
    """Queue an extraction job. Returns job_id immediately."""
    if not text or not text.strip():
        raise ValueError("No source text provided")
    if model not in SOURCE_MODELS:
        model = DEFAULT_MODEL
    ensure_source_tables()
    _ensure_worker()
    job_id = uuid.uuid4().hex[:12]
    title = title.strip() or f"Pasted source {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    _job_queue.put({"job_id": job_id, "title": title, "model": model,
                    "text": text})
    _write_status(state=read_status().get("state", "idle"))  # refresh queued count
    return job_id


# ==============================================================================
# CODE VALIDATION (standalone copy of the discovery harness)
# ==============================================================================

def validate_strategy_code(code: str, name: str) -> Tuple[bool, str, int]:
    """
    1. Syntax  2. Import/exec  3. bt.Strategy subclass  4. Dummy backtest
    Includes a second correlated data feed for pairs/cointegration strategies.
    Returns (success, error_message, trade_count).
    """
    try:
        compile(code, f"<{name}>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}", 0

    namespace: Dict[str, Any] = {}
    exec_holder: Dict[str, Any] = {}

    def _exec():
        try:
            exec(code, namespace)
        except Exception as exc:
            exec_holder["error"] = exc

    t = threading.Thread(target=_exec, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        return False, "Import timeout: module-level code hung", 0
    if "error" in exec_holder:
        e = exec_holder["error"]
        if isinstance(e, ImportError):
            return False, f"ImportError: {e}", 0
        return False, f"Import/exec error: {type(e).__name__}: {e}", 0

    strategy_class = None
    try:
        import backtrader as bt
        for obj_name, obj in namespace.items():
            if (isinstance(obj, type) and issubclass(obj, bt.Strategy)
                    and obj is not bt.Strategy):
                strategy_class = obj
                break
    except ImportError:
        for obj_name, obj in namespace.items():
            if isinstance(obj, type) and obj_name != "Strategy":
                strategy_class = obj
                break
        if strategy_class:
            return True, "", 0
        return False, "No strategy class found", 0

    if strategy_class is None:
        return False, "No bt.Strategy subclass found", 0

    try:
        import backtrader as bt
        import numpy as np
        import pandas as pd

        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeanalyzer")

        bars = _VALIDATION_BARS
        rng = np.random.RandomState(42)
        close = 100 + np.cumsum(rng.randn(bars) * 0.5)
        high = close + np.abs(rng.randn(bars) * 0.3)
        low = close - np.abs(rng.randn(bars) * 0.3)
        open_ = close + rng.randn(bars) * 0.1
        volume = rng.randint(1000, 10000, bars).astype(float)
        dates = pd.date_range("2020-01-01", periods=bars, freq="D")
        df = pd.DataFrame({"open": open_, "high": high, "low": low,
                           "close": close, "volume": volume}, index=dates)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))  # type: ignore[call-arg]

        # Second correlated feed for pairs / cointegration strategies
        close2 = close * 0.98 + np.cumsum(rng.randn(bars) * 0.1)
        df2 = pd.DataFrame({
            "open": close2 + rng.randn(bars) * 0.1,
            "high": close2 + np.abs(rng.randn(bars) * 0.3),
            "low": close2 - np.abs(rng.randn(bars) * 0.3),
            "close": close2,
            "volume": rng.randint(1000, 10000, bars).astype(float),
        }, index=dates)
        cerebro.adddata(bt.feeds.PandasData(dataname=df2))  # type: ignore[call-arg]

        cerebro.broker.setcash(10000)

        results_holder: Dict[str, Any] = {}

        def _run():
            try:
                results_holder["results"] = cerebro.run()
            except Exception as exc:
                results_holder["error"] = exc

        rt = threading.Thread(target=_run, daemon=True)
        rt.start()
        rt.join(timeout=_VALIDATION_TIMEOUT)
        if rt.is_alive():
            return False, "Backtest timeout: strategy hung (possible infinite loop)", 0
        if "error" in results_holder:
            raise results_holder["error"]

        strat_instance = results_holder["results"][0]
        trade_count = 0
        if hasattr(strat_instance, "analyzers"):
            try:
                ta = strat_instance.analyzers.getbyname("tradeanalyzer")
                trade_count = ta.get_analysis().get("total", {}).get("total", 0)
            except Exception:
                pass
        return True, "", trade_count
    except Exception as e:
        return False, f"Backtest error: {type(e).__name__}: {e}", 0


# ==============================================================================
# APPROVE -> CODEGEN -> VALIDATE -> PROMOTE
# ==============================================================================

def _codegen_and_promote(row_id: str):
    row = get_source_strategy(row_id)
    if not row:
        return
    model = row.get("model_used") or DEFAULT_MODEL
    if model not in SOURCE_MODELS:
        model = DEFAULT_MODEL
    name = row.get("name") or "UnnamedStrategy"

    try:
        update_source_strategy(row_id, codegen_status="generating")

        user_prompt = CODEGEN_USER_TEMPLATE.format(
            name=name,
            summary=row.get("summary", ""),
            hypothesis=row.get("hypothesis", ""),
            entry_rules=row.get("entry_rules", "Not specified"),
            exit_rules=row.get("exit_rules", "Not specified"),
            stop_loss=row.get("stop_loss", "Not specified"),
            take_profit=row.get("take_profit", "Not specified"),
            indicators=row.get("indicators", "None specified"),
            parameters=row.get("parameters", "{}"),
            timeframe=row.get("timeframe", "Not specified"),
            position_sizing=row.get("position_sizing", "Not specified"),
        )

        # ast.parse truncation guard: retry on SyntaxError (rate-limit
        # truncation on free tier produces syntactically broken code)
        code = ""
        delay = _RETRY_DELAY
        last_syntax_error = ""
        for attempt in range(1, _MAX_RETRIES + 1):
            raw = ollama_chat(model, CODEGEN_SYSTEM_PROMPT, user_prompt,
                              max_tokens=CODEGEN_MAX_TOKENS, temperature=0.1)
            candidate = _clean_code_response(raw)
            if not candidate or "class " not in candidate:
                last_syntax_error = "No class found in generated output"
                time.sleep(delay)
                delay *= _RETRY_BACKOFF
                continue
            try:
                ast.parse(candidate)
                code = candidate
                break
            except SyntaxError as e:
                last_syntax_error = f"SyntaxError (truncation?): {e}"
                time.sleep(delay)
                delay *= _RETRY_BACKOFF

        if not code:
            update_source_strategy(row_id, codegen_status="failed",
                                   validation_error=last_syntax_error or "Codegen failed")
            return

        ok, err, trades = validate_strategy_code(code, name)

        # Promote into the main strategies table
        sid = _insert_into_strategies(row, code, ok, err, model)

        update_source_strategy(
            row_id,
            status="approved",
            codegen_status="done",
            code_valid=1 if ok else 0,
            validation_error=err or "",
            promoted_strategy_id=sid,
        )
    except Exception as e:
        update_source_strategy(row_id, codegen_status="failed",
                               validation_error=f"{type(e).__name__}: {e}")


def _clean_code_response(text: str) -> str:
    text = _strip_think(text).strip()
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```$", "", text)
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if (s.startswith("import ") or s.startswith("from ")
                or s.startswith("class ") or s.startswith("#")):
            start_idx = i
            break
    return "\n".join(lines[start_idx:]).rstrip()


def _ensure_strategies_table():
    """Make sure the main strategies table exists. Prefer StrategyInbox's
    schema (it also runs migrations); fall back to a minimal create."""
    try:
        from strategy_inbox import StrategyInbox
        StrategyInbox(DB_PATH)  # __init__ calls _ensure_tables()
        return
    except Exception:
        pass
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id TEXT PRIMARY KEY, strategy_name TEXT, summary TEXT,
            description TEXT, generated_code TEXT, origin_source TEXT,
            source_url TEXT, quality_score REAL, has_code INTEGER,
            code_validates INTEGER, validation_error TEXT, status TEXT,
            created_at TEXT, updated_at TEXT, asset_class TEXT,
            timeframe TEXT, hypothesis TEXT, tags TEXT, extraction_model TEXT
        )
    """)
    c.commit()
    c.close()


def _insert_into_strategies(row: Dict, code: str, code_valid: bool,
                            validation_error: str, model: str) -> str:
    """Insert the approved strategy into the main strategies table."""
    import hashlib
    _ensure_strategies_table()
    ts = datetime.now().isoformat()
    name = row.get("name", "UnnamedStrategy")
    sid = hashlib.sha256(f"{name}_{ts}".encode()).hexdigest()[:16]

    description = json.dumps({
        "summary": row.get("summary", ""),
        "hypothesis": row.get("hypothesis", ""),
        "entry_rules": row.get("entry_rules", ""),
        "exit_rules": row.get("exit_rules", ""),
        "stop_loss": row.get("stop_loss", ""),
        "take_profit": row.get("take_profit", ""),
        "indicators": row.get("indicators", ""),
        "parameters": row.get("parameters", ""),
        "position_sizing": row.get("position_sizing", ""),
        "source_title": row.get("source_title", ""),
        "source_quote": row.get("source_quote", ""),
    }, indent=2)

    c = _conn()
    c.execute("""
        INSERT INTO strategies (
            strategy_id, strategy_name, summary, description, generated_code,
            origin_source, source_url, quality_score, has_code,
            code_validates, validation_error, status, created_at, updated_at,
            asset_class, timeframe, hypothesis, tags, extraction_model
        ) VALUES (?,?,?,?,?, 'transcript', '', 75.0, 1, ?, ?, ?, ?, ?, ?, ?, ?, 'transcript', ?)
    """, (sid, name, (row.get("summary") or "")[:200], description, code,
          1 if code_valid else 0, validation_error or "",
          "extracted" if code_valid else "validation_failed",
          ts, ts,
          row.get("asset_class", "forex"),
          row.get("timeframe", "1hour"),
          row.get("hypothesis", ""),
          model))
    c.commit()
    c.close()
    return sid


def approve_strategy(row_id: str) -> None:
    """Approve a source strategy: fires codegen + validation in the background."""
    update_source_strategy(row_id, codegen_status="queued")
    t = threading.Thread(target=_codegen_and_promote, args=(row_id,),
                         daemon=True, name=f"codegen-{row_id}")
    t.start()


# ==============================================================================
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description="TradingLab Source Extraction")
    p.add_argument("--status", action="store_true", help="Show worker status")
    p.add_argument("--list", action="store_true", help="List extracted strategies")
    p.add_argument("--extract-file", type=str, help="Extract from a text file")
    p.add_argument("--title", type=str, default="", help="Source title")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"Model (options: {', '.join(SOURCE_MODELS)})")
    args = p.parse_args()

    if args.status:
        print(json.dumps(read_status(), indent=2))
    elif args.list:
        for r in list_source_strategies():
            badge = r["status"]
            if r["codegen_status"]:
                badge += f"/{r['codegen_status']}"
            print(f"  [{badge:>20}] {r['name']:<35} ({r['confidence']}, "
                  f"{r['model_used']}, {r['source_title'][:30]})")
    elif args.extract_file:
        text = Path(args.extract_file).read_text(encoding="utf-8", errors="replace")
        job_id = submit_extraction(text, title=args.title or Path(args.extract_file).stem,
                                   model=args.model)
        print(f"Job {job_id} queued. Waiting...")
        while True:
            time.sleep(2)
            s = read_status()
            print(f"  state={s.get('state')} found={s.get('strategies_found', 0)}")
            if s.get("state") in ("complete", "error") and s.get("job_id") == job_id:
                break
        if s.get("state") == "error":
            print(f"  ERROR: {s.get('error')}")
        else:
            print(f"  Done. {s.get('strategies_found', 0)} strategies extracted.")
    else:
        p.print_help()


if __name__ == "__main__":
    main()