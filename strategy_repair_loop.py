"""
strategy_repair_loop.py
=======================

Take a freshly generated strategy file, run the SAME checks the rest of the
project would run against it, and -- if it has genuine bugs -- hand the file
plus the exact errors to the local model, loop until it passes or a retry
budget is spent. Strategies that cannot be repaired are quarantined rather than
allowed into the pool.

WHY THIS EXISTS
    The generated strategy files (disc_*, variant_*) are the single largest
    source of type errors in the project. Hand-fixing them is throwaway work:
    they regenerate. The durable answer is a gate that makes a strategy earn
    its place by actually integrating -- generate, check, repair, recheck.

THE ONE THING THAT MATTERS MOST HERE
    Most of the type-checker output on these files is NOT bugs. Backtrader
    builds indicator classes and line objects through a metaclass at import
    time, so a static checker reports `bt.indicators.SMA(...)` as "Module is
    not callable" and `self.params.x` as "Cannot access attribute" -- for code
    that runs perfectly. A repair loop that treats those as bugs would burn the
    model rewriting correct code, and likely break it.

    So the loop separates REAL failures (syntax errors, import failures, a
    smoke backtest that raises, the known bug patterns) from FRAMEWORK NOISE,
    and only real failures drive a repair. If the only findings are framework
    noise, the strategy passes untouched.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:                                      # pragma: no cover
    requests = None                                      # type: ignore


# ==============================================================================
# What counts as a real problem vs. framework noise
# ==============================================================================

# Pyright messages that are backtrader's metaclass being invisible to the type
# checker, NOT bugs. Matched as substrings against the diagnostic message.
FRAMEWORK_NOISE_PATTERNS = (
    "Module is not callable",              # bt.indicators.SMA(...)
    'is not a known attribute of module "..indicators"',
    'is not a known attribute of module "backtrader',
    "Expected 0 positional arguments",     # metaclass __init__ signatures
    'No parameter named "dataname"',       # PandasData metaclass params
    'No parameter named "datetime"',
    'No parameter named "open"',
    'No parameter named "high"',
    'No parameter named "low"',
    'No parameter named "close"',
    'No parameter named "volume"',
    'No parameter named "openinterest"',
    'for class "tuple[tuple[str,',         # self.params.x on the params tuple
)

# Substrings that mark a diagnostic as a GENUINE bug even if it might otherwise
# look framework-ish. These win over the noise list.
REAL_BUG_OVERRIDES = (
    "OnBalanceVolume",                     # wrong indicator name; OBV is right
    "is not defined",                      # undefined name
    "is possibly unbound",
)


def is_framework_noise(message: str) -> bool:
    """
    True when a pyright message is backtrader metaclass invisibility rather than
    a real defect. Overrides win, so a genuinely wrong indicator name is never
    waved through as noise.
    """
    if any(o in message for o in REAL_BUG_OVERRIDES):
        return False
    return any(p in message for p in FRAMEWORK_NOISE_PATTERNS)


# Known-bug signatures checked by static inspection, independent of pyright.
def _bug_obv(code: str) -> Optional[str]:
    if "OnBalanceVolume" in code:
        return ("Uses bt.indicators.OnBalanceVolume, which does not exist. "
                "The correct name is bt.indicators.OBV.")
    return None


def _bug_unguarded_position_price(code: str) -> Optional[str]:
    if "self.position.price" in code:
        if ("if self.position" not in code
                and "if not self.position" not in code):
            return ("Reads self.position.price without first checking that a "
                    "position exists; guard with `if self.position:`.")
    return None


def _bug_missing_bar_guard(code: str) -> Optional[str]:
    idx = code.find("def next(self):")
    if idx == -1:
        return None
    body = code[idx:idx + 600]
    if "if len(self)" not in body and "if len(self.data)" not in body:
        return ("next() appears to lack a minimum-bars guard; indicators are "
                "not ready on the first bars. Add e.g. `if len(self) < "
                "self.params.slow: return`.")
    return None


STATIC_BUG_CHECKS = (
    ("obv_name", _bug_obv),
    ("unguarded_position_price", _bug_unguarded_position_price),
    ("missing_bar_guard", _bug_missing_bar_guard),
)


# ==============================================================================
# Diagnosis result
# ==============================================================================

@dataclass
class Diagnosis:
    """Everything found wrong with a strategy file on one pass."""
    syntax_error: Optional[str] = None
    import_error: Optional[str] = None
    smoke_error: Optional[str] = None
    static_bugs: List[str] = field(default_factory=list)
    real_type_errors: List[str] = field(default_factory=list)
    framework_noise_count: int = 0

    @property
    def is_clean(self) -> bool:
        """
        Clean means no REAL problem. Framework noise does not count -- a
        strategy that only trips the metaclass blind spot is fine as written.
        """
        return not (self.syntax_error or self.import_error or self.smoke_error
                    or self.static_bugs or self.real_type_errors)

    def as_prompt_block(self) -> str:
        """The findings, formatted for the repair model. Real problems only."""
        lines: List[str] = []
        if self.syntax_error:
            lines.append(f"SYNTAX ERROR:\n{self.syntax_error}")
        if self.import_error:
            lines.append(f"IMPORT ERROR:\n{self.import_error}")
        if self.smoke_error:
            lines.append(f"RUNTIME ERROR when the strategy was backtested on a "
                         f"tiny synthetic series:\n{self.smoke_error}")
        for b in self.static_bugs:
            lines.append(f"KNOWN BUG: {b}")
        for e in self.real_type_errors:
            lines.append(f"TYPE ERROR: {e}")
        return "\n\n".join(lines)

    def summary(self) -> str:
        parts = []
        if self.syntax_error:
            parts.append("syntax")
        if self.import_error:
            parts.append("import")
        if self.smoke_error:
            parts.append("smoke")
        if self.static_bugs:
            parts.append(f"{len(self.static_bugs)} known-bug")
        if self.real_type_errors:
            parts.append(f"{len(self.real_type_errors)} type")
        core = ", ".join(parts) if parts else "clean"
        return f"{core} (+{self.framework_noise_count} framework noise ignored)"


# ==============================================================================
# The checks
# ==============================================================================

class StrategyChecker:
    """Runs the real checks against a strategy file and returns a Diagnosis."""

    def __init__(self, project_root: Path, run_pyright: bool = True,
                 run_smoke: bool = True):
        self.project_root = project_root
        self.run_pyright = run_pyright
        self.run_smoke = run_smoke

    def diagnose(self, path: Path) -> Diagnosis:
        code = path.read_text(encoding="utf-8", errors="replace")
        d = Diagnosis()

        # 1. Syntax -- cheapest, and everything else depends on it.
        try:
            ast.parse(code)
        except SyntaxError as e:
            d.syntax_error = f"line {e.lineno}: {e.msg}"
            return d          # nothing else can run against unparseable code

        # 2. Static known-bug checks.
        for _name, check in STATIC_BUG_CHECKS:
            msg = check(code)
            if msg:
                d.static_bugs.append(msg)

        # 3. Import + smoke backtest in a subprocess (isolation: a strategy
        #    that hangs or crashes takes down only the probe).
        if self.run_smoke:
            err = self._smoke_test(path)
            if err is not None:
                if err.startswith("IMPORT::"):
                    d.import_error = err[len("IMPORT::"):]
                else:
                    d.smoke_error = err

        # 4. Pyright, with framework noise filtered out.
        if self.run_pyright:
            real, noise = self._pyright(path)
            d.real_type_errors = real
            d.framework_noise_count = noise

        return d

    def _smoke_test(self, path: Path) -> Optional[str]:
        """
        Import the file and, if it exposes a bt.Strategy subclass, run it on a
        tiny synthetic feed. Returns None on success, else an error string.
        """
        probe = _SMOKE_PROBE.replace("__TARGET__", json.dumps(str(path)))
        try:
            proc = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True, text=True, timeout=120,
                cwd=str(self.project_root), stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "smoke backtest timed out after 120s (possible infinite loop)"
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            return None
        m = re.search(r"^(IMPORT|RUN)::(.*)$", out, re.M | re.S)
        if m:
            kind, detail = m.group(1), m.group(2).strip()
            detail = detail[:1500]
            return ("IMPORT::" + detail) if kind == "IMPORT" else detail
        return out.strip()[-1500:] or "smoke test failed with no output"

    def _pyright(self, path: Path) -> Tuple[List[str], int]:
        """Return (real_errors, framework_noise_count)."""
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pyright", "--outputjson", str(path)],
                capture_output=True, text=True, timeout=180,
                cwd=str(self.project_root), stdin=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return [], 0
        try:
            report = json.loads(proc.stdout)
        except (ValueError, json.JSONDecodeError):
            return [], 0

        real: List[str] = []
        noise = 0
        for diag in report.get("generalDiagnostics", []):
            if diag.get("severity") != "error":
                continue
            msg = diag.get("message", "")
            if is_framework_noise(msg):
                noise += 1
                continue
            line = diag.get("range", {}).get("start", {}).get("line", 0) + 1
            real.append(f"line {line}: {msg}")
        return real, noise


# Runs in a subprocess. Imports the target module, finds a bt.Strategy
# subclass, runs a minimal cerebro on synthetic data. Prints IMPORT:: or RUN::
# on failure so the parent can classify.
_SMOKE_PROBE = r"""
import sys, traceback, importlib.util
path = __TARGET__
try:
    spec = importlib.util.spec_from_file_location("strat_probe", path)
    if spec is None or spec.loader is None:
        print("IMPORT::could not create import spec for the file"); sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: backtrader's metaclass looks the class's module up
    # in sys.modules, and a spec-loaded module is not there by default.
    sys.modules["strat_probe"] = mod
    spec.loader.exec_module(mod)
except Exception:
    print("IMPORT::" + traceback.format_exc()); sys.exit(1)

try:
    import backtrader as bt
    import pandas as pd, numpy as np
    strat = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj is not bt.Strategy:
            strat = obj
    if strat is None:
        sys.exit(0)          # importing cleanly is enough
    n = 200
    try:
        idx = pd.date_range("2020-01-01", periods=n, freq="h")   # pandas >=2.2
    except ValueError:
        idx = pd.date_range("2020-01-01", periods=n, freq="H")   # older pandas
    base = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 0.0005, n))
    df = pd.DataFrame({
        "open": base, "high": base + 0.0003, "low": base - 0.0003,
        "close": base, "volume": 1000}, index=idx)
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(strat)
    cerebro.broker.setcash(100000.0)
    cerebro.run()
    sys.exit(0)
except Exception:
    print("RUN::" + traceback.format_exc()); sys.exit(1)
"""


# ==============================================================================
# The local-model repair call
# ==============================================================================

REPAIR_SYSTEM_PROMPT = """\
You repair Backtrader trading-strategy Python files so they integrate with an \
existing quantitative-research codebase. You are given a strategy file that \
has real defects and the exact errors it produced. Return the COMPLETE \
corrected file and nothing else.

Hard rules:
- Return only Python code. No prose, no markdown fences, no explanation.
- Preserve the strategy's TRADING LOGIC and intent. Fix bugs; do not redesign.
- Keep the same class name and the same parameters.
- bt.indicators.OBV is the correct name (never OnBalanceVolume).
- Guard self.position.price behind `if self.position:`.
- Guard next() against insufficient bars before reading indicators.
- Do NOT try to satisfy a type checker about bt.indicators.* or self.params.*; \
those are framework attributes and are correct as written.
"""


class LocalModelRepairer:
    """Sends a broken strategy + its errors to the local model for repair."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                 timeout: int = 180, temperature: float = 0.1,
                 max_tokens: int = 8192):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        if requests is None:                             # pragma: no cover
            raise RuntimeError("The 'requests' package is required for repair.")
        self.session = requests.Session()

    def repair(self, code: str, diagnosis: Diagnosis) -> Optional[str]:
        """Return repaired code, or None if the call failed or gave nothing."""
        user = (
            "Here is the strategy file:\n\n"
            "```python\n" + code + "\n```\n\n"
            "These are the errors that must be fixed:\n\n"
            + diagnosis.as_prompt_block()
            + "\n\nReturn the complete corrected file."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        url = self.base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = self.session.post(url, json=payload, headers=headers,
                                     timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:                           # pragma: no cover
            print(f"   [repair] model call failed: {e}")
            return None

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        return _extract_code(text)


def _extract_code(text: str) -> Optional[str]:
    """
    Pull Python out of a model response. Handles bare code, ```python fences,
    and thinking-model <think> leakage.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if fence:
        return fence.group(1).strip()
    if "import" in text and ("class " in text or "def " in text):
        return text.strip()
    return None


# ==============================================================================
# The loop
# ==============================================================================

@dataclass
class RepairOutcome:
    path: Path
    success: bool
    attempts: int
    final_diagnosis: Diagnosis
    repaired: bool                 # whether the file was rewritten at all
    history: List[str] = field(default_factory=list)


class StrategyRepairLoop:
    """generate -> check -> (repair -> recheck)* until clean or budget spent."""

    def __init__(self, checker: StrategyChecker,
                 repairer: Optional[LocalModelRepairer],
                 max_attempts: int = 3,
                 quarantine_dir: Optional[Path] = None):
        self.checker = checker
        self.repairer = repairer
        self.max_attempts = max_attempts
        self.quarantine_dir = quarantine_dir

    def process(self, path: Path, write: bool = True) -> RepairOutcome:
        history: List[str] = []
        diagnosis = self.checker.diagnose(path)
        history.append(f"initial: {diagnosis.summary()}")

        if diagnosis.is_clean:
            return RepairOutcome(path, True, 0, diagnosis, False, history)

        if self.repairer is None:
            history.append("no repairer configured; leaving as-is")
            self._quarantine(path, diagnosis, write)
            return RepairOutcome(path, False, 0, diagnosis, False, history)

        current = path.read_text(encoding="utf-8", errors="replace")

        for attempt in range(1, self.max_attempts + 1):
            repaired_code = self.repairer.repair(current, diagnosis)
            if repaired_code is None:
                history.append(f"attempt {attempt}: model returned nothing")
                break

            # Never accept a repair that does not even parse.
            try:
                ast.parse(repaired_code)
            except SyntaxError as e:
                history.append(f"attempt {attempt}: repair had a syntax error "
                               f"(line {e.lineno}); discarded")
                continue

            # Check the repaired version against a temp file so a failed repair
            # never touches the real one.
            with tempfile.NamedTemporaryFile(
                    "w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(repaired_code)
                tmp_path = Path(tmp.name)
            try:
                new_diag = self.checker.diagnose(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            history.append(f"attempt {attempt}: {new_diag.summary()}")

            if new_diag.is_clean:
                if write:
                    path.write_text(_crlf(repaired_code), encoding="utf-8",
                                    newline="")
                return RepairOutcome(path, True, attempt, new_diag, True,
                                     history)

            # Stop if the repair made things strictly worse rather than chase a
            # regression; otherwise iterate with the new state.
            if self._worse(new_diag, diagnosis):
                history.append(f"attempt {attempt}: repair regressed; stopping")
                break
            current, diagnosis = repaired_code, new_diag

        # Exhausted or gave up. Quarantine; do not ship a half-repaired file.
        self._quarantine(path, diagnosis, write)
        return RepairOutcome(path, False, self.max_attempts, diagnosis,
                             False, history)

    @staticmethod
    def _worse(new: Diagnosis, old: Diagnosis) -> bool:
        def weight(d: Diagnosis) -> int:
            return ((1 if d.syntax_error else 0) * 100
                    + (1 if d.import_error else 0) * 50
                    + (1 if d.smoke_error else 0) * 20
                    + len(d.static_bugs) * 5
                    + len(d.real_type_errors))
        return weight(new) > weight(old)

    def _quarantine(self, path: Path, diagnosis: Diagnosis, write: bool) -> None:
        if not (self.quarantine_dir and write):
            return
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        note = self.quarantine_dir / (path.stem + ".why.txt")
        note.write_text(
            f"Could not integrate {path.name}.\n\n"
            f"{diagnosis.as_prompt_block()}\n",
            encoding="utf-8")


def _crlf(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\n", "\r\n")


# ==============================================================================
# CLI
# ==============================================================================

def build_repairer_from_config() -> Optional[LocalModelRepairer]:
    """
    Construct a repairer from the discovery config's code_generator endpoint,
    falling back to env vars. Returns None if nothing is reachable.
    """
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
    model = os.getenv("REPAIR_MODEL", "")
    try:
        sys.path.insert(0, os.getcwd())
        import discovery_config                          # type: ignore
        cfg = discovery_config.LLMConfig()
        base_url = cfg.code_generator.base_url
        if not model:
            model = cfg.code_generator.model
    except Exception:
        if not model:
            model = "qwen2.5-coder:7b"
    try:
        return LocalModelRepairer(base_url=base_url, model=model)
    except Exception:
        return None


def collect_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("*.py")))
        elif pp.is_file():
            files.append(pp)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check and repair generated strategy files so they "
                    "integrate with the project.")
    ap.add_argument("paths", nargs="+", help="strategy .py files or a dir")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--no-pyright", action="store_true",
                    help="skip the type check (syntax + smoke only)")
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip the smoke backtest")
    ap.add_argument("--check-only", action="store_true",
                    help="diagnose and report; never call the model or write")
    ap.add_argument("--quarantine-dir", default="quarantined_strategies")
    ap.add_argument("--project-root", default=os.getcwd())
    args = ap.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("No .py files found.")
        return 1

    root = Path(args.project_root)
    checker = StrategyChecker(root, run_pyright=not args.no_pyright,
                              run_smoke=not args.no_smoke)
    repairer = None if args.check_only else build_repairer_from_config()
    if not args.check_only and repairer is None:
        print("[WARN] No local model reachable; running in check-only mode.")
    loop = StrategyRepairLoop(
        checker, repairer, max_attempts=args.max_attempts,
        quarantine_dir=Path(args.quarantine_dir))

    n_clean = n_repaired = n_failed = 0
    for f in files:
        outcome = loop.process(f, write=not args.check_only)
        if outcome.success and not outcome.repaired:
            n_clean += 1
            tag = "[CLEAN]"
        elif outcome.success:
            n_repaired += 1
            tag = "[REPAIRED]"
        else:
            n_failed += 1
            tag = "[QUARANTINE]"
        print(f"  {tag:<12} {f.name}")
        for h in outcome.history:
            print(f"                 {h}")

    print("\n" + "=" * 60)
    print(f"  clean {n_clean}   repaired {n_repaired}   "
          f"could-not-fix {n_failed}   of {len(files)}")
    print("=" * 60)
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
