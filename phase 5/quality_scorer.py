# ==============================================================================
# quality_scorer.py
# ==============================================================================
# Scores scraped documents and extracted strategies for quality.
#
# Quality scoring happens at two stages:
#   1. Document-level: Before extraction (should we even bother extracting?)
#   2. Strategy-level: After extraction (is this strategy worth backtesting?)
#
# Follows the same pattern as cost_adjusted_scoring.py in main TradingLab.
#
# Usage:
#     from quality_scorer import QualityScorer
#
#     scorer = QualityScorer()
#     doc_score = scorer.score_document(doc_dict)
#     strat_score = scorer.score_strategy(strategy_dict)
#
# ==============================================================================

import re
import logging
from typing import Dict, Any, Tuple

from discovery_config import DISCOVERY_CONFIG as cfg

logger = logging.getLogger(__name__)


# ==============================================================================
# PATTERNS FOR CONTENT DETECTION
# ==============================================================================

# Math / equation indicators
MATH_PATTERNS = [
    r'\$.*?\\.*?\$',                    # LaTeX inline math
    r'\\begin\{equation\}',             # LaTeX equation blocks
    r'\\frac\{',                         # Fractions
    r'\\sum',                            # Summation
    r'\\int',                            # Integrals
    r'E\[.*?\]',                         # Expected value notation
    r'σ|μ|α|β|λ|Σ',                     # Greek letters (common in quant)
    r'Sharpe\s*(?:ratio|=)',             # Sharpe ratio mentions
    r'(?:return|r)_[it]',               # Return subscripts (r_i, r_t)
    r'\bN\s*\(\s*\d',                   # Normal distribution N(0,1)
    r'(?:log|ln|exp)\s*\(',             # Math functions
    r'argmax|argmin',                    # Optimization notation
    r'(?:maximize|minimize)\s+',         # Optimization language
    r'R\^2|R-squared',                   # R-squared
    r'p-value|p\s*<\s*0\.\d',          # Statistical significance
    r'confidence\s+interval',            # CI
    r'standard\s+deviation',             # Std dev
    r'(?:covariance|correlation)\s+matrix', # Matrix references
]

# Backtest result indicators
BACKTEST_PATTERNS = [
    r'(?:total|cumulative|annualized)\s+return',
    r'(?:sharpe|sortino|calmar)\s+ratio\s*[=:]\s*[\d.-]',
    r'max(?:imum)?\s+drawdown\s*[=:]\s*[\d.-]',
    r'win\s+rate\s*[=:]\s*[\d.]',
    r'profit\s+factor\s*[=:]\s*[\d.]',
    r'number\s+of\s+trades\s*[=:]\s*\d',
    r'backtest(?:ed|ing)\s+(?:from|over|period)',
    r'(?:in-sample|out-of-sample)\s+(?:period|results|performance)',
    r'walk[- ]forward',
    r'(?:equity|performance)\s+curve',
    r'(?:risk|reward)\s+ratio',
]

# Code snippet indicators
CODE_PATTERNS = [
    r'```(?:python|py)',                 # Markdown code blocks
    r'import\s+(?:backtrader|pandas|numpy|ta|talib)',
    r'class\s+\w+\((?:bt\.)?Strategy\)',  # Backtrader strategy class
    r'def\s+(?:__init__|next|notify_order)',  # Backtrader methods
    r'bt\.indicators\.\w+',              # Backtrader indicators
    r'self\.buy\(\)|self\.sell\(\)',      # Backtrader orders
    r'(?:pandas|pd)\.(?:DataFrame|read_csv|rolling)',
    r'(?:numpy|np)\.(?:array|mean|std)',
]

# Explicit parameter indicators
PARAM_PATTERNS = [
    r'(?:fast|slow|signal)\s*(?:period|length|window)\s*[=:]\s*\d+',
    r'(?:stop\s*loss|take\s*profit)\s*[=:]\s*[\d.]+',
    r'(?:lookback|holding)\s*(?:period|days)\s*[=:]\s*\d+',
    r'threshold\s*[=:]\s*[\d.]+',
    r'(?:RSI|ATR|ADX|Bollinger)\s+(?:period|length)\s*[=:]\s*\d+',
    r'params\s*=\s*\(',                  # Backtrader params tuple
    r'(?:entry|exit)\s+(?:signal|condition|rule)',
]


class QualityScorer:
    """
    Scores documents and strategies for quality.

    Document scoring determines whether to proceed with LLM extraction.
    Strategy scoring determines whether to proceed with backtesting.
    """

    def __init__(self):
        self.cfg = cfg.quality

    # ==========================================================================
    # DOCUMENT-LEVEL SCORING
    # ==========================================================================

    def score_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a raw document before extraction.

        Args:
            doc: Dict with keys: content, source_type, source_bias, title

        Returns:
            Dict with: quality_score, has_math, has_backtest, has_code,
                       has_explicit_params, signals (detail dict)
        """
        content = doc.get("content", "")
        source_type = doc.get("source_type", "general")
        title = doc.get("title", "")
        full_text = f"{title}\n{content}"

        # 1. Base score from source type
        source_weight = self.cfg.source_weights.get(source_type, 0.5)

        # 2. Content signal detection
        has_math = self._has_pattern(full_text, MATH_PATTERNS)
        has_backtest = self._has_pattern(full_text, BACKTEST_PATTERNS)
        has_code = self._has_pattern(full_text, CODE_PATTERNS)
        has_params = self._has_pattern(full_text, PARAM_PATTERNS)

        # 3. Count how many patterns match (depth of coverage)
        math_count = self._count_patterns(full_text, MATH_PATTERNS)
        backtest_count = self._count_patterns(full_text, BACKTEST_PATTERNS)
        code_count = self._count_patterns(full_text, CODE_PATTERNS)
        param_count = self._count_patterns(full_text, PARAM_PATTERNS)

        # 4. Clarity / specificity score
        clarity = self._assess_clarity(full_text)

        # 5. Calculate composite score
        score = source_weight

        if has_math:
            score *= self.cfg.has_math_bonus
        if has_backtest:
            score *= self.cfg.has_backtest_bonus
        if has_code:
            score *= self.cfg.has_code_bonus
        if has_params:
            score *= self.cfg.has_params_bonus

        score *= clarity

        # Normalize to 0-1 range (max theoretical: 1.0 * 1.2 * 1.3 * 1.15 * 1.1 * 1.0 ≈ 1.98)
        score = min(score / 2.0, 1.0)

        result = {
            "quality_score": round(score, 4),
            "has_math": has_math,
            "has_backtest": has_backtest,
            "has_code": has_code,
            "has_explicit_params": has_params,
            "signals": {
                "source_weight": source_weight,
                "math_matches": math_count,
                "backtest_matches": backtest_count,
                "code_matches": code_count,
                "param_matches": param_count,
                "clarity": round(clarity, 3),
            },
        }

        logger.debug(
            f"Doc score: {score:.3f} "
            f"(src={source_weight}, math={math_count}, bt={backtest_count}, "
            f"code={code_count}, params={param_count}, clarity={clarity:.2f})"
        )
        return result

    def passes_extraction_threshold(self, doc: Dict[str, Any]) -> Tuple[bool, Dict]:
        """
        Check if a document is worth extracting.

        Returns:
            (passes: bool, score_details: dict)
        """
        score_result = self.score_document(doc)
        passes = score_result["quality_score"] >= self.cfg.min_quality_threshold
        return passes, score_result

    # ==========================================================================
    # STRATEGY-LEVEL SCORING
    # ==========================================================================

    def score_strategy(self, strategy: Dict[str, Any]) -> float:
        """
        Score an extracted strategy (after LLM extraction).

        This checks the generated code and summary, not just the source document.

        Args:
            strategy: Dict with keys: summary, generated_code, source_type,
                      has_math, has_backtest, has_code, has_explicit_params

        Returns:
            Quality score (0.0 - 1.0)
        """
        code = strategy.get("generated_code", "")
        summary = strategy.get("summary", "")
        source_type = strategy.get("source_type", "general")

        source_weight = self.cfg.source_weights.get(source_type, 0.5)
        score = source_weight

        # Carry forward document-level signals
        if strategy.get("has_math"):
            score *= self.cfg.has_math_bonus
        if strategy.get("has_backtest"):
            score *= self.cfg.has_backtest_bonus

        # Code quality checks
        code_quality = self._assess_code_quality(code)
        score *= code_quality

        # Summary quality
        summary_quality = self._assess_summary_quality(summary)
        score *= summary_quality

        # Normalize
        score = min(score / 2.0, 1.0)
        return round(score, 4)

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    @staticmethod
    def _has_pattern(text: str, patterns: list) -> bool:
        """Check if any pattern matches."""
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _count_patterns(text: str, patterns: list) -> int:
        """Count how many distinct patterns match."""
        count = 0
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                count += 1
        return count

    @staticmethod
    def _assess_clarity(text: str) -> float:
        """
        Assess how clear/specific a document is about a trading strategy.

        Returns 0.5 - 1.0 multiplier.
        """
        score = 0.5  # Base

        # Longer documents tend to be more detailed (up to a point)
        length = len(text)
        if length > 1000:
            score += 0.1
        if length > 3000:
            score += 0.1

        # Specific strategy keywords boost clarity
        strategy_keywords = [
            "entry", "exit", "signal", "indicator", "position",
            "long", "short", "buy", "sell", "stop loss", "take profit",
            "timeframe", "lookback", "period", "threshold",
            "momentum", "mean reversion", "breakout", "trend following",
            "crossover", "divergence", "volatility",
        ]
        keyword_hits = sum(1 for kw in strategy_keywords if kw.lower() in text.lower())

        # Scale: 0-3 hits = base, 4-8 = +0.15, 9+ = +0.3
        if keyword_hits >= 9:
            score += 0.3
        elif keyword_hits >= 4:
            score += 0.15

        return min(score, 1.0)

    @staticmethod
    def _assess_code_quality(code: str) -> float:
        """
        Assess quality of generated Backtrader code.

        Returns 0.5 - 1.3 multiplier.
        """
        if not code:
            return 0.5

        score = 0.7  # Base for having code at all

        # Must-have patterns
        if "class " in code and "bt.Strategy" in code:
            score += 0.1
        if "def __init__" in code:
            score += 0.05
        if "def next" in code:
            score += 0.05
        if "bt.indicators." in code:
            score += 0.05

        # Safety patterns (from Backtrader coding rules)
        if "if not self.position" in code or "if self.position" in code:
            score += 0.05
        if "if len(self)" in code:  # Minimum bar check
            score += 0.05
        if "self.entry_price" in code:  # Tracks entry price
            score += 0.05

        # Penalize suspicious patterns
        if "OnBalanceVolume" in code:  # Known bad indicator name
            score -= 0.1
        if code.count("self.close()") > 2:  # Overuse of close()
            score -= 0.05

        return max(0.3, min(score, 1.3))

    @staticmethod
    def _assess_summary_quality(summary: str) -> float:
        """
        Assess quality of strategy summary.

        Returns 0.7 - 1.1 multiplier.
        """
        if not summary:
            return 0.7

        score = 0.8
        words = summary.split()

        # Too short = vague
        if len(words) < 10:
            return 0.7

        # Good length with substance
        if len(words) >= 30:
            score += 0.1

        # Contains actionable detail
        action_words = ["when", "if", "above", "below", "crosses", "exceeds",
                        "entry", "exit", "buy", "sell", "signal"]
        if any(w.lower() in summary.lower() for w in action_words):
            score += 0.1

        return min(score, 1.1)


# ==============================================================================
# CLI: Test scorer on sample documents
# ==============================================================================

if __name__ == "__main__":
    scorer = QualityScorer()

    # Test documents with varying quality
    test_docs = [
        {
            "title": "RSI works great for day trading",
            "content": "I use RSI and it works. Buy when RSI is low, sell when high. Made good money.",
            "source_type": "reddit",
            "source_bias": "retail",
        },
        {
            "title": "A Momentum Strategy with Moving Average Crossover",
            "content": """
                This paper presents a systematic momentum trading strategy using a fast/slow
                moving average crossover. Entry signal: when the 10-period SMA crosses above
                the 50-period SMA. Exit: reverse crossover or stop loss at 2x ATR.

                Parameters: fast_period=10, slow_period=50, atr_stop_mult=2.0
                Lookback period: 252 trading days.

                Backtest results (2010-2023):
                Sharpe ratio: 1.2
                Max drawdown: -15.3%
                Win rate: 54%
                Total trades: 847
                Annualized return: 12.4%

                The strategy performs well in trending markets (ADX > 25) but suffers
                during ranging periods. E[r_t] = μ + β * r_{t-1} + ε_t
            """,
            "source_type": "arxiv",
            "source_bias": "academic",
        },
        {
            "title": "Backtrader RSI Mean Reversion Strategy",
            "content": """
                ```python
                import backtrader as bt

                class RSIMeanReversion(bt.Strategy):
                    params = (('rsi_period', 14), ('oversold', 30), ('overbought', 70))

                    def __init__(self):
                        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

                    def next(self):
                        if len(self) < self.params.rsi_period:
                            return
                        if not self.position:
                            if self.rsi[0] < self.params.oversold:
                                self.buy()
                        else:
                            if self.rsi[0] > self.params.overbought:
                                self.sell()
                ```
            """,
            "source_type": "github",
            "source_bias": "retail",
        },
    ]

    for doc in test_docs:
        print(f"\n{'='*60}")
        print(f"Title: {doc['title']}")
        print(f"Source: {doc['source_type']} ({doc['source_bias']})")
        passes, result = scorer.passes_extraction_threshold(doc)
        print(f"Quality Score: {result['quality_score']:.3f}")
        print(f"Passes Threshold ({scorer.cfg.min_quality_threshold}): {'✓ YES' if passes else '✗ NO'}")
        print(f"Signals: math={result['signals']['math_matches']}, "
              f"backtest={result['signals']['backtest_matches']}, "
              f"code={result['signals']['code_matches']}, "
              f"params={result['signals']['param_matches']}, "
              f"clarity={result['signals']['clarity']:.2f}")
