# ==============================================================================
# adversarial_reviewer.py
# ==============================================================================
# Adversarial Reviewer Agent
#
# Uses Claude API to critically analyze your strategy and try to find flaws.
# Acts as a "red team" that attempts to break your strategy's logic.
#
# Cost: ~$0.15-0.25 per review
#
# Usage:
#     from adversarial_reviewer import AdversarialReviewer
#     
#     reviewer = AdversarialReviewer()
#     
#     # Review a strategy's code
#     review = reviewer.review_strategy_code(strategy_code)
#     
#     # Review backtest results
#     review = reviewer.review_backtest_results(results)
#     
#     # Full adversarial analysis
#     review = reviewer.full_adversarial_review(
#         strategy_code=code,
#         backtest_results=results,
#         robustness_results=robustness
#     )
#
# ==============================================================================

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
import config


@dataclass
class AdversarialReview:
    """Results from adversarial review"""
    timestamp: str
    strategy_name: str
    overall_risk_score: int  # 1-10, 10 = highest risk
    critical_flaws: List[str]
    warnings: List[str]
    suggestions: List[str]
    overfitting_indicators: List[str]
    market_conditions_vulnerable: List[str]
    recommended_action: str  # "REJECT", "REVISE", "PROCEED_WITH_CAUTION", "APPROVE"
    full_analysis: str


class AdversarialReviewer:
    """
    Adversarial Reviewer Agent
    
    Uses Claude to critically analyze strategies and find flaws.
    Think of it as a "red team" for your trading strategies.
    """
    
    def __init__(self):
        if not config.CLAUDE_API_KEY:
            raise ValueError("Claude API key not configured")
        
        self.client = Anthropic(api_key=config.CLAUDE_API_KEY)
        self.model = config.CLAUDE_MODEL
        self.reviews = []
    
    # =========================================================================
    # REVIEW STRATEGY CODE
    # =========================================================================
    
    def review_strategy_code(
        self,
        strategy_code: str,
        strategy_name: str = "Unknown"
    ) -> AdversarialReview:
        """
        Critically review strategy code for flaws and risks.
        
        Args:
            strategy_code: The Python code of the strategy
            strategy_name: Name of the strategy
        
        Returns:
            AdversarialReview with findings
        """
        
        print(f"\n{'='*60}")
        print(f"🔴 ADVERSARIAL CODE REVIEW")
        print(f"{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"Sending to Claude for critical analysis...")
        
        prompt = f"""You are an adversarial reviewer for trading strategies. Your job is to find FLAWS, not praise the code.

## STRATEGY CODE TO REVIEW:
```python
{strategy_code}
```

## YOUR TASK:
Critically analyze this strategy and try to BREAK it. Look for:

1. **CRITICAL FLAWS** - Issues that will definitely cause problems:
   - Logic errors
   - Look-ahead bias
   - Survivorship bias potential
   - Division by zero risks
   - Array index errors
   - NoneType access errors

2. **OVERFITTING INDICATORS** - Signs the strategy is curve-fitted:
   - Magic numbers without justification
   - Too many parameters
   - Extremely specific conditions
   - Complex rules that seem arbitrary

3. **MARKET CONDITION VULNERABILITIES** - When will this strategy FAIL:
   - Bull markets only?
   - Low volatility only?
   - Trending markets only?
   - Will it blow up in crashes?

4. **EXECUTION RISKS** - Real-world problems:
   - Slippage sensitivity
   - Latency requirements
   - Liquidity assumptions
   - Gap risk

5. **MISSING SAFEGUARDS**:
   - No stop loss?
   - No position sizing?
   - No maximum drawdown check?

## RESPONSE FORMAT:
Respond with a JSON object (no markdown code blocks):
{{
    "overall_risk_score": <1-10, where 10 is extremely risky>,
    "critical_flaws": ["flaw1", "flaw2"],
    "warnings": ["warning1", "warning2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "overfitting_indicators": ["indicator1", "indicator2"],
    "market_conditions_vulnerable": ["condition1", "condition2"],
    "recommended_action": "<REJECT|REVISE|PROCEED_WITH_CAUTION|APPROVE>",
    "full_analysis": "<detailed paragraph explaining your concerns>"
}}

Be HARSH. Your job is to protect the user from losing money on a flawed strategy."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            review = AdversarialReview(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                overall_risk_score=data.get('overall_risk_score', 5),
                critical_flaws=data.get('critical_flaws', []),
                warnings=data.get('warnings', []),
                suggestions=data.get('suggestions', []),
                overfitting_indicators=data.get('overfitting_indicators', []),
                market_conditions_vulnerable=data.get('market_conditions_vulnerable', []),
                recommended_action=data.get('recommended_action', 'REVISE'),
                full_analysis=data.get('full_analysis', '')
            )
            
            self._print_review(review)
            self.reviews.append(review)
            
            return review
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse Claude response as JSON: {e}")
            print(f"Raw response: {response_text[:500]}...")
            
            # Return a default review
            return AdversarialReview(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                overall_risk_score=5,
                critical_flaws=["Could not parse review"],
                warnings=[],
                suggestions=[],
                overfitting_indicators=[],
                market_conditions_vulnerable=[],
                recommended_action="REVISE",
                full_analysis=response_text
            )
            
        except Exception as e:
            print(f"❌ API error: {e}")
            raise
    
    # =========================================================================
    # REVIEW BACKTEST RESULTS
    # =========================================================================
    
    def review_backtest_results(
        self,
        results: Dict,
        strategy_name: str = "Unknown"
    ) -> AdversarialReview:
        """
        Critically review backtest results for red flags.
        
        Args:
            results: Dictionary with backtest metrics
            strategy_name: Name of the strategy
        
        Returns:
            AdversarialReview with findings
        """
        
        print(f"\n{'='*60}")
        print(f"🔴 ADVERSARIAL RESULTS REVIEW")
        print(f"{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"Sending results to Claude for critical analysis...")
        
        prompt = f"""You are an adversarial reviewer for trading strategy backtest results. Your job is to find RED FLAGS.

## BACKTEST RESULTS TO REVIEW:
```json
{json.dumps(results, indent=2, default=str)}
```

## YOUR TASK:
Critically analyze these results and look for:

1. **TOO GOOD TO BE TRUE** indicators:
   - Sharpe > 3 is suspicious
   - Win rate > 70% needs scrutiny
   - Profit factor > 3 is unusual
   - Very low drawdown with high returns

2. **OVERFITTING SIGNS**:
   - Very few trades (< 30)
   - Inconsistent performance across timeframes
   - Results too specific to test period

3. **SURVIVORSHIP/LOOK-AHEAD BIAS POTENTIAL**:
   - Were there regime changes in the test period?
   - Is the test period representative?

4. **RISK CONCERNS**:
   - What's the worst-case scenario?
   - Is drawdown realistic for live trading?
   - Position sizing appropriate?

5. **MISSING INFORMATION**:
   - What data would you need to validate this?
   - What tests should be run next?

## RESPONSE FORMAT:
Respond with a JSON object (no markdown code blocks):
{{
    "overall_risk_score": <1-10, where 10 is extremely risky>,
    "critical_flaws": ["flaw1", "flaw2"],
    "warnings": ["warning1", "warning2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "overfitting_indicators": ["indicator1", "indicator2"],
    "market_conditions_vulnerable": ["condition1", "condition2"],
    "recommended_action": "<REJECT|REVISE|PROCEED_WITH_CAUTION|APPROVE>",
    "full_analysis": "<detailed paragraph explaining your concerns>"
}}

Be SKEPTICAL. Assume results are too good until proven otherwise."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse JSON
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            review = AdversarialReview(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                overall_risk_score=data.get('overall_risk_score', 5),
                critical_flaws=data.get('critical_flaws', []),
                warnings=data.get('warnings', []),
                suggestions=data.get('suggestions', []),
                overfitting_indicators=data.get('overfitting_indicators', []),
                market_conditions_vulnerable=data.get('market_conditions_vulnerable', []),
                recommended_action=data.get('recommended_action', 'REVISE'),
                full_analysis=data.get('full_analysis', '')
            )
            
            self._print_review(review)
            self.reviews.append(review)
            
            return review
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse response: {e}")
            return AdversarialReview(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                overall_risk_score=5,
                critical_flaws=["Could not parse review"],
                warnings=[],
                suggestions=[],
                overfitting_indicators=[],
                market_conditions_vulnerable=[],
                recommended_action="REVISE",
                full_analysis=response_text
            )
            
        except Exception as e:
            print(f"❌ API error: {e}")
            raise
    
    # =========================================================================
    # FULL ADVERSARIAL REVIEW
    # =========================================================================
    
    def full_adversarial_review(
        self,
        strategy_code: str,
        backtest_results: Dict,
        robustness_results: Dict = None,
        validation_results: Dict = None,
        strategy_name: str = "Unknown"
    ) -> AdversarialReview:
        """
        Comprehensive adversarial review combining all available data.
        
        Args:
            strategy_code: Python code of the strategy
            backtest_results: Backtest performance metrics
            robustness_results: Results from robustness tests (optional)
            validation_results: Results from validation framework (optional)
            strategy_name: Name of the strategy
        
        Returns:
            AdversarialReview with comprehensive findings
        """
        
        print(f"\n{'='*60}")
        print(f"🔴 FULL ADVERSARIAL REVIEW")
        print(f"{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"Conducting comprehensive adversarial analysis...")
        
        # Build comprehensive context
        context_parts = []
        
        context_parts.append(f"""## STRATEGY CODE:
```python
{strategy_code}
```""")
        
        context_parts.append(f"""## BACKTEST RESULTS:
```json
{json.dumps(backtest_results, indent=2, default=str)}
```""")
        
        if robustness_results:
            context_parts.append(f"""## ROBUSTNESS TEST RESULTS:
```json
{json.dumps(robustness_results, indent=2, default=str)}
```""")
        
        if validation_results:
            context_parts.append(f"""## VALIDATION RESULTS (Bootstrap/Monte Carlo/Walk-Forward):
```json
{json.dumps(validation_results, indent=2, default=str)}
```""")
        
        full_context = "\n\n".join(context_parts)
        
        prompt = f"""You are a senior quantitative analyst conducting an adversarial review. Your job is to PROTECT the user from deploying a flawed strategy.

{full_context}

## YOUR MISSION:
You must try to BREAK this strategy. Find every possible flaw, weakness, and risk. Be thorough and harsh.

Consider:
1. **Code Quality** - Logic errors, edge cases, bugs
2. **Backtest Validity** - Overfitting, look-ahead bias, data snooping
3. **Robustness** - How does it handle adverse conditions?
4. **Statistical Validity** - Are the results statistically significant?
5. **Market Reality** - Will this actually work in live trading?
6. **Risk Management** - What could go catastrophically wrong?
7. **Regime Dependence** - Does it only work in certain markets?

## RESPONSE FORMAT:
Respond with a JSON object (no markdown code blocks):
{{
    "overall_risk_score": <1-10, where 10 is extremely risky>,
    "critical_flaws": ["flaw1", "flaw2"],
    "warnings": ["warning1", "warning2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "overfitting_indicators": ["indicator1", "indicator2"],
    "market_conditions_vulnerable": ["condition1", "condition2"],
    "recommended_action": "<REJECT|REVISE|PROCEED_WITH_CAUTION|APPROVE>",
    "full_analysis": "<detailed 2-3 paragraph analysis explaining all your concerns and the reasoning behind your recommendation>"
}}

Remember: It's better to reject a good strategy than approve a bad one. Be CONSERVATIVE."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse JSON
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            review = AdversarialReview(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                overall_risk_score=data.get('overall_risk_score', 5),
                critical_flaws=data.get('critical_flaws', []),
                warnings=data.get('warnings', []),
                suggestions=data.get('suggestions', []),
                overfitting_indicators=data.get('overfitting_indicators', []),
                market_conditions_vulnerable=data.get('market_conditions_vulnerable', []),
                recommended_action=data.get('recommended_action', 'REVISE'),
                full_analysis=data.get('full_analysis', '')
            )
            
            self._print_review(review)
            self.reviews.append(review)
            
            # Calculate approximate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000) * 3 + (output_tokens / 1_000_000) * 15
            print(f"\n💰 API Cost: ~${cost:.4f}")
            
            return review
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse response: {e}")
            return AdversarialReview(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                overall_risk_score=5,
                critical_flaws=["Could not parse review"],
                warnings=[],
                suggestions=[],
                overfitting_indicators=[],
                market_conditions_vulnerable=[],
                recommended_action="REVISE",
                full_analysis=response_text
            )
            
        except Exception as e:
            print(f"❌ API error: {e}")
            raise
    
    # =========================================================================
    # PRINT REVIEW
    # =========================================================================
    
    def _print_review(self, review: AdversarialReview):
        """Print formatted review"""
        
        print(f"\n{'─'*60}")
        print(f"ADVERSARIAL REVIEW: {review.strategy_name}")
        print(f"{'─'*60}")
        
        # Risk score with visual
        risk_bar = "🔴" * review.overall_risk_score + "⚪" * (10 - review.overall_risk_score)
        print(f"\n🎯 RISK SCORE: {review.overall_risk_score}/10")
        print(f"   {risk_bar}")
        
        # Recommendation
        rec_emoji = {
            "REJECT": "❌",
            "REVISE": "⚠️",
            "PROCEED_WITH_CAUTION": "🟡",
            "APPROVE": "✅"
        }
        print(f"\n📋 RECOMMENDATION: {rec_emoji.get(review.recommended_action, '❓')} {review.recommended_action}")
        
        # Critical flaws
        if review.critical_flaws:
            print(f"\n🚨 CRITICAL FLAWS:")
            for flaw in review.critical_flaws:
                print(f"   • {flaw}")
        
        # Warnings
        if review.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in review.warnings:
                print(f"   • {warning}")
        
        # Overfitting indicators
        if review.overfitting_indicators:
            print(f"\n📈 OVERFITTING INDICATORS:")
            for indicator in review.overfitting_indicators:
                print(f"   • {indicator}")
        
        # Market vulnerabilities
        if review.market_conditions_vulnerable:
            print(f"\n🌊 VULNERABLE IN:")
            for condition in review.market_conditions_vulnerable:
                print(f"   • {condition}")
        
        # Suggestions
        if review.suggestions:
            print(f"\n💡 SUGGESTIONS:")
            for suggestion in review.suggestions:
                print(f"   • {suggestion}")
        
        # Full analysis
        print(f"\n📝 FULL ANALYSIS:")
        print(f"   {review.full_analysis}")
        
        print(f"\n{'='*60}")
    
    # =========================================================================
    # SAVE REVIEWS
    # =========================================================================
    
    def save_reviews(self, filepath: str = None):
        """Save all reviews to a JSON file"""
        
        if filepath is None:
            filepath = f"adversarial_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        reviews_data = []
        for review in self.reviews:
            reviews_data.append({
                'timestamp': review.timestamp,
                'strategy_name': review.strategy_name,
                'overall_risk_score': review.overall_risk_score,
                'critical_flaws': review.critical_flaws,
                'warnings': review.warnings,
                'suggestions': review.suggestions,
                'overfitting_indicators': review.overfitting_indicators,
                'market_conditions_vulnerable': review.market_conditions_vulnerable,
                'recommended_action': review.recommended_action,
                'full_analysis': review.full_analysis
            })
        
        with open(filepath, 'w') as f:
            json.dump(reviews_data, f, indent=2)
        
        print(f"💾 Saved {len(self.reviews)} reviews to {filepath}")


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def adversarial_review_strategy(
    strategy_path: str,
    backtest_results: Dict = None
) -> AdversarialReview:
    """
    Quick adversarial review of a strategy file.
    
    Args:
        strategy_path: Path to the strategy .py file
        backtest_results: Optional backtest results dict
    
    Returns:
        AdversarialReview
    """
    
    # Load strategy code
    with open(strategy_path, 'r') as f:
        code = f.read()
    
    strategy_name = Path(strategy_path).stem
    
    reviewer = AdversarialReviewer()
    
    if backtest_results:
        return reviewer.full_adversarial_review(
            strategy_code=code,
            backtest_results=backtest_results,
            strategy_name=strategy_name
        )
    else:
        return reviewer.review_strategy_code(code, strategy_name)


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ADVERSARIAL REVIEWER TEST")
    print("="*70)
    
    # Check API key
    if not config.CLAUDE_API_KEY:
        print("❌ No Claude API key configured")
        print("   Add your key to BacktestingAgent_API_KEY.txt")
        exit()
    
    print("✅ Claude API key found")
    
    # Try to load a strategy
    try:
        strategy_path = Path(__file__).parent / 'strategies' / 'simple_strategy.py'
        if strategy_path.exists():
            with open(strategy_path, 'r') as f:
                code = f.read()
            
            print(f"\n📄 Loaded strategy: {strategy_path.name}")
            
            # Ask for confirmation (costs money)
            print(f"\n⚠️  This will make a Claude API call (~$0.15-0.25)")
            confirm = input("Proceed with adversarial review? (Y/N): ").strip().upper()
            
            if confirm == 'Y':
                reviewer = AdversarialReviewer()
                review = reviewer.review_strategy_code(code, "SimpleMovingAverageCrossover")
                print("\n✅ Adversarial review complete!")
            else:
                print("Cancelled.")
        else:
            print(f"⚠️  Strategy file not found: {strategy_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("="*70)