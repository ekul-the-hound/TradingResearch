# ==============================================================================
# failures_tracker.py
# ==============================================================================
# FAILURES.md Auto-Generation
#
# Tracks failed strategies, why they failed, and lessons learned.
# Prevents "mutation collapse" where the same bad ideas keep getting generated.
#
# Usage:
#     from failures_tracker import FailuresTracker
#     
#     tracker = FailuresTracker()
#     
#     # Log a failure
#     tracker.log_failure(
#         strategy_name="Variant_07_RSI_Filter",
#         failure_type="NEGATIVE_RETURNS",
#         metrics={'return': -15.2, 'sharpe': -0.5},
#         reason="RSI filter too aggressive, missed all good entries"
#     )
#     
#     # Generate FAILURES.md
#     tracker.generate_failures_md()
#     
#     # Get failure patterns for mutation agent
#     patterns = tracker.get_failure_patterns()
#
# ==============================================================================

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter

import config


@dataclass
class FailureRecord:
    """Record of a single strategy failure"""
    timestamp: str
    strategy_name: str
    variant_id: Optional[str]
    failure_type: str  # NEGATIVE_RETURNS, OVERFITTING, FRAGILE, HIGH_DRAWDOWN, etc.
    metrics: Dict[str, Any]
    reason: str
    lessons_learned: Optional[str]
    code_snippet: Optional[str]  # The problematic code pattern


class FailuresTracker:
    """
    Tracks strategy failures and generates FAILURES.md
    
    This helps prevent the mutation agent from repeatedly generating
    the same bad ideas (mutation collapse).
    """
    
    # Failure type definitions
    FAILURE_TYPES = {
        'NEGATIVE_RETURNS': 'Strategy lost money overall',
        'HIGH_DRAWDOWN': 'Maximum drawdown exceeded acceptable threshold',
        'OVERFITTING': 'Good in-sample, poor out-of-sample performance',
        'FRAGILE': 'Failed robustness tests (latency/slippage sensitive)',
        'NO_TRADES': 'Strategy generated zero trades',
        'TOO_FEW_TRADES': 'Insufficient trades for statistical significance',
        'POOR_SHARPE': 'Sharpe ratio below acceptable threshold',
        'CODE_ERROR': 'Strategy code had bugs/errors',
        'LOGIC_FLAW': 'Fundamental logic error in strategy design',
        'REGIME_DEPENDENT': 'Only works in specific market conditions',
        'COST_SENSITIVE': 'Profits eliminated by realistic transaction costs',
        'LATENCY_SENSITIVE': 'Requires unrealistic execution speed',
    }
    
    def __init__(self, failures_file: str = None):
        """
        Args:
            failures_file: Path to JSON file storing failures
        """
        if failures_file is None:
            failures_file = Path(config.BASE_DIR) / 'results' / 'failures.json'
        
        self.failures_file = Path(failures_file)
        self.failures_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.failures: List[FailureRecord] = []
        self._load_failures()
    
    def _load_failures(self):
        """Load existing failures from file"""
        if self.failures_file.exists():
            try:
                with open(self.failures_file, 'r') as f:
                    data = json.load(f)
                    self.failures = [
                        FailureRecord(**record) for record in data
                    ]
                print(f"📂 Loaded {len(self.failures)} failure records")
            except Exception as e:
                print(f"⚠️  Could not load failures file: {e}")
                self.failures = []
        else:
            self.failures = []
    
    def _save_failures(self):
        """Save failures to file"""
        try:
            with open(self.failures_file, 'w') as f:
                json.dump([asdict(f) for f in self.failures], f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save failures: {e}")
    
    # =========================================================================
    # LOG FAILURES
    # =========================================================================
    
    def log_failure(
        self,
        strategy_name: str,
        failure_type: str,
        metrics: Dict[str, Any],
        reason: str,
        variant_id: str = None,
        lessons_learned: str = None,
        code_snippet: str = None
    ) -> FailureRecord:
        """
        Log a strategy failure.
        
        Args:
            strategy_name: Name of the failed strategy
            failure_type: Type of failure (use FAILURE_TYPES keys)
            metrics: Performance metrics dict
            reason: Human-readable explanation of why it failed
            variant_id: Optional variant identifier
            lessons_learned: Optional lessons for future mutations
            code_snippet: Optional problematic code pattern
        
        Returns:
            The created FailureRecord
        """
        
        if failure_type not in self.FAILURE_TYPES:
            print(f"⚠️  Unknown failure type: {failure_type}")
            print(f"   Valid types: {list(self.FAILURE_TYPES.keys())}")
        
        record = FailureRecord(
            timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            variant_id=variant_id,
            failure_type=failure_type,
            metrics=metrics,
            reason=reason,
            lessons_learned=lessons_learned,
            code_snippet=code_snippet
        )
        
        self.failures.append(record)
        self._save_failures()
        
        print(f"📝 Logged failure: {strategy_name} ({failure_type})")
        
        return record
    
    def log_from_backtest_result(
        self,
        result: Dict,
        failure_type: str = None,
        reason: str = None,
        lessons_learned: str = None
    ) -> Optional[FailureRecord]:
        """
        Log a failure from a backtest result dict.
        
        Automatically detects failure type if not provided.
        """
        
        # Extract metrics
        metrics = {
            'total_return_pct': result.get('total_return_pct', 0),
            'sharpe_ratio': result.get('sharpe_ratio'),
            'max_drawdown_pct': result.get('max_drawdown_pct', 0),
            'total_trades': result.get('total_trades', 0),
            'win_rate': result.get('win_rate'),
        }
        
        # Auto-detect failure type if not provided
        if failure_type is None:
            failure_type = self._detect_failure_type(metrics)
        
        if failure_type is None:
            print(f"ℹ️  No failure detected for {result.get('strategy_name', 'Unknown')}")
            return None
        
        # Auto-generate reason if not provided
        if reason is None:
            reason = self._generate_reason(failure_type, metrics)
        
        return self.log_failure(
            strategy_name=result.get('strategy_name', 'Unknown'),
            variant_id=result.get('variant_id'),
            failure_type=failure_type,
            metrics=metrics,
            reason=reason,
            lessons_learned=lessons_learned
        )
    
    def _detect_failure_type(self, metrics: Dict) -> Optional[str]:
        """Auto-detect failure type from metrics"""
        
        ret = metrics.get('total_return_pct', 0)
        sharpe = metrics.get('sharpe_ratio')
        dd = metrics.get('max_drawdown_pct', 0)
        trades = metrics.get('total_trades', 0)
        
        # Check in order of severity
        if trades == 0:
            return 'NO_TRADES'
        
        if trades < 10:
            return 'TOO_FEW_TRADES'
        
        if dd > 30:
            return 'HIGH_DRAWDOWN'
        
        if ret < -10:
            return 'NEGATIVE_RETURNS'
        
        if sharpe is not None and sharpe < 0:
            return 'POOR_SHARPE'
        
        if ret < 0:
            return 'NEGATIVE_RETURNS'
        
        return None  # No failure detected
    
    def _generate_reason(self, failure_type: str, metrics: Dict) -> str:
        """Generate human-readable reason"""
        
        ret = metrics.get('total_return_pct', 0)
        sharpe = metrics.get('sharpe_ratio')
        dd = metrics.get('max_drawdown_pct', 0)
        trades = metrics.get('total_trades', 0)
        
        reasons = {
            'NEGATIVE_RETURNS': f"Strategy lost {abs(ret):.2f}% overall",
            'HIGH_DRAWDOWN': f"Maximum drawdown of {dd:.2f}% exceeds risk tolerance",
            'NO_TRADES': "Strategy generated zero trades - entry conditions too restrictive",
            'TOO_FEW_TRADES': f"Only {trades} trades - insufficient for statistical significance",
            'POOR_SHARPE': f"Sharpe ratio of {sharpe:.2f} indicates poor risk-adjusted returns",
            'OVERFITTING': "Performance degraded significantly out-of-sample",
            'FRAGILE': "Strategy failed robustness tests",
            'CODE_ERROR': "Strategy code contained bugs",
            'LOGIC_FLAW': "Fundamental logic error in strategy design",
            'REGIME_DEPENDENT': "Strategy only profitable in specific market conditions",
            'COST_SENSITIVE': "Profits eliminated when realistic costs applied",
            'LATENCY_SENSITIVE': "Strategy requires unrealistic execution speed",
        }
        
        return reasons.get(failure_type, f"Failed with {failure_type}")
    
    # =========================================================================
    # ANALYZE FAILURES
    # =========================================================================
    
    def get_failure_patterns(self) -> Dict:
        """
        Analyze failure patterns for the mutation agent.
        
        Returns dict with common failure patterns to avoid.
        """
        
        if not self.failures:
            return {'patterns': [], 'avoid': []}
        
        # Count failure types
        type_counts = Counter(f.failure_type for f in self.failures)
        
        # Extract code snippets that caused problems
        bad_patterns = []
        for f in self.failures:
            if f.code_snippet:
                bad_patterns.append(f.code_snippet)
        
        # Extract lessons learned
        lessons = []
        for f in self.failures:
            if f.lessons_learned:
                lessons.append(f.lessons_learned)
        
        # Find most common failure types
        common_failures = type_counts.most_common(5)
        
        return {
            'total_failures': len(self.failures),
            'common_failure_types': common_failures,
            'bad_code_patterns': bad_patterns,
            'lessons_learned': lessons,
            'avoid': self._generate_avoid_list()
        }
    
    def _generate_avoid_list(self) -> List[str]:
        """Generate list of things to avoid for mutation agent"""
        
        avoid = []
        type_counts = Counter(f.failure_type for f in self.failures)
        
        # Add advice based on common failures
        if type_counts.get('NO_TRADES', 0) > 2:
            avoid.append("Avoid overly restrictive entry conditions")
        
        if type_counts.get('LATENCY_SENSITIVE', 0) > 2:
            avoid.append("Avoid strategies requiring immediate execution")
        
        if type_counts.get('COST_SENSITIVE', 0) > 2:
            avoid.append("Avoid high-frequency strategies with many trades")
        
        if type_counts.get('HIGH_DRAWDOWN', 0) > 2:
            avoid.append("Always include stop losses or position limits")
        
        if type_counts.get('OVERFITTING', 0) > 2:
            avoid.append("Avoid too many parameters or overly specific conditions")
        
        if type_counts.get('REGIME_DEPENDENT', 0) > 2:
            avoid.append("Add regime filters or adaptive logic")
        
        return avoid
    
    def get_summary(self) -> Dict:
        """Get summary statistics of failures"""
        
        if not self.failures:
            return {'total': 0, 'by_type': {}, 'recent': []}
        
        type_counts = Counter(f.failure_type for f in self.failures)
        
        # Get 5 most recent
        recent = sorted(self.failures, key=lambda x: x.timestamp, reverse=True)[:5]
        
        return {
            'total': len(self.failures),
            'by_type': dict(type_counts),
            'recent': [
                {
                    'name': f.strategy_name,
                    'type': f.failure_type,
                    'reason': f.reason
                }
                for f in recent
            ]
        }
    
    # =========================================================================
    # GENERATE FAILURES.MD
    # =========================================================================
    
    def generate_failures_md(self, output_path: str = None) -> str:
        """
        Generate FAILURES.md file documenting all failures.
        
        Args:
            output_path: Path to save the file (default: FAILURES.md)
        
        Returns:
            The generated markdown content
        """
        
        if output_path is None:
            output_path = Path(config.BASE_DIR) / 'FAILURES.md'
        
        lines = []
        
        # Header
        lines.append("# Strategy Failures Log")
        lines.append("")
        lines.append(f"*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        lines.append("This document tracks failed strategies to prevent repeating mistakes.")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Failures:** {len(self.failures)}")
        
        if self.failures:
            type_counts = Counter(f.failure_type for f in self.failures)
            lines.append("")
            lines.append("### Failures by Type")
            lines.append("")
            lines.append("| Type | Count | Description |")
            lines.append("|------|-------|-------------|")
            for ftype, count in type_counts.most_common():
                desc = self.FAILURE_TYPES.get(ftype, "Unknown")
                lines.append(f"| {ftype} | {count} | {desc} |")
        
        lines.append("")
        
        # Lessons Learned
        lessons = [f.lessons_learned for f in self.failures if f.lessons_learned]
        if lessons:
            lines.append("## Lessons Learned")
            lines.append("")
            for lesson in set(lessons):
                lines.append(f"- {lesson}")
            lines.append("")
        
        # Things to Avoid
        avoid = self._generate_avoid_list()
        if avoid:
            lines.append("## Things to Avoid")
            lines.append("")
            lines.append("Based on failure patterns, the mutation agent should avoid:")
            lines.append("")
            for item in avoid:
                lines.append(f"- {item}")
            lines.append("")
        
        # Detailed Failure Log
        lines.append("## Detailed Failure Log")
        lines.append("")
        
        if self.failures:
            # Sort by date, most recent first
            sorted_failures = sorted(self.failures, key=lambda x: x.timestamp, reverse=True)
            
            for f in sorted_failures:
                lines.append(f"### {f.strategy_name}")
                lines.append("")
                lines.append(f"- **Date:** {f.timestamp[:10]}")
                lines.append(f"- **Type:** {f.failure_type}")
                lines.append(f"- **Reason:** {f.reason}")
                
                if f.metrics:
                    lines.append(f"- **Metrics:**")
                    for k, v in f.metrics.items():
                        if v is not None:
                            lines.append(f"  - {k}: {v}")
                
                if f.lessons_learned:
                    lines.append(f"- **Lesson:** {f.lessons_learned}")
                
                if f.code_snippet:
                    lines.append(f"- **Problematic Code:**")
                    lines.append("```python")
                    lines.append(f.code_snippet)
                    lines.append("```")
                
                lines.append("")
        else:
            lines.append("No failures logged yet.")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*This file is auto-generated by failures_tracker.py*")
        
        content = "\n".join(lines)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"📄 Generated {output_path}")
        
        return content
    
    # =========================================================================
    # INTEGRATION WITH MUTATION AGENT
    # =========================================================================
    
    def get_mutation_context(self) -> str:
        """
        Get context string for the mutation agent prompt.
        
        This tells Claude what patterns to avoid based on past failures.
        """
        
        if not self.failures:
            return ""
        
        patterns = self.get_failure_patterns()
        
        lines = []
        lines.append("## PAST FAILURES TO AVOID")
        lines.append("")
        lines.append(f"We have logged {len(self.failures)} failed strategies. Learn from these mistakes:")
        lines.append("")
        
        # Common failures
        if patterns['common_failure_types']:
            lines.append("### Most Common Failure Types:")
            for ftype, count in patterns['common_failure_types'][:3]:
                desc = self.FAILURE_TYPES.get(ftype, "Unknown")
                lines.append(f"- {ftype} ({count}x): {desc}")
        
        # Things to avoid
        if patterns['avoid']:
            lines.append("")
            lines.append("### DO NOT:")
            for item in patterns['avoid']:
                lines.append(f"- {item}")
        
        # Lessons learned
        if patterns['lessons_learned']:
            lines.append("")
            lines.append("### Lessons Learned:")
            for lesson in patterns['lessons_learned'][:5]:
                lines.append(f"- {lesson}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    
    def print_report(self):
        """Print failure report to console"""
        
        print(f"\n{'='*60}")
        print(f"FAILURES REPORT")
        print(f"{'='*60}")
        
        if not self.failures:
            print("\n✅ No failures logged yet!")
            print(f"{'='*60}")
            return
        
        summary = self.get_summary()
        
        print(f"\n📊 Total Failures: {summary['total']}")
        
        print(f"\n📈 By Type:")
        for ftype, count in sorted(summary['by_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {ftype}: {count}")
        
        print(f"\n📅 Recent Failures:")
        for f in summary['recent']:
            print(f"   • {f['name']}: {f['type']}")
            print(f"     {f['reason']}")
        
        # Things to avoid
        avoid = self._generate_avoid_list()
        if avoid:
            print(f"\n⚠️  Things to Avoid:")
            for item in avoid:
                print(f"   • {item}")
        
        print(f"\n{'='*60}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def log_failure(
    strategy_name: str,
    failure_type: str,
    metrics: Dict,
    reason: str,
    **kwargs
) -> FailureRecord:
    """Quick function to log a failure"""
    tracker = FailuresTracker()
    return tracker.log_failure(strategy_name, failure_type, metrics, reason, **kwargs)


def generate_failures_report():
    """Generate FAILURES.md and print report"""
    tracker = FailuresTracker()
    tracker.print_report()
    tracker.generate_failures_md()


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FAILURES TRACKER TEST")
    print("="*70)
    
    tracker = FailuresTracker()
    
    # Print current status
    tracker.print_report()
    
    # Example: Log a test failure
    print("\n📝 Logging a test failure...")
    tracker.log_failure(
        strategy_name="Test_Strategy_Example",
        failure_type="NEGATIVE_RETURNS",
        metrics={
            'total_return_pct': -5.2,
            'sharpe_ratio': -0.3,
            'max_drawdown_pct': 12.5,
            'total_trades': 45
        },
        reason="Strategy lost money due to poor entry timing",
        lessons_learned="Need to add trend filter before entry"
    )
    
    # Generate FAILURES.md
    print("\n📄 Generating FAILURES.md...")
    tracker.generate_failures_md()
    
    # Show mutation context
    print("\n📋 Mutation Agent Context:")
    print(tracker.get_mutation_context())
    
    print("\n" + "="*70)
    print("✅ Failures tracker working!")
    print("="*70)