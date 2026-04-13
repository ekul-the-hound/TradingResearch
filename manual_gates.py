# ==============================================================================
# manual_gates.py
# ==============================================================================
# Manual Validation Gates for Trading Research Pipeline
#
# Provides pause points where you approve before expensive operations run.
# Helps prevent:
# - Accidental API cost overruns
# - Running massive backtests unintentionally
# - Mutation agent loops without review
#
# Usage:
#     from manual_gates import ValidationGate, require_approval
#     
#     gate = ValidationGate()
#     
#     if gate.approve("Run 500 backtests", estimated_cost=0):
#         run_backtests()
#     
#     # Or as decorator:
#     @require_approval("Claude analysis")
#     def analyze_with_claude():
#         ...
#
# ==============================================================================

import functools
from typing import Optional, Callable
from datetime import datetime


class ValidationGate:
    """
    Manual validation gate system for expensive operations.
    
    Features:
    - Pause for user approval
    - Track approved operations
    - "Approve all" mode
    - Cost tracking
    - Session logging
    """
    
    def __init__(
        self,
        enabled: bool = True,
        log_file: Optional[str] = None,
        auto_approve_under: float = 0.0  # Auto-approve if cost under this threshold
    ):
        """
        Args:
            enabled: If False, all operations auto-approved
            log_file: Optional file to log all gate decisions
            auto_approve_under: Auto-approve operations with cost below this
        """
        self.enabled = enabled
        self.log_file = log_file
        self.auto_approve_under = auto_approve_under
        
        # Tracking
        self.session_start = datetime.now()
        self.total_approved_cost = 0.0
        self.total_blocked = 0
        self.total_approved = 0
        self.decisions = []
    
    def approve(
        self,
        description: str,
        estimated_cost: float = 0.0,
        details: Optional[str] = None,
        category: str = "general"
    ) -> bool:
        """
        Request approval for an operation.
        
        Args:
            description: Short description of what will happen
            estimated_cost: Estimated cost in dollars
            details: Additional details to show
            category: Category for grouping (e.g., "backtest", "api", "mutation")
        
        Returns:
            True if approved, False if rejected
        """
        
        # Check if gates are disabled
        if not self.enabled:
            self._log_decision(description, estimated_cost, "auto-approved (gates disabled)")
            self.total_approved += 1
            self.total_approved_cost += estimated_cost
            return True
        
        # Check auto-approve threshold
        if estimated_cost > 0 and estimated_cost < self.auto_approve_under:
            self._log_decision(description, estimated_cost, f"auto-approved (under ${self.auto_approve_under})")
            self.total_approved += 1
            self.total_approved_cost += estimated_cost
            return True
        
        # Display gate
        print("\n" + "="*70)
        print("🚦 MANUAL VALIDATION GATE")
        print("="*70)
        print(f"\n[LIST] Action: {description}")
        
        if estimated_cost > 0:
            print(f"[COST] Estimated cost: ${estimated_cost:.2f}")
            print(f"   Session total so far: ${self.total_approved_cost:.2f}")
        
        if details:
            print(f"\n[NOTE] Details:\n{details}")
        
        print("\n" + "-"*70)
        print("Options:")
        print("  [Y] Yes - Approve this action")
        print("  [N] No  - Skip this action")
        print("  [A] All - Approve all remaining (disable gates)")
        print("  [Q] Quit - Stop the entire process")
        print("-"*70)
        
        while True:
            try:
                response = input("\nYour choice (Y/N/A/Q): ").strip().upper()
            except EOFError:
                # Non-interactive mode
                print("[WARN]  Non-interactive mode - auto-approving")
                response = 'Y'
            
            if response == 'Y':
                self._log_decision(description, estimated_cost, "approved")
                self.total_approved += 1
                self.total_approved_cost += estimated_cost
                print("[OK] Approved")
                return True
            
            elif response == 'N':
                self._log_decision(description, estimated_cost, "rejected")
                self.total_blocked += 1
                print("[SKIP]  Skipped")
                return False
            
            elif response == 'A':
                self._log_decision(description, estimated_cost, "approved (+ disabled gates)")
                self.enabled = False
                self.total_approved += 1
                self.total_approved_cost += estimated_cost
                print("[OK] Approved - All gates now disabled for this session")
                return True
            
            elif response == 'Q':
                self._log_decision(description, estimated_cost, "quit")
                print("🛑 Process stopped by user")
                raise KeyboardInterrupt("User quit at validation gate")
            
            else:
                print("[FAIL] Invalid choice. Please enter Y, N, A, or Q")
    
    def approve_batch(
        self,
        description: str,
        n_items: int,
        cost_per_item: float = 0.0,
        show_items: Optional[list] = None
    ) -> bool:
        """
        Request approval for a batch operation.
        
        Args:
            description: What will be done
            n_items: Number of items to process
            cost_per_item: Cost per item
            show_items: Optional list of items to show (first 10)
        
        Returns:
            True if approved
        """
        
        total_cost = n_items * cost_per_item
        
        details = f"Items to process: {n_items}"
        if cost_per_item > 0:
            details += f"\nCost per item: ${cost_per_item:.4f}"
        
        if show_items:
            preview = show_items[:10]
            details += f"\n\nPreview (first {len(preview)}):\n"
            for item in preview:
                details += f"  - {item}\n"
            if len(show_items) > 10:
                details += f"  ... and {len(show_items) - 10} more"
        
        return self.approve(
            description=f"{description} ({n_items} items)",
            estimated_cost=total_cost,
            details=details,
            category="batch"
        )
    
    def require_positive_sharpe(
        self,
        result: dict,
        threshold: float = 0.0,
        action_if_pass: str = "Continue to next phase"
    ) -> bool:
        """
        Gate that requires minimum Sharpe ratio to proceed.
        
        Args:
            result: Backtest result dictionary with 'sharpe_ratio' key
            threshold: Minimum Sharpe to auto-pass
            action_if_pass: What happens if criteria met
        
        Returns:
            True if approved to continue
        """
        
        sharpe = result.get('sharpe_ratio', 0) or 0
        ret = result.get('total_return_pct', 0)
        
        if sharpe >= threshold:
            print(f"[OK] Gate passed: Sharpe {sharpe:.2f} >= {threshold}")
            return True
        
        print("\n" + "="*70)
        print("🚦 PERFORMANCE GATE")
        print("="*70)
        print(f"\nResults did not meet automatic criteria:")
        print(f"  Sharpe Ratio: {sharpe:.2f} (threshold: {threshold})")
        print(f"  Total Return: {ret:+.2f}%")
        print(f"\nIf passed, will: {action_if_pass}")
        
        return self.approve("Continue despite below-threshold performance?")
    
    def _log_decision(self, description: str, cost: float, decision: str):
        """Log a gate decision"""
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'cost': cost,
            'decision': decision
        }
        
        self.decisions.append(record)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{record['timestamp']} | {decision} | ${cost:.2f} | {description}\n")
            except Exception as e:
                print(f"[WARN]  Could not write to log: {e}")
    
    def get_session_summary(self) -> dict:
        """Get summary of this session's gate decisions"""
        
        return {
            'session_start': self.session_start.isoformat(),
            'session_duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60,
            'total_gates': self.total_approved + self.total_blocked,
            'approved': self.total_approved,
            'blocked': self.total_blocked,
            'total_approved_cost': self.total_approved_cost,
            'gates_enabled': self.enabled,
            'decisions': self.decisions
        }
    
    def print_session_summary(self):
        """Print formatted session summary"""
        
        summary = self.get_session_summary()
        
        print("\n" + "="*70)
        print("[STATS] VALIDATION GATE SESSION SUMMARY")
        print("="*70)
        print(f"  Duration:        {summary['session_duration_minutes']:.1f} minutes")
        print(f"  Total gates:     {summary['total_gates']}")
        print(f"  Approved:        {summary['approved']}")
        print(f"  Blocked:         {summary['blocked']}")
        print(f"  Total cost:      ${summary['total_approved_cost']:.2f}")
        print("="*70)


# ==============================================================================
# DECORATOR
# ==============================================================================

# Global gate instance for decorator use
_global_gate = ValidationGate(enabled=True)


def require_approval(
    description: str,
    estimated_cost: float = 0.0,
    category: str = "general"
):
    """
    Decorator to require manual approval before running a function.
    
    Usage:
        @require_approval("Run expensive Claude analysis", estimated_cost=0.25)
        def analyze_with_claude(results):
            ...
    """
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _global_gate.approve(
                description=description,
                estimated_cost=estimated_cost,
                category=category
            ):
                return func(*args, **kwargs)
            else:
                return None
        return wrapper
    return decorator


def set_global_gate(gate: ValidationGate):
    """Set the global gate instance for decorators"""
    global _global_gate
    _global_gate = gate


def disable_global_gates():
    """Disable all gates globally"""
    global _global_gate
    _global_gate.enabled = False


def enable_global_gates():
    """Enable all gates globally"""
    global _global_gate
    _global_gate.enabled = True


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_gate(description: str, cost: float = 0.0) -> bool:
    """Quick one-off gate check"""
    return _global_gate.approve(description, cost)


def cost_gate(operation: str, cost: float) -> bool:
    """Gate specifically for cost approval"""
    return _global_gate.approve(
        description=f"API call: {operation}",
        estimated_cost=cost,
        category="api"
    )


# ==============================================================================
# MAIN (Testing)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MANUAL GATES TEST")
    print("="*70)
    
    # Create gate with logging
    gate = ValidationGate(
        enabled=True,
        auto_approve_under=0.01  # Auto-approve costs under $0.01
    )
    
    # Test 1: Basic approval
    print("\nTest 1: Basic approval gate")
    if gate.approve("Test operation 1"):
        print("  -> Operation 1 would run")
    
    # Test 2: Cost gate
    print("\nTest 2: Cost gate ($0.25)")
    if gate.approve("Claude API call", estimated_cost=0.25):
        print("  -> Claude call would run")
    
    # Test 3: Auto-approve (under threshold)
    print("\nTest 3: Auto-approve (under $0.01)")
    if gate.approve("Tiny operation", estimated_cost=0.005):
        print("  -> Auto-approved (under threshold)")
    
    # Test 4: Batch approval
    print("\nTest 4: Batch operation")
    items = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    if gate.approve_batch(
        "Run backtests",
        n_items=len(items),
        cost_per_item=0,
        show_items=items
    ):
        print("  -> Batch would run")
    
    # Test 5: Decorator
    print("\nTest 5: Decorated function")
    
    @require_approval("Run test function", estimated_cost=0.10)
    def test_function():
        return "Function executed!"
    
    result = test_function()
    if result:
        print(f"  -> {result}")
    
    # Print session summary
    gate.print_session_summary()
    
    print("\n[OK] Manual gates working!")
    print("="*70)