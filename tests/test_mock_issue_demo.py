#!/usr/bin/env python3
"""
Demonstration of the Mock() issue in utility calculation tests.
This script shows how using Mock() objects can lead to false test passes.
"""

import sys
import os
from unittest.mock import Mock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_utility_functions import calculate_action_utility, Goal
from actions import Action

def demonstrate_mock_issue():
    """Show how Mock() objects can give false positives in tests."""
    print("=== Demonstrating Mock() Issue ===\n")
    
    # Test setup - character state
    char_state = {"hunger": 0.9, "energy": 0.5}
    
    # Create a goal with priority
    goal = Goal(name="SatisfyHunger", target_effects={"hunger": -0.8}, priority=0.7)
    
    print("1. Testing with Mock() action (problematic approach):")
    print("-" * 50)
    
    # Create action using Mock() - this is the problematic approach
    mock_action = Mock()
    mock_action.name = "EatFood"
    # Notice: We're NOT setting cost or effects - Mock() will return Mock() for these
    
    try:
        utility_mock = calculate_action_utility(char_state, mock_action, current_goal=goal)
        print(f"✓ Mock action utility: {utility_mock}")
        print("✓ Test PASSED with Mock() (but this might be wrong!)")
        print(f"  - mock_action.cost: {mock_action.cost} (type: {type(mock_action.cost)})")
        print(f"  - mock_action.effects: {mock_action.effects} (type: {type(mock_action.effects)})")
    except Exception as e:
        print(f"✗ Mock action failed: {e}")
    
    print("\n2. Testing with real Action class (correct approach):")
    print("-" * 52)
    
    try:
        # Create a real Action with proper attributes
        real_action = Action(
            name="EatFood",
            preconditions={},
            effects=[{"attribute": "hunger", "change_value": -0.7}],
            cost=0.1
        )
        
        utility_real = calculate_action_utility(char_state, real_action, current_goal=goal)
        print(f"✓ Real action utility: {utility_real}")
        print("✓ Test PASSED with real Action class")
        print(f"  - real_action.cost: {real_action.cost} (type: {type(real_action.cost)})")
        print(f"  - real_action.effects: {real_action.effects} (type: {type(real_action.effects)})")
        
    except Exception as e:
        print(f"✗ Real action failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing with MockAction test class (good test approach):")
    print("-" * 58)
    
    # Import the proper MockAction class
    from tests.test_tiny_utility_functions import MockAction
    
    # Create action using MockAction test class
    test_action = MockAction(
        name="EatFood",
        cost=0.1,
        effects=[{"attribute": "hunger", "change_value": -0.7}]
    )
    
    try:
        utility_test = calculate_action_utility(char_state, test_action, current_goal=goal)
        print(f"✓ MockAction utility: {utility_test}")
        print("✓ Test PASSED with MockAction test class")
        print(f"  - test_action.cost: {test_action.cost} (type: {type(test_action.cost)})")
        print(f"  - test_action.effects: {test_action.effects} (type: {type(test_action.effects)})")
        
        # Verify the calculation is correct
        expected = 0.9 * 0.7 * 20 + 0.7 * 25.0 - 0.1 * 10  # need_fulfillment + goal_progress - cost
        print(f"  - Expected utility: {expected}")
        print(f"  - Matches expected: {abs(utility_test - expected) < 0.001}")
        
    except Exception as e:
        print(f"✗ MockAction failed: {e}")
    
    print("\n=== Summary ===")
    print("The issue with Mock() is that it doesn't enforce the expected interface.")
    print("Tests can pass even when the utility function expects specific attributes.")
    print("Use real classes or proper test doubles with the correct interface instead.")

if __name__ == "__main__":
    demonstrate_mock_issue()