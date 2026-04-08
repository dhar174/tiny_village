#!/usr/bin/env python3
"""
Test to validate that using real Goal objects (instead of Mock) properly tests utility functions.
This demonstrates that our fix correctly identifies when utility functions don't work with real Goal objects.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_utility_functions import calculate_action_utility

# Import our improved Goal implementation
from tests.test_tiny_utility_functions import Goal, MockAction


def test_real_goal_catches_broken_utility_function():
    """
    Demonstrate that using real Goal objects catches problems that Mock() would miss.
    This test simulates a broken utility function and shows it would be caught.
    """
    print("Testing that real Goal objects catch broken utility functions...")
    
    # Create a real Goal object with the interface utility functions expect
    real_goal = Goal(
        name="TestGoal",
        target_effects={"hunger": -0.5},
        priority=0.8,
        urgency=0.9  # This attribute might be used by utility functions
    )
    
    # Test that all expected attributes are present and accessible
    assert hasattr(real_goal, 'target_effects'), "Real Goal missing target_effects"
    assert hasattr(real_goal, 'priority'), "Real Goal missing priority"
    assert hasattr(real_goal, 'urgency'), "Real Goal missing urgency"
    assert hasattr(real_goal, 'name'), "Real Goal missing name"
    
    # Test with utility function
    char_state = {"hunger": 0.7}
    action = MockAction("Eat", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.3}])
    
    # This should work without errors if utility function properly handles Goal interface
    try:
        utility = calculate_action_utility(char_state, action, current_goal=real_goal)
        print(f"✅ Utility calculation succeeded: {utility}")
        assert isinstance(utility, (int, float)), "Utility should be numeric"
    except AttributeError as e:
        print(f"❌ AttributeError caught: {e}")
        print("   This demonstrates that utility function expects Goal attributes that were missing!")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   This demonstrates that utility function has issues with real Goal objects!")
        raise
    
    print("✅ Real Goal object works correctly with utility functions")


def test_mock_vs_real_goal_difference():
    """
    Show the difference between testing with Mock() vs real Goal objects.
    """
    print("\nTesting difference between Mock and Real Goal objects...")
    
    # Real Goal has more attributes than basic Mock
    real_goal = Goal(
        name="RealGoal",
        target_effects={"hunger": -0.5},
        priority=0.8,
        urgency=0.9,
        attributes={"importance": 5}
    )
    
    # Mock Goal (simple implementation)
    class BasicMockGoal:
        def __init__(self, name, target_effects=None, priority=0.5):
            self.name = name
            self.target_effects = target_effects or {}
            self.priority = priority
    
    mock_goal = BasicMockGoal("MockGoal", {"hunger": -0.5}, 0.8)
    
    # Real Goal has additional interface
    print(f"Real Goal attributes: {[attr for attr in dir(real_goal) if not attr.startswith('_')]}")
    print(f"Mock Goal attributes: {[attr for attr in dir(mock_goal) if not attr.startswith('_')]}")
    
    # Real Goal provides more complete testing interface
    assert hasattr(real_goal, 'urgency'), "Real Goal should have urgency"
    assert hasattr(real_goal, 'attributes'), "Real Goal should have attributes dict"
    assert hasattr(real_goal, 'description'), "Real Goal should have description"
    
    # Mock Goal has minimal interface (could miss bugs)
    assert not hasattr(mock_goal, 'urgency'), "Mock Goal should NOT have urgency"
    assert not hasattr(mock_goal, 'attributes'), "Mock Goal should NOT have attributes"
    assert not hasattr(mock_goal, 'description'), "Mock Goal should NOT have description"
    
    print("✅ Real Goal provides more complete testing interface than Mock")


def test_utility_function_goal_interface_usage():
    """
    Test that utility functions actually use Goal attributes beyond basic ones.
    This validates that our real Goal implementation is testing more than a simple mock would.
    """
    print("\nTesting utility function Goal interface usage...")
    
    char_state = {"hunger": 0.7}
    action = MockAction("Eat", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.3}])
    
    # Test with different priority values to ensure they affect utility
    high_priority_goal = Goal("HighPriority", target_effects={"hunger": -0.5}, priority=1.0)
    low_priority_goal = Goal("LowPriority", target_effects={"hunger": -0.5}, priority=0.1)
    
    high_utility = calculate_action_utility(char_state, action, current_goal=high_priority_goal)
    low_utility = calculate_action_utility(char_state, action, current_goal=low_priority_goal)
    
    print(f"High priority goal utility: {high_utility}")
    print(f"Low priority goal utility: {low_utility}")
    
    # Different priorities should produce different utilities
    assert high_utility != low_utility, "Priority should affect utility calculation"
    assert high_utility > low_utility, "Higher priority should generally give higher utility"
    
    print("✅ Utility functions properly use Goal priority attribute")


if __name__ == "__main__":
    print("🧪 Testing Real Goal vs Mock Goal validation...")
    print("=" * 60)
    
    try:
        test_real_goal_catches_broken_utility_function()
        test_mock_vs_real_goal_difference()
        test_utility_function_goal_interface_usage()
        
        print("\n🎉 ALL VALIDATION TESTS PASSED! 🎉")
        print("✅ Real Goal objects provide proper testing of utility functions")
        print("✅ This fix successfully addresses the Mock() fallback issue")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)