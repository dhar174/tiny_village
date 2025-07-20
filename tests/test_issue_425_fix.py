#!/usr/bin/env python3
"""
Demonstrates the fix for issue #425: Replace Mock() fallback with real Goal object creation.

This file specifically addresses the problematic pattern found in PR #417's test_fix_validation.py:

PROBLEMATIC CODE:
    if hasattr(tiny_utility_functions, 'Goal'):
        goal = tiny_utility_functions.Goal(...)
    else:
        # Fallback to mock if Goal not available
        goal = Mock()

FIXED CODE:
    # Always use real Goal object - no Mock() fallback needed
    goal = Goal(name="test_goal", target_effects={...}, priority=0.7)

This demonstrates that the Mock() fallback is unnecessary because the Goal class
is always available in tiny_utility_functions.py.
"""

import unittest
import sys
import os
from unittest.mock import Mock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the utility functions and Goal class
from tiny_utility_functions import calculate_action_utility, Goal


class MockAction:
    """Simple mock action for testing."""
    def __init__(self, name, cost=0.0, effects=None):
        self.name = name
        self.cost = cost
        self.effects = effects or []


class TestIssue425Fix(unittest.TestCase):
    """Test that demonstrates the fix for issue #425."""

    def test_problematic_pattern_from_issue(self):
        """
        This recreates the problematic pattern from the issue to show why it's bad.
        
        The original code from PR #417 test_fix_validation.py had:
            if hasattr(tiny_utility_functions, 'Goal'):
                goal = tiny_utility_functions.Goal(...)
            else:
                # Fallback to mock if Goal not available
                goal = Mock()
        """
        
        # Simulate the problematic pattern by forcing the fallback
        # (We'll pretend Goal doesn't exist to test the Mock fallback)
        use_mock_fallback = True  # This simulates when Goal is "not available"
        
        if not use_mock_fallback and hasattr(Goal, '__real_goal_class__'):
            # This branch would create a real Goal
            goal = Goal(
                name="test_goal",
                target_effects={"hunger": 0.2, "energy": 0.8},
                priority=0.7
            )
        else:
            # This is the PROBLEMATIC fallback pattern
            goal = Mock()
            goal.target_effects = {"hunger": 0.2, "energy": 0.8}
            goal.priority = 0.7
            goal.name = "test_goal"
        
        # Verify we got a Mock (which is the problem)
        self.assertIsInstance(goal, Mock)
        
        # Create test action and state
        action = MockAction(
            name="test_action",
            cost=1.0,
            effects=[
                {"attribute": "hunger", "change_value": -0.3},
                {"attribute": "energy", "change_value": -0.1}
            ]
        )
        
        char_state = {"hunger": 0.5, "energy": 0.7}
        
        # Test utility calculation with Mock goal
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        
        print(f"❌ Problematic Mock fallback utility: {utility}")
        
        # The problem: This might work, but it's not testing real Goal behavior
        self.assertIsInstance(utility, (int, float))

    def test_correct_fix_no_fallback_needed(self):
        """
        CORRECT FIX: Shows that Mock() fallback is unnecessary.
        
        The Goal class is always available in tiny_utility_functions.py,
        so we can always create real Goal objects.
        """
        
        # FIXED CODE: Always create real Goal object
        goal = Goal(
            name="test_goal",
            target_effects={"hunger": 0.2, "energy": 0.8},
            priority=0.7
        )
        
        # Verify we got a real Goal object
        self.assertIsInstance(goal, Goal)
        self.assertNotIsInstance(goal, Mock)
        
        # Verify Goal has the expected attributes
        self.assertEqual(goal.name, "test_goal")
        self.assertEqual(goal.target_effects, {"hunger": 0.2, "energy": 0.8})
        self.assertEqual(goal.priority, 0.7)
        
        # Create test action and state
        action = MockAction(
            name="test_action", 
            cost=1.0,
            effects=[
                {"attribute": "hunger", "change_value": -0.3},
                {"attribute": "energy", "change_value": -0.1}
            ]
        )
        
        char_state = {"hunger": 0.5, "energy": 0.7}
        
        # Test utility calculation with real Goal
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        
        print(f"✅ Correct real Goal utility: {utility}")
        
        # This tests actual Goal integration, not Mock behavior
        self.assertIsInstance(utility, (int, float))

    def test_direct_comparison_mock_vs_real(self):
        """
        Direct comparison showing the difference between Mock and real Goal.
        """
        
        # Create Mock goal (problematic approach)
        mock_goal = Mock()
        mock_goal.target_effects = {"hunger": -0.5}
        mock_goal.priority = 0.8
        mock_goal.name = "mock_goal"
        
        # Create real Goal (correct approach)
        real_goal = Goal(
            name="real_goal",
            target_effects={"hunger": -0.5},
            priority=0.8
        )
        
        # Test action
        action = MockAction(
            name="eat_food",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.4}]
        )
        
        char_state = {"hunger": 0.9}
        
        # Calculate utility with both
        mock_utility = calculate_action_utility(char_state, action, current_goal=mock_goal)
        real_utility = calculate_action_utility(char_state, action, current_goal=real_goal)
        
        print(f"Mock goal utility: {mock_utility}")
        print(f"Real goal utility: {real_utility}")
        
        # Both should work, but only the real Goal tests actual integration
        self.assertIsInstance(mock_utility, (int, float))
        self.assertIsInstance(real_utility, (int, float))
        
        # The utilities should be the same in this simple case
        self.assertAlmostEqual(mock_utility, real_utility, places=5)
        
        # But the key difference is that real_goal tests actual Goal class behavior
        self.assertIsInstance(real_goal, Goal)
        self.assertIsInstance(mock_goal, Mock)
        
        # Real Goal has actual implementation, Mock just returns whatever we set
        self.assertTrue(hasattr(real_goal, 'target_effects'))
        self.assertTrue(hasattr(real_goal, 'priority'))
        
        # Mock might have attributes we didn't set, but they're fake
        self.assertEqual(real_goal.priority, 0.8)
        self.assertEqual(mock_goal.priority, 0.8)  # Same value, but Mock is fake

    def test_recommended_pattern_for_tests(self):
        """
        Demonstrates the recommended pattern that should replace the problematic code.
        """
        
        def create_test_utility_goal(name, target_effects, priority=0.5):
            """
            Recommended pattern: Helper function to create test Goals.
            
            This ensures consistent Goal creation and eliminates need for Mock() fallbacks.
            """
            return Goal(
                name=name,
                target_effects=target_effects,
                priority=priority
            )
        
        # Use the helper to create various test goals
        hunger_goal = create_test_utility_goal(
            name="satisfy_hunger",
            target_effects={"hunger": -0.8},
            priority=0.9
        )
        
        energy_goal = create_test_utility_goal(
            name="restore_energy", 
            target_effects={"energy": 0.6},
            priority=0.7
        )
        
        complex_goal = create_test_utility_goal(
            name="improve_wellbeing",
            target_effects={"hunger": -0.3, "energy": 0.4, "happiness": 0.2},
            priority=0.8
        )
        
        # Verify all are real Goal objects
        goals = [hunger_goal, energy_goal, complex_goal]
        for goal in goals:
            self.assertIsInstance(goal, Goal)
            self.assertNotIsInstance(goal, Mock)
        
        # Test that all work with utility functions
        action = MockAction(
            name="balanced_meal",
            cost=0.2,
            effects=[
                {"attribute": "hunger", "change_value": -0.5},
                {"attribute": "energy", "change_value": 0.2}
            ]
        )
        
        char_state = {"hunger": 0.8, "energy": 0.4, "happiness": 0.6}
        
        for goal in goals:
            utility = calculate_action_utility(char_state, action, current_goal=goal)
            self.assertIsInstance(utility, (int, float))
            print(f"✅ {goal.name} utility: {utility}")


def demonstrate_fix():
    """Demonstrate the specific fix for issue #425."""
    print("=" * 80)
    print("FIX FOR ISSUE #425: Replace Mock() fallback with real Goal objects")
    print("=" * 80)
    print()
    
    print("ORIGINAL PROBLEMATIC CODE (from PR #417):")
    print("```python")
    print("if hasattr(tiny_utility_functions, 'Goal'):")
    print("    goal = tiny_utility_functions.Goal(")
    print("        name=\"test_goal\",")
    print("        target_effects={\"hunger\": 0.2, \"energy\": 0.8},")
    print("        priority=0.7")
    print("    )")
    print("else:")
    print("    # Fallback to mock if Goal not available")
    print("    goal = Mock()  # ❌ THIS IS THE PROBLEM")
    print("    goal.target_effects = {\"hunger\": 0.2, \"energy\": 0.8}")
    print("    goal.priority = 0.7")
    print("```")
    print()
    
    print("FIXED CODE:")
    print("```python")
    print("# Always use real Goal object - no Mock() fallback needed")
    print("goal = Goal(")
    print("    name=\"test_goal\",")
    print("    target_effects={\"hunger\": 0.2, \"energy\": 0.8},")
    print("    priority=0.7")
    print(")")
    print("```")
    print()
    
    print("WHY THE FIX WORKS:")
    print("• Goal class is always available in tiny_utility_functions.py")
    print("• No need for hasattr() check or fallback logic") 
    print("• Real Goal objects test actual functionality")
    print("• Simpler, more reliable code")
    print("• No risk of Mock objects hiding bugs")
    print()


if __name__ == "__main__":
    # Demonstrate the fix
    demonstrate_fix()
    
    # Run the tests
    print("RUNNING FIX VALIDATION TESTS...")
    print("=" * 80)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIssue425Fix)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("FIX VALIDATION RESULTS")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 Fix validation passed!")
        print("✅ Demonstrated the problematic Mock() fallback pattern")
        print("✅ Showed the correct approach using real Goal objects")
        print("✅ Proved that Mock() fallback is unnecessary")
        print("✅ Provided recommended patterns for test Goal creation")
        print()
        print("RECOMMENDATION FOR PR #417:")
        print("Replace the Mock() fallback with direct Goal object creation:")
        print("goal = Goal(name='test_goal', target_effects={...}, priority=0.7)")
    else:
        print("❌ Fix validation failed")
        for test, traceback in result.failures:
            print(f"Failure: {test}")
        for test, traceback in result.errors:
            print(f"Error: {test}")