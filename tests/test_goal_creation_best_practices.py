#!/usr/bin/env python3
"""
Best practices for creating concrete goal objects in utility tests.

These tests demonstrate the issue #425 fix: when a utility test needs a goal object,
use a lightweight concrete class or factory with the expected interface instead of a
`Mock()` fallback branch.
"""

import os
import sys
import unittest
from unittest.mock import Mock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the utility function we want to test
from tiny_utility_functions import calculate_action_utility


class MockAction:
    """Simple mock action for testing when Action class is not available."""
    def __init__(self, name, cost=0.0, effects=None):
        self.name = name
        self.cost = cost
        self.effects = effects or []


class UtilityTestGoal:
    """Concrete goal object for utility tests."""

    def __init__(self, name, target_effects=None, priority=0.5, score=None):
        self.name = name
        self.target_effects = target_effects or {}
        self.priority = priority
        self.score = priority if score is None else score
        self.urgency = priority
        self.attributes = {}


class TestGoalCreationBestPractices(unittest.TestCase):
    """Test suite demonstrating correct Goal object creation patterns."""

    def test_use_lightweight_concrete_goal_for_utility_functions(self):
        """
        CORRECT APPROACH: use a lightweight concrete goal class for utility tests.
        """
        goal = UtilityTestGoal(
            name="TestGoal",
            target_effects={"hunger": -0.5, "energy": 0.3},
            priority=0.8,
        )

        self.assertIsInstance(goal, UtilityTestGoal)
        self.assertEqual(goal.name, "TestGoal")
        self.assertEqual(goal.target_effects, {"hunger": -0.5, "energy": 0.3})
        self.assertEqual(goal.priority, 0.8)

        action = MockAction(
            name="EatFood",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.4}],
        )

        # Test utility calculation with real Goal object
        char_state = {"hunger": 0.8, "energy": 0.6}
        utility = calculate_action_utility(char_state, action, current_goal=goal)

        self.assertAlmostEqual(utility, 25.4)

    def test_problematic_mock_fallback_pattern(self):
        """
        PROBLEMATIC PATTERN: Using Mock() as fallback.
        
        This demonstrates why the Mock() fallback is problematic:
        1. Mock objects may not have the expected attributes/behavior
        2. Tests may pass even when real Goal objects would fail
        3. It doesn't test the actual integration with real Goal objects
        """
        goal = Mock()
        goal.target_effects = {"hunger": -0.5}
        goal.priority = 0.8
        goal.name = "MockedGoal"

        # Verify this creates a Mock object (which is problematic)
        self.assertIsInstance(goal, Mock)

        action = MockAction(
            name="EatFood",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.4}],
        )

        char_state = {"hunger": 0.8, "energy": 0.6}

        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertAlmostEqual(utility, 25.4)

    def test_correct_goal_creation_multiple_approaches(self):
        """
        CORRECT APPROACHES: Multiple ways to create real Goal objects.
        
        This demonstrates several correct approaches that avoid Mock() fallbacks.
        """
        
        # Approach 1: instantiate the concrete goal class directly
        simple_goal = UtilityTestGoal(
            name="SimpleGoal",
            target_effects={"hunger": -0.6},
            priority=0.9,
        )

        self.assertIsInstance(simple_goal, UtilityTestGoal)
        self.assertEqual(simple_goal.name, "SimpleGoal")

        # Approach 2: use a helper factory for consistency
        def create_test_goal(name, target_effects, priority=0.5):
            """Helper function to create test goals consistently."""
            return UtilityTestGoal(
                name=name,
                target_effects=target_effects,
                priority=priority,
            )

        helper_goal = create_test_goal(
            name="HelperGoal",
            target_effects={"energy": 0.4, "happiness": 0.2},
            priority=0.7,
        )

        # Approach 3: adapt a score-first representation to the priority interface
        adapted_goal = UtilityTestGoal(
            name="AdaptedGoal",
            target_effects={"hunger": -0.7},
            priority=0.8,
            score=0.8,
        )

        self.assertIsInstance(helper_goal, UtilityTestGoal)
        self.assertEqual(helper_goal.priority, 0.7)

        # Test all goals work with utility functions
        action = MockAction(
            name="TestAction",
            cost=0.2,
            effects=[{"attribute": "hunger", "change_value": -0.3}],
        )

        char_state = {"hunger": 0.9, "energy": 0.4}

        utility1 = calculate_action_utility(char_state, action, current_goal=simple_goal)
        utility2 = calculate_action_utility(char_state, action, current_goal=adapted_goal)
        utility3 = calculate_action_utility(char_state, action, current_goal=helper_goal)

        self.assertAlmostEqual(utility1, 25.9)
        self.assertAlmostEqual(utility2, 23.4)
        self.assertAlmostEqual(utility3, 3.4)

    def test_goal_creation_error_handling(self):
        """
        Goal helpers should produce objects with the attributes utility code expects.
        """

        goal = UtilityTestGoal(
            name="ErrorHandlingGoal",
            target_effects={"test_attribute": 0.5},
            priority=0.6,
        )
        action = MockAction("TestAction", cost=0.1, effects=[])
        char_state = {"test_attribute": 0.3}

        utility = calculate_action_utility(char_state, action, current_goal=goal)

        self.assertAlmostEqual(utility, -1.0)
        self.assertEqual(goal.score, 0.6)

    def test_best_practice_goal_factory(self):
        """
        BEST PRACTICE: Goal factory function for consistent test Goal creation.
        
        This provides a reusable pattern for creating test Goals consistently.
        """
        
        def create_utility_test_goal(
            name="TestGoal",
            target_effects=None,
            priority=0.5,
            score=None
        ):
            """
            Factory function to create Goal objects for utility function testing.
            
            This ensures consistent Goal creation and avoids Mock() fallbacks.
            """
            if target_effects is None:
                target_effects = {"hunger": -0.3, "energy": 0.2}

            return UtilityTestGoal(
                name=name,
                target_effects=target_effects,
                priority=priority,
                score=score,
            )

        # Use the factory to create test goals
        survival_goal = create_utility_test_goal(
            name="Survival",
            target_effects={"hunger": -0.8, "health": 0.2},
            priority=0.9,
        )

        comfort_goal = create_utility_test_goal(
            name="Comfort",
            target_effects={"energy": 0.5, "happiness": 0.3},
            priority=0.6,
        )

        self.assertIsInstance(survival_goal, UtilityTestGoal)
        self.assertIsInstance(comfort_goal, UtilityTestGoal)

        # Test both goals work with utility functions
        action = MockAction(
            name="Rest",
            cost=0.3,
            effects=[
                {"attribute": "energy", "change_value": 0.4},
                {"attribute": "hunger", "change_value": 0.1},
            ],
        )

        char_state = {"hunger": 0.7, "energy": 0.3, "health": 0.8, "happiness": 0.5}

        survival_utility = calculate_action_utility(char_state, action, current_goal=survival_goal)
        comfort_utility = calculate_action_utility(char_state, action, current_goal=comfort_goal)

        self.assertAlmostEqual(survival_utility, 1.2)
        self.assertAlmostEqual(comfort_utility, 16.2)


def demonstrate_correct_patterns():
    """Demonstrate the correct patterns vs problematic Mock() fallback."""
    print("=" * 80)
    print("GOAL CREATION BEST PRACTICES")
    print("=" * 80)
    print()
    
    print("❌ PROBLEMATIC PATTERN (from issue #425):")
    print("   if hasattr(tiny_utility_functions, 'Goal'):")
    print("       goal = tiny_utility_functions.Goal(...)")
    print("   else:")
    print("       # Fallback to mock if Goal not available")
    print("       goal = Mock()  # <-- THIS IS THE PROBLEM")
    print()
    
    print("✅ CORRECT PATTERNS:")
    print("   1. Use a lightweight concrete goal class (RECOMMENDED):")
    print("      goal = UtilityTestGoal(name='TestGoal', target_effects={...}, priority=0.8)")
    print()
    print("   2. Use goal factory function:")
    print("      def create_test_goal(name, effects, priority):")
    print("          return UtilityTestGoal(name=name, target_effects=effects, priority=priority)")
    print()
    print("   3. Provide both score and priority when adapting goal-like test fixtures.")
    print()
    
    print("WHY MOCK() FALLBACK IS PROBLEMATIC:")
    print("• Mock objects may not have correct attributes/methods")
    print("• Tests may pass even when real Goal objects would fail")
    print("• Doesn't test actual integration with real Goal classes")
    print("• Can hide bugs that only appear with real Goal objects")
    print("• Provides false confidence in test coverage")
    print()


if __name__ == "__main__":
    # First demonstrate the patterns
    demonstrate_correct_patterns()
    
    # Then run the tests
    print("RUNNING BEST PRACTICES TESTS...")
    print("=" * 80)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGoalCreationBestPractices)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 All best practices tests passed!")
        print("✓ Demonstrated correct Goal object creation patterns")
        print("✓ Showed why Mock() fallbacks are problematic")
        print("✓ Provided reusable patterns for test Goal creation")
        print("\nRECOMMENDATION:")
        print("Replace Mock() fallbacks with concrete goal object creation using:")
        print("goal = UtilityTestGoal(name='TestGoal', target_effects={...}, priority=0.8)")
    else:
        print("❌ Some tests failed")
        for test, traceback in result.failures:
            print(f"Failure: {test}")
        for test, traceback in result.errors:
            print(f"Error: {test}")
