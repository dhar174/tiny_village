#!/usr/bin/env python3
"""
Best practices for creating real Goal objects in tests instead of using Mock() fallbacks.

This test demonstrates the correct approach to address issue #425 where Mock() fallbacks
create fake objects that may not correctly test utility function behavior.

The issue was that code like this:
    if hasattr(tiny_utility_functions, 'Goal'):
        goal = tiny_utility_functions.Goal(...)
    else:
        # Fallback to mock if Goal not available  <-- PROBLEMATIC
        goal = Mock()

Should be replaced with proper Goal object creation that tests real functionality.
"""

import unittest
import sys
import os
from unittest.mock import Mock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the utility functions we want to test
from tiny_utility_functions import calculate_action_utility, Goal

# Also import the comprehensive Goal class from tiny_characters
try:
    from tiny_characters import Goal as ComprehensiveGoal
    COMPREHENSIVE_GOAL_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_GOAL_AVAILABLE = False

# Import Action for testing
try:
    from actions import Action
    ACTION_AVAILABLE = True
except ImportError:
    ACTION_AVAILABLE = False


class MockAction:
    """Simple mock action for testing when Action class is not available."""
    def __init__(self, name, cost=0.0, effects=None):
        self.name = name
        self.cost = cost
        self.effects = effects or []


class TestGoalCreationBestPractices(unittest.TestCase):
    """Test suite demonstrating correct Goal object creation patterns."""

    def test_use_simple_goal_from_utility_functions(self):
        """
        CORRECT APPROACH: Use the simple Goal class from tiny_utility_functions.
        
        This is the preferred approach for utility function testing because:
        1. It's specifically designed for utility calculations
        2. It has a simple constructor that's easy to use in tests
        3. It provides real Goal object behavior, not fake Mock behavior
        """
        # Create a real Goal object using the simple Goal class
        goal = Goal(
            name="TestGoal",
            target_effects={"hunger": -0.5, "energy": 0.3},
            priority=0.8
        )
        
        # Verify it's a real Goal object, not a Mock
        self.assertIsInstance(goal, Goal)
        self.assertEqual(goal.name, "TestGoal")
        self.assertEqual(goal.target_effects, {"hunger": -0.5, "energy": 0.3})
        self.assertEqual(goal.priority, 0.8)
        
        # Create a test action
        if ACTION_AVAILABLE:
            action = Action(
                name="EatFood",
                preconditions={},
                effects=[{"attribute": "hunger", "change_value": -0.4}],
                cost=0.1
            )
        else:
            action = MockAction(
                name="EatFood",
                cost=0.1,
                effects=[{"attribute": "hunger", "change_value": -0.4}]
            )
        
        # Test utility calculation with real Goal object
        char_state = {"hunger": 0.8, "energy": 0.6}
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        
        # This should work correctly because we're using a real Goal object
        self.assertIsInstance(utility, (int, float))
        print(f"✓ Utility calculation with real Goal: {utility}")

    def test_problematic_mock_fallback_pattern(self):
        """
        PROBLEMATIC PATTERN: Using Mock() as fallback.
        
        This demonstrates why the Mock() fallback is problematic:
        1. Mock objects may not have the expected attributes/behavior
        2. Tests may pass even when real Goal objects would fail
        3. It doesn't test the actual integration with real Goal objects
        """
        # This is the PROBLEMATIC pattern from the issue
        if hasattr(Goal, '__fake_attribute_that_does_not_exist__'):
            # This will never be true, so we fall back to Mock
            goal = Goal("TestGoal", target_effects={"hunger": -0.5}, priority=0.8)
        else:
            # Fallback to mock if Goal not available - THIS IS THE PROBLEM
            goal = Mock()
            goal.target_effects = {"hunger": -0.5}
            goal.priority = 0.8
            goal.name = "MockedGoal"
        
        # Verify this creates a Mock object (which is problematic)
        self.assertIsInstance(goal, Mock)
        
        # Create a test action
        action = MockAction(
            name="EatFood",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.4}]
        )
        
        # Test utility calculation with Mock object
        char_state = {"hunger": 0.8, "energy": 0.6}
        
        # This might work with Mock, but it doesn't test real Goal behavior
        try:
            utility = calculate_action_utility(char_state, action, current_goal=goal)
            print(f"⚠️  Mock fallback 'worked' but doesn't test real functionality: {utility}")
            
            # The problem: This test might pass even if the utility function
            # doesn't work correctly with real Goal objects
            self.assertIsInstance(utility, (int, float))
            
        except Exception as e:
            print(f"✓ Mock fallback failed as expected: {e}")
            # If it fails, that actually shows the Mock doesn't behave like a real Goal

    def test_correct_goal_creation_multiple_approaches(self):
        """
        CORRECT APPROACHES: Multiple ways to create real Goal objects.
        
        This demonstrates several correct approaches that avoid Mock() fallbacks.
        """
        
        # Approach 1: Always use the simple Goal class (RECOMMENDED)
        simple_goal = Goal(
            name="SimpleGoal",
            target_effects={"hunger": -0.6},
            priority=0.9
        )
        
        self.assertIsInstance(simple_goal, Goal)
        self.assertEqual(simple_goal.name, "SimpleGoal")
        
        # Approach 2: Use the comprehensive Goal class if available
        if COMPREHENSIVE_GOAL_AVAILABLE:
            # Create minimal required objects for comprehensive Goal
            mock_character = Mock()
            mock_character.name = "TestCharacter"
            
            mock_graph_manager = Mock()
            
            # Simple condition for completion
            mock_condition = Mock()
            mock_condition.name = "test_condition"
            
            comprehensive_goal = ComprehensiveGoal(
                name="ComprehensiveGoal",
                description="Test comprehensive goal",
                character=mock_character,
                target=mock_character,
                score=0.8,
                completion_conditions={False: [mock_condition]},
                evaluate_utility_function=lambda char, goal, env: 0.8,
                difficulty=lambda char, env: 0.5,
                completion_reward=lambda char, env: 10,
                failure_penalty=lambda char, env: -5,
                completion_message=lambda char, env: "Goal completed!",
                failure_message=lambda char, env: "Goal failed!",
                criteria=[],
                graph_manager=mock_graph_manager,
                goal_type="test",
                target_effects={"hunger": -0.7}
            )
            
            self.assertIsInstance(comprehensive_goal, ComprehensiveGoal)
            self.assertEqual(comprehensive_goal.name, "ComprehensiveGoal")
            self.assertEqual(comprehensive_goal.target_effects, {"hunger": -0.7})
            
            print("✓ Created comprehensive Goal object successfully")
        else:
            print("ℹ️  Comprehensive Goal class not available, using simple Goal")
            comprehensive_goal = simple_goal
        
        # Approach 3: Create helper function for consistent Goal creation
        def create_test_goal(name, target_effects, priority=0.5):
            """Helper function to create test goals consistently."""
            return Goal(
                name=name,
                target_effects=target_effects,
                priority=priority
            )
        
        helper_goal = create_test_goal(
            name="HelperGoal",
            target_effects={"energy": 0.4, "happiness": 0.2},
            priority=0.7
        )
        
        self.assertIsInstance(helper_goal, Goal)
        self.assertEqual(helper_goal.priority, 0.7)
        
        # Test all goals work with utility functions
        action = MockAction(
            name="TestAction",
            cost=0.2,
            effects=[{"attribute": "hunger", "change_value": -0.3}]
        )
        
        char_state = {"hunger": 0.9, "energy": 0.4}
        
        # Test with simple goal
        utility1 = calculate_action_utility(char_state, action, current_goal=simple_goal)
        self.assertIsInstance(utility1, (int, float))
        
        # Test with comprehensive goal
        utility2 = calculate_action_utility(char_state, action, current_goal=comprehensive_goal)
        self.assertIsInstance(utility2, (int, float))
        
        # Test with helper-created goal
        utility3 = calculate_action_utility(char_state, action, current_goal=helper_goal)
        self.assertIsInstance(utility3, (int, float))
        
        print(f"✓ All real Goal objects work with utility functions")
        print(f"   Simple goal utility: {utility1}")
        print(f"   Comprehensive goal utility: {utility2}")
        print(f"   Helper goal utility: {utility3}")

    def test_goal_creation_error_handling(self):
        """
        CORRECT ERROR HANDLING: How to handle missing dependencies properly.
        
        Instead of falling back to Mock(), handle missing dependencies gracefully.
        """
        
        # Correct approach: Skip test if dependencies are missing
        try:
            # Try to import what we need
            from tiny_utility_functions import Goal, calculate_action_utility
            
            # Create real Goal object
            goal = Goal(
                name="ErrorHandlingGoal",
                target_effects={"test_attribute": 0.5},
                priority=0.6
            )
            
            # Test with real Goal
            action = MockAction("TestAction", cost=0.1, effects=[])
            char_state = {"test_attribute": 0.3}
            
            utility = calculate_action_utility(char_state, action, current_goal=goal)
            self.assertIsInstance(utility, (int, float))
            
            print("✓ Error handling test passed with real Goal object")
            
        except ImportError as e:
            # Correct approach: Skip the test, don't use Mock()
            self.skipTest(f"Required dependencies not available: {e}")
        
        except Exception as e:
            # Handle other errors appropriately
            self.fail(f"Unexpected error with real Goal object: {e}")

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
            
            return Goal(
                name=name,
                target_effects=target_effects,
                priority=priority,
                score=score
            )
        
        # Use the factory to create test goals
        survival_goal = create_utility_test_goal(
            name="Survival",
            target_effects={"hunger": -0.8, "health": 0.2},
            priority=0.9
        )
        
        comfort_goal = create_utility_test_goal(
            name="Comfort",
            target_effects={"energy": 0.5, "happiness": 0.3},
            priority=0.6
        )
        
        # Verify factory creates real Goal objects
        self.assertIsInstance(survival_goal, Goal)
        self.assertIsInstance(comfort_goal, Goal)
        
        # Test both goals work with utility functions
        action = MockAction(
            name="Rest",
            cost=0.3,
            effects=[
                {"attribute": "energy", "change_value": 0.4},
                {"attribute": "hunger", "change_value": 0.1}
            ]
        )
        
        char_state = {"hunger": 0.7, "energy": 0.3, "health": 0.8, "happiness": 0.5}
        
        survival_utility = calculate_action_utility(char_state, action, current_goal=survival_goal)
        comfort_utility = calculate_action_utility(char_state, action, current_goal=comfort_goal)
        
        self.assertIsInstance(survival_utility, (int, float))
        self.assertIsInstance(comfort_utility, (int, float))
        
        print(f"✓ Goal factory pattern works correctly")
        print(f"   Survival goal utility: {survival_utility}")
        print(f"   Comfort goal utility: {comfort_utility}")


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
    print("   1. Always use tiny_utility_functions.Goal (RECOMMENDED):")
    print("      goal = Goal(name='TestGoal', target_effects={...}, priority=0.8)")
    print()
    print("   2. Use goal factory function:")
    print("      def create_test_goal(name, effects, priority):")
    print("          return Goal(name=name, target_effects=effects, priority=priority)")
    print()
    print("   3. Skip test if dependencies missing:")
    print("      try:")
    print("          from tiny_utility_functions import Goal")
    print("          goal = Goal(...)")
    print("      except ImportError:")
    print("          self.skipTest('Dependencies not available')")
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
        print("Replace Mock() fallbacks with real Goal object creation using:")
        print("goal = Goal(name='TestGoal', target_effects={...}, priority=0.8)")
    else:
        print("❌ Some tests failed")
        for test, traceback in result.failures:
            print(f"Failure: {test}")
        for test, traceback in result.errors:
            print(f"Error: {test}")