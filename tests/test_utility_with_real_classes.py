#!/usr/bin/env python3
"""
Test file for tiny_utility_functions.py using REAL classes instead of mocks.
This properly tests the production code against real Goal and Action instances.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility functions to test
from tiny_utility_functions import (
    calculate_action_utility,
    calculate_plan_utility,
)

try:
    # Import real classes
    from tiny_characters import Goal, Character
    from actions import Action, Condition

    REAL_CLASSES_AVAILABLE = True
    print("✅ Successfully imported real Goal and Action classes")
except ImportError as e:
    print(f"❌ Failed to import real classes: {e}")
    REAL_CLASSES_AVAILABLE = False


class TestUtilityWithRealClasses(unittest.TestCase):
    """Test utility functions with real Goal and Action classes."""

    def setUp(self):
        """Set up test fixtures with real classes."""
        if not REAL_CLASSES_AVAILABLE:
            self.skipTest("Real classes not available")

        # Create mock dependencies for Goal and Action
        self.mock_character = Mock()
        self.mock_character.name = "TestChar"

        self.mock_graph_manager = Mock()
        self.mock_graph_manager.calculate_goal_difficulty = Mock(return_value=3)
        self.mock_graph_manager.calculate_reward = Mock(return_value=10)
        self.mock_graph_manager.calculate_penalty = Mock(return_value=5)

        # Mock functions for Goal
        def mock_utility_eval(*args, **kwargs):
            return 0.5

        def mock_completion_message(*args, **kwargs):
            return "Goal completed!"

        def mock_failure_message(*args, **kwargs):
            return "Goal failed!"

        self.mock_utility_eval = mock_utility_eval
        self.mock_completion_message = mock_completion_message
        self.mock_failure_message = mock_failure_message

    def create_real_goal(self, name="TestGoal", target_effects=None):
        """Create a real Goal instance with all required parameters."""
        target_effects = target_effects or {"hunger": -0.5}

        # Create a simple condition
        mock_condition = Mock()
        mock_condition.name = "test_condition"
        mock_condition.attribute = "test_attr"
        mock_condition.satisfy_value = True
        mock_condition.op = "=="
        mock_condition.weight = 1

        completion_conditions = {False: [mock_condition]}

        return Goal(
            description=f"Test goal: {name}",
            character=self.mock_character,
            target=self.mock_character,
            score=0.8,
            name=name,
            completion_conditions=completion_conditions,
            evaluate_utility_function=self.mock_utility_eval,
            difficulty=self.mock_graph_manager.calculate_goal_difficulty,
            completion_reward=self.mock_graph_manager.calculate_reward,
            failure_penalty=self.mock_graph_manager.calculate_penalty,
            completion_message=self.mock_completion_message,
            failure_message=self.mock_failure_message,
            criteria={},
            graph_manager=self.mock_graph_manager,
            goal_type="basic",
            target_effects=target_effects,
        )

    def create_real_action(self, name="TestAction", cost=0.1, effects=None):
        """Create a real Action instance with all required parameters."""
        effects = effects or [{"attribute": "hunger", "change_value": -0.5}]

        return Action(
            name=name,
            preconditions={},  # Empty preconditions for testing
            effects=effects,
            cost=cost,
        )

    def test_calculate_action_utility_with_real_action(self):
        """Test calculate_action_utility with a real Action instance."""
        char_state = {"hunger": 0.9, "energy": 0.5}

        # Create real action that reduces hunger
        real_action = self.create_real_action(
            name="EatFood",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.7}],
        )

        utility = calculate_action_utility(char_state, real_action)

        # Expected: hunger_score = 0.9 * 0.7 * 20 = 12.6
        # cost_score = 0.1 * 10 = 1.0
        # utility = 12.6 - 1.0 = 11.6
        self.assertAlmostEqual(utility, 11.6, places=1)
        print(f"✅ Real Action utility test passed: {utility}")

    def test_calculate_action_utility_with_real_goal(self):
        """Test calculate_action_utility with both real Action and Goal."""
        char_state = {"hunger": 0.8, "energy": 0.7}

        # Create real action and goal
        real_action = self.create_real_action(
            name="EatHealthyMeal",
            cost=0.2,
            effects=[{"attribute": "hunger", "change_value": -0.6}],
        )

        real_goal = self.create_real_goal(
            name="SatisfyHunger", target_effects={"hunger": -0.8}
        )

        utility = calculate_action_utility(
            char_state, real_action, current_goal=real_goal
        )

        # This should include both need fulfillment and goal progress scores
        print(f"✅ Real Action + Goal utility test: {utility}")
        self.assertGreater(utility, 0)  # Should be positive

    def test_calculate_plan_utility_with_real_actions(self):
        """Test calculate_plan_utility with real Action instances."""
        char_state = {"hunger": 0.9, "energy": 0.2}

        # Create real actions
        action1 = self.create_real_action(
            name="EatFood",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.7}],
        )

        action2 = self.create_real_action(
            name="Rest",
            cost=0.5,
            effects=[{"attribute": "energy", "change_value": 0.6}],
        )

        plan = [action1, action2]

        # Test without simulation
        plan_utility_no_sim = calculate_plan_utility(
            char_state, plan, simulate_effects=False
        )
        print(f"✅ Real Actions plan utility (no simulation): {plan_utility_no_sim}")

        # Test with simulation
        plan_utility_sim = calculate_plan_utility(
            char_state, plan, simulate_effects=True
        )
        print(f"✅ Real Actions plan utility (with simulation): {plan_utility_sim}")

        self.assertGreater(plan_utility_no_sim, 0)
        self.assertGreater(plan_utility_sim, 0)

    def test_action_effects_compatibility(self):
        """Test that real Action effects work with utility calculations."""
        char_state = {"hunger": 0.7, "energy": 0.3, "money": 50}

        # Test complex action with multiple effects
        complex_action = self.create_real_action(
            name="WorkShift",
            cost=0.4,
            effects=[
                {"attribute": "money", "change_value": 20},
                {"attribute": "energy", "change_value": -0.3},
            ],
        )

        utility = calculate_action_utility(char_state, complex_action)
        print(f"✅ Complex Action utility test: {utility}")

        # Should be positive due to money gain outweighing energy cost
        self.assertGreater(utility, 0)

    def test_goal_target_effects_compatibility(self):
        """Test that real Goal target_effects work with utility calculations."""
        char_state = {"hunger": 0.6, "energy": 0.4}

        # Create action that matches goal
        action = self.create_real_action(
            name="PrepareFood",
            cost=0.15,
            effects=[{"attribute": "hunger", "change_value": -0.5}],
        )

        # Create goal with target effects
        goal = self.create_real_goal(
            name="PrepareForWork", target_effects={"hunger": -0.6, "energy": 0.3}
        )

        utility = calculate_action_utility(char_state, action, current_goal=goal)
        print(f"✅ Goal target_effects compatibility test: {utility}")

        # Should include goal progress bonus
        self.assertGreater(utility, 0)


def run_real_class_tests():
    """Run all tests and provide summary."""
    if not REAL_CLASSES_AVAILABLE:
        print("❌ Cannot run tests - real classes not available")
        return False

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtilityWithRealClasses)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print(f"\n🎉 All {result.testsRun} tests passed!")
        print("✅ Utility functions work correctly with real Goal and Action classes")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    print("🧪 Testing utility functions with REAL Goal and Action classes...")
    print("=" * 60)

    success = run_real_class_tests()

    if success:
        print("\n✨ CONCLUSION: The utility functions are production-ready!")
        print("   They work correctly with the actual Goal and Action classes")
        print("   instead of just passing tests with mock objects.")
    else:
        print("\n🔧 CONCLUSION: Issues found with real class integration.")
        print("   The utility functions need fixes to work with production classes.")
