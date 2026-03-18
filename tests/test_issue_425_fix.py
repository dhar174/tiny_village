#!/usr/bin/env python3
"""
Regression coverage for issue #425: avoid Mock() fallback branches in utility tests.

These tests focus on the pattern described in the issue report: when a utility test
needs a goal object, use a lightweight concrete goal implementation that exposes the
attributes `calculate_action_utility()` actually reads instead of falling back to
`Mock()`.
"""

import os
import sys
import unittest
from unittest.mock import Mock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the utility function under test
from tiny_utility_functions import calculate_action_utility


class UtilityTestAction:
    """Concrete action fixture that mirrors the utility function contract."""

    def __init__(
        self,
        name,
        cost=0.0,
        effects=None,
        preconditions=None,
        priority=0.5,
        target=None,
        initiator=None,
        action_id=None,
    ):
        self.name = name
        self.cost = float(cost)
        self.effects = [] if effects is None else effects
        self.preconditions = [] if preconditions is None else preconditions
        self.priority = priority
        self.target = target
        self.initiator = initiator
        self.action_id = id(self) if action_id is None else action_id

    def preconditions_met(self, state=None):
        """
        Evaluate whether all preconditions are met.

        Supports:
        - bool values
        - callables (optionally accepting `state`)
        - objects with a `check_condition` method (optionally accepting `state`)

        Any unsupported precondition type will raise a TypeError to avoid
        silently treating it as truthy.
        """
        if not self.preconditions:
            return True

        for cond in self.preconditions:
            # Direct boolean precondition
            if isinstance(cond, bool):
                result = cond
            # Callable precondition (function, lambda, etc.)
            elif callable(cond):
                try:
                    result = cond(state)
                except TypeError:
                    # Fallback for callables that do not accept `state`
                    result = cond()
            # Condition-like object with `check_condition`
            elif hasattr(cond, "check_condition"):
                check = cond.check_condition
                try:
                    result = check(state)
                except TypeError:
                    # Fallback for `check_condition` without `state` parameter
                    result = check()
            else:
                raise TypeError(
                    f"Unsupported precondition type: {type(cond)!r} in {self!r}"
                )

            if not bool(result):
                return False

        return True


class UtilityTestGoal:
    """
    Concrete goal fixture for utility tests.

    `calculate_action_utility()` currently reads `target_effects` and `priority`,
    and related tests often also expect `name`, `score`, `urgency`, and
    `attributes` to exist on goal-like objects.
    """

    def __init__(self, name, target_effects=None, priority=0.5):
        self.name = name
        self.target_effects = target_effects or {}
        self.priority = priority
        self.score = priority
        self.urgency = priority
        self.attributes = {}


class TestIssue425Fix(unittest.TestCase):
    """Test that demonstrates the fix for issue #425."""

    def test_problematic_pattern_from_issue(self):
        """
        This recreates the problematic fallback pattern from the issue.

        The point of the regression is that a Mock goal can return a number here
        without proving that a concrete goal object is compatible with the utility
        calculation.
        """

        goal = Mock()
        goal.target_effects = {"hunger": -0.3, "energy": 0.8}
        goal.priority = 0.7
        goal.name = "test_goal"

        # Verify we got a Mock (which is the problem)
        self.assertIsInstance(goal, Mock)

        # Create test action and state
        action = UtilityTestAction(
            name="test_action",
            cost=1.0,
            effects=[
                {"attribute": "hunger", "change_value": -0.3},
                {"attribute": "energy", "change_value": -0.1},
            ],
        )

        char_state = {"hunger": 0.5, "energy": 0.7}

        # Test utility calculation with Mock goal
        utility = calculate_action_utility(char_state, action, current_goal=goal)

        # The problem: This might work, but it's not testing real Goal behavior
        self.assertIsInstance(utility, (int, float))

    def test_correct_fix_no_fallback_needed(self):
        """
        CORRECT FIX: use a lightweight concrete goal object instead of Mock().
        """

        goal = UtilityTestGoal(
            name="test_goal",
            target_effects={"hunger": -0.3, "energy": 0.8},
            priority=0.7,
        )

        # Verify we got a concrete goal object
        self.assertNotIsInstance(goal, Mock)

        # Verify Goal has the expected attributes
        self.assertEqual(goal.name, "test_goal")
        self.assertEqual(goal.target_effects, {"hunger": -0.3, "energy": 0.8})
        self.assertEqual(goal.priority, 0.7)
        self.assertEqual(goal.score, 0.7)

        # Create test action and state
        action = UtilityTestAction(
            name="test_action",
            cost=1.0,
            effects=[
                {"attribute": "hunger", "change_value": -0.3},
                {"attribute": "energy", "change_value": -0.1},
            ],
        )

        self.assertNotIsInstance(action, Mock)
        self.assertTrue(callable(action.preconditions_met))
        self.assertEqual(action.priority, 0.5)

        char_state = {"hunger": 0.5, "energy": 0.7}

        # Test utility calculation with a concrete goal object
        utility = calculate_action_utility(char_state, action, current_goal=goal)

        # 3.0 = 0.5*0.3*20 hunger need fulfillment.
        # Energy contributes 0 because the action reduces energy and the scorer only
        # rewards positive energy changes; the positive energy target also does not
        # add a goal bonus for a negative action effect. Total: 3.0 + 17.5 - 10.0 = 10.5.
        self.assertAlmostEqual(utility, 10.5)

    def test_direct_comparison_mock_vs_real(self):
        """
        Direct comparison between a Mock fallback and a concrete goal object.
        """

        # Create Mock goal (problematic approach)
        mock_goal = Mock()
        mock_goal.target_effects = {"hunger": -0.5}
        mock_goal.priority = 0.8
        mock_goal.name = "mock_goal"

        # Create concrete goal (correct approach)
        real_goal = UtilityTestGoal(
            name="real_goal",
            target_effects={"hunger": -0.5},
            priority=0.8,
        )

        # Test action
        action = UtilityTestAction(
            name="eat_food",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.4}],
        )

        char_state = {"hunger": 0.9}

        # Calculate utility with both
        mock_utility = calculate_action_utility(char_state, action, current_goal=mock_goal)
        real_utility = calculate_action_utility(char_state, action, current_goal=real_goal)

        # Both should work, but only the concrete goal tests an explicit interface
        # 0.9*0.4*20 hunger + 0.8*25 goal bonus - 0.1*10 cost = 26.2
        self.assertAlmostEqual(mock_utility, 26.2)
        self.assertAlmostEqual(real_utility, 26.2)

        # The key difference is that real_goal is an explicit concrete implementation
        self.assertIsInstance(mock_goal, Mock)
        self.assertIsInstance(real_goal, UtilityTestGoal)
        self.assertEqual(real_goal.score, 0.8)
        self.assertEqual(real_goal.priority, 0.8)
        self.assertEqual(mock_goal.priority, 0.8)

    def test_recommended_pattern_for_tests(self):
        """
        Demonstrates the recommended pattern that should replace the problematic code.
        """

        def create_test_utility_goal(name, target_effects, priority=0.5):
            """Create a lightweight concrete goal for utility tests."""

            return UtilityTestGoal(
                name=name,
                target_effects=target_effects,
                priority=priority,
            )

        # Use the helper to create various test goals
        hunger_goal = create_test_utility_goal(
            name="satisfy_hunger",
            target_effects={"hunger": -0.8},
            priority=0.9,
        )

        energy_goal = create_test_utility_goal(
            name="restore_energy",
            target_effects={"energy": 0.6},
            priority=0.7,
        )

        complex_goal = create_test_utility_goal(
            name="improve_wellbeing",
            target_effects={"hunger": -0.3, "energy": 0.4, "happiness": 0.2},
            priority=0.8,
        )

        # Verify all are concrete goal objects
        goals = [hunger_goal, energy_goal, complex_goal]
        for goal in goals:
            self.assertIsInstance(goal, UtilityTestGoal)
            self.assertNotIsInstance(goal, Mock)

        # Test that all work with utility functions
        action = UtilityTestAction(
            name="balanced_meal",
            cost=0.2,
            effects=[
                {"attribute": "hunger", "change_value": -0.5},
                {"attribute": "energy", "change_value": 0.2},
            ],
        )

        char_state = {"hunger": 0.8, "energy": 0.4, "happiness": 0.6}

        utilities = [
            calculate_action_utility(char_state, action, current_goal=goal)
            for goal in goals
        ]
        self.assertEqual(len(utilities), 3)
        # utilities[0]: 8.0 = 0.8*0.5*20 hunger, 1.8 = (1-0.4)*0.2*15 energy,
        # 22.5 = 0.9*25 hunger-goal bonus, and -2.0 = 0.2*10 cost => 30.3.
        # utilities[1]: same 8.0 hunger + 1.8 energy, 17.5 = 0.7*25 energy-goal
        # bonus, and -2.0 cost => 25.3.
        # utilities[2]: same 8.0 hunger + 1.8 energy, 20.0 = 0.8*25 first matching
        # goal bonus from hunger, and -2.0 cost => 27.8.
        self.assertAlmostEqual(utilities[0], 30.3)
        self.assertAlmostEqual(utilities[1], 25.3)
        self.assertAlmostEqual(utilities[2], 27.8)


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
    print("# Use a lightweight concrete goal object - no Mock() fallback needed")
    print("goal = UtilityTestGoal(")
    print("    name=\"test_goal\",")
    print("    target_effects={\"hunger\": -0.3, \"energy\": 0.8},")
    print("    priority=0.7")
    print(")")
    print("```")
    print()
    
    print("WHY THE FIX WORKS:")
    print("• No need for hasattr() check or fallback logic")
    print("• Concrete goal objects test actual functionality")
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
        print("✅ Showed the correct approach using concrete Goal objects")
        print("✅ Proved that Mock() fallback is unnecessary")
        print("✅ Provided recommended patterns for test Goal creation")
        print()
        print("RECOMMENDATION:")
        print("Replace the Mock() fallback with direct Goal object creation:")
        print("goal = UtilityTestGoal(name='test_goal', target_effects={...}, priority=0.7)")
    else:
        print("❌ Fix validation failed")
        for test, traceback in result.failures:
            print(f"Failure: {test}")
        for test, traceback in result.errors:
            print(f"Error: {test}")
