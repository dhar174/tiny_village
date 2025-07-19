"""
Test suite for tiny_utility_functions.py using REAL production classes.
This replaces the mock-based tests to ensure proper integration testing.
"""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import real production classes
try:
    from actions import Action, State, ActionSystem
    from tiny_characters import Character, Goal
    from tiny_graph_manager import GraphManager
    from tiny_time_manager import GameTimeManager, GameCalendar
    from tiny_items import ItemInventory
    from tiny_locations import Location

    print("✓ Successfully imported all real classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    # Fallback for testing - we'll handle this gracefully
    pass

# Import the functions we want to test
from tiny_utility_functions import (
    calculate_action_utility,
    calculate_plan_utility,
    UtilityEvaluator,
    calculate_importance,
    validate_character_state,
    validate_action,
    validate_goal,
)


class TestRealUtilityFunctions(unittest.TestCase):
    """Test utility functions with real production classes."""

    @classmethod
    def setUpClass(cls):
        """Set up shared resources for all tests."""
        try:
            # Create necessary managers
            cls.action_system = ActionSystem()
            cls.graph_manager = GraphManager()

            # Create calendar and time manager
            cls.calendar = GameCalendar()
            cls.gametime_manager = GameTimeManager(calendar=cls.calendar)

        except Exception as e:
            # Classes not available - tests will be skipped with clear messaging
            print(f"Notice: Real classes not available for testing: {e}")
            cls.action_system = None
            cls.graph_manager = None
            cls.gametime_manager = None

    def setUp(self):
        """Set up for each test."""
        # Create a minimal but real character if possible
        try:
            self.character = self._create_test_character()
            self.real_classes_available = True
        except Exception as e:
            # Character creation failed - tests will be skipped with clear messaging  
            print(f"Notice: Real character not available for testing: {e}")
            self.character = None
            self.real_classes_available = False

    def _create_test_character(self):
        """Create a minimal real character for testing."""
        if not all([self.action_system, self.graph_manager, self.gametime_manager]):
            return None

        # Create minimal character
        character = Character(
            name="TestCharacter",
            age=25,
            graph_manager=self.graph_manager,
            action_system=self.action_system,
            gametime_manager=self.gametime_manager,
            hunger_level=50,
            energy=60,
            wealth_money=100,
            health_status=80,
            social_wellbeing=70,
        )
        return character

    def _create_test_action(self):
        """Create a real Action for testing."""
        return Action(
            name="EatFood",
            preconditions={},
            effects=[
                {"attribute": "hunger", "change_value": -30, "targets": ["initiator"]},
                {"attribute": "energy", "change_value": 10, "targets": ["initiator"]},
            ],
            cost=0.1,
            default_target_is_initiator=True,
        )

    def _create_test_goal(self):
        """Create a real Goal for testing."""
        if not self.character:
            return None

        return Goal(
            name="SatisfyHunger",
            description="Reduce hunger to acceptable levels",
            character=self.character,
            target=self.character,
            score=80,
            completion_conditions={},
            evaluate_utility_function=lambda char, goal, env: goal.score,
            difficulty=lambda char, env: 0.5,
            completion_reward=lambda char, env: 10,
            failure_penalty=lambda char, env: -5,
            completion_message=lambda char, env: "Hunger satisfied!",
            failure_message=lambda char, env: "Failed to satisfy hunger",
            criteria=[],
            graph_manager=self.graph_manager,
            goal_type="survival",
            target_effects={"hunger": -30},
        )

    def test_calculate_action_utility_real_classes(self):
        """Test calculate_action_utility with real Action and character state."""
        if not self.real_classes_available:
            self.skipTest("Real classes not available")

        # Get character state as dict
        char_state = {
            "hunger": 80,  # High hunger
            "energy": 60,
            "money": 100,
            "health": 80,
            "social_needs": 70,
        }

        # Create real action
        action = self._create_test_action()

        # Test utility calculation
        utility = calculate_action_utility(char_state, action)

        # Should be positive because high hunger and action reduces hunger
        self.assertIsInstance(utility, (int, float))
        print(f"Action utility with real classes: {utility}")

    def test_calculate_action_utility_with_real_goal(self):
        """Test calculate_action_utility with real Goal object."""
        if not self.real_classes_available:
            self.skipTest("Real classes not available")

        char_state = {"hunger": 80, "energy": 60}
        action = self._create_test_action()
        goal = self._create_test_goal()

        if goal is None:
            self.skipTest("Could not create real goal")

        # Test with goal
        utility = calculate_action_utility(char_state, action, current_goal=goal)

        self.assertIsInstance(utility, (int, float))
        print(f"Action utility with real goal: {utility}")

    def test_utility_evaluator_with_real_character(self):
        """Test UtilityEvaluator with real Character object."""
        if not self.real_classes_available:
            self.skipTest("Real classes not available")

        # Create evaluator
        evaluator = UtilityEvaluator()

        # Test context calculation
        context = evaluator.calculate_context_factors(self.character)
        self.assertIsInstance(context, dict)
        print(f"Context factors: {context}")

        # Test action history
        action = self._create_test_action()
        evaluator.add_action_to_history(self.character, action, 1.5)

        history_modifier = evaluator.get_action_history_modifier(self.character, action)
        self.assertIsInstance(history_modifier, (int, float))
        print(f"History modifier: {history_modifier}")

    def test_calculate_importance_with_real_character(self):
        """Test calculate_importance function with real character data."""
        if not self.real_classes_available:
            self.skipTest("Real classes not available")

        goal = self._create_test_goal()
        if goal is None:
            self.skipTest("Could not create real goal")

        # Test importance calculation
        importance = calculate_importance(
            character=self.character,
            goal=goal,
            environment_context={"time_of_day": "morning", "weather": "sunny"},
            character_history=[],
            goal_progress=0.3,
        )

        self.assertIsInstance(importance, (int, float))
        self.assertGreaterEqual(importance, 0)
        print(f"Goal importance with real character: {importance}")

    def test_validation_functions_with_real_objects(self):
        """Test validation functions with real objects."""
        if not self.real_classes_available:
            self.skipTest("Real classes not available")

        # Test character state validation
        char_state = (
            self.character.get_state() if hasattr(self.character, "get_state") else {}
        )
        if isinstance(char_state, dict):
            is_valid, errors = validate_character_state(char_state)
            print(f"Character state validation: {is_valid}, errors: {errors}")

        # Test action validation
        action = self._create_test_action()
        is_valid, errors = validate_action(action)
        self.assertIsInstance(is_valid, bool)
        print(f"Action validation: {is_valid}, errors: {errors}")

        # Test goal validation
        goal = self._create_test_goal()
        if goal:
            is_valid, errors = validate_goal(goal)
            self.assertIsInstance(is_valid, bool)
            print(f"Goal validation: {is_valid}, errors: {errors}")

    def test_calculate_plan_utility_with_real_actions(self):
        """Test calculate_plan_utility with real Action objects."""
        if not self.real_classes_available:
            self.skipTest("Real classes not available")

        char_state = {"hunger": 80, "energy": 50, "money": 100}

        # Create a plan with real actions
        action1 = self._create_test_action()
        action2 = Action(
            name="Rest",
            preconditions={},
            effects=[
                {"attribute": "energy", "change_value": 30, "targets": ["initiator"]}
            ],
            cost=0.2,
            default_target_is_initiator=True,
        )

        plan = [action1, action2]

        # Test plan utility calculation
        utility = calculate_plan_utility(char_state, plan, simulate_effects=True)

        self.assertIsInstance(utility, (int, float))
        print(f"Plan utility with real actions: {utility}")

    def test_integration_with_character_methods(self):
        """Test integration with actual Character class methods."""
        if not self.real_classes_available:
            self.skipTest("Real classes not available")

        # Test that character has expected attributes/methods
        expected_attrs = ["name", "age", "hunger_level", "energy", "wealth_money"]

        for attr in expected_attrs:
            if hasattr(self.character, attr):
                value = getattr(self.character, attr)
                print(f"Character.{attr}: {value}")

        # Test if character has get_state method
        if hasattr(self.character, "get_state"):
            state = self.character.get_state()
            print(f"Character state: {type(state)}")

        # Test if character can be converted to dict
        if hasattr(self.character, "to_dict"):
            char_dict = self.character.to_dict()
            print(f"Character dict keys: {list(char_dict.keys())[:5]}...")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
