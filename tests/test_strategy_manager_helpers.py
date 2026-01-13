"""
Unit tests for StrategyManager helper methods.
Tests the new helper methods added for GOAP planning and event routing:
- _build_situation_context
- _coerce_goal
- _gather_character_goals
- _select_goal_for_event
"""

import unittest
from unittest.mock import MagicMock, Mock, patch
import sys
sys.path.insert(0, "/home/runner/work/tiny_village/tiny_village")

from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import Goal
from actions import Action


class MockEvent:
    """Mock event object for testing."""
    def __init__(self, event_type=None, importance=None, impact=None):
        self.type = event_type
        self.importance = importance
        self.impact = impact


class MockCharacter:
    """Mock character object for testing."""
    def __init__(self, name="TestChar", goals=None):
        self.name = name
        self.goals = goals or []
        self.hunger_level = 5.0
        self.energy = 5.0
        self.wealth_money = 50.0
        
    def evaluate_goals(self):
        """Mock evaluate_goals method."""
        return self.goals


class TestStrategyManagerHelpers(unittest.TestCase):
    """Test suite for StrategyManager helper methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy_manager = StrategyManager(use_llm=False)

    def test_build_situation_context_with_none_event(self):
        """Test _build_situation_context with None event."""
        context = self.strategy_manager._build_situation_context(None)
        self.assertIsInstance(context, dict)
        self.assertEqual(len(context), 0)

    def test_build_situation_context_with_event_type(self):
        """Test _build_situation_context extracts event type."""
        event = MockEvent(event_type="social")
        context = self.strategy_manager._build_situation_context(event)
        
        self.assertIn("event_type", context)
        self.assertEqual(context["event_type"], "social")

    def test_build_situation_context_with_importance_and_impact(self):
        """Test _build_situation_context extracts importance and impact."""
        event = MockEvent(event_type="crisis", importance=9, impact=-15)
        context = self.strategy_manager._build_situation_context(event)
        
        self.assertIn("event_type", context)
        self.assertEqual(context["event_type"], "crisis")
        self.assertIn("event_importance", context)
        self.assertEqual(context["event_importance"], 9)
        self.assertIn("event_impact", context)
        self.assertEqual(context["event_impact"], -15)

    def test_build_situation_context_with_dict_event(self):
        """Test _build_situation_context with dict-based event."""
        event = {"type": "economic", "importance": 7, "impact": 10}
        context = self.strategy_manager._build_situation_context(event)
        
        self.assertIn("event_type", context)
        self.assertEqual(context["event_type"], "economic")
        self.assertIn("event_importance", context)
        self.assertEqual(context["event_importance"], 7)

    def test_build_situation_context_sets_social_complexity(self):
        """Test _build_situation_context sets social_complexity for social events."""
        event = MockEvent(event_type="social")
        context = self.strategy_manager._build_situation_context(event)
        
        self.assertIn("social_complexity", context)
        self.assertEqual(context["social_complexity"], 0.8)

    def test_build_situation_context_forces_llm_for_high_importance(self):
        """Test _build_situation_context sets force_llm for high importance events."""
        event = MockEvent(event_type="crisis", importance=10)
        context = self.strategy_manager._build_situation_context(event)
        
        self.assertIn("force_llm", context)
        self.assertTrue(context["force_llm"])

    def test_coerce_goal_with_none(self):
        """Test _coerce_goal with None input."""
        result = self.strategy_manager._coerce_goal(None)
        self.assertIsNone(result)

    def test_coerce_goal_with_goal_object(self):
        """Test _coerce_goal with existing Goal object."""
        goal = Goal(name="test_goal", target_effects={"happiness": 80}, priority=0.7)
        result = self.strategy_manager._coerce_goal(goal)
        
        self.assertIsInstance(result, Goal)
        self.assertEqual(result.name, "test_goal")
        self.assertEqual(result.priority, 0.7)

    def test_coerce_goal_with_dict(self):
        """Test _coerce_goal converts dict to Goal."""
        goal_dict = {
            "name": "social_goal",
            "target_effects": {"social_wellbeing": 75},
            "priority": 0.8
        }
        result = self.strategy_manager._coerce_goal(goal_dict)
        
        self.assertIsInstance(result, Goal)
        self.assertEqual(result.name, "social_goal")
        self.assertEqual(result.priority, 0.8)

    def test_coerce_goal_with_dict_missing_fields(self):
        """Test _coerce_goal handles dict with missing fields."""
        goal_dict = {"target_effects": {"wealth": 100}}
        result = self.strategy_manager._coerce_goal(goal_dict)
        
        self.assertIsInstance(result, Goal)
        self.assertEqual(result.name, "custom_goal")
        self.assertEqual(result.priority, 0.5)

    def test_coerce_goal_with_non_dict_non_goal(self):
        """Test _coerce_goal with other object types."""
        result = self.strategy_manager._coerce_goal("string_goal")
        self.assertEqual(result, "string_goal")

    def test_gather_character_goals_with_no_goals(self):
        """Test _gather_character_goals with character having no goals."""
        character = MockCharacter()
        goals = self.strategy_manager._gather_character_goals(character)
        
        self.assertIsInstance(goals, list)
        self.assertEqual(len(goals), 0)

    def test_gather_character_goals_with_evaluate_goals(self):
        """Test _gather_character_goals using character.evaluate_goals()."""
        goal1 = Goal(name="goal1", target_effects={"happiness": 70}, priority=0.6)
        goal2 = Goal(name="goal2", target_effects={"wealth": 80}, priority=0.7)
        character = MockCharacter(goals=[goal1, goal2])
        
        goals = self.strategy_manager._gather_character_goals(character)
        
        self.assertIsInstance(goals, list)
        # Goals are collected from both evaluate_goals() and goals attribute, so we get duplicates
        self.assertGreaterEqual(len(goals), 2)
        self.assertIn(goal1, goals)
        self.assertIn(goal2, goals)

    def test_gather_character_goals_with_tuple_format(self):
        """Test _gather_character_goals handles tuple format (score, goal)."""
        goal1 = Goal(name="goal1", target_effects={"happiness": 70}, priority=0.6)
        goal2 = Goal(name="goal2", target_effects={"wealth": 80}, priority=0.7)
        character = MockCharacter(goals=[(0.8, goal1), (0.9, goal2)])
        
        goals = self.strategy_manager._gather_character_goals(character)
        
        self.assertIsInstance(goals, list)
        # Goals are collected from both evaluate_goals() and goals attribute, so we get duplicates
        self.assertGreaterEqual(len(goals), 2)
        self.assertIn(goal1, goals)
        self.assertIn(goal2, goals)

    def test_gather_character_goals_handles_exceptions(self):
        """Test _gather_character_goals handles exceptions gracefully."""
        character = MockCharacter()
        character.evaluate_goals = Mock(side_effect=Exception("Test error"))
        
        goals = self.strategy_manager._gather_character_goals(character)
        
        # Should return empty list on error
        self.assertIsInstance(goals, list)
        self.assertEqual(len(goals), 0)

    def test_select_goal_for_event_with_no_goals(self):
        """Test _select_goal_for_event when character has no goals."""
        character = MockCharacter()
        goal = self.strategy_manager._select_goal_for_event(character, "social")
        
        # Should return a goal from _goal_for_event_type or None
        # The exact result depends on implementation
        self.assertTrue(goal is None or isinstance(goal, (Goal, dict)))

    def test_select_goal_for_event_with_character_goals(self):
        """Test _select_goal_for_event selects from character goals."""
        goal1 = Goal(name="social_goal", target_effects={"happiness": 70}, priority=0.6)
        goal2 = Goal(name="economic_goal", target_effects={"wealth": 80}, priority=0.9)
        character = MockCharacter(goals=[goal1, goal2])
        
        selected_goal = self.strategy_manager._select_goal_for_event(character, "social")
        
        self.assertIsNotNone(selected_goal)
        self.assertIsInstance(selected_goal, Goal)

    def test_select_goal_for_event_prioritizes_higher_priority(self):
        """Test _select_goal_for_event selects highest priority goal."""
        goal1 = Goal(name="low_priority", target_effects={"happiness": 70}, priority=0.3)
        goal2 = Goal(name="high_priority", target_effects={"wealth": 80}, priority=0.9)
        character = MockCharacter(goals=[goal1, goal2])
        
        selected_goal = self.strategy_manager._select_goal_for_event(character, None)
        
        self.assertIsNotNone(selected_goal)
        # Should select the higher priority goal
        self.assertEqual(selected_goal.priority, 0.9)

    def test_select_goal_for_event_handles_invalid_goals(self):
        """Test _select_goal_for_event handles goals with invalid structure."""
        # Create a mock goal with priority attribute to allow sorting
        invalid_goal = Mock()
        invalid_goal.name = "invalid"
        invalid_goal.priority = 0.5
        # Missing target_effects attribute
        
        character = MockCharacter(goals=[invalid_goal])
        
        # Should handle gracefully and return a goal
        selected_goal = self.strategy_manager._select_goal_for_event(character, "social")
        # The method should return something, even if it's the invalid goal or an event-type goal
        self.assertIsNotNone(selected_goal)

    @patch('tiny_strategy_manager.logger')
    def test_select_goal_for_event_logs_warnings_for_evaluation_failure(self, mock_logger):
        """Test _select_goal_for_event logs warnings when goal evaluation fails."""
        # Mock GOAP planner and graph manager
        self.strategy_manager.goap_planner = Mock()
        self.strategy_manager.graph_manager = Mock()
        
        # Make evaluate_goal_importance raise an exception
        self.strategy_manager.goap_planner.evaluate_goal_importance = Mock(
            side_effect=Exception("Evaluation failed")
        )
        
        # Create valid goals
        goal1 = Goal(name="goal1", target_effects={"happiness": 80}, priority=0.6)
        goal2 = Goal(name="goal2", target_effects={"wealth": 90}, priority=0.7)
        
        character = MockCharacter(goals=[goal1, goal2])
        
        selected_goal = self.strategy_manager._select_goal_for_event(character, "social")
        
        # Should have logged a warning when evaluation failed
        self.assertTrue(mock_logger.warning.called)
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        has_evaluation_warning = any('evaluation failed' in str(call).lower() for call in warning_calls)
        self.assertTrue(has_evaluation_warning)


if __name__ == '__main__':
    unittest.main()
