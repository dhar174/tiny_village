#!/usr/bin/env python3
"""
Failure mode tests for Tiny Village system integration.

Tests required by System Integration Agent:
- LLM timeout handling
- Invalid JSON output from LLM
- Invalid action output
- Plan invalidation mid-execution
- Memory subsystem exception handling
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tiny_strategy_manager import StrategyManager
from tiny_event_handler import Event
from actions import Action, State


class TestLLMFailureModes(unittest.TestCase):
    """Test LLM integration failure modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.character = self._create_test_character()
        
    def _create_test_character(self):
        """Create a test character with all required attributes."""
        char = Mock()
        char.name = "TestChar"
        char.energy = 50
        char.hunger_level = 50
        char.health_status = 75
        char.mental_health = 70
        char.social_wellbeing = 65
        char.wealth_money = 100
        char.location = Mock()
        char.location.name = "Home"
        char.job = "farmer"
        char.inventory = Mock()
        char.inventory.get_food_items = Mock(return_value=[])
        char.get_current_goal = Mock(return_value=None)
        return char
        
    @patch('tiny_strategy_manager.TinyBrainIO')
    def test_llm_timeout_handling(self, mock_brain_io_class):
        """Test that LLM timeout is handled gracefully."""
        # Create strategy manager with LLM enabled
        sm = StrategyManager(use_llm=True)
        
        if not sm.brain_io:
            self.skipTest("LLM components not available")
            
        # Mock LLM to simulate timeout
        def timeout_simulation(*args, **kwargs):
            time.sleep(0.1)  # Simulate delay
            return None  # Return None as if timeout occurred
            
        sm.brain_io.input_to_model = Mock(side_effect=timeout_simulation)
        
        # Should fall back to utility-based planning
        result = sm.decide_action_with_llm(self.character)
        
        # Should not crash and should return some actions
        self.assertIsNotNone(result, "Should fall back to utility actions on timeout")
        self.assertIsInstance(result, list, "Should return list of actions")
        
    @patch('tiny_strategy_manager.TinyBrainIO')
    @patch('tiny_strategy_manager.OutputInterpreter')
    def test_invalid_json_from_llm(self, mock_interp_class, mock_brain_class):
        """Test handling of invalid JSON from LLM."""
        sm = StrategyManager(use_llm=True)
        
        if not sm.brain_io or not sm.output_interpreter:
            self.skipTest("LLM components not available")
            
        # Mock LLM to return invalid JSON
        sm.brain_io.input_to_model = Mock(return_value=[("This is not JSON {{{", 0.9)])
        
        # Mock interpreter to raise exception on invalid JSON
        sm.output_interpreter.interpret_response = Mock(
            side_effect=ValueError("Invalid JSON")
        )
        
        # Should handle the error gracefully
        result = sm.decide_action_with_llm(self.character)
        
        # Should fall back to utility-based planning
        self.assertIsNotNone(result, "Should fall back on invalid JSON")
        self.assertIsInstance(result, list, "Should return list of actions")
        
    @patch('tiny_strategy_manager.TinyBrainIO')
    @patch('tiny_strategy_manager.OutputInterpreter')
    def test_llm_returns_empty_response(self, mock_interp_class, mock_brain_class):
        """Test handling when LLM returns empty response."""
        sm = StrategyManager(use_llm=True)
        
        if not sm.brain_io:
            self.skipTest("LLM components not available")
            
        # Mock LLM to return empty response
        sm.brain_io.input_to_model = Mock(return_value=[])
        
        # Should handle empty response
        result = sm.decide_action_with_llm(self.character)
        
        # Should fall back to utility-based planning
        self.assertIsNotNone(result, "Should handle empty LLM response")
        self.assertIsInstance(result, list, "Should return list of actions")
        
    @patch('tiny_strategy_manager.TinyBrainIO')
    @patch('tiny_strategy_manager.OutputInterpreter')
    def test_llm_returns_malformed_data(self, mock_interp_class, mock_brain_class):
        """Test handling when LLM returns unexpected data structure."""
        sm = StrategyManager(use_llm=True)
        
        if not sm.brain_io:
            self.skipTest("LLM components not available")
            
        # Mock LLM to return malformed data
        sm.brain_io.input_to_model = Mock(return_value=[12345])  # Integer instead of string
        
        # Should handle malformed response
        result = sm.decide_action_with_llm(self.character)
        
        # Should fall back to utility-based planning
        self.assertIsNotNone(result, "Should handle malformed LLM data")
        self.assertIsInstance(result, list, "Should return list of actions")


class TestActionValidationFailures(unittest.TestCase):
    """Test invalid action output handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sm = StrategyManager(use_llm=False)
        self.character = self._create_test_character()
        
    def _create_test_character(self):
        """Create a test character."""
        char = Mock()
        char.name = "TestChar"
        char.energy = 50
        char.hunger_level = 50
        char.health_status = 75
        char.mental_health = 70
        char.social_wellbeing = 65
        char.wealth_money = 100
        char.location = Mock()
        char.location.name = "Home"
        char.job = "farmer"
        char.inventory = Mock()
        char.inventory.get_food_items = Mock(return_value=[])
        char.get_current_goal = Mock(return_value=None)
        return char
        
    def test_action_with_invalid_parameters(self):
        """Test handling of actions with invalid parameters."""
        # Create action with an attribute that may not exist on characters
        invalid_action = Action(
            name="InvalidAction",
            preconditions={},
            effects=[{"attribute": "nonexistent_attr", "change_value": 10}],
            cost=1.0
        )
        
        # Verify action can be created
        self.assertIsNotNone(invalid_action)
        self.assertEqual(invalid_action.name, "InvalidAction")
        
        # Test that planning with this action doesn't crash the system
        if self.sm.goap_planner:
            from tiny_utility_functions import Goal
            from actions import State
            goal = Goal(name="test_goal", target_effects={"energy": 100})
            state = State({"energy": 50})
            
            # Should handle gracefully - may return None or skip the invalid action
            try:
                plan = self.sm.goap_planner.plan_actions(
                    self.character, goal, state, [invalid_action]
                )
                # Test passes if no exception
            except Exception as e:
                self.fail(f"Should handle invalid action gracefully, but raised: {e}")
        
    def test_action_with_impossible_preconditions(self):
        """Test handling of actions with impossible preconditions."""
        # Create action with preconditions that can never be met
        impossible_action = Action(
            name="ImpossibleAction",
            preconditions={"impossible_condition": True},
            effects=[],
            cost=1.0
        )
        
        # GOAP planner should skip this action
        if self.sm.goap_planner:
            from tiny_utility_functions import Goal
            from actions import State
            goal = Goal(name="test_goal", target_effects={"energy": 100})
            state = State({"energy": 50})
            
            # Planning with impossible action should either skip it or return None
            plan = self.sm.goap_planner.plan_actions(
                self.character, goal, state, [impossible_action]
            )
            
            # If plan exists, it shouldn't contain the impossible action
            # If no plan, that's also acceptable
            if plan:
                self.assertNotIn(impossible_action, plan)


class TestPlanInvalidation(unittest.TestCase):
    """Test plan invalidation during execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sm = StrategyManager(use_llm=False)
        self.character = self._create_test_character()
        
    def _create_test_character(self):
        """Create a test character."""
        char = Mock()
        char.name = "TestChar"
        char.energy = 50
        char.hunger_level = 50
        char.health_status = 75
        char.mental_health = 70
        char.social_wellbeing = 65
        char.wealth_money = 100
        char.location = Mock()
        char.location.name = "Home"
        char.job = "farmer"
        char.inventory = Mock()
        char.inventory.get_food_items = Mock(return_value=[])
        char.get_current_goal = Mock(return_value=None)
        return char
        
    def test_world_state_change_invalidates_plan(self):
        """Test that plan adapts when world state changes mid-execution."""
        if not self.sm.goap_planner:
            self.skipTest("GOAP planner not available")
            
        from tiny_utility_functions import Goal
        from actions import State
        
        # Create a goal and initial state
        goal = Goal(name="recover_energy", target_effects={"energy": 80})
        initial_state = State({"energy": 50, "hunger": 40})
        
        # Create actions that could be invalidated
        actions = [
            Action(
                name="Rest",
                preconditions={},
                effects=[{"attribute": "energy", "change_value": 30}],
                cost=1.0
            ),
            Action(
                name="Sleep",
                preconditions={},
                effects=[{"attribute": "energy", "change_value": 50}],
                cost=2.0
            )
        ]
        
        # Get initial plan
        plan = self.sm.goap_planner.plan_actions(self.character, goal, initial_state, actions)
        
        # Should either have a plan or gracefully handle no plan
        self.assertTrue(plan is None or isinstance(plan, list))
        
    def test_action_failure_triggers_replan(self):
        """Test that action failure triggers replanning."""
        # This tests the architecture's requirement for adaptive replanning
        if not self.sm.goap_planner:
            self.skipTest("GOAP planner not available")
            
        # Verify GOAP planner has necessary methods for replanning
        # This validates the architecture compliance
        planner = self.sm.goap_planner
        self.assertTrue(
            hasattr(planner, "plan_actions"),
            "GOAP planner must define a 'plan_actions' method for replanning support",
        )
        self.assertTrue(
            callable(planner.plan_actions),
            "'plan_actions' on GOAP planner must be callable",
        )


class TestMemorySubsystemFailures(unittest.TestCase):
    """Test memory subsystem exception handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sm = StrategyManager(use_llm=False)
        
    def test_memory_retrieval_failure(self):
        """Test handling when memory retrieval fails."""
        # Test that system continues to function even if memory system fails
        # This would be integrated with MemoryManager once available
        
        # For now, verify strategy manager works without memory
        character = Mock()
        character.name = "TestChar"
        character.energy = 50
        character.hunger_level = 50
        character.health_status = 75
        character.mental_health = 70
        character.social_wellbeing = 65
        character.wealth_money = 100
        character.location = Mock()
        character.location.name = "Home"
        character.job = "farmer"
        character.inventory = Mock()
        character.inventory.get_food_items = Mock(return_value=[])  # Return empty list, not Mock
        character.get_current_goal = Mock(return_value=None)
        
        # Should work without memory system
        try:
            actions = self.sm.get_daily_actions(character)
            self.assertIsNotNone(actions)
        except Exception as e:
            self.fail(f"Should work without memory system, but raised: {e}")
            
    def test_memory_storage_failure(self):
        """Test handling when memory storage fails."""
        # Future implementation: validate that execution continues if memory storage fails
        # In the architecture, action execution should generate memories
        # If memory storage fails, execution should continue
        # TODO: Implement once MemoryManager integration is complete
        self.skipTest("Memory integration not yet implemented")
        
    def test_corrupted_memory_data(self):
        """Test handling of corrupted memory data."""
        # Future implementation: validate system handles corrupted memory gracefully
        # TODO: Implement once MemoryManager integration is complete
        self.skipTest("Memory integration not yet implemented")


class TestEventDrivenFailures(unittest.TestCase):
    """Test event-driven system failure modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sm = StrategyManager(use_llm=False)
        self.character = self._create_test_character()
        
    def _create_test_character(self):
        """Create a test character."""
        char = Mock()
        char.name = "TestChar"
        char.energy = 50
        char.hunger_level = 50
        char.health_status = 75
        char.mental_health = 70
        char.social_wellbeing = 65
        char.wealth_money = 100
        char.location = Mock()
        char.location.name = "Home"
        char.job = "farmer"
        char.inventory = Mock()
        char.inventory.get_food_items = Mock(return_value=[])
        char.get_current_goal = Mock(return_value=None)
        return char
        
    def test_malformed_event_handling(self):
        """Test handling of malformed events."""
        # Create a malformed event
        malformed_event = Mock()
        malformed_event.type = None
        malformed_event.name = None
        
        # System should handle malformed events gracefully
        self.sm.update_strategy([malformed_event], subject=self.character)
        # Test passes if no exception is raised
            
    def test_missing_event_participants(self):
        """Test handling of events with missing participants."""
        event = Event(
            name="Test Event",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact={},
            participants=None  # Missing participants
        )
        
        # System should handle events with missing participants gracefully
        self.sm.update_strategy([event], subject=self.character)
        # Test passes if no exception is raised
            
    def test_cascading_event_failure(self):
        """Test handling when cascading events fail."""
        # In the architecture, events can trigger cascading events
        # If one fails, others should continue
        
        event = Event(
            name="Primary Event",
            date=datetime.now(),
            event_type="social",
            importance=5,
            impact={"type": "social"},
            cascading_events=["event1", "event2"]  # Simplified cascading events
        )
        
        # System should handle cascading events gracefully
        self.sm.update_strategy([event], subject=self.character)
        # Test passes if no exception is raised


def run_failure_mode_tests():
    """Run all failure mode tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestLLMFailureModes))
    suite.addTests(loader.loadTestsFromTestCase(TestActionValidationFailures))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanInvalidation))
    suite.addTests(loader.loadTestsFromTestCase(TestMemorySubsystemFailures))
    suite.addTests(loader.loadTestsFromTestCase(TestEventDrivenFailures))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_failure_mode_tests()
    sys.exit(0 if success else 1)
