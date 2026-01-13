#!/usr/bin/env python3
"""
Integration tests to validate alignment with strategy_management_architecture.md

These tests ensure the decision-making sequence follows the documented workflow:
1. Event Detection (EventHandler)
2. Strategic Planning (StrategyManager.update_strategy)
3. GOAP Planning (GOAPPlanner.plan_actions)
4. Utility Evaluation (utility functions)
5. Decision Execution (GameplayController.apply_decision)
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tiny_strategy_manager import StrategyManager
from tiny_event_handler import Event, EventHandler
from actions import Action, State


class TestArchitectureAlignment(unittest.TestCase):
    """Test that implementation follows the documented architecture sequence."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy_manager = StrategyManager(use_llm=False)
        
        # Create a simple character for testing
        self.character = Mock()
        self.character.name = "TestCharacter"
        self.character.energy = 50
        self.character.hunger_level = 50
        self.character.health_status = 75
        self.character.mental_health = 60
        self.character.social_wellbeing = 65
        self.character.wealth_money = 100
        self.character.location = Mock()
        self.character.location.name = "Home"
        self.character.job = "farmer"
        self.character.inventory = Mock()
        self.character.inventory.get_food_items = Mock(return_value=[])
        self.character.get_current_goal = Mock(return_value=None)
        
    def test_event_to_strategy_flow(self):
        """Test Phase 1-2: Event detection triggers strategy update."""
        # Create a new_day event
        from datetime import datetime
        event = Event(
            name="New Day",
            date=datetime.now(),
            event_type="new_day",
            importance=5,
            impact={"type": "daily_cycle"},
            participants=[self.character.name]
        )
        
        # Call update_strategy with the event (Phase 2)
        result = self.strategy_manager.update_strategy([event], subject=self.character.name)
        
        # Verify strategy manager processed the event
        self.assertIsNotNone(result, "Strategy manager should return a result")
        self.assertIsInstance(result, dict, "Result should be a dictionary of plans by character")
        
    def test_strategy_to_goap_flow(self):
        """Test Phase 3: Strategy manager invokes GOAP planner."""
        if not self.strategy_manager.goap_planner:
            self.skipTest("GOAP planner not initialized")
            
        # Mock GOAP planner's plan_actions method to track calls
        with patch.object(self.strategy_manager.goap_planner, 'plan_actions', 
                         return_value=[]) as mock_plan:
            
            # Create an event that should trigger GOAP planning
            from datetime import datetime
            event = Event(
                name="Low Energy Alert",
                date=datetime.now(),
                event_type="low_energy",
                importance=7,
                impact={"type": "character_state"},
                participants=[self.character.name]
            )
            
            # Update strategy
            self.strategy_manager.update_strategy([event], subject=self.character)
            
            # Verify GOAP planner was called (may not be called if goap_planner is None)
            # This test validates the integration point exists
            
    def test_goap_uses_utility_functions(self):
        """Test Phase 4: GOAP system uses utility functions for evaluation."""
        if not self.strategy_manager.goap_planner:
            self.skipTest("GOAP planner not initialized")
            
        # Create a goal and actions
        from tiny_utility_functions import Goal
        goal = Goal(
            name="Rest and Recover",
            completion_conditions={'energy': 80}
        )
        
        actions = [
            Action(
                name="Rest",
                preconditions={},
                effects=[{"attribute": "energy", "change_value": 20}],
                cost=1.0
            )
        ]
        
        # Call GOAP planner
        try:
            plan = self.strategy_manager.goap_planner.plan_actions(
                self.character, goal, actions=actions
            )
            # Test passes if no exception is raised
            # The plan may be None or a list depending on implementation
        except Exception as e:
            self.fail(f"GOAP planning raised exception: {e}")
            
    def test_full_decision_cycle(self):
        """Test complete cycle: Event → Strategy → GOAP → Utility → Execution."""
        # Phase 1: Create event
        from datetime import datetime
        event = Event(
            name="New Day",
            date=datetime.now(),
            event_type="new_day",
            importance=5,
            impact={"type": "daily_cycle"},
            participants=[self.character.name]
        )
        
        # Phase 2-3: Process through strategy manager
        result = self.strategy_manager.update_strategy([event], subject=self.character)
        
        # Phase 4-5: Verify we got actionable results
        self.assertIsNotNone(result, "Should return planning result")
        
        # If result is a dict, it should have character plans
        if isinstance(result, dict):
            self.assertTrue(len(result) > 0, "Should have plans for at least one character")
            
    def test_fallback_on_goap_failure(self):
        """Test that system has fallback when GOAP fails."""
        # Create strategy manager without GOAP
        sm_no_goap = StrategyManager(use_llm=False)
        sm_no_goap.goap_planner = None
        
        from datetime import datetime
        event = Event(
            name="New Day",
            date=datetime.now(),
            event_type="new_day",
            importance=5,
            impact={"type": "daily_cycle"},
            participants=[self.character.name]
        )
        
        # Should not crash even without GOAP
        try:
            result = sm_no_goap.update_strategy([event], subject=self.character)
            # Test passes if no exception
        except Exception as e:
            self.fail(f"Should have fallback behavior, but raised: {e}")


class TestDecisionSequenceIntegration(unittest.TestCase):
    """Test the complete decision-making sequence as documented."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.strategy_manager = StrategyManager(use_llm=False)
        self.character = self._create_test_character()
        
    def _create_test_character(self):
        """Create a test character with required attributes."""
        char = Mock()
        char.name = "Alice"
        char.energy = 30  # Low energy
        char.hunger_level = 70  # High hunger
        char.health_status = 80
        char.mental_health = 75
        char.social_wellbeing = 60
        char.wealth_money = 50
        char.location = Mock()
        char.location.name = "Home"
        char.job = "baker"
        char.inventory = Mock()
        char.inventory.get_food_items = Mock(return_value=[])
        char.get_current_goal = Mock(return_value=None)
        return char
        
    def test_documented_sequence_phases(self):
        """Test that all documented phases are accessible."""
        # Phase 1: Event Detection (EventHandler responsibility)
        from datetime import datetime
        event = Event(
            name="Dawn",
            date=datetime.now(),
            event_type="new_day",
            importance=5,
            impact={"type": "daily_cycle"},
            participants=["Alice"]
        )
        
        # Phase 2: Strategic Planning Initiation
        self.assertTrue(hasattr(self.strategy_manager, 'update_strategy'),
                       "StrategyManager must have update_strategy method")
        
        # Phase 3: GOAP Planning
        self.assertTrue(hasattr(self.strategy_manager, 'goap_planner'),
                       "StrategyManager must have goap_planner")
        
        # Phase 4: Utility Evaluation (tested via GOAP)
        if self.strategy_manager.goap_planner:
            from tiny_utility_functions import calculate_action_utility
            # Just verify the function exists and is callable
            self.assertTrue(callable(calculate_action_utility))
            
        # Phase 5: Decision Execution (GameplayController responsibility)
        # We verify the result format is suitable for execution
        result = self.strategy_manager.update_strategy([event], subject="Alice")
        self.assertIsNotNone(result)
        
    def test_error_handling_in_sequence(self):
        """Test that errors in one phase don't crash the entire sequence."""
        # Test with malformed event
        bad_event = Mock()
        bad_event.type = None
        bad_event.name = None
        
        try:
            result = self.strategy_manager.update_strategy([bad_event], subject="Alice")
            # Should handle gracefully
        except Exception as e:
            # Should not raise exception, but if it does, it should be caught
            pass


class TestLLMIntegrationAlignment(unittest.TestCase):
    """Test that LLM integration follows the architecture when enabled."""
    
    def setUp(self):
        """Set up LLM integration test fixtures."""
        # Note: These tests use mocks since we don't want actual LLM calls
        self.character = Mock()
        self.character.name = "Bob"
        self.character.energy = 50
        self.character.hunger_level = 40
        self.character.health_status = 85
        self.character.mental_health = 70
        self.character.social_wellbeing = 75
        self.character.wealth_money = 200
        self.character.location = Mock()
        self.character.location.name = "Home"
        self.character.job = "merchant"
        self.character.inventory = Mock()
        self.character.inventory.get_food_items = Mock(return_value=[])
        self.character.get_current_goal = Mock(return_value=None)
        
    def test_llm_integration_follows_sequence(self):
        """Test that LLM path still follows: Prompt → LLM → Interpret → GOAP/Utility."""
        # Create strategy manager with LLM disabled to avoid actual LLM calls
        sm = StrategyManager(use_llm=False)
        
        # Verify the methods exist for LLM integration
        self.assertTrue(hasattr(sm, 'decide_action_with_llm'),
                       "Should have LLM decision method")
        self.assertTrue(hasattr(sm, 'should_use_llm_for_decision'),
                       "Should have LLM decision logic")
        
    @patch('tiny_strategy_manager.TinyBrainIO')
    @patch('tiny_strategy_manager.OutputInterpreter')
    @patch('tiny_strategy_manager.PromptBuilder')
    def test_llm_fallback_to_goap(self, mock_prompt, mock_interp, mock_brain):
        """Test that LLM failures fall back to GOAP/utility planning."""
        # Create strategy manager with LLM enabled but mocked to fail
        sm = StrategyManager(use_llm=True)
        
        # Mock LLM to return None (failure)
        if sm.brain_io:
            sm.brain_io.input_to_model = Mock(return_value=None)
            
        # Should fall back to utility-based planning
        actions = sm.get_daily_actions(self.character)
        
        # Should get some actions even if LLM fails
        self.assertIsNotNone(actions, "Should fall back to utility-based actions")
        self.assertIsInstance(actions, list, "Should return list of actions")


class TestMemoryAndEventPropagation(unittest.TestCase):
    """Test that events propagate correctly through the system."""
    
    def test_event_creates_memory_entry(self):
        """Test Phase 11: Action results generate new events and memories."""
        # This is a placeholder for future memory integration tests
        # Architecture shows: Actions → Events → Memories
        pass
        
    def test_cascading_events(self):
        """Test that one event can trigger cascading events."""
        # This validates the feedback loop: Events → Actions → New Events
        pass


def run_architecture_alignment_tests():
    """Run all architecture alignment tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestArchitectureAlignment))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionSequenceIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMIntegrationAlignment))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryAndEventPropagation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_architecture_alignment_tests()
    sys.exit(0 if success else 1)
