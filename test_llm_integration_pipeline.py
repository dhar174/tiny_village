#!/usr/bin/env python3
"""Tests for LLM integration with StrategyManager decision-making loop.

These tests validate that the LLM components are properly integrated into the
character decision-making pipeline as described in documentation_summary.txt.
"""

import unittest
import logging
from unittest.mock import Mock, patch, MagicMock

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestLLMIntegrationPipeline(unittest.TestCase):
    """Test LLM integration in the strategy manager decision loop."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test character with LLM enabled
        self.character = Mock()
        self.character.name = "TestCharacter"
        self.character.id = "TestCharacter"
        self.character.use_llm_decisions = True
        self.character.hunger_level = 6.0
        self.character.energy = 4.0
        self.character.wealth_money = 25.0
        self.character.social_wellbeing = 7.0
        self.character.mental_health = 6.0
        self.character.health_status = 5.0
        self.character.job = "farmer"
        self.character.location = Mock()
        self.character.location.name = "Home"
        self.character.inventory = Mock()
        self.character.inventory.get_food_items = Mock(return_value=[])

    def test_strategy_manager_llm_initialization(self):
        """Test that StrategyManager can be initialized with LLM support."""
        # Mock the LLM components to avoid dependencies
        with patch('tiny_strategy_manager.TinyBrainIO') as mock_brain_io, \
             patch('tiny_strategy_manager.PromptBuilder') as mock_prompt_builder, \
             patch('tiny_strategy_manager.OutputInterpreter') as mock_output_interpreter:
            
            # Make the imports available
            mock_brain_io.return_value = Mock()
            mock_output_interpreter.return_value = Mock()
            
            from tiny_strategy_manager import StrategyManager
            
            # Test LLM-enabled initialization
            manager = StrategyManager(use_llm=True, model_name="test-model")
            
            self.assertTrue(manager.use_llm)
            self.assertIsNotNone(manager.brain_io)
            self.assertIsNotNone(manager.output_interpreter)

    def test_character_llm_configuration(self):
        """Test character LLM configuration utilities."""
        try:
            from llm_integration_utils import (
                setup_character_llm_integration,
                validate_llm_integration,
                create_llm_test_character
            )
            
            # Test character setup
            test_char = create_llm_test_character("Alice", enable_llm=True)
            self.assertTrue(test_char.use_llm_decisions)
            self.assertEqual(test_char.name, "Alice")
            
            # Test LLM enabling/disabling
            setup_character_llm_integration(test_char, enable_llm=False)
            self.assertFalse(test_char.use_llm_decisions)
            
            setup_character_llm_integration(test_char, enable_llm=True)
            self.assertTrue(test_char.use_llm_decisions)
            
        except ImportError:
            self.skipTest("LLM integration utils not available")

    def test_update_strategy_llm_integration(self):
        """Test that update_strategy can use LLM decisions for new day events."""
        with patch('tiny_strategy_manager.TinyBrainIO') as mock_brain_io, \
             patch('tiny_strategy_manager.PromptBuilder') as mock_prompt_builder, \
             patch('tiny_strategy_manager.OutputInterpreter') as mock_output_interpreter, \
             patch('tiny_strategy_manager.GraphManager'), \
             patch('tiny_strategy_manager.GOAPPlanner'):
            
            # Mock the LLM components
            mock_brain_io.return_value = Mock()
            mock_output_interpreter.return_value = Mock()
            
            from tiny_strategy_manager import StrategyManager
            
            # Create LLM-enabled strategy manager
            manager = StrategyManager(use_llm=True)
            
            # Mock the decide_action_with_llm method
            manager.decide_action_with_llm = Mock(return_value=[Mock(name="LLM_Action")])
            
            # Create a new day event
            new_day_event = Mock()
            new_day_event.type = "new_day"
            
            # Enable LLM for the character in the manager
            manager.enable_llm_for_character(self.character)
            
            # Test that update_strategy uses LLM for new day events
            result = manager.update_strategy([new_day_event], self.character)
            
            # Verify LLM decision method was called
            manager.decide_action_with_llm.assert_called_once()
            self.assertIsNotNone(result)

    def test_decide_action_with_llm_pipeline(self):
        """Test the complete LLM decision-making pipeline."""
        with patch('tiny_strategy_manager.TinyBrainIO') as mock_brain_io_class, \
             patch('tiny_strategy_manager.PromptBuilder') as mock_prompt_builder_class, \
             patch('tiny_strategy_manager.OutputInterpreter') as mock_output_interpreter_class, \
             patch('tiny_strategy_manager.GraphManager'), \
             patch('tiny_strategy_manager.GOAPPlanner'):
            
            # Create mock instances
            mock_brain_io = Mock()
            mock_prompt_builder = Mock()
            mock_output_interpreter = Mock()
            
            mock_brain_io_class.return_value = mock_brain_io
            mock_prompt_builder_class.return_value = mock_prompt_builder
            mock_output_interpreter_class.return_value = mock_output_interpreter
            
            # Mock the LLM pipeline components
            mock_brain_io.input_to_model.return_value = ["I choose to eat"]  # Simple string response
            mock_output_interpreter.interpret_response.return_value = [Mock(name="EatAction")]
            
            # Mock prompt generation
            mock_prompt_builder.generate_decision_prompt.return_value = "Mock prompt for decision making"
            
            from tiny_strategy_manager import StrategyManager
            from actions import Action
            
            # Create LLM-enabled strategy manager
            manager = StrategyManager(use_llm=True)
            
            # Mock get_daily_actions to return potential actions
            manager.get_daily_actions = Mock(return_value=[
                Action(name="Eat", preconditions=[], effects=[], cost=0.1),
                Action(name="Sleep", preconditions=[], effects=[], cost=0.2)
            ])
            
            # Test LLM decision making
            result = manager.decide_action_with_llm(self.character, time="morning", weather="sunny")
            
            # Verify the pipeline was called
            mock_brain_io.input_to_model.assert_called_once()
            mock_output_interpreter.interpret_response.assert_called_once()
            self.assertIsNotNone(result)
            self.assertTrue(len(result) > 0)

    def test_llm_fallback_to_utility(self):
        """Test that LLM failures properly fall back to utility-based decisions."""
        with patch('tiny_strategy_manager.TinyBrainIO') as mock_brain_io_class, \
             patch('tiny_strategy_manager.PromptBuilder') as mock_prompt_builder_class, \
             patch('tiny_strategy_manager.OutputInterpreter') as mock_output_interpreter_class, \
             patch('tiny_strategy_manager.GraphManager'), \
             patch('tiny_strategy_manager.GOAPPlanner'):
            
            # Create mock instances
            mock_brain_io = Mock()
            mock_brain_io_class.return_value = mock_brain_io
            mock_prompt_builder_class.return_value = Mock()
            mock_output_interpreter_class.return_value = Mock()
            
            # Make the LLM fail
            mock_brain_io.input_to_model.side_effect = Exception("LLM service unavailable")
            
            from tiny_strategy_manager import StrategyManager
            from actions import Action
            
            # Create LLM-enabled strategy manager
            manager = StrategyManager(use_llm=True)
            
            # Mock get_daily_actions to return utility-based actions
            utility_actions = [Action(name="UtilityAction", preconditions=[], effects=[], cost=0.1)]
            manager.get_daily_actions = Mock(return_value=utility_actions)
            
            # Test LLM decision making with failure
            result = manager.decide_action_with_llm(self.character, time="morning", weather="sunny")
            
            # Verify fallback to utility actions
            manager.get_daily_actions.assert_called()
            self.assertEqual(result, utility_actions)

    def test_character_llm_flag_integration(self):
        """Test that character use_llm_decisions flag is properly respected."""
        with patch('tiny_strategy_manager.TinyBrainIO') as mock_brain_io, \
             patch('tiny_strategy_manager.PromptBuilder') as mock_prompt_builder, \
             patch('tiny_strategy_manager.OutputInterpreter') as mock_output_interpreter, \
             patch('tiny_strategy_manager.GraphManager'), \
             patch('tiny_strategy_manager.GOAPPlanner'):
            
            mock_brain_io.return_value = Mock()
            mock_output_interpreter.return_value = Mock()
            
            from tiny_strategy_manager import StrategyManager
            
            manager = StrategyManager(use_llm=True)
            
            # Test with LLM disabled character
            self.character.use_llm_decisions = False
            manager.disable_llm_for_character(self.character)
            
            # Verify character is not tracked for LLM
            self.assertNotIn(self.character.name, manager._characters_using_llm)
            
            # Test with LLM enabled character
            self.character.use_llm_decisions = True
            manager.enable_llm_for_character(self.character)
            
            # Verify character is tracked for LLM
            self.assertIn(self.character.name, manager._characters_using_llm)


class TestLLMIntegrationUtils(unittest.TestCase):
    """Test LLM integration utility functions."""

    def test_enable_llm_for_characters(self):
        """Test enabling LLM for multiple characters."""
        try:
            from llm_integration_utils import enable_llm_for_characters, create_llm_test_character
            
            # Create test characters
            char1 = create_llm_test_character("Alice", enable_llm=False)
            char2 = create_llm_test_character("Bob", enable_llm=False)
            char3 = create_llm_test_character("Charlie", enable_llm=False)
            
            characters = [char1, char2, char3]
            
            # Enable LLM for specific characters
            enabled = enable_llm_for_characters(characters, ["Alice", "Bob"])
            
            self.assertTrue(char1.use_llm_decisions)
            self.assertTrue(char2.use_llm_decisions)
            self.assertFalse(char3.use_llm_decisions)
            self.assertEqual(len(enabled), 2)
            
        except ImportError:
            self.skipTest("LLM integration utils not available")

    def test_validate_llm_integration(self):
        """Test LLM integration validation."""
        try:
            from llm_integration_utils import validate_llm_integration, create_llm_test_character
            
            # Create test character
            character = create_llm_test_character("TestChar", enable_llm=True)
            
            # Create mock strategy manager
            strategy_manager = Mock()
            strategy_manager.use_llm = True
            strategy_manager.brain_io = Mock()
            strategy_manager.output_interpreter = Mock()
            strategy_manager.decide_action_with_llm = Mock()
            
            # Validate integration
            results = validate_llm_integration(character, strategy_manager)
            
            self.assertTrue(results['character_llm_enabled'])
            self.assertTrue(results['strategy_manager_llm_enabled'])
            self.assertTrue(results['brain_io_available'])
            self.assertTrue(results['output_interpreter_available'])
            self.assertTrue(results['decide_action_with_llm_method'])
            self.assertTrue(results['fully_integrated'])
            
        except ImportError:
            self.skipTest("LLM integration utils not available")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)