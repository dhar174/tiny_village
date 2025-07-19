import unittest
from unittest.mock import MagicMock, patch
import logging

from tiny_strategy_manager import StrategyManager
from tiny_prompt_builder import PromptBuilder
from tiny_output_interpreter import OutputInterpreter
from tiny_brain_io import TinyBrainIO
from tiny_gameplay_controller import GameplayController

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MockCharacter:
    def __init__(self, name="TestChar"):
        self.name = name
        self.hunger_level = 5.0
        self.energy = 5.0
        self.wealth_money = 50.0
        self.social_wellbeing = 5.0
        self.mental_health = 5.0
        self.inventory = MagicMock()
        self.location = MagicMock()
        self.job = "unemployed"
        self.use_llm_decisions = False  # For testing LLM integration

        # Mock inventory behavior
        self.mock_food_items = []
        self.inventory.get_food_items = MagicMock(return_value=self.mock_food_items)

    def add_food_item(self, name, calories):
        food_item = MagicMock()
        food_item.name = name
        food_item.calories = calories
        self.mock_food_items.append(food_item)


class MockLocation:
    def __init__(self, name="TestLocation"):
        self.name = name


class TestLLMIntegration(unittest.TestCase):
    """Test the complete LLM decision-making integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.character = MockCharacter("Alice")
        self.character.location = MockLocation("Home")

        # Set up StrategyManager with LLM enabled
        self.strategy_manager = StrategyManager(
            use_llm=True, model_name="gpt-3.5-turbo"
        )

        # Set up other components
        self.prompt_builder = PromptBuilder()
        self.output_interpreter = OutputInterpreter()
        self.brain_io = TinyBrainIO()

    @patch("tiny_strategy_manager.TinyBrainIO")
    @patch("tiny_strategy_manager.PromptBuilder")
    @patch("tiny_strategy_manager.OutputInterpreter")
    def test_decide_action_with_llm_success(
        self, mock_interpreter_class, mock_prompt_builder_class, mock_brain_io_class
    ):
        """Test successful LLM decision-making flow"""
        # Set up mocks
        mock_brain_io = MagicMock()
        mock_prompt_builder = MagicMock()
        mock_interpreter = MagicMock()

        mock_brain_io_class.return_value = mock_brain_io
        mock_prompt_builder_class.return_value = mock_prompt_builder
        mock_interpreter_class.return_value = mock_interpreter

        # Mock LLM response
        llm_response = "I choose to sleep because Alice needs to restore energy."
        mock_brain_io.input_to_model.return_value = llm_response

        # Mock interpreter response
        mock_action = MagicMock()
        mock_action.name = "sleep"
        mock_interpreter.interpret_response.return_value = mock_action

        # Mock prompt generation
        mock_prompt_builder.generate_decision_prompt.return_value = "Test prompt"

        # Execute LLM decision
        result = self.strategy_manager.decide_action_with_llm(
            self.character, "morning", "sunny"
        )

        # Verify the flow
        self.assertIsNotNone(result)
        mock_prompt_builder.generate_decision_prompt.assert_called_once()
        mock_brain_io.send_request.assert_called_once()
        mock_interpreter.interpret_response.assert_called_once()

    @patch("tiny_strategy_manager.TinyBrainIO")
    def test_decide_action_with_llm_fallback(self, mock_brain_io_class):
        """Test fallback to utility-based decisions when LLM fails"""
        # Mock LLM failure
        mock_brain_io = MagicMock()
        mock_brain_io_class.return_value = mock_brain_io
        mock_brain_io.send_request.side_effect = Exception("LLM service unavailable")

        # Execute decision (should fallback to utility-based)
        with patch.object(
            self.strategy_manager, "get_daily_actions"
        ) as mock_get_actions:
            mock_action = MagicMock()
            mock_get_actions.return_value = [mock_action]

            result = self.strategy_manager.decide_action_with_llm(
                self.character, "morning", "sunny"
            )

            # Should return utility-based action
            self.assertEqual(result, [mock_action])
            mock_get_actions.assert_called_once()

    def test_prompt_builder_decision_prompt_generation(self):
        """Test PromptBuilder generates decision prompts correctly"""
        action_choices = [
            {"action": "eat", "reason": "Hunger level is 6/10"},
            {"action": "sleep", "reason": "Energy level is 4/10"},
            {"action": "work", "reason": "Need income, current wealth: $50"},
        ]

        prompt = self.prompt_builder.generate_decision_prompt(
            character_name="Alice",
            character_state="hungry and tired",
            current_time="morning",
            weather="sunny",
            action_choices=action_choices,
        )

        # Verify prompt contains key elements
        self.assertIn("Alice", prompt)
        self.assertIn("morning", prompt)
        self.assertIn("sunny", prompt)
        self.assertIn("eat", prompt)
        self.assertIn("sleep", prompt)
        self.assertIn("work", prompt)

    def test_output_interpreter_action_matching(self):
        """Test OutputInterpreter matches LLM responses to actions"""
        # Mock potential actions
        eat_action = MagicMock()
        eat_action.name = "eat"
        sleep_action = MagicMock()
        sleep_action.name = "sleep"
        potential_actions = [eat_action, sleep_action]

        # Test various LLM response formats
        test_responses = [
            "I should eat something to satisfy my hunger.",
            "Alice needs to sleep to restore energy.",
            "The best choice is to eat.",
            "Sleep is the most important action right now.",
        ]

        for response in test_responses:
            result = self.output_interpreter.interpret_response(
                response, potential_actions
            )
            self.assertIsNotNone(result)
            self.assertIn(result.name, ["eat", "sleep"])

    @patch("tiny_gameplay_controller.logging")
    def test_gameplay_controller_llm_integration(self, mock_logging):
        """Test GameplayController integrates LLM decisions correctly"""
        # Create a mock character with LLM enabled
        character = MockCharacter("Bob")
        character.use_llm_decisions = True

        # Mock the strategy manager
        mock_strategy_manager = MagicMock()
        mock_action = MagicMock()
        mock_strategy_manager.decide_action_with_llm.return_value = mock_action

        # Create gameplay controller
        controller = GameplayController()
        controller.strategy_manager = mock_strategy_manager

        # Mock other dependencies
        with patch.object(controller, "_validate_and_execute_action") as mock_execute:
            mock_execute.return_value = True

            # Execute character actions
            controller._execute_character_actions([character], "morning", "sunny")

            # Verify LLM decision was called
            mock_strategy_manager.decide_action_with_llm.assert_called_once_with(
                character, "morning", "sunny"
            )
            mock_execute.assert_called_once_with(character, mock_action)

    def test_character_llm_enable_method(self):
        """Test enabling LLM decisions for a character"""
        # This test verifies that we can enable LLM for characters
        character = MockCharacter("Charlie")
        self.assertFalse(
            hasattr(character, "use_llm_decisions") and character.use_llm_decisions
        )

        # Enable LLM decisions
        character.use_llm_decisions = True
        self.assertTrue(character.use_llm_decisions)

    @patch("tiny_strategy_manager.logging")
    def test_error_handling_in_llm_pipeline(self, mock_logging):
        """Test error handling throughout the LLM pipeline"""
        # Test various failure points
        with patch("tiny_strategy_manager.TinyBrainIO") as mock_brain_io_class:
            mock_brain_io = MagicMock()
            mock_brain_io_class.return_value = mock_brain_io

            # Test LLM request failure
            mock_brain_io.send_request.side_effect = Exception("Network error")

            with patch.object(
                self.strategy_manager, "get_daily_actions"
            ) as mock_get_actions:
                mock_action = MagicMock()
                mock_get_actions.return_value = [mock_action]

                result = self.strategy_manager.decide_action_with_llm(
                    self.character, "morning", "sunny"
                )

                # Should fallback gracefully
                self.assertEqual(result, mock_action)


if __name__ == "__main__":
    # Run the tests
    print("Starting LLM Integration Tests...")
    unittest.main(verbosity=2)
