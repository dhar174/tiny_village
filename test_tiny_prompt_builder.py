import logging
import unittest
from unittest.mock import MagicMock
from venv import logger

# Assuming tc and ActionOptions are defined elsewhere and imported correctly
from tiny_characters import Character
from tiny_locations import Location
from tiny_prompt_builder import PromptBuilder, DescriptorMatrices
from actions import ActionSystem
import tiny_time_manager
from tiny_graph_manager import GraphManager

logging.basicConfig(level=logging.DEBUG)


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        # Create a mock character
        self.graph_manager = GraphManager()
        alice = Character(
            name="Alice",
            age=25,
            pronouns="she/her",
            job="Waitress",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=ActionSystem(),
            gametime_manager=tiny_time_manager,
            location=Location("Alice", 0, 0, 1, 1, ActionSystem()),
            graph_manager=self.graph_manager,
        )
        bob = Character(
            name="Bob",
            age=25,
            pronouns="he/him",
            job="Mayor",
            health_status=10,
            hunger_level=2,
            wealth_money=10,
            mental_health=8,
            social_wellbeing=8,
            job_performance=20,
            community=5,
            friendship_grid=[],
            recent_event="new job",
            goal="Save for college",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=ActionSystem(),
            gametime_manager=tiny_time_manager,
            location=Location("Bob", 0, 0, 1, 1, ActionSystem()),
            graph_manager=self.graph_manager,
        )
        self.character = Character(
            "Emily",
            21,
            "she/her",
            "software engineer",
            10,
            2,
            10,
            8,
            8,
            20,
            5,
            [],
            "new job",
            "Work for OpenAI",
            personality_traits={
                "extraversion": 50,
                "openness": 80,
                "conscientiousness": 70,
                "agreeableness": 60,
                "neuroticism": 30,
            },
            action_system=ActionSystem(),
            gametime_manager=tiny_time_manager,
            location=Location("Emily", 0, 0, 1, 1, ActionSystem()),
            graph_manager=self.graph_manager,
        )

        # Mock the NeedsPriorities class
        self.mock_needs_priorities = MagicMock()
        self.mock_needs_priorities.calculate_needs_priorities.return_value = {
            "need1": 10,
            "need2": 20,
            "need3": 30,
        }
        # Mock the ActionOptions class
        self.mock_action_options = MagicMock()
        self.prompt_builder = PromptBuilder(self.character)
        self.prompt_builder.needs_priorities_func = self.mock_needs_priorities
        self.prompt_builder.action_options = self.mock_action_options

    def test_initialization(self):
        self.assertEqual(self.prompt_builder.character, self.character)
        self.assertIsInstance(self.prompt_builder.needs_priorities_func, MagicMock)
        self.assertIsInstance(self.prompt_builder.action_options, MagicMock)

    def test_calculate_needs_priorities(self):
        self.prompt_builder.calculate_needs_priorities()
        self.mock_needs_priorities.calculate_needs_priorities.assert_called_once_with(
            self.character
        )
        self.assertEqual(
            self.prompt_builder.needs_priorities,
            {
                "need1": 10,
                "need2": 20,
                "need3": 30,
            },
        )

    def test_generate_prompt(self):
        # Mock the DescriptorMatrices class
        mock_descriptor_matrices = MagicMock()
        mock_descriptor_matrices.generate.return_value = "Generated Prompt"
        self.prompt_builder.descriptor_matrices = mock_descriptor_matrices

        prompt = self.prompt_builder.generate_prompt()
        mock_descriptor_matrices.generate.assert_called_once()
        self.assertEqual(prompt, "Generated Prompt")

    def test_get_action_options(self):
        self.prompt_builder.get_action_options()
        self.mock_action_options.get_options.assert_called_once_with(self.character)


if __name__ == "__main__":
    unittest.main()
