#!/usr/bin/env python3
"""Focused integration tests for the prompt/brain/output LLM stack."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockInventory:
    def count_food_items_total(self):
        return 3

    def count_food_calories_total(self):
        return 150

    def get_food_items(self):
        return []


class MockMotive:
    def __init__(self, score):
        self.score = score


class MockMotives:
    def to_dict(self):
        return {
            "health": MockMotive(5),
            "wealth": MockMotive(6),
            "social_wellbeing": MockMotive(4),
            "job_performance": MockMotive(6),
        }


class MockCharacter:
    """Concrete character double that exposes the interfaces PromptBuilder uses."""

    def __init__(self, name="TestCharacter"):
        self.id = name
        self.name = name
        self.job = "Engineer"
        self.health_status = 7
        self.hunger_level = 4
        self.mental_health = 6
        self.social_wellbeing = 5
        self.energy = 8
        self.wealth_money = 50
        self.recent_event = "learning"
        self.long_term_goal = "career_advancement"
        self.personality_traits = {"extraversion": 60, "conscientiousness": 70}
        self.motives = MockMotives()
        self.inventory = MockInventory()
        self.use_llm_decisions = True
        self.location = None

    def get_hunger_level(self):
        return self.hunger_level

    def get_health_status(self):
        return self.health_status

    def get_mental_health(self):
        return self.mental_health

    def get_social_wellbeing(self):
        return self.social_wellbeing

    def get_wealth_money(self):
        return self.wealth_money

    def get_wealth(self):
        return self.wealth_money

    def get_happiness(self):
        return 5

    def get_shelter(self):
        return 5

    def get_stability(self):
        return 5

    def get_luxury(self):
        return 3

    def get_hope(self):
        return 6

    def get_success(self):
        return 5

    def get_control(self):
        return 5

    def get_job_performance(self):
        return 6

    def get_beauty(self):
        return 5

    def get_community(self):
        return 4

    def get_material_goods(self):
        return 4

    def get_friendship_grid(self):
        return 5

    def get_long_term_goal(self):
        return self.long_term_goal

    def get_inventory(self):
        return self.inventory


class TestLLMInterfaceIntegration(unittest.TestCase):
    def setUp(self):
        self.character = MockCharacter()

    def test_modules_import_with_optional_dependency_fallbacks(self):
        import tiny_memories
        import tiny_prompt_builder
        import tiny_brain_io
        import tiny_output_interpreter
        from tiny_strategy_manager import StrategyManager

        self.assertTrue(hasattr(tiny_memories, "MemoryManager"))
        self.assertTrue(hasattr(tiny_prompt_builder, "PromptBuilder"))
        self.assertTrue(hasattr(tiny_brain_io, "TinyBrainIO"))
        self.assertTrue(hasattr(tiny_output_interpreter, "OutputInterpreter"))
        self.assertIsNotNone(StrategyManager(use_llm=False))

    def test_prompt_builder_decision_prompt_contains_actionable_context(self):
        from tiny_prompt_builder import PromptBuilder

        prompt_builder = PromptBuilder(self.character)
        prompt = prompt_builder.generate_decision_prompt(
            time="morning",
            weather="sunny",
            action_choices=["1. Rest", "2. Work as Engineer"],
            character_state_dict={"energy": 0.4, "money": 50.0},
            include_conversation_context=False,
            include_few_shot_examples=False,
            include_memory_integration=False,
        )

        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn(self.character.name, prompt)
        self.assertIn("CURRENT ACTIVE GOALS", prompt)
        self.assertIn("PRESSING NEEDS", prompt)
        self.assertIn("Available actions:", prompt)
        self.assertIn("1. Rest", prompt)
        self.assertIn("2. Work as Engineer", prompt)
        self.assertIn("Additional state:", prompt)
        self.assertIn("Energy: 0.4", prompt)
        self.assertIn("morning", prompt)
        self.assertTrue(prompt.endswith(f"{self.character.name}, I choose "))

    def test_brain_io_no_model_fallback_preserves_each_prompt(self):
        from tiny_brain_io import TinyBrainIO

        with patch.object(TinyBrainIO, "load_model", autospec=True, return_value=None):
            brain_io = TinyBrainIO("test-model")

        brain_io.model = None
        brain_io.tokenizer = None

        prompts = ["Plan breakfast", "Visit the market"]
        results = brain_io.input_to_model(prompts)

        self.assertEqual(len(results), len(prompts))
        for prompt, (response, elapsed) in zip(prompts, results):
            self.assertIn(prompt, response)
            self.assertEqual(elapsed, "0.0")
            self.assertTrue(response.startswith("Model not available:"))

    def test_output_interpreter_selects_matching_potential_action(self):
        from actions import Action
        from tiny_output_interpreter import OutputInterpreter

        interpreter = OutputInterpreter()
        actions = [
            Action(name="Rest", preconditions=[], effects=[], cost=0.1),
            Action(name="Sleep", preconditions=[], effects=[], cost=0.2),
        ]

        selected = interpreter.interpret_response(
            '{"action": "Sleep", "parameters": {}}',
            self.character,
            actions,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "Sleep")

    def test_strategy_manager_routes_complex_decisions_through_llm_path(self):
        from tiny_brain_io import TinyBrainIO
        from tiny_strategy_manager import StrategyManager

        with patch.object(TinyBrainIO, "load_model", autospec=True, return_value=None):
            manager = StrategyManager(use_llm=True, model_name="test-model")

        manager.brain_io.model = None
        manager.brain_io.tokenizer = None

        should_use_llm = manager.should_use_llm_for_decision(
            self.character, {"social_complexity": 0.9}
        )
        actions = manager.get_enhanced_daily_actions(
            self.character,
            time="morning",
            weather="sunny",
            situation_context={"force_llm": True},
        )

        self.assertTrue(manager.use_llm)
        self.assertTrue(should_use_llm)
        self.assertGreater(len(actions), 0)
        self.assertTrue(hasattr(actions[0], "name"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
