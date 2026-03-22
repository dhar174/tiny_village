#!/usr/bin/env python3
"""Behavior-focused tests for the StrategyManager LLM pipeline."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from actions import Action
import tiny_strategy_manager


class ScenarioCharacter:
    def __init__(self):
        self.name = "TestCharacter"
        self.id = "TestCharacter"
        self.use_llm_decisions = True
        self.hunger_level = 6.0
        self.energy = 4.0
        self.wealth_money = 25.0
        self.social_wellbeing = 7.0
        self.mental_health = 6.0
        self.health_status = 5.0
        self.job = "farmer"
        self.location = SimpleNamespace(name="Home")
        self.inventory = SimpleNamespace(get_food_items=lambda: [])


class RecordingPromptBuilder:
    last_instance = None

    def __init__(self, character):
        self.character = character
        self.calls = []
        RecordingPromptBuilder.last_instance = self

    def generate_decision_prompt(
        self, time, weather, action_choices, character_state_dict
    ):
        self.calls.append(
            {
                "time": time,
                "weather": weather,
                "action_choices": list(action_choices),
                "character_state_dict": dict(character_state_dict),
            }
        )
        return (
            f"Prompt for {self.character.name} at {time} in {weather}\n"
            + "\n".join(action_choices)
        )


class RecordingBrainIO:
    last_instance = None

    def __init__(self, model_name, model_special_args=None):
        self.model_name = model_name
        self.prompts = []
        RecordingBrainIO.last_instance = self

    def input_to_model(self, prompts, reset_model=True):
        batch = prompts if isinstance(prompts, list) else [prompts]
        self.prompts.extend(batch)
        return [("I choose Sleep", "0.01") for _ in batch]


class RecordingOutputInterpreter:
    last_instance = None

    def __init__(self):
        self.calls = []
        RecordingOutputInterpreter.last_instance = self

    def interpret_response(self, llm_response_text, character, potential_actions):
        self.calls.append(
            {
                "response": llm_response_text,
                "character": character.name,
                "potential_actions": [action.name for action in potential_actions],
            }
        )
        for action in potential_actions:
            if action.name == "Sleep":
                return [action]
        return [potential_actions[0]]


class MinimalPlanner:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager


class TestLLMIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        RecordingPromptBuilder.last_instance = None
        RecordingBrainIO.last_instance = None
        RecordingOutputInterpreter.last_instance = None
        self.character = ScenarioCharacter()
        self.actions = [
            Action(name="Eat", preconditions=[], effects=[], cost=0.1),
            Action(name="Sleep", preconditions=[], effects=[], cost=0.2),
        ]

    def _patched_components(self):
        return patch.multiple(
            tiny_strategy_manager,
            PromptBuilder=RecordingPromptBuilder,
            TinyBrainIO=RecordingBrainIO,
            OutputInterpreter=RecordingOutputInterpreter,
            GOAPPlanner=MinimalPlanner,
        )

    def test_decide_action_with_llm_passes_prompt_response_and_actions_through_pipeline(
        self,
    ):
        with self._patched_components():
            manager = tiny_strategy_manager.StrategyManager(
                use_llm=True, model_name="fake-model"
            )
            result = manager.decide_action_with_llm(
                self.character,
                time="morning",
                weather="sunny",
                potential_actions=self.actions,
            )

        prompt_call = RecordingPromptBuilder.last_instance.calls[-1]
        interpreter_call = RecordingOutputInterpreter.last_instance.calls[-1]
        sent_prompt = RecordingBrainIO.last_instance.prompts[-1]

        self.assertEqual([action.name for action in result], ["Sleep"])
        self.assertEqual(prompt_call["time"], "morning")
        self.assertEqual(prompt_call["weather"], "sunny")
        self.assertIn("1. Eat", prompt_call["action_choices"][0])
        self.assertIn("2. Sleep", prompt_call["action_choices"][1])
        self.assertIn(self.character.name, sent_prompt)
        self.assertEqual(interpreter_call["response"], "I choose Sleep")
        self.assertEqual(interpreter_call["potential_actions"], ["Eat", "Sleep"])

    def test_update_strategy_new_day_uses_real_llm_pipeline_for_llm_enabled_character(
        self,
    ):
        with self._patched_components():
            manager = tiny_strategy_manager.StrategyManager(
                use_llm=True, model_name="fake-model"
            )
            self.character.energy = 2.0
            manager.enable_llm_for_character(self.character)
            plans = manager.update_strategy(
                [SimpleNamespace(type="new_day")],
                self.character,
            )

        self.assertIn("testcharacter", plans)
        self.assertEqual([action.name for action in plans["testcharacter"]], ["Sleep"])
        self.assertEqual(len(RecordingBrainIO.last_instance.prompts), 1)

    def test_character_llm_flag_integration_tracks_enabled_characters(self):
        with self._patched_components():
            manager = tiny_strategy_manager.StrategyManager(
                use_llm=True, model_name="fake-model"
            )

        self.character.use_llm_decisions = False
        manager.disable_llm_for_character(self.character)
        self.assertNotIn(self.character.name, manager._characters_using_llm)

        self.character.use_llm_decisions = True
        manager.enable_llm_for_character(self.character)
        self.assertIn(self.character.name, manager._characters_using_llm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
