import importlib
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


@contextmanager
def temporary_test_path():
    previous_sys_path = list(sys.path)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    try:
        yield
    finally:
        sys.path[:] = previous_sys_path


def load_realistic_mock_character():
    with temporary_test_path():
        mock_character = importlib.import_module("mock_character")
        return mock_character.MockCharacter


def build_character(mock_character_class):
    return mock_character_class(
        name="Tom",
        age=31,
        job="Carpenter",
        recent_event="market day",
        wealth_money=5,
        health_status=8,
        hunger_level=4,
        energy=6,
        mental_health=7,
        social_wellbeing=6,
    )


def render_prompt(prompt_factory):
    realistic_mock_character = load_realistic_mock_character()
    stub_tiny_characters = ModuleType("tiny_characters")
    stub_tiny_characters.Character = realistic_mock_character
    stub_attr = ModuleType("attr")
    previous_prompt_builder = sys.modules.get("tiny_prompt_builder")

    try:
        with patch.dict(
            sys.modules,
            {"tiny_characters": stub_tiny_characters, "attr": stub_attr},
        ):
            sys.modules.pop("tiny_prompt_builder", None)
            tiny_prompt_builder = importlib.import_module("tiny_prompt_builder")
            builder = tiny_prompt_builder.PromptBuilder(
                build_character(realistic_mock_character)
            )
            return prompt_factory(builder)
    finally:
        if previous_prompt_builder is not None:
            sys.modules["tiny_prompt_builder"] = previous_prompt_builder
        else:
            sys.modules.pop("tiny_prompt_builder", None)


class ScenarioPromptTests(unittest.TestCase):
    def test_social_prompt_matches_expected_structure(self):
        actions = ["greet villager", "share meal"]

        prompt = render_prompt(
            lambda builder: builder.generate_social_interaction_prompt(actions)
        )

        expected = (
            "<|system|>"
            "You are Tom, a Carpenter."
            "<|user|>"
            "You are about to interact with another villager."
            " Current state: Health 8/10, Hunger 4/10, Energy 6.0/10."
            "\nAvailable actions:\n"
            "greet villager\n"
            "share meal\n"
            "</s><|assistant|>"
            "Tom, I choose "
        )

        self.assertEqual(prompt, expected)

    def test_travel_prompt_matches_expected_structure(self):
        actions = ["pack supplies", "set off"]

        prompt = render_prompt(
            lambda builder: builder.generate_travel_prompt("Riverside", actions)
        )

        expected = (
            "<|system|>"
            "You are Tom, a Carpenter."
            "<|user|>"
            "You are considering travelling to Riverside."
            " Current state: Health 8/10, Hunger 4/10, Energy 6.0/10."
            "\nAvailable actions:\n"
            "pack supplies\n"
            "set off\n"
            "</s><|assistant|>"
            "Tom, I choose "
        )

        self.assertEqual(prompt, expected)


if __name__ == "__main__":
    unittest.main()
