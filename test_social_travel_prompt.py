import sys
import unittest
from types import ModuleType
from unittest.mock import patch

# Import proper MockCharacter instead of using object
sys.path.insert(0, 'tests')
from mock_character import MockCharacter

# Stub modules required by tiny_prompt_builder
stub_tc = ModuleType('tiny_characters')
stub_tc.Character = MockCharacter
stub_attr = ModuleType('attr')

class MockCharacter:
    def __init__(self):
        self.name = "Tom"
        self.job = "Carpenter"
        self.recent_event = "market day"
        self.wealth_money = 5
        self.health_status = 8
        self.hunger_level = 4
        self.energy = 6
        self.mental_health = 7
        self.social_wellbeing = 6

class ScenarioPromptTests(unittest.TestCase):
    def test_social_prompt_includes_actions(self):
        actions = ["greet villager", "share meal"]
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            PromptBuilder = tiny_prompt_builder.PromptBuilder
            builder = PromptBuilder(MockCharacter())
            prompt = builder.generate_social_interaction_prompt(actions)
        self.assertIn("greet villager", prompt)
        self.assertIn("<|assistant|>", prompt)

    def test_travel_prompt_includes_destination(self):
        actions = ["pack supplies", "set off"]
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            PromptBuilder = tiny_prompt_builder.PromptBuilder
            builder = PromptBuilder(MockCharacter())
            prompt = builder.generate_travel_prompt("Riverside", actions)
        self.assertIn("Riverside", prompt)
        self.assertIn("<|assistant|>", prompt)

if __name__ == "__main__":
    unittest.main()
