import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

# Create stub modules to satisfy imports in tiny_prompt_builder
stub_tc = ModuleType('tiny_characters')
stub_tc.Character = object
stub_attr = ModuleType('attr')

class MockCharacter:
    def __init__(self):
        self.name = "Eve"
        self.job = "Farmer"
        self.recent_event = "outbreak"
        self.wealth_money = 10
        self.health_status = 7
        self.hunger_level = 5
        self.energy = 5
        self.mental_health = 6
        self.social_wellbeing = 6

class CrisisPromptTests(unittest.TestCase):
    def test_prompt_contains_description_and_assistant_cue(self):
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            tiny_prompt_builder.descriptors.event_recent.setdefault("default", [""])
            tiny_prompt_builder.descriptors.financial_situation.setdefault("default", [""])
            PromptBuilder = tiny_prompt_builder.PromptBuilder
            char = MockCharacter()
            builder = PromptBuilder(char)
            prompt = builder.generate_crisis_response_prompt("barn fire", urgency="high")
        self.assertIn("barn fire", prompt)
        self.assertTrue(prompt.strip().endswith("<|assistant|>"))

if __name__ == "__main__":
    unittest.main()
