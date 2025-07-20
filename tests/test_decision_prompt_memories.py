import sys
import unittest
from types import ModuleType
from unittest.mock import patch

# Stub modules to satisfy imports in tiny_prompt_builder
stub_tc = ModuleType('tiny_characters')
stub_tc.Character = object
stub_attr = ModuleType('attr')

class MockMemory:
    def __init__(self, description):
        self.description = description

class MockCharacter:
    def __init__(self):
        self.name = "Eve"
        self.job = "Farmer"
        self.recent_event = "harvest"
        self.wealth_money = 5
        self.health_status = 7
        self.hunger_level = 4
        self.energy = 6
        self.mental_health = 6
        self.social_wellbeing = 5
        self.long_term_goal = "grow the best crops"
        self.motives = None

    def evaluate_goals(self):
        return []

class DecisionPromptMemoryTests(unittest.TestCase):
    def test_memories_in_prompt(self):
        with patch.dict(sys.modules, {"tiny_characters": stub_tc, "attr": stub_attr}):
            import tiny_prompt_builder
            # ensure descriptor dictionaries have defaults to avoid KeyError
            tiny_prompt_builder.descriptors.event_recent.setdefault("default", [""])
            tiny_prompt_builder.descriptors.financial_situation.setdefault("default", [""])
            PromptBuilder = tiny_prompt_builder.PromptBuilder

            char = MockCharacter()
            builder = PromptBuilder(char)
            memories = [MockMemory("won a pie contest"), MockMemory("lost keys at market")]
            prompt = builder.generate_decision_prompt(
                time="noon",
                weather="sunny",
                action_choices=["1. Eat lunch"],
                memories=memories,
            )

        self.assertIn("won a pie contest", prompt)
        self.assertIn("lost keys at market", prompt)

if __name__ == "__main__":
    unittest.main()
