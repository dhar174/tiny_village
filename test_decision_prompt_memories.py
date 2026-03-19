import sys
import unittest
from types import ModuleType


stub_tiny_characters = ModuleType("tiny_characters")


class StubCharacter:
    pass


stub_tiny_characters.Character = StubCharacter


class Memory:
    def __init__(self, description):
        self.description = description


class Character:
    def __init__(self):
        self.name = "Eve"
        self.job = "Farmer"
        self.recent_event = "default"
        self.wealth_money = 5
        self.health_status = 7
        self.hunger_level = 4
        self.energy = 6
        self.mental_health = 6
        self.social_wellbeing = 5
        self.long_term_goal = "grow the best crops"
        self.personality_traits = {}
        self.motives = None

    def evaluate_goals(self):
        return []


class DecisionPromptMemoryTests(unittest.TestCase):
    def test_memories_in_prompt(self):
        original_module = sys.modules.get("tiny_characters")
        sys.modules["tiny_characters"] = stub_tiny_characters
        try:
            if "tiny_prompt_builder" in sys.modules:
                del sys.modules["tiny_prompt_builder"]
            import tiny_prompt_builder

            builder = tiny_prompt_builder.PromptBuilder(Character())
            builder.context_manager.assemble_complete_context = lambda *args, **kwargs: {
                "character": {"basic_info": {"name": "Eve", "job": "Farmer"}},
                "goals": {"active_goals": [], "needs_priorities": {}},
                "memories": [],
            }

            prompt = builder.generate_decision_prompt(
                time="noon",
                weather="sunny",
                action_choices=["1. Eat lunch"],
                memories=[
                    Memory("won a pie contest"),
                    Memory("lost keys at market"),
                ],
                include_conversation_context=False,
                include_few_shot_examples=False,
                include_memory_integration=False,
                output_format="text",
            )
        finally:
            if original_module is None:
                sys.modules.pop("tiny_characters", None)
            else:
                sys.modules["tiny_characters"] = original_module
            sys.modules.pop("tiny_prompt_builder", None)

        self.assertIn("won a pie contest", prompt)
        self.assertIn("lost keys at market", prompt)


if __name__ == "__main__":
    unittest.main()
