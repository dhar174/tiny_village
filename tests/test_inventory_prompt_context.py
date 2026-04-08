import sys
import types
import unittest


tc_stub = types.ModuleType("tiny_characters")


class MockInventory:
    def to_prompt_context(self):
        return {
            "summary": {
                "total_items": 4,
                "total_stacks": 2,
                "total_value": 9,
                "total_weight": 3,
                "counts_by_type": {"food": 3, "tools": 1},
            },
            "items_by_type": {
                "food": [{"name": "Apple", "quantity": 3, "item_type": "food"}],
                "tools": [{"name": "Hammer", "quantity": 1, "item_type": "tools"}],
            },
            "all_items": [
                {"name": "Apple", "quantity": 3, "item_type": "food"},
                {"name": "Hammer", "quantity": 1, "item_type": "tools"},
            ],
            "surplus_items": [{"name": "Apple", "quantity": 3, "item_type": "food"}],
            "trade_candidates": [
                {"name": "Apple", "quantity": 3, "item_type": "food"},
                {"name": "Hammer", "quantity": 1, "item_type": "tools"},
            ],
            "drop_candidates": [{"name": "Apple", "quantity": 3, "item_type": "food"}],
        }


class MockCharacter:
    def __init__(self):
        self.name = "Iris"
        self.job = "Engineer"
        self.health_status = 8
        self.hunger_level = 5
        self.mental_health = 7
        self.social_wellbeing = 6
        self.energy = 8
        self.wealth_money = 50
        self.long_term_goal = "build resilient systems"
        self.recent_event = "default"
        self.personality_traits = {"extraversion": 60, "conscientiousness": 80}
        self.inventory = MockInventory()
        self.motives = None

    def evaluate_goals(self):
        return []


tc_stub.Character = MockCharacter
sys.modules["tiny_characters"] = tc_stub

from tiny_prompt_builder import PromptBuilder


class TestInventoryPromptContext(unittest.TestCase):
    def test_decision_prompt_includes_inventory_trade_and_drop_context(self):
        prompt = PromptBuilder(MockCharacter()).generate_decision_prompt(
            "morning",
            "sunny",
            action_choices=["1. Trade goods (Utility: 0.8)"],
            include_memory_integration=False,
            include_conversation_context=False,
            include_few_shot_examples=False,
        )

        self.assertIn("Inventory overview: 4 total items across 2 stacks", prompt)
        self.assertIn("Potential trade items: Apple x3, Hammer x1.", prompt)
        self.assertIn("Potential drop candidates: Apple x3.", prompt)


if __name__ == "__main__":
    unittest.main()
