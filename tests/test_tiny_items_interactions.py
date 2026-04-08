import importlib
import sys
import types
import unittest
from unittest.mock import patch


_module_patcher = None
_original_modules = {}


def setUpModule():
    global _module_patcher
    global _original_modules

    for module_name in ("actions", "tiny_locations", "tiny_items"):
        if module_name in sys.modules:
            _original_modules[module_name] = sys.modules.pop(module_name)

    fake_actions = types.ModuleType("actions")

    class FakeAction:
        def __init__(self, name, preconditions, effects, cost=0, **kwargs):
            self.name = name
            self.preconditions = preconditions
            self.effects = effects
            self.cost = cost

    class FakeActionSystem:
        def instantiate_conditions(self, conditions_list):
            return conditions_list

    class FakeState:
        pass

    fake_actions.Action = FakeAction
    fake_actions.ActionSystem = FakeActionSystem
    fake_actions.State = FakeState

    _module_patcher = patch.dict(sys.modules, {"actions": fake_actions})
    _module_patcher.start()


def tearDownModule():
    if _module_patcher is not None:
        _module_patcher.stop()

    for module_name in ("tiny_locations", "tiny_items"):
        sys.modules.pop(module_name, None)

    for module_name, original_module in _original_modules.items():
        sys.modules[module_name] = original_module


class TestItemInteractions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiny_items = importlib.import_module("tiny_items")
        cls.fake_action_type = sys.modules["actions"].Action

    def test_tool_items_have_equip_interaction(self):
        item = self.tiny_items.ItemObject(
            name="Hammer",
            description="A sturdy tool",
            value=10,
            weight=2,
            quantity=1,
            item_type="tool",
        )

        self.assertEqual(
            [action.name for action in item.get_possible_interactions()],
            ["Equip Tool"],
        )

    def test_armor_items_have_wear_interaction_case_insensitive(self):
        item = self.tiny_items.ItemObject(
            name="Chainmail",
            description="Protective armor",
            value=25,
            weight=8,
            quantity=1,
            item_type="Armor",
        )

        self.assertEqual(
            [action.name for action in item.get_possible_interactions()],
            ["Wear Clothing"],
        )

    def test_resource_items_can_be_used_for_crafting(self):
        item = self.tiny_items.ItemObject(
            name="Wood",
            description="Crafting resource",
            value=3,
            weight=1,
            quantity=4,
            item_type="misc",
            item_subtype="resource",
        )

        self.assertEqual(
            [action.name for action in item.get_possible_interactions()],
            ["Use Resource for Crafting"],
        )

    def test_explicit_item_interactions_are_preserved(self):
        custom_interaction = self.fake_action_type(
            "Polish Armor",
            [],
            [],
            cost=0,
        )
        item = self.tiny_items.ItemObject(
            name="Breastplate",
            description="Shiny armor",
            value=40,
            weight=6,
            quantity=1,
            item_type="Armor",
            possible_interactions=[custom_interaction],
        )

        self.assertEqual(item.get_possible_interactions(), [custom_interaction])

