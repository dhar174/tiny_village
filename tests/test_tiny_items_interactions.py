import importlib
import sys
import types
import unittest
from unittest.mock import patch


MODULE_PATCHER = None
ORIGINAL_MODULES = {}


def setUpModule():
    global MODULE_PATCHER
    global ORIGINAL_MODULES

    for module_name in ("actions", "tiny_locations", "tiny_items"):
        if module_name in sys.modules:
            ORIGINAL_MODULES[module_name] = sys.modules.pop(module_name)

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

    MODULE_PATCHER = patch.dict(sys.modules, {"actions": fake_actions})
    MODULE_PATCHER.start()


def tearDownModule():
    if MODULE_PATCHER is not None:
        MODULE_PATCHER.stop()

    for module_name in ("tiny_locations", "tiny_items"):
        sys.modules.pop(module_name, None)

    for module_name, original_module in ORIGINAL_MODULES.items():
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
