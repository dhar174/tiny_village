import unittest
import sys
import types


actions_stub = types.ModuleType("actions")


class DummyAction:
    def __init__(self, *args, **kwargs):
        pass


class DummyActionSystem:
    def instantiate_conditions(self, conditions):
        return conditions


actions_stub.Action = DummyAction
actions_stub.ActionSystem = DummyActionSystem
sys.modules["actions"] = actions_stub

tiny_locations_stub = types.ModuleType("tiny_locations")


class DummyLocation:
    def __init__(self, *args, **kwargs):
        self.coordinates_location = (0, 0)

    def get_coordinates(self):
        return self.coordinates_location

    def set_coordinates(self, coordinates):
        self.coordinates_location = coordinates

    def to_dict(self):
        return {"coordinates_location": self.coordinates_location}

    def __eq__(self, other):
        return isinstance(other, DummyLocation) and self.coordinates_location == other.coordinates_location


tiny_locations_stub.Location = DummyLocation
sys.modules["tiny_locations"] = tiny_locations_stub

tiny_types_stub = types.ModuleType("tiny_types")
tiny_types_stub.Action = DummyAction
tiny_types_stub.ActionSystem = DummyActionSystem
tiny_types_stub.GraphManager = object
sys.modules["tiny_types"] = tiny_types_stub

from tiny_items import ItemInventory, ItemObject


def build_item(name, quantity, item_type="misc", value=1, weight=1):
    return ItemObject(
        name=name,
        description=f"{name} description",
        value=value,
        weight=weight,
        quantity=quantity,
        item_type=item_type,
    )


class TestInventoryManagement(unittest.TestCase):
    def test_check_has_item_by_type_accepts_strings_and_lists(self):
        inventory = ItemInventory(
            food_items=[build_item("Apple", 2, item_type="food")],
            misc_items=[build_item("Rope", 1, item_type="misc")],
        )

        self.assertTrue(inventory.check_has_item_by_type("food", amount=2))
        self.assertTrue(inventory.check_has_item_by_type(["food"], amount=2))
        self.assertTrue(inventory.check_has_item_by_type(["food", "misc"], amount=3))
        self.assertFalse(inventory.check_has_item_by_type(["medicine"], amount=1))

    def test_transfer_item_to_moves_quantity_between_inventories(self):
        source = ItemInventory(
            food_items=[build_item("Apple", 3, item_type="food", value=2, weight=1)]
        )
        target = ItemInventory()

        source.transfer_item_to(build_item("Apple", 2, item_type="food", value=2, weight=1), target)

        self.assertEqual(source.count_total_items_by_name("Apple"), 1)
        self.assertEqual(target.count_total_items_by_name("Apple"), 2)

    def test_transfer_item_to_raises_when_quantity_is_unavailable(self):
        source = ItemInventory(
            food_items=[build_item("Apple", 1, item_type="food", value=2, weight=1)]
        )
        target = ItemInventory()

        with self.assertRaises(ValueError):
            source.transfer_item_to(
                build_item("Apple", 2, item_type="food", value=2, weight=1), target
            )

    def test_prompt_context_exposes_trade_and_drop_candidates(self):
        inventory = ItemInventory(
            food_items=[build_item("Apple", 3, item_type="food", value=2, weight=1)],
            tools_items=[build_item("Hammer", 1, item_type="tools", value=5, weight=2)],
            misc_items=[build_item("Stone", 1, item_type="misc", value=0, weight=1)],
        )

        context = inventory.to_prompt_context()

        self.assertEqual(context["summary"]["total_items"], 5)
        self.assertEqual(context["summary"]["counts_by_type"]["food"], 3)
        self.assertEqual(
            [item["name"] for item in context["trade_candidates"]],
            ["Apple", "Hammer", "Stone"],
        )
        self.assertEqual(
            [item["name"] for item in context["drop_candidates"]],
            ["Apple", "Stone"],
        )


if __name__ == "__main__":
    unittest.main()
