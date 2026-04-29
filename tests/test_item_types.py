import unittest
from types import SimpleNamespace
from unittest.mock import patch
import importlib as stdlib_importlib

import tiny_items
from tiny_items import (
    ClothingItem,
    Door,
    ItemInventory,
    MedicineItem,
    QuestItem,
    ResourceItem,
    ToolItem,
    WeaponItem,
)


class FakeActionSystem:
    def instantiate_conditions(self, conditions):
        return list(conditions or [])


class FakeAction:
    def __init__(self, name, preconditions, effects, cost, **kwargs):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects
        self.cost = cost


class FakeLocation:
    def __init__(
        self,
        name,
        x,
        y,
        width,
        height,
        action_system,
        security=0,
        threat_level=0,
        popularity=0,
    ):
        self.name = name
        self.coordinates_location = (x, y)
        self.action_system = action_system

    def get_coordinates(self):
        return self.coordinates_location

    def set_coordinates(self, coordinates):
        self.coordinates_location = tuple(coordinates)

    def to_dict(self):
        return {
            "name": self.name,
            "coordinates_location": self.coordinates_location,
        }

    def __eq__(self, other):
        return isinstance(other, FakeLocation) and self.to_dict() == other.to_dict()


class TestItemTypes(unittest.TestCase):
    def setUp(self):
        self.action_system = FakeActionSystem()
        self.location_patch = patch.object(tiny_items, "Location", FakeLocation)
        self.import_patch = patch.object(
            tiny_items.importlib,
            "import_module",
            side_effect=self._fake_import_module,
        )
        self.location_patch.start()
        self.import_patch.start()

    def tearDown(self):
        self.import_patch.stop()
        self.location_patch.stop()

    def _fake_import_module(self, module_name):
        if module_name == "actions":
            return SimpleNamespace(Action=FakeAction, ActionSystem=FakeActionSystem)
        return stdlib_importlib.import_module(module_name)

    def test_specialized_items_define_specific_attributes_and_interactions(self):
        clothing = ClothingItem(
            "Warm Coat",
            "Keeps villagers warm",
            15,
            2,
            1,
            action_system=self.action_system,
            clothing_type="coat",
            insulation=8,
            durability=90,
        )
        tool = ToolItem(
            "Iron Hammer",
            "Useful for repairs",
            20,
            4,
            1,
            action_system=self.action_system,
            tool_type="hammer",
            durability=85,
            efficiency=1.5,
        )
        resource = ResourceItem(
            "Oak Lumber",
            "Construction material",
            8,
            5,
            3,
            action_system=self.action_system,
            resource_type="wood",
            renewable=True,
            quality=2,
        )
        medicine = MedicineItem(
            "Herbal Remedy",
            "Improves recovery",
            12,
            1,
            2,
            action_system=self.action_system,
            potency=6,
            cure_type="healing",
        )
        weapon = WeaponItem(
            "Training Sword",
            "Simple practice weapon",
            18,
            3,
            1,
            action_system=self.action_system,
            damage=7,
            weapon_type="sword",
            durability=70,
        )
        quest_item = QuestItem(
            "Royal Seal",
            "Needed to finish the courier quest",
            0,
            1,
            1,
            action_system=self.action_system,
            quest_name="Courier Duty",
            objective="Deliver the seal to the mayor",
            key_item=True,
        )

        self.assertEqual(clothing.item_type, "clothing")
        self.assertEqual(clothing.get_type_specific_attributes()["clothing_type"], "coat")
        self.assertEqual(clothing.possible_interactions[0].name, "Wear Clothing")

        self.assertEqual(tool.item_type, "tools")
        self.assertEqual(tool.get_type_specific_attributes()["tool_type"], "hammer")
        self.assertEqual(tool.possible_interactions[0].name, "Use Tool")

        self.assertEqual(resource.item_type, "resources")
        self.assertTrue(resource.get_type_specific_attributes()["renewable"])
        self.assertEqual(resource.possible_interactions[0].name, "Gather Resource")

        self.assertEqual(medicine.item_type, "medicine")
        self.assertEqual(medicine.get_type_specific_attributes()["potency"], 6)
        self.assertEqual(medicine.possible_interactions[0].name, "Use Medicine")

        self.assertEqual(weapon.item_type, "weapons")
        self.assertEqual(weapon.get_type_specific_attributes()["damage"], 7)
        self.assertEqual(weapon.possible_interactions[0].name, "Wield Weapon")

        self.assertEqual(quest_item.item_type, "quest")
        self.assertTrue(quest_item.get_type_specific_attributes()["key_item"])
        self.assertEqual(quest_item.possible_interactions[0].name, "Inspect Quest Item")

    def test_inventory_supports_new_categories_and_aliases(self):
        inventory = ItemInventory()
        tool = ToolItem(
            "Wood Axe",
            "Cuts lumber",
            10,
            3,
            2,
            action_system=self.action_system,
            tool_type="axe",
        )
        resource = ResourceItem(
            "Stone Block",
            "Basic building resource",
            4,
            6,
            3,
            action_system=self.action_system,
            resource_type="stone",
        )
        quest_item = QuestItem(
            "Lost Necklace",
            "Return this to the villager who lost it",
            1,
            1,
            1,
            action_system=self.action_system,
            quest_name="Lost Necklace",
            objective="Return the necklace",
        )
        weapon = WeaponItem(
            "Spear",
            "Simple reach weapon",
            14,
            4,
            1,
            action_system=self.action_system,
            weapon_type="spear",
            damage=5,
        )

        inventory.add_item(tool)
        inventory.add_item(resource)
        inventory.add_item(quest_item)
        inventory.add_item(weapon)

        self.assertEqual(inventory.count_total_items_by_type("tool"), 2)
        self.assertEqual(inventory.count_total_items_by_type("resource"), 3)
        self.assertEqual(inventory.count_total_items_by_type("quest_item"), 1)
        self.assertEqual(inventory.count_total_items_by_type("weapon"), 1)
        self.assertTrue(inventory.check_has_item_by_type(["tool", "resource", "quest"], amount=6))
        self.assertIn(resource, inventory.get_resources_items())
        self.assertIn(quest_item, inventory.get_quest_items())
        self.assertIn("resources", inventory.to_dict())
        self.assertIn("quest", inventory.to_dict())

    def test_unrecognized_item_types_fall_back_to_misc_inventory(self):
        inventory = ItemInventory()
        door = Door(
            "Tavern Door",
            "A sturdy oak door",
            5,
            10,
            1,
            action_system=self.action_system,
        )

        inventory.add_item(door)

        self.assertIn(door, inventory.get_misc_items())
        self.assertTrue(inventory.check_has_item_by_type("misc"))


if __name__ == "__main__":
    unittest.main()
