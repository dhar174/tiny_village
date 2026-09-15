import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tiny_building_manager import BuildingManager


Building = None
_ORIGINAL_MODULES = {}


def _build_stub_modules():
    stub_actions = types.ModuleType("actions")

    class Action:
        def __init__(self, name, preconditions, effects):
            self.name = name
            self.preconditions = preconditions
            self.effects = effects

    class ActionSystem:
        def instantiate_conditions(self, conditions):
            return conditions

    stub_actions.Action = Action
    stub_actions.ActionSystem = ActionSystem

    stub_locations = types.ModuleType("tiny_locations")

    class Location:
        def __init__(
            self,
            name,
            x,
            y,
            width,
            height,
            action_system,
            security=0,
            popularity=0,
            threat_level=0,
        ):
            self.name = name
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.action_system = action_system
            self.security = security
            self.popularity = popularity
            self.threat_level = threat_level
            self.activities_available = []
            self.current_visitors = []

        def add_activity(self, activity):
            self.activities_available.append(activity)

        def contains_point(self, x, y):
            return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

        def get_coordinates(self):
            return (self.x, self.y)

    class LocationManager:
        pass

    stub_locations.Location = Location
    stub_locations.LocationManager = LocationManager

    return {
        "actions": stub_actions,
        "tiny_locations": stub_locations,
    }


def setUpModule():
    global Building, _ORIGINAL_MODULES

    module_names = ("actions", "tiny_locations", "tiny_buildings")
    _ORIGINAL_MODULES = {name: sys.modules.get(name) for name in module_names}

    sys.modules.pop("tiny_buildings", None)

    with patch.dict(sys.modules, _build_stub_modules()):
        Building = importlib.import_module("tiny_buildings").Building


def tearDownModule():
    for module_name in ("actions", "tiny_locations", "tiny_buildings"):
        original_module = _ORIGINAL_MODULES.get(module_name)
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


class TestBuildingOwnership(unittest.TestCase):
    def test_private_building_allows_owner_identity_match(self):
        owner = SimpleNamespace(name="Alice", uuid="owner-identity", wealth_money=100, energy=100)
        house = Building(
            "Alice House",
            0,
            0,
            10,
            10,
            10,
            building_type="house",
            owner=owner,
        )

        interactions = house.get_possible_interactions(owner)
        self.assertGreaterEqual(len(interactions), 1)
        self.assertIn("Enter Building", [action.name for action in interactions])

    def test_private_building_blocks_non_owner_interactions(self):
        owner = SimpleNamespace(name="Alice", uuid="owner-1", wealth_money=100)
        visitor = SimpleNamespace(name="Bob", uuid="visitor-2", energy=100)
        house = Building(
            "Alice House",
            0,
            0,
            10,
            10,
            10,
            building_type="house",
            owner=owner,
        )

        self.assertEqual(house.get_possible_interactions(visitor), [])

        owner_interactions = house.get_possible_interactions(
            SimpleNamespace(name="Alice", uuid="owner-1", energy=100)
        )
        self.assertGreaterEqual(len(owner_interactions), 1)
        self.assertIn("Enter Building", [action.name for action in owner_interactions])

    def test_owned_public_building_remains_accessible(self):
        owner = SimpleNamespace(name="Merchant", uuid="owner-3", wealth_money=100)
        visitor = SimpleNamespace(name="Buyer", uuid="visitor-4", energy=100)
        market = Building(
            "Store",
            0,
            0,
            10,
            10,
            10,
            building_type="commercial",
            owner=owner,
        )

        interactions = market.get_possible_interactions(visitor)
        self.assertGreater(len(interactions), 0)
        self.assertIn("Buy Items", [action.name for action in interactions])

    def test_paid_service_generates_owner_income(self):
        owner = SimpleNamespace(name="Merchant", uuid="owner-5", wealth_money=50)
        customer = SimpleNamespace(
            name="Buyer",
            uuid="visitor-6",
            wealth_money=100,
            current_satisfaction=0,
        )
        building = Building(
            "Market",
            0,
            0,
            10,
            10,
            10,
            building_type="market",
            owner=owner,
            building_manager=BuildingManager(),
        )

        success, _ = building.provide_service("buy_goods", customer)

        self.assertTrue(success)
        self.assertEqual(customer.wealth_money, 90)
        self.assertEqual(owner.wealth_money, 60)
        self.assertEqual(building.owner_income_generated, 10)

    def test_production_generates_owner_income(self):
        owner = SimpleNamespace(name="Farmer", uuid="owner-7", wealth_money=20)
        farm = Building(
            "Farm",
            0,
            0,
            10,
            10,
            10,
            building_type="farm",
            owner=owner,
            building_manager=BuildingManager(),
        )

        produced = farm.process_production(20)

        self.assertTrue(produced)
        self.assertEqual(owner.wealth_money, 30)
        self.assertEqual(farm.owner_income_generated, 10)
