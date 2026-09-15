import unittest
from types import SimpleNamespace

from tiny_building_manager import BuildingManager
from tiny_economic_simulation import EconomicSimulation
from tiny_items import ItemInventory


class StubCharacter:
    def __init__(self, name="Character", job="unemployed", wealth_money=25, hunger_level=0):
        self.name = name
        self.job = job
        self.wealth_money = wealth_money
        self.hunger_level = hunger_level
        self.inventory = ItemInventory([], [], [], [], [], [])
        self.action_system = SimpleNamespace(instantiate_conditions=lambda *_args, **_kwargs: [])


class TestEconomicSimulation(unittest.TestCase):
    def setUp(self):
        self.economic_simulation = EconomicSimulation()

    def test_job_production_adds_items_to_inventory(self):
        farmer = StubCharacter(name="Farmer", job="farmer")

        produced_items = self.economic_simulation.produce_items_for_job(farmer, current_tick=10)

        self.assertEqual(len(produced_items), 1)
        self.assertEqual(farmer.inventory.count_food_items_total(), 2)
        self.assertEqual(farmer.inventory.count_total_items_by_name("Farm Produce"), 2)

    def test_unmapped_job_produces_no_items(self):
        character = StubCharacter(name="Clerk", job="accountant")

        produced_items = self.economic_simulation.produce_items_for_job(character, current_tick=10)

        self.assertEqual(produced_items, [])
        self.assertEqual(character.inventory.count_total_items(), 0)

    def test_job_production_is_rate_limited_per_character(self):
        farmer = StubCharacter(name="Farmer", job="farmer")

        self.economic_simulation.produce_items_for_job(farmer, current_tick=10)
        second_production = self.economic_simulation.produce_items_for_job(farmer, current_tick=15)

        self.assertEqual(second_production, [])
        self.assertEqual(farmer.inventory.count_food_items_total(), 2)

    def test_need_consumption_uses_food_inventory(self):
        farmer = StubCharacter(name="Farmer", job="farmer", hunger_level=8)
        self.economic_simulation.produce_items_for_job(farmer, current_tick=10)

        consumed_item = self.economic_simulation.consume_items_for_needs(farmer)

        self.assertIsNotNone(consumed_item)
        self.assertEqual(consumed_item.get_name(), "Farm Produce")
        self.assertLess(farmer.hunger_level, 8)
        self.assertEqual(farmer.inventory.count_food_items_total(), 1)

    def test_need_consumption_skips_when_food_missing(self):
        hungry_character = StubCharacter(name="Hungry", hunger_level=8)

        consumed_item = self.economic_simulation.consume_items_for_needs(hungry_character)

        self.assertIsNone(consumed_item)
        self.assertEqual(hungry_character.hunger_level, 8)

    def test_trade_transfers_item_and_wealth(self):
        seller = StubCharacter(name="Seller", job="merchant", wealth_money=10)
        buyer = StubCharacter(name="Buyer", wealth_money=30)
        self.economic_simulation.produce_items_for_job(seller, current_tick=10)

        success, _message = self.economic_simulation.trade_item(
            seller,
            buyer,
            "Trade Goods",
            quantity=1,
            unit_price=7,
        )

        self.assertTrue(success)
        self.assertEqual(seller.wealth_money, 17)
        self.assertEqual(buyer.wealth_money, 23)
        self.assertEqual(seller.inventory.count_total_items_by_name("Trade Goods"), 1)
        self.assertEqual(buyer.inventory.count_total_items_by_name("Trade Goods"), 1)

    def test_trade_fails_when_buyer_cannot_afford_item(self):
        seller = StubCharacter(name="Seller", job="merchant", wealth_money=10)
        buyer = StubCharacter(name="Buyer", wealth_money=2)
        self.economic_simulation.produce_items_for_job(seller, current_tick=10)

        success, message = self.economic_simulation.trade_item(
            seller,
            buyer,
            "Trade Goods",
            quantity=1,
            unit_price=5,
        )

        self.assertFalse(success)
        self.assertEqual(message, "Buyer cannot afford trade")
        self.assertEqual(seller.inventory.count_total_items_by_name("Trade Goods"), 2)
        self.assertEqual(buyer.inventory.count_total_items(), 0)

    def test_trade_fails_when_seller_lacks_quantity(self):
        seller = StubCharacter(name="Seller", job="merchant", wealth_money=10)
        buyer = StubCharacter(name="Buyer", wealth_money=30)
        self.economic_simulation.produce_items_for_job(seller, current_tick=10)

        success, message = self.economic_simulation.trade_item(
            seller,
            buyer,
            "Trade Goods",
            quantity=5,
            unit_price=5,
        )

        self.assertFalse(success)
        self.assertEqual(message, "Seller lacks 5 Trade Goods")

    def test_sync_item_availability_includes_buildings_and_characters(self):
        building_manager = BuildingManager()
        building_manager.register_building("farm-1", "farm")
        building_manager.process_production("farm-1", "farm", current_tick=20)

        farmer = StubCharacter(name="Farmer", job="farmer")
        self.economic_simulation.produce_items_for_job(farmer, current_tick=20)

        availability = self.economic_simulation.sync_item_availability(
            building_manager=building_manager,
            characters=[farmer],
        )

        self.assertGreaterEqual(availability["food"], 110)
        self.assertEqual(availability["Farm Produce"], 2)


if __name__ == "__main__":
    unittest.main()
