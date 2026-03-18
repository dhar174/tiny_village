"""
Integration tests for Building Manager with GameplayController.

Tests that building functionality is properly integrated into the game loop.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_building_manager import BuildingManager, ResourceType
from tiny_buildings import Building
try:
    from actions import ActionSystem
except ImportError:
    ActionSystem = Mock


class MockCharacter:
    """Mock character for testing."""
    
    def __init__(self, name="TestChar", wealth_money=100, **attributes):
        self.name = name
        self.wealth_money = wealth_money
        self.uuid = attributes.get('uuid', f'{name}_uuid')
        self.hunger_level = attributes.get('hunger', 5)
        self.thirst = attributes.get('thirst', 50)
        self.energy = attributes.get('energy', 50)
        self.social_wellbeing = attributes.get('social_wellbeing', 50)
        self.current_satisfaction = attributes.get('satisfaction', 50)
        self.intelligence = attributes.get('intelligence', 50)
        self.knowledge = attributes.get('knowledge', 50)
        
        # Mock skills
        self.skills = Mock()
        self.skills.crafting = attributes.get('crafting_skill', 20)


class TestBuildingIntegration(unittest.TestCase):
    """Test building integration with game systems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.building_manager = BuildingManager()
        self.mock_character = MockCharacter(wealth_money=100)
    
    def test_building_with_manager_integration(self):
        """Test that Building class integrates with BuildingManager."""
        # Create a building with the manager
        building = Building(
            name="Test Market",
            x=10,
            y=20,
            height=10,
            width=20,
            length=15,
            building_type="market",
            building_manager=self.building_manager
        )
        
        # Building should be registered with the manager
        resources = building.get_resource_levels()
        self.assertIsNotNone(resources)
        self.assertGreater(resources['goods'], 0)
    
    def test_building_service_provision(self):
        """Test providing services through Building class."""
        building = Building(
            name="Test Tavern",
            x=10,
            y=20,
            height=10,
            width=20,
            length=15,
            building_type="tavern",
            building_manager=self.building_manager
        )
        
        initial_wealth = self.mock_character.wealth_money
        initial_hunger = self.mock_character.hunger_level
        
        success, message = building.provide_service("buy_meal", self.mock_character)
        
        self.assertTrue(success)
        self.assertLess(self.mock_character.wealth_money, initial_wealth)
        self.assertLess(self.mock_character.hunger_level, initial_hunger)
    
    def test_building_production_through_building_class(self):
        """Test resource production through Building class."""
        building = Building(
            name="Test Farm",
            x=10,
            y=20,
            height=10,
            width=20,
            length=15,
            building_type="farm",
            building_manager=self.building_manager
        )
        
        initial_food = building.get_resource_levels()['food']
        
        # Process production (farm interval is 20 ticks)
        success = building.process_production(20)
        self.assertTrue(success)
        
        final_food = building.get_resource_levels()['food']
        self.assertGreater(final_food, initial_food)
    
    def test_building_get_available_services(self):
        """Test getting available services through Building class."""
        building = Building(
            name="Test Blacksmith",
            x=10,
            y=20,
            height=10,
            width=20,
            length=15,
            building_type="blacksmith",
            building_manager=self.building_manager
        )
        
        services = building.get_available_services(self.mock_character)
        self.assertGreater(len(services), 0)
        
        service_names = [s.name for s in services]
        self.assertIn("Repair Tools", service_names)
    
    def test_building_full_info(self):
        """Test getting full building information."""
        building = Building(
            name="Test Market",
            x=10,
            y=20,
            height=10,
            width=20,
            length=15,
            building_type="market",
            building_manager=self.building_manager
        )
        
        info = building.get_full_building_info()
        
        self.assertIn('name', info)
        self.assertIn('type', info)
        self.assertIn('resources', info)
        self.assertIn('available_services', info)
        self.assertIn('production_interval', info)
        self.assertEqual(info['name'], "Test Market")
        self.assertEqual(info['type'], "market")
    
    def test_multiple_buildings_different_types(self):
        """Test managing multiple buildings of different types."""
        market = Building(
            name="Village Market",
            x=10, y=20, height=10, width=20, length=15,
            building_type="market",
            building_manager=self.building_manager
        )
        
        tavern = Building(
            name="Cozy Tavern",
            x=30, y=40, height=10, width=20, length=15,
            building_type="tavern",
            building_manager=self.building_manager
        )
        
        farm = Building(
            name="Sunny Farm",
            x=50, y=60, height=10, width=30, length=25,
            building_type="farm",
            building_manager=self.building_manager
        )
        
        # Each building should have appropriate resources
        market_resources = market.get_resource_levels()
        tavern_resources = tavern.get_resource_levels()
        farm_resources = farm.get_resource_levels()
        
        self.assertGreater(market_resources['goods'], 0)
        self.assertGreater(tavern_resources['food'], 0)
        self.assertGreater(farm_resources['food'], 0)
        
        # Each should provide different services
        market_services = [s.name for s in market.get_available_services(self.mock_character)]
        tavern_services = [s.name for s in tavern.get_available_services(self.mock_character)]
        
        self.assertIn("Buy Goods", market_services)
        self.assertIn("Buy Meal", tavern_services)
    
    def test_character_uses_multiple_building_services(self):
        """Test character using services from multiple buildings."""
        market = Building(
            name="Village Market",
            x=10, y=20, height=10, width=20, length=15,
            building_type="market",
            building_manager=self.building_manager
        )
        
        tavern = Building(
            name="Cozy Tavern",
            x=30, y=40, height=10, width=20, length=15,
            building_type="tavern",
            building_manager=self.building_manager
        )
        
        character = MockCharacter(wealth_money=100, hunger=8)
        
        initial_wealth = character.wealth_money
        initial_satisfaction = character.current_satisfaction
        initial_hunger = character.hunger_level
        
        # Use market service
        success1, _ = market.provide_service("buy_goods", character)
        self.assertTrue(success1)
        self.assertLess(character.wealth_money, initial_wealth)
        self.assertGreater(character.current_satisfaction, initial_satisfaction)
        
        # Use tavern service
        success2, _ = tavern.provide_service("buy_meal", character)
        self.assertTrue(success2)
        self.assertLess(character.hunger_level, initial_hunger)
    
    def test_building_resource_depletion_and_production(self):
        """Test that resources deplete with use and replenish with production."""
        tavern = Building(
            name="Busy Tavern",
            x=10, y=20, height=10, width=20, length=15,
            building_type="tavern",
            building_manager=self.building_manager
        )
        
        # Consume food multiple times
        characters = [MockCharacter(f"Customer{i}", 100) for i in range(5)]
        
        initial_food = tavern.get_resource_levels()['food']
        
        for character in characters:
            tavern.provide_service("buy_meal", character)
        
        after_consumption_food = tavern.get_resource_levels()['food']
        self.assertLess(after_consumption_food, initial_food)
        
        # Run production to replenish (tavern needs materials to produce)
        # Add materials first
        resources = self.building_manager.building_resources[str(tavern.uuid)]
        resources.add(ResourceType.MATERIALS, 20)
        
        for tick in [5, 10, 15, 20, 25]:
            tavern.process_production(tick)
        
        after_production_food = tavern.get_resource_levels()['food']
        self.assertGreater(after_production_food, after_consumption_food)
    
    def test_economic_transaction_flow(self):
        """Test complete economic flow: character buys and sells at market."""
        market = Building(
            name="Trading Post",
            x=10, y=20, height=10, width=20, length=15,
            building_type="market",
            building_manager=self.building_manager
        )
        
        trader = MockCharacter("Trader", wealth_money=50)
        
        initial_wealth = trader.wealth_money
        
        # Sell goods to market (gains money)
        success1, _ = market.provide_service("sell_goods", trader)
        self.assertTrue(success1)
        after_sell_wealth = trader.wealth_money
        self.assertGreater(after_sell_wealth, initial_wealth)
        
        # Buy goods from market (loses money)
        success2, _ = market.provide_service("buy_goods", trader)
        self.assertTrue(success2)
        final_wealth = trader.wealth_money
        self.assertLess(final_wealth, after_sell_wealth)  # Less than after selling


class TestBuildingGameplayScenarios(unittest.TestCase):
    """Test realistic gameplay scenarios with buildings."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.building_manager = BuildingManager()
    
    def test_village_economy_scenario(self):
        """Test a complete village economy scenario."""
        # Create village buildings
        farm = Building(
            "Village Farm", 10, 10, 10, 20, 15,
            building_type="farm",
            building_manager=self.building_manager
        )
        
        market = Building(
            "Village Market", 30, 30, 10, 20, 15,
            building_type="market",
            building_manager=self.building_manager
        )
        
        tavern = Building(
            "Village Tavern", 50, 50, 10, 20, 15,
            building_type="tavern",
            building_manager=self.building_manager
        )
        
        blacksmith = Building(
            "Village Blacksmith", 70, 70, 10, 15, 15,
            building_type="blacksmith",
            building_manager=self.building_manager
        )
        
        # Create villagers
        farmer = MockCharacter("Farmer Bob", 50)
        merchant = MockCharacter("Merchant Alice", 100)
        craftsperson = MockCharacter("Blacksmith Joe", 75)
        
        # Simulate production cycles
        for tick in range(0, 100, 5):
            farm.process_production(tick)
            market.process_production(tick)
            tavern.process_production(tick)
            blacksmith.process_production(tick)
        
        # Villagers use services
        farm.provide_service("buy_food", farmer)
        market.provide_service("buy_goods", merchant)
        tavern.provide_service("buy_meal", craftsperson)
        blacksmith.provide_service("repair_tools", merchant)
        
        # All buildings should have resources
        self.assertGreater(farm.get_resource_levels()['food'], 0)
        self.assertGreater(market.get_resource_levels()['goods'], 0)
        self.assertGreater(tavern.get_resource_levels()['food'], 0)
        self.assertGreater(blacksmith.get_resource_levels()['tools'], 0)
        
        # All villagers should have spent money
        self.assertLess(farmer.wealth_money, 50)
        self.assertLess(merchant.wealth_money, 100)
        self.assertLess(craftsperson.wealth_money, 75)
    
    def test_survival_scenario(self):
        """Test character survival using building services."""
        tavern = Building(
            "Safety Tavern", 10, 10, 10, 20, 15,
            building_type="tavern",
            building_manager=self.building_manager
        )
        
        farm = Building(
            "Food Farm", 30, 30, 10, 30, 25,
            building_type="farm",
            building_manager=self.building_manager
        )
        
        # Character starts hungry and thirsty
        survivor = MockCharacter("Survivor", 50, hunger=9, thirst=80)
        
        # Character visits tavern for meal and drink
        tavern.provide_service("buy_meal", survivor)
        tavern.provide_service("buy_drink", survivor)
        
        # Character should be less hungry and thirsty
        self.assertLess(survivor.hunger_level, 9)
        self.assertLess(survivor.thirst, 80)
        
        # Character can also get food from farm
        farm.provide_service("buy_food", survivor)
        self.assertLess(survivor.hunger_level, 6)  # Should be even less hungry


if __name__ == '__main__':
    unittest.main()
