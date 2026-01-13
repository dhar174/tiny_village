"""
Tests for the Building Manager system.

Tests resource production, consumption, and service functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock
from tiny_building_manager import (
    BuildingManager,
    ResourceType,
    ResourcePool,
    BuildingService
)


class MockCharacter:
    """Mock character for testing."""
    
    def __init__(self, wealth_money=100, **attributes):
        self.wealth_money = wealth_money
        self.name = attributes.get('name', 'TestChar')
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


class TestResourcePool(unittest.TestCase):
    """Test ResourcePool functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pool = ResourcePool()
    
    def test_initial_state(self):
        """Test that resource pool starts empty."""
        self.assertEqual(self.pool.food, 0)
        self.assertEqual(self.pool.materials, 0)
        self.assertEqual(self.pool.tools, 0)
    
    def test_add_resources(self):
        """Test adding resources to pool."""
        self.pool.add(ResourceType.FOOD, 10)
        self.assertEqual(self.pool.food, 10)
        
        self.pool.add(ResourceType.FOOD, 5)
        self.assertEqual(self.pool.food, 15)
    
    def test_add_negative_raises_error(self):
        """Test that adding negative resources raises error."""
        with self.assertRaises(ValueError):
            self.pool.add(ResourceType.FOOD, -5)
    
    def test_consume_resources_success(self):
        """Test consuming resources when sufficient."""
        self.pool.add(ResourceType.MATERIALS, 20)
        
        success = self.pool.consume(ResourceType.MATERIALS, 10)
        self.assertTrue(success)
        self.assertEqual(self.pool.materials, 10)
    
    def test_consume_resources_failure(self):
        """Test consuming resources when insufficient."""
        self.pool.add(ResourceType.TOOLS, 5)
        
        success = self.pool.consume(ResourceType.TOOLS, 10)
        self.assertFalse(success)
        self.assertEqual(self.pool.tools, 5)  # Should remain unchanged
    
    def test_consume_negative_raises_error(self):
        """Test that consuming negative resources raises error."""
        with self.assertRaises(ValueError):
            self.pool.consume(ResourceType.FOOD, -5)
    
    def test_get_resource_amount(self):
        """Test getting resource amounts."""
        self.pool.add(ResourceType.GOODS, 25)
        self.assertEqual(self.pool.get(ResourceType.GOODS), 25)
    
    def test_to_dict(self):
        """Test converting pool to dictionary."""
        self.pool.add(ResourceType.FOOD, 10)
        self.pool.add(ResourceType.MATERIALS, 20)
        
        result = self.pool.to_dict()
        
        self.assertEqual(result['food'], 10)
        self.assertEqual(result['materials'], 20)
        self.assertEqual(result['tools'], 0)


class TestBuildingService(unittest.TestCase):
    """Test BuildingService functionality."""
    
    def test_can_provide_with_sufficient_resources(self):
        """Test service availability with sufficient resources."""
        character = MockCharacter(wealth_money=50)
        pool = ResourcePool()
        pool.add(ResourceType.FOOD, 10)
        
        service = BuildingService(
            name="Test Service",
            cost=20,
            resource_requirements={ResourceType.FOOD: 5}
        )
        
        self.assertTrue(service.can_provide(character, pool))
    
    def test_can_provide_insufficient_money(self):
        """Test service availability with insufficient money."""
        character = MockCharacter(wealth_money=10)
        pool = ResourcePool()
        pool.add(ResourceType.FOOD, 10)
        
        service = BuildingService(
            name="Test Service",
            cost=20,
            resource_requirements={ResourceType.FOOD: 5}
        )
        
        self.assertFalse(service.can_provide(character, pool))
    
    def test_can_provide_insufficient_resources(self):
        """Test service availability with insufficient resources."""
        character = MockCharacter(wealth_money=50)
        pool = ResourcePool()
        pool.add(ResourceType.FOOD, 2)
        
        service = BuildingService(
            name="Test Service",
            cost=20,
            resource_requirements={ResourceType.FOOD: 5}
        )
        
        self.assertFalse(service.can_provide(character, pool))


class TestBuildingManager(unittest.TestCase):
    """Test BuildingManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = BuildingManager()
        self.mock_character = MockCharacter(wealth_money=100)
    
    def test_register_building(self):
        """Test registering a new building."""
        self.manager.register_building("market_1", "market")
        
        self.assertIn("market_1", self.manager.building_resources)
        self.assertIn("market_1", self.manager.last_production_tick)
    
    def test_initialize_building_resources_market(self):
        """Test that markets initialize with goods and materials."""
        self.manager.register_building("market_1", "market")
        
        resources = self.manager.get_building_resources("market_1")
        self.assertIsNotNone(resources)
        self.assertGreater(resources['goods'], 0)
        self.assertGreater(resources['materials'], 0)
    
    def test_initialize_building_resources_tavern(self):
        """Test that taverns initialize with food and services."""
        self.manager.register_building("tavern_1", "tavern")
        
        resources = self.manager.get_building_resources("tavern_1")
        self.assertIsNotNone(resources)
        self.assertGreater(resources['food'], 0)
        self.assertGreater(resources['services'], 0)
    
    def test_initialize_building_resources_blacksmith(self):
        """Test that blacksmiths initialize with materials and tools."""
        self.manager.register_building("blacksmith_1", "blacksmith")
        
        resources = self.manager.get_building_resources("blacksmith_1")
        self.assertIsNotNone(resources)
        self.assertGreater(resources['materials'], 0)
        self.assertGreater(resources['tools'], 0)
    
    def test_initialize_building_resources_farm(self):
        """Test that farms initialize with food."""
        self.manager.register_building("farm_1", "farm")
        
        resources = self.manager.get_building_resources("farm_1")
        self.assertIsNotNone(resources)
        self.assertGreater(resources['food'], 0)
    
    def test_initialize_building_resources_school(self):
        """Test that schools initialize with knowledge."""
        self.manager.register_building("school_1", "school")
        
        resources = self.manager.get_building_resources("school_1")
        self.assertIsNotNone(resources)
        self.assertGreater(resources['knowledge'], 0)
    
    def test_process_production_too_early(self):
        """Test that production doesn't occur before interval."""
        self.manager.register_building("farm_1", "farm")
        
        # Production interval for farm is 20 ticks
        result = self.manager.process_production("farm_1", "farm", 10)
        self.assertFalse(result)
    
    def test_process_production_success(self):
        """Test successful resource production."""
        self.manager.register_building("farm_1", "farm")
        
        initial_food = self.manager.building_resources["farm_1"].get(ResourceType.FOOD)
        
        # Production interval for farm is 20 ticks
        result = self.manager.process_production("farm_1", "farm", 20)
        self.assertTrue(result)
        
        final_food = self.manager.building_resources["farm_1"].get(ResourceType.FOOD)
        self.assertGreater(final_food, initial_food)
    
    def test_process_production_with_consumption(self):
        """Test production that requires resource consumption."""
        self.manager.register_building("market_1", "market")
        
        # Market produces goods but consumes materials
        initial_goods = self.manager.building_resources["market_1"].get(ResourceType.GOODS)
        initial_materials = self.manager.building_resources["market_1"].get(ResourceType.MATERIALS)
        
        # Production interval for market is 10 ticks
        result = self.manager.process_production("market_1", "market", 10)
        
        if result:
            final_goods = self.manager.building_resources["market_1"].get(ResourceType.GOODS)
            final_materials = self.manager.building_resources["market_1"].get(ResourceType.MATERIALS)
            
            self.assertGreater(final_goods, initial_goods)
            self.assertLess(final_materials, initial_materials)
    
    def test_process_production_insufficient_resources(self):
        """Test production fails when insufficient input resources."""
        self.manager.register_building("blacksmith_1", "blacksmith")
        
        # Consume all materials
        resources = self.manager.building_resources["blacksmith_1"]
        resources.consume(ResourceType.MATERIALS, resources.get(ResourceType.MATERIALS))
        
        # Production should fail due to insufficient materials
        result = self.manager.process_production("blacksmith_1", "blacksmith", 100)
        self.assertFalse(result)
    
    def test_get_available_services_market(self):
        """Test getting available services for market."""
        services = self.manager.get_available_services("market", self.mock_character)
        
        self.assertGreater(len(services), 0)
        service_names = [s.name for s in services]
        self.assertIn("Buy Goods", service_names)
    
    def test_get_available_services_tavern(self):
        """Test getting available services for tavern."""
        services = self.manager.get_available_services("tavern", self.mock_character)
        
        self.assertGreater(len(services), 0)
        service_names = [s.name for s in services]
        self.assertIn("Buy Meal", service_names)
        self.assertIn("Buy Drink", service_names)
    
    def test_get_available_services_blacksmith(self):
        """Test getting available services for blacksmith."""
        services = self.manager.get_available_services("blacksmith", self.mock_character)
        
        self.assertGreater(len(services), 0)
        service_names = [s.name for s in services]
        self.assertIn("Repair Tools", service_names)
    
    def test_provide_service_success(self):
        """Test successful service provision."""
        self.manager.register_building("tavern_1", "tavern")
        
        initial_wealth = self.mock_character.wealth_money
        initial_hunger = self.mock_character.hunger_level
        
        success, message = self.manager.provide_service(
            "tavern_1", "tavern", "buy_meal", self.mock_character
        )
        
        self.assertTrue(success)
        self.assertLess(self.mock_character.wealth_money, initial_wealth)
        self.assertLess(self.mock_character.hunger_level, initial_hunger)
    
    def test_provide_service_insufficient_money(self):
        """Test service provision fails with insufficient money."""
        self.manager.register_building("market_1", "market")
        
        poor_character = MockCharacter(wealth_money=1)
        
        success, message = self.manager.provide_service(
            "market_1", "market", "buy_goods", poor_character
        )
        
        self.assertFalse(success)
        self.assertIn("Insufficient", message)
    
    def test_provide_service_money_transfer(self):
        """Test that service costs are properly deducted."""
        self.manager.register_building("tavern_1", "tavern")
        
        initial_wealth = self.mock_character.wealth_money
        
        success, message = self.manager.provide_service(
            "tavern_1", "tavern", "buy_drink", self.mock_character
        )
        
        self.assertTrue(success)
        # Buy drink costs 3 money
        expected_wealth = initial_wealth - 3
        self.assertEqual(self.mock_character.wealth_money, expected_wealth)
    
    def test_provide_service_resource_consumption(self):
        """Test that service consumes building resources."""
        self.manager.register_building("tavern_1", "tavern")
        
        initial_food = self.manager.building_resources["tavern_1"].get(ResourceType.FOOD)
        
        success, message = self.manager.provide_service(
            "tavern_1", "tavern", "buy_meal", self.mock_character
        )
        
        self.assertTrue(success)
        final_food = self.manager.building_resources["tavern_1"].get(ResourceType.FOOD)
        self.assertLess(final_food, initial_food)
    
    def test_provide_service_effects_applied(self):
        """Test that service effects are applied to character."""
        self.manager.register_building("school_1", "school")
        
        initial_intelligence = self.mock_character.intelligence
        
        success, message = self.manager.provide_service(
            "school_1", "school", "take_lesson", self.mock_character
        )
        
        self.assertTrue(success)
        self.assertGreater(self.mock_character.intelligence, initial_intelligence)
    
    def test_provide_service_invalid_service(self):
        """Test providing invalid service name."""
        self.manager.register_building("market_1", "market")
        
        success, message = self.manager.provide_service(
            "market_1", "market", "invalid_service", self.mock_character
        )
        
        self.assertFalse(success)
        self.assertIn("not available", message)
    
    def test_service_history_tracking(self):
        """Test that service history is tracked."""
        self.manager.register_building("market_1", "market")
        
        initial_history_len = len(self.manager.service_history)
        
        self.manager.provide_service(
            "market_1", "market", "buy_goods", self.mock_character
        )
        
        self.assertEqual(len(self.manager.service_history), initial_history_len + 1)
        
        last_service = self.manager.service_history[-1]
        self.assertEqual(last_service['building_id'], "market_1")
        self.assertEqual(last_service['building_type'], "market")
    
    def test_get_building_info(self):
        """Test getting comprehensive building information."""
        self.manager.register_building("market_1", "market")
        
        info = self.manager.get_building_info("market_1", "market")
        
        self.assertEqual(info['building_id'], "market_1")
        self.assertEqual(info['building_type'], "market")
        self.assertIn('resources', info)
        self.assertIn('available_services', info)
        self.assertIn('production_interval', info)
        self.assertGreater(info['production_interval'], 0)
    
    def test_multiple_buildings(self):
        """Test managing multiple buildings simultaneously."""
        self.manager.register_building("market_1", "market")
        self.manager.register_building("tavern_1", "tavern")
        self.manager.register_building("farm_1", "farm")
        
        self.assertEqual(len(self.manager.building_resources), 3)
        
        # Each should have appropriate resources
        market_resources = self.manager.get_building_resources("market_1")
        tavern_resources = self.manager.get_building_resources("tavern_1")
        farm_resources = self.manager.get_building_resources("farm_1")
        
        self.assertGreater(market_resources['goods'], 0)
        self.assertGreater(tavern_resources['food'], 0)
        self.assertGreater(farm_resources['food'], 0)
    
    def test_production_cycles(self):
        """Test multiple production cycles."""
        self.manager.register_building("farm_1", "farm")
        
        initial_food = self.manager.building_resources["farm_1"].get(ResourceType.FOOD)
        
        # Farm production interval is 20 ticks
        # Run production at tick 20, 40, 60
        self.manager.process_production("farm_1", "farm", 20)
        food_after_1 = self.manager.building_resources["farm_1"].get(ResourceType.FOOD)
        
        self.manager.process_production("farm_1", "farm", 40)
        food_after_2 = self.manager.building_resources["farm_1"].get(ResourceType.FOOD)
        
        self.manager.process_production("farm_1", "farm", 60)
        food_after_3 = self.manager.building_resources["farm_1"].get(ResourceType.FOOD)
        
        # Food should increase with each production cycle
        self.assertGreater(food_after_1, initial_food)
        self.assertGreater(food_after_2, food_after_1)
        self.assertGreater(food_after_3, food_after_2)


class TestBuildingTypeFunctionality(unittest.TestCase):
    """Test specific functionality for each building type."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = BuildingManager()
    
    def test_market_buy_and_sell(self):
        """Test market buy and sell functionality."""
        self.manager.register_building("market_1", "market")
        
        # Test buying
        buyer = MockCharacter(wealth_money=50)
        success, _ = self.manager.provide_service(
            "market_1", "market", "buy_goods", buyer
        )
        self.assertTrue(success)
        self.assertLess(buyer.wealth_money, 50)
        
        # Test selling
        seller = MockCharacter(wealth_money=10)
        initial_wealth = seller.wealth_money
        success, _ = self.manager.provide_service(
            "market_1", "market", "sell_goods", seller
        )
        self.assertTrue(success)
        self.assertGreater(seller.wealth_money, initial_wealth)
    
    def test_tavern_meal_and_drink(self):
        """Test tavern meal and drink services."""
        self.manager.register_building("tavern_1", "tavern")
        
        character = MockCharacter(wealth_money=50, hunger=8, thirst=80)
        
        # Test buying meal
        initial_hunger = character.hunger_level
        success, _ = self.manager.provide_service(
            "tavern_1", "tavern", "buy_meal", character
        )
        self.assertTrue(success)
        self.assertLess(character.hunger_level, initial_hunger)
        
        # Test buying drink
        initial_thirst = character.thirst
        success, _ = self.manager.provide_service(
            "tavern_1", "tavern", "buy_drink", character
        )
        self.assertTrue(success)
        self.assertLess(character.thirst, initial_thirst)
    
    def test_blacksmith_repairs_and_commissions(self):
        """Test blacksmith repair and commission services."""
        self.manager.register_building("blacksmith_1", "blacksmith")
        
        character = MockCharacter(wealth_money=100)
        
        # Test repair tools
        success, _ = self.manager.provide_service(
            "blacksmith_1", "blacksmith", "repair_tools", character
        )
        self.assertTrue(success)
        
        # Test commission weapon (expensive)
        rich_character = MockCharacter(wealth_money=100)
        success, _ = self.manager.provide_service(
            "blacksmith_1", "blacksmith", "commission_weapon", rich_character
        )
        self.assertTrue(success)
        self.assertLess(rich_character.wealth_money, 100)
    
    def test_farm_food_production(self):
        """Test farm produces food consistently."""
        self.manager.register_building("farm_1", "farm")
        
        # Run multiple production cycles
        for tick in [20, 40, 60, 80, 100]:
            self.manager.process_production("farm_1", "farm", tick)
        
        final_food = self.manager.building_resources["farm_1"].get(ResourceType.FOOD)
        
        # Farm should have produced significant food
        self.assertGreater(final_food, 100)
    
    def test_school_education_service(self):
        """Test school provides education services."""
        self.manager.register_building("school_1", "school")
        
        student = MockCharacter(wealth_money=50, intelligence=30)
        
        initial_intelligence = student.intelligence
        
        success, _ = self.manager.provide_service(
            "school_1", "school", "take_lesson", student
        )
        
        self.assertTrue(success)
        self.assertGreater(student.intelligence, initial_intelligence)


if __name__ == '__main__':
    unittest.main()
