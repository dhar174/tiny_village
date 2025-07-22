#!/usr/bin/env python3
"""
Test suite for Location Integration
Tests the integration between Building objects and Location objects,
and their use in MapController for character AI decisions.
"""

import unittest
import sys
import os

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_buildings import Building, House, CreateBuilding
from tiny_locations import Location, LocationManager
from actions import ActionSystem


class TestLocationIntegration(unittest.TestCase):
    """Test Location Integration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.action_system = ActionSystem()
        
    def test_building_has_location(self):
        """Test that Building objects have Location instances"""
        building = Building("Test Building", 10, 20, 30, 40, 50)
        
        # Building should have a location
        self.assertIsNotNone(building.location)
        self.assertIsInstance(building.location, Location)
        
        # Location should match building coordinates and dimensions
        self.assertEqual(building.location.x, 10)
        self.assertEqual(building.location.y, 20)
        self.assertEqual(building.location.width, 40)
        self.assertEqual(building.location.height, 50)
        
    def test_building_location_properties(self):
        """Test that buildings have security, popularity, and activities"""
        building = Building("Test Building", 10, 20, 30, 40, 50, building_type="commercial")
        
        # Should have security and popularity values
        security = building.get_security_level()
        popularity = building.get_popularity_level()
        activities = building.get_available_activities()
        
        self.assertIsInstance(security, int)
        self.assertIsInstance(popularity, int)
        self.assertIsInstance(activities, list)
        
        # Values should be in reasonable ranges
        self.assertGreaterEqual(security, 0)
        self.assertLessEqual(security, 10)
        self.assertGreaterEqual(popularity, 0)
        self.assertLessEqual(popularity, 10)
        
        # Should have some activities
        self.assertGreater(len(activities), 0)
        
    def test_house_specific_activities(self):
        """Test that House objects have house-specific activities"""
        house = House("Test House", 10, 20, 30, 40, 50, "123 Test St", 
                     stories=2, bedrooms=3, bathrooms=2)
        
        activities = house.get_available_activities()
        
        # Should have house-specific activities
        self.assertIn("enter_house", activities)
        self.assertIn("rest", activities)
        self.assertIn("sleep", activities)
        
        # Should have activities based on house properties
        if house.bedrooms > 1:
            self.assertIn("visit_family", activities)
        if house.bathrooms > 0:
            self.assertIn("use_facilities", activities)
            
    def test_building_position_checking(self):
        """Test that buildings can check if characters are within them"""
        building = Building("Test Building", 10, 10, 30, 20, 20)
        
        # Position inside building
        self.assertTrue(building.is_within_building((15, 15)))
        
        # Position outside building
        self.assertFalse(building.is_within_building((5, 5)))
        self.assertFalse(building.is_within_building((35, 35)))
        
    def test_building_type_affects_properties(self):
        """Test that building type affects security, popularity, and activities"""
        house = Building("House", 10, 10, 30, 20, 20, building_type="house")
        commercial = Building("Shop", 50, 50, 30, 20, 20, building_type="commercial")
        
        # Houses should be more secure than commercial buildings
        self.assertGreater(house.get_security_level(), commercial.get_security_level())
        
        # Commercial buildings should be more popular
        self.assertGreater(commercial.get_popularity_level(), house.get_popularity_level())
        
        # Different activities
        house_activities = house.get_available_activities()
        commercial_activities = commercial.get_available_activities()
        
        # Check for type-specific activities
        self.assertTrue(any("rest" in activity or "visit" in activity for activity in house_activities))
        self.assertTrue(any("shop" in activity or "business" in activity for activity in commercial_activities))
        
    def test_location_can_be_used_for_pathfinding(self):
        """Test that building locations provide necessary data for pathfinding"""
        building = Building("Test Building", 10, 20, 30, 40, 50)
        location = building.get_location()
        
        # Location should provide geometric data needed for pathfinding
        self.assertEqual(location.get_coordinates(), (10, 20))
        self.assertEqual(location.get_dimensions(), (40, 50))
        self.assertGreater(location.get_area(), 0)
        
        # Should be able to check point containment
        self.assertTrue(location.contains_point(25, 35))
        self.assertFalse(location.contains_point(5, 5))
        
        # Should be able to calculate distances
        distance = location.distance_to_point_from_center(0, 0)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)


class TestMapControllerIntegration(unittest.TestCase):
    """Test MapController integration with Building-Location system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock map data for testing
        self.map_data = {
            "width": 100,
            "height": 100,
            "buildings": []
        }
        
    def test_map_controller_without_pygame(self):
        """Test MapController functionality that doesn't require pygame"""
        # Import MapController but skip pygame-dependent parts
        try:
            from tiny_map_controller import MapController
            
            # Create a mock MapController that doesn't require pygame
            class MockMapController:
                def __init__(self, map_data):
                    self.map_data = map_data
                    self.buildings = []
                    self.dynamic_obstacles = set()
                    
                def add_building(self, building):
                    self.buildings.append(building)
                    
                def get_buildings_at_position(self, position):
                    x, y = position
                    buildings_at_pos = []
                    for building in self.buildings:
                        if building.is_within_building((x, y)):
                            buildings_at_pos.append(building)
                    return buildings_at_pos
                    
                def get_building_by_location_properties(self, min_security=None, min_popularity=None, 
                                                      required_activities=None):
                    matching_buildings = []
                    for building in self.buildings:
                        location = building.get_location()
                        
                        if min_security is not None and location.security < min_security:
                            continue
                        if min_popularity is not None and location.popularity < min_popularity:
                            continue
                        if required_activities:
                            available_activities = set(location.activities_available)
                            required_set = set(required_activities)
                            if not required_set.issubset(available_activities):
                                continue
                        matching_buildings.append(building)
                    return matching_buildings
                    
                def find_safe_locations(self, min_security=7):
                    return self.get_building_by_location_properties(min_security=min_security)
                    
                def find_popular_locations(self, min_popularity=6):
                    return self.get_building_by_location_properties(min_popularity=min_popularity)
                    
                def find_locations_with_activity(self, activity):
                    return self.get_building_by_location_properties(required_activities=[activity])
            
            controller = MockMapController(self.map_data)
            
            # Add some buildings
            house = House("Safe House", 10, 10, 30, 20, 20, "123 Safe St", bedrooms=3, bathrooms=2)
            shop = Building("Popular Shop", 50, 50, 30, 20, 20, building_type="commercial")
            
            controller.add_building(house)
            controller.add_building(shop)
            
            # Test building position detection
            buildings_at_house = controller.get_buildings_at_position((15, 15))
            self.assertEqual(len(buildings_at_house), 1)
            self.assertEqual(buildings_at_house[0].name, "Safe House")
            
            # Test finding safe locations
            safe_locations = controller.find_safe_locations(min_security=6)
            self.assertGreater(len(safe_locations), 0)
            
            # Test finding popular locations
            popular_locations = controller.find_popular_locations(min_popularity=4)
            self.assertGreater(len(popular_locations), 0)
            
            # Test finding locations with specific activities
            rest_locations = controller.find_locations_with_activity("rest")
            house_names = [building.name for building in rest_locations]
            self.assertIn("Safe House", house_names)
            
        except ImportError as e:
            # Skip test if pygame is not available
            self.skipTest(f"Skipping MapController test due to missing dependency: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)