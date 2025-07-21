#!/usr/bin/env python3
"""
Test script for Map and Location System Integration
Tests the core functionality without requiring heavy dependencies like pygame or numpy
"""

import sys
import os

# Add current directory to path to avoid import issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock dependencies to avoid circular imports and missing modules
class MockState:
    def __init__(self, dict_or_obj):
        self.dict_or_obj = dict_or_obj

class MockActionSystem:
    def instantiate_conditions(self, conditions):
        return conditions

class MockAction:
    def __init__(self, name, conditions, effects, cost=0):
        self.name = name
        self.conditions = conditions
        self.effects = effects
        self.cost = cost
    
    def can_execute(self, character):
        return True

# Mock the modules before importing the actual modules
sys.modules['actions'] = type(sys)('MockActions')
sys.modules['actions'].Action = MockAction
sys.modules['actions'].ActionSystem = MockActionSystem
sys.modules['actions'].State = MockState

sys.modules['tiny_types'] = type(sys)('MockTinyTypes')
sys.modules['tiny_types'].Action = MockAction
sys.modules['tiny_types'].ActionSystem = MockActionSystem
sys.modules['tiny_types'].State = MockState
sys.modules['tiny_types'].Character = type('MockCharacter', (), {})
sys.modules['tiny_types'].GraphManager = type('MockGraphManager', (), {})

class MockCharacter:
    def __init__(self, name, x=0, y=0):
        self.name = name
        self.location = MockLocation("char_location", x, y, 1, 1)
        self.safety_threshold = 2
        self.energy = 50
        self.social_preference = 50
        self.movement_preferences = {
            'avoid_danger': True,
            'prefer_roads': False
        }

class MockLocation:
    def __init__(self, name, x, y, width, height):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.coordinates_location = (x, y)

def test_building_location_integration():
    """Test that Building instances are properly connected to Location instances"""
    print("Testing Building-Location Integration...")
    
    try:
        from tiny_buildings import Building
        from tiny_locations import Location, LocationManager, PointOfInterest
        
        # Test building creation with automatic location creation
        action_system = MockActionSystem()
        building = Building(
            name="Test Shop",
            x=10, y=20,
            height=15, width=20, length=25,
            action_system=action_system,
            building_type="shop"
        )
        
        # Verify building has a location
        assert hasattr(building, 'location'), "Building should have a location attribute"
        assert building.location is not None, "Building location should not be None"
        assert building.location.name == "Test Shop Location", f"Expected 'Test Shop Location', got '{building.location.name}'"
        
        # Verify location has shop-specific activities
        assert "Shopping" in building.location.activities_available, "Shop should have Shopping activity"
        
        # Test building utility methods
        info = building.get_building_info()
        assert 'location' in info, "Building info should include location data"
        assert 'security' in info['location'], "Building info should include location security"
        
        entrance = building.get_entrance_point()
        assert entrance == (20, 20), f"Expected entrance at (20, 20), got {entrance}"
        
        print("✓ Building-Location integration test passed")
        
    except Exception as e:
        print(f"✗ Building-Location integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_point_of_interest_system():
    """Test the Points of Interest system"""
    print("Testing Points of Interest System...")
    
    try:
        from tiny_locations import PointOfInterest
        
        # Test POI creation
        poi = PointOfInterest(
            name="Village Well",
            x=50, y=50,
            poi_type="well",
            interaction_radius=8,
            action_system=MockActionSystem(),
            description="An old stone well"
        )
        
        # Test POI properties
        assert poi.name == "Village Well", f"Expected 'Village Well', got '{poi.name}'"
        assert poi.poi_type == "well", f"Expected 'well', got '{poi.poi_type}'"
        assert poi.max_users == 3, f"Expected 3 max users for well, got {poi.max_users}"
        
        # Test distance calculation
        distance = poi.distance_to_point(53, 54)
        expected_distance = 5.0  # sqrt(3^2 + 4^2)
        assert abs(distance - expected_distance) < 0.1, f"Expected distance ~5.0, got {distance}"
        
        # Test interaction capability
        character = MockCharacter("Test Character", 52, 52)
        can_interact = poi.can_interact(character)
        assert can_interact, "Character should be able to interact with nearby POI"
        
        # Test adding user
        success = poi.add_user(character)
        assert success, "Should successfully add user to POI"
        assert character in poi.current_users, "Character should be in current users"
        
        # Test POI info
        info = poi.get_info()
        assert info['available'] == True, "POI should still be available"
        assert info['current_users'] == 1, "POI should have 1 current user"
        
        print("✓ Points of Interest system test passed")
        
    except Exception as e:
        print(f"✗ Points of Interest system test failed: {e}")
        return False
    
    return True

def test_location_ai_utility_methods():
    """Test Location utility methods for AI decision making"""
    print("Testing Location AI Utility Methods...")
    
    try:
        from tiny_locations import Location
        
        # Create test location
        location = Location(
            name="Test Park",
            x=0, y=0, width=50, height=50,
            action_system=MockActionSystem(),
            security=5, popularity=3
        )
        location.threat_level = 1
        location.add_activity("Walking")
        location.add_activity("Relaxing")
        
        # Test safety score calculation
        safety_score = location.get_safety_score()
        expected_safety = 5 - 1  # security - threat_level
        assert safety_score >= expected_safety, f"Expected safety score >= {expected_safety}, got {safety_score}"
        
        # Test attractiveness score
        attractiveness = location.get_attractiveness_score()
        assert attractiveness > 0, f"Expected positive attractiveness, got {attractiveness}"
        
        # Test character suitability
        character = MockCharacter("Test Character")
        character.safety_threshold = 2
        is_suitable = location.is_suitable_for_character(character)
        assert is_suitable, "Location should be suitable for character with low safety threshold"
        
        character.safety_threshold = 10  # Very high safety requirement
        is_suitable = location.is_suitable_for_character(character)
        assert not is_suitable, "Location should not be suitable for character with very high safety threshold"
        
        # Test recommended activities
        character.energy = 20  # Low energy
        recommended = location.get_recommended_activities_for_character(character)
        assert isinstance(recommended, list), "Recommended activities should be a list"
        
        print("✓ Location AI utility methods test passed")
        
    except Exception as e:
        print(f"✗ Location AI utility methods test failed: {e}")
        return False
    
    return True

def test_location_manager():
    """Test LocationManager functionality"""
    print("Testing LocationManager...")
    
    try:
        from tiny_locations import Location, LocationManager
        
        manager = LocationManager()
        
        # Create test locations
        location1 = Location("Park", 0, 0, 20, 20, MockActionSystem())
        location2 = Location("Shop", 30, 30, 15, 15, MockActionSystem())
        
        manager.add_location(location1)
        manager.add_location(location2)
        
        # Test finding locations by point
        locations_at_point = manager.find_locations_containing_point(10, 10)
        assert len(locations_at_point) == 1, f"Expected 1 location at (10,10), got {len(locations_at_point)}"
        assert locations_at_point[0].name == "Park", f"Expected 'Park', got '{locations_at_point[0].name}'"
        
        # Test finding overlapping locations
        overlap_location = Location("Overlap", 10, 10, 25, 25, MockActionSystem())
        overlapping = manager.find_overlapping_locations(overlap_location)
        assert len(overlapping) >= 1, f"Expected at least 1 overlapping location, got {len(overlapping)}"
        
        print("✓ LocationManager test passed")
        
    except Exception as e:
        print(f"✗ LocationManager test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all integration tests"""
    print("Running Map and Location System Integration Tests...")
    print("=" * 60)
    
    tests = [
        test_building_location_integration,
        test_point_of_interest_system,
        test_location_ai_utility_methods,
        test_location_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! ✓")
        return True
    else:
        print(f"{failed} test(s) failed! ✗")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)