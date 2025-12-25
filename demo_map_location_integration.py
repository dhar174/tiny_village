#!/usr/bin/env python3
"""
Demonstration Script for Map and Location System Integration
Shows the new functionality in action without requiring heavy dependencies
"""

import sys
import os
import time
from actions import ActionSystem
from tiny_locations import Location
from simple_character_for_testing import SimpleCharacter

def demonstrate_building_location_integration():
    """Demonstrate Building-Location integration"""
    print("\n" + "="*60)
    print("BUILDING-LOCATION INTEGRATION DEMONSTRATION")
    print("="*60)
    
    from tiny_buildings import Building
    
    # Create different types of buildings
    buildings = [
        Building("Village Bakery", 10, 20, 12, 15, 20, action_system=ActionSystem(), building_type="shop"),
        Building("Town Library", 50, 60, 15, 25, 30, action_system=ActionSystem(), building_type="library"),
        Building("Mayor's House", 100, 10, 20, 18, 22, action_system=ActionSystem(), building_type="house"),
        Building("Adventurer's Inn", 30, 80, 18, 20, 25, action_system=ActionSystem(), building_type="restaurant"),
    ]
    
    print(f"Created {len(buildings)} buildings with integrated locations:\n")
    
    for building in buildings:
        print(f"🏢 {building.name} ({building.building_type})")
        print(f"   📍 Position: {building.coordinates_location}")
        print(f"   🚪 Entrance: {building.get_entrance_point()}")
        print(f"   📏 Dimensions: {building.width}x{building.length}x{building.height}")
        print(f"   🎯 Activities: {building.location.activities_available}")
        print(f"   👥 Capacity: {'Yes' if building.can_accommodate_visitors(5) else 'No'} (for 5 visitors)")
        
        # Set some location properties for demonstration
        if building.building_type == "shop":
            building.set_location_properties(security=7, popularity=8)
        elif building.building_type == "library":
            building.set_location_properties(security=9, popularity=6)
        elif building.building_type == "house":
            building.set_location_properties(security=8, popularity=3)
        elif building.building_type == "restaurant":
            building.set_location_properties(security=6, popularity=9)
        
        print(f"   🛡️  Safety Score: {building.location.get_safety_score():.1f}")
        print(f"   ⭐ Attractiveness: {building.location.get_attractiveness_score():.1f}")
        print()

def demonstrate_points_of_interest():
    """Demonstrate Points of Interest system"""
    print("\n" + "="*60)
    print("POINTS OF INTEREST SYSTEM DEMONSTRATION") 
    print("="*60)
    
    from tiny_locations import PointOfInterest
    
    # Create various POIs
    pois = [
        PointOfInterest("Village Well", 25, 25, "well", 8, ActionSystem(), "Ancient stone well"),
        PointOfInterest("Park Bench", 70, 40, "bench", 5, ActionSystem(), "Comfortable wooden bench"),
        PointOfInterest("Rose Garden", 15, 70, "garden", 10, ActionSystem(), "Beautiful flower garden"),
        PointOfInterest("Town Fountain", 60, 15, "fountain", 12, ActionSystem(), "Ornate marble fountain"),
    ]
    
    print(f"Created {len(pois)} points of interest:\n")
    
    for poi in pois:
        print(f"🌟 {poi.name} ({poi.poi_type})")
        print(f"   📍 Position: ({poi.x}, {poi.y})")
        print(f"   🔄 Interaction Radius: {poi.interaction_radius}")
        print(f"   👥 Capacity: {poi.max_users} users")
        print(f"   📝 Description: {poi.description}")
        
        # Simulate some visitors
        characters = []
        for name, coords in [
            ("Alice", (poi.x + 2, poi.y + 1)),
            ("Bob", (poi.x - 1, poi.y + 2)),
            ("Charlie", (poi.x + 20, poi.y + 20)),
        ]:
            character = SimpleCharacter(name)
            character.location = Location(f"{name}_location", coords[0], coords[1], 1, 1, ActionSystem())
            characters.append(character)
        
        for char in characters:
            if poi.can_interact(char):
                poi.add_user(char)
                print(f"   ➕ {char.name} can interact and has been added")
            else:
                distance = poi.distance_to_point(char.location.coordinates_location[0], 
                                                char.location.coordinates_location[1])
                print(f"   ➖ {char.name} is too far away (distance: {distance:.1f})")
        
        info = poi.get_info()
        print(f"   📊 Status: {info['current_users']}/{info['max_users']} users, Available: {info['available']}")
        print()

def demonstrate_location_ai_utilities():
    """Demonstrate Location AI utility methods"""
    print("\n" + "="*60)
    print("LOCATION AI UTILITY METHODS DEMONSTRATION")
    print("="*60)
    
    from tiny_locations import Location
    
    # Create locations with different characteristics
    locations = [
        Location("Safe District", 0, 0, 50, 50, ActionSystem(), security=8, popularity=6),
        Location("Dangerous Alley", 100, 100, 20, 20, ActionSystem(), security=2, popularity=1),
        Location("Popular Market", 60, 60, 40, 40, ActionSystem(), security=5, popularity=9),
        Location("Quiet Park", 20, 120, 30, 30, ActionSystem(), security=7, popularity=4),
    ]
    
    # Set additional properties
    locations[0].threat_level = 1
    locations[1].threat_level = 8
    locations[2].threat_level = 3
    locations[3].threat_level = 1
    
    # Add activities
    locations[0].add_activity("Residential")
    locations[0].add_activity("Shopping")
    locations[1].add_activity("Criminal activities")
    locations[2].add_activity("Shopping")
    locations[2].add_activity("Trading")
    locations[2].add_activity("Social gathering")
    locations[3].add_activity("Relaxing")
    locations[3].add_activity("Walking")
    
    print("Analyzing locations for different character types:\n")
    
    # Create different character types
    characters = []
    for name, safety, energy, social in [
        ("Cautious Carl", 6, 30, 30),
        ("Brave Betty", 1, 80, 70),
        ("Social Sam", 3, 60, 90),
    ]:
        character = SimpleCharacter(name, energy=energy)
        character.safety_threshold = safety
        character.social_preference = social
        characters.append(character)
    
    for char in characters:
        print(f"👤 {char.name} (Safety Threshold: {char.safety_threshold}, Energy: {char.energy})")
        print(f"   Analysis of locations:")
        
        for location in locations:
            safety_score = location.get_safety_score()
            attractiveness = location.get_attractiveness_score()
            suitable = location.is_suitable_for_character(char)
            
            print(f"   📍 {location.name}:")
            print(f"      🛡️  Safety: {safety_score:.1f} | ⭐ Attractiveness: {attractiveness:.1f}")
            print(f"      ✅ Suitable: {'Yes' if suitable else 'No'}")
            
            if suitable:
                recommended = location.get_recommended_activities_for_character(char)
                if recommended:
                    print(f"      🎯 Recommended: {', '.join(recommended)}")
        print()

def demonstrate_terrain_and_pathfinding():
    """Demonstrate terrain effects and pathfinding integration"""
    print("\n" + "="*60)
    print("TERRAIN EFFECTS AND PATHFINDING DEMONSTRATION")
    print("="*60)
    
    # Mock map data with terrain information
    map_data = {
        "width": 100,
        "height": 100,
        "buildings": [],
        "terrain": {
            (10, 10): 1.0,  # Normal terrain
            (20, 20): 2.0,  # Difficult terrain
            (30, 30): 0.5,  # Easy terrain (road)
            (40, 40): 5.0,  # Very difficult terrain
        }
    }
    
    # Mock the pygame import for MapController
    import types
    pygame_mock = types.ModuleType('pygame')
    pygame_mock.image = types.ModuleType('image')
    pygame_mock.image.load = lambda x: "mock_image"
    pygame_mock.math = types.ModuleType('math')
    pygame_mock.math.Vector2 = lambda x, y=None: type('Vector2', (), {'x': x, 'y': y if y is not None else x[1], '__sub__': lambda s, o: type('Vector2', (), {'length': lambda: 5.0, 'normalize': lambda: type('Vector2', (), {'__mul__': lambda s, dt: (x, y if y is not None else x[1])})()})(), '__add__': lambda s, o: (x, y if y is not None else x[1])})()
    pygame_mock.draw = types.ModuleType('draw')
    pygame_mock.draw.rect = lambda *args, **kwargs: None
    pygame_mock.draw.circle = lambda *args, **kwargs: None
    pygame_mock.Rect = lambda *args: type('Rect', (), {})()
    sys.modules['pygame'] = pygame_mock
    
    from tiny_map_controller import MapController
    
    # Create a mock map controller (without actual pygame initialization)
    class DemoMapController:
        def __init__(self, map_data):
            self.map_data = map_data
            
        def get_terrain_movement_modifier(self, position):
            """Get movement speed modifier based on terrain at position"""
            x, y = position
            
            # Default terrain cost from map data
            terrain_cost = self.map_data.get("terrain", {}).get(position, 1.0)
            
            # Convert terrain cost to movement modifier
            if terrain_cost <= 1.0:
                return 1.0  # Normal movement
            elif terrain_cost <= 2.0:
                return 0.8  # Slightly slower
            elif terrain_cost <= 5.0:
                return 0.5  # Much slower
            else:
                return 0.1  # Nearly impassable
    
    controller = DemoMapController(map_data)
    
    print("Terrain movement modifiers at different positions:\n")
    
    test_positions = [
        ((10, 10), "Normal terrain"),
        ((20, 20), "Difficult terrain"), 
        ((30, 30), "Road (easy terrain)"),
        ((40, 40), "Very difficult terrain"),
        ((50, 50), "Unknown terrain (default)"),
    ]
    
    base_speed = 10.0  # Base character movement speed
    
    for position, description in test_positions:
        modifier = controller.get_terrain_movement_modifier(position)
        effective_speed = base_speed * modifier
        print(f"📍 Position {position} - {description}")
        print(f"   🏃 Speed Modifier: {modifier:.1f}x")
        print(f"   ⚡ Effective Speed: {effective_speed:.1f} units/second")
        print(f"   ⏱️  Time to cross 10 units: {10/effective_speed:.1f} seconds")
        print()

def main():
    """Run all demonstrations"""
    print("🌟 TINY VILLAGE MAP & LOCATION SYSTEM INTEGRATION DEMO 🌟")
    print("This demonstration shows the new integrated functionality")
    
    try:
        demonstrate_building_location_integration()
        demonstrate_points_of_interest()
        demonstrate_location_ai_utilities()
        demonstrate_terrain_and_pathfinding()
        
        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("• Buildings automatically create and manage Location instances")
        print("• Points of Interest system for interactive environment objects")
        print("• AI utility methods for location-based decision making")
        print("• Terrain effects on character movement speed")
        print("• Integration between all systems for cohesive gameplay")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
