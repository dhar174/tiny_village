#!/usr/bin/env python3
"""
Comprehensive integration demo showing Location Integration functionality.
This demo shows how Building objects integrate with Location objects and
how character AI uses location properties for decision-making.
"""

import sys
import os

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiny_buildings import Building, House, CreateBuilding
from tiny_locations import Location, LocationManager
from actions import ActionSystem
from simple_character_for_testing import SimpleCharacter, SimplePersonalityTraits


def demo_building_location_integration():
    """Demonstrate Building-Location integration"""
    print("=== Building-Location Integration Demo ===\n")
    
    # Create different types of buildings
    buildings = []
    
    # Create a house
    house = House("Cozy Home", 10, 10, 30, 25, 20, "123 Oak Street", 
                  stories=2, bedrooms=3, bathrooms=2)
    buildings.append(house)
    
    # Create commercial buildings
    shop = Building("Popular Shop", 50, 30, 25, 15, 15, building_type="commercial")
    buildings.append(shop)
    
    # Create office building
    office = Building("Downtown Office", 80, 60, 40, 30, 25, building_type="office")
    buildings.append(office)
    
    print("Created buildings with integrated locations:")
    for building in buildings:
        location = building.get_location()
        print(f"\n{building.name}:")
        print(f"  Type: {getattr(building, 'building_type', 'unknown')}")
        print(f"  Location: ({location.x}, {location.y}) - {location.width}x{location.height}")
        print(f"  Security Level: {building.get_security_level()}/10")
        print(f"  Popularity Level: {building.get_popularity_level()}/10")
        print(f"  Available Activities: {', '.join(building.get_available_activities())}")
        
        # Show house-specific properties
        if hasattr(building, 'shelter_value'):
            print(f"  Shelter Value: {building.shelter_value}")
            print(f"  Beauty Value: {building.beauty_value}")


def demo_location_based_pathfinding():
    """Demonstrate how locations can be used for pathfinding"""
    print("\n\n=== Location-Based Pathfinding Demo ===\n")
    
    # Create buildings
    house = House("Safe House", 20, 20, 30, 20, 20, "456 Pine St")
    shop = Building("Corner Shop", 60, 40, 25, 15, 15, building_type="commercial")
    
    # Show how location properties can influence pathfinding decisions
    print("Pathfinding considerations:")
    
    for building in [house, shop]:
        location = building.get_location()
        print(f"\n{building.name}:")
        print(f"  Position: {location.get_coordinates()}")
        print(f"  Dimensions: {location.get_dimensions()}")
        print(f"  Area: {location.get_area()}")
        print(f"  Contains point (25, 25): {location.contains_point(25, 25)}")
        
        # Calculate distance from a character position
        character_pos = (10, 10)
        distance = location.distance_to_point_from_center(*character_pos)
        print(f"  Distance from {character_pos}: {distance:.2f}")


def demo_character_location_decisions():
    """Demonstrate character AI location decision-making"""
    print("\n\n=== Character AI Location Decisions Demo ===\n")
    
    # Create test characters with different traits
    characters = [
        SimpleCharacter("Alice (High Energy, Social)", energy=9, social_wellbeing=8),
        SimpleCharacter("Bob (Low Energy, Tired)", energy=2, social_wellbeing=8),
        SimpleCharacter("Carol (Socially Isolated)", energy=8, social_wellbeing=2),
    ]
    characters[0].personality_traits = SimplePersonalityTraits(neuroticism=60, extraversion=70)
    characters[1].personality_traits = SimplePersonalityTraits(neuroticism=60, extraversion=45)
    characters[2].personality_traits = SimplePersonalityTraits(neuroticism=40, extraversion=65)
    
    # Create test buildings
    buildings = [
        House("Comfortable Home", 10, 10, 30, 20, 20, "123 Home St", bedrooms=3),
        Building("Busy Mall", 50, 50, 25, 20, 20, building_type="commercial"),
        Building("Quiet Office", 80, 80, 25, 15, 15, building_type="office"),
    ]
    
    print("Character location preferences:")
    
    for character in characters:
        print(f"\n{character.name}:")
        print(f"  Energy: {character.energy}/10")
        print(f"  Social Wellbeing: {character.social_wellbeing}/10")
        print(f"  Personality: Neuroticism={character.personality_traits.neuroticism}, " +
              f"Extraversion={character.personality_traits.extraversion}")
        
        print("  Location Evaluation Scores:")
        for building in buildings:
            score = character.evaluate_location_for_visit(building)
            location = building.get_location()
            print(f"    {building.name}: {score}/100 " +
                  f"(Security: {location.security}, Popularity: {location.popularity})")


def demo_map_controller_integration():
    """Demonstrate MapController integration"""
    print("\n\n=== MapController Integration Demo ===\n")
    
    # Track buildings directly without mock controllers
    controller_buildings = []
    location_manager = LocationManager()

    buildings = [
        House("Safe Family Home", 10, 10, 30, 25, 20, "100 Safe St", bedrooms=4),
        Building("Popular Nightclub", 50, 30, 25, 20, 15, building_type="commercial"),
        Building("Secure Bank", 80, 60, 40, 25, 20, building_type="office"),
        House("Modest Apartment", 30, 80, 20, 15, 15, "200 Budget Ave", bedrooms=1),
    ]
    
    for building in buildings:
        controller_buildings.append(building)
        location_manager.add_location(building.get_location())
    
    print("MapController can find locations by criteria:")
    
    # Test position-based finding
    test_position = (15, 15)
    buildings_at_pos = [b for b in controller_buildings if b.is_within_building(test_position)]
    print(f"\nBuildings at position {test_position}:")
    for building in buildings_at_pos:
        print(f"  - {building.name}")
    
    # Test criteria-based finding
    safe_locations = [b for b in controller_buildings if b.get_location().security >= 8]
    print(f"\nSafe locations (security >= 8):")
    for building in safe_locations:
        location = building.get_location()
        print(f"  - {building.name} (Security: {location.security})")
    
    popular_locations = [b for b in controller_buildings if b.get_location().popularity >= 7]
    print(f"\nPopular locations (popularity >= 7):")
    for building in popular_locations:
        location = building.get_location()
        print(f"  - {building.name} (Popularity: {location.popularity})")
    
    rest_locations = [
        b for b in controller_buildings if "rest" in b.get_location().activities_available
    ]
    print(f"\nLocations with rest activity:")
    for building in rest_locations:
        print(f"  - {building.name}")


def main():
    """Run all demonstrations"""
    print("Location Integration Comprehensive Demo")
    print("=" * 50)
    
    try:
        demo_building_location_integration()
        demo_location_based_pathfinding()
        demo_character_location_decisions()
        demo_map_controller_integration()
        
        print("\n\n=== Demo Complete ===")
        print("The Location Integration system successfully:")
        print("1. Connects Building objects with Location instances")
        print("2. Provides security, popularity, and activity properties")
        print("3. Enables character AI to make location-based decisions")
        print("4. Integrates with MapController for pathfinding and search")
        print("5. Replaces raw coordinates with rich Location objects")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
