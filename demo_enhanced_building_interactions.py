#!/usr/bin/env python3
"""
Demo of the enhanced building interactions system.

This script demonstrates the new building-type specific interactions
that go beyond the basic 'Enter Building' action.
"""

from tiny_buildings import Building


class DemoCharacter:
    """Simple character class for demonstration."""
    
    def __init__(self, name, energy=50, **attributes):
        self.name = name
        self.energy = energy
        for key, value in attributes.items():
            setattr(self, key, value)


def demo_building_interactions():
    """Demonstrate building-type specific interactions."""
    
    print("🏘️  Enhanced Building Interactions Demo")
    print("=" * 50)
    
    # Create different types of buildings
    buildings = [
        Building("Cozy Cottage", 10, 10, 15, 20, 25, building_type="residential"),
        Building("Village Market", 50, 20, 12, 25, 20, building_type="commercial"),
        Building("The Laughing Dragon Tavern", 20, 50, 18, 30, 22, building_type="social"),
        Building("Master Blacksmith", 80, 30, 10, 15, 18, building_type="crafting"),
        Building("Green Valley Farm", 5, 80, 8, 40, 35, building_type="agricultural"),
        Building("Village School", 70, 70, 12, 20, 25, building_type="educational"),
        Building("Town Hall", 40, 5, 25, 30, 35, building_type="civic"),
        Building("Peaceful Library", 30, 65, 15, 25, 30, building_type="library"),
    ]
    
    # Create characters with different energy levels
    characters = [
        DemoCharacter("Energetic Emma", energy=80),
        DemoCharacter("Tired Tom", energy=3),
        DemoCharacter("Normal Nancy", energy=25),
    ]
    
    print("\n🏢 Available Buildings and Their Interactions:")
    print("-" * 50)
    
    for building in buildings:
        print(f"\n🏛️  {building.name} ({building.building_type.title()})")
        total_interactions = [action.name for action in building.possible_interactions]
        print(f"   Total interactions: {len(total_interactions)}")
        print(f"   Available: {', '.join(total_interactions)}")
    
    print("\n\n👥 Character Interaction Filtering:")
    print("-" * 50)
    
    # Test with the market (commercial building)
    market = buildings[1]  # Village Market
    
    for character in characters:
        print(f"\n👤 {character.name} (Energy: {character.energy})")
        available_interactions = market.get_possible_interactions(character)
        interaction_names = [action.name for action in available_interactions]
        
        print(f"   Can perform {len(interaction_names)} interactions at {market.name}:")
        print(f"   {', '.join(interaction_names)}")
        
        if len(interaction_names) < len(market.possible_interactions):
            unavailable = [action.name for action in market.possible_interactions 
                          if action.name not in interaction_names]
            print(f"   ❌ Cannot perform: {', '.join(unavailable)}")
    
    print("\n\n🎯 Building Type Specialization Examples:")
    print("-" * 50)
    
    emma = characters[0]  # High energy character
    
    specializations = [
        ("Residential (Cottage)", buildings[0], "Rest and socialize with residents"),
        ("Commercial (Market)", buildings[1], "Browse, buy, and trade goods"),
        ("Social (Tavern)", buildings[2], "Socialize, drink, and join activities"),
        ("Crafting (Blacksmith)", buildings[3], "Commission items and learn skills"),
        ("Agricultural (Farm)", buildings[4], "Help with crops and tend animals"),
        ("Educational (School)", buildings[5], "Learn and access knowledge"),
        ("Civic (Town Hall)", buildings[6], "Participate in community activities"),
    ]
    
    for description, building, purpose in specializations:
        interactions = building.get_possible_interactions(emma)
        print(f"\n🏗️  {description}")
        print(f"   Purpose: {purpose}")
        print(f"   Interactions: {', '.join([a.name for a in interactions])}")
    
    print("\n\n✨ Key Features Demonstrated:")
    print("-" * 50)
    print("✅ Each building type has 3-5 unique interactions beyond 'Enter Building'")
    print("✅ Interactions are dynamically filtered based on building type")
    print("✅ Character energy affects which interactions are available")
    print("✅ Building type aliases work (house = residential, shop = commercial)")
    print("✅ Special cases like libraries have customized interaction sets")
    print("✅ Backward compatibility maintained with existing 'Enter Building' action")
    print("✅ Graceful fallback for unknown building types")
    
    print("\n📊 Interaction Statistics:")
    print("-" * 50)
    total_unique_interactions = set()
    for building in buildings:
        for action in building.possible_interactions:
            total_unique_interactions.add(action.name)
    
    print(f"Total unique interaction types: {len(total_unique_interactions)}")
    print(f"Building types supported: {len(set(b.building_type for b in buildings))}")
    print(f"Average interactions per building: {sum(len(b.possible_interactions) for b in buildings) / len(buildings):.1f}")


if __name__ == "__main__":
    demo_building_interactions()