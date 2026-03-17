#!/usr/bin/env python3
"""
Demo script showcasing the Building Functionality System.

This script demonstrates:
1. Building creation with different types
2. Resource production over time
3. Characters using building services
4. Economic transactions
5. Complete village economy simulation
"""

import sys
sys.path.insert(0, '.')

from tiny_building_manager import BuildingManager, ResourceType
from tiny_buildings import Building


class MockCharacter:
    """Simple character for demo purposes."""
    def __init__(self, name, wealth_money=100, **attributes):
        self.name = name
        self.wealth_money = wealth_money
        self.hunger_level = attributes.get('hunger', 5)
        self.thirst = attributes.get('thirst', 50)
        self.energy = attributes.get('energy', 50)
        self.social_wellbeing = attributes.get('social_wellbeing', 50)
        self.current_satisfaction = attributes.get('satisfaction', 50)
        self.intelligence = attributes.get('intelligence', 50)
        self.knowledge = attributes.get('knowledge', 50)
        
        # Mock skills
        class Skills:
            def __init__(self):
                self.crafting = 20
        self.skills = Skills()


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_building_status(building, manager):
    """Print building status."""
    resources = building.get_resource_levels()
    print(f"  {building.name} ({building.building_type}):")
    print(f"    Resources: Food={resources['food']}, Materials={resources['materials']}, "
          f"Tools={resources['tools']}, Goods={resources['goods']}")
    print(f"    Services={resources['services']}, Knowledge={resources['knowledge']}")


def print_character_status(character):
    """Print character status."""
    print(f"  {character.name}:")
    print(f"    Wealth: ${character.wealth_money}")
    print(f"    Hunger: {character.hunger_level}/10, Thirst: {character.thirst}/100")
    print(f"    Energy: {character.energy}/100, Satisfaction: {character.current_satisfaction}/100")


def main():
    """Run the building functionality demo."""
    
    print_header("Tiny Village - Building Functionality Demo")
    
    # Initialize the building manager
    print("Initializing Building Manager...")
    manager = BuildingManager()
    
    # Create village buildings
    print("\nCreating village buildings...")
    
    farm = Building(
        "Sunny Farm", 10, 10, 10, 30, 25,
        building_type="farm",
        building_manager=manager
    )
    print(f"  ✓ Created {farm.name}")
    
    market = Building(
        "Village Market", 50, 50, 10, 20, 15,
        building_type="market",
        building_manager=manager
    )
    print(f"  ✓ Created {market.name}")
    
    tavern = Building(
        "Cozy Tavern", 90, 90, 10, 20, 15,
        building_type="tavern",
        building_manager=manager
    )
    print(f"  ✓ Created {tavern.name}")
    
    blacksmith = Building(
        "Iron Forge", 130, 130, 10, 15, 15,
        building_type="blacksmith",
        building_manager=manager
    )
    print(f"  ✓ Created {blacksmith.name}")
    
    school = Building(
        "Village School", 170, 170, 10, 25, 20,
        building_type="school",
        building_manager=manager
    )
    print(f"  ✓ Created {school.name}")
    
    # Show initial building resources
    print_header("Initial Building Resources")
    print_building_status(farm, manager)
    print_building_status(market, manager)
    print_building_status(tavern, manager)
    print_building_status(blacksmith, manager)
    print_building_status(school, manager)
    
    # Create villagers
    print_header("Creating Villagers")
    
    farmer = MockCharacter("Farmer Bob", wealth_money=75, hunger=7)
    print(f"  ✓ Created {farmer.name}")
    
    merchant = MockCharacter("Merchant Alice", wealth_money=150, hunger=4)
    print(f"  ✓ Created {merchant.name}")
    
    student = MockCharacter("Student Eve", wealth_money=50, intelligence=40)
    print(f"  ✓ Created {student.name}")
    
    craftsperson = MockCharacter("Blacksmith Joe", wealth_money=100)
    print(f"  ✓ Created {craftsperson.name}")
    
    # Show initial character status
    print_header("Initial Character Status")
    print_character_status(farmer)
    print_character_status(merchant)
    print_character_status(student)
    print_character_status(craftsperson)
    
    # Simulate production cycles
    print_header("Simulating Production Cycles (100 ticks)")
    print("Processing production for all buildings...")
    
    production_results = {
        'farm': 0, 'market': 0, 'tavern': 0, 'blacksmith': 0, 'school': 0
    }
    
    for tick in range(0, 100, 5):
        if farm.process_production(tick):
            production_results['farm'] += 1
        if market.process_production(tick):
            production_results['market'] += 1
        if tavern.process_production(tick):
            production_results['tavern'] += 1
        if blacksmith.process_production(tick):
            production_results['blacksmith'] += 1
        if school.process_production(tick):
            production_results['school'] += 1
    
    print(f"\n  Production cycles completed:")
    for building, count in production_results.items():
        print(f"    {building.capitalize()}: {count} cycles")
    
    # Show updated building resources
    print_header("Building Resources After Production")
    print_building_status(farm, manager)
    print_building_status(market, manager)
    print_building_status(tavern, manager)
    print_building_status(blacksmith, manager)
    print_building_status(school, manager)
    
    # Characters use building services
    print_header("Characters Using Building Services")
    
    print("\n1. Farmer Bob buys food from the farm:")
    success, message = farm.provide_service("buy_food", farmer)
    print(f"   Result: {message}")
    print(f"   Hunger before: 7, after: {farmer.hunger_level}")
    print(f"   Wealth before: $75, after: ${farmer.wealth_money}")
    
    print("\n2. Merchant Alice buys goods from the market:")
    initial_satisfaction = merchant.current_satisfaction
    success, message = market.provide_service("buy_goods", merchant)
    print(f"   Result: {message}")
    print(f"   Satisfaction before: {initial_satisfaction}, after: {merchant.current_satisfaction}")
    print(f"   Wealth before: $150, after: ${merchant.wealth_money}")
    
    print("\n3. Student Eve takes a lesson at the school:")
    initial_intelligence = student.intelligence
    success, message = school.provide_service("take_lesson", student)
    print(f"   Result: {message}")
    print(f"   Intelligence before: {initial_intelligence}, after: {student.intelligence}")
    print(f"   Wealth before: $50, after: ${student.wealth_money}")
    
    print("\n4. Merchant Alice sells goods at the market:")
    wealth_before_sell = merchant.wealth_money
    success, message = market.provide_service("sell_goods", merchant)
    print(f"   Result: {message}")
    print(f"   Wealth before: ${wealth_before_sell}, after: ${merchant.wealth_money}")
    
    print("\n5. Blacksmith Joe repairs tools:")
    success, message = blacksmith.provide_service("repair_tools", craftsperson)
    print(f"   Result: {message}")
    print(f"   Wealth before: $100, after: ${craftsperson.wealth_money}")
    
    print("\n6. Everyone visits the tavern for a meal:")
    for character in [farmer, merchant, student, craftsperson]:
        hunger_before = character.hunger_level
        success, message = tavern.provide_service("buy_meal", character)
        print(f"   {character.name}: Hunger {hunger_before} → {character.hunger_level}, Wealth: ${character.wealth_money}")
    
    # Final status
    print_header("Final Character Status")
    print_character_status(farmer)
    print_character_status(merchant)
    print_character_status(student)
    print_character_status(craftsperson)
    
    print_header("Final Building Resources")
    print_building_status(farm, manager)
    print_building_status(market, manager)
    print_building_status(tavern, manager)
    print_building_status(blacksmith, manager)
    print_building_status(school, manager)
    
    # Economic summary
    print_header("Economic Summary")
    
    total_wealth = sum([
        farmer.wealth_money,
        merchant.wealth_money,
        student.wealth_money,
        craftsperson.wealth_money
    ])
    
    print(f"  Total Village Wealth: ${total_wealth}")
    print(f"  Average Wealth: ${total_wealth / 4:.2f}")
    
    print("\n  Service History:")
    history = manager.service_history
    print(f"    Total transactions: {len(history)}")
    if history:
        print(f"    Recent transactions:")
        for transaction in history[-5:]:
            print(f"      - {transaction['character']} used {transaction['service']} at "
                  f"{transaction['building_type']} (cost: ${transaction['cost']})")
    
    print_header("Demo Complete!")
    print("The building functionality system successfully:")
    print("  ✓ Managed resources for 5 different building types")
    print("  ✓ Processed production cycles over time")
    print("  ✓ Handled character-building service interactions")
    print("  ✓ Executed economic transactions")
    print("  ✓ Tracked resource consumption and production")
    print("\nAll building types working as expected! 🎉")


if __name__ == "__main__":
    main()
