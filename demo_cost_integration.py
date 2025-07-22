#!/usr/bin/env python3
"""
Demonstration of Cost Function Integration in GOAP Planning System

This script demonstrates how the enhanced GOAP planner now properly
integrates and utilizes action costs in the planning search.
"""

from tiny_goap_system import GOAPPlanner
from actions import Action, State
from tiny_utility_functions import Goal


class DemoCharacter:
    """Simple character for demonstration"""
    def __init__(self, name="DemoChar"):
        self.name = name
        self.hunger = 0.9  # Very hungry
        self.energy = 0.3  # Low energy
        self.money = 50
        
    def get_state(self):
        return State({"hunger": self.hunger, "energy": self.energy, "money": self.money})


def demonstrate_cost_based_planning():
    """Demonstrate cost-based planning with different action costs"""
    
    print("=== Cost Function Integration Demonstration ===\n")
    
    # Create planner and character
    planner = GOAPPlanner(None)
    character = DemoCharacter()
    
    print(f"Character initial state:")
    print(f"  Hunger: {character.hunger} (0.0=full, 1.0=very hungry)")
    print(f"  Energy: {character.energy} (0.0=exhausted, 1.0=energized)")
    print(f"  Money: {character.money}")
    print()
    
    # Create goal to reduce hunger
    goal = Goal("ReduceHunger", target_effects={"hunger": 0.2}, priority=1.0)
    print(f"Goal: Reduce hunger to {goal.target_effects['hunger']}")
    print()
    
    # Create initial state
    current_state = character.get_state()
    
    # Create actions with different cost/utility profiles
    actions = [
        Action(
            name="ExpensiveMeal",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.8}],
            cost=3.0  # High cost, high effect
        ),
        Action(
            name="RegularMeal", 
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.7}],
            cost=1.0  # Medium cost, good effect
        ),
        Action(
            name="CheapSnack",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.6}],
            cost=0.2  # Low cost, decent effect
        ),
        Action(
            name="TinyBite",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.1}],
            cost=0.1  # Very low cost, minimal effect
        )
    ]
    
    print("Available actions:")
    for action in actions:
        effect = action.effects[0]['change_value']
        print(f"  {action.name}: Cost={action.cost}, Hunger reduction={abs(effect)}")
    print()
    
    # Find plan using cost-aware algorithm
    print("Planning with cost-aware GOAP algorithm...")
    plan = planner.plan_actions(character, goal, current_state, actions)
    
    if plan:
        print(f"✓ Found plan with {len(plan)} action(s):")
        total_cost = 0
        simulated_hunger = character.hunger
        
        for i, action in enumerate(plan, 1):
            effect = action.effects[0]['change_value']
            simulated_hunger += effect
            total_cost += action.cost
            print(f"  {i}. {action.name}")
            print(f"     Cost: {action.cost}")
            print(f"     Effect: {effect} (hunger becomes {simulated_hunger:.2f})")
        
        print(f"\nPlan summary:")
        print(f"  Total cost: {total_cost}")
        print(f"  Final hunger: {simulated_hunger:.2f}")
        print(f"  Goal achieved: {'✓' if simulated_hunger <= goal.target_effects['hunger'] + 0.1 else '✗'}")
        
    else:
        print("✗ No plan found")
    
    print()


def demonstrate_utility_vs_cost_tradeoff():
    """Demonstrate how utility and cost are balanced in planning"""
    
    print("=== Utility vs Cost Tradeoff Demonstration ===\n")
    
    planner = GOAPPlanner(None)
    character = DemoCharacter()
    character.hunger = 0.95  # Extremely hungry
    
    print(f"Character in extreme hunger state: {character.hunger}")
    print()
    
    # Goal to reduce hunger significantly
    goal = Goal("SurvivalMeal", target_effects={"hunger": 0.1}, priority=1.0)
    current_state = character.get_state()
    
    # Actions with very different cost/utility profiles
    actions = [
        Action(
            name="FastFood",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.9}],  # Solves problem in one go
            cost=5.0  # Expensive but efficient
        ),
        Action(
            name="GranolaBites",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.3}],  # Multiple needed
            cost=0.5  # Cheap but need multiple
        )
    ]
    
    print("Action comparison:")
    for action in actions:
        effect = action.effects[0]['change_value']
        efficiency = abs(effect) / action.cost
        print(f"  {action.name}: Effect={effect}, Cost={action.cost}, Efficiency={efficiency:.2f}")
    print()
    
    plan = planner.plan_actions(character, goal, current_state, actions)
    
    if plan:
        print("Cost-aware planner decision:")
        total_cost = sum(action.cost for action in plan)
        total_effect = sum(action.effects[0]['change_value'] for action in plan)
        
        print(f"  Selected plan: {[action.name for action in plan]}")
        print(f"  Total actions: {len(plan)}")
        print(f"  Total cost: {total_cost}")
        print(f"  Total hunger reduction: {abs(total_effect)}")
        print(f"  Cost efficiency: {abs(total_effect) / total_cost:.2f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    demonstrate_cost_based_planning()
    demonstrate_utility_vs_cost_tradeoff()
    print("Demonstration complete! The GOAP planner now properly")
    print("integrates action costs and utility functions in planning search.")