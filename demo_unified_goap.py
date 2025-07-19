#!/usr/bin/env python3
"""
Demo script showing the unified GOAP planning system in action.
This demonstrates how the static and instance methods now work together.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from tiny_goap_system import GOAPPlanner
from actions import Action, State


def demo_unified_goap():
    """Demonstrate the unified GOAP planning system."""
    print("=== GOAP Unified Planning Demo ===\n")
    
    # Create a simple character state
    class SimpleCharacter:
        def __init__(self):
            self.name = "Alice"
            self.energy = 60
            self.hunger = 80  # High hunger (needs food)
            self.happiness = 40

    # Create a simple goal
    class SimpleGoal:
        def __init__(self, requirements):
            self.requirements = requirements
            
        def check_completion(self, state):
            return all(state.get(k, 0) >= v for k, v in self.requirements.items())

    # Setup test scenario
    character = SimpleCharacter()
    goal = SimpleGoal({'energy': 70, 'hunger': 50})  # Want more energy, less hunger
    current_state = State({
        'energy': character.energy,
        'hunger': character.hunger,
        'happiness': character.happiness
    })

    print(f"Character: {character.name}")
    print(f"Current state: Energy={character.energy}, Hunger={character.hunger}, Happiness={character.happiness}")
    print(f"Goal: Energy >= 70, Hunger <= 50\n")

    # Create available actions
    eat_action = Action(
        name="Eat Food",
        preconditions=[],  # No prerequisites
        effects=[
            {'attribute': 'hunger', 'change_value': -20, 'targets': ['initiator']},
            {'attribute': 'energy', 'change_value': 5, 'targets': ['initiator']}
        ],
        cost=3
    )

    rest_action = Action(
        name="Take Rest",
        preconditions=[],
        effects=[
            {'attribute': 'energy', 'change_value': 15, 'targets': ['initiator']},
            {'attribute': 'hunger', 'change_value': 5, 'targets': ['initiator']}  # Rest makes you hungrier
        ],
        cost=5
    )

    actions = [eat_action, rest_action]
    print("Available actions:")
    for action in actions:
        print(f"  - {action.name} (cost: {action.cost})")
    print()

    # Test the unified planner
    planner = GOAPPlanner(None)
    
    print("Planning with instance method (plan_actions):")
    plan = planner.plan_actions(character, goal, current_state, actions)
    
    if plan:
        print(f"Found plan with {len(plan)} actions:")
        for i, action in enumerate(plan, 1):
            print(f"  {i}. {action.name}")
    else:
        print("No plan found or goal already satisfied")
    print()

    print("Planning with static method (goap_planner):")
    static_plan = GOAPPlanner.goap_planner(character, goal, current_state, actions)
    
    if static_plan:
        print(f"Found plan with {len(static_plan)} actions:")
        for i, action in enumerate(static_plan, 1):
            print(f"  {i}. {action.name}")
    else:
        print("No plan found or goal already satisfied")
    print()

    # Verify unification
    if plan == static_plan:
        print("✓ SUCCESS: Both methods return identical plans!")
        print("✓ Core GOAP planner logic has been successfully unified!")
    else:
        print("⚠ WARNING: Methods return different plans")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_unified_goap()