#!/usr/bin/env python3
"""
Simple debug test for StrategyManager and GOAPPlanner interaction.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import Goal
from actions import Action, State


class MockCharacter:
    def __init__(self, name="TestChar"):
        self.name = name
        self.hunger_level = 5.0
        self.energy = 5.0
        self.wealth_money = 50.0
        self.social_wellbeing = 5.0
        self.mental_health = 5.0
        self.location = None
        self.job = "unemployed"


def simple_test():
    print("=== Simple StrategyManager Test ===\n")
    
    try:
        # Create StrategyManager
        sm = StrategyManager()
        print("✓ StrategyManager created")
        
        # Create character
        character = MockCharacter("Alice")
        print("✓ Character created")
        
        # Test character state extraction
        state_dict = sm.get_character_state_dict(character)
        print(f"✓ Character state: {state_dict}")
        
        # Test action generation
        actions = sm.get_daily_actions(character)
        print(f"✓ Generated {len(actions)} actions")
        for action in actions:
            print(f"  - {action.name}")
        
        # Test Goal creation
        goal = Goal(
            name="test_goal", 
            target_effects={"energy": 0.7},
            priority=0.8
        )
        print(f"✓ Goal created: {goal.name} with target_effects: {goal.target_effects}")
        
        # Test State creation
        current_state = State(state_dict)
        print(f"✓ State created with data: {current_state}")
        
        # Test direct GOAP planning (simple)
        print("\n--- Testing GOAP Planning ---")
        if actions:
            print("Calling GOAP planner...")
            plan = sm.goap_planner.plan_actions(character, goal, current_state, actions)
            
            if plan:
                print(f"✓ GOAP planning successful - {len(plan)} actions")
                for action in plan:
                    print(f"  - {action.name}")
            else:
                print("✓ GOAP planning completed (goal satisfied or no plan needed)")
        else:
            print("⚠ No actions available for planning")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n✅ Simple test passed!")
    else:
        print("\n❌ Simple test failed!")
    sys.exit(0 if success else 1)