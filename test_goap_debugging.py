#!/usr/bin/env python3
"""
Debugging test to understand why GOAP planning isn't finding valid plans.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import Goal
from actions import State


def test_goap_debugging():
    """Test GOAP planning with debugging output."""
    print("=== GOAP Planning Debug Test ===\n")
    
    # Create strategy manager
    strategy_manager = StrategyManager()
    
    # Create simple character state
    character_state = {
        "satisfaction": 50.0,
        "energy": 50.0,
        "happiness": 40.0
    }
    current_state = State(character_state)
    print(f"Initial state: {current_state}")
    
    # Create a simple goal
    goal = Goal(
        name="improve_satisfaction",
        target_effects={"satisfaction": 55.0},  # Only small increase needed
        priority=0.8
    )
    print(f"Goal: {goal.name} - Target effects: {goal.target_effects}")
    
    # Get actions
    actions = strategy_manager.get_daily_actions("TestChar")
    print(f"Available actions: {[a.name for a in actions]}")
    
    # Test goal satisfaction with current state
    print(f"\nGoal satisfied with current state? {strategy_manager.goap_planner._goal_satisfied(goal, current_state)}")
    
    # Test applying an action
    if actions:
        test_action = actions[0]  # Use first action
        print(f"\nTesting action: {test_action.name}")
        print(f"Action effects: {test_action.effects}")
        
        # Apply action effects
        new_state = strategy_manager.goap_planner._apply_action_effects(test_action, current_state)
        print(f"State after action: {new_state}")
        print(f"Goal satisfied after action? {strategy_manager.goap_planner._goal_satisfied(goal, new_state)}")
    
    # Try planning
    print(f"\nTrying GOAP planning...")
    plan = strategy_manager.goap_planner.plan_actions("TestChar", goal, current_state, actions)
    if plan:
        print(f"Plan found: {[a.name for a in plan]}")
    else:
        print("No plan found")
    
    # Test with higher goal that requires multiple actions
    print(f"\n=== Testing with higher goal ===")
    big_goal = Goal(
        name="high_satisfaction",
        target_effects={"satisfaction": 80.0},
        priority=0.9
    )
    print(f"Big goal: {big_goal.name} - Target effects: {big_goal.target_effects}")
    
    # Test with exact same goal as plan_daily_activities uses
    print(f"\n=== Testing with daily activities goal ===")
    daily_goal = Goal(
        name="daily_wellbeing",
        target_effects={"satisfaction": 60.0, "energy": 65.0},  # Updated constants
        priority=0.8
    )
    print(f"Daily goal: {daily_goal.name} - Target effects: {daily_goal.target_effects}")
    
    daily_plan = strategy_manager.goap_planner.plan_actions("TestChar", daily_goal, current_state, actions)
    if daily_plan:
        print(f"Daily plan found: {[a.name for a in daily_plan]}")
        
        # Simulate executing the plan
        sim_state = State(character_state.copy())
        for action in daily_plan:
            print(f"  Applying {action.name}...")
            sim_state = strategy_manager.goap_planner._apply_action_effects(action, sim_state)
            print(f"    State: {sim_state}")
    else:
        print("No daily plan found")
        
        # Debug: Let's see if the actions even modify the right attributes
        print("Debug: Testing individual action effects...")
        for action in actions:
            if hasattr(action, 'effects') and action.effects:
                test_state = State(character_state.copy())
                print(f"  {action.name}: {action.effects}")
                new_test_state = strategy_manager.goap_planner._apply_action_effects(action, test_state)
                print(f"    Before: {test_state}")
                print(f"    After:  {new_test_state}")
                print(f"    Goal satisfied? {strategy_manager.goap_planner._goal_satisfied(daily_goal, new_test_state)}")
                print()
    
    big_plan = strategy_manager.goap_planner.plan_actions("TestChar", big_goal, current_state, actions)
    if big_plan:
        print(f"Big plan found: {[a.name for a in big_plan]}")
    else:
        print("No big plan found")


if __name__ == "__main__":
    test_goap_debugging()