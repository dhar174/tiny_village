#!/usr/bin/env python3
"""
Deep debugging of GOAP planner to find infinite loop issue.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import Goal
from actions import State


def test_planner_deep_debug():
    """Debug the GOAP planner with detailed output."""
    print("=== Deep GOAP Planner Debug ===\n")
    
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
    
    # Create the problematic goal
    goal = Goal(
        name="daily_wellbeing",
        target_effects={"satisfaction": 60.0, "energy": 65.0},
        priority=0.8
    )
    print(f"Goal: {goal.name} - Target effects: {goal.target_effects}")
    
    # Get actions
    actions = strategy_manager.get_daily_actions("TestChar")
    print(f"Available actions: {[a.name for a in actions]}")
    
    # Test a simple combination manually
    print(f"\n=== Manual Combination Test ===")
    
    # Test: Rest + Leisure Activity
    state1 = current_state
    print(f"Step 0: {state1}")
    
    # Apply Rest action
    rest_action = next(a for a in actions if a.name == "Rest")
    state1 = strategy_manager.goap_planner._apply_action_effects(rest_action, state1)
    print(f"Step 1 (Rest): {state1}")
    print(f"Goal satisfied? {strategy_manager.goap_planner._goal_satisfied(goal, state1)}")
    
    # Apply Leisure Activity
    leisure_action = next(a for a in actions if a.name == "Leisure Activity")
    state1 = strategy_manager.goap_planner._apply_action_effects(leisure_action, state1)
    print(f"Step 2 (Leisure Activity): {state1}")
    print(f"Goal satisfied? {strategy_manager.goap_planner._goal_satisfied(goal, state1)}")
    
    print(f"\n=== Testing Goal Satisfaction Logic ===")
    # Test goal satisfaction with exact values
    test_state = State({"satisfaction": 60.0, "energy": 65.0, "happiness": 48.0})
    print(f"Test state with exact goals: {test_state}")
    print(f"Goal satisfied? {strategy_manager.goap_planner._goal_satisfied(goal, test_state)}")
    
    # Test with slightly over
    test_state2 = State({"satisfaction": 61.0, "energy": 66.0, "happiness": 48.0})
    print(f"Test state with over goals: {test_state2}")
    print(f"Goal satisfied? {strategy_manager.goap_planner._goal_satisfied(goal, test_state2)}")
    
    # Test with slightly under
    test_state3 = State({"satisfaction": 59.9, "energy": 64.9, "happiness": 48.0})
    print(f"Test state with under goals: {test_state3}")
    print(f"Goal satisfied? {strategy_manager.goap_planner._goal_satisfied(goal, test_state3)}")

if __name__ == "__main__":
    test_planner_deep_debug()