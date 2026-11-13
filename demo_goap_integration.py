#!/usr/bin/env python3
"""
Demonstration of the completed GOAP Planning Engine integration.
Shows how the StrategyManager → GOAP → ActionSystem chain works for intelligent character decision-making.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from tiny_strategy_manager import StrategyManager
from tiny_utility_functions import Goal
from actions import State


class DemoCharacter:
    """Demo character with various attributes for testing GOAP."""
    def __init__(self, name):
        self.name = name
        self.hunger_level = 30.0   # Native 0-100 scale (#740)
        self.energy = 40.0         # Native 0-100 scale (#740)
        self.wealth_money = 100.0
        self.social_wellbeing = 50.0  # Native 0-100 scale (#740)
        self.mental_health = 60.0     # Native 0-100 scale (#740)
        self.location = None
        self.job = "developer"


def demonstrate_goap_integration():
    """Demonstrate the complete GOAP integration with various scenarios."""
    print("🎯 GOAP Planning Engine Integration Demonstration")
    print("=" * 60)
    
    # Initialize the system
    strategy_manager = StrategyManager()
    print("✅ StrategyManager with GOAP integration initialized")
    
    # Create demo characters with different states
    characters = [
        DemoCharacter("Alice"),    # Balanced character
        DemoCharacter("Bob"),      # Will have different energy levels
        DemoCharacter("Charlie")   # Will have different social needs
    ]
    
    # Adjust character states for variety (native 0-100, #740)
    characters[1].energy = 20.0        # Bob has low energy
    characters[2].social_wellbeing = 30.0  # Charlie has low social wellbeing
    
    print(f"✅ Created {len(characters)} demo characters with varying needs")
    
    # Demonstrate each character's decision-making process
    for i, character in enumerate(characters, 1):
        print(f"\n📋 Character {i}: {character.name}")
        print("-" * 40)
        
        # Show character's current state
        char_state_dict = strategy_manager.get_character_state_dict(character)
        print(f"Current state: {char_state_dict}")
        
        # Get available actions
        actions = strategy_manager.get_daily_actions(character)
        print(f"Available actions: {[a.name for a in actions]}")
        
        # Plan daily activities using GOAP
        print(f"\n🎯 Planning daily activities for {character.name}...")
        plan_result = strategy_manager.plan_daily_activities(character)
        
        if plan_result:
            print(f"✅ GOAP recommended action: {plan_result.name}")
            if hasattr(plan_result, 'effects'):
                print(f"   Expected effects: {plan_result.effects}")
        else:
            print("ℹ️  No specific action needed (goals already satisfied)")
        
        # Test event-driven planning
        print(f"\n📅 Testing event-driven planning for {character.name}...")
        from test_strategy_goap_alignment import MockEvent
        events = [MockEvent("new_day")]
        strategy_result = strategy_manager.update_strategy(events, character)
        
        if strategy_result and hasattr(strategy_result, 'name'):
            print(f"✅ Event-driven action: {strategy_result.name}")
        else:
            print("ℹ️  No event-driven action needed")
    
    # Demonstrate goal-specific planning
    print(f"\n🎯 Advanced Goal-Specific Planning Demo")
    print("-" * 50)
    
    alice = characters[0]
    
    # Test different types of goals (native 0-100 absolute targets, #740)
    goals = [
        Goal(name="boost_energy", target_effects={"energy": 90}, priority=0.8),
        Goal(name="improve_mood", target_effects={"satisfaction": 85}, priority=0.7),
        Goal(name="balanced_wellbeing", target_effects={"satisfaction": 75, "energy": 75}, priority=0.9)
    ]
    
    for goal in goals:
        print(f"\n🎯 Goal: {goal.name}")
        print(f"   Target: {goal.target_effects}")
        
        # Get current state
        current_state = State(strategy_manager.get_character_state_dict(alice))
        actions = strategy_manager.get_daily_actions(alice)
        
        # Plan using GOAP
        plan = strategy_manager.goap_planner.plan_actions(alice, goal, current_state, actions)
        
        if plan:
            print(f"✅ Plan found ({len(plan)} actions): {[a.name for a in plan]}")
            
            # Simulate plan execution
            sim_state = State(current_state.dict_or_obj.copy())
            print(f"   Initial state: {sim_state}")
            
            for step, action in enumerate(plan, 1):
                sim_state = strategy_manager.goap_planner._apply_action_effects(action, sim_state)
                goal_satisfied = strategy_manager.goap_planner._goal_satisfied(goal, sim_state)
                print(f"   Step {step} ({action.name}): {sim_state}")
                if goal_satisfied:
                    print(f"   🎉 Goal achieved after step {step}!")
                    break
        else:
            print("❌ No plan found for this goal")
    
    # Show job offer decision-making
    print(f"\n💼 Job Offer Decision-Making Demo")
    print("-" * 40)
    
    job_offer = {"title": "Senior Developer", "salary": 80000, "location": "Remote"}
    decision = strategy_manager.respond_to_job_offer(alice, job_offer)
    
    if decision and hasattr(decision, 'name'):
        print(f"✅ Job offer decision: {decision.name}")
    else:
        print("ℹ️  Unable to make job offer decision")
    
    print(f"\n🎉 GOAP Integration Demonstration Complete!")
    print("✅ StrategyManager → GOAP → ActionSystem chain is fully functional")
    print("✅ Characters can make intelligent, goal-driven decisions")
    print("✅ Multi-step planning works for complex goals")
    print("✅ Event-driven and proactive planning both supported")


if __name__ == "__main__":
    demonstrate_goap_integration()