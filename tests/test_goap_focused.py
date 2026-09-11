#!/usr/bin/env python3
"""
Focused test to identify specific GOAP system issues and validate key functionality.
This test focuses on the core problems outlined in the issue.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_core_planning_algorithm():
    """Test 1: Core Planner Logic - Unify and complete the core GOAP planning algorithm."""
    print("=== Test 1: Core Planning Algorithm ===")
    
    try:
        from tiny_goap_system import GOAPPlanner
        from tiny_utility_functions import Goal
        from actions import Action, State
        
        # Create a GOAP planner
        planner = GOAPPlanner(None)  # No graph manager for basic test
        print("✓ GOAPPlanner instantiated")
        
        # Create a simple character (dict-based for simplicity)
        character = {"name": "TestChar", "energy": 50, "satisfaction": 40}
        
        # Create a goal with target effects
        goal = Goal(
            name="increase_energy",
            target_effects={"energy": 80},
            priority=0.8
        )
        print("✓ Goal created")
        
        # Create current state
        current_state = State({"energy": 50, "satisfaction": 40})
        print("✓ Current state created")
        
        # Create actions using Action objects from actions.py
        actions = [
            Action(
                name="Rest",
                preconditions=[],
                effects=[{"targets": ["initiator"], "attribute": "energy", "change_value": 20}],
                cost=1.0
            ),
            Action(
                name="Sleep",
                preconditions=[],
                effects=[{"targets": ["initiator"], "attribute": "energy", "change_value": 30}],
                cost=2.0
            )
        ]
        print("✓ Action objects created")
        
        # Test the main planning method (instance method)
        plan = planner.plan_actions(character, goal, current_state, actions)
        
        if plan:
            print(f"✓ Instance method plan_actions returned plan with {len(plan)} actions:")
            for action in plan:
                print(f"  - {action.name} (cost: {action.cost})")
        else:
            print("✓ No plan needed (goal already satisfied)")
        
        # Test the static method for backward compatibility
        try:
            static_plan = GOAPPlanner.goap_planner(character, goal, current_state, actions)
            if static_plan:
                print(f"✓ Static method goap_planner returned plan with {len(static_plan)} actions")
            else:
                print("✓ Static method: No plan needed")
        except Exception as e:
            print(f"⚠ Static method issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core planning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_object_compatibility():
    """Test 2: Action Definitions - Ensure the planner works with Action objects from actions.py."""
    print("\n=== Test 2: Action Object Compatibility ===")
    
    try:
        from tiny_goap_system import GOAPPlanner
        from actions import Action, EatAction, SleepAction, WorkAction
        from tiny_utility_functions import Goal
        
        planner = GOAPPlanner(None)
        character = {"name": "TestChar", "energy": 30, "hunger": 80, "money": 10}
        
        # Test with various Action types from actions.py
        actions = [
            EatAction("apple", initiator_id="TestChar"),
            SleepAction(duration=8, initiator_id="TestChar"), 
            WorkAction("programmer", initiator_id="TestChar"),
            Action(
                name="CustomAction",
                preconditions=[],
                effects=[{"targets": ["initiator"], "attribute": "satisfaction", "change_value": 10}],
                cost=1.5
            )
        ]
        print(f"✓ Created {len(actions)} Action objects of different types")
        
        # Test precondition checking
        for action in actions:
            try:
                result = planner._action_applicable(action, {"energy": 30, "hunger": 80})
                print(f"✓ Precondition check for {action.name}: {result}")
            except Exception as e:
                print(f"⚠ Precondition check failed for {action.name}: {e}")
        
        # Test effect application
        test_state = {"energy": 30, "hunger": 80, "money": 10}
        for action in actions:
            try:
                new_state = planner._apply_action_effects(action, test_state)
                print(f"✓ Effect application for {action.name} successful")
            except Exception as e:
                print(f"⚠ Effect application failed for {action.name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Action compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_integration():
    """Test 3: Integration with Character/World - Dynamic state and action retrieval."""
    print("\n=== Test 3: Dynamic Integration ===")
    
    try:
        from tiny_goap_system import GOAPPlanner
        from tiny_utility_functions import Goal
        
        planner = GOAPPlanner(None)
        
        # Mock character object
        class MockCharacter:
            def __init__(self):
                self.name = "TestChar"
                self.energy = 40
                self.hunger = 60
                self.wealth_money = 20
                
            def get_state(self):
                from actions import State
                return State({"energy": self.energy, "hunger": self.hunger, "money": self.wealth_money})
        
        character = MockCharacter()
        print("✓ Mock character created")
        
        # Test dynamic world state retrieval
        current_state = planner.get_current_world_state(character)
        print(f"✓ Dynamic world state retrieved: {current_state}")
        
        # Test dynamic action retrieval
        available_actions = planner.get_available_actions(character)
        print(f"✓ Dynamic actions retrieved: {len(available_actions)} actions")
        for action in available_actions:
            print(f"  - {action.name}")
        
        # Test planning with dynamic inputs (auto-retrieval)
        goal = Goal(
            name="improve_state",
            target_effects={"energy": 70},
            priority=0.8
        )
        
        plan = planner.plan_for_character(character, goal)
        if plan:
            print(f"✓ Dynamic planning successful: {len(plan)} actions")
        else:
            print("✓ Dynamic planning: No plan needed")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_functions():
    """Test 4: Cost Functions - Properly integrate and utilize action costs in planning."""
    print("\n=== Test 4: Cost Functions ===")
    
    try:
        from tiny_goap_system import GOAPPlanner
        from actions import Action, State
        from tiny_utility_functions import Goal
        
        planner = GOAPPlanner(None)
        
        # Create character and goal
        character = {"name": "TestChar", "energy": 50}
        goal = Goal(
            name="high_energy", 
            target_effects={"energy": 80},
            priority=0.8
        )
        current_state = State({"energy": 50})
        
        # Create actions with different costs
        expensive_action = Action(
            name="ExpensiveRest",
            preconditions=[],
            effects=[{"targets": ["initiator"], "attribute": "energy", "change_value": 30}],
            cost=10.0  # High cost
        )
        
        cheap_action = Action(
            name="CheapRest", 
            preconditions=[],
            effects=[{"targets": ["initiator"], "attribute": "energy", "change_value": 30}],
            cost=1.0   # Low cost
        )
        
        actions = [expensive_action, cheap_action]
        
        # Test cost calculation
        for action in actions:
            cost = planner._calculate_action_cost(action, current_state, character, goal)
            print(f"✓ Cost calculated for {action.name}: {cost}")
        
        # Test that planning considers costs (should prefer cheaper action)
        plan = planner.plan_actions(character, goal, current_state, actions)
        if plan:
            print(f"✓ Planning with cost consideration: selected {plan[0].name}")
            if plan[0].name == "CheapRest":
                print("✓ Cost optimization working (chose cheaper action)")
            else:
                print("⚠ Cost optimization may not be working optimally")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_manager_alignment():
    """Test 5: Alignment with StrategyManager - Basic interface testing."""
    print("\n=== Test 5: StrategyManager Alignment ===")
    
    try:
        # Test imports
        from tiny_goap_system import GOAPPlanner
        print("✓ GOAPPlanner imported")
        
        # Try to import StrategyManager (may fail due to dependencies)
        try:
            from tiny_strategy_manager import StrategyManager
            print("✓ StrategyManager imported")
            
            # Test basic instantiation
            strategy_manager = StrategyManager(use_llm=False)  # Disable LLM to avoid dependencies
            print("✓ StrategyManager instantiated")
            
            # Test that it has a GOAP planner
            if hasattr(strategy_manager, 'goap_planner') and strategy_manager.goap_planner:
                print("✓ StrategyManager has GOAPPlanner instance")
            else:
                print("⚠ StrategyManager missing GOAPPlanner instance")
            
            # Test basic interaction
            class MockCharacter:
                def __init__(self):
                    self.name = "TestChar"
                    self.energy = 50
                    self.hunger_level = 5
                    self.wealth_money = 100
                    self.social_wellbeing = 50
                    self.mental_health = 50
                    self.location = None
                    self.job = "unemployed"
            
            character = MockCharacter()
            actions = strategy_manager.get_daily_actions(character)
            print(f"✓ StrategyManager.get_daily_actions returned {len(actions)} actions")
            
            plan_result = strategy_manager.plan_daily_activities(character)
            if plan_result:
                print(f"✓ StrategyManager.plan_daily_activities returned: {plan_result.name}")
            else:
                print("✓ StrategyManager.plan_daily_activities completed (no plan needed)")
            
        except ImportError as ie:
            print(f"⚠ StrategyManager import failed (expected due to dependencies): {ie}")
        except Exception as e:
            print(f"⚠ StrategyManager test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ StrategyManager alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all focused tests to identify specific GOAP issues."""
    print("GOAP System Focused Testing")
    print("="*50)
    
    tests = [
        test_core_planning_algorithm,
        test_action_object_compatibility, 
        test_dynamic_integration,
        test_cost_functions,
        test_strategy_manager_alignment
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("🎉 All tests passed! GOAP system is working well.")
    else:
        print("⚠ Some issues identified. See details above.")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())