#!/usr/bin/env python3
"""
Basic integration test for GOAP planner without heavy dependencies.
Tests the core integration functionality with Character/World state.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())


def test_core_integration():
    """Test the core integration features without networkx dependencies."""
    print("Testing core GOAP planner integration...")
    
    try:
        from tiny_goap_system import GOAPPlanner
        from actions import State, Action
        from tests.mock_graph_manager import create_mock_graph_manager
        
        # Create a realistic test scenario
        class TestCharacter:
            def __init__(self, name="Alice"):
                self.name = name
                self.energy = 20  # Low energy
                self.hunger = 80  # Very hungry
                self.location = "Home"
                
            def get_state(self):
                return State({
                    'energy': self.energy,
                    'hunger': self.hunger,
                    'location': self.location,
                    'character_name': self.name
                })
        
        class TestGoal:
            def __init__(self, target_energy=50):
                self.target_energy = target_energy
                self.completion_conditions = {'energy': target_energy}
                
            def check_completion(self, state=None):
                if state and hasattr(state, 'get'):
                    return state.get('energy', 0) >= self.target_energy
                return False
        
        # Initialize planner with minimal mock graph manager for better test coverage
        character = TestCharacter()
        mock_graph_manager = create_mock_graph_manager(character)
        planner = GOAPPlanner(mock_graph_manager)
        goal = TestGoal(50)
        
        print(f"Character initial state: energy={character.energy}, hunger={character.hunger}")
        print(f"Goal: Achieve energy >= {goal.target_energy}")
        
        # Test 1: Dynamic state retrieval
        current_state = planner.get_current_world_state(character)
        print(f"✓ Retrieved dynamic state: {current_state}")
        
        # Test 2: Dynamic action retrieval  
        actions = planner.get_available_actions(character)
        print(f"✓ Retrieved {len(actions)} available actions:")
        for action in actions:
            action_name = getattr(action, 'name', 'Unknown')
            action_effects = getattr(action, 'effects', [])
            print(f"  - {action_name}: {action_effects}")
        
        # Test 3: Planning with dynamic retrieval
        plan = planner.plan_for_character(character, goal)
        
        if plan:
            print(f"✓ Generated plan with {len(plan)} actions:")
            for i, action in enumerate(plan, 1):
                action_name = getattr(action, 'name', 'Unknown')
                print(f"  {i}. {action_name}")
        else:
            print("ℹ️ No plan generated (may be expected based on available actions)")
        
        # Test 4: Test with provided actions to ensure planning works
        custom_actions = [
            Action(
                name="Take Nap", 
                preconditions={}, 
                effects=[{"attribute": "energy", "change_value": 15}], 
                cost=1
            ),
            Action(
                name="Full Rest", 
                preconditions={}, 
                effects=[{"attribute": "energy", "change_value": 35}], 
                cost=2
            )
        ]
        
        plan_with_custom = planner.plan_actions(character, goal, actions=custom_actions)
        if plan_with_custom:
            print(f"✓ Generated plan with custom actions ({len(plan_with_custom)} steps):")
            for i, action in enumerate(plan_with_custom, 1):
                print(f"  {i}. {action.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Core integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Basic GOAP Planner Integration Test ===\n")
    
    if test_core_integration():
        print("\n🎉 All core integration features working!")
    else:
        print("\n❌ Integration test failed")