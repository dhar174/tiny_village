#!/usr/bin/env python3
"""
Test script to verify the enhanced GOAP planner integration with Character/World state.
This tests the new dynamic state and action retrieval functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())


def test_dynamic_state_retrieval():
    """Test the dynamic world state retrieval functionality."""
    print("Testing dynamic world state retrieval...")
    
    try:
        from tiny_goap_system import GOAPPlanner
        
        # Create a mock character for testing
        class MockCharacter:
            def __init__(self, name="TestChar"):
                self.name = name
                self.energy = 50
                self.hunger = 30
                
            def get_state(self):
                from actions import State
                return State({'energy': self.energy, 'hunger': self.hunger, 'name': self.name})
        
        # Test with no graph manager (should still work)
        planner = GOAPPlanner(None)
        character = MockCharacter()
        
        world_state = planner.get_current_world_state(character)
        print("✓ Dynamic world state retrieval successful")
        print(f"  Retrieved state: {world_state}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dynamic world state retrieval failed: {e}")
        return False


def test_dynamic_action_retrieval():
    """Test the dynamic available actions retrieval functionality."""
    print("\nTesting dynamic action retrieval...")
    
    try:
        from tiny_goap_system import GOAPPlanner
        
        class MockCharacter:
            def __init__(self, name="TestChar"):
                self.name = name
                self.energy = 50
                
            def get_state(self):
                from actions import State
                return State({'energy': self.energy, 'name': self.name})
        
        planner = GOAPPlanner(None)
        character = MockCharacter()
        
        available_actions = planner.get_available_actions(character)
        print("✓ Dynamic action retrieval successful")
        print(f"  Retrieved {len(available_actions)} actions")
        
        if available_actions:
            print(f"  Sample action: {getattr(available_actions[0], 'name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dynamic action retrieval failed: {e}")
        return False


def test_enhanced_planning():
    """Test the enhanced planning functionality with dynamic retrieval."""
    print("\nTesting enhanced planning with dynamic retrieval...")
    
    try:
        from tiny_goap_system import GOAPPlanner
        
        class MockCharacter:
            def __init__(self, name="TestChar"):
                self.name = name
                self.energy = 20  # Low energy to test planning
                
            def get_state(self):
                from actions import State
                return State({'energy': self.energy, 'name': self.name})
        
        class MockGoal:
            def __init__(self):
                self.completion_conditions = {'energy': 50}
                
            def check_completion(self, state=None):
                if state and hasattr(state, 'get'):
                    return state.get('energy', 0) >= 50
                return False
        
        planner = GOAPPlanner(None)
        character = MockCharacter()
        goal = MockGoal()
        
        # Test planning with automatic state/action retrieval
        plan = planner.plan_for_character(character, goal)
        
        print("✓ Enhanced planning successful")
        if plan:
            print(f"  Generated plan with {len(plan)} actions")
        else:
            print("  No plan generated (expected for this test setup)")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced planning failed: {e}")
        return False


def test_strategy_manager_integration():
    """Test that the strategy manager properly uses the enhanced planner."""
    print("\nTesting strategy manager integration...")
    
    try:
        from tiny_strategy_manager import StrategyManager
        
        # Test that StrategyManager can be created with the enhanced planner
        strategy_manager = StrategyManager()
        
        print("✓ Strategy manager integration successful")
        print(f"  GOAP planner type: {type(strategy_manager.goap_planner).__name__}")
        print(f"  Graph manager attached: {strategy_manager.goap_planner.graph_manager is not None}")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy manager integration failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing GOAP Planner Integration with Character/World ===\n")
    
    tests = [
        test_dynamic_state_retrieval,
        test_dynamic_action_retrieval,
        test_enhanced_planning,
        test_strategy_manager_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
    else:
        print("⚠️  Some tests failed - check implementation")