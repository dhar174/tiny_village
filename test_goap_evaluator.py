#!/usr/bin/env python3
"""
Test for GoapEvaluator functionality
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_goap_evaluator_import():
    """Test that we can import the GoapEvaluator"""
    try:
        from goap_evaluator import GoapEvaluator, WorldState
        print("✓ Successfully imported GoapEvaluator and WorldState")
        return True
    except ImportError as e:
        print(f"✗ Failed to import GoapEvaluator: {e}")
        return False

def test_goap_evaluator_instantiation():
    """Test that we can instantiate GoapEvaluator"""
    try:
        from goap_evaluator import GoapEvaluator, WorldState
        evaluator = GoapEvaluator()
        world_state = WorldState()
        print("✓ Successfully instantiated GoapEvaluator and WorldState")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate GoapEvaluator: {e}")
        return False

def test_graph_manager_integration():
    """Test that GraphManager can use GoapEvaluator"""
    try:
        from tiny_graph_manager import GraphManager
        from goap_evaluator import GoapEvaluator
        
        # Create GraphManager (which should create GoapEvaluator internally)
        gm = GraphManager()
        
        # Check that it has the goap_evaluator attribute
        if hasattr(gm, 'goap_evaluator') and isinstance(gm.goap_evaluator, GoapEvaluator):
            print("✓ GraphManager successfully integrated with GoapEvaluator")
            return True
        else:
            print("✗ GraphManager does not have GoapEvaluator integration")
            return False
    except Exception as e:
        print(f"✗ Failed GraphManager integration test: {e}")
        return False

def test_method_delegation():
    """Test that GraphManager methods delegate to GoapEvaluator"""
    try:
        from tiny_graph_manager import GraphManager
        
        gm = GraphManager()
        
        # Check that key methods exist
        methods_to_check = [
            'calculate_motives',
            'calculate_goal_difficulty', 
            'calculate_how_goal_impacts_character',
            'calculate_action_effect_cost',
            'calculate_action_viability_cost',
            'will_action_fulfill_goal',
            'evaluate_action_plan'
        ]
        
        for method_name in methods_to_check:
            if hasattr(gm, method_name):
                print(f"✓ GraphManager has {method_name} method")
            else:
                print(f"✗ GraphManager missing {method_name} method")
                return False
        
        print("✓ All GOAP methods are available in GraphManager")
        return True
    except Exception as e:
        print(f"✗ Failed method delegation test: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=== Testing GoapEvaluator Implementation ===\n")
    
    tests = [
        test_goap_evaluator_import,
        test_goap_evaluator_instantiation,
        test_graph_manager_integration,
        test_method_delegation
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print("")  # Add spacing between tests
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}\n")
    
    print(f"=== Results: {passed}/{len(tests)} tests passed ===")
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)