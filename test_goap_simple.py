#!/usr/bin/env python3
"""
Simple test for GOAP refactoring functionality
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_basic_functionality():
    """Test basic GOAP evaluator functionality"""
    print("=== Testing Basic GOAP Evaluator Functionality ===\n")
    
    try:
        # Test 1: Import
        from goap_evaluator import GoapEvaluator, WorldState
        print("✓ Successfully imported GoapEvaluator and WorldState")
        
        # Test 2: Instantiation
        evaluator = GoapEvaluator()
        world_state = WorldState()
        print("✓ Successfully instantiated GoapEvaluator and WorldState")
        
        # Test 3: Check methods exist
        required_methods = [
            'calculate_motives',
            'calculate_goal_difficulty',
            'calculate_how_goal_impacts_character',
            'calculate_action_effect_cost',
            'calculate_action_viability_cost',
            'will_action_fulfill_goal',
            'evaluate_action_plan'
        ]
        
        for method in required_methods:
            if hasattr(evaluator, method):
                print(f"✓ GoapEvaluator has {method} method")
            else:
                print(f"✗ GoapEvaluator missing {method} method")
                return False
        
        print("\n✓ All tests passed! GOAP refactoring is structurally correct.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)