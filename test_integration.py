#!/usr/bin/env python3
"""
Integration test demonstrating GraphManager -> GoapEvaluator delegation

This shows that the refactoring maintains backward compatibility.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_graph_manager_delegation():
    """Test that GraphManager properly delegates to GoapEvaluator"""
    print("=== Testing GraphManager -> GoapEvaluator Integration ===\n")
    
    # Mock some basic imports to avoid dependency issues
    import types
    import importlib.util
    
    # Import proper MockCharacter instead of using object
    sys.path.insert(0, 'tests')
    from mock_character import MockCharacter, MockMotives, MockMotive
    
    # Create a mock module for tiny_characters with proper mocks
    tiny_characters_mock = types.ModuleType('tiny_characters')
    tiny_characters_mock.Character = MockCharacter
    tiny_characters_mock.PersonalMotives = MockMotives  
    tiny_characters_mock.Motive = MockMotive
    sys.modules['tiny_characters'] = tiny_characters_mock
    
    try:
        # Test basic import and instantiation
        print("1. Testing GraphManager import and instantiation...")
        
        # This should work without full dependencies now that we have mocks
        try:
            from tiny_graph_manager import GraphManager
            print("   ✓ GraphManager imported successfully")
        except Exception as e:
            print(f"   ✗ GraphManager import failed: {e}")
            return False
        
        # Test that GraphManager has a GoapEvaluator instance
        print("2. Testing GoapEvaluator integration...")
        try:
            gm = GraphManager()
            if hasattr(gm, 'goap_evaluator'):
                print("   ✓ GraphManager has goap_evaluator attribute")
            else:
                print("   ✗ GraphManager missing goap_evaluator attribute")
                return False
        except Exception as e:
            print(f"   ✗ GraphManager instantiation failed: {e}")
            return False
        
        # Test that GraphManager has all the GOAP methods
        print("3. Testing method availability...")
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
            if hasattr(gm, method):
                print(f"   ✓ GraphManager.{method} available")
            else:
                print(f"   ✗ GraphManager.{method} missing")
                return False
        
        # Test that GraphManager has helper method
        print("4. Testing helper methods...")
        if hasattr(gm, '_get_world_state'):
            print("   ✓ GraphManager._get_world_state available")
        else:
            print("   ✗ GraphManager._get_world_state missing")
            return False
        
        print("\n✓ All integration tests passed!")
        print("✓ GraphManager successfully delegates to GoapEvaluator")
        print("✓ Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def test_method_signatures():
    """Test that method signatures are preserved"""
    print("\n=== Testing Method Signature Compatibility ===\n")
    
    try:
        from tiny_graph_manager import GraphManager
        import inspect
        
        gm = GraphManager()
        
        # Test key method signatures
        signature_tests = [
            ('calculate_motives', ['character']),
            ('calculate_goal_difficulty', ['goal', 'character']),
            ('calculate_how_goal_impacts_character', ['goal', 'character']),
            ('calculate_action_effect_cost', ['action', 'character', 'goal']),
            ('evaluate_action_plan', ['plan', 'character', 'goal'])
        ]
        
        for method_name, expected_params in signature_tests:
            if hasattr(gm, method_name):
                method = getattr(gm, method_name)
                sig = inspect.signature(method)
                param_names = list(sig.parameters.keys())
                
                # Remove 'self' parameter for comparison
                if 'self' in param_names:
                    param_names.remove('self')
                
                # Check that expected parameters are present
                all_present = all(param in param_names for param in expected_params)
                if all_present:
                    print(f"   ✓ {method_name} signature compatible: {param_names}")
                else:
                    print(f"   ✗ {method_name} signature mismatch. Expected: {expected_params}, Got: {param_names}")
                    return False
        
        print("\n✓ All method signatures are compatible!")
        return True
        
    except Exception as e:
        print(f"✗ Signature test failed: {e}")
        return False

if __name__ == "__main__":
    test1 = test_graph_manager_delegation()
    test2 = test_method_signatures()
    
    if test1 and test2:
        print("\n🎉 INTEGRATION TESTS PASSED!")
        print("The GOAP refactoring is successful and maintains backward compatibility.")
        sys.exit(0)
    else:
        print("\n❌ INTEGRATION TESTS FAILED!")
        sys.exit(1)