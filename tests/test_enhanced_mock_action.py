#!/usr/bin/env python3
"""
Test for enhanced MockAction classes with meaningful precondition checking.

This test validates that the enhanced MockAction classes properly implement
precondition checking and will fail when real precondition logic is broken,
rather than masking bugs by always returning True.
"""

import unittest
import sys
import os

# Add the parent directory to sys.path to import test modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced MockAction classes from test files
from test_tiny_utility_functions import MockAction as UtilityMockAction
from test_llm_integration_isolated import MockAction as LLMMockAction


class TestEnhancedMockAction(unittest.TestCase):
    """Test cases for enhanced MockAction implementations."""

    def test_mockaction_no_preconditions_returns_true(self):
        """Test that actions with no preconditions return True."""
        action = UtilityMockAction("TestAction", cost=1.0)
        self.assertTrue(action.preconditions_met())
        self.assertTrue(action.preconditions_met({"energy": 50}))

    def test_mockaction_dict_preconditions_energy_requirement(self):
        """Test dict-style preconditions with energy requirement."""
        preconditions = [
            {"attribute": "energy", "operator": ">=", "value": 50}
        ]
        action = UtilityMockAction("RestAction", cost=1.0, preconditions=preconditions)
        
        # Test with sufficient energy
        state = {"energy": 60, "hunger": 0.3}
        self.assertTrue(action.preconditions_met(state))
        
        # Test with insufficient energy
        state = {"energy": 40, "hunger": 0.3}
        self.assertFalse(action.preconditions_met(state))
        
        # Test with exact energy requirement
        state = {"energy": 50, "hunger": 0.3}
        self.assertTrue(action.preconditions_met(state))

    def test_mockaction_dict_preconditions_multiple_requirements(self):
        """Test multiple dict-style preconditions."""
        preconditions = [
            {"attribute": "energy", "operator": ">=", "value": 50},
            {"attribute": "hunger", "operator": "<=", "value": 0.5}
        ]
        action = UtilityMockAction("EatAction", cost=1.0, preconditions=preconditions)
        
        # Test with all conditions met
        state = {"energy": 60, "hunger": 0.3}
        self.assertTrue(action.preconditions_met(state))
        
        # Test with energy too low
        state = {"energy": 40, "hunger": 0.3}
        self.assertFalse(action.preconditions_met(state))
        
        # Test with hunger too high
        state = {"energy": 60, "hunger": 0.8}
        self.assertFalse(action.preconditions_met(state))
        
        # Test with both conditions failing
        state = {"energy": 40, "hunger": 0.8}
        self.assertFalse(action.preconditions_met(state))

    def test_mockaction_different_operators(self):
        """Test different comparison operators in preconditions."""
        # Test greater than (>)
        action = UtilityMockAction("Action1", cost=1.0, 
                                 preconditions=[{"attribute": "energy", "operator": ">", "value": 50}])
        self.assertTrue(action.preconditions_met({"energy": 51}))
        self.assertFalse(action.preconditions_met({"energy": 50}))
        
        # Test less than (<)
        action = UtilityMockAction("Action2", cost=1.0,
                                 preconditions=[{"attribute": "hunger", "operator": "<", "value": 0.5}])
        self.assertTrue(action.preconditions_met({"hunger": 0.4}))
        self.assertFalse(action.preconditions_met({"hunger": 0.5}))
        
        # Test equals (==)
        action = UtilityMockAction("Action3", cost=1.0,
                                 preconditions=[{"attribute": "level", "operator": "==", "value": 5}])
        self.assertTrue(action.preconditions_met({"level": 5}))
        self.assertFalse(action.preconditions_met({"level": 4}))

    def test_mockaction_no_state_provided_with_preconditions(self):
        """Test that actions with preconditions fail when no state is provided."""
        preconditions = [
            {"attribute": "energy", "operator": ">=", "value": 50}
        ]
        action = UtilityMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Should fail when no state is provided but preconditions exist
        self.assertFalse(action.preconditions_met())
        self.assertFalse(action.preconditions_met(None))

    def test_mockaction_callable_preconditions(self):
        """Test function-style preconditions."""
        def energy_check(state):
            return state.get("energy", 0) >= 50
            
        def hunger_check(state):
            return state.get("hunger", 1.0) <= 0.5
            
        action = UtilityMockAction("TestAction", cost=1.0, 
                                 preconditions=[energy_check, hunger_check])
        
        # Test with conditions met
        state = {"energy": 60, "hunger": 0.3}
        self.assertTrue(action.preconditions_met(state))
        
        # Test with energy condition failing
        state = {"energy": 40, "hunger": 0.3}
        self.assertFalse(action.preconditions_met(state))

    def test_mockaction_mixed_precondition_types(self):
        """Test mixing different precondition types."""
        def custom_check(state):
            return state.get("custom_attr", 0) > 10
            
        preconditions = [
            {"attribute": "energy", "operator": ">=", "value": 50},
            custom_check
        ]
        action = UtilityMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Test with all conditions met
        state = {"energy": 60, "custom_attr": 15}
        self.assertTrue(action.preconditions_met(state))
        
        # Test with dict condition failing
        state = {"energy": 40, "custom_attr": 15}
        self.assertFalse(action.preconditions_met(state))
        
        # Test with callable condition failing
        state = {"energy": 60, "custom_attr": 5}
        self.assertFalse(action.preconditions_met(state))

    def test_mockaction_invalid_operator_fails_safe(self):
        """Test that invalid operators fail safe."""
        preconditions = [
            {"attribute": "energy", "operator": "invalid_op", "value": 50}
        ]
        action = UtilityMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Should fail safe with invalid operator
        state = {"energy": 60}
        self.assertFalse(action.preconditions_met(state))

    def test_mockaction_invalid_precondition_type_fails_safe(self):
        """Test that invalid precondition types fail safe."""
        preconditions = [
            "invalid_precondition_type"  # String instead of dict or callable
        ]
        action = UtilityMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Should fail safe with invalid precondition type
        state = {"energy": 60}
        self.assertFalse(action.preconditions_met(state))

    def test_llm_mockaction_has_same_behavior(self):
        """Test that LLM MockAction has the same enhanced behavior."""
        preconditions = [
            {"attribute": "energy", "operator": ">=", "value": 50}
        ]
        action = LLMMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Test with sufficient energy
        state = {"energy": 60}
        self.assertTrue(action.preconditions_met(state))
        
        # Test with insufficient energy
        state = {"energy": 40}
        self.assertFalse(action.preconditions_met(state))

    def test_mockaction_interface_compatibility(self):
        """Test that enhanced MockAction has all required interface methods."""
        action = UtilityMockAction("TestAction", cost=1.0)
        
        # Check that interface methods exist
        self.assertTrue(hasattr(action, 'preconditions_met'))
        self.assertTrue(callable(action.preconditions_met))
        self.assertTrue(hasattr(action, 'to_dict'))
        self.assertTrue(callable(action.to_dict))
        
        # Check that to_dict returns expected structure
        action_dict = action.to_dict()
        self.assertIn('name', action_dict)
        self.assertIn('preconditions', action_dict)
        self.assertIn('effects', action_dict)
        self.assertIn('cost', action_dict)
        
        # Check that additional attributes exist for compatibility
        self.assertTrue(hasattr(action, 'satisfaction'))
        self.assertTrue(hasattr(action, 'urgency'))
        self.assertTrue(hasattr(action, 'action_id'))


def demonstrate_enhanced_vs_naive_mocking():
    """Demonstrate why enhanced mocking is better than naive always-True mocking."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Enhanced vs Naive MockAction Precondition Checking")
    print("="*60)
    
    # Simulate a broken GOAP system that doesn't properly check preconditions
    def broken_goap_planner(actions, state):
        """Simulate a broken GOAP planner that ignores preconditions."""
        print(f"Planning with state: {state}")
        for action in actions:
            # Broken planner doesn't check preconditions!
            print(f"  Selecting action: {action.name}")
            return action
        return None
    
    def working_goap_planner(actions, state):
        """Simulate a working GOAP planner that checks preconditions."""
        print(f"Planning with state: {state}")
        for action in actions:
            if action.preconditions_met(state):
                print(f"  Selected action: {action.name} (preconditions met)")
                return action
            else:
                print(f"  Skipped action: {action.name} (preconditions not met)")
        print("  No valid action found!")
        return None
    
    # Test scenario: Character with low energy trying to work
    state = {"energy": 30, "hunger": 0.2}
    
    # Create actions with energy requirements
    rest_action = UtilityMockAction("Rest", cost=1.0, 
                                  preconditions=[{"attribute": "energy", "operator": "<=", "value": 50}])
    work_action = UtilityMockAction("Work", cost=2.0,
                                   preconditions=[{"attribute": "energy", "operator": ">=", "value": 60}])
    
    actions = [work_action, rest_action]  # Work first in list
    
    print("\n1. NAIVE MOCKING (always returns True):")
    print("   If MockAction.preconditions_met() always returned True:")
    print("   - Broken GOAP planner would select 'Work' action")
    print("   - Working GOAP planner would also select 'Work' action")  
    print("   - Test would PASS even though GOAP planner is broken!")
    print("   - Bug would be MASKED!")
    
    print("\n2. ENHANCED MOCKING (meaningful checking):")
    print("   With enhanced MockAction.preconditions_met():")
    
    print("\n   Broken GOAP planner (ignores preconditions):")
    selected = broken_goap_planner(actions, state)
    print(f"   -> Selected: {selected.name if selected else 'None'}")
    
    print("\n   Working GOAP planner (checks preconditions):")
    selected = working_goap_planner(actions, state)
    print(f"   -> Selected: {selected.name if selected else 'None'}")
    
    print("\n   RESULT: Enhanced mocking would catch the difference!")
    print("   - Test with broken planner would select 'Work' (wrong)")
    print("   - Test with working planner would select 'Rest' (correct)")
    print("   - Test would FAIL and expose the bug!")
    
    print("\nConclusion: Enhanced MockAction prevents masking real implementation bugs!")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_enhanced_vs_naive_mocking()
    
    # Run the tests
    print("\n" + "="*60)
    print("RUNNING ENHANCED MOCKACTION TESTS")
    print("="*60)
    unittest.main(verbosity=2)