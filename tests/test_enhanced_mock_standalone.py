#!/usr/bin/env python3
"""
Standalone test for enhanced MockAction classes with meaningful precondition checking.

This test validates that the enhanced MockAction classes properly implement
precondition checking and will fail when real precondition logic is broken,
rather than masking bugs by always returning True.
"""

import unittest


class EnhancedMockAction:
    """Enhanced MockAction with meaningful precondition checking.
    
    This mock implements realistic precondition validation to ensure tests
    fail when real precondition logic is broken, rather than masking bugs
    by always returning True.
    """
    
    def __init__(self, name, cost, effects=None, preconditions=None, satisfaction=None):
        self.name = name
        self.cost = float(cost)
        self.effects = effects if effects else []
        self.preconditions = preconditions if preconditions else []
        
        # Additional attributes to match real Action interface
        self.satisfaction = satisfaction if satisfaction is not None else 5.0
        self.urgency = 1.0
        self.action_id = id(self)
        self.target = None
        self.initiator = None
        self.priority = 1.0
        self.related_goal = None

    def preconditions_met(self, state=None):
        """Check if preconditions are met - matches real Action interface.
        
        This implementation provides meaningful precondition checking rather
        than always returning True, ensuring tests will fail when real
        precondition logic is broken.
        
        Args:
            state: Optional state object or dict to check preconditions against
            
        Returns:
            bool: True if all preconditions are satisfied, False otherwise
        """
        if not self.preconditions:
            return True
            
        # Handle different precondition formats for testing flexibility
        for precondition in self.preconditions:
            if isinstance(precondition, dict):
                # Handle dict-style preconditions: {"attribute": "energy", "operator": ">=", "value": 50}
                attribute = precondition.get("attribute")
                operator = precondition.get("operator", ">=")
                required_value = precondition.get("value", 0)
                
                if state is None:
                    # No state provided - cannot verify preconditions
                    return False
                    
                # Get current value from state
                if isinstance(state, dict):
                    current_value = state.get(attribute, 0)
                else:
                    current_value = getattr(state, attribute, 0)
                
                # Check condition based on operator
                if operator == ">=":
                    if current_value < required_value:
                        return False
                elif operator == "<=":
                    if current_value > required_value:
                        return False
                elif operator == "==":
                    if current_value != required_value:
                        return False
                elif operator == ">":
                    if current_value <= required_value:
                        return False
                elif operator == "<":
                    if current_value >= required_value:
                        return False
                else:
                    # Unknown operator - fail safe
                    return False
                    
            elif hasattr(precondition, 'check_condition'):
                # Handle Condition-like objects
                try:
                    if not precondition.check_condition(state):
                        return False
                except Exception:
                    # If condition checking fails, precondition is not met
                    return False
            elif callable(precondition):
                # Handle function-style preconditions
                try:
                    if not precondition(state):
                        return False
                except Exception:
                    return False
            else:
                # Unknown precondition type - fail safe to catch bugs
                return False
                
        return True

    def to_dict(self):
        """Serialize action for compatibility with real Action interface."""
        return {
            "name": self.name,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "cost": self.cost,
        }


class TestEnhancedMockAction(unittest.TestCase):
    """Test cases for enhanced MockAction implementations."""

    def test_mockaction_no_preconditions_returns_true(self):
        """Test that actions with no preconditions return True."""
        action = EnhancedMockAction("TestAction", cost=1.0)
        self.assertTrue(action.preconditions_met())
        self.assertTrue(action.preconditions_met({"energy": 50}))

    def test_mockaction_dict_preconditions_energy_requirement(self):
        """Test dict-style preconditions with energy requirement."""
        preconditions = [
            {"attribute": "energy", "operator": ">=", "value": 50}
        ]
        action = EnhancedMockAction("RestAction", cost=1.0, preconditions=preconditions)
        
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
        action = EnhancedMockAction("EatAction", cost=1.0, preconditions=preconditions)
        
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
        action = EnhancedMockAction("Action1", cost=1.0, 
                                 preconditions=[{"attribute": "energy", "operator": ">", "value": 50}])
        self.assertTrue(action.preconditions_met({"energy": 51}))
        self.assertFalse(action.preconditions_met({"energy": 50}))
        
        # Test less than (<)
        action = EnhancedMockAction("Action2", cost=1.0,
                                 preconditions=[{"attribute": "hunger", "operator": "<", "value": 0.5}])
        self.assertTrue(action.preconditions_met({"hunger": 0.4}))
        self.assertFalse(action.preconditions_met({"hunger": 0.5}))
        
        # Test equals (==)
        action = EnhancedMockAction("Action3", cost=1.0,
                                 preconditions=[{"attribute": "level", "operator": "==", "value": 5}])
        self.assertTrue(action.preconditions_met({"level": 5}))
        self.assertFalse(action.preconditions_met({"level": 4}))

    def test_mockaction_no_state_provided_with_preconditions(self):
        """Test that actions with preconditions fail when no state is provided."""
        preconditions = [
            {"attribute": "energy", "operator": ">=", "value": 50}
        ]
        action = EnhancedMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Should fail when no state is provided but preconditions exist
        self.assertFalse(action.preconditions_met())
        self.assertFalse(action.preconditions_met(None))

    def test_mockaction_callable_preconditions(self):
        """Test function-style preconditions."""
        def energy_check(state):
            return state.get("energy", 0) >= 50
            
        def hunger_check(state):
            return state.get("hunger", 1.0) <= 0.5
            
        action = EnhancedMockAction("TestAction", cost=1.0, 
                                 preconditions=[energy_check, hunger_check])
        
        # Test with conditions met
        state = {"energy": 60, "hunger": 0.3}
        self.assertTrue(action.preconditions_met(state))
        
        # Test with energy condition failing
        state = {"energy": 40, "hunger": 0.3}
        self.assertFalse(action.preconditions_met(state))

    def test_mockaction_invalid_operator_fails_safe(self):
        """Test that invalid operators fail safe."""
        preconditions = [
            {"attribute": "energy", "operator": "invalid_op", "value": 50}
        ]
        action = EnhancedMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Should fail safe with invalid operator
        state = {"energy": 60}
        self.assertFalse(action.preconditions_met(state))

    def test_mockaction_invalid_precondition_type_fails_safe(self):
        """Test that invalid precondition types fail safe."""
        preconditions = [
            "invalid_precondition_type"  # String instead of dict or callable
        ]
        action = EnhancedMockAction("TestAction", cost=1.0, preconditions=preconditions)
        
        # Should fail safe with invalid precondition type
        state = {"energy": 60}
        self.assertFalse(action.preconditions_met(state))


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
    rest_action = EnhancedMockAction("Rest", cost=1.0, 
                                  preconditions=[{"attribute": "energy", "operator": "<=", "value": 50}])
    work_action = EnhancedMockAction("Work", cost=2.0,
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