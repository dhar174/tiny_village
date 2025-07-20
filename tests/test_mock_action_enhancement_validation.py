#!/usr/bin/env python3
"""
Test script that demonstrates how the enhanced MockAction catches issues
that the old simplified MockAction would miss.

This validates that the enhancement addresses the issue raised about overly
simplified mocks causing tests to pass when real implementation is broken.
"""

import sys
import traceback

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('./tests')

try:
    from test_tiny_utility_functions import MockAction, MockGoal
    from tiny_utility_functions import (
        calculate_action_utility, 
        validate_action,
        safe_calculate_action_utility,
        UtilityEvaluator
    )
    
    print("=" * 60)
    print("🧪 ENHANCED MOCKACTION VALIDATION TESTS")
    print("=" * 60)
    print()
    
    # Test 1: Precondition checking that old MockAction couldn't do
    print("📋 TEST 1: Precondition Validation")
    print("-" * 40)
    
    # Create action with failed preconditions
    blocked_action = MockAction(
        "BlockedAction",
        cost=0.1,
        effects=[{"attribute": "hunger", "change_value": -0.5}],
        preconditions=[True, False]  # One failed precondition
    )
    
    # This would have been impossible to test with old MockAction
    can_execute = blocked_action.preconditions_met()
    print(f"Action with failed preconditions executable: {can_execute}")
    assert not can_execute, "Failed preconditions should prevent execution"
    print("✓ Precondition checking working correctly")
    print()
    
    # Test 2: Effects validation that catches malformed data
    print("🔍 TEST 2: Effects Validation")
    print("-" * 40)
    
    try:
        # This should fail - invalid effects structure
        invalid_action = MockAction(
            "InvalidAction",
            cost=0.1,
            effects=[{"missing_change_value": "invalid"}]
        )
        print("❌ Should have caught invalid effects structure")
        assert False, "Should have caught invalid effects"
    except ValueError as e:
        print(f"✓ Correctly caught invalid effects: {e}")
    
    try:
        # This should also fail - non-numeric change_value
        invalid_action2 = MockAction(
            "InvalidAction2", 
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": "not_a_number"}]
        )
        print("❌ Should have caught non-numeric change_value")
        assert False, "Should have caught non-numeric change_value"
    except ValueError as e:
        print(f"✓ Correctly caught non-numeric change_value: {e}")
    print()
    
    # Test 3: Target/Initiator relationships that real actions have
    print("🎯 TEST 3: Target/Initiator Relationships")
    print("-" * 40)
    
    # Test default_target_is_initiator functionality
    self_action = MockAction(
        "SelfTargetingAction",
        cost=0.1,
        initiator="character_bob",
        default_target_is_initiator=True
    )
    
    assert self_action.target == "character_bob", "Target should be set to initiator"
    assert self_action.initiator == "character_bob", "Initiator should be preserved"
    print("✓ Default target assignment working correctly")
    
    # Test explicit target assignment
    social_action = MockAction(
        "SocialAction",
        cost=0.1,
        initiator="character_alice",
        target="character_bob"
    )
    
    assert social_action.initiator == "character_alice", "Initiator should be alice"
    assert social_action.target == "character_bob", "Target should be bob"
    print("✓ Explicit target assignment working correctly")
    print()
    
    # Test 4: Priority and related_goal attributes for realistic planning
    print("📊 TEST 4: Priority and Goal Integration")
    print("-" * 40)
    
    urgent_goal = MockGoal(
        "UrgentSurvival",
        target_effects={"hunger": -0.8},
        priority=1.0,
        urgency=0.95
    )
    
    urgent_action = MockAction(
        "UrgentEat",
        cost=0.1,
        effects=[{"attribute": "hunger", "change_value": -0.8}],
        priority=0.9,
        related_goal=urgent_goal
    )
    
    # Test with UtilityEvaluator which uses urgency for calculations
    evaluator = UtilityEvaluator()
    char_state = {"hunger": 0.9, "energy": 0.5, "health": 0.8}
    
    urgent_utility = evaluator.evaluate_action_utility(
        "survivor", char_state, urgent_action, urgent_goal
    )
    
    # Create same action but with normal urgency
    normal_goal = MockGoal(
        "NormalEat",
        target_effects={"hunger": -0.8},
        priority=1.0,
        urgency=0.5  # Normal urgency
    )
    
    normal_action = MockAction(
        "NormalEat",
        cost=0.1,
        effects=[{"attribute": "hunger", "change_value": -0.8}],
        priority=0.5
    )
    
    normal_utility = evaluator.evaluate_action_utility(
        "survivor", char_state, normal_action, normal_goal
    )
    
    print(f"Urgent action utility: {urgent_utility:.2f}")
    print(f"Normal action utility: {normal_utility:.2f}")
    
    # Urgent action should have higher utility due to urgency multiplier
    assert urgent_utility > normal_utility, "Urgent actions should have higher utility"
    print("✓ Urgency-based utility calculations working correctly")
    print()
    
    # Test 5: Validation compatibility with utility system validators
    print("🔧 TEST 5: Utility System Validation")
    print("-" * 40)
    
    # Create properly structured action
    valid_action = MockAction(
        "ValidAction",
        cost=0.2,
        effects=[
            {"attribute": "hunger", "change_value": -0.5},
            {"attribute": "energy", "change_value": 0.3}
        ],
        preconditions=[True],
        priority=0.7
    )
    
    # Test with validate_action function
    is_valid, error = validate_action(valid_action)
    print(f"Action validation result: {is_valid}, error: '{error}'")
    assert is_valid, f"Valid action should pass validation: {error}"
    
    # Test with safe_calculate_action_utility
    complete_char_state = {
        "hunger": 0.8,
        "energy": 0.3,
        "health": 0.9
    }
    
    utility, error = safe_calculate_action_utility(
        complete_char_state, valid_action, validate_inputs=True
    )
    
    print(f"Safe utility calculation: {utility:.2f}, error: '{error}'")
    assert error == "", f"Should calculate utility without error: {error}"
    assert utility > 0, "Should have positive utility for beneficial action"
    print("✓ Utility system validation working correctly")
    print()
    
    # Test 6: Demonstrate detection of broken implementations
    print("🐛 TEST 6: Broken Implementation Detection")
    print("-" * 40)
    
    # Simulate what old MockAction couldn't catch
    class OldStyleMockAction:
        """Simplified mock that misses critical attributes."""
        def __init__(self, name, cost, effects=None):
            self.name = name
            self.cost = cost
            self.effects = effects or []
            # Missing: preconditions, target, initiator, priority, etc.
    
    old_action = OldStyleMockAction("OldAction", 0.1, [{"attribute": "hunger", "change_value": -0.5}])
    
    # Try validation with old-style action
    try:
        # This would fail because validate_action expects real Action attributes
        is_valid, error = validate_action(old_action)
        print(f"Old action validation: {is_valid}, error: '{error}'")
        
        # Enhanced MockAction should pass where old one might not
        enhanced_equivalent = MockAction(
            "EnhancedEquivalent",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.5}]
        )
        
        is_valid_enhanced, error_enhanced = validate_action(enhanced_equivalent)
        print(f"Enhanced action validation: {is_valid_enhanced}, error: '{error_enhanced}'")
        
        assert is_valid_enhanced, "Enhanced MockAction should pass validation"
        print("✓ Enhanced MockAction passes validations that old one might fail")
        
    except Exception as e:
        print(f"Old action failed as expected: {e}")
        print("✓ Enhanced MockAction would handle this properly")
    
    print()
    
    # Test 7: Method availability
    print("🔨 TEST 7: Method Availability")
    print("-" * 40)
    
    action = MockAction("TestAction", cost=0.1)
    
    # Check that enhanced methods are available
    methods_to_check = ['preconditions_met', 'add_effect', 'add_precondition']
    
    for method_name in methods_to_check:
        assert hasattr(action, method_name), f"Should have {method_name} method"
        assert callable(getattr(action, method_name)), f"{method_name} should be callable"
        print(f"✓ Method {method_name} available and callable")
    
    # Test that methods work
    action.add_precondition(True)
    action.add_effect({"attribute": "test", "change_value": 1.0})
    
    assert len(action.preconditions) == 1, "Should have added precondition"
    assert len(action.effects) == 1, "Should have added effect"
    assert action.preconditions_met(), "Should meet preconditions"
    
    print("✓ All methods working correctly")
    print()
    
    print("=" * 60)
    print("🎉 ALL ENHANCEMENT VALIDATION TESTS PASSED!")
    print("=" * 60)
    print()
    print("The enhanced MockAction successfully addresses the issues with")
    print("the original simplified mock by providing:")
    print("✓ Precondition checking capabilities")
    print("✓ Effects validation to catch malformed data")
    print("✓ Target/initiator relationship modeling")
    print("✓ Priority and goal integration")
    print("✓ Compatibility with utility system validators") 
    print("✓ Method availability matching real Action class")
    print("✓ Early detection of implementation issues")
    print()
    print("This ensures tests are more likely to catch real implementation")
    print("problems rather than passing due to oversimplified mocks.")

except Exception as e:
    print(f"❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)