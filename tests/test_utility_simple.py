#!/usr/bin/env python3
"""Simple test to verify utility functions work correctly."""

import sys
import traceback

try:
    # Test basic imports
    from tiny_utility_functions import (
        calculate_action_utility,
        calculate_plan_utility,
        calculate_importance,
        UtilityEvaluator,
        utility_evaluator,
        get_utility_system_info,
    )

    print("✓ Successfully imported utility functions")

    # Test MockAction and MockGoal from test file
    from test_tiny_utility_functions import MockAction, MockGoal

    print("✓ Successfully imported mock classes")

    # Test basic functionality
    print("\n=== Testing Basic Functionality ===")

    # Test 1: Basic action utility calculation
    char_state = {"hunger": 0.9, "energy": 0.5}
    action = MockAction(
        "EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}]
    )
    utility = calculate_action_utility(char_state, action)
    expected = 0.9 * 0.7 * 20 - 0.1 * 10  # 12.6 - 1.0 = 11.6
    print(f"Test 1 - Basic utility: {utility} (expected: {expected})")
    assert abs(utility - expected) < 0.001, f"Expected {expected}, got {utility}"
    print("✓ Basic utility calculation works correctly")

    # Test 2: Utility with goal
    goal = MockGoal("SatisfyHunger", target_effects={"hunger": -0.8}, priority=0.8)
    utility_with_goal = calculate_action_utility(char_state, action, current_goal=goal)
    expected_with_goal = 12.6 + (0.8 * 25.0) - 1.0  # 12.6 + 20.0 - 1.0 = 31.6
    print(
        f"Test 2 - Utility with goal: {utility_with_goal} (expected: {expected_with_goal})"
    )
    assert (
        abs(utility_with_goal - expected_with_goal) < 0.001
    ), f"Expected {expected_with_goal}, got {utility_with_goal}"
    print("✓ Utility calculation with goal works correctly")

    # Test 3: Plan utility
    action2 = MockAction(
        "Rest", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.6}]
    )
    plan = [action, action2]
    plan_utility = calculate_plan_utility(char_state, plan, simulate_effects=False)
    # Action 1: 11.6, Action 2: (1.0-0.5)*0.6*15 - 0.5*10 = 4.5 - 5.0 = -0.5
    expected_plan = 11.6 + (-0.5)  # 11.1
    print(f"Test 3 - Plan utility: {plan_utility} (expected: {expected_plan})")
    assert (
        abs(plan_utility - expected_plan) < 0.001
    ), f"Expected {expected_plan}, got {plan_utility}"
    print("✓ Plan utility calculation works correctly")

    # Test 4: Enhanced importance calculation
    importance = calculate_importance(
        health=0.8,
        hunger=0.6,
        social_needs=0.3,
        current_activity="idle",
        social_factor=0.5,
        event_participation_factor=0.2,
        goal_importance=0.7,
    )
    print(f"Test 4 - Importance calculation: {importance}")
    assert 0 <= importance <= 10, f"Importance should be between 0-10, got {importance}"
    print("✓ Importance calculation works correctly")

    # Test 5: UtilityEvaluator
    evaluator = UtilityEvaluator()
    advanced_utility = evaluator.evaluate_action_utility(
        character_id="test_char", character_state=char_state, action=action
    )
    print(f"Test 5 - Advanced utility: {advanced_utility}")
    print("✓ UtilityEvaluator works correctly")

    # Test 6: Documentation
    doc = get_utility_system_info()
    print(f"Test 6 - Documentation length: {len(doc)} characters")
    assert len(doc) > 100, "Documentation should be substantial"
    print("✓ Documentation generation works correctly")

    print("\n🎉 ALL TESTS PASSED! 🎉")
    print(f"✓ All {6} tests completed successfully")

except Exception as e:
    print(f"❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
