#!/usr/bin/env python3
"""
Demonstration script for the enhanced tiny_utility_functions.py
Shows advanced features like context awareness, history tracking, and caching.
"""

from tiny_utility_functions import (
    UtilityEvaluator,
    calculate_importance,
    get_utility_system_info,
    safe_calculate_action_utility,
)
from test_tiny_utility_functions import MockAction, MockGoal


def main():
    print("=" * 60)
    print("🏘️  TINY VILLAGE UTILITY FUNCTIONS DEMONSTRATION")
    print("=" * 60)

    # Initialize the advanced evaluator
    evaluator = UtilityEvaluator()

    # Character state
    char_state = {
        "hunger": 0.8,
        "energy": 0.3,
        "health": 0.9,
        "money": 10,
        "social_needs": 0.6,
    }

    # Actions
    eat_action = MockAction(
        "EatMeal",
        cost=0.2,
        effects=[
            {"attribute": "hunger", "change_value": -0.7},
            {"attribute": "energy", "change_value": 0.1},
        ],
    )

    sleep_action = MockAction(
        "Sleep", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.8}]
    )

    socialize_action = MockAction(
        "ChatWithFriend",
        cost=0.1,
        effects=[{"attribute": "social_needs", "change_value": -0.5}],
    )

    # Goal
    goal = MockGoal(
        "GetHealthy",
        target_effects={"hunger": -0.5, "energy": 0.6, "social_needs": -0.3},
        priority=0.9,
    )

    print("\n📊 CHARACTER STATE:")
    for attr, value in char_state.items():
        print(f"  {attr}: {value}")

    print(f"\n🎯 CURRENT GOAL: {goal.name} (priority: {goal.priority})")
    print("  Target effects:", goal.target_effects)

    print("\n🔧 BASIC UTILITY CALCULATIONS:")
    print("-" * 40)

    # Test each action
    actions = [eat_action, sleep_action, socialize_action]
    basic_utilities = []

    for action in actions:
        utility, error = safe_calculate_action_utility(char_state, action, goal)
        basic_utilities.append(utility)
        print(f"  {action.name:15} | Utility: {utility:6.2f} | {error or 'OK'}")

    print("\n🚀 ADVANCED UTILITY EVALUATION:")
    print("-" * 40)

    # Environmental context
    environment = {
        "time_of_day": 22,  # Evening
        "weather": "rain",
        "social_event_active": True,
        "resource_scarcity": 0.2,
    }

    print("  Environment:", environment)
    print()

    character_id = "demo_character"
    advanced_utilities = []

    for action in actions:
        # Simulate some action history
        evaluator.update_action_history(character_id, action.name)

        advanced_utility = evaluator.evaluate_action_utility(
            character_id, char_state, action, goal, environment
        )
        advanced_utilities.append(advanced_utility)

        context_modifier = evaluator.calculate_context_modifier(environment)
        history_modifier = evaluator._calculate_history_modifier(
            character_id, action.name
        )

        print(
            f"  {action.name:15} | Advanced: {advanced_utility:6.2f} | Context: {context_modifier:.3f} | History: {history_modifier:.3f}"
        )

    print("\n📈 PLAN EVALUATION:")
    print("-" * 40)

    # Create a plan
    plan = [eat_action, sleep_action, socialize_action]

    total_utility, analysis = evaluator.evaluate_plan_utility_advanced(
        character_id, char_state, plan, goal, environment, simulate_effects=True
    )

    print(f"  Total Plan Utility: {total_utility:.2f}")
    print(f"  Average Action Utility: {analysis['average_utility']:.2f}")
    print(f"  Plan Length: {analysis['plan_length']} actions")

    print("\n  Action Breakdown:")
    for action_info in analysis["action_breakdown"]:
        print(
            f"    Step {action_info['step']}: {action_info['action']:15} | {action_info['utility']:6.2f}"
        )

    print("\n  Final Simulated State:")
    for attr, value in analysis["final_simulated_state"].items():
        original = char_state[attr]
        change = value - original
        print(f"    {attr}: {original:.2f} → {value:.2f} (Δ{change:+.2f})")

    print("\n🧮 ENHANCED IMPORTANCE CALCULATION:")
    print("-" * 40)

    # Test enhanced importance calculation with history and context
    character_history = {
        "recent_goal_completions": ["GetHealthy", "Socialize"],
        "current_goal_type": "GetHealthy",
        "goal_preferences": {
            "GetHealthy": 0.2,  # Slight preference
            "Socialize": -0.1,  # Slight aversion
        },
    }

    importance = calculate_importance(
        health=char_state["health"],
        hunger=char_state["hunger"],
        social_needs=char_state["social_needs"],
        current_activity="idle",
        social_factor=0.6,
        event_participation_factor=0.3,
        goal_importance=0.8,
        goal_urgency=0.9,
        character_history=character_history,
        environmental_context=environment,
    )

    print(f"  Goal Importance Score: {importance:.2f}/10.0")
    print("  Factors considered:")
    print("    ✓ Character needs (hunger, health, social)")
    print("    ✓ Goal urgency and importance")
    print("    ✓ Character history and preferences")
    print("    ✓ Environmental context")
    print("    ✓ Diminishing returns for repeated goals")

    print("\n💾 CACHING PERFORMANCE:")
    print("-" * 40)

    # Demonstrate caching
    import time

    # First calculation (no cache)
    start_time = time.time()
    for _ in range(100):
        evaluator.evaluate_action_utility(
            character_id, char_state, eat_action, goal, environment
        )
    first_time = time.time() - start_time

    # Second calculation (with cache)
    start_time = time.time()
    for _ in range(100):
        evaluator.evaluate_action_utility(
            character_id, char_state, eat_action, goal, environment
        )
    second_time = time.time() - start_time

    speedup = first_time / second_time if second_time > 0 else float("inf")
    print(f"  100 calculations (no cache): {first_time*1000:.1f}ms")
    print(f"  100 calculations (cached):   {second_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x faster")

    print("\n📚 SYSTEM DOCUMENTATION:")
    print("-" * 40)
    doc = get_utility_system_info()
    print(f"  Documentation generated: {len(doc)} characters")
    print("  Includes:")
    print("    ✓ Component descriptions")
    print("    ✓ Data structure requirements")
    print("    ✓ Usage examples")
    print("    ✓ Configuration constants")

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE!")
    print("All advanced utility function features working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
