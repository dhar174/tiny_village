#!/usr/bin/env python3
"""
Simple test to verify our specific fixes work
"""

import sys
import os
from unittest.mock import Mock, patch

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_goal_constructor_fix():
    """Test that Goal constructor works with target_effects parameter"""
    print("🧪 Testing Goal constructor with target_effects...")

    try:
        # Mock dependencies
        with patch.dict(
            "sys.modules",
            {
                "tiny_strategy_manager": Mock(),
                "tiny_event_handler": Mock(),
                "tiny_types": Mock(),
                "tiny_map_controller": Mock(),
                "actions": Mock(),
            },
        ):
            from tiny_characters import Goal, Condition

            # Mock required objects
            mock_character = Mock()
            mock_graph_manager = Mock()
            mock_goap_planner = Mock()
            mock_prompt_builder = Mock()

            mock_goap_planner.evaluate_goal_importance = Mock(return_value=0.5)
            mock_graph_manager.calculate_goal_difficulty = Mock(return_value=3)
            mock_graph_manager.calculate_reward = Mock(return_value=10)
            mock_graph_manager.calculate_penalty = Mock(return_value=5)
            mock_prompt_builder.generate_completion_message = Mock(
                return_value="Success!"
            )
            mock_prompt_builder.generate_failure_message = Mock(return_value="Failed!")

            # Test Goal creation with target_effects
            goal = Goal(
                name="Test Goal",
                description="A test goal",
                score=0.8,
                character=mock_character,
                target=mock_character,
                completion_conditions={
                    False: [
                        Condition(
                            name="test_condition",
                            attribute="test_attr",
                            target=mock_character,
                            satisfy_value=True,
                            op="==",
                            weight=1,
                        )
                    ]
                },
                evaluate_utility_function=mock_goap_planner.evaluate_goal_importance,
                difficulty=mock_graph_manager.calculate_goal_difficulty,
                completion_reward=mock_graph_manager.calculate_reward,
                failure_penalty=mock_graph_manager.calculate_penalty,
                completion_message=mock_prompt_builder.generate_completion_message,
                failure_message=mock_prompt_builder.generate_failure_message,
                criteria={},
                graph_manager=mock_graph_manager,
                goal_type="basic",
                target_effects={"hunger_level": -5},
            )

            print("  ✅ Goal constructor with target_effects works")
            print(f"  ✅ Goal target_effects: {goal.target_effects}")
            return True

    except Exception as e:
        print(f"  ❌ Goal constructor test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_character_inventory_fix():
    """Test that Character constructor handles ItemInventory correctly"""
    print("\n🧪 Testing Character constructor inventory handling...")

    try:
        from tiny_items import ItemInventory
        from tiny_characters import Character

        # Mock required objects
        with patch("tiny_characters.JobManager") as mock_job_manager, patch(
            "tiny_characters.House"
        ) as mock_house, patch("tiny_characters.GoalGenerator") as mock_goal_gen:

            mock_job_manager.return_value.get_random_job.return_value = Mock()
            mock_house.return_value = Mock()
            mock_goal_gen.return_value.generate_goals.return_value = []

            # Test with ItemInventory object
            inventory = ItemInventory([], [], [], [], [], [])

            # Test character creation (minimal required args)
            character = Character(
                name="Test Character",
                age=25,
                coordinates_location=(0, 0),
                inventory=inventory,
            )

            print("  ✅ Character constructor handles ItemInventory correctly")
            print(f"  ✅ Character inventory type: {type(character.inventory)}")
            return True

    except Exception as e:
        print(f"  ❌ Character inventory test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run our specific tests"""
    print("=" * 60)
    print("TESTING OUR SPECIFIC FIXES")
    print("=" * 60)

    tests = [
        ("Goal Constructor Fix", test_goal_constructor_fix),
        ("Character Inventory Fix", test_character_inventory_fix),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:.<40} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All our fixes work correctly!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
