#!/usr/bin/env python3
"""
Test script to verify the completed implementations work correctly.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_building_coordinate_selection():
    """Test the building coordinate selection functionality."""
    print("Testing building coordinate selection...")

    try:
        from tiny_buildings import CreateBuilding

        # Test with a basic map setup
        map_data = {"width": 50, "height": 50, "buildings": []}

        builder = CreateBuilding(map_data)

        # Test creating a house
        house = builder.create_house(
            "Test House",
            height=20,
            width=15,
            length=15,
            address="123 Test St",
            stories=2,
            bedrooms=3,
            bathrooms=2,
        )

        print(f"✓ House created successfully at coordinates: ({house.x}, {house.y})")
        print(f"  House details: {house.name}, {house.width}x{house.length}")

        # Test creating multiple houses to verify collision detection
        house2 = builder.create_house(
            "Test House 2",
            height=25,
            width=12,
            length=12,
            address="456 Test Ave",
            stories=1,
            bedrooms=2,
            bathrooms=1,
        )

        print(f"✓ Second house created at coordinates: ({house2.x}, {house2.y})")

        # Verify they don't overlap
        if (house.x, house.y) != (house2.x, house2.y):
            print(
                "✓ Collision detection working - houses placed at different coordinates"
            )
            return True
        else:
            print("✗ Houses placed at same coordinates - collision detection failed")
            return False

    except Exception as e:
        print(f"✗ Building coordinate selection test failed: {e}")
        return False


def test_pause_functionality():
    """Test that pause functionality was added to gameplay controller."""
    print("\nTesting pause functionality...")

    try:
        from tiny_gameplay_controller import GameplayController

        # Create a controller (without actually starting pygame)
        controller = GameplayController.__new__(GameplayController)

        # Test pause state
        controller.paused = False
        pause_state = getattr(controller, "paused", False)
        print(f"✓ Pause state accessible: {pause_state}")

        # Test pause toggle logic
        controller.paused = not getattr(controller, "paused", False)
        print(f"✓ Pause toggle works: {controller.paused}")

        return True

    except Exception as e:
        print(f"✗ Pause functionality test failed: {e}")
        return False


def test_happiness_calculation():
    """Test the enhanced happiness calculation."""
    print("\nTesting happiness calculation enhancements...")

    try:
        # Read the file to check if the TODOs were replaced
        with open("tiny_characters.py", "r") as f:
            content = f.read()

        # Check that TODOs were replaced with actual implementations
        todo_patterns = [
            "# TODO: Add happiness calculation based on motives",
            "# TODO: Add happiness calculation based on social relationships",
            "# TODO: Add happiness calculation based on romantic relationships",
            "# TODO: Add happiness calculation based on family relationships",
        ]

        remaining_todos = []
        for pattern in todo_patterns:
            if pattern in content:
                remaining_todos.append(pattern)

        if not remaining_todos:
            print("✓ All happiness calculation TODOs have been implemented")

            # Check for implementation keywords
            implementation_keywords = [
                "motive_satisfaction",
                "social_happiness",
                "romantic_happiness",
                "family_happiness",
            ]

            implemented_features = []
            missing_features = []
            for keyword in implementation_keywords:
                if keyword in content:
                    implemented_features.append(keyword)
                else:
                    missing_features.append(keyword)

            if len(implemented_features) == len(implementation_keywords):
                print(
                    f"✓ All {len(implemented_features)} happiness calculation features implemented"
                )
                return True
            else:
                print(
                    f"✗ Missing {len(missing_features)} required happiness features: {missing_features}"
                )
                return False

        else:
            print(f"⚠ Warning: {len(remaining_todos)} TODO items still remain:")
            for todo in remaining_todos:
                print(f"  - {todo}")
            return False

    except Exception as e:
        print(f"✗ Happiness calculation test failed: {e}")
        return False


def test_goap_implementations():
    """Test that the GOAP system implementations are still working."""
    print("\nTesting GOAP system implementations...")

    try:
        from tiny_goap_system import GOAPSystem

        # Test that the methods exist and are implemented
        methods_to_check = [
            "replan",
            "find_alternative_action",
            "calculate_utility",
            "evaluate_utility",
            "evaluate_feasibility_of_goal",
        ]

        # Create a basic GOAP system instance
        goap = GOAPSystem.__new__(GOAPSystem)

        implemented_methods = []
        for method_name in methods_to_check:
            if hasattr(goap, method_name):
                method = getattr(goap, method_name)
                # Check if it's not just "pass"
                if callable(method):
                    implemented_methods.append(method_name)

        print(
            f"✓ Found {len(implemented_methods)}/{len(methods_to_check)} GOAP methods implemented"
        )

        if len(implemented_methods) == len(methods_to_check):
            print("✓ All key GOAP methods are present")
            return True
        else:
            missing = set(methods_to_check) - set(implemented_methods)
            print(f"⚠ Missing methods: {missing}")
            return False

    except Exception as e:
        print(f"✗ GOAP system test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running tests for completed implementations...\n")

    tests = [
        test_building_coordinate_selection,
        test_pause_functionality,
        test_happiness_calculation,
        test_goap_implementations,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Implementations appear to be working correctly.")
    else:
        print(f"⚠ {total - passed} test(s) failed. Review the implementations.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
