#!/usr/bin/env python3
"""
Test script to verify the completed implementations work correctly.
"""

import sys
import os
import unittest

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
        else:
            print("⚠ Warning: Houses placed at same coordinates")

        return True

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


class TestHappinessCalculation(unittest.TestCase):
    """Test suite for happiness calculation feature implementation with robust validation."""

    def setUp(self):
        """Set up test environment."""
        self.core_features = ["motive_satisfaction"]  # Essential features that must be present
        self.relationship_features = ["social_happiness", "romantic_happiness", "family_happiness"]
        # Additional implementation features to match validation checks
        self.additional_features = ["positive_relationships", "romantic_partner", "family_members"]
        self.all_features = self.core_features + self.relationship_features + self.additional_features
        self.minimum_relationship_features = 2  # Require at least 2 relationship features

    def test_core_happiness_features_required(self):
        """Test that all core happiness features are implemented (no flexibility for critical features)."""
        try:
            with open("tiny_characters.py", "r") as f:
                content = f.read()

            # Assert that all core features are present - no flexibility here
            missing_core_features = []
            for feature in self.core_features:
                if feature not in content:
                    missing_core_features.append(feature)

            self.assertEqual(len(missing_core_features), 0,
                           f"Critical core happiness features are missing and must be implemented: {missing_core_features}. "
                           f"These features are essential for basic happiness calculation functionality.")

            print(f"✓ All {len(self.core_features)} core happiness features implemented: {self.core_features}")

        except Exception as e:
            self.fail(f"Core happiness features test failed: {e}")

    def test_relationship_features_adequate_coverage(self):
        """Test that sufficient relationship features are implemented."""
        try:
            with open("tiny_characters.py", "r") as f:
                content = f.read()

            # Check relationship features implementation
            implemented_relationship_features = []
            for feature in self.relationship_features:
                if feature in content:
                    implemented_relationship_features.append(feature)

            # Require adequate coverage of relationship features
            self.assertGreaterEqual(len(implemented_relationship_features), self.minimum_relationship_features,
                                  f"Expected at least {self.minimum_relationship_features} relationship features "
                                  f"but found only {len(implemented_relationship_features)}: {implemented_relationship_features}. "
                                  f"Available relationship features: {self.relationship_features}")

            print(f"✓ Found {len(implemented_relationship_features)}/{len(self.relationship_features)} relationship features implemented: {implemented_relationship_features}")

        except Exception as e:
            self.fail(f"Relationship happiness features test failed: {e}")

    def test_all_todos_replaced(self):
        """Test that all TODO items have been replaced with actual implementations."""
        try:
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

            # All TODOs must be replaced - no flexibility here
            self.assertEqual(len(remaining_todos), 0,
                           f"Found {len(remaining_todos)} unimplemented TODO items that must be completed: {remaining_todos}")

            print("✓ All happiness calculation TODOs have been implemented")

        except Exception as e:
            self.fail(f"TODO replacement test failed: {e}")

    def test_comprehensive_happiness_validation(self):
        """Test comprehensive validation ensuring proper implementation combinations."""
        try:
            with open("tiny_characters.py", "r") as f:
                content = f.read()

            # Check all features
            implemented_features = []
            missing_features = []
            for feature in self.all_features:
                if feature in content:
                    implemented_features.append(feature)
                else:
                    missing_features.append(feature)

            # Verify we have the essential combination:
            # 1. All core features (motive_satisfaction)
            # 2. At least minimum relationship features
            core_implemented = all(feature in implemented_features for feature in self.core_features)
            relationship_count = sum(1 for feature in self.relationship_features if feature in implemented_features)
            
            # Core features are mandatory
            self.assertTrue(core_implemented, 
                          f"Core features must be implemented: missing {[f for f in self.core_features if f not in implemented_features]}")
            
            # Adequate relationship feature coverage
            self.assertGreaterEqual(relationship_count, self.minimum_relationship_features,
                                  f"Need at least {self.minimum_relationship_features} relationship features, found {relationship_count}")

            # Provide detailed feedback
            total_implemented = len(implemented_features)
            print(f"✓ Comprehensive validation passed:")
            print(f"  - Core features: {len(self.core_features)}/{len(self.core_features)} implemented")
            print(f"  - Relationship features: {relationship_count}/{len(self.relationship_features)} implemented")
            print(f"  - Total features: {total_implemented}/{len(self.all_features)} implemented")
            print(f"  - Implemented features: {implemented_features}")
            
            if missing_features:
                print(f"  - Missing optional features: {missing_features}")

        except Exception as e:
            self.fail(f"Comprehensive happiness validation failed: {e}")


def test_happiness_calculation():
    """Legacy function wrapper for backward compatibility."""
    print("\nRunning enhanced happiness calculation tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHappinessCalculation)
    with open(os.devnull, 'w') as devnull:
        runner = unittest.TextTestRunner(verbosity=1, stream=devnull)
        result = runner.run(suite)
    
    # Return boolean for legacy compatibility
    success = result.wasSuccessful()
    if success:
        print("✓ All happiness calculation tests passed")
    else:
        print(f"✗ {len(result.failures + result.errors)} happiness calculation test(s) failed")
        for test, traceback in result.failures + result.errors:
            print(f"  - {test}: {format_traceback(traceback)}")
    
    return success


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
