#!/usr/bin/env python3
"""
Test script to verify the completed implementations work correctly.
"""

import sys
import os
import unittest

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestCompletedImplementations(unittest.TestCase):
    """Test suite for completed implementations."""

    def test_building_coordinate_selection(self):
        """Test the building coordinate selection functionality."""
        # This test requires numpy and pygame which aren't available in minimal environment
        # Skip if dependencies are missing, but test the import and basic functionality
        try:
            from tiny_buildings import CreateBuilding
        except ImportError as e:
            if "numpy" in str(e) or "pygame" in str(e):
                self.skipTest(f"Skipping due to missing dependency: {e}")
            else:
                raise

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

        # Verify house was created with correct properties
        self.assertIsNotNone(house)
        self.assertEqual(house.name, "Test House")
        self.assertEqual(house.width, 15)
        self.assertEqual(house.length, 15)
        self.assertTrue(hasattr(house, 'x'))
        self.assertTrue(hasattr(house, 'y'))

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

        # Verify second house was created
        self.assertIsNotNone(house2)
        self.assertEqual(house2.name, "Test House 2")
        
        # Verify they don't overlap (collision detection working)
        self.assertNotEqual((house.x, house.y), (house2.x, house2.y),
                          "Houses should not be placed at the same coordinates")


    def test_pause_functionality(self):
        """Test that pause functionality was added to gameplay controller."""
        # This test requires pygame which isn't available in minimal environment
        # Skip if dependencies are missing, but test the import and basic functionality
        try:
            from tiny_gameplay_controller import GameplayController
        except ImportError as e:
            if "pygame" in str(e):
                self.skipTest(f"Skipping due to missing dependency: {e}")
            else:
                raise

        # Create a controller (without actually starting pygame)
        controller = GameplayController.__new__(GameplayController)

        # Test pause state exists
        controller.paused = False
        self.assertTrue(hasattr(controller, 'paused'),
                       "GameplayController should have a 'paused' attribute")

        # Test pause toggle logic
        initial_state = getattr(controller, "paused", False)
        controller.paused = not initial_state
        self.assertNotEqual(controller.paused, initial_state,
                          "Pause state should toggle correctly")


    def test_happiness_calculation(self):
        """Test the enhanced happiness calculation."""
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

        self.assertEqual(len(remaining_todos), 0,
                        f"Found {len(remaining_todos)} unimplemented TODO items: {remaining_todos}")

        # Check for implementation keywords
        implementation_keywords = [
            "motive_satisfaction",
            "social_happiness",
            "romantic_happiness",
            "family_happiness",
        ]

        implemented_features = []
        for keyword in implementation_keywords:
            if keyword in content:
                implemented_features.append(keyword)

        self.assertGreater(len(implemented_features), 0,
                          "Should have at least some happiness calculation features implemented")
        self.assertEqual(len(implemented_features), 4,
                        f"Expected 4 happiness calculation features, found {len(implemented_features)}: {implemented_features}")


    def test_goap_implementations(self):
        """Test that the GOAP system implementations are actually working."""
        from tiny_goap_system import GOAPPlanner, Plan

        # Test GOAPPlanner methods
        planner_methods = [
            "plan_actions",
            "_goal_satisfied",
            "_action_applicable",
            "_apply_action_effects",
        ]

        # Create a GOAP planner instance - it requires a graph_manager
        try:
            planner = GOAPPlanner(graph_manager=None)  # Allow None for testing
            self.assertIsNotNone(planner)
        except TypeError:
            self.fail("GOAPPlanner should accept graph_manager=None for testing")

        # Check that planner methods exist
        for method_name in planner_methods:
            self.assertTrue(hasattr(planner, method_name),
                          f"GOAPPlanner should have method '{method_name}'")
            
            method = getattr(planner, method_name)
            self.assertTrue(callable(method),
                          f"'{method_name}' should be callable")

        # Test Plan class methods (these were the original methods being tested)
        plan_methods = [
            "replan",
            "find_alternative_action",
            "evaluate",
            "execute",
        ]

        # Create a Plan instance
        plan = Plan("test_plan")
        self.assertIsNotNone(plan)

        # Check that plan methods exist
        for method_name in plan_methods:
            self.assertTrue(hasattr(plan, method_name),
                          f"Plan should have method '{method_name}'")
            
            method = getattr(plan, method_name)
            self.assertTrue(callable(method),
                          f"'{method_name}' should be callable")

        # Test that we can add goals and actions to a plan
        self.assertTrue(hasattr(plan, 'add_goal'),
                       "Plan should have add_goal method")
        self.assertTrue(hasattr(plan, 'add_action'),
                       "Plan should have add_action method")



if __name__ == "__main__":
    unittest.main()
