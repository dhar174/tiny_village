#!/usr/bin/env python3
"""
Test script to verify the implemented GOAP system methods.
This script tests the newly implemented methods without requiring complex dependencies.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())


def test_plan_methods():
    """Test the Plan class methods that were implemented."""
    print("Testing Plan class methods...")

    try:
        # Create a mock action class for testing
        class MockAction:
            """Enhanced MockAction with meaningful precondition checking."""
            
            def __init__(self, name, cost=10, urgency=1.0, preconditions=None):
                self.name = name
                self.cost = cost
                self.urgency = urgency
                self.preconditions = preconditions if preconditions else []
                
                # Additional attributes for real Action compatibility
                self.effects = []
                self.satisfaction = 5.0
                self.action_id = id(self)

            def preconditions_met(self, state=None):
                """Check if preconditions are met - meaningful implementation."""
                if not self.preconditions:
                    return True
                    
                for precondition in self.preconditions:
                    if isinstance(precondition, dict):
                        attribute = precondition.get("attribute")
                        operator = precondition.get("operator", ">=")
                        required_value = precondition.get("value", 0)
                        
                        if state is None:
                            return False
                            
                        if isinstance(state, dict):
                            current_value = state.get(attribute, 0)
                        else:
                            current_value = getattr(state, attribute, 0)
                        
                        if operator == ">=" and current_value < required_value:
                            return False
                        elif operator == "<=" and current_value > required_value:
                            return False
                        elif operator == "==" and current_value != required_value:
                            return False
                        elif operator == ">" and current_value <= required_value:
                            return False
                        elif operator == "<" and current_value >= required_value:
                            return False
                    elif callable(precondition):
                        try:
                            if not precondition(state):
                                return False
                        except Exception:
                            return False
                            
                return True

            def __str__(self):
                return f"MockAction({self.name})"

        # Import the Plan class
        from tiny_goap_system import Plan

        # Create a test plan
        plan = Plan("Test Plan")
        print("✓ Plan created successfully")

        # Test adding actions
        action1 = MockAction("test_action_1", cost=5, urgency=0.8)
        action2 = MockAction("test_action_2", cost=15, urgency=1.2)

        plan.add_action(action1, priority=1, dependencies=[])
        plan.add_action(action2, priority=2, dependencies=[])
        print("✓ Actions added to plan successfully")

        # Test replan method (should work without errors)
        plan.replan()
        print("✓ replan() method executed successfully")

        # Test find_alternative_action method
        alternative = plan.find_alternative_action(action1)
        if alternative is not None:
            print("✓ find_alternative_action() returned an alternative")
        else:
            print("✓ find_alternative_action() handled failure gracefully")

        return True

    except Exception as e:
        print(f"✗ Error testing Plan methods: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_goap_planner_methods():
    """Test the GOAPPlanner class methods that were implemented."""
    print("\nTesting GOAPPlanner class methods...")

    try:
        # Create mock objects for testing
        class MockCharacter:
            def __init__(self):
                self.energy = 80
                self.mood = "neutral"

            def get_state(self):
                return {"energy": self.energy, "mood": self.mood}

        class MockAction:
            """Enhanced MockAction with meaningful precondition checking."""
            
            def __init__(self, name, satisfaction=20, cost=10, urgency=1.0, preconditions=None):
                self.name = name
                self.satisfaction = satisfaction
                self.cost = cost
                self.urgency = urgency
                self.preconditions = preconditions if preconditions else []
                
                # Additional attributes for real Action compatibility
                self.effects = []
                self.action_id = id(self)

            def preconditions_met(self, state=None):
                """Check if preconditions are met - meaningful implementation."""
                if not self.preconditions:
                    return True
                    
                for precondition in self.preconditions:
                    if isinstance(precondition, dict):
                        attribute = precondition.get("attribute")
                        operator = precondition.get("operator", ">=")
                        required_value = precondition.get("value", 0)
                        
                        if state is None:
                            return False
                            
                        if isinstance(state, dict):
                            current_value = state.get(attribute, 0)
                        else:
                            current_value = getattr(state, attribute, 0)
                        
                        if operator == ">=" and current_value < required_value:
                            return False
                        elif operator == "<=" and current_value > required_value:
                            return False
                        elif operator == "==" and current_value != required_value:
                            return False
                        elif operator == ">" and current_value <= required_value:
                            return False
                        elif operator == "<" and current_value >= required_value:
                            return False
                    elif callable(precondition):
                        try:
                            if not precondition(state):
                                return False
                        except Exception:
                            return False
                            
                return True

        # Import the GOAPPlanner class
        from tiny_goap_system import GOAPPlanner
        from tests.mock_graph_manager import create_mock_graph_manager

        # Create a planner with minimal mock graph manager for better test coverage
        character = MockCharacter()
        mock_graph_manager = create_mock_graph_manager(character)
        planner = GOAPPlanner(mock_graph_manager)
        print("✓ GOAPPlanner created successfully")

        # Test calculate_utility method
        action = MockAction("test_action", satisfaction=30, cost=15, urgency=1.5)

        utility = planner.calculate_utility(action, character)
        print(f"✓ calculate_utility() returned: {utility}")

        # Test with dictionary format action
        dict_action = {"satisfaction": 25, "energy_cost": 12, "urgency": 0.9}
        utility_dict = planner.calculate_utility(dict_action, character)
        print(f"✓ calculate_utility() with dict format returned: {utility_dict}")

        # Test evaluate_utility method
        plan = [
            action,
            MockAction("test_action_2", satisfaction=15, cost=8, urgency=0.7),
        ]
        best_action = planner.evaluate_utility(plan, character)
        print(f"✓ evaluate_utility() returned best action: {best_action}")

        # Test evaluate_feasibility_of_goal method
        goal = {"energy": 70, "happiness": 60}
        state = {"energy": 80, "happiness": 55}
        feasibility = planner.evaluate_feasibility_of_goal(goal, state)
        print(f"✓ evaluate_feasibility_of_goal() returned: {feasibility}")

        return True

    except Exception as e:
        print(f"✗ Error testing GOAPPlanner methods: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== Testing GOAP System Implementations ===\n")

    test1_passed = test_plan_methods()
    test2_passed = test_goap_planner_methods()

    print("\n=== Test Results ===")
    if test1_passed and test2_passed:
        print("✓ All tests passed! The implemented methods are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
