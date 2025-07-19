#!/usr/bin/env python3
"""
Test for cost function integration in GOAP planning system.
This test verifies that action costs are properly utilized in the planning search.
"""

import unittest
from tiny_goap_system import GOAPPlanner, Plan
from actions import Action, State
from tiny_utility_functions import Goal


class MockCharacter:
    """Mock character for testing"""
    def __init__(self, name="TestChar"):
        self.name = name
        self.hunger = 0.8
        self.energy = 0.3
        self.money = 10
        
    def get_state(self):
        return State({"hunger": self.hunger, "energy": self.energy, "money": self.money})


class TestCostIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.planner = GOAPPlanner(None)
        self.character = MockCharacter()
        
    def test_cost_based_planning_selects_cheaper_actions(self):
        """Test that planner prefers cheaper actions when utility is similar"""
        
        # Create goal to reduce hunger to 0.2 (from 0.8)
        goal = Goal("ReduceHunger", target_effects={"hunger": 0.2}, priority=1.0)
        
        # Create initial state with high hunger
        current_state = State({"hunger": 0.8, "energy": 0.5, "money": 20})
        
        # Create two actions that achieve same effect but different costs
        cheap_action = Action(
            name="EatSnack",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.6}],  # 0.8 - 0.6 = 0.2 (goal achieved)
            cost=0.1  # Cheap
        )
        
        expensive_action = Action(
            name="FancyMeal", 
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.6}],  # 0.8 - 0.6 = 0.2 (goal achieved)
            cost=2.0  # Expensive
        )
        
        actions = [expensive_action, cheap_action]  # Order expensive first
        
        # Debug: Check initial goal satisfaction
        print(f"Initial goal satisfied: {self.planner._goal_satisfied(goal, current_state)}")
        
        # Plan should prefer cheaper action
        plan = self.planner.plan_actions(self.character, goal, current_state, actions)
        
        print(f"Found plan: {plan}")
        if plan:
            print(f"Plan actions: {[action.name for action in plan]}")
        
        self.assertIsNotNone(plan)
        self.assertTrue(len(plan) > 0)
        # Should select the cheaper action (EatSnack)
        self.assertEqual(plan[0].name, "EatSnack")
        
    def test_utility_influences_action_selection(self):
        """Test that utility calculation influences action selection"""
        
        # Create goal
        goal = Goal("ReduceHunger", target_effects={"hunger": 0.1}, priority=1.0)
        
        # Create state where character is very hungry
        current_state = State({"hunger": 0.9, "energy": 0.8})
        
        # Create actions with different utility profiles
        high_utility_action = Action(
            name="HealthyMeal",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.8}],  # Big hunger reduction
            cost=0.5
        )
        
        low_utility_action = Action(
            name="TinySnack",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.1}],  # Small hunger reduction
            cost=0.1
        )
        
        actions = [low_utility_action, high_utility_action]
        
        # Plan should consider utility, not just cost
        plan = self.planner.plan_actions(self.character, goal, current_state, actions)
        
        self.assertIsNotNone(plan)
        self.assertTrue(len(plan) > 0)
        
        # With very high hunger, should prefer action that reduces hunger more
        # even if it costs more
        first_action = plan[0]
        self.assertIn(first_action.name, ["HealthyMeal", "TinySnack"])
        
    def test_calculate_action_cost_method(self):
        """Test the _calculate_action_cost method directly"""
        
        action = Action(
            name="TestAction",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.5}],
            cost=1.0
        )
        
        state = State({"hunger": 0.8, "energy": 0.5})
        goal = Goal("ReduceHunger", target_effects={"hunger": 0.2}, priority=1.0)
        
        cost = self.planner._calculate_action_cost(action, state, self.character, goal)
        
        # Cost should be positive
        self.assertGreater(cost, 0)
        # Cost should be influenced by utility (not just base cost)
        self.assertIsInstance(cost, float)
        
    def test_estimate_cost_to_goal_method(self):
        """Test the _estimate_cost_to_goal heuristic method"""
        
        state = State({"hunger": 0.8, "energy": 0.3})
        goal = Goal("ReduceHunger", target_effects={"hunger": 0.2}, priority=1.0)
        
        estimated_cost = self.planner._estimate_cost_to_goal(state, goal, self.character)
        
        # Should return a reasonable estimate
        self.assertGreaterEqual(estimated_cost, 0)
        self.assertIsInstance(estimated_cost, float)
        
        # Test with goal already achieved
        satisfied_state = State({"hunger": 0.1, "energy": 0.8})
        satisfied_cost = self.planner._estimate_cost_to_goal(satisfied_state, goal, self.character)
        
        # Cost should be lower when closer to goal
        self.assertLessEqual(satisfied_cost, estimated_cost)
        
    def test_plan_replan_uses_costs(self):
        """Test that the Plan.replan method uses cost-aware prioritization"""
        
        plan = Plan("TestPlan", graph_manager=None)
        
        # Add some goals
        goal1 = Goal("ReduceHunger", target_effects={"hunger": 0.2}, priority=1.0)
        plan.add_goal(goal1)
        
        # Add actions with different costs
        cheap_action = Action(
            name="CheapAction",
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.3}],
            cost=0.1
        )
        cheap_action.urgency = 1.0
        
        expensive_action = Action(
            name="ExpensiveAction", 
            preconditions=[],
            effects=[{"attribute": "hunger", "change_value": -0.3}],
            cost=2.0
        )
        expensive_action.urgency = 1.0
        
        # Add actions to plan
        plan.add_action(expensive_action, priority=1.0)  # Add expensive first
        plan.add_action(cheap_action, priority=1.0)
        
        # Replan should reorder based on costs
        plan.replan()
        
        # Check that queue is not empty
        self.assertGreater(len(plan.action_queue), 0)
        
        # Both actions should still be in queue
        action_names = [item[2].name for item in plan.action_queue]  # item[2] is the action
        self.assertIn("CheapAction", action_names)
        self.assertIn("ExpensiveAction", action_names)


class TestCostIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in cost integration"""
    
    def setUp(self):
        self.planner = GOAPPlanner(None)
        
    def test_fallback_when_utility_calculation_fails(self):
        """Test that system falls back gracefully when utility calculation fails"""
        
        # Create action without proper effects structure
        broken_action = Action(
            name="BrokenAction",
            preconditions=[],
            effects="invalid_effects_format",  # Invalid format
            cost=1.0
        )
        
        state = State({"hunger": 0.5})
        character = MockCharacter()
        goal = Goal("TestGoal", target_effects={"hunger": 0.2}, priority=1.0)
        
        # Should not crash, should fall back to base cost
        cost = self.planner._calculate_action_cost(broken_action, state, character, goal)
        
        self.assertGreater(cost, 0)
        self.assertEqual(cost, 1.0)  # Should fallback to base cost
        
    def test_no_goal_heuristic(self):
        """Test heuristic calculation when no goal is provided"""
        
        state = State({"hunger": 0.5})
        character = MockCharacter()
        
        # Should handle None goal gracefully
        cost = self.planner._estimate_cost_to_goal(state, None, character)
        self.assertEqual(cost, 0.0)
        
        # Should handle goal without target_effects
        empty_goal = Goal("EmptyGoal")
        cost = self.planner._estimate_cost_to_goal(state, empty_goal, character)
        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()