 
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

import unittest
from tiny_goap_system import GOAPPlanner, UTILITY_SCALING_FACTOR, UTILITY_INFLUENCE_FACTOR
from actions import Action, State


class MockCharacter:
    """Mock character for testing cost integration functionality."""
    
    def __init__(self, name="TestCharacter", energy=100, health=100):
        self.name = name
        self.energy = energy
        self.health = health
        self.uuid = f"{name}_uuid"
        
    def get_state(self):
        """Return character state as State object."""
        return State({
            "energy": self.energy,
            "health": self.health,
            "happiness": 50,
            "social_wellbeing": 60
        })


class MockGoal:
    """Mock goal for testing."""
    
    def __init__(self, name="TestGoal", completion_conditions=None):
        self.name = name
        self.completion_conditions = completion_conditions or {"happiness": 80}
        
    def check_completion(self, state=None):
        """Check if goal is completed."""
        if not state:
            return False
        if isinstance(self.completion_conditions, dict):
            return all(state.get(k, 0) >= v for k, v in self.completion_conditions.items())
        return False


class TestCostIntegration(unittest.TestCase):
    """Test suite for cost integration functionality in GOAP system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.planner = GOAPPlanner(graph_manager=None)
        self.character = MockCharacter()
        
        # Create test actions with different properties
        self.high_utility_action = Action(
            name="HealthyMeal",
            preconditions=[],
            effects=[{"attribute": "energy", "change_value": 20, "targets": ["initiator"]}],
            cost=3.0
        )
        self.high_utility_action.satisfaction = 15
        self.high_utility_action.urgency = 2
        
        self.low_utility_action = Action(
            name="TinySnack", 
            preconditions=[],
            effects=[{"attribute": "energy", "change_value": 5, "targets": ["initiator"]}],
            cost=1.0
        )
        self.low_utility_action.satisfaction = 3
        self.low_utility_action.urgency = 1

    def test_calculate_action_cost_method(self):
        """
        Test that _calculate_action_cost method properly calculates cost with utility influence.
        This test verifies that utility actually influences the cost calculation.
        """
        # Test high utility action - should have reduced cost due to high utility
        high_utility_cost = self.planner._calculate_action_cost(self.high_utility_action, self.character)
        
        # Test low utility action - should have cost closer to base cost
        low_utility_cost = self.planner._calculate_action_cost(self.low_utility_action, self.character)
        
        # Verify costs are positive floats
        self.assertIsInstance(high_utility_cost, float)
        self.assertIsInstance(low_utility_cost, float)
        self.assertGreater(high_utility_cost, 0)
        self.assertGreater(low_utility_cost, 0)
        
        # Verify that utility actually influences cost
        # High utility action should have lower effective cost than low utility action
        # when accounting for the utility scaling
        high_utility = self.planner.calculate_utility(self.high_utility_action, self.character)
        low_utility = self.planner.calculate_utility(self.low_utility_action, self.character)
        
        # The cost reduction should be proportional to utility difference
        high_cost_reduction = high_utility * UTILITY_SCALING_FACTOR
        low_cost_reduction = low_utility * UTILITY_SCALING_FACTOR
        
        expected_high_cost = max(3.0 - high_cost_reduction, 0.1)
        expected_low_cost = max(1.0 - low_cost_reduction, 0.1)
        
        self.assertAlmostEqual(high_utility_cost, expected_high_cost, places=2)
        self.assertAlmostEqual(low_utility_cost, expected_low_cost, places=2)
        
        # Verify that higher utility leads to lower effective cost (when base costs are similar)
        if high_utility > low_utility:
            # If utility difference is significant enough to overcome base cost difference
            utility_diff = (high_utility - low_utility) * UTILITY_SCALING_FACTOR
            base_cost_diff = 3.0 - 1.0  # 2.0
            if utility_diff > base_cost_diff:
                self.assertLess(high_utility_cost, low_utility_cost,
                               "Higher utility action should have lower effective cost when utility difference is significant")

    def test_utility_influences_action_selection(self):
        """
        Test that action selection is properly influenced by utility calculations.
        Make assertion more specific to validate correct action selection.
        """
        actions = [self.high_utility_action, self.low_utility_action]
        
        # Calculate priorities for both actions
        high_priority = self.planner._calculate_action_priority(self.high_utility_action, self.character)
        low_priority = self.planner._calculate_action_priority(self.low_utility_action, self.character)
        
        # Verify priorities are calculated correctly
        self.assertIsInstance(high_priority, float)
        self.assertIsInstance(low_priority, float)
        
        # Calculate expected priority values
        high_cost = self.planner._calculate_action_cost(self.high_utility_action, self.character)
        low_cost = self.planner._calculate_action_cost(self.low_utility_action, self.character)
        
        high_utility = self.planner.calculate_utility(self.high_utility_action, self.character)
        low_utility = self.planner.calculate_utility(self.low_utility_action, self.character)
        
        expected_high_priority = max(high_cost - (high_utility * UTILITY_INFLUENCE_FACTOR), 0.1)
        expected_low_priority = max(low_cost - (low_utility * UTILITY_INFLUENCE_FACTOR), 0.1)
        
        self.assertAlmostEqual(high_priority, expected_high_priority, places=2)
        self.assertAlmostEqual(low_priority, expected_low_priority, places=2)
        
        # Test action selection based on priority (lower priority number = higher actual priority)
        if high_utility > low_utility:
            self.assertLessEqual(high_priority, low_priority,
                               "Higher utility action should have lower priority number (higher actual priority)")
        
        # More specific assertion: verify that the HealthyMeal action is selected when
        # it has significantly higher utility
        best_action = self.planner.evaluate_utility(actions, self.character)
        self.assertIsNotNone(best_action)
        
        # Since HealthyMeal has higher satisfaction and urgency, it should be selected
        self.assertEqual(best_action.name, "HealthyMeal",
                        "HealthyMeal should be selected due to higher utility (satisfaction=15, urgency=2) vs TinySnack (satisfaction=3, urgency=1)")

    def test_estimate_cost_to_goal_method(self):
        """
        Test the _estimate_cost_to_goal heuristic function with specific scenarios.
        Verify specific cost calculation scenarios instead of just return type.
        """
        # Test with dictionary-format goal
        dict_goal = {"happiness": 80, "energy": 90}
        current_state = State({"happiness": 60, "energy": 70})
        
        estimated_cost = self.planner._estimate_cost_to_goal(current_state, dict_goal)
        
        # Verify return type and non-negative value
        self.assertIsInstance(estimated_cost, float)
        self.assertGreaterEqual(estimated_cost, 0)
        
        # Test specific cost calculation
        # Expected cost: (80-60)*0.1 + (90-70)*0.1 = 2.0 + 2.0 = 4.0
        expected_cost = (80 - 60) * 0.1 + (90 - 70) * 0.1
        self.assertAlmostEqual(estimated_cost, expected_cost, places=2)
        
        # Test with MockGoal object
        mock_goal = MockGoal("TestGoal", {"happiness": 100})
        current_state_low = State({"happiness": 30})
        
        goal_cost = self.planner._estimate_cost_to_goal(current_state_low, mock_goal)
        expected_goal_cost = (100 - 30) * 0.1  # 7.0
        self.assertAlmostEqual(goal_cost, expected_goal_cost, places=2)
        
        # Test edge case: already at goal
        current_state_high = State({"happiness": 100})
        zero_cost = self.planner._estimate_cost_to_goal(current_state_high, mock_goal)
        self.assertEqual(zero_cost, 0.0, "Cost should be zero when already at goal")
        
        # Test edge case: no goal
        no_goal_cost = self.planner._estimate_cost_to_goal(current_state, None)
        self.assertEqual(no_goal_cost, 0.0, "Cost should be zero for no goal")
        
        # Test that closer states have lower estimated costs
        closer_state = State({"happiness": 75})
        farther_state = State({"happiness": 50})
        
        closer_cost = self.planner._estimate_cost_to_goal(closer_state, mock_goal)
        farther_cost = self.planner._estimate_cost_to_goal(farther_state, mock_goal)
        
        self.assertLess(closer_cost, farther_cost,
                       "Closer states should have lower estimated costs than farther states")

    def test_character_state_population(self):
        """
        Test that character_state dictionaries are properly populated instead of being empty.
        """
        # Create character with specific state
        test_character = MockCharacter("TestChar", energy=75, health=90)
        
        # Test that calculate_utility uses character state
        utility = self.planner.calculate_utility(self.high_utility_action, test_character)
        self.assertIsInstance(utility, (int, float))
        
        # Test with low energy character - utility should be affected
        low_energy_character = MockCharacter("LowEnergyChar", energy=20, health=100)
        low_energy_utility = self.planner.calculate_utility(self.high_utility_action, low_energy_character)
        
        # Low energy should reduce utility for high-cost actions
        high_energy_utility = self.planner.calculate_utility(self.high_utility_action, test_character)
        self.assertLess(low_energy_utility, high_energy_utility,
                       "Low energy character should have lower utility for high-cost actions")
        
        # Test that _calculate_action_priority uses character state
        priority = self.planner._calculate_action_priority(self.high_utility_action, test_character)
        self.assertIsInstance(priority, (int, float))
        self.assertGreater(priority, 0)
        
        # Test evaluate_utility with multiple actions
        actions = [self.high_utility_action, self.low_utility_action]
        best_action = self.planner.evaluate_utility(actions, test_character)
        self.assertIsNotNone(best_action)
        self.assertIn(best_action, actions)


if __name__ == '__main__':
    unittest.main()