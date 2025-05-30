import unittest
from tiny_utility_functions import (
    calculate_action_utility,
    calculate_plan_utility,
    Goal # Import the placeholder Goal
)
# We need a mock or placeholder for the Action class.
# If actions.py is too complex to import, we define a simple one here for testing utility.
# The actual Action class from actions.py has a complex __init__ and dependencies.
# For testing utility functions, we only need name, cost, and effects.

class MockAction:
    def __init__(self, name, cost, effects=None):
        self.name = name
        self.cost = float(cost)
        # Effects is a list of dictionaries, e.g., [{'attribute': 'hunger', 'change_value': -0.5}]
        self.effects = effects if effects else []

    def __repr__(self):
        return f"MockAction(name='{self.name}', cost={self.cost}, effects={self.effects})"

class TestTinyUtilityFunctions(unittest.TestCase):

    def test_calculate_action_utility_satisfy_high_hunger(self):
        char_state = {"hunger": 0.9, "energy": 0.5} # High hunger
        action = MockAction("EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}])
        # Expected: hunger_score = 0.9 * 0.7 * 20 = 12.6
        # cost_score = 0.1 * 10 = 1.0
        # utility = 12.6 - 1.0 = 11.6
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, 11.6)

    def test_calculate_action_utility_satisfy_low_hunger(self):
        char_state = {"hunger": 0.1, "energy": 0.5} # Low hunger
        action = MockAction("EatSnack", cost=0.05, effects=[{"attribute": "hunger", "change_value": -0.2}])
        # Expected: hunger_score = 0.1 * 0.2 * 20 = 0.4
        # cost_score = 0.05 * 10 = 0.5
        # utility = 0.4 - 0.5 = -0.1
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, -0.1)

    def test_calculate_action_utility_increase_low_energy(self):
        char_state = {"hunger": 0.3, "energy": 0.2} # Low energy
        action = MockAction("Rest", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.6}])
        # Expected: energy_score = (1.0 - 0.2) * 0.6 * 15 = 0.8 * 0.6 * 15 = 7.2
        # cost_score = 0.5 * 10 = 5.0
        # utility = 7.2 - 5.0 = 2.2
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, 2.2)

    def test_calculate_action_utility_increase_high_energy_low_benefit(self):
        char_state = {"hunger": 0.3, "energy": 0.9} # High energy
        action = MockAction("ShortNap", cost=0.1, effects=[{"attribute": "energy", "change_value": 0.1}])
        # Expected: energy_score = (1.0 - 0.9) * 0.1 * 15 = 0.1 * 0.1 * 15 = 0.15
        # cost_score = 0.1 * 10 = 1.0
        # utility = 0.15 - 1.0 = -0.85
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, -0.85)
        
    def test_calculate_action_utility_get_money(self):
        char_state = {"money": 50, "energy": 0.6}
        action = MockAction("WorkShift", cost=0.4, effects=[
            {"attribute": "money", "change_value": 20},
            {"attribute": "energy", "change_value": -0.3} # Work costs energy
        ])
        # Expected: money_score = 20 * 0.5 = 10.0
        # energy_score for cost: (1.0 - 0.6) * (-0.3) * 15 -> this is tricky, energy change is a cost not a direct need fulfillment here.
        # The current logic for energy in need_fulfillment is only for positive changes.
        # Negative energy changes from effects are not directly penalized as negative need fulfillment.
        # They would be implicitly handled if 'energy' was a cost of the action, or if plan simulation is on.
        # For now, only money_score applies from effects.
        # cost_score = 0.4 * 10 = 4.0
        # utility = 10.0 - 4.0 = 6.0
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, 6.0)


    def test_calculate_action_utility_high_cost(self):
        char_state = {"hunger": 0.5, "energy": 0.5}
        action = MockAction("ExpensiveTask", cost=2.0, effects=[{"attribute": "hunger", "change_value": -0.1}])
        # Expected: hunger_score = 0.5 * 0.1 * 20 = 1.0
        # cost_score = 2.0 * 10 = 20.0
        # utility = 1.0 - 20.0 = -19.0
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, -19.0)

    def test_calculate_action_utility_with_goal_match(self):
        char_state = {"hunger": 0.8, "energy": 0.7}
        action = MockAction("EatHealthyMeal", cost=0.2, effects=[{"attribute": "hunger", "change_value": -0.6}])
        goal = Goal("SatisfyHunger", target_effects={"hunger": -0.8}, priority=0.8)
        # Expected: hunger_score = 0.8 * 0.6 * 20 = 9.6
        # goal_progress_score = 0.8 * 25.0 = 20.0 (action effect 'hunger' matches goal target_effect 'hunger' and is beneficial)
        # cost_score = 0.2 * 10 = 2.0
        # utility = 9.6 + 20.0 - 2.0 = 27.6
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertAlmostEqual(utility, 27.6)

    def test_calculate_action_utility_with_goal_no_match_effect(self):
        char_state = {"energy": 0.3}
        action = MockAction("Rest", cost=0.1, effects=[{"attribute": "energy", "change_value": 0.5}])
        goal = Goal("GetFood", target_effects={"hunger": -0.5}, priority=0.9) # Goal is different
        # Expected: energy_score = (1.0 - 0.3) * 0.5 * 15 = 0.7 * 0.5 * 15 = 5.25
        # goal_progress_score = 0.0 (action effect 'energy' does not match goal target_effect 'hunger')
        # cost_score = 0.1 * 10 = 1.0
        # utility = 5.25 + 0.0 - 1.0 = 4.25
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertAlmostEqual(utility, 4.25)
        
    def test_calculate_action_utility_goal_effect_opposite_direction(self):
        char_state = {"energy": 0.8} # Already high energy
        # Goal is to *reduce* energy (e.g. "Calm Down" goal, not well represented by target_effects this way)
        # Let's test a goal to *gain* energy, but action *reduces* it.
        goal = Goal("GainEnergy", target_effects={"energy": 0.5}, priority=1.0)
        action = MockAction("RunMarathon", cost=1.0, effects=[{"attribute": "energy", "change_value": -0.8}])
        # Expected: energy_score = 0 (action reduces energy, not a direct "need fulfillment" for positive gain)
        # goal_progress_score = 0 (action reduces energy, goal wants to increase)
        # cost_score = 1.0 * 10 = 10.0
        # utility = 0 + 0 - 10.0 = -10.0
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertAlmostEqual(utility, -10.0)


    # --- Tests for calculate_plan_utility ---

    def test_calculate_plan_utility_single_action(self):
        char_state = {"hunger": 0.9, "energy": 0.5}
        action1 = MockAction("EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}])
        # Utility of action1 = 11.6 (from previous test)
        plan = [action1]
        plan_utility = calculate_plan_utility(char_state, plan)
        self.assertAlmostEqual(plan_utility, 11.6)

    def test_calculate_plan_utility_multiple_actions_no_simulation(self):
        char_state = {"hunger": 0.9, "energy": 0.2}
        action1 = MockAction("EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}]) # Util = 0.9 * 0.7 * 20 - 0.1*10 = 12.6 - 1 = 11.6
        action2 = MockAction("Rest", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.6}])    # Util = (1-0.2)*0.6*15 - 0.5*10 = 7.2 - 5 = 2.2
        plan = [action1, action2]
        # Expected: 11.6 + 2.2 = 13.8
        plan_utility = calculate_plan_utility(char_state, plan, simulate_effects=False)
        self.assertAlmostEqual(plan_utility, 13.8)

    def test_calculate_plan_utility_multiple_actions_with_simulation(self):
        char_state = {"hunger": 0.9, "energy": 0.2, "money": 10}
        # Action 1: EatFood
        # Initial state: hunger=0.9, energy=0.2
        # Effects: hunger -= 0.7. Cost: 0.1
        # Utility1 = (0.9 * 0.7 * 20) - (0.1 * 10) = 12.6 - 1.0 = 11.6
        action1 = MockAction("EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}])
        
        # Simulated state after action1: hunger = 0.9 - 0.7 = 0.2, energy = 0.2
        
        # Action 2: Rest
        # State for utility calc: hunger=0.2, energy=0.2
        # Effects: energy += 0.6. Cost: 0.5
        # Utility2 = ((1.0 - 0.2) * 0.6 * 15) - (0.5 * 10) = (0.8 * 0.6 * 15) - 5.0 = 7.2 - 5.0 = 2.2
        action2 = MockAction("Rest", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.6}])
        
        plan = [action1, action2]
        # Expected total utility = 11.6 + 2.2 = 13.8
        plan_utility = calculate_plan_utility(char_state, plan, simulate_effects=True)
        self.assertAlmostEqual(plan_utility, 13.8) # Note: In this case, simulation didn't change second action's utility because effects were on different attributes.

    def test_calculate_plan_utility_simulation_affects_utility(self):
        char_state = {"hunger": 0.9, "energy": 0.5}
        # Action 1: EatBigMeal
        # Initial state: hunger=0.9
        # Effects: hunger -= 0.8. Cost: 0.2
        # Utility1 = (0.9 * 0.8 * 20) - (0.2 * 10) = 14.4 - 2.0 = 12.4
        action1 = MockAction("EatBigMeal", cost=0.2, effects=[{"attribute": "hunger", "change_value": -0.8}])
        
        # Simulated state after action1: hunger = 0.9 - 0.8 = 0.1
        
        # Action 2: EatSnack
        # State for utility calc: hunger=0.1
        # Effects: hunger -= 0.2. Cost: 0.05
        # Utility2 = (0.1 * 0.2 * 20) - (0.05 * 10) = 0.4 - 0.5 = -0.1
        action2 = MockAction("EatSnack", cost=0.05, effects=[{"attribute": "hunger", "change_value": -0.2}])
        
        plan = [action1, action2]
        # Expected total utility = 12.4 + (-0.1) = 12.3
        plan_utility_simulated = calculate_plan_utility(char_state, plan, simulate_effects=True)
        self.assertAlmostEqual(plan_utility_simulated, 12.3)

        # For comparison, without simulation:
        # Utility1 (no change) = 12.4
        # Utility2 (state is still hunger=0.9) = (0.9 * 0.2 * 20) - (0.05 * 10) = 3.6 - 0.5 = 3.1
        # Expected total utility (no sim) = 12.4 + 3.1 = 15.5
        plan_utility_no_sim = calculate_plan_utility(char_state, plan, simulate_effects=False)
        self.assertAlmostEqual(plan_utility_no_sim, 15.5)
        self.assertNotAlmostEqual(plan_utility_simulated, plan_utility_no_sim)

    def test_plan_utility_with_goal(self):
        char_state = {"hunger": 0.7, "energy": 0.4}
        goal = Goal("GetFullAndEnergized", target_effects={"hunger": -0.7, "energy": 0.6}, priority=1.0)

        # Action 1: Eat (helps hunger goal)
        # Utility1: hunger_score = 0.7 * 0.5 * 20 = 7.0
        # goal_score (hunger part) = 1.0 * 25.0 = 25.0
        # cost_score = 0.1 * 10 = 1.0
        # Util1 = 7.0 + 25.0 - 1.0 = 31.0
        action1 = MockAction("Eat", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.5}])
        
        # Simulated state after action1: hunger = 0.7 - 0.5 = 0.2, energy = 0.4

        # Action 2: Rest (helps energy goal)
        # Utility2: energy_score = (1.0 - 0.4) * 0.5 * 15 = 0.6 * 0.5 * 15 = 4.5
        # goal_score (energy part) = 1.0 * 25.0 = 25.0
        # cost_score = 0.2 * 10 = 2.0
        # Util2 = 4.5 + 25.0 - 2.0 = 27.5
        action2 = MockAction("Rest", cost=0.2, effects=[{"attribute": "energy", "change_value": 0.5}])
        
        plan = [action1, action2]
        # Expected: 31.0 + 27.5 = 58.5
        plan_utility = calculate_plan_utility(char_state, plan, current_goal=goal, simulate_effects=True)
        self.assertAlmostEqual(plan_utility, 58.5)

if __name__ == '__main__':
    unittest.main()
