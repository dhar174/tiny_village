import unittest
from tiny_utility_functions import (
    calculate_action_utility,
    calculate_plan_utility,
)
import importlib

# Importing the Goal class from tiny_characters module (but using a mock for testing)
# Goal = importlib.import_module("tiny_characters").Goal


# We need mock classes for testing since the real classes have complex dependencies
class MockGoal:
    """
    Enhanced MockGoal that more closely mirrors real Goal class attributes.
    
    This includes additional attributes that goal systems might use for
    more comprehensive testing of utility calculations.
    """

    def __init__(self, name, target_effects=None, priority=0.5, urgency=None, 
                 deadline=None, description=None):
        self.name = name
        self.target_effects = target_effects if target_effects else {}
        self.priority = priority
        self.score = priority  # alias for compatibility
        
        # Enhanced attributes that real Goal classes might have
        self.urgency = urgency if urgency is not None else priority  # Defaults to priority
        self.deadline = deadline
        self.description = description or f"Goal: {name}"


# Alias for test compatibility
Goal = MockGoal

# We need a mock or placeholder for the Action class.
# If actions.py is too complex to import, we define a simple one here for testing utility.
# The actual Action class from actions.py has a complex __init__ and dependencies.
# For testing utility functions, we only need name, cost, and effects.


class MockAction:
    """
    Enhanced MockAction that more closely mirrors the real Action class attributes.
    
    This includes critical attributes that the real Action class has to ensure
    tests properly validate utility calculations and don't pass with broken implementations.
    """
    def __init__(self, name, cost, effects=None, preconditions=None, target=None, 
                 initiator=None, priority=None, related_goal=None, action_id=None,
                 default_target_is_initiator=False):
        self.name = name
        self.cost = float(cost)
        # Effects is a list of dictionaries, e.g., [{'attribute': 'hunger', 'change_value': -0.5}]
        self.effects = effects if effects else []
        
        # Enhanced attributes that real Action class has
        self.preconditions = preconditions if preconditions else []
        self.target = target
        self.initiator = initiator
        self.priority = priority if priority is not None else 0.5
        self.related_goal = related_goal
        self.action_id = action_id if action_id else id(self)
        self.default_target_is_initiator = default_target_is_initiator
        
        # Set target to initiator if default_target_is_initiator is True and no target provided
        if self.default_target_is_initiator and self.target is None and self.initiator is not None:
            self.target = self.initiator
        
        # Validate effects structure to catch issues early
        self._validate_effects()
    
    def _validate_effects(self):
        """Validate effects structure similar to real Action class."""
        if not isinstance(self.effects, list):
            raise ValueError("Effects must be a list")
        
        for i, effect in enumerate(self.effects):
            if not isinstance(effect, dict):
                raise ValueError(f"Effect {i} must be a dictionary")
            
            if "attribute" not in effect:
                raise ValueError(f"Effect {i} must have 'attribute' key")
            
            if "change_value" not in effect:
                raise ValueError(f"Effect {i} must have 'change_value' key")
            
            if not isinstance(effect["change_value"], (int, float)):
                raise ValueError(f"Effect {i} change_value must be numeric")
    
    def preconditions_met(self, character_state=None):
        """
        Check if preconditions are met. Simple implementation for testing.
        
        Args:
            character_state: Optional state to check against (for future enhancements)
        
        Returns:
            bool: True if preconditions are met (always True for mock unless explicitly set)
        """
        if not self.preconditions:
            return True
        
        # Simple validation - in real implementation this would check actual conditions
        # For now, assume preconditions are met unless they're explicitly set to False
        for precondition in self.preconditions:
            if isinstance(precondition, bool) and not precondition:
                return False
            # Could add more sophisticated checking here if needed for specific tests
        
        return True
    
    def add_precondition(self, precondition):
        """Add a precondition to the action."""
        self.preconditions.append(precondition)
    
    def add_effect(self, effect):
        """Add an effect to the action."""
        # Validate the new effect
        if not isinstance(effect, dict):
            raise ValueError("Effect must be a dictionary")
        if "attribute" not in effect or "change_value" not in effect:
            raise ValueError("Effect must have 'attribute' and 'change_value' keys")
        if not isinstance(effect["change_value"], (int, float)):
            raise ValueError("Effect change_value must be numeric")
        
        self.effects.append(effect)

    def __repr__(self):
        return (
            f"MockAction(name='{self.name}', cost={self.cost}, effects={self.effects}, "
            f"preconditions={len(self.preconditions)}, target={self.target}, "
            f"priority={self.priority})"
        )


class TestTinyUtilityFunctions(unittest.TestCase):

    def setUp(self):
        """Set up any necessary state before each test."""
        # This can be used to initialize common variables or states for tests.
        

    def test_calculate_action_utility_satisfy_high_hunger(self):
        char_state = {"hunger": 0.9, "energy": 0.5}  # High hunger
        action = MockAction(
            "EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}]
        )
        # Expected: hunger_score = 0.9 * 0.7 * 20 = 12.6
        # cost_score = 0.1 * 10 = 1.0
        # utility = 12.6 - 1.0 = 11.6
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, 11.6)

    def test_calculate_action_utility_satisfy_low_hunger(self):
        char_state = {"hunger": 0.1, "energy": 0.5}  # Low hunger
        action = MockAction(
            "EatSnack",
            cost=0.05,
            effects=[{"attribute": "hunger", "change_value": -0.2}],
        )
        # Expected: hunger_score = 0.1 * 0.2 * 20 = 0.4
        # cost_score = 0.05 * 10 = 0.5
        # utility = 0.4 - 0.5 = -0.1
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, -0.1)

    def test_calculate_action_utility_increase_low_energy(self):
        char_state = {"hunger": 0.3, "energy": 0.2}  # Low energy
        action = MockAction(
            "Rest", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.6}]
        )
        # Expected: energy_score = (1.0 - 0.2) * 0.6 * 15 = 0.8 * 0.6 * 15 = 7.2
        # cost_score = 0.5 * 10 = 5.0
        # utility = 7.2 - 5.0 = 2.2
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, 2.2)

    def test_calculate_action_utility_increase_high_energy_low_benefit(self):
        char_state = {"hunger": 0.3, "energy": 0.9}  # High energy
        action = MockAction(
            "ShortNap", cost=0.1, effects=[{"attribute": "energy", "change_value": 0.1}]
        )
        # Expected: energy_score = (1.0 - 0.9) * 0.1 * 15 = 0.1 * 0.1 * 15 = 0.15
        # cost_score = 0.1 * 10 = 1.0
        # utility = 0.15 - 1.0 = -0.85
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, -0.85)

    def test_calculate_action_utility_get_money(self):
        char_state = {"money": 50, "energy": 0.6}
        action = MockAction(
            "WorkShift",
            cost=0.4,
            effects=[
                {"attribute": "money", "change_value": 20},
                {"attribute": "energy", "change_value": -0.3},  # Work costs energy
            ],
        )
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
        action = MockAction(
            "ExpensiveTask",
            cost=2.0,
            effects=[{"attribute": "hunger", "change_value": -0.1}],
        )
        # Expected: hunger_score = 0.5 * 0.1 * 20 = 1.0
        # cost_score = 2.0 * 10 = 20.0
        # utility = 1.0 - 20.0 = -19.0
        utility = calculate_action_utility(char_state, action)
        self.assertAlmostEqual(utility, -19.0)

    def test_calculate_action_utility_with_goal_match(self):
        char_state = {"hunger": 0.8, "energy": 0.7}
        action = MockAction(
            "EatHealthyMeal",
            cost=0.2,
            effects=[{"attribute": "hunger", "change_value": -0.6}],
        )
        goal = Goal("SatisfyHunger", target_effects={"hunger": -0.8}, priority=0.8)
        # Expected: hunger_score = 0.8 * 0.6 * 20 = 9.6
        # goal_progress_score = 0.8 * 25.0 = 20.0 (action effect 'hunger' matches goal target_effect 'hunger' and is beneficial)
        # cost_score = 0.2 * 10 = 2.0
        # utility = 9.6 + 20.0 - 2.0 = 27.6
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertAlmostEqual(utility, 27.6)

    def test_calculate_action_utility_with_goal_no_match_effect(self):
        char_state = {"energy": 0.3}
        action = MockAction(
            "Rest", cost=0.1, effects=[{"attribute": "energy", "change_value": 0.5}]
        )
        goal = Goal(
            "GetFood", target_effects={"hunger": -0.5}, priority=0.9
        )  # Goal is different
        # Expected: energy_score = (1.0 - 0.3) * 0.5 * 15 = 0.7 * 0.5 * 15 = 5.25
        # goal_progress_score = 0.0 (action effect 'energy' does not match goal target_effect 'hunger')
        # cost_score = 0.1 * 10 = 1.0
        # utility = 5.25 + 0.0 - 1.0 = 4.25
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertAlmostEqual(utility, 4.25)

    def test_calculate_action_utility_goal_effect_opposite_direction(self):
        char_state = {"energy": 0.8}  # Already high energy
        # Goal is to *reduce* energy (e.g. "Calm Down" goal, not well represented by target_effects this way)
        # Let's test a goal to *gain* energy, but action *reduces* it.
        goal = Goal("GainEnergy", target_effects={"energy": 0.5}, priority=1.0)
        action = MockAction(
            "RunMarathon",
            cost=1.0,
            effects=[{"attribute": "energy", "change_value": -0.8}],
        )
        # Expected: energy_score = 0 (action reduces energy, not a direct "need fulfillment" for positive gain)
        # goal_progress_score = 0 (action reduces energy, goal wants to increase)
        # cost_score = 1.0 * 10 = 10.0
        # utility = 0 + 0 - 10.0 = -10.0
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertAlmostEqual(utility, -10.0)

    # --- Tests for calculate_plan_utility ---

    def test_calculate_plan_utility_single_action(self):
        char_state = {"hunger": 0.9, "energy": 0.5}
        action1 = MockAction(
            "EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}]
        )
        # Utility of action1 = 11.6 (from previous test)
        plan = [action1]
        plan_utility = calculate_plan_utility(char_state, plan)
        self.assertAlmostEqual(plan_utility, 11.6)

    def test_calculate_plan_utility_multiple_actions_no_simulation(self):
        char_state = {"hunger": 0.9, "energy": 0.2}
        action1 = MockAction(
            "EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}]
        )  # Util = 0.9 * 0.7 * 20 - 0.1*10 = 12.6 - 1 = 11.6
        action2 = MockAction(
            "Rest", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.6}]
        )  # Util = (1-0.2)*0.6*15 - 0.5*10 = 7.2 - 5 = 2.2
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
        action1 = MockAction(
            "EatFood", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.7}]
        )

        # Simulated state after action1: hunger = 0.9 - 0.7 = 0.2, energy = 0.2

        # Action 2: Rest
        # State for utility calc: hunger=0.2, energy=0.2
        # Effects: energy += 0.6. Cost: 0.5
        # Utility2 = ((1.0 - 0.2) * 0.6 * 15) - (0.5 * 10) = (0.8 * 0.6 * 15) - 5.0 = 7.2 - 5.0 = 2.2
        action2 = MockAction(
            "Rest", cost=0.5, effects=[{"attribute": "energy", "change_value": 0.6}]
        )

        plan = [action1, action2]
        # Expected total utility = 11.6 + 2.2 = 13.8
        plan_utility = calculate_plan_utility(char_state, plan, simulate_effects=True)
        self.assertAlmostEqual(
            plan_utility, 13.8
        )  # Note: In this case, simulation didn't change second action's utility because effects were on different attributes.

    def test_calculate_plan_utility_simulation_affects_utility(self):
        char_state = {"hunger": 0.9, "energy": 0.5}
        # Action 1: EatBigMeal
        # Initial state: hunger=0.9
        # Effects: hunger -= 0.8. Cost: 0.2
        # Utility1 = (0.9 * 0.8 * 20) - (0.2 * 10) = 14.4 - 2.0 = 12.4
        action1 = MockAction(
            "EatBigMeal",
            cost=0.2,
            effects=[{"attribute": "hunger", "change_value": -0.8}],
        )

        # Simulated state after action1: hunger = 0.9 - 0.8 = 0.1

        # Action 2: EatSnack
        # State for utility calc: hunger=0.1
        # Effects: hunger -= 0.2. Cost: 0.05
        # Utility2 = (0.1 * 0.2 * 20) - (0.05 * 10) = 0.4 - 0.5 = -0.1
        action2 = MockAction(
            "EatSnack",
            cost=0.05,
            effects=[{"attribute": "hunger", "change_value": -0.2}],
        )

        plan = [action1, action2]
        # Expected total utility = 12.4 + (-0.1) = 12.3
        plan_utility_simulated = calculate_plan_utility(
            char_state, plan, simulate_effects=True
        )
        self.assertAlmostEqual(plan_utility_simulated, 12.3)

        # For comparison, without simulation:
        # Utility1 (no change) = 12.4
        # Utility2 (state is still hunger=0.9) = (0.9 * 0.2 * 20) - (0.05 * 10) = 3.6 - 0.5 = 3.1
        # Expected total utility (no sim) = 12.4 + 3.1 = 15.5
        plan_utility_no_sim = calculate_plan_utility(
            char_state, plan, simulate_effects=False
        )
        self.assertAlmostEqual(plan_utility_no_sim, 15.5)
        self.assertNotAlmostEqual(plan_utility_simulated, plan_utility_no_sim)

    def test_plan_utility_with_goal(self):
        char_state = {"hunger": 0.7, "energy": 0.4}
        goal = Goal(
            "GetFullAndEnergized",
            target_effects={"hunger": -0.7, "energy": 0.6},
            priority=1.0,
        )

        # Action 1: Eat (helps hunger goal)
        # Utility1: hunger_score = 0.7 * 0.5 * 20 = 7.0
        # goal_score (hunger part) = 1.0 * 25.0 = 25.0
        # cost_score = 0.1 * 10 = 1.0
        # Util1 = 7.0 + 25.0 - 1.0 = 31.0
        action1 = MockAction(
            "Eat", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.5}]
        )

        # Simulated state after action1: hunger = 0.7 - 0.5 = 0.2, energy = 0.4

        # Action 2: Rest (helps energy goal)
        # Utility2: energy_score = (1.0 - 0.4) * 0.5 * 15 = 0.6 * 0.5 * 15 = 4.5
        # goal_score (energy part) = 1.0 * 25.0 = 25.0
        # cost_score = 0.2 * 10 = 2.0
        # Util2 = 4.5 + 25.0 - 2.0 = 27.5
        action2 = MockAction(
            "Rest", cost=0.2, effects=[{"attribute": "energy", "change_value": 0.5}]
        )

        plan = [action1, action2]
        # Expected: 31.0 + 27.5 = 58.5
        plan_utility = calculate_plan_utility(
            char_state, plan, current_goal=goal, simulate_effects=True
        )
        self.assertAlmostEqual(plan_utility, 58.5)

    # --- Tests for Enhanced MockAction Features ---

    def test_mock_action_enhanced_attributes(self):
        """Test that MockAction now has enhanced attributes like the real Action class."""
        action = MockAction(
            "TestAction",
            cost=0.5,
            effects=[{"attribute": "hunger", "change_value": -0.3}],
            preconditions=[True],  # Simple boolean precondition for testing
            target="target_obj",
            initiator="initiator_obj",
            priority=0.8,
            related_goal="test_goal"
        )
        
        # Verify all attributes are present
        self.assertEqual(action.name, "TestAction")
        self.assertEqual(action.cost, 0.5)
        self.assertEqual(action.effects, [{"attribute": "hunger", "change_value": -0.3}])
        self.assertEqual(action.preconditions, [True])
        self.assertEqual(action.target, "target_obj")
        self.assertEqual(action.initiator, "initiator_obj")
        self.assertEqual(action.priority, 0.8)
        self.assertEqual(action.related_goal, "test_goal")
        self.assertTrue(hasattr(action, "action_id"))

    def test_mock_action_default_target_is_initiator(self):
        """Test default_target_is_initiator functionality."""
        action = MockAction(
            "SelfAction",
            cost=0.1,
            initiator="self_character",
            default_target_is_initiator=True
        )
        
        # Target should be set to initiator
        self.assertEqual(action.target, "self_character")
        self.assertEqual(action.initiator, "self_character")

    def test_mock_action_preconditions_met(self):
        """Test preconditions_met method."""
        # Action with no preconditions should always be executable
        action1 = MockAction("NoPrereqs", cost=0.1)
        self.assertTrue(action1.preconditions_met())
        
        # Action with True preconditions should be executable
        action2 = MockAction("ValidPrereqs", cost=0.1, preconditions=[True, True])
        self.assertTrue(action2.preconditions_met())
        
        # Action with False precondition should not be executable
        action3 = MockAction("InvalidPrereqs", cost=0.1, preconditions=[True, False])
        self.assertFalse(action3.preconditions_met())

    def test_mock_action_effects_validation(self):
        """Test that MockAction validates effects structure properly."""
        # Valid effects should work
        valid_effects = [
            {"attribute": "hunger", "change_value": -0.5},
            {"attribute": "energy", "change_value": 0.3}
        ]
        action = MockAction("ValidAction", cost=0.1, effects=valid_effects)
        self.assertEqual(action.effects, valid_effects)
        
        # Invalid effects should raise errors
        with self.assertRaises(ValueError):
            MockAction("InvalidEffect1", cost=0.1, effects="not_a_list")
        
        with self.assertRaises(ValueError):
            MockAction("InvalidEffect2", cost=0.1, effects=[{"missing_attribute": "value"}])
        
        with self.assertRaises(ValueError):
            MockAction("InvalidEffect3", cost=0.1, effects=[{"attribute": "hunger"}])  # missing change_value
        
        with self.assertRaises(ValueError):
            MockAction("InvalidEffect4", cost=0.1, effects=[{"attribute": "hunger", "change_value": "not_numeric"}])

    def test_mock_action_add_methods(self):
        """Test add_effect and add_precondition methods."""
        action = MockAction("TestAction", cost=0.1)
        
        # Add a valid effect
        action.add_effect({"attribute": "health", "change_value": 0.2})
        self.assertEqual(len(action.effects), 1)
        self.assertEqual(action.effects[0]["attribute"], "health")
        
        # Add a precondition
        action.add_precondition(True)
        self.assertEqual(len(action.preconditions), 1)
        self.assertTrue(action.preconditions[0])
        
        # Try to add invalid effect
        with self.assertRaises(ValueError):
            action.add_effect({"invalid": "effect"})

    def test_mock_action_priority_defaults(self):
        """Test that priority defaults to 0.5 like real actions."""
        action = MockAction("DefaultPriority", cost=0.1)
        self.assertEqual(action.priority, 0.5)
        
        action_with_priority = MockAction("CustomPriority", cost=0.1, priority=0.9)
        self.assertEqual(action_with_priority.priority, 0.9)

    def test_enhanced_mock_action_utility_calculation_compatibility(self):
        """Test that enhanced MockAction works correctly with utility calculations."""
        char_state = {"hunger": 0.8, "energy": 0.3}
        
        # Create enhanced action with priority and goal
        action = MockAction(
            "EnhancedEat",
            cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.6}],
            priority=0.9,
            target="food_item",
            initiator="character"
        )
        
        goal = Goal("SatisfyHunger", target_effects={"hunger": -0.5}, priority=0.8)
        
        # Should work the same as before
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        expected = 0.8 * 0.6 * 20 + 0.8 * 25.0 - 0.1 * 10  # hunger + goal + cost
        self.assertAlmostEqual(utility, expected)
        
        # Verify action has enhanced attributes
        self.assertEqual(action.priority, 0.9)
        self.assertEqual(action.target, "food_item")
        self.assertEqual(action.initiator, "character")
        self.assertTrue(action.preconditions_met())

    def test_enhanced_mock_goal_with_urgency(self):
        """Test enhanced MockGoal with urgency attribute."""
        # Test goal with urgency
        urgent_goal = Goal("UrgentTask", target_effects={"energy": 0.5}, 
                          priority=0.7, urgency=0.9)
        
        self.assertEqual(urgent_goal.name, "UrgentTask")
        self.assertEqual(urgent_goal.priority, 0.7)
        self.assertEqual(urgent_goal.urgency, 0.9)
        self.assertTrue(hasattr(urgent_goal, "deadline"))
        self.assertTrue(hasattr(urgent_goal, "description"))
        
        # Test that urgency defaults to priority if not specified
        normal_goal = Goal("NormalTask", priority=0.6)
        self.assertEqual(normal_goal.urgency, 0.6)

    def test_utility_evaluator_with_urgent_goal(self):
        """Test that UtilityEvaluator properly handles urgent goals."""
        from tiny_utility_functions import UtilityEvaluator
        
        evaluator = UtilityEvaluator()
        char_state = {"hunger": 0.8, "energy": 0.3, "health": 0.9}
        
        action = MockAction(
            "UrgentAction",
            cost=0.1,
            effects=[{"attribute": "energy", "change_value": 0.5}]
        )
        
        # Create urgent goal (urgency > 0.8 should trigger urgency multiplier)
        urgent_goal = Goal("UrgentEnergyBoost", target_effects={"energy": 0.4}, 
                          priority=0.8, urgency=0.9)
        
        # Calculate utility with urgent goal
        utility_with_urgent = evaluator.evaluate_action_utility(
            "test_char", char_state, action, current_goal=urgent_goal
        )
        
        # Calculate utility with normal goal
        normal_goal = Goal("NormalEnergyBoost", target_effects={"energy": 0.4}, 
                          priority=0.8, urgency=0.5)
        
        utility_with_normal = evaluator.evaluate_action_utility(
            "test_char", char_state, action, current_goal=normal_goal
        )
        
        # Urgent goal should result in higher utility due to urgency multiplier
        self.assertGreater(utility_with_urgent, utility_with_normal)
        print(f"Urgent goal utility: {utility_with_urgent}")
        print(f"Normal goal utility: {utility_with_normal}")

    def test_realistic_action_scenario_with_enhanced_mocks(self):
        """Test a realistic scenario using enhanced MockAction and MockGoal."""
        # Character in critical state
        char_state = {"hunger": 0.9, "energy": 0.1, "health": 0.7, "money": 5}
        
        # Critical survival goal
        survival_goal = Goal(
            "Survive", 
            target_effects={"hunger": -0.8, "energy": 0.7}, 
            priority=1.0, 
            urgency=0.95,
            description="Critical survival needs"
        )
        
        # Actions with realistic attributes
        eat_action = MockAction(
            "EatEmergencyFood",
            cost=0.2,
            effects=[{"attribute": "hunger", "change_value": -0.8}],
            preconditions=[True],  # Has food available
            priority=0.9,
            related_goal=survival_goal
        )
        
        sleep_action = MockAction(
            "TakeNap",
            cost=0.1,
            effects=[{"attribute": "energy", "change_value": 0.6}],
            preconditions=[True],  # Can rest
            priority=0.8
        )
        
        impossible_action = MockAction(
            "ImpossibleTask",
            cost=0.5,
            effects=[{"attribute": "energy", "change_value": 0.3}],
            preconditions=[False],  # Cannot be performed
            priority=0.1
        )
        
        # Test preconditions
        self.assertTrue(eat_action.preconditions_met())
        self.assertTrue(sleep_action.preconditions_met())
        self.assertFalse(impossible_action.preconditions_met())
        
        # Test utility calculations
        eat_utility = calculate_action_utility(char_state, eat_action, survival_goal)
        sleep_utility = calculate_action_utility(char_state, sleep_action, survival_goal)
        
        # Eating should have much higher utility due to critical hunger and goal match
        self.assertGreater(eat_utility, sleep_utility)
        
        # Test with UtilityEvaluator for urgency bonus
        from tiny_utility_functions import UtilityEvaluator
        evaluator = UtilityEvaluator()
        
        advanced_eat_utility = evaluator.evaluate_action_utility(
            "survivor", char_state, eat_action, survival_goal
        )
        
        # Should be higher due to urgency multiplier (urgency 0.95 > 0.8)
        self.assertGreater(advanced_eat_utility, eat_utility)
        
        print(f"Basic eat utility: {eat_utility:.2f}")
        print(f"Advanced eat utility with urgency: {advanced_eat_utility:.2f}")
        print(f"Sleep utility: {sleep_utility:.2f}")
        print(f"Urgency multiplier applied: {advanced_eat_utility > eat_utility}")

    def test_action_priority_in_goal_contexts(self):
        """Test how action priority interacts with goal-based utility."""
        char_state = {"hunger": 0.6, "energy": 0.6}
        
        goal = Goal("Balance", target_effects={"hunger": -0.3, "energy": 0.3}, priority=0.8)
        
        # Same effect, different priorities
        high_priority_action = MockAction(
            "HighPriorityEat", cost=0.1, 
            effects=[{"attribute": "hunger", "change_value": -0.3}],
            priority=0.9
        )
        
        low_priority_action = MockAction(
            "LowPriorityEat", cost=0.1,
            effects=[{"attribute": "hunger", "change_value": -0.3}],
            priority=0.1
        )
        
        # Both should have same utility since priority doesn't directly affect utility calculation
        # (utility calculation focuses on effects, cost, and goal progress)
        high_utility = calculate_action_utility(char_state, high_priority_action, goal)
        low_utility = calculate_action_utility(char_state, low_priority_action, goal)
        
        self.assertAlmostEqual(high_utility, low_utility)
        
        # But verify the actions have different priorities
        self.assertEqual(high_priority_action.priority, 0.9)
        self.assertEqual(low_priority_action.priority, 0.1)


if __name__ == "__main__":
    unittest.main()
# This code is designed to be run as a script, so it will execute the tests when run directly.
# If you want to run this in an interactive environment, you can comment out the last line.
# or run the tests using a test runner like pytest.
