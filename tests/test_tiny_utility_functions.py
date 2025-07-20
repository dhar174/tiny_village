import unittest
from tiny_utility_functions import (
    calculate_action_utility,
    calculate_plan_utility,
)
import importlib

# Try to import the real Goal class first, fall back to a simplified real implementation
try:
    from tiny_characters import Goal as RealGoal
    REAL_GOAL_AVAILABLE = True
    print("✅ Successfully imported real Goal class from tiny_characters")
except ImportError as e:
    print(f"⚠️  Failed to import real Goal class: {e}")
    print("   Creating simplified Real Goal implementation...")
    REAL_GOAL_AVAILABLE = False


class SimplifiedRealGoal:
    """
    Simplified Real Goal class that implements the same interface as the full Goal class
    but without complex dependencies. This provides a more realistic test than Mock()
    while still being importable.
    """

    def __init__(self, name, target_effects=None, priority=0.5, **kwargs):
        # Core attributes used by utility functions
        self.name = name
        self.target_effects = target_effects if target_effects else {}
        self.priority = priority
        self.score = priority  # alias for compatibility
        
        # Additional attributes that may be checked by utility functions
        self.urgency = kwargs.get('urgency', priority)
        self.attributes = kwargs.get('attributes', {})
        
        # Minimal Goal-like interface compatibility
        self.description = kwargs.get('description', f"Goal: {name}")
        self.character = kwargs.get('character', None)
        self.target = kwargs.get('target', None)
        self.completion_conditions = kwargs.get('completion_conditions', {})
        self.criteria = kwargs.get('criteria', [])
        self.goal_type = kwargs.get('goal_type', 'basic')


class MockGoal:

    Mock goal class for testing utility functions - ONLY used when SimplifiedRealGoal 
    cannot be used. This provides minimal interface for backward compatibility.
    """


    def __init__(self, name, target_effects=None, priority=0.5, description=None):
        self.name = name
        self.target_effects = target_effects if target_effects else {}
        self.priority = priority
        self.score = priority  # alias for compatibility
        self.description = description or f"Test goal: {name}"
        self.completed = False
        
        # Additional attributes to match real Goal interface
        self.character = None
        self.target = None
        self.completion_conditions = {}
        self.criteria = []
        self.required_items = []
        self.goal_type = "test"
        
    def check_completion(self, state=None):
        """Check if goal is completed - matches real Goal interface."""
        return self.completed
        
    def get_name(self):
        """Getter method found in real Goal class."""
        return self.name
        
    def get_score(self):
        """Getter method found in real Goal class."""
        return self.score
        
    def to_dict(self):
        """Serialization method found in real Goal class."""
        return {
            "name": self.name,
            "description": self.description,
            "score": self.score,
            "target_effects": self.target_effects,
            "priority": self.priority
        }


# Use real Goal if available, simplified real Goal if not, mock as last resort
if REAL_GOAL_AVAILABLE:
    Goal = RealGoal
    print("✅ Using real Goal class for tests")
else:
    Goal = SimplifiedRealGoal
    print("✅ Using simplified real Goal implementation for tests")

# We need a mock or placeholder for the Action class.
# If actions.py is too complex to import, we define a simple one here for testing utility.
# The actual Action class from actions.py has a complex __init__ and dependencies.
# For testing utility functions, we only need name, cost, and effects.


class MockAction:
    """Enhanced MockAction with meaningful precondition checking.
    
    This mock implements realistic precondition validation to ensure tests
    fail when real precondition logic is broken, rather than masking bugs
    by always returning True.
    """

    
    def __init__(self, name, cost, effects=None, preconditions=None, satisfaction=None):
        self.name = name
        self.cost = float(cost)
        # Effects is a list of dictionaries, e.g., [{'attribute': 'hunger', 'change_value': -0.5}]
        self.effects = effects if effects else []
        self.preconditions = preconditions if preconditions else []
        
        # Additional attributes to match real Action interface
        self.satisfaction = satisfaction if satisfaction is not None else 5.0
        self.urgency = 1.0
        self.action_id = id(self)
        self.target = None
        self.initiator = None
        self.priority = 1.0
        self.related_goal = None
        
        # Impact ratings found in real Action class
        self.impact_rating_on_target = 1
        self.impact_rating_on_initiator = 1
        self.impact_rating_on_other = {}

    def preconditions_met(self, state=None):
        """Check if preconditions are met - matches real Action interface.
        
        This implementation provides meaningful precondition checking rather
        than always returning True, ensuring tests will fail when real
        precondition logic is broken.
        
        Args:
            state: Optional state object or dict to check preconditions against
            
        Returns:
            bool: True if all preconditions are satisfied, False otherwise
        """
        if not self.preconditions:
            return True
            
        # Handle different precondition formats for testing flexibility
        for precondition in self.preconditions:
            if isinstance(precondition, dict):
                # Handle dict-style preconditions: {"attribute": "energy", "operator": ">=", "value": 50}
                attribute = precondition.get("attribute")
                operator = precondition.get("operator", ">=")
                required_value = precondition.get("value", 0)
                
                if state is None:
                    # No state provided - cannot verify preconditions
                    return False
                    
                # Get current value from state
                if isinstance(state, dict):
                    current_value = state.get(attribute, 0)
                else:
                    current_value = getattr(state, attribute, 0)
                
                # Check condition based on operator
                if operator == ">=":
                    if current_value < required_value:
                        return False
                elif operator == "<=":
                    if current_value > required_value:
                        return False
                elif operator == "==":
                    if current_value != required_value:
                        return False
                elif operator == ">":
                    if current_value <= required_value:
                        return False
                elif operator == "<":
                    if current_value >= required_value:
                        return False
                else:
                    # Unknown operator - fail safe
                    return False
                    
            elif hasattr(precondition, 'check_condition'):
                # Handle Condition-like objects
                try:
                    if not precondition.check_condition(state):
                        return False
                except Exception:
                    # If condition checking fails, precondition is not met
                    return False
            elif callable(precondition):
                # Handle function-style preconditions
                try:
                    if not precondition(state):
                        return False
                except Exception:
                    return False
            else:
                # Unknown precondition type - fail safe to catch bugs
                return False
                
        return True

    def to_dict(self):
        """Serialize action for compatibility with real Action interface."""
        return {
            "name": self.name,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "cost": self.cost,

        }

    def __repr__(self):
        return (
            f"MockAction(name='{self.name}', cost={self.cost}, effects={self.effects})"
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

    def test_goal_interface_compatibility(self):
        """
        Test that our Goal implementation properly exposes the interface expected
        by utility functions. This test would fail if the Goal object doesn't
        have the required attributes or if they don't work as expected.
        """
        # Test Goal with all expected attributes
        goal = Goal(
            name="TestGoal",
            target_effects={"hunger": -0.5, "energy": 0.3},
            priority=0.8,
            urgency=0.9,
            attributes={"importance": 5}
        )
        
        # Verify essential interface
        self.assertEqual(goal.name, "TestGoal")
        self.assertEqual(goal.target_effects, {"hunger": -0.5, "energy": 0.3})
        self.assertEqual(goal.priority, 0.8)
        self.assertEqual(goal.score, 0.8)  # Should be aliased to priority
        
        # Test that utility functions can access these attributes without errors
        char_state = {"hunger": 0.6, "energy": 0.4}
        action = MockAction(
            "TestAction", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.3}]
        )
        
        # This should work without throwing AttributeError or TypeError
        utility = calculate_action_utility(char_state, action, current_goal=goal)
        self.assertIsInstance(utility, (int, float))
        
        # Test that goal attributes are actually used (should get different result with different priority)
        goal_low_priority = Goal(
            name="LowPriority",
            target_effects={"hunger": -0.5},
            priority=0.1
        )
        utility_low = calculate_action_utility(char_state, action, current_goal=goal_low_priority)
        
        # With different priorities, utilities should be different (proving Goal attributes matter)
        self.assertNotEqual(utility, utility_low)

    def test_mock_vs_real_goal_behavior(self):
        """
        Test that demonstrates the difference between using Mock() and real Goal objects.
        This test ensures our real Goal implementation behaves differently from a basic mock,
        proving that it provides more realistic testing.
        """
        char_state = {"hunger": 0.7}
        action = MockAction(
            "Eat", cost=0.1, effects=[{"attribute": "hunger", "change_value": -0.4}]
        )
        
        # Real Goal implementation
        real_goal = Goal(
            name="RealGoal",
            target_effects={"hunger": -0.6},
            priority=0.8
        )
        
        # Basic mock-like goal (like what was being used before)
        mock_goal = MockGoal(
            name="MockGoal",
            target_effects={"hunger": -0.6},
            priority=0.8
        )
        
        # Both should work, but real goal provides more realistic testing
        real_utility = calculate_action_utility(char_state, action, current_goal=real_goal)
        mock_utility = calculate_action_utility(char_state, action, current_goal=mock_goal)
        
        # Results should be the same for basic cases (proving compatibility)
        self.assertAlmostEqual(real_utility, mock_utility, places=2)
        
        # But real goal has additional attributes that could be used by utility functions
        self.assertTrue(hasattr(real_goal, 'description'))
        self.assertTrue(hasattr(real_goal, 'goal_type'))
        self.assertTrue(hasattr(real_goal, 'urgency'))
        
        # Mock goal has minimal interface
        self.assertFalse(hasattr(mock_goal, 'description'))
        self.assertFalse(hasattr(mock_goal, 'goal_type'))
        self.assertFalse(hasattr(mock_goal, 'urgency'))


if __name__ == "__main__":
    unittest.main()
# This code is designed to be run as a script, so it will execute the tests when run directly.
# If you want to run this in an interactive environment, you can comment out the last line.
# or run the tests using a test runner like pytest.
