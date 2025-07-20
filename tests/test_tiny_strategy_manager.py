import unittest
from unittest.mock import MagicMock, patch

from tiny_strategy_manager import StrategyManager
# Assuming Character, Action, Goal, ItemObject, FoodItem, Location can be simplified or mocked
# For these tests, we'll primarily mock Character and calculate_action_utility

# Minimal placeholder for Character and related classes if not easily importable/mockable
# These are simplified to support attributes accessed by get_daily_actions
class MockCharacter:
    def __init__(self, name="TestChar"):
        self.name = name
        self.hunger_level = 5.0 # Scale of 0-10, where 0 is full
        self.energy = 5.0       # Scale of 0-10, where 10 is full
        self.wealth_money = 50.0
        self.social_wellbeing = 5.0
        self.mental_health = 5.0
        self.inventory = MagicMock()
        self.location = MagicMock()
        self.job = "unemployed" # Can be a string or a mock object

        # Mocking get_food_items behavior
        self.mock_food_items = []
        self.inventory.get_food_items = MagicMock(return_value=self.mock_food_items)
    
    def add_food_item(self, name, calories):
        food_item = MagicMock()
        food_item.name = name
        food_item.calories = calories
        self.mock_food_items.append(food_item)

class MockLocation:
    def __init__(self, name="NeutralPlace"):
        self.name = name

class MockJob:
    def __init__(self, job_title="Worker"):
        self.job_title = job_title


# The Action classes (EatAction, SleepAction, etc.) are defined in tiny_strategy_manager.py
# and inherit from actions.Action. We will rely on those definitions.
# Goal is also defined in tiny_utility_functions and imported by strategy_manager.

class TestStrategyManager(unittest.TestCase):

    def setUp(self):
        self.strategy_manager = StrategyManager()
        self.character = MockCharacter("TestCharacter")
        self.character.location = MockLocation("Home") # Default location

    @patch('tiny_strategy_manager.calculate_action_utility')
    def test_get_daily_actions_hungry_with_food(self, mock_calculate_utility):
        self.character.hunger_level = 8.0 # High hunger
        self.character.add_food_item(name="Apple", calories=50)
        
        # Define return values for calculate_action_utility
        # Higher utility for EatApple when hungry
        def utility_side_effect(char_state, action, current_goal):
            if "Eat Apple" in action.name:
                return 100.0
            elif "Sleep" in action.name:
                return 50.0
            elif "Work" in action.name:
                return 30.0
            else: # NoOp, Wander
                return 10.0
        mock_calculate_utility.side_effect = utility_side_effect

        actions = self.strategy_manager.get_daily_actions(self.character)
        
        self.assertTrue(len(actions) > 0)
        self.assertIn("Eat Apple", actions[0].name) # Eat Apple should be top action
        
        # Check if EatAction was generated
        found_eat = any("Eat Apple" in action.name for action in actions)
        self.assertTrue(found_eat, "EatAction for Apple should be generated")

    @patch('tiny_strategy_manager.calculate_action_utility')
    def test_get_daily_actions_tired_at_home(self, mock_calculate_utility):
        self.character.energy = 2.0 # Low energy (assuming 0-10 scale)
        self.character.location.name = "Home"

        def utility_side_effect(char_state, action, current_goal):
            if "Sleep" in action.name:
                return 100.0
            return 10.0 # Other actions
        mock_calculate_utility.side_effect = utility_side_effect

        actions = self.strategy_manager.get_daily_actions(self.character)
        self.assertTrue(len(actions) > 0)
        self.assertIn("Sleep", actions[0].name)

        found_sleep = any("Sleep" in action.name for action in actions)
        self.assertTrue(found_sleep, "SleepAction should be generated when tired at home")

    @patch('tiny_strategy_manager.calculate_action_utility')
    def test_get_daily_actions_has_job(self, mock_calculate_utility):
        self.character.job = MockJob("Programmer")

        def utility_side_effect(char_state, action, current_goal):
            if "Work as Programmer" in action.name:
                return 100.0
            return 10.0
        mock_calculate_utility.side_effect = utility_side_effect

        actions = self.strategy_manager.get_daily_actions(self.character)
        self.assertTrue(len(actions) > 0)
        self.assertIn("Work as Programmer", actions[0].name)
        
        found_work = any("Work as Programmer" in action.name for action in actions)
        self.assertTrue(found_work, "WorkAction should be generated when character has a job")

    @patch('tiny_strategy_manager.calculate_action_utility')
    def test_get_daily_actions_sorting(self, mock_calculate_utility):
        # Ensure actions are sorted by utility
        self.character.hunger_level = 7.0
        self.character.add_food_item("Pear", 30)
        self.character.energy = 2.0
        self.character.location.name = "Home"
        self.character.job = MockJob("Gardener")

        utility_values = {
            "Eat Pear": 80.0,
            "Sleep": 100.0,
            "Work as Gardener": 60.0,
            "Wander": 5.0,
            "NoOp": 0.0
        }
        def utility_side_effect(char_state, action, current_goal):
            for name_key, util_val in utility_values.items():
                if name_key in action.name:
                    return util_val
            return -1.0 # Should not happen if all actions covered
        mock_calculate_utility.side_effect = utility_side_effect

        actions = self.strategy_manager.get_daily_actions(self.character)
        
        self.assertTrue(len(actions) >= 4) # Eat, Sleep, Work, Wander, NoOp
        action_names_sorted = [a.name for a in actions]
        
        expected_order = ["Sleep", "Eat Pear", "Work as Gardener", "Wander", "NoOp"]
        # Allow for other generic actions if any, but these should be in this relative order
        
        # Check that the top actions appear in the expected order based on utility
        # This is a bit more robust than checking exact list equality if other actions get added
        # Find indices of our key actions
        indices = {}
        for name in expected_order:
            try:
                indices[name] = next(i for i, act in enumerate(actions) if name in act.name)
            except StopIteration:
                self.fail(f"Action containing '{name}' not found in results: {action_names_sorted}")

        self.assertTrue(indices["Sleep"] < indices["Eat Pear"])
        self.assertTrue(indices["Eat Pear"] < indices["Work as Gardener"])
        self.assertTrue(indices["Work as Gardener"] < indices["Wander"])
        self.assertTrue(indices["Wander"] < indices["NoOp"])


    @patch('tiny_strategy_manager.calculate_action_utility')
    def test_get_daily_actions_no_specific_needs(self, mock_calculate_utility):
        # Character is not particularly hungry or tired, no job.
        self.character.hunger_level = 2.0
        self.character.energy = 8.0
        self.character.job = "unemployed"
        self.character.location.name = "Park" # Not home, so Sleep not prioritized

        def utility_side_effect(char_state, action, current_goal):
            if "Wander" in action.name: return 20.0
            if "NoOp" in action.name: return 10.0
            return 5.0 # Other actions get low utility
        mock_calculate_utility.side_effect = utility_side_effect
        
        actions = self.strategy_manager.get_daily_actions(self.character)
        self.assertTrue(len(actions) > 0)
        # Expect Wander or NoOp to be high, Eat/Sleep/Work should not be generated or have low utility
        action_names = [a.name for a in actions]
        self.assertNotIn("Sleep", action_names) # Should not generate sleep if not at home / not tired
        
        # Check that Wander and NoOp are present
        self.assertTrue(any("Wander" in name for name in action_names))
        self.assertTrue(any("NoOp" in name for name in action_names))
        
        if actions: # Ensure there's at least one action to check
            self.assertIn(actions[0].name, ["Wander", "NoOp"]) # Or whichever has higher utility based on mock

if __name__ == '__main__':
    unittest.main()
