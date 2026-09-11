import unittest
from unittest.mock import MagicMock

from tiny_strategy_manager import StrategyManager
# Assuming Character, Action, Goal, ItemObject, FoodItem, Location can be simplified or mocked
# For these tests, we'll primarily mock Character

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

    def test_get_daily_actions_hungry_with_food(self):
        self.character.hunger_level = 8.0 # High hunger
        self.character.add_food_item(name="Apple", calories=50)

        actions = self.strategy_manager.get_daily_actions(self.character)

        self.assertTrue(len(actions) > 0)
        self.assertIn("Eat Apple", actions[0].name) # Eat Apple should be top action

        # Check if EatAction was generated
        found_eat = any("Eat Apple" in action.name for action in actions)
        self.assertTrue(found_eat, "EatAction for Apple should be generated")

    def test_get_daily_actions_tired_at_home(self):
        self.character.energy = 2.0 # Low energy (assuming 0-10 scale)
        self.character.location.name = "Home"

        actions = self.strategy_manager.get_daily_actions(self.character)
        self.assertTrue(len(actions) > 0)
        self.assertIn("Sleep", actions[0].name)

        found_sleep = any("Sleep" in action.name for action in actions)
        self.assertTrue(found_sleep, "SleepAction should be generated when tired at home")

    def test_get_daily_actions_has_job(self):
        self.character.job = MockJob("Programmer")

        actions = self.strategy_manager.get_daily_actions(self.character)
        self.assertTrue(len(actions) > 0)
        self.assertIn("Work as Programmer", actions[0].name)

        found_work = any("Work as Programmer" in action.name for action in actions)
        self.assertTrue(found_work, "WorkAction should be generated when character has a job")

    def test_get_daily_actions_sorting(self):
        # Ensure actions are sorted by utility
        self.character.hunger_level = 7.0
        self.character.add_food_item("Pear", 30)
        self.character.energy = 2.0
        self.character.location.name = "Home"
        self.character.job = MockJob("Gardener")

        actions = self.strategy_manager.get_daily_actions(self.character)

        self.assertTrue(len(actions) >= 4) # Eat, Sleep, Work, Wander, NoOp
        action_names_sorted = [a.name for a in actions]

        expected_order = ["Eat Pear", "Sleep", "Work as Gardener", "NoOp", "Wander"]
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

        self.assertTrue(indices["Eat Pear"] < indices["Sleep"])
        self.assertTrue(indices["Sleep"] < indices["Work as Gardener"])
        self.assertTrue(indices["Work as Gardener"] < indices["NoOp"])
        self.assertTrue(indices["NoOp"] < indices["Wander"])


    def test_get_daily_actions_no_specific_needs(self):
        # Character is not particularly hungry or tired, no job.
        self.character.hunger_level = 2.0
        self.character.energy = 8.0
        self.character.job = "unemployed"
        self.character.location.name = "Park" # Not home, so Sleep not prioritized

        actions = self.strategy_manager.get_daily_actions(self.character)
        self.assertTrue(len(actions) > 0)
        # Expect NoOp to be highest due to zero cost; others should be low utility
        action_names = [a.name for a in actions]
        self.assertNotIn("Sleep", action_names) # Should not generate sleep if not at home / not tired
        
        # Check that Wander and NoOp are present
        self.assertTrue(any("Wander" in name for name in action_names))
        self.assertTrue(any("NoOp" in name for name in action_names))
        
        if actions: # Ensure there's at least one action to check
            self.assertEqual(actions[0].name, "NoOp") # Utility-based sort should prefer low-cost default

if __name__ == '__main__':
    unittest.main()
