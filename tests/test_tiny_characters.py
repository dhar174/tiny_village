import unittest
from unittest.mock import MagicMock, patch # Added patch for potential future use

from tiny_characters import CreateCharacter, Character, SimpleSocialGoal # Added SimpleSocialGoal
import tiny_buildings as tb
from tiny_items import ItemInventory # For Character instantiation
from tiny_locations import Location # For Character instantiation
from actions import ActionSystem # For Character instantiation (if its __init__ is not too complex)
from tiny_time_manager import GameTimeManager # For Character instantiation
# Note: GraphManager and MemoryManager in Character are now more robust to import errors


class TestSimpleSocialGoal(unittest.TestCase):
    def test_goal_instantiation(self):
        goal = SimpleSocialGoal(
            name="Make Friend",
            description="Befriend a new person.",
            completion_conditions={"relationship_with_target": {">": 10}},
            priority=0.7,
            reward={"happiness_boost": 0.2},
            target_character_id="npc123"
        )
        self.assertEqual(goal.name, "Make Friend")
        self.assertEqual(goal.description, "Befriend a new person.")
        self.assertEqual(goal.completion_conditions, {"relationship_with_target": {">": 10}})
        self.assertEqual(goal.priority, 0.7)
        self.assertEqual(goal.reward, {"happiness_boost": 0.2})
        self.assertEqual(goal.target_character_id, "npc123")
        self.assertFalse(goal.achieved)
        self.assertTrue(goal.active)

    def test_set_achieved(self):
        goal = SimpleSocialGoal("Test Goal", "", {}, 0.5, {})
        goal.set_achieved(True)
        self.assertTrue(goal.achieved)
        self.assertFalse(goal.active)
        goal.set_achieved(False) # Should allow resetting if needed by game logic
        self.assertFalse(goal.achieved)
        self.assertTrue(goal.active) # Should reactivate if no longer achieved

    def test_set_active(self):
        goal = SimpleSocialGoal("Test Goal", "", {}, 0.5, {})
        goal.set_active(False)
        self.assertFalse(goal.active)
        goal.set_active(True)
        self.assertTrue(goal.active)

class TestCharacterGoalIntegration(unittest.TestCase):
    def setUp(self):
        # Mock dependencies for Character that might still cause issues or are complex
        # The try-except blocks in Character.__init__ should handle None for these if imports fail
        self.mock_graph_manager = None
        from tiny_time_manager import GameCalendar # Import GameCalendar
        self.mock_calendar = GameCalendar() # Create a calendar instance
        self.mock_gametime_manager = GameTimeManager(calendar=self.mock_calendar) # Pass calendar
        
        # ActionSystem from actions.py might be okay if its __init__ is simple
        # and its methods are not called deeply during Character init.
        # The modified actions.py should allow this.
        self.mock_action_system = ActionSystem() 

        # Create a default character for tests
        # This relies on the robust __init__ in Character class
        try:
            self.character = Character(
                name="TestSubject",
                age=30,
                graph_manager=self.mock_graph_manager,
                gametime_manager=self.mock_gametime_manager,
                action_system=self.mock_action_system,
                # Provide other minimal required args if any, or ensure defaults are safe
                inventory=ItemInventory(),
                location=Location("TestLocation",0,0,0,0, self.mock_action_system)
            )
        except Exception as e:
            # This will help diagnose if Character instantiation is still failing
            print(f"ERROR DURING CHARACTER SETUP FOR TESTS: {e}")
            # Potentially re-raise or self.fail() if Character must be created
            # For now, allow tests to proceed and fail if self.character is None
            self.character = None 


    def test_character_instantiation_for_goals(self):
        # This test primarily checks if the setUp's Character instantiation worked,
        # especially with the MemoryManager workaround.
        self.assertIsNotNone(self.character, "Character object could not be instantiated. Check __init__ and its dependencies.")
        if self.character: # Proceed only if character was created
            self.assertTrue(hasattr(self.character, 'social_goals'))
            self.assertEqual(self.character.social_goals, [])

    def test_add_simple_social_goal(self):
        if not self.character:
            self.skipTest("Character object not instantiated, skipping goal integration test.")
            
        goal_desc = "Make a new friend (NPC001)"
        goal_conditions = {"relationship_status_NPC001": {">": 10}}
        goal_reward = {"happiness": 0.5}
        social_goal = SimpleSocialGoal(
            name="Befriend NPC001",
            description=goal_desc,
            completion_conditions=goal_conditions,
            priority=0.8,
            reward=goal_reward,
            target_character_id="NPC001_ID"
        )
        self.character.add_simple_social_goal(social_goal)
        self.assertIn(social_goal, self.character.social_goals)
        self.assertEqual(len(self.character.social_goals), 1)
        self.assertEqual(self.character.social_goals[0].name, "Befriend NPC001")

class TestCreateCharacter(unittest.TestCase): # Original test class
    def setUp(self):
        self.character = None
        # For CreateCharacter tests, we need a more functional GraphManager mock if its methods are called
        # However, CreateCharacter itself instantiates GraphManager if not provided.
        # The try-except in Character.__init__ is key.

    # def test_create_new_character_manual(self):
    #     # This test is commented out in the original, keeping it that way.
    #     pass

    def test_create_new_character_auto(self):
        # Test creating a character automatically
        # This will internally try to create GraphManager, GameTimeManager, ActionSystem
        # The robustness added to Character.__init__ should handle potential failures.
        try:
            creator = CreateCharacter()
            # Pass None for managers that might cause issues to test Character's internal fallbacks
            self.character = creator.create_new_character(
                mode="auto",
                graph_manager_instance=None, # Test fallback
                gametime_manager_instance=GameTimeManager(calendar=GameCalendar()), # Assumed safe, needs calendar
                action_system_instance=ActionSystem() # Assumed safe with modified actions.py
            )
        except Exception as e:
            self.fail(f"CreateCharacter().create_new_character(mode='auto') raised an exception: {e}")

        self.assertIsNotNone(self.character)
        if self.character:
            self.assertNotEqual(self.character.name, "John Doe")
            # print(f"Auto-created character: {self.character.name}")
            # for key, val in self.character.to_dict().items():
            #     print(key, val)
            # print("\n")
            # if self.character.motives:
            #    for motive_obj in self.character.get_motives(): # Iterate through Motive objects
            #        print(motive_obj.name, motive_obj.score)


if __name__ == "__main__":
    unittest.main()
