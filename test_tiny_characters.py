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

# New Test Class for Goal
from tiny_characters import Goal # Ensure Goal is imported

class TestGoal(unittest.TestCase):
    def setUp(self):
        # Basic setup that might be common for several tests
        # Mock dependencies that Goal's constructor might need
        self.mock_character = Mock(spec=Character)
        self.mock_character.name = "TestCharacter"
        # If Character has complex attributes accessed in Goal, mock them too
        # For example, if goal accesses character.inventory.get_item_count:
        # self.mock_character.inventory = Mock()
        # self.mock_character.inventory.get_item_count = Mock(return_value=0)

        # It's good practice to mock specific classes if they are defined,
        # otherwise, a generic Mock will do for simple attribute/method access.
        # from tiny_graph_manager import GraphManager # Assuming these exist
        # from tiny_action_system import ActionSystem
        # from tiny_prompt_builder import PromptBuilder

        self.mock_graph_manager = Mock() # Mock(spec=GraphManager)
        self.mock_action_system = Mock() # Mock(spec=ActionSystem)
        # self.mock_gametime_manager is already set up in TestCharacterGoalIntegration,
        # but for TestGoal, we might need its own or ensure Goal doesn't need it deeply.
        # For now, assume Goal can take a generic mock for gametime_manager if it's a direct param.
        self.mock_gametime_manager = Mock() # Mock(spec=GameTimeManager)
        self.mock_prompt_builder = Mock() # Mock(spec=PromptBuilder)

        # If Goal.__init__ expects these on the character object, ensure they are there.
        # self.mock_character.graph_manager = self.mock_graph_manager
        # self.mock_character.action_system = self.mock_action_system
        # self.mock_character.gametime_manager = self.mock_gametime_manager
        # self.mock_character.prompt_builder = self.mock_prompt_builder


    def test_goal_creation_successful(self):
        """Test basic successful creation of a Goal object."""
        goal_name = "Test Goal"
        goal_description = "This is a test goal."
        goal_score = 0.75
        completion_conditions = [
            {"condition_type": "attribute", "attribute": "energy", "operator": ">=", "value": 50}
        ]
        criteria = [
            {"criterion_type": "item_present", "item_name": "key", "value": True}
        ]
        priority = 5
        deadline = "2024-12-31"

        goal = Goal(
            character=self.mock_character, # Passed directly
            name=goal_name,
            description=goal_description,
            completion_conditions=completion_conditions,
            criteria=criteria,
            score=goal_score,
            priority=priority,
            deadline=deadline,
            graph_manager=self.mock_graph_manager, # Passed directly
            action_system=self.mock_action_system, # Passed directly
            gametime_manager=self.mock_gametime_manager, # Passed directly
            prompt_builder=self.mock_prompt_builder # Passed directly
        )

        self.assertIsNotNone(goal, "Goal object should be created.")
        self.assertEqual(goal.name, goal_name, "Goal name not set correctly.")
        self.assertEqual(goal.description, goal_description, "Goal description not set correctly.")
        self.assertEqual(goal.score, goal_score, "Goal score not set correctly.")
        self.assertEqual(goal.priority, priority, "Goal priority not set correctly.")
        self.assertEqual(goal.deadline, deadline, "Goal deadline not set correctly.")
        self.assertEqual(goal.completion_conditions, completion_conditions, "Completion conditions not set correctly.")
        self.assertEqual(goal.criteria, criteria, "Criteria not set correctly.")
        self.assertIsInstance(goal.required_items, list, "Required items should be a list.")
        self.assertIsInstance(goal.target_effects, dict, "Target effects should be a dict.")
        self.assertIsInstance(goal.tasks, list, "Tasks should be a list.")

    def test_extract_required_items_basic(self):
        """Test basic functionality of extract_required_items."""
        completion_conditions = [
            {"condition_type": "item_check", "item_type": "food", "quantity": 1, "source": "inventory.check_has_item_by_type(['food'])"}
        ]
        criteria = [
            # Criteria processing for items is not explicitly shown in Goal.extract_required_items
            # It mainly processes 'item_check' from completion_conditions.
            # {"criterion_type": "item_present", "item_name": "key", "value": True}
        ]

        goal = Goal(
            character=self.mock_character,
            name="Food Goal",
            description="Need to find food.",
            completion_conditions=completion_conditions,
            criteria=criteria,
            score=0.8,
            priority=7,
            deadline="2024-01-01",
            graph_manager=self.mock_graph_manager,
            action_system=self.mock_action_system,
            gametime_manager=self.mock_gametime_manager,
            prompt_builder=self.mock_prompt_builder
        )

        # extract_required_items is called in __init__.
        # We expect it to find 'food' from completion_conditions.
        # Expected format from condition: ({'item_type': 'food', 'quantity': 1}, True)

        found_food = False
        for item_dict, is_essential in goal.required_items:
            if item_dict.get('item_type') == 'food' and item_dict.get('quantity') == 1:
                found_food = True
                self.assertTrue(is_essential, "Food item from item_check should be essential.")

        self.assertTrue(found_food, "Required item 'food' not extracted correctly from completion_conditions.")


if __name__ == "__main__":
    unittest.main()
