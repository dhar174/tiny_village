import unittest

from actions import Action, ActionSystem
from demo_character_factory import Character as DemoCharacter
from tiny_locations import Location


class RecordingGraphManager:
    def __init__(self, characters=None):
        self.characters = characters if characters is not None else {}
        self.updated_nodes = []

    def update_node_attribute(self, node_id, attribute_name, value):
        self.updated_nodes.append((node_id, attribute_name, value))


class ExplodingCharacterRegistry:
    def values(self):
        raise AssertionError(
            "graph-wide character lookup should not run when current_visitors are available"
        )


class TestCharacter(DemoCharacter):
    def __init__(self, name, location, social_wellbeing):
        super().__init__(name=name, social_wellbeing=social_wellbeing)
        self.uuid = f"{name.lower()}-uuid"
        self.location = location


class TestActionNamedTargetResolution(unittest.TestCase):
    def test_execute_uses_current_visitors_for_target_character_in_location(self):
        graph_manager = RecordingGraphManager(characters=ExplodingCharacterRegistry())
        action_system = ActionSystem(graph_manager=graph_manager)

        town_square = Location("Town Square", 0, 0, 10, 10, action_system)
        elsewhere = Location("Elsewhere", 20, 20, 10, 10, action_system)

        initiator = TestCharacter("Initiator", town_square, social_wellbeing=5)
        nearby_target = TestCharacter("Target", town_square, social_wellbeing=3)
        other_character = TestCharacter("Other", elsewhere, social_wellbeing=9)

        town_square.current_visitors.extend([initiator, nearby_target])

        action = Action(
            name="Encourage",
            preconditions=[],
            effects=[
                {
                    "targets": ["target_character_in_location"],
                    "attribute": "social_wellbeing",
                    "change_value": 2,
                }
            ],
            initiator=initiator,
            target=nearby_target.name,
            graph_manager=graph_manager,
        )

        result = action.execute()

        self.assertTrue(result)
        self.assertEqual(nearby_target.social_wellbeing, 5)
        self.assertEqual(other_character.social_wellbeing, 9)
        self.assertEqual(
            graph_manager.updated_nodes,
            [(nearby_target.uuid, "social_wellbeing", 5)],
        )

    def test_execute_falls_back_to_registered_characters_when_visitors_missing(self):
        graph_manager = RecordingGraphManager()
        action_system = ActionSystem(graph_manager=graph_manager)

        town_square = Location("Town Square", 0, 0, 10, 10, action_system)
        elsewhere = Location("Elsewhere", 20, 20, 10, 10, action_system)

        initiator = TestCharacter("Initiator", town_square, social_wellbeing=5)
        nearby_target = TestCharacter("Target", town_square, social_wellbeing=3)
        other_character = TestCharacter("Other", elsewhere, social_wellbeing=9)

        graph_manager.characters = {
            character.uuid: character
            for character in (initiator, nearby_target, other_character)
        }

        action = Action(
            name="Encourage",
            preconditions=[],
            effects=[
                {
                    "targets": ["target_character_in_location"],
                    "attribute": "social_wellbeing",
                    "change_value": 2,
                }
            ],
            initiator=initiator,
            target=nearby_target.uuid,
            graph_manager=graph_manager,
        )

        result = action.execute()

        self.assertTrue(result)
        self.assertEqual(nearby_target.social_wellbeing, 5)
        self.assertEqual(other_character.social_wellbeing, 9)
        self.assertEqual(
            graph_manager.updated_nodes,
            [(nearby_target.uuid, "social_wellbeing", 5)],
        )


if __name__ == "__main__":
    unittest.main()
