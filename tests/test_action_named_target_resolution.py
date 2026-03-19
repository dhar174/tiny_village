"""
Regression tests for the target_character_in_location named-target token.

These tests verify that Action.execute() correctly resolves the
``target_character_in_location`` token: the first character found in the
same location as the initiator whose identity matches the action's requested
target is selected, and the declared effects are applied to that resolved
character rather than to someone else.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from actions import Action, ActionSystem
from demo_character_factory import Character as DemoCharacter
from tiny_locations import Location


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingGraphManager:
    """Graph manager that records update_node_attribute calls for assertions."""

    def __init__(self, characters=None):
        self.characters = characters if characters is not None else {}
        self.updated_nodes = []

    def update_node_attribute(self, node_id, attribute_name, value):
        self.updated_nodes.append((node_id, attribute_name, value))


class ExplodingCharacterRegistry:
    """Raises if iterated — used to assert the graph-wide scan is NOT triggered."""

    def values(self):
        raise AssertionError(
            "graph-wide character lookup should not run when current_visitors are available"
        )


class TestCharacter(DemoCharacter):
    """DemoCharacter with an explicit location and UUID for test fixtures."""

    def __init__(self, name, location, social_wellbeing=0):
        super().__init__(name=name, social_wellbeing=social_wellbeing)
        self.uuid = f"{name.lower()}-uuid"
        self.location = location


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestActionNamedTargetResolution(unittest.TestCase):
    """Action.execute() correctly resolves target_character_in_location."""

    # ------------------------------------------------------------------
    # current_visitors path
    # ------------------------------------------------------------------

    def test_execute_uses_current_visitors_for_target_character_in_location(self):
        """Effect is applied to the co-located character via current_visitors."""
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

    # ------------------------------------------------------------------
    # graph_manager.characters fallback path
    # ------------------------------------------------------------------

    def test_execute_falls_back_to_registered_characters_when_visitors_missing(self):
        """Falls back to graph_manager.characters when current_visitors is empty."""
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

    # ------------------------------------------------------------------
    # No match: initiator is alone
    # ------------------------------------------------------------------

    def test_no_effect_when_no_other_character_in_location(self):
        """No effect applied when the initiator is alone in the location."""
        graph_manager = RecordingGraphManager()
        action_system = ActionSystem(graph_manager=graph_manager)
        location = Location("Empty Room", 0, 0, 5, 5, action_system)
        initiator = TestCharacter("Alice", location, social_wellbeing=0)
        location.current_visitors.append(initiator)

        action = Action(
            name="Greet",
            preconditions=[],
            effects=[
                {
                    "targets": ["target_character_in_location"],
                    "attribute": "social_wellbeing",
                    "change_value": 5,
                }
            ],
            initiator=initiator,
            graph_manager=graph_manager,
        )
        result = action.execute()

        self.assertTrue(result, "execute() should still return True")
        self.assertEqual(
            initiator.social_wellbeing,
            0,
            "No effect when the initiator is alone",
        )
        self.assertEqual(graph_manager.updated_nodes, [])

    # ------------------------------------------------------------------
    # No match: target name does not match any co-located character
    # ------------------------------------------------------------------

    def test_no_effect_when_target_name_not_in_location(self):
        """No effect when the requested target name is not in the location."""
        graph_manager = RecordingGraphManager()
        action_system = ActionSystem(graph_manager=graph_manager)
        location = Location("Square", 0, 0, 10, 10, action_system)
        elsewhere = Location("Far Away", 50, 50, 10, 10, action_system)

        initiator = TestCharacter("Alice", location, social_wellbeing=0)
        bob = TestCharacter("Bob", elsewhere, social_wellbeing=0)
        location.current_visitors.append(initiator)
        elsewhere.current_visitors.append(bob)

        # Register both characters in the graph manager so that Bob exists
        # in the global registry but is not co-located with the initiator.
        graph_manager.characters = {
            initiator.name: initiator,
            bob.name: bob,
        }

        action = Action(
            name="Greet",
            preconditions=[],
            effects=[
                {
                    "targets": ["target_character_in_location"],
                    "attribute": "social_wellbeing",
                    "change_value": 5,
                }
            ],
            initiator=initiator,
            target=bob.name,  # Bob is not in the same location
            graph_manager=graph_manager,
        )
        result = action.execute()

        self.assertTrue(result, "execute() should still return True when no target is resolved")
        self.assertEqual(
            initiator.social_wellbeing,
            0,
            "Initiator should not be affected when the named target is elsewhere",
        )
        self.assertEqual(
            bob.social_wellbeing,
            0,
            "Bob is elsewhere and should not be affected",
        )
        self.assertEqual(
            graph_manager.updated_nodes,
            [],
            "No graph updates should occur when the named target is not co-located",
        )

    # ------------------------------------------------------------------
    # Named token alongside initiator token
    # ------------------------------------------------------------------

    def test_named_token_alongside_initiator_token(self):
        """An effect with both 'initiator' and 'target_character_in_location' targets."""
        graph_manager = RecordingGraphManager()
        action_system = ActionSystem(graph_manager=graph_manager)
        location = Location("Guild", 0, 0, 10, 10, action_system)

        alice = TestCharacter("Alice", location, social_wellbeing=0)
        bob = TestCharacter("Bob", location, social_wellbeing=0)
        location.current_visitors.extend([alice, bob])

        action = Action(
            name="CheerUp",
            preconditions=[],
            effects=[
                {
                    "targets": ["initiator", "target_character_in_location"],
                    "attribute": "social_wellbeing",
                    "change_value": 3,
                }
            ],
            initiator=alice,
            graph_manager=graph_manager,
        )
        action.execute()

        self.assertEqual(alice.social_wellbeing, 3, "Initiator token should apply to Alice")
        self.assertEqual(bob.social_wellbeing, 3, "Named token should apply to Bob")

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def test_deduplication_no_double_apply(self):
        """A character appearing twice in visitors is only affected once."""
        graph_manager = RecordingGraphManager()
        action_system = ActionSystem(graph_manager=graph_manager)
        location = Location("Well", 0, 0, 10, 10, action_system)

        initiator = TestCharacter("Alice", location, social_wellbeing=0)
        bob = TestCharacter("Bob", location, social_wellbeing=0)
        # Add Bob twice to simulate a stale list.
        location.current_visitors.extend([initiator, bob, bob])

        action = Action(
            name="Greet",
            preconditions=[],
            effects=[
                {
                    "targets": ["target_character_in_location"],
                    "attribute": "social_wellbeing",
                    "change_value": 5,
                }
            ],
            initiator=initiator,
            graph_manager=graph_manager,
        )
        action.execute()

        self.assertEqual(
            bob.social_wellbeing,
            5,
            "Effect should be applied exactly once despite duplicate visitor entry",
        )


if __name__ == "__main__":
    unittest.main()
