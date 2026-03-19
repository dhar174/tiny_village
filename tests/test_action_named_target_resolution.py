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

from actions import Action


# ---------------------------------------------------------------------------
# Minimal stand-ins for Character and Location.
# These are lightweight plain-Python objects that carry only the attributes
# the resolution path actually reads.  Using full game classes would pull in
# heavy optional dependencies (pygame, networkx, …) that are not available in
# a CI environment.
# ---------------------------------------------------------------------------


class _Location:
    """Minimal Location with a name, uuid, and visitor list."""

    def __init__(self, name):
        self.name = name
        self.uuid = f"loc_{name}"
        self.current_visitors = []

    def add_visitor(self, character):
        self.current_visitors.append(character)
        character.location = self


class _Character:
    """Minimal Character carrying the attributes Action resolution reads."""

    def __init__(self, name, location=None):
        self.name = name
        self.uuid = f"char_{name}"
        self.location = location
        # Attribute that test effects will modify.
        self.social_wellbeing = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNamedTargetResolution(unittest.TestCase):
    """Action.execute() correctly resolves target_character_in_location."""

    def _make_action(self, initiator, target=None):
        """Return an Action whose effect targets target_character_in_location."""
        return Action(
            name="Greet",
            preconditions=[],
            effects=[
                {
                    "targets": ["target_character_in_location"],
                    "attribute": "social_wellbeing",
                    "change_value": 5,
                }
            ],
            cost=0,
            initiator=initiator,
            target=target,
            graph_manager=None,
        )

    # ------------------------------------------------------------------
    # Happy-path: a single other character shares the location
    # ------------------------------------------------------------------

    def test_effect_applied_to_character_in_location(self):
        """Effect is applied to the co-located character."""
        location = _Location("tavern")
        alice = _Character("Alice")
        bob = _Character("Bob")
        location.add_visitor(alice)
        location.add_visitor(bob)

        action = self._make_action(initiator=alice)
        result = action.execute()

        self.assertTrue(result, "execute() should return True when preconditions pass")
        self.assertEqual(
            bob.social_wellbeing,
            5,
            "Effect should be applied to the co-located character (Bob)",
        )
        # The initiator must not receive the effect (they are excluded from
        # _iter_characters_in_initiator_location).
        self.assertEqual(
            alice.social_wellbeing,
            0,
            "Effect must NOT be applied to the initiator",
        )

    # ------------------------------------------------------------------
    # Target filtering: only the character matching target is resolved
    # ------------------------------------------------------------------

    def test_resolves_matching_target_by_name(self):
        """When self.target is a name string only the matching character is affected."""
        location = _Location("market")
        alice = _Character("Alice")
        bob = _Character("Bob")
        carol = _Character("Carol")
        location.add_visitor(alice)
        location.add_visitor(bob)
        location.add_visitor(carol)

        # Request the action specifically at Carol by name.
        action = self._make_action(initiator=alice, target="Carol")
        action.execute()

        self.assertEqual(carol.social_wellbeing, 5, "Carol should be the resolved target")
        self.assertEqual(bob.social_wellbeing, 0, "Bob should not be affected")
        self.assertEqual(alice.social_wellbeing, 0, "Alice (initiator) should not be affected")

    def test_resolves_matching_target_by_object(self):
        """When self.target is an object only the matching character is affected."""
        location = _Location("forest")
        alice = _Character("Alice")
        bob = _Character("Bob")
        carol = _Character("Carol")
        location.add_visitor(alice)
        location.add_visitor(bob)
        location.add_visitor(carol)

        action = self._make_action(initiator=alice, target=carol)
        action.execute()

        self.assertEqual(carol.social_wellbeing, 5, "Carol should be the resolved target")
        self.assertEqual(bob.social_wellbeing, 0, "Bob should not be affected")

    # ------------------------------------------------------------------
    # No match: no other character is in the same location
    # ------------------------------------------------------------------

    def test_no_effect_when_no_other_character_in_location(self):
        """No effect applied when the initiator is alone in the location."""
        location = _Location("empty_room")
        alice = _Character("Alice")
        location.add_visitor(alice)

        action = self._make_action(initiator=alice)
        result = action.execute()

        self.assertTrue(result, "execute() should still return True")
        self.assertEqual(
            alice.social_wellbeing,
            0,
            "No effect when there is no co-located character",
        )

    def test_no_effect_when_initiator_has_no_location(self):
        """No effect applied when the initiator has no location set."""
        alice = _Character("Alice", location=None)
        bob = _Character("Bob", location=None)

        action = self._make_action(initiator=alice)
        result = action.execute()

        self.assertTrue(result, "execute() should still return True")
        self.assertEqual(alice.social_wellbeing, 0)
        self.assertEqual(bob.social_wellbeing, 0)

    # ------------------------------------------------------------------
    # Multiple visitors: only one effect per unique character
    # ------------------------------------------------------------------

    def test_deduplication_no_double_apply(self):
        """The same character appearing twice in visitors is only affected once."""
        location = _Location("well")
        alice = _Character("Alice")
        bob = _Character("Bob")
        # Intentionally add bob twice to simulate a stale list.
        location.add_visitor(alice)
        location.current_visitors.append(bob)  # first occurrence
        bob.location = location
        location.current_visitors.append(bob)  # duplicate

        action = self._make_action(initiator=alice)
        action.execute()

        self.assertEqual(
            bob.social_wellbeing,
            5,
            "Effect should be applied exactly once despite duplicate visitor entry",
        )

    # ------------------------------------------------------------------
    # Effect is cumulative when execute() is called multiple times
    # ------------------------------------------------------------------

    def test_repeated_execute_accumulates_effect(self):
        """Calling execute() twice applies the effect twice."""
        location = _Location("inn")
        alice = _Character("Alice")
        bob = _Character("Bob")
        location.add_visitor(alice)
        location.add_visitor(bob)

        action = self._make_action(initiator=alice)
        action.execute()
        action.execute()

        self.assertEqual(
            bob.social_wellbeing,
            10,
            "Second execute() should accumulate the effect",
        )

    # ------------------------------------------------------------------
    # Named token mixed with other target tokens in the same effect
    # ------------------------------------------------------------------

    def test_named_token_alongside_initiator_token(self):
        """An effect with both 'initiator' and 'target_character_in_location' targets."""
        location = _Location("guild")
        alice = _Character("Alice")
        bob = _Character("Bob")
        location.add_visitor(alice)
        location.add_visitor(bob)

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
            cost=0,
            initiator=alice,
            graph_manager=None,
        )
        action.execute()

        self.assertEqual(alice.social_wellbeing, 3, "Initiator token should apply to Alice")
        self.assertEqual(bob.social_wellbeing, 3, "Named token should apply to Bob")


if __name__ == "__main__":
    unittest.main()
