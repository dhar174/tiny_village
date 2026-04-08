#!/usr/bin/env python3

"""Behavior-focused coverage for social action execution."""

import unittest
from unittest.mock import MagicMock

from actions import Condition, GreetAction, OfferComplimentAction, ShareNewsAction, TalkAction
from tests.social_action_test_utils import build_character


class TestSocialInteractions(unittest.TestCase):
    def setUp(self):
        self.alice = build_character("Alice", default_energy=8.0)
        self.bob = build_character("Bob", default_energy=8.0)
        self.graph_manager = MagicMock()
        self.graph_manager.characters = {
            self.alice.name: self.alice,
            self.bob.name: self.bob,
        }

    def test_talk_action_uses_real_preconditions_and_wrapped_response(self):
        initial_social = self.bob.social_wellbeing
        self.bob.respond_to_talk = MagicMock(wraps=self.bob.respond_to_talk)
        precondition = Condition("HasEnergy", "energy", self.alice, 5, ">=")

        action = TalkAction(
            initiator=self.alice,
            target=self.bob,
            preconditions=[precondition],
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.bob.respond_to_talk.assert_called_once_with(self.alice)
        self.assertGreater(self.bob.social_wellbeing, initial_social)
        self.graph_manager.add_character_character_edge.assert_called_once_with(
            self.alice,
            self.bob,
            impact_factor=1.0,
            impact_value=0.1,
        )

    def test_talk_action_stops_when_preconditions_fail(self):
        self.alice.energy = 1.0
        self.bob.respond_to_talk = MagicMock(wraps=self.bob.respond_to_talk)
        precondition = Condition("HasEnergy", "energy", self.alice, 5, ">=")

        action = TalkAction(
            initiator=self.alice,
            target=self.bob,
            preconditions=[precondition],
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertFalse(result)
        self.bob.respond_to_talk.assert_not_called()
        self.graph_manager.add_character_character_edge.assert_not_called()

    def test_greet_action_uses_general_talk_fallback(self):
        initial_social = self.bob.social_wellbeing
        self.bob.respond_to_talk = MagicMock(wraps=self.bob.respond_to_talk)

        action = GreetAction(
            initiator=self.alice,
            target=self.bob,
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.bob.respond_to_talk.assert_called_once_with(self.alice)
        self.assertGreater(self.bob.social_wellbeing, initial_social + 0.5)
        self.graph_manager.add_character_character_edge.assert_called_once_with(
            self.alice,
            self.bob,
            impact_factor=1.0,
            impact_value=0.05,
        )

    def test_share_news_action_records_memory_and_updates_target_state(self):
        news_item = "The market opens at dawn"
        initial_social = self.bob.social_wellbeing
        self.graph_manager.memory_manager = MagicMock()

        action = ShareNewsAction(
            initiator=self.alice,
            target=self.bob,
            news_item=news_item,
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.assertEqual(self.bob.social_wellbeing, initial_social + 1)
        self.assertEqual(self.bob.knowledge, news_item)
        self.graph_manager.add_character_character_edge.assert_called_once_with(
            self.alice,
            self.bob,
            impact_factor=1.0,
            impact_value=0.15,
        )
        self.graph_manager.memory_manager.add_memory.assert_called_once_with(
            f"{self.alice.name} shared news with {self.bob.name}: {news_item}",
            importance_score=4,
        )

    def test_offer_compliment_action_uses_fallback_response_after_effects(self):
        initial_social = self.bob.social_wellbeing
        self.bob.respond_to_talk = MagicMock(wraps=self.bob.respond_to_talk)

        action = OfferComplimentAction(
            initiator=self.alice,
            target=self.bob,
            compliment_topic="their baking",
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.assertEqual(self.bob.relationship_strength, 1)
        self.assertGreater(self.bob.social_wellbeing, initial_social + 1.5)
        self.bob.respond_to_talk.assert_called_once_with(self.alice)
        self.graph_manager.add_character_character_edge.assert_called_once_with(
            self.alice,
            self.bob,
            impact_factor=1.0,
            impact_value=0.2,
        )


if __name__ == "__main__":
    unittest.main()
