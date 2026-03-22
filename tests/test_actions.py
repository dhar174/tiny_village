import unittest
from unittest.mock import MagicMock

from actions import Action, ActionTemplate, Condition, State, TalkAction
from tests.mock_character import MockCharacter, MockPersonalityTraits


class CharacterWithState(MockCharacter):
    """MockCharacter variant that can participate in real Condition checks."""

    def get_state(self):
        return State(self)


def build_character(name, **overrides):
    defaults = {
        "name": name,
        "energy": 10.0,
        "friendship_grid": {},
        "personality_traits": MockPersonalityTraits(
            agreeableness=5,
            extraversion=5,
        ),
    }
    defaults.update(overrides)
    return CharacterWithState(**defaults)


class TestActionExecution(unittest.TestCase):
    def setUp(self):
        self.graph_manager = MagicMock()
        self.alice = build_character("Alice")
        self.bob = build_character("Bob", status="idle")

    def test_action_execute_applies_effects_and_updates_graph(self):
        action = Action(
            name="BoostAndSetStatus",
            preconditions=[],
            effects=[
                {"targets": ["initiator"], "attribute": "energy", "change_value": -2},
                {"targets": ["target"], "attribute": "status", "change_value": "set:active"},
            ],
            initiator=self.alice,
            target=self.bob,
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.assertEqual(self.alice.energy, 8.0)
        self.assertEqual(self.bob.status, "active")
        self.graph_manager.update_node_attribute.assert_any_call(
            self.alice.uuid,
            "energy",
            8.0,
        )
        self.graph_manager.update_node_attribute.assert_any_call(
            self.bob.uuid,
            "status",
            "active",
        )
        self.assertEqual(self.graph_manager.update_node_attribute.call_count, 2)

    def test_action_execute_honors_real_condition_preconditions(self):
        condition = Condition("HasEnergy", "energy", self.alice, 20, ">=")
        action = Action(
            name="RequiresEnergy",
            preconditions=[condition],
            effects=[
                {"targets": ["initiator"], "attribute": "energy", "change_value": -1},
            ],
            initiator=self.alice,
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertFalse(result)
        self.assertEqual(self.alice.energy, 10.0)
        self.graph_manager.update_node_attribute.assert_not_called()

    def test_action_template_passes_graph_manager_into_instantiated_action(self):
        template = ActionTemplate(
            "TemplateAction",
            [],
            [{"targets": ["initiator"], "attribute": "energy", "change_value": -1}],
            cost=0.25,
        )

        action = template.instantiate(
            {
                "initiator": self.alice,
                "target": self.bob,
                "graph_manager": self.graph_manager,
            }
        )

        self.assertIsInstance(action, Action)
        self.assertIs(action.graph_manager, self.graph_manager)
        self.assertIs(action.initiator, self.alice)
        self.assertIs(action.target, self.bob)


class TestTalkActionIntegration(unittest.TestCase):
    def setUp(self):
        self.graph_manager = MagicMock()
        self.alice = build_character("Alice", social_wellbeing=5.0)
        self.bob = build_character("Bob", social_wellbeing=6.0)

    def test_talk_action_uses_wrapped_real_response_without_hardcoded_delta(self):
        initial_target_social = self.bob.social_wellbeing
        action_only_social = initial_target_social + 1.0
        self.bob.respond_to_talk = MagicMock(wraps=self.bob.respond_to_talk)

        action = TalkAction(
            initiator=self.alice,
            target=self.bob,
            effects=[
                {
                    "targets": ["target"],
                    "attribute": "social_wellbeing",
                    "change_value": 1.0,
                }
            ],
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.bob.respond_to_talk.assert_called_once_with(self.alice)
        self.assertGreater(self.bob.social_wellbeing, action_only_social)
        self.graph_manager.update_node_attribute.assert_called_once_with(
            self.bob.uuid,
            "social_wellbeing",
            action_only_social,
        )
        self.graph_manager.add_character_character_edge.assert_called_once_with(
            self.alice,
            self.bob,
            impact_factor=1.0,
            impact_value=0.1,
        )

    def test_talk_action_default_execution_relies_on_target_response(self):
        initial_target_social = self.bob.social_wellbeing
        self.bob.respond_to_talk = MagicMock(wraps=self.bob.respond_to_talk)

        action = TalkAction(
            initiator=self.alice,
            target=self.bob,
            graph_manager=self.graph_manager,
        )

        result = action.execute(character=self.alice)

        self.assertTrue(result)
        self.assertGreater(self.bob.social_wellbeing, initial_target_social)
        self.bob.respond_to_talk.assert_called_once_with(self.alice)
        self.graph_manager.update_node_attribute.assert_not_called()
        self.graph_manager.add_character_character_edge.assert_called_once_with(
            self.alice,
            self.bob,
            impact_factor=1.0,
            impact_value=0.1,
        )


if __name__ == "__main__":
    unittest.main()
