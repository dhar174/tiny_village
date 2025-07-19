#!/usr/bin/env python3

"""
Test to demonstrate that TalkAction is now decoupled from hardcoded social_wellbeing values.
This test shows that the fix for issue #332 is working correctly.
"""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actions import TalkAction


class MockCharacter:
    def __init__(self, name="TestCharacter", uuid=None):
        self.name = name
        self.uuid = uuid or f"{name}_uuid"
        self.social_wellbeing = 10.0


class TestTalkActionDecoupling(unittest.TestCase):
    """Test that TalkAction no longer has hardcoded social_wellbeing increments."""

    def setUp(self):
        self.initiator = MockCharacter(name="Alice", uuid="alice_uuid")
        self.target = MockCharacter(name="Bob", uuid="bob_uuid")
        self.mock_graph_manager = MagicMock()

    def test_talk_action_no_hardcoded_social_wellbeing_effects(self):
        """Test that TalkAction with default effects doesn't apply hardcoded social_wellbeing changes."""
        
        # Store initial values
        initial_target_social = self.target.social_wellbeing
        initial_initiator_social = self.initiator.social_wellbeing
        
        # Mock respond_to_talk to simulate it changing social_wellbeing
        def mock_respond_to_talk(initiator):
            # Simulate respond_to_talk method handling social_wellbeing changes
            self.target.social_wellbeing += 2.5  # Different value than the old hardcoded 1.0
            
        self.target.respond_to_talk = MagicMock(side_effect=mock_respond_to_talk)

        # Create TalkAction with default effects (should be empty)
        talk_action = TalkAction(
            initiator=self.initiator,
            target=self.target,
            graph_manager=self.mock_graph_manager
        )
        
        # Mock preconditions to pass
        talk_action.preconditions_met = MagicMock(return_value=True)

        # Execute the action
        result = talk_action.execute(character=self.initiator)
        self.assertTrue(result)

        # Verify that TalkAction itself didn't apply any hardcoded social_wellbeing changes
        # The initiator's social_wellbeing should be unchanged by TalkAction's effects
        self.assertEqual(self.initiator.social_wellbeing, initial_initiator_social)
        
        # The target's social_wellbeing should only change due to respond_to_talk method
        expected_target_social = initial_target_social + 2.5  # From our mock respond_to_talk
        self.assertEqual(self.target.social_wellbeing, expected_target_social)

        # Verify respond_to_talk was called
        self.target.respond_to_talk.assert_called_once_with(self.initiator)

        # Verify no graph updates occurred from TalkAction's effects (since effects are empty)
        self.mock_graph_manager.update_node_attribute.assert_not_called()

    def test_talk_action_respects_custom_effects(self):
        """Test that TalkAction still respects explicitly provided custom effects."""
        
        # Define custom effects
        custom_effects = [
            {"targets": ["target"], "attribute": "social_wellbeing", "change_value": 3.0},
            {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 1.5}
        ]
        
        initial_target_social = self.target.social_wellbeing
        initial_initiator_social = self.initiator.social_wellbeing
        
        self.target.respond_to_talk = MagicMock()

        # Create TalkAction with custom effects
        talk_action = TalkAction(
            initiator=self.initiator,
            target=self.target,
            effects=custom_effects,
            graph_manager=self.mock_graph_manager
        )
        
        talk_action.preconditions_met = MagicMock(return_value=True)

        result = talk_action.execute(character=self.initiator)
        self.assertTrue(result)

        # Verify custom effects were applied
        expected_target_social = initial_target_social + 3.0
        expected_initiator_social = initial_initiator_social + 1.5
        
        self.assertEqual(self.target.social_wellbeing, expected_target_social)
        self.assertEqual(self.initiator.social_wellbeing, expected_initiator_social)

        # Verify respond_to_talk was called
        self.target.respond_to_talk.assert_called_once_with(self.initiator)

        # Verify graph updates occurred for custom effects
        self.assertEqual(self.mock_graph_manager.update_node_attribute.call_count, 2)
        self.mock_graph_manager.update_node_attribute.assert_any_call(
            self.target.uuid, "social_wellbeing", expected_target_social
        )
        self.mock_graph_manager.update_node_attribute.assert_any_call(
            self.initiator.uuid, "social_wellbeing", expected_initiator_social
        )


if __name__ == "__main__":
    unittest.main()