#!/usr/bin/env python3

"""
Test the enhanced social interaction system
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actions import TalkAction, GreetAction, ShareNewsAction, OfferComplimentAction


class MockPersonalityTraits:
    def __init__(self, extraversion=50, agreeableness=50, neuroticism=50, openness=50):
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
        self.openness = openness


class MockCharacter:
    def __init__(self, name="TestCharacter", uuid=None):
        self.name = name
        self.uuid = uuid or f"{name}_uuid"
        self.social_wellbeing = 10.0
        self.friendship_grid = {}
        self.personality_traits = MockPersonalityTraits()

    def respond_to_talk(self, initiator):
        """Mock respond_to_talk method that mimics the enhanced version"""
        self.social_wellbeing += 0.1
        if hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.1
            else:
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.05, 1.0
                )
        return f"{self.name} responds to {getattr(initiator, 'name', 'someone')}"

    def respond_to_greeting(self, initiator):
        """Mock respond_to_greeting method"""
        self.social_wellbeing += 0.05
        if hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.05
            else:
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.02, 1.0
                )
        return f"{self.name} greets {getattr(initiator, 'name', 'someone')} back"

    def respond_to_compliment(self, initiator, compliment_topic):
        """Mock respond_to_compliment method"""
        self.social_wellbeing += 0.2
        if hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.15
            else:
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.1, 1.0
                )
        return f"{self.name} thanks {getattr(initiator, 'name', 'someone')} for the compliment about {compliment_topic}"


class TestSocialInteractions(unittest.TestCase):

    def setUp(self):
        self.alice = MockCharacter(name="Alice", uuid="alice_uuid")
        self.bob = MockCharacter(name="Bob", uuid="bob_uuid")
        self.mock_graph_manager = MagicMock()
        
        # Mock the add_character_character_edge method
        self.mock_graph_manager.add_character_character_edge = MagicMock()
        self.mock_graph_manager.characters = {"Alice": self.alice, "Bob": self.bob}

    def test_talk_action_enhances_relationships(self):
        """Test that TalkAction updates relationships and memories"""
        
        # Create TalkAction
        talk_action = TalkAction(
            initiator=self.alice,
            target=self.bob,
            graph_manager=self.mock_graph_manager
        )
        
        # Mock preconditions to pass
        talk_action.preconditions_met = MagicMock(return_value=True)

        # Execute the action
        result = talk_action.execute(character=self.alice)
        self.assertTrue(result)

        # Verify relationship was updated
        self.assertIn(self.alice.name, self.bob.friendship_grid)
        self.assertEqual(self.bob.friendship_grid[self.alice.name], 0.1)

        # Verify social wellbeing increased
        self.assertEqual(self.bob.social_wellbeing, 10.1)

        # Verify graph manager was called to update relationship
        self.mock_graph_manager.add_character_character_edge.assert_called_once()

    def test_greet_action_creates_initial_relationship(self):
        """Test that GreetAction creates initial relationships"""
        
        # Create GreetAction
        greet_action = GreetAction(
            initiator=self.alice,
            target=self.bob,
            graph_manager=self.mock_graph_manager
        )
        
        greet_action.preconditions_met = MagicMock(return_value=True)

        # Execute the action
        result = greet_action.execute(character=self.alice)
        self.assertTrue(result)

        # Verify relationship was created with smaller value for greeting
        self.assertIn(self.alice.name, self.bob.friendship_grid)
        self.assertEqual(self.bob.friendship_grid[self.alice.name], 0.05)

        # Verify social wellbeing increased (base effect 0.5 + respond_to_greeting 0.05)
        self.assertEqual(self.bob.social_wellbeing, 10.55)

        # Verify graph manager was called
        self.mock_graph_manager.add_character_character_edge.assert_called_once()

    def test_compliment_action_strong_relationship_impact(self):
        """Test that OfferComplimentAction has strong relationship impact"""
        
        # Create OfferComplimentAction
        compliment_action = OfferComplimentAction(
            initiator=self.alice,
            target=self.bob,
            compliment_topic="your cooking skills",
            graph_manager=self.mock_graph_manager
        )
        
        compliment_action.preconditions_met = MagicMock(return_value=True)

        # Execute the action
        result = compliment_action.execute(character=self.alice)
        self.assertTrue(result)

        # Verify relationship was created with larger value for compliment
        self.assertIn(self.alice.name, self.bob.friendship_grid)
        self.assertEqual(self.bob.friendship_grid[self.alice.name], 0.15)

        # Verify social wellbeing increased significantly (base effect 1.5 + respond_to_compliment 0.2)
        self.assertEqual(self.bob.social_wellbeing, 11.7)

        # Verify graph manager was called
        self.mock_graph_manager.add_character_character_edge.assert_called_once()

    def test_share_news_action_moderate_impact(self):
        """Test that ShareNewsAction has moderate relationship impact"""
        
        # Create ShareNewsAction
        news_action = ShareNewsAction(
            initiator=self.alice,
            target=self.bob,
            news_item="The village market is having a sale tomorrow",
            graph_manager=self.mock_graph_manager
        )
        
        news_action.preconditions_met = MagicMock(return_value=True)

        # Execute the action
        result = news_action.execute(character=self.alice)
        self.assertTrue(result)

        # Verify graph manager was called with moderate impact
        self.mock_graph_manager.add_character_character_edge.assert_called_once()

    def test_repeated_interactions_strengthen_relationships(self):
        """Test that repeated interactions strengthen existing relationships"""
        
        # Start with a greeting to establish initial relationship
        greet_action = GreetAction(
            initiator=self.alice,
            target=self.bob,
            graph_manager=self.mock_graph_manager
        )
        greet_action.preconditions_met = MagicMock(return_value=True)
        greet_action.execute(character=self.alice)
        
        initial_friendship = self.bob.friendship_grid[self.alice.name]
        initial_wellbeing = self.bob.social_wellbeing

        # Follow up with a talk action
        talk_action = TalkAction(
            initiator=self.alice,
            target=self.bob,
            graph_manager=self.mock_graph_manager
        )
        talk_action.preconditions_met = MagicMock(return_value=True)
        talk_action.execute(character=self.alice)

        # Verify relationship strengthened
        self.assertGreater(self.bob.friendship_grid[self.alice.name], initial_friendship)
        self.assertGreater(self.bob.social_wellbeing, initial_wellbeing)

        # Verify graph manager was called multiple times
        self.assertEqual(self.mock_graph_manager.add_character_character_edge.call_count, 2)

    def test_social_action_impact_values(self):
        """Test that different social actions have appropriate impact values"""
        
        actions = [
            (TalkAction, 0.1),
            (GreetAction, 0.05),
            (ShareNewsAction, 0.15),
            (OfferComplimentAction, 0.2)
        ]
        
        for action_class, expected_impact in actions:
            with self.subTest(action_class=action_class):
                if action_class in [ShareNewsAction, OfferComplimentAction]:
                    # These require additional parameters
                    if action_class == ShareNewsAction:
                        action = action_class(
                            initiator=self.alice,
                            target=self.bob,
                            news_item="test news",
                            graph_manager=self.mock_graph_manager
                        )
                    else:  # OfferComplimentAction
                        action = action_class(
                            initiator=self.alice,
                            target=self.bob,
                            compliment_topic="test",
                            graph_manager=self.mock_graph_manager
                        )
                else:
                    action = action_class(
                        initiator=self.alice,
                        target=self.bob,
                        graph_manager=self.mock_graph_manager
                    )
                
                # Test the _get_social_impact method
                self.assertEqual(action._get_social_impact(), expected_impact)


if __name__ == "__main__":
    unittest.main()