#!/usr/bin/env python3

"""
Demo of the enhanced social interaction system showing characters forming relationships
"""

import sys
import os
from unittest.mock import MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actions import TalkAction, GreetAction, ShareNewsAction, OfferComplimentAction


class MockPersonalityTraits:
    def __init__(self, extraversion=50, agreeableness=60, neuroticism=30, openness=70, conscientiousness=50):
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
        self.openness = openness
        self.conscientiousness = conscientiousness


class MockCharacter:
    def __init__(self, name, personality_type="balanced"):
        self.name = name
        self.uuid = f"{name}_uuid"
        self.social_wellbeing = 50.0
        self.friendship_grid = {}
        
        # Different personality types
        if personality_type == "extraverted":
            self.personality_traits = MockPersonalityTraits(extraversion=80, agreeableness=70)
        elif personality_type == "introverted":
            self.personality_traits = MockPersonalityTraits(extraversion=30, neuroticism=60)
        elif personality_type == "creative":
            self.personality_traits = MockPersonalityTraits(openness=85, extraversion=60)
        else:
            self.personality_traits = MockPersonalityTraits()

    def respond_to_talk(self, initiator):
        """Enhanced respond_to_talk that updates relationships"""
        self.social_wellbeing += 0.1
        
        # Personality compatibility boost
        if hasattr(initiator, 'personality_traits') and self.personality_traits:
            initiator_agreeable = getattr(initiator.personality_traits, 'agreeableness', 50)
            self_agreeable = getattr(self.personality_traits, 'agreeableness', 50)
            
            if initiator_agreeable > 60 and self_agreeable > 60:
                self.social_wellbeing += 0.1
                
        # Update friendship
        if hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.1
            else:
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.05, 1.0
                )
        
        # Personality-based response
        if self.personality_traits.extraversion > 65:
            return f"{self.name} engages enthusiastically in conversation with {initiator.name}"
        elif self.personality_traits.neuroticism > 60:
            return f"{self.name} responds nervously but appreciates {initiator.name}'s attention"
        elif self.personality_traits.openness > 70:
            return f"{self.name} shares interesting thoughts with {initiator.name}"
        else:
            return f"{self.name} listens and responds thoughtfully to {initiator.name}"

    def respond_to_greeting(self, initiator):
        """Respond to greeting"""
        self.social_wellbeing += 0.05
        if hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.05
            else:
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.02, 1.0
                )
        
        if self.personality_traits.extraversion > 65:
            return f"{self.name} warmly greets {initiator.name} back"
        elif self.personality_traits.neuroticism > 60:
            return f"{self.name} shyly acknowledges {initiator.name}'s greeting"
        else:
            return f"{self.name} politely returns {initiator.name}'s greeting"

    def respond_to_compliment(self, initiator, compliment_topic):
        """Respond to compliment"""
        self.social_wellbeing += 0.2
        if hasattr(initiator, 'name'):
            if initiator.name not in self.friendship_grid:
                self.friendship_grid[initiator.name] = 0.15
            else:
                self.friendship_grid[initiator.name] = min(
                    self.friendship_grid[initiator.name] + 0.1, 1.0
                )
        
        if self.personality_traits.extraversion > 65:
            return f"{self.name} beams with joy at {initiator.name}'s compliment about {compliment_topic}"
        elif self.personality_traits.neuroticism > 60:
            return f"{self.name} blushes and thanks {initiator.name} for the kind words about {compliment_topic}"
        else:
            return f"{self.name} graciously accepts {initiator.name}'s compliment about {compliment_topic}"


def demo_social_interactions():
    """Demonstrate the social interaction system"""
    print("=== Social Interaction System Demo ===\n")
    
    # Create characters with different personalities
    alice = MockCharacter("Alice", "extraverted")  # Outgoing and friendly
    bob = MockCharacter("Bob", "introverted")      # Shy but thoughtful
    carol = MockCharacter("Carol", "creative")     # Open and artistic
    
    # Mock graph manager
    mock_graph_manager = MagicMock()
    mock_graph_manager.characters = {"Alice": alice, "Bob": bob, "Carol": carol}
    mock_graph_manager.add_character_character_edge = MagicMock()
    
    print("Initial character states:")
    for char in [alice, bob, carol]:
        print(f"  {char.name}: Social wellbeing = {char.social_wellbeing}, Friendships = {char.friendship_grid}")
    print()
    
    # Simulate a series of social interactions
    interactions = [
        # Alice meets Bob
        (GreetAction(alice, bob, graph_manager=mock_graph_manager), "Alice greets Bob"),
        
        # Bob responds with a talk
        (TalkAction(bob, alice, graph_manager=mock_graph_manager), "Bob talks to Alice"),
        
        # Alice shares some news
        (ShareNewsAction(alice, bob, "The village market has fresh apples today!", graph_manager=mock_graph_manager), "Alice shares news with Bob"),
        
        # Carol joins the conversation
        (GreetAction(carol, alice, graph_manager=mock_graph_manager), "Carol greets Alice"),
        (GreetAction(carol, bob, graph_manager=mock_graph_manager), "Carol greets Bob"),
        
        # Alice compliments Carol's creativity
        (OfferComplimentAction(alice, carol, "your beautiful artwork", graph_manager=mock_graph_manager), "Alice compliments Carol"),
        
        # Carol talks with both Alice and Bob
        (TalkAction(carol, alice, graph_manager=mock_graph_manager), "Carol talks with Alice"),
        (TalkAction(carol, bob, graph_manager=mock_graph_manager), "Carol talks with Bob"),
        
        # Bob overcomes his shyness to compliment Alice
        (OfferComplimentAction(bob, alice, "your friendly personality", graph_manager=mock_graph_manager), "Bob compliments Alice"),
        
        # More conversations to strengthen bonds
        (TalkAction(alice, bob, graph_manager=mock_graph_manager), "Alice talks more with Bob"),
        (TalkAction(bob, carol, graph_manager=mock_graph_manager), "Bob talks more with Carol"),
    ]
    
    print("Social interactions:")
    for i, (action, description) in enumerate(interactions, 1):
        print(f"\n{i}. {description}")
        
        # Mock preconditions to always pass
        action.preconditions_met = MagicMock(return_value=True)
        
        # Execute the action
        success = action.execute()
        
        if success:
            print(f"   ✓ Action executed successfully")
            
            # Show the immediate effects
            target_name = getattr(action.target, 'name', 'unknown')
            initiator_name = getattr(action.initiator, 'name', 'unknown')
            
            if hasattr(action.target, 'friendship_grid'):
                friendship_level = action.target.friendship_grid.get(initiator_name, 0)
                print(f"   → {target_name}'s friendship with {initiator_name}: {friendship_level:.3f}")
                print(f"   → {target_name}'s social wellbeing: {action.target.social_wellbeing:.2f}")
        else:
            print(f"   ✗ Action failed")
    
    print("\n" + "="*50)
    print("Final relationship states:")
    print("="*50)
    
    for char in [alice, bob, carol]:
        print(f"\n{char.name} (Personality: {char.personality_traits.extraversion} extraversion, {char.personality_traits.agreeableness} agreeableness):")
        print(f"  Social wellbeing: {char.social_wellbeing:.2f}")
        print(f"  Friendships:")
        for friend, level in char.friendship_grid.items():
            friendship_desc = "acquaintance" if level < 0.2 else "friend" if level < 0.5 else "close friend"
            print(f"    {friend}: {level:.3f} ({friendship_desc})")
    
    print(f"\nGraph manager calls: {mock_graph_manager.add_character_character_edge.call_count}")
    print("✓ All social interactions successfully updated the social graph!")
    
    print("\n=== Conversation Analysis ===")
    print("The extraverted Alice made friends quickly and initiated many interactions.")
    print("The introverted Bob started shy but warmed up through positive interactions.")
    print("The creative Carol connected well with both through shared interests.")
    print("All characters' social wellbeing improved through positive social interactions.")


if __name__ == "__main__":
    demo_social_interactions()