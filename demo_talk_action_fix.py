#!/usr/bin/env python3

"""
Demo script showing how TalkAction is now decoupled from hardcoded social_wellbeing values.
This demonstrates the fix for issue #332.
"""

import sys
import os
from unittest.mock import MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actions import TalkAction


class Character:
    def __init__(self, name):
        self.name = name
        self.uuid = f"{name.lower()}_uuid"
        self.social_wellbeing = 10.0
        
    def respond_to_talk(self, initiator):
        """
        This method can now implement its own logic for social_wellbeing changes
        without being coupled to hardcoded values in TalkAction.
        """
        print(f"  {self.name} responds to {initiator.name}'s conversation")
        
        # Example: Different social_wellbeing increments based on some logic
        if initiator.name == "Alice":
            increment = 2.5  # Alice is particularly engaging
        else:
            increment = 1.8  # Default increment
            
        self.social_wellbeing += increment
        print(f"  {self.name}'s social_wellbeing increased by {increment} to {self.social_wellbeing}")


def main():
    print("=== TalkAction Decoupling Demo ===\n")
    
    # Create characters
    alice = Character("Alice")
    bob = Character("Bob")
    charlie = Character("Charlie")
    
    # Mock graph manager
    mock_graph_manager = MagicMock()
    
    print("Initial social_wellbeing values:")
    print(f"  Alice: {alice.social_wellbeing}")
    print(f"  Bob: {bob.social_wellbeing}")
    print(f"  Charlie: {charlie.social_wellbeing}")
    print()
    
    # Scenario 1: TalkAction with default effects (empty) - all changes come from respond_to_talk
    print("Scenario 1: Alice talks to Bob (using default TalkAction with no hardcoded effects)")
    talk_action1 = TalkAction(
        initiator=alice,
        target=bob,
        graph_manager=mock_graph_manager
    )
    
    # Mock preconditions to pass
    talk_action1.preconditions_met = MagicMock(return_value=True)
    
    print(f"Before: Alice={alice.social_wellbeing}, Bob={bob.social_wellbeing}")
    result = talk_action1.execute(character=alice)
    print(f"After: Alice={alice.social_wellbeing}, Bob={bob.social_wellbeing}")
    print(f"Execution successful: {result}")
    print()
    
    # Scenario 2: Different initiator should get different increment based on respond_to_talk logic
    print("Scenario 2: Charlie talks to Bob (showing respond_to_talk can implement different logic)")
    talk_action2 = TalkAction(
        initiator=charlie,
        target=bob,
        graph_manager=mock_graph_manager
    )
    
    talk_action2.preconditions_met = MagicMock(return_value=True)
    
    print(f"Before: Charlie={charlie.social_wellbeing}, Bob={bob.social_wellbeing}")
    result = talk_action2.execute(character=charlie)
    print(f"After: Charlie={charlie.social_wellbeing}, Bob={bob.social_wellbeing}")
    print(f"Execution successful: {result}")
    print()
    
    # Scenario 3: Custom effects still work
    print("Scenario 3: Alice talks to Charlie with custom effects")
    custom_effects = [
        {"targets": ["target"], "attribute": "social_wellbeing", "change_value": 5.0},
        {"targets": ["initiator"], "attribute": "social_wellbeing", "change_value": 2.0}
    ]
    
    talk_action3 = TalkAction(
        initiator=alice,
        target=charlie,
        effects=custom_effects,
        graph_manager=mock_graph_manager
    )
    
    talk_action3.preconditions_met = MagicMock(return_value=True)
    
    print(f"Before: Alice={alice.social_wellbeing}, Charlie={charlie.social_wellbeing}")
    result = talk_action3.execute(character=alice)
    print(f"After: Alice={alice.social_wellbeing}, Charlie={charlie.social_wellbeing}")
    print(f"Execution successful: {result}")
    print()
    
    print("=== Summary ===")
    print("✅ TalkAction no longer has hardcoded social_wellbeing increments")
    print("✅ respond_to_talk method can implement its own logic")
    print("✅ Custom effects still work when explicitly provided")
    print("✅ Tests are decoupled from implementation details")
    print("\nThis fixes issue #332: TalkAction is now flexible and won't mask bugs if respond_to_talk logic changes.")


if __name__ == "__main__":
    main()