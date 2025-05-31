#!/usr/bin/env python3
"""
Minimal integration test for StrategyManager using REAL production classes.
Avoids heavy dependencies that might cause hanging.
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only essential real classes, avoid heavy memory manager
from actions import Action, ActionSystem
from tiny_strategy_manager import StrategyManager
from tiny_goap_system import GOAPPlanner


class MockEvent:
    """Simple event class for testing."""

    def __init__(self, event_type, description, participants=None, location=None):
        self.type = event_type
        self.event_type = event_type
        self.description = description
        self.participants = participants or []
        self.location = location
        self.timestamp = datetime.now()


class MockGraphManager:
    """Mock GraphManager to avoid heavy dependencies."""

    def __init__(self):
        self.characters = {
            "Emma": {"name": "Emma", "hunger": 60, "energy": 40},
            "Bob": {"name": "Bob", "hunger": 30, "energy": 80},
            "Alice": {"name": "Alice", "hunger": 45, "energy": 70},
        }

    def get_character_state(self, character_name):
        return self.characters.get(character_name, {})

    def get_possible_actions(self, character_name):
        return [
            Action(name="Eat", effects=[{"attribute": "hunger", "change_value": -20}]),
            Action(name="Sleep", effects=[{"attribute": "energy", "change_value": 30}]),
            Action(name="Work", effects=[{"attribute": "money", "change_value": 50}]),
        ]

    def get_characters_at_location(self, location):
        if location == "Park":
            return ["Alice"]
        elif location == "Market":
            return ["Bob"]
        else:
            return []


def test_strategy_manager_basic():
    """Test basic functionality of StrategyManager with real classes."""
    print("🧪 Testing StrategyManager with minimal real classes...")

    # Create StrategyManager with mocked dependencies
    strategy_manager = StrategyManager()

    # Replace the heavy graph_manager with our mock
    strategy_manager.graph_manager = MockGraphManager()

    # Create test events
    events = [
        MockEvent("new_day", "A new day begins"),
        MockEvent("interaction", "Emma meets Bob", ["Emma", "Bob"], "Market"),
        MockEvent("location_event", "Food festival", [], "Park"),
        MockEvent("global_event", "Village celebration"),
    ]

    print(f"Created {len(events)} test events")

    # Test get_affected_characters
    print("\n--- Testing get_affected_characters ---")
    for i, event in enumerate(events):
        affected = strategy_manager.get_affected_characters(event)
        print(f"Event {i+1} ({event.type}): {affected}")

        # Basic validation
        assert isinstance(affected, list), "Should return a list"
        assert len(affected) > 0, "Should affect at least one character"

    print("✅ get_affected_characters passed!")

    # Test update_strategy
    print("\n--- Testing update_strategy ---")
    try:
        plans = strategy_manager.update_strategy(events)

        print(f"Generated plans for {len(plans)} characters:")
        for char_name, char_plans in plans.items():
            print(f"  {char_name}: {len(char_plans)} plans")

        # Validate structure
        assert isinstance(plans, dict), "Plans should be a dictionary"

        # Should have plans for multiple characters
        assert (
            len(plans) >= 2
        ), f"Should have plans for multiple characters, got {len(plans)}"

        print("✅ update_strategy passed!")
        return True

    except Exception as e:
        print(f"❌ update_strategy failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_character_state_extraction():
    """Test character state extraction without heavy dependencies."""
    print("\n--- Testing character state extraction ---")

    # Create a minimal character object
    class MockCharacter:
        def __init__(self, name, hunger_level=50, energy=60, wealth_money=100):
            self.name = name
            self.hunger_level = hunger_level
            self.energy = energy
            self.wealth_money = wealth_money
            self.health_status = 80
            self.social_wellbeing = 70
            self.mental_health = 75

    strategy_manager = StrategyManager()
    character = MockCharacter("TestChar")

    # Test state extraction
    state_dict = strategy_manager.get_character_state_dict(character)
    print(f"Character state: {state_dict}")

    # Validate state structure
    assert isinstance(state_dict, dict), "State should be a dictionary"
    assert "hunger" in state_dict, "State should include hunger"
    assert "energy" in state_dict, "State should include energy"
    assert "money" in state_dict, "State should include money"

    print("✅ Character state extraction passed!")


def main():
    """Run the minimal integration tests."""
    print("🚀 Minimal StrategyManager Integration Test")
    print("=" * 50)

    try:
        # Run individual tests
        test_character_state_extraction()
        success = test_strategy_manager_basic()

        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ The refactored update_strategy method works with real classes!")
            return True
        else:
            print("\n❌ SOME TESTS FAILED!")
            return False

    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
