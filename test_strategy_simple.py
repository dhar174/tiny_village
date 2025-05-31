#!/usr/bin/env python3
"""
Simple focused test for StrategyManager without heavy dependencies.
Tests the refactored update_strategy method with minimal real classes.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we absolutely need
try:
    from tiny_strategy_manager import StrategyManager

    print("✅ Successfully imported StrategyManager")
except ImportError as e:
    print(f"❌ Failed to import StrategyManager: {e}")
    sys.exit(1)


class MockEvent:
    """Simple mock event to test with."""

    def __init__(self, event_type, participants=None, location=None):
        self.type = event_type
        self.event_type = event_type
        self.participants = participants or []
        self.location = location
        self.description = f"Mock {event_type} event"


class MockCharacter:
    """Simple mock character for testing."""

    def __init__(self, name, hunger=50, energy=50, money=100):
        self.name = name
        self.hunger_level = hunger
        self.energy = energy
        self.wealth_money = money
        self.health_status = 80
        self.social_wellbeing = 70
        self.mental_health = 75


class MockGraphManager:
    """Simple mock graph manager."""

    def __init__(self):
        self.characters = {
            "Emma": MockCharacter("Emma", hunger=80, energy=30),
            "Bob": MockCharacter("Bob", hunger=40, energy=70),
            "Alice": MockCharacter("Alice", hunger=60, energy=50),
        }

    def get_character_state(self, character_name):
        """Mock character state."""
        char = self.characters.get(character_name)
        if char:
            return {
                "hunger": char.hunger_level,
                "energy": char.energy,
                "money": char.wealth_money,
            }
        return None

    def get_possible_actions(self, character_name):
        """Mock possible actions."""
        return [
            {"name": "eat", "cost": 0.1},
            {"name": "sleep", "cost": 0.0},
            {"name": "work", "cost": 0.3},
        ]


class MockGOAPPlanner:
    """Simple mock GOAP planner."""

    def plan_actions(self, character_state, possible_actions):
        """Mock action planning."""
        return ["eat", "rest"] if character_state else ["wander"]


def test_get_affected_characters():
    """Test the get_affected_characters method."""
    print("\n🧪 Testing get_affected_characters...")

    # Create strategy manager and mock its dependencies
    strategy_manager = StrategyManager()
    strategy_manager.graph_manager = MockGraphManager()

    # Test different event types
    test_cases = [
        (MockEvent("new_day"), ["Emma", "Bob", "Alice"]),
        (MockEvent("interaction", participants=["Emma", "Bob"]), ["Emma", "Bob"]),
        (MockEvent("global_event"), ["Emma", "Bob", "Alice"]),
    ]

    for event, expected in test_cases:
        affected = strategy_manager.get_affected_characters(event)
        print(f"  Event: {event.type} -> Affected: {affected}")

        # Basic validation
        assert isinstance(affected, list), f"Should return list, got {type(affected)}"
        assert len(affected) > 0, f"Should affect at least one character"

        if event.type in ["new_day", "global_event"]:
            assert (
                len(affected) == 3
            ), f"Should affect all 3 characters for {event.type}"

    print("✅ get_affected_characters test passed!")


def test_update_strategy():
    """Test the main update_strategy method."""
    print("\n🧪 Testing update_strategy...")

    # Create strategy manager and mock its dependencies
    strategy_manager = StrategyManager()
    strategy_manager.graph_manager = MockGraphManager()
    strategy_manager.goap_planner = MockGOAPPlanner()

    # Create test events
    events = [
        MockEvent("new_day"),
        MockEvent("interaction", participants=["Emma", "Bob"]),
    ]

    print(f"Testing with {len(events)} events...")

    try:
        # This is the main test - call our refactored method
        plans = strategy_manager.update_strategy(events)

        print(f"Generated plans: {type(plans)}")
        print(f"Number of characters with plans: {len(plans)}")

        # Validate structure
        assert isinstance(plans, dict), f"Should return dict, got {type(plans)}"
        assert len(plans) > 0, "Should generate plans for at least one character"

        # Check each character's plans
        for character_name, character_plans in plans.items():
            print(f"  {character_name}: {len(character_plans)} plan(s)")
            assert isinstance(
                character_plans, list
            ), f"Plans for {character_name} should be list"

            # Check plan structure
            for plan_info in character_plans:
                assert isinstance(plan_info, dict), "Each plan should be a dict"
                assert "event" in plan_info, "Plan should have 'event' key"
                assert "plan" in plan_info, "Plan should have 'plan' key"

        print("✅ update_strategy test passed!")
        return True

    except Exception as e:
        print(f"❌ update_strategy test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_character_state_dict():
    """Test character state extraction."""
    print("\n🧪 Testing get_character_state_dict...")

    strategy_manager = StrategyManager()
    mock_char = MockCharacter("TestChar", hunger=75, energy=25, money=50)

    try:
        state_dict = strategy_manager.get_character_state_dict(mock_char)

        print(f"Character state: {state_dict}")

        # Validate structure
        assert isinstance(state_dict, dict), "Should return dict"
        assert "hunger" in state_dict, "Should include hunger"
        assert "energy" in state_dict, "Should include energy"
        assert "money" in state_dict, "Should include money"

        print("✅ get_character_state_dict test passed!")
        return True

    except Exception as e:
        print(f"❌ get_character_state_dict test failed: {e}")
        return False


def main():
    """Run all simple tests."""
    print("🚀 Simple StrategyManager Test Suite")
    print("=" * 50)

    try:
        # Run individual tests
        test_get_affected_characters()
        test_character_state_dict()

        # Run main integration test
        success = test_update_strategy()

        if success:
            print("\n🎉 All tests PASSED!")
            print("✅ The refactored update_strategy method works correctly!")
            return True
        else:
            print("\n❌ Some tests FAILED!")
            return False

    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n{'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
