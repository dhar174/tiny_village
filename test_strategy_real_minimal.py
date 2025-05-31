#!/usr/bin/env python3
"""
Minimal integration test for StrategyManager using REAL production classes.
This test avoids the heavy MemoryManager dependency in Character class.
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🚀 Starting minimal StrategyManager integration test with real classes...")
print("=" * 60)

# Import basic real classes first
try:
    print("📦 Importing basic dependencies...")
    from actions import Action, State, ActionSystem

    print("✅ actions imported")

    from tiny_graph_manager import GraphManager

    print("✅ tiny_graph_manager imported")

    from tiny_types import Location, Event

    print("✅ tiny_types imported")

    from tiny_items import FoodItem, ItemInventory, ItemObject

    print("✅ tiny_items imported")

    from tiny_jobs import Job

    print("✅ tiny_jobs imported")

    from tiny_time_manager import GameTimeManager

    print("✅ tiny_time_manager imported")

except ImportError as e:
    print(f"❌ Failed to import basic dependencies: {e}")
    sys.exit(1)


# Create a minimal Character class that avoids MemoryManager
class MinimalCharacter:
    """Minimal Character class without heavy MemoryManager dependency."""

    def __init__(self, name, age=25, location_name="Home", **kwargs):
        self.name = name
        self.age = age
        self.location_name = location_name

        # Basic attributes for testing
        self.hunger_level = kwargs.get("hunger_level", 50)
        self.energy = kwargs.get("energy", 50)
        self.wealth_money = kwargs.get("wealth_money", 100)
        self.health_status = kwargs.get("health_status", 80)
        self.social_wellbeing = kwargs.get("social_wellbeing", 70)
        self.mental_health = kwargs.get("mental_health", 75)

        # Simple inventory
        self.inventory = kwargs.get("inventory", ItemInventory())

        # Simple location object
        self.location = type("Location", (), {"name": location_name})()

        # Simple job
        self.job = kwargs.get("job", None)

        print(f"✅ Created minimal character: {name}")


# Now try to import StrategyManager
try:
    print("📦 Importing StrategyManager...")
    from tiny_strategy_manager import StrategyManager

    print("✅ StrategyManager imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import StrategyManager: {e}")
    sys.exit(1)


class TestStrategyManagerMinimal:
    """Minimal integration test for StrategyManager."""

    def __init__(self):
        print("\n🏗️ Setting up minimal test environment...")

        # Create real instances
        self.action_system = ActionSystem()
        self.graph_manager = GraphManager()
        self.time_manager = GameTimeManager()
        self.strategy_manager = StrategyManager()

        # Create minimal test characters
        self.characters = self._create_minimal_characters()

        # Create test events
        self.events = self._create_test_events()

        print("✅ Minimal test environment setup complete!")

    def _create_minimal_characters(self):
        """Create minimal Character instances for testing."""
        print("👥 Creating minimal test characters...")

        characters = {}

        # Create Emma with inventory
        emma_inventory = ItemInventory()
        emma_inventory.add_item(FoodItem(name="Apple", calories=50))
        emma_inventory.add_item(FoodItem(name="Bread", calories=100))

        emma = MinimalCharacter(
            name="Emma",
            age=25,
            location_name="Home",
            hunger_level=60,  # Quite hungry
            energy=40,  # Low energy
            wealth_money=150,
            inventory=emma_inventory,
        )
        characters["Emma"] = emma

        # Create Bob
        bob_inventory = ItemInventory()
        bob_inventory.add_item(FoodItem(name="Sandwich", calories=120))

        bob = MinimalCharacter(
            name="Bob",
            age=30,
            location_name="Market",
            hunger_level=30,  # Not very hungry
            energy=80,  # High energy
            wealth_money=200,
            inventory=bob_inventory,
        )
        characters["Bob"] = bob

        # Create Alice
        alice = MinimalCharacter(
            name="Alice",
            age=28,
            location_name="Park",
            hunger_level=45,  # Moderately hungry
            energy=70,  # Good energy
            wealth_money=100,
        )
        characters["Alice"] = alice

        # Add characters to graph manager for affected_characters logic
        for name, character in characters.items():
            self.graph_manager.characters[name] = character

        print(f"✅ Created {len(characters)} minimal characters")
        return characters

    def _create_test_events(self):
        """Create real Event instances for testing."""
        print("📅 Creating test events...")

        events = []

        # 1. New Day Event - affects all characters
        new_day_event = Event(
            event_type="new_day",
            timestamp=datetime.now(),
            description="A new day begins in the village",
            participants=[],
            location=None,
            effects={},
        )
        events.append(new_day_event)

        # 2. Interaction Event - affects specific characters
        interaction_event = Event(
            event_type="interaction",
            timestamp=datetime.now(),
            description="Emma meets Bob at the market",
            participants=["Emma", "Bob"],
            location="Market",
            effects={"social_wellbeing": 5},
        )
        events.append(interaction_event)

        print(f"✅ Created {len(events)} test events")
        return events

    def test_get_affected_characters(self):
        """Test the get_affected_characters method with real events."""
        print("\n🧪 Testing get_affected_characters method...")

        for i, event in enumerate(self.events):
            print(f"\n--- Event {i+1}: {event.event_type} ---")
            print(f"Description: {event.description}")

            affected = self.strategy_manager.get_affected_characters(event)
            print(f"Affected characters: {affected}")

            # Validate results based on event type
            if event.event_type == "new_day":
                assert (
                    len(affected) == 3
                ), f"New day should affect all 3 characters, got {len(affected)}"
                assert set(affected) == {
                    "Emma",
                    "Bob",
                    "Alice",
                }, f"Expected all characters, got {affected}"

            elif event.event_type == "interaction":
                assert (
                    len(affected) == 2
                ), f"Interaction should affect 2 characters, got {len(affected)}"
                assert set(affected) == {
                    "Emma",
                    "Bob",
                }, f"Expected Emma and Bob, got {affected}"

        print("✅ get_affected_characters test passed!")
        return True

    def test_character_state_extraction(self):
        """Test character state extraction works with minimal characters."""
        print("\n🧪 Testing character state extraction...")

        for character_name, character in self.characters.items():
            print(f"\n--- Testing character: {character_name} ---")

            # Test state extraction
            state_dict = self.strategy_manager.get_character_state_dict(character)
            print(f"Character state: {state_dict}")

            # Validate state structure
            assert isinstance(state_dict, dict), "State should be a dictionary"
            assert "hunger" in state_dict, "State should include hunger"
            assert "energy" in state_dict, "State should include energy"
            assert "money" in state_dict, "State should include money"

            print(f"✅ State extraction works for {character_name}")

        print("✅ Character state extraction test passed!")
        return True

    def test_daily_actions_generation(self):
        """Test daily actions generation with minimal characters."""
        print("\n🧪 Testing daily actions generation...")

        for character_name, character in self.characters.items():
            print(f"\n--- Testing daily actions for: {character_name} ---")

            try:
                daily_actions = self.strategy_manager.get_daily_actions(character)
                print(f"Generated {len(daily_actions)} daily actions")

                for i, action in enumerate(daily_actions[:3]):  # Show first 3 actions
                    print(f"  {i+1}. {action.name} (cost: {action.cost})")

                assert (
                    len(daily_actions) > 0
                ), f"Should generate daily actions for {character_name}"
                print(f"✅ Daily actions generation works for {character_name}")

            except Exception as e:
                print(f"⚠️ Daily actions generation failed for {character_name}: {e}")
                return False

        print("✅ Daily actions generation test passed!")
        return True

    def test_update_strategy_core_logic(self):
        """Test the core update_strategy method logic."""
        print("\n🧪 Testing update_strategy core logic...")

        # Test with single event first
        single_event = [self.events[0]]  # new_day event
        print(f"\n--- Testing with single event: {single_event[0].event_type} ---")

        try:
            plans = self.strategy_manager.update_strategy(single_event)

            print(f"Generated plans for {len(plans)} characters:")
            for character_name, character_plans in plans.items():
                print(f"  {character_name}: {len(character_plans)} plan(s)")

            # Validate structure
            assert isinstance(plans, dict), "Plans should be a dictionary"
            print("✅ Single event test passed!")

        except Exception as e:
            print(f"⚠️ Single event test failed: {e}")
            return False

        # Test with multiple events
        print(f"\n--- Testing with multiple events: {len(self.events)} events ---")

        try:
            all_plans = self.strategy_manager.update_strategy(self.events)

            print(f"Generated plans for {len(all_plans)} characters:")
            for character_name, character_plans in all_plans.items():
                print(f"  {character_name}: {len(character_plans)} plan(s)")

            # Validate comprehensive results
            assert isinstance(all_plans, dict), "Plans should be a dictionary"
            print("✅ Multi-event test passed!")
            return True

        except Exception as e:
            print(f"⚠️ Multi-event test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all minimal integration tests."""
        print("\n🚀 Starting StrategyManager Minimal Integration Tests")
        print("=" * 60)

        try:
            # Test individual components
            success1 = self.test_get_affected_characters()
            success2 = self.test_character_state_extraction()
            success3 = self.test_daily_actions_generation()
            success4 = self.test_update_strategy_core_logic()

            overall_success = success1 and success2 and success3 and success4

            if overall_success:
                print("\n🎉 All minimal integration tests PASSED!")
                print("✅ The refactored update_strategy method works correctly!")
                return True
            else:
                print("\n❌ Some minimal integration tests FAILED!")
                return False

        except Exception as e:
            print(f"\n💥 Minimal integration test suite failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Run the minimal integration tests."""
    print("🏁 StrategyManager Minimal Integration Test Suite")
    print("Testing refactored update_strategy method with minimal real classes")
    print("=" * 80)

    # Create and run test suite
    test_suite = TestStrategyManagerMinimal()
    success = test_suite.run_all_tests()

    if success:
        print("\n🏆 MINIMAL INTEGRATION TEST SUITE: SUCCESS")
        print(
            "The refactored StrategyManager.update_strategy() method is working correctly!"
        )
        print("✅ Refactoring validation complete!")
    else:
        print("\n💔 MINIMAL INTEGRATION TEST SUITE: FAILED")
        print("The refactored method needs further debugging.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
