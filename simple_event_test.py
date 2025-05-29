#!/usr/bin/env python3
"""
Simple test to verify enhanced event handler functionality.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project directory to path
sys.path.insert(0, "/workspaces/tiny_village")


def test_event_creation():
    """Test basic event creation and functionality."""
    print("Testing Event creation...")

    try:
        from tiny_event_handler import Event

        # Create basic event
        event = Event(
            name="Test Event",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact=3,
        )

        print(f"✓ Basic event created: {event.name}")

        # Test recurring event
        recurring_event = Event(
            name="Daily Meeting",
            date=datetime.now(),
            event_type="social",
            importance=4,
            impact=2,
            recurrence_pattern={"type": "daily", "interval": 1},
        )

        print(f"✓ Recurring event created: {recurring_event.name}")
        print(f"  - Is recurring: {recurring_event.is_recurring()}")

        # Test complex event with effects
        complex_event = Event(
            name="Village Festival",
            date=datetime.now(),
            event_type="celebration",
            importance=9,
            impact=6,
            effects=[
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "happiness",
                    "change_value": 10,
                }
            ],
            cascading_events=[
                {"name": "festival_cleanup", "type": "work", "delay": 24}
            ],
            max_participants=50,
        )

        print(f"✓ Complex event created: {complex_event.name}")
        print(f"  - Has effects: {len(complex_event.effects)}")
        print(f"  - Has cascading events: {len(complex_event.cascading_events)}")

        return True

    except Exception as e:
        print(f"✗ Event creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_event_handler():
    """Test basic event handler functionality."""
    print("\nTesting EventHandler...")

    try:
        from tiny_event_handler import Event, EventHandler
        from unittest.mock import Mock

        # Create mock graph manager
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()

        # Create event handler
        handler = EventHandler(mock_graph_manager)
        print("✓ EventHandler created")

        # Create test event
        event = Event(
            name="Test Event",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact=3,
        )

        # Test adding event
        handler.add_event(event)
        print("✓ Event added to handler")

        # Test event templates
        templates = handler.get_event_templates()
        print(f"✓ Event templates loaded: {len(templates)} templates")

        # Test creating event from template
        festival = handler.create_event_from_template(
            "village_festival", "Summer Festival", datetime.now()
        )

        if festival:
            print(f"✓ Event created from template: {festival.name}")
        else:
            print("✗ Failed to create event from template")

        # Test event creation helpers
        holiday = handler.create_holiday_event("Test Holiday", datetime.now())
        market = handler.create_market_event(datetime.now())
        weather = handler.create_weather_event("storm", datetime.now(), 7)

        print(
            f"✓ Helper methods work: holiday={holiday.name}, market={market.name}, weather={weather.name}"
        )

        return True

    except Exception as e:
        print(f"✗ EventHandler test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_event_processing():
    """Test event processing functionality."""
    print("\nTesting Event Processing...")

    try:
        from tiny_event_handler import Event, EventHandler
        from unittest.mock import Mock

        # Create mock graph manager
        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()

        handler = EventHandler(mock_graph_manager)

        # Create event with effects
        event = Event(
            name="Happiness Event",
            date=datetime.now(),
            event_type="social",
            importance=5,
            impact=3,
            effects=[
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "happiness",
                    "change_value": 10,
                }
            ],
        )

        handler.add_event(event)

        # Test event processing
        results = handler.process_events()
        print(f"✓ Event processing completed")
        print(f"  - Processed: {len(results.get('processed_events', []))}")
        print(f"  - Failed: {len(results.get('failed_events', []))}")
        print(f"  - Cascading: {len(results.get('cascading_events', []))}")

        return True

    except Exception as e:
        print(f"✗ Event processing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_daily_events():
    """Test daily event checking."""
    print("\nTesting Daily Events...")

    try:
        from tiny_event_handler import EventHandler
        from unittest.mock import Mock

        mock_graph_manager = Mock()
        mock_graph_manager.G = Mock()
        mock_graph_manager.add_event_node = Mock()

        handler = EventHandler(mock_graph_manager)

        # Test daily events check
        daily_results = handler.check_daily_events()
        print(f"✓ Daily events check completed")
        print(f"  - Event type: {daily_results.get('event')}")
        print(f"  - Date: {daily_results.get('date')}")
        print(f"  - Total events: {daily_results.get('total_events', 0)}")

        return True

    except Exception as e:
        print(f"✗ Daily events test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Enhanced Event Handler - Simple Tests")
    print("=" * 50)

    tests = [
        test_event_creation,
        test_event_handler,
        test_event_processing,
        test_daily_events,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
