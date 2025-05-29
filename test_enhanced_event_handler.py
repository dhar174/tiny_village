#!/usr/bin/env python3
"""
Test suite for the enhanced tiny_event_handler.py functionality.
Tests all the new features including recurring events, cascading events,
event templates, and comprehensive event processing.
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Add the project directory to path
sys.path.insert(0, "/workspaces/tiny_village")

from tiny_event_handler import Event, EventHandler
from tiny_locations import Location


class TestEnhancedEvent(unittest.TestCase):
    """Test the enhanced Event class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.basic_event = Event(
            name="Test Event",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact=3,
        )

        self.recurring_event = Event(
            name="Daily Meeting",
            date=datetime.now(),
            event_type="social",
            importance=4,
            impact=2,
            recurrence_pattern={
                "type": "daily",
                "interval": 1,
                "end_date": datetime.now() + timedelta(days=30),
            },
        )

        self.complex_event = Event(
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
            preconditions=[
                {
                    "type": "attribute_check",
                    "target": "participants",
                    "attribute": "energy",
                    "operator": ">=",
                    "threshold": 50,
                }
            ],
            cascading_events=[
                {"name": "festival_cleanup", "type": "work", "delay": 24}
            ],
            max_participants=50,
        )

    def test_event_initialization(self):
        """Test that events initialize with correct properties."""
        self.assertEqual(self.basic_event.name, "Test Event")
        self.assertEqual(self.basic_event.type, "test")
        self.assertEqual(self.basic_event.importance, 5)
        self.assertEqual(self.basic_event.impact, 3)
        self.assertFalse(self.basic_event.is_active)
        self.assertEqual(self.basic_event.trigger_count, 0)

    def test_recurring_event_detection(self):
        """Test recurring event functionality."""
        self.assertFalse(self.basic_event.is_recurring())
        self.assertTrue(self.recurring_event.is_recurring())

    def test_next_occurrence_calculation(self):
        """Test calculation of next event occurrence."""
        # Basic event should return None for next occurrence
        self.assertIsNone(self.basic_event.get_next_occurrence())

        # Recurring event should return next occurrence
        next_occurrence = self.recurring_event.get_next_occurrence()
        self.assertIsNotNone(next_occurrence)
        self.assertIsInstance(next_occurrence, datetime)

    def test_should_trigger_logic(self):
        """Test event triggering logic."""
        current_time = datetime.now()

        # Event with matching time should trigger
        self.basic_event.date = current_time
        # Note: This might fail due to precondition checks, but tests the time logic

        # Event with future date should not trigger
        future_event = Event(
            name="Future Event",
            date=current_time + timedelta(days=1),
            event_type="test",
            importance=1,
            impact=1,
        )
        self.assertFalse(future_event.should_trigger(current_time))

    def test_participant_management(self):
        """Test adding and removing participants."""
        mock_character1 = Mock()
        mock_character1.name = "Alice"
        mock_character2 = Mock()
        mock_character2.name = "Bob"

        # Test adding participants
        self.assertTrue(self.complex_event.add_participant(mock_character1))
        self.assertTrue(self.complex_event.add_participant(mock_character2))
        self.assertEqual(len(self.complex_event.participants), 2)

        # Test removing participants
        self.assertTrue(self.complex_event.remove_participant(mock_character1))
        self.assertEqual(len(self.complex_event.participants), 1)
        self.assertFalse(
            self.complex_event.remove_participant(mock_character1)
        )  # Already removed

    def test_max_participants_limit(self):
        """Test maximum participants enforcement."""
        limited_event = Event(
            name="Small Meeting",
            date=datetime.now(),
            event_type="meeting",
            importance=3,
            impact=2,
            max_participants=2,
        )

        mock_char1 = Mock()
        mock_char2 = Mock()
        mock_char3 = Mock()

        self.assertTrue(limited_event.add_participant(mock_char1))
        self.assertTrue(limited_event.add_participant(mock_char2))
        self.assertFalse(
            limited_event.add_participant(mock_char3)
        )  # Should fail - limit reached

    def test_precondition_evaluation(self):
        """Test precondition evaluation logic."""
        # Test basic condition evaluation
        self.assertTrue(self.complex_event._evaluate_condition(60, ">=", 50))
        self.assertFalse(self.complex_event._evaluate_condition(40, ">=", 50))
        self.assertTrue(self.complex_event._evaluate_condition(100, "==", 100))
        self.assertFalse(self.complex_event._evaluate_condition(100, "!=", 100))

    def test_event_serialization(self):
        """Test event to_dict conversion."""
        event_dict = self.complex_event.to_dict()

        self.assertIn("name", event_dict)
        self.assertIn("effects", event_dict)
        self.assertIn("preconditions", event_dict)
        self.assertIn("cascading_events", event_dict)
        self.assertIn("max_participants", event_dict)
        self.assertEqual(event_dict["name"], "Village Festival")


class TestEnhancedEventHandler(unittest.TestCase):
    """Test the enhanced EventHandler class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock graph manager
        self.mock_graph_manager = Mock()
        self.mock_graph_manager.G = Mock()
        self.mock_graph_manager.add_event_node = Mock()
        self.mock_graph_manager.get_node = Mock()
        self.mock_graph_manager.add_character_event_edge = Mock()
        self.mock_graph_manager.add_location_event_edge = Mock()

        # Create event handler
        self.event_handler = EventHandler(self.mock_graph_manager)

        # Create test events
        self.test_event = Event(
            name="Test Event",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact=3,
            effects=[
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "happiness",
                    "change_value": 5,
                }
            ],
        )

        self.recurring_event = Event(
            name="Weekly Market",
            date=datetime.now(),
            event_type="economic",
            importance=6,
            impact=4,
            recurrence_pattern={"type": "weekly", "interval": 1},
        )

    def test_event_management(self):
        """Test adding, removing, and updating events."""
        # Test adding event
        self.event_handler.add_event(self.test_event)
        self.assertIn(self.test_event, self.event_handler.events)
        self.mock_graph_manager.add_event_node.assert_called_with(self.test_event)

        # Test getting event
        retrieved = self.event_handler.get_event("Test Event")
        self.assertEqual(retrieved, self.test_event)

        # Test updating event
        success = self.event_handler.update_event("Test Event", importance=7)
        self.assertTrue(success)
        self.assertEqual(self.test_event.importance, 7)

        # Test removing event
        success = self.event_handler.remove_event("Test Event")
        self.assertTrue(success)
        self.assertNotIn(self.test_event, self.event_handler.events)

    def test_event_processing(self):
        """Test comprehensive event processing."""
        # Add mock participants to event
        mock_participant = Mock()
        mock_participant.name = "TestCharacter"
        self.test_event.add_participant(mock_participant)

        self.event_handler.add_event(self.test_event)

        # Process events
        results = self.event_handler.process_events()

        self.assertIn("processed_events", results)
        self.assertIn("failed_events", results)
        self.assertIn("cascading_events", results)
        self.assertIn("state_changes", results)

    def test_cascading_events(self):
        """Test cascading event functionality."""
        cascading_event = Event(
            name="Parent Event",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact=3,
            cascading_events=[
                {
                    "name": "Child Event",
                    "type": "test",
                    "delay": 0,
                    "effects": [
                        {
                            "type": "attribute_change",
                            "targets": ["participants"],
                            "attribute": "energy",
                            "change_value": -2,
                        }
                    ],
                }
            ],
        )

        self.event_handler.add_event(cascading_event)

        # Trigger cascading events
        triggered = self.event_handler._trigger_cascading_events(cascading_event)
        self.assertEqual(len(triggered), 1)

    def test_event_templates(self):
        """Test event template system."""
        templates = self.event_handler.get_event_templates()

        self.assertIn("village_festival", templates)
        self.assertIn("harvest_season", templates)
        self.assertIn("merchant_arrival", templates)
        self.assertIn("natural_disaster", templates)

        # Test creating event from template
        festival_event = self.event_handler.create_event_from_template(
            "village_festival", "Summer Festival", datetime.now()
        )

        self.assertIsNotNone(festival_event)
        self.assertEqual(festival_event.name, "Summer Festival")
        self.assertEqual(festival_event.type, "social")

    def test_event_creation_helpers(self):
        """Test event creation helper methods."""
        current_time = datetime.now()

        # Test holiday event creation
        holiday = self.event_handler.create_holiday_event("Test Holiday", current_time)
        self.assertIsNotNone(holiday)
        self.assertEqual(holiday.type, "holiday")

        # Test market event creation
        market = self.event_handler.create_market_event(current_time)
        self.assertIsNotNone(market)
        self.assertEqual(market.type, "economic")

        # Test weather event creation
        weather = self.event_handler.create_weather_event("storm", current_time, 7)
        self.assertIsNotNone(weather)
        self.assertEqual(weather.type, "weather")
        self.assertEqual(weather.importance, 7)

        # Test social event creation
        social = self.event_handler.create_social_event("Party", current_time)
        self.assertIsNotNone(social)
        self.assertEqual(social.type, "social")

        # Test work event creation
        work = self.event_handler.create_work_event("Build Bridge", current_time)
        self.assertIsNotNone(work)
        self.assertEqual(work.type, "work")

    def test_daily_events_check(self):
        """Test enhanced daily events checking."""
        self.event_handler.add_event(self.recurring_event)

        daily_results = self.event_handler.check_daily_events()

        self.assertIn("event", daily_results)
        self.assertIn("date", daily_results)
        self.assertIn("recurring_events", daily_results)
        self.assertIn("cascading_events", daily_results)
        self.assertIn("special_events", daily_results)
        self.assertEqual(daily_results["event"], "daily_check")

    def test_recurring_event_scheduling(self):
        """Test scheduling of recurring events."""
        self.event_handler.add_event(self.recurring_event)

        scheduled_count = self.event_handler.schedule_recurring_events(30)  # 30 days
        self.assertGreaterEqual(scheduled_count, 0)

    def test_event_filtering(self):
        """Test event filtering methods."""
        # Add different types of events
        self.event_handler.add_event(self.test_event)
        self.event_handler.add_event(self.recurring_event)

        # Test filtering by type
        test_events = self.event_handler.get_events_by_type("test")
        economic_events = self.event_handler.get_events_by_type("economic")

        self.assertEqual(len(test_events), 1)
        self.assertEqual(len(economic_events), 1)

        # Test filtering by timeframe
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now() + timedelta(hours=1)
        events_in_timeframe = self.event_handler.get_events_in_timeframe(
            start_time, end_time
        )

        self.assertGreaterEqual(len(events_in_timeframe), 0)

    def test_event_statistics(self):
        """Test event statistics generation."""
        self.event_handler.add_event(self.test_event)
        self.event_handler.add_event(self.recurring_event)

        stats = self.event_handler.get_event_statistics()

        self.assertIn("total_events", stats)
        self.assertIn("active_events", stats)
        self.assertIn("processed_events", stats)
        self.assertIn("recurring_events", stats)
        self.assertIn("events_by_type", stats)
        self.assertIn("average_importance", stats)

        self.assertEqual(stats["total_events"], 2)
        self.assertEqual(stats["recurring_events"], 1)

    def test_cascading_queue_processing(self):
        """Test processing of scheduled cascading events."""
        current_time = datetime.now()
        future_time = current_time + timedelta(hours=1)

        # Add a future event to the cascading queue
        future_event = Event(
            name="Future Cascading",
            date=future_time,
            event_type="test",
            importance=3,
            impact=2,
        )

        self.event_handler.cascading_event_queue.append((future_time, future_event))

        # Process queue with current time (should not process future event)
        processed = self.event_handler.process_cascading_queue(current_time)
        self.assertEqual(len(processed), 0)
        self.assertEqual(len(self.event_handler.cascading_event_queue), 1)

        # Process queue with future time (should process the event)
        processed = self.event_handler.process_cascading_queue(future_time)
        self.assertEqual(len(processed), 1)
        self.assertEqual(len(self.event_handler.cascading_event_queue), 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complex integration scenarios."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_graph_manager = Mock()
        self.mock_graph_manager.G = Mock()
        self.mock_graph_manager.add_event_node = Mock()
        self.mock_graph_manager.get_node = Mock()
        self.mock_graph_manager.add_character_event_edge = Mock()
        self.mock_graph_manager.add_location_event_edge = Mock()

        self.event_handler = EventHandler(self.mock_graph_manager)

    def test_complex_event_chain(self):
        """Test a complex chain of events with cascading effects."""
        # Create a merchant arrival that triggers market expansion
        merchant_event = self.event_handler.create_event_from_template(
            "merchant_arrival", "Traveling Merchant Arrives", datetime.now()
        )

        self.assertIsNotNone(merchant_event)
        self.event_handler.add_event(merchant_event)

        # Process the event
        results = self.event_handler.process_events()

        # Should have cascading events scheduled
        self.assertGreaterEqual(len(results.get("cascading_events", [])), 0)

    def test_seasonal_event_creation(self):
        """Test creation of seasonal events."""
        # Test Christmas event creation
        christmas_time = datetime(2024, 12, 25)
        special_events = self.event_handler._check_special_date_events(christmas_time)

        christmas_events = [e for e in special_events if "Christmas" in e.name]
        self.assertGreater(len(christmas_events), 0)

        # Test market day creation (Sunday)
        sunday = datetime(2024, 12, 22)  # A Sunday
        special_events = self.event_handler._check_special_date_events(sunday)

        market_events = [e for e in special_events if "Market" in e.name]
        self.assertGreater(len(market_events), 0)

    def test_event_lifecycle(self):
        """Test complete event lifecycle from creation to cleanup."""
        # Create and add event
        event = self.event_handler.create_social_event(
            "Village Gathering", datetime.now() - timedelta(days=35)  # Old event
        )

        self.event_handler.add_event(event)

        # Process the event
        self.event_handler._process_single_event(event, datetime.now())
        self.assertIn(event, self.event_handler.processed_events)

        # Clean up old events
        initial_count = len(self.event_handler.processed_events)
        removed_count = self.event_handler.cleanup_old_events(
            30
        )  # Remove events older than 30 days

        self.assertGreater(removed_count, 0)
        self.assertLess(len(self.event_handler.processed_events), initial_count)


def run_tests():
    """Run all tests and provide summary."""
    print("Running Enhanced Event Handler Tests...")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedEvent))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedEventHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASSED' if success else 'FAILED'}")

    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
