import os
import sys
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tiny_event_handler import Event, EventHandler


class TestEventHandlerPayloads(unittest.TestCase):
    """Focused tests for mixed Event and event-like payload handling."""

    def setUp(self):
        self.mock_graph_manager = Mock()
        self.mock_graph_manager.G = Mock()
        self.mock_graph_manager.add_event_node = Mock()
        self.mock_graph_manager.get_node = Mock(return_value=None)
        self.mock_graph_manager.add_character_event_edge = Mock()
        self.mock_graph_manager.add_location_event_edge = Mock()

        self.event_handler = EventHandler(self.mock_graph_manager)
        self.event = Event(
            name="Structured Event",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact=3,
            action_system=Mock(),
        )

    def test_dict_payload_triggers_immediately(self):
        payload = {"event_type": "payload_trigger"}

        self.event_handler.add_event(payload)
        triggered_events = self.event_handler.check_events()

        self.assertIn(payload, triggered_events)
        self.assertEqual(self.event_handler._event_name(payload), "payload_trigger")

    def test_successful_payload_processing_removes_payload_and_wraps_context(self):
        participant = SimpleNamespace(happiness=0)
        location = SimpleNamespace(name="Town Square")
        payload = {
            "event_type": "payload_success",
            "participants": [participant],
            "location": location,
            "effects": [
                {
                    "type": "attribute_change",
                    "targets": ["participants"],
                    "attribute": "happiness",
                    "change_value": 2,
                }
            ],
        }

        self.event_handler.effect_dispatcher.apply_effect = Mock(return_value=True)
        self.event_handler.add_event(payload)

        results = self.event_handler.process_events()

        self.assertIn("payload_success", results["processed_events"])
        self.assertNotIn(payload, self.event_handler.events)
        self.assertEqual(len(self.event_handler.processed_events), 0)

        self.event_handler.effect_dispatcher.apply_effect.assert_called_once()
        _, payload_context = self.event_handler.effect_dispatcher.apply_effect.call_args.args
        self.assertEqual(payload_context.participants, [participant])
        self.assertIs(payload_context.location, location)

    def test_failed_payload_processing_removes_payload_from_queue(self):
        payload = {
            "event_type": "payload_failure",
            "effects": [{}],
        }

        self.event_handler.add_event(payload)
        results = self.event_handler.process_events()

        self.assertIn("payload_failure", results["failed_events"])
        self.assertNotIn(payload, self.event_handler.events)

    def test_event_only_helpers_ignore_payloads(self):
        payload = {"event_type": "payload_stats"}

        self.event_handler.add_event(self.event)
        self.event_handler.add_event(payload)

        self.assertEqual(self.event_handler.get_events_by_type("test"), [self.event])
        self.assertEqual(self.event_handler.get_events_by_location(None), [self.event])

        stats = self.event_handler.get_event_statistics()

        self.assertEqual(stats["total_events"], 2)
        self.assertEqual(stats["events_by_type"]["test"], 1)
        self.assertEqual(stats["average_importance"], self.event.importance)


if __name__ == "__main__":
    unittest.main()
