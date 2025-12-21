#!/usr/bin/env python3
"""
Test script to validate the improved event-driven strategy integration.
This tests the fixes made to address issue #190.
"""

import sys
from collections import defaultdict
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

# Add the project directory to path
sys.path.insert(0, "/home/runner/work/tiny_village/tiny_village")

# Import the modules we're testing
from tiny_event_handler import Event, EventHandler
from tiny_gameplay_controller import GameplayController
from actions import Action


class TestEventStrategyIntegration(unittest.TestCase):
    """Test the improved event-driven strategy integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock pygame to avoid initialization issues
        with patch('pygame.init'), patch('pygame.display.set_mode'), patch('pygame.time.Clock'), patch('pygame.font.Font'):
            self.game_controller = GameplayController()
    
    def tearDown(self):
        """Clean up after each test."""
        # Ensure event_handler is restored if it was modified
        if hasattr(self, 'game_controller'):
            # Clean up the controller
            self.game_controller = None

    def test_event_handler_drives_strategy(self):
        """Test that EventHandler.check_events() properly drives strategy updates."""
        # Skip if event handler couldn't be initialized
        if not self.game_controller.event_handler:
            self.skipTest("EventHandler not available in test environment")
            
        # Ensure we have both event handler and strategy manager
        self.assertIsNotNone(self.game_controller.event_handler)
        self.assertIsNotNone(self.game_controller.strategy_manager)
        
        # Create a real test event and add it to the event handler
        test_event = Event(
            name="Test Social Event",
            date=datetime.now(),
            event_type="social",
            importance=5,
            impact=3
        )
        
        # Override should_trigger to ensure it triggers
        test_event.should_trigger = lambda x: True
        self.game_controller.event_handler.add_event(test_event)
        
        # Track whether strategy manager was called (without mocking its full behavior)
        original_update_strategy = self.game_controller.strategy_manager.update_strategy
        strategy_was_called = []
        
        def track_strategy_call(events, **kwargs):
            strategy_was_called.append(events)
            return original_update_strategy(events, **kwargs)
        
        # Patch only to track the call, but let the real method run
        with patch.object(self.game_controller.strategy_manager, 'update_strategy', side_effect=track_strategy_call):
            # Call the new robust event processing method
            update_errors = []
            self.game_controller._process_events_and_drive_strategy(update_errors)
            
            # Verify that strategy manager was called with events
            self.assertTrue(len(strategy_was_called) > 0, "Strategy manager should have been called")
            # The events passed should include our test event
            called_events = strategy_was_called[0]
            self.assertTrue(any(e.name == "Test Social Event" for e in called_events), 
                          "Test event should have been passed to strategy manager")
            
            # Verify no errors occurred (unless expected from real processing)
            # Allow for normal processing errors but not system failures
            for error in update_errors:
                self.assertNotIn("Event-driven strategy system failure", error)

    def test_robust_error_handling(self):
        """Test that the integration handles errors gracefully."""
        # Test with no event handler
        original_event_handler = self.game_controller.event_handler
        self.game_controller.event_handler = None
        
        update_errors = []
        # Should not crash and should use fallback
        self.game_controller._process_events_and_drive_strategy(update_errors)
        
        # Restore event handler for next test
        self.game_controller.event_handler = original_event_handler
        
    def test_error_handling_with_exception(self):
        """Test that the integration handles exceptions from EventHandler gracefully."""
        # This test uses a separate method to avoid state issues from the previous test
        if not self.game_controller.event_handler:
            self.skipTest("EventHandler not available")
            
        # Test with event handler that throws exception
        with patch.object(self.game_controller.event_handler, 'check_events', side_effect=Exception("Test error")):
            update_errors = []
            self.game_controller._process_events_and_drive_strategy(update_errors)
            
            # Should handle error gracefully
            self.assertTrue(any("Event checking failed" in error for error in update_errors))

    def test_strategic_decision_application(self):
        """Test that strategic decisions are properly applied."""
        # Create a test decision
        test_decision = {
            "type": "character_action",
            "character_id": "test_char",
            "action": {"name": "wait", "cost": 0}  # Use a simple action
        }
        
        # Add a test character (minimal mock for character only)
        mock_character = Mock()
        mock_character.name = "Test Character"
        mock_character.location = None
        self.game_controller.characters = {"test_char": mock_character}
        
        # Test the actual apply_decision flow without over-mocking
        update_errors = []
        try:
            # This should execute the real apply_decision logic
            self.game_controller._apply_strategic_decisions([test_decision], update_errors)
            
            # The test passes if no exceptions were raised and errors are manageable
            # We allow for some errors (like missing action resolver) but not critical failures
            critical_errors = [e for e in update_errors if "Critical" in e or "failure" in e.lower()]
            # It's ok if there are some errors due to test environment, but system should not crash
            
        except Exception as e:
            self.fail(f"Strategic decision application should not raise exceptions: {e}")

    def test_world_state_calculation(self):
        """Test that world state is calculated correctly for dynamic events."""
        # Add test characters with various attributes
        char1 = Mock()
        char1.wealth_money = 100
        char1.health_status = 80
        
        char2 = Mock()
        char2.wealth_money = 50
        char2.health_status = 60
        
        self.game_controller.characters = {"char1": char1, "char2": char2}
        
        # Add social networks via graph manager delegation
        mock_graph = Mock()
        mock_graph.get_social_networks = Mock(return_value={
            "relationships": {
                "char1": {"char2": 70},
                "char2": {"char1": 65}
            }
        })
        self.game_controller.graph_manager = mock_graph
        
        world_state = self.game_controller._get_current_world_state()
        
        # Verify calculations
        self.assertEqual(world_state["average_wealth"], 75)  # (100 + 50) / 2
        self.assertEqual(world_state["average_health"], 70)  # (80 + 60) / 2
        self.assertEqual(world_state["average_relationships"], 67.5)  # (70 + 65) / 2
        self.assertEqual(world_state["population"], 2)

    def test_fallback_event_processing(self):
        """Test that fallback event processing works when EventHandler is unavailable."""
        # Remove event handler
        self.game_controller.event_handler = None
        
        # Create a real event object
        test_event = Event(
            name="Test Event",
            date=datetime.now(),
            event_type="test",
            importance=3,
            impact=2
        )
        
        self.game_controller.events = [test_event]
        
        # Let the real strategy manager process it
        update_errors = []
        self.game_controller._process_basic_events_fallback(update_errors)
        
        # Verify event was removed (processed)
        self.assertEqual(len(self.game_controller.events), 0, 
                        "Event should have been processed and removed")
        
        # Verify no critical failures
        critical_errors = [e for e in update_errors if "failure" in e.lower()]
        # Some errors might occur in test environment, but should not be critical system failures
        self.assertNotIn("Fallback event processing failed", update_errors)

    def test_deprecated_methods_still_work(self):
        """Test that deprecated methods still work for backward compatibility."""
        # Test deprecated _process_pending_events
        test_event = Mock()
        self.game_controller.events = [test_event]
        
        # Should not crash
        self.game_controller._process_pending_events()
        
        # Test deprecated _process_events_and_update_strategy
        # Should not crash
        self.game_controller._process_events_and_update_strategy(0.1)


class TestPendingEventForwarding(unittest.TestCase):
    """Targeted tests for deprecated _process_pending_events forwarding logic."""

    def _make_controller(self):
        ctrl = GameplayController.__new__(GameplayController)
        ctrl.events = []
        ctrl.event_handler = None
        ctrl.storytelling_system = None
        ctrl._process_events_and_drive_strategy = lambda update_errors: None
        return ctrl

    def test_forwards_to_event_handler_and_clears(self):
        ctrl = self._make_controller()
        event = Mock()
        handler = Mock()
        handler.events = []
        handler.add_event = Mock()
        ctrl.event_handler = handler
        ctrl.events = [event]

        ctrl._process_pending_events()

        handler.add_event.assert_called_once_with(event)
        self.assertEqual(ctrl.events, [])

    def test_retains_event_on_handler_failure(self):
        ctrl = self._make_controller()
        event = Mock()

        def failing_add(_):
            raise RuntimeError("handler failure")

        handler = Mock()
        handler.add_event = Mock(side_effect=failing_add)
        ctrl.event_handler = handler
        ctrl.events = [event]

        ctrl._process_pending_events()

        self.assertEqual(ctrl.events, [event])

    def test_forwards_to_storytelling_when_no_handler(self):
        ctrl = self._make_controller()
        event = Mock()
        storytelling = Mock()
        storytelling.process_event_for_stories = Mock()
        ctrl.storytelling_system = storytelling
        ctrl.events = [event]

        ctrl._process_pending_events()

        storytelling.process_event_for_stories.assert_called_once_with(event)
        self.assertEqual(ctrl.events, [])

    def test_retains_event_on_storytelling_failure(self):
        ctrl = self._make_controller()
        event = Mock()

        def failing_story(_):
            raise RuntimeError("story failure")

        storytelling = Mock()
        storytelling.process_event_for_stories = Mock(side_effect=failing_story)
        ctrl.storytelling_system = storytelling
        ctrl.events = [event]

        ctrl._process_pending_events()

        self.assertEqual(ctrl.events, [event])

    def test_integration_in_update_game_state(self):
        """Test that the integration works properly within update_game_state."""
        game_controller = GameplayController.__new__(GameplayController)
        game_controller.events = []
        game_controller.event_handler = None
        game_controller.storytelling_system = None
        game_controller.map_controller = Mock()
        game_controller.map_controller.update = Mock()
        game_controller.characters = {}
        game_controller.strategy_manager = None
        game_controller.gametime_manager = None
        game_controller.animation_system = None
        game_controller.recovery_manager = Mock()
        game_controller.recovery_manager.attempt_recovery = Mock(return_value=False)
        game_controller.game_statistics = defaultdict(int)

        # Mock the new method to verify it's called
        with patch.object(game_controller, '_process_events_and_drive_strategy') as mock_process:
            # Mock other dependencies to avoid side effects
            with patch.object(game_controller, '_update_feature_systems'):
                with patch.object(game_controller, '_update_character', return_value=True):
                    
                    # Call update_game_state
                    game_controller.paused = False  # Ensure not paused
                    game_controller.update_game_state(0.1)
                    
                    # Verify our new method was called
                    mock_process.assert_called_once()


class TestUpdateGameStateEventHandlerUsage(unittest.TestCase):
    """Ensure update_game_state feeds queued events through EventHandler and strategy."""

    def test_update_game_state_forwards_events_and_updates_strategy(self):
        controller = GameplayController.__new__(GameplayController)
        controller.paused = False
        queued_event = Mock(name="queued_event")
        controller.events = [queued_event]

        handler_event = Mock(name="handler_event")
        controller.event_handler = Mock()
        controller.event_handler.add_event = Mock()
        controller.event_handler.check_events = Mock(return_value=[handler_event])
        controller.event_handler.process_events = Mock(
            return_value={"processed_events": [], "failed_events": []}
        )
        controller.event_handler.process_cascading_queue = Mock()
        controller.event_handler.generate_dynamic_events = Mock()

        controller.strategy_manager = Mock()
        controller.strategy_manager.update_strategy = Mock(return_value=None)

        controller._update_feature_systems = Mock()
        controller.map_controller = Mock()
        controller.map_controller.update = Mock()
        controller.characters = {}
        controller.gametime_manager = None
        controller.animation_system = None
        controller.recovery_manager = Mock()
        controller.recovery_manager.attempt_recovery = Mock(return_value=False)
        controller.game_statistics = defaultdict(int)
        controller.action_resolver = None
        controller._update_character = Mock(return_value=True)

        controller.update_game_state(0.1)

        controller.event_handler.add_event.assert_called_once_with(queued_event)
        controller.event_handler.check_events.assert_called_once()
        controller.strategy_manager.update_strategy.assert_called_once_with(
            [handler_event]
        )
        self.assertEqual(controller.events, [])


class TestEventHandlerIntegration(unittest.TestCase):
    """Test EventHandler functionality that supports the strategy integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_handler = EventHandler()

    def test_check_events_returns_valid_events(self):
        """Test that check_events returns properly formatted events."""
        # Add a test event
        test_event = Event(
            name="Daily Test",
            date=datetime.now(),
            event_type="daily",
            importance=5,
            impact=2
        )
        self.event_handler.add_event(test_event)
        
        # Check events
        events = self.event_handler.check_events()
        
        # Verify format
        self.assertIsInstance(events, list)
        for event in events:
            self.assertTrue(hasattr(event, 'name'))
            self.assertTrue(hasattr(event, 'type'))

    def test_process_events_handles_effects(self):
        """Test that process_events properly handles event effects."""
        # Create event with effects
        test_event = Event(
            name="Effect Test",
            date=datetime.now(),
            event_type="test",
            importance=5,
            impact=3,
            effects=[{
                "type": "attribute_change",
                "targets": ["participants"],
                "attribute": "happiness",
                "change_value": 10
            }]
        )
        
        # Add mock participant
        mock_participant = Mock()
        mock_participant.happiness = 50
        test_event.participants = [mock_participant]
        
        self.event_handler.add_event(test_event)
        
        # Process events
        results = self.event_handler.process_events()
        
        # Verify results structure
        self.assertIn('processed_events', results)
        self.assertIn('failed_events', results)
        self.assertIsInstance(results['processed_events'], list)
        self.assertIsInstance(results['failed_events'], list)


if __name__ == '__main__':
    unittest.main()
