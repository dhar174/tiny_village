#!/usr/bin/env python3
"""
Test script to validate the improved event-driven strategy integration.
This tests the fixes made to address issue #190.
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

# Add the project directory to path
sys.path.insert(0, "/home/runner/work/tiny_village/tiny_village")

# Import the modules we're testing
from tiny_event_handler import Event, EventHandler
from tiny_strategy_manager import StrategyManager
from tiny_gameplay_controller import GameplayController


class TestEventStrategyIntegration(unittest.TestCase):
    """Test the improved event-driven strategy integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock pygame to avoid initialization issues
        with patch('pygame.init'), patch('pygame.display.set_mode'), patch('pygame.time.Clock'):
            self.game_controller = GameplayController()

    def test_event_handler_drives_strategy(self):
        """Test that EventHandler.check_events() properly drives strategy updates."""
        # Ensure we have both event handler and strategy manager
        self.assertIsNotNone(self.game_controller.event_handler)
        self.assertIsNotNone(self.game_controller.strategy_manager)
        
        # Mock the event handler to return test events
        test_event = Event(
            name="Test Social Event",
            date=datetime.now(),
            event_type="social",
            importance=5,
            impact=3
        )
        
        with patch.object(self.game_controller.event_handler, 'check_events', return_value=[test_event]):
            with patch.object(self.game_controller.event_handler, 'process_events', return_value={'processed_events': ['Test Social Event'], 'failed_events': []}):
                with patch.object(self.game_controller.strategy_manager, 'update_strategy', return_value=[]) as mock_update_strategy:
                    
                    # Call the new robust event processing method
                    update_errors = []
                    self.game_controller._process_events_and_drive_strategy(update_errors)
                    
                    # Verify that strategy manager was called with the events
                    mock_update_strategy.assert_called_once_with([test_event])
                    
                    # Verify no errors occurred
                    self.assertEqual(len(update_errors), 0, f"Should have no errors, but got: {update_errors}")

    def test_robust_error_handling(self):
        """Test that the integration handles errors gracefully."""
        # Test with no event handler
        original_event_handler = self.game_controller.event_handler
        self.game_controller.event_handler = None
        
        update_errors = []
        # Should not crash and should use fallback
        self.game_controller._process_events_and_drive_strategy(update_errors)
        
        # Restore event handler
        self.game_controller.event_handler = original_event_handler
        
        # Test with event handler that throws exception
        with patch.object(self.game_controller.event_handler, 'check_events', side_effect=Exception("Test error")):
            update_errors = []
            self.game_controller._process_events_and_drive_strategy(update_errors)
            
            # Should handle error gracefully
            self.assertTrue(any("Event checking failed" in error for error in update_errors))

    def test_strategic_decision_application(self):
        """Test that strategic decisions are properly applied."""
        # Mock a strategic decision
        test_decision = {
            "type": "character_action",
            "character_id": "test_char",
            "action": {"name": "Test Action", "cost": 1}
        }
        
        # Add a test character
        mock_character = Mock()
        mock_character.name = "Test Character"
        self.game_controller.characters = {"test_char": mock_character}
        
        # Mock action resolver
        mock_action = Mock()
        mock_action.execute.return_value = True
        mock_action.name = "Test Action"
        
        with patch.object(self.game_controller.action_resolver, 'resolve_action', return_value=mock_action):
            update_errors = []
            self.game_controller._apply_strategic_decisions([test_decision], update_errors)
            
            # Verify action was executed
            mock_action.execute.assert_called_once()
            
            # Verify no errors
            self.assertEqual(len(update_errors), 0)

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
        
        # Add social networks
        self.game_controller.social_networks = {
            "relationships": {
                "char1": {"char2": 70},
                "char2": {"char1": 65}
            }
        }
        
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
        
        # Add basic events
        test_event = Mock()
        test_event.type = "test"
        test_event.name = "Test Event"
        test_event.importance = 3
        
        self.game_controller.events = [test_event]
        
        # Mock strategy manager
        with patch.object(self.game_controller.strategy_manager, 'update_strategy', return_value=[]) as mock_update_strategy:
            update_errors = []
            self.game_controller._process_basic_events_fallback(update_errors)
            
            # Verify strategy manager was called
            mock_update_strategy.assert_called_once()
            
            # Verify event was removed
            self.assertEqual(len(self.game_controller.events), 0)

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

    def test_integration_in_update_game_state(self):
        """Test that the integration works properly within update_game_state."""
        # Mock the new method to verify it's called
        with patch.object(self.game_controller, '_process_events_and_drive_strategy') as mock_process:
            # Mock other dependencies to avoid side effects
            with patch.object(self.game_controller, '_update_feature_systems'):
                with patch.object(self.game_controller, '_update_character', return_value=True):
                    
                    # Call update_game_state
                    self.game_controller.paused = False  # Ensure not paused
                    self.game_controller.update_game_state(0.1)
                    
                    # Verify our new method was called
                    mock_process.assert_called_once()


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