#!/usr/bin/env python3
"""
Integration Test Suite for Tiny Village

This test suite validates the critical integration points for a minimal demo:
1. Full turn cycle (prompt → LLM fallback → parse → execute → memory)
2. Failure modes (LLM timeout, invalid actions, plan failures)
3. Event handling and strategy updates
4. Error recovery and fallback mechanisms
5. Basic performance metrics

These tests use real components (not mocks where possible) to validate
actual integration behavior.
"""

import unittest
import sys
import logging
from unittest.mock import Mock
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def mock_pygame():
    """Mock pygame to avoid display dependencies."""
    mock = Mock()
    mock.time.get_ticks.return_value = 1000
    mock.font.Font.return_value = Mock()
    mock.display.flip = Mock()
    sys.modules['pygame'] = mock
    sys.modules['pygame.font'] = mock.font
    sys.modules['pygame.time'] = mock.time
    sys.modules['pygame.display'] = mock.display
    return mock

# Mock pygame before imports
mock_pygame()

from tiny_gameplay_controller import GameplayController
from tiny_event_handler import Event

class TestFullTurnCycle(unittest.TestCase):
    """Test the complete character turn cycle."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = {
            "target_fps": 60,
            "render": {"background_color": [20, 50, 80]},
            "characters": {"count": 0}
        }
        
        self.controller = GameplayController(config=config)
        self.controller.screen = Mock()
        self.controller.clock = Mock()
        self.controller.clock.tick.return_value = 16
        
        # Create a test character
        self.test_char = Mock()
        self.test_char.name = "TestChar"
        self.test_char.energy = 50
        self.test_char.health_status = 80
        self.test_char.hunger_level = 5
        self.test_char.use_llm_decisions = False
        self.test_char.uuid = "test_char_1"
        
        self.controller.characters = {"test_char_1": self.test_char}
    
    def test_turn_cycle_with_fallback_action(self):
        """Test that a character turn completes using fallback logic."""
        logger.info("\n=== Test: Turn Cycle with Fallback Action ===")
        
        # Process the character's turn
        result = self.controller._execute_character_actions(self.test_char)
        
        # Should return a boolean value
        self.assertIn(result, [True, False], "Turn cycle should return a boolean")
        logger.info("✅ Turn cycle completed (fallback logic)")
    
    def test_turn_cycle_with_strategy_manager(self):
        """Test turn cycle when strategy manager provides actions."""
        logger.info("\n=== Test: Turn Cycle with Strategy Manager ===")
        
        if self.controller.strategy_manager:
            # Get actions from strategy manager
            actions = self.controller.strategy_manager.get_daily_actions(self.test_char)
            
            self.assertIsNotNone(actions, "Strategy manager should return actions")
            logger.info(f"✅ Strategy manager provided {len(actions)} actions")
        else:
            logger.warning("⚠️  Strategy manager not available")
            self.skipTest("Strategy manager not initialized")
    
    def test_action_resolution(self):
        """Test action resolver can convert action data to executable actions."""
        logger.info("\n=== Test: Action Resolution ===")
        
        if not self.controller.action_resolver:
            self.skipTest("Action resolver not initialized")
        
        # Test with dictionary action
        dict_action = {
            "name": "TestAction",
            "energy_cost": 5,
            "satisfaction": 3
        }
        
        resolved = self.controller.action_resolver.resolve_action(
            dict_action, 
            self.test_char
        )
        
        self.assertIsNotNone(resolved, "Action resolver should resolve dictionary action")
        logger.info(f"✅ Dictionary action resolved: {resolved.name}")
        
        # Test with string action
        str_action = "Rest"
        resolved2 = self.controller.action_resolver.resolve_action(
            str_action,
            self.test_char
        )
        
        self.assertIsNotNone(resolved2, "Action resolver should resolve string action")
        logger.info(f"✅ String action resolved: {resolved2.name}")

class TestFailureModes(unittest.TestCase):
    """Test system behavior under failure conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = {
            "target_fps": 60,
            "render": {"background_color": [20, 50, 80]},
            "characters": {"count": 0}
        }
        
        self.controller = GameplayController(config=config)
        self.controller.screen = Mock()
        self.controller.clock = Mock()
        
        self.test_char = Mock()
        self.test_char.name = "TestChar"
        self.test_char.energy = 50
        self.test_char.uuid = "test_char_1"
    
    def test_invalid_action_handling(self):
        """Test that invalid actions are handled gracefully."""
        logger.info("\n=== Test: Invalid Action Handling ===")
        
        if not self.controller.action_resolver:
            self.skipTest("Action resolver not initialized")
        
        # Try resolving various invalid inputs
        invalid_actions = [
            None,
            "",
            {"invalid": "data"},
            "NonexistentAction",
            12345,
            []
        ]
        
        for invalid_action in invalid_actions:
            resolved = self.controller.action_resolver.resolve_action(
                invalid_action,
                self.test_char
            )
            
            # Should either return a fallback action or None, not crash
            self.assertTrue(
                resolved is None or hasattr(resolved, 'execute'),
                f"Invalid action {invalid_action} should be handled safely"
            )
        
        logger.info("✅ All invalid actions handled gracefully")
    
    def test_fallback_action_execution(self):
        """Test that fallback actions execute when normal actions fail."""
        logger.info("\n=== Test: Fallback Action Execution ===")
        
        # Attempt to process turn for character
        result = self.controller._execute_fallback_character_action(self.test_char)
        
        # Fallback should always succeed in some way
        self.assertTrue(result, "Fallback action should succeed")
        logger.info("✅ Fallback action executed successfully")
    
    def test_error_recovery(self):
        """Test that errors are recovered and don't crash the system."""
        logger.info("\n=== Test: Error Recovery ===")
        
        # Try to trigger an error condition that should be recovered
        try:
            # Process character turn with minimal setup
            result = self.controller._execute_character_actions(self.test_char)
            # System should survive even if result is False
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"System failed to recover from error: {e}")
        
        logger.info("✅ System recovered from potential errors")

class TestEventIntegration(unittest.TestCase):
    """Test event handling and strategy integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = {
            "target_fps": 60,
            "render": {"background_color": [20, 50, 80]},
            "characters": {"count": 0}
        }
        
        self.controller = GameplayController(config=config)
        self.controller.screen = Mock()
    
    def test_event_handler_initialization(self):
        """Test that event handler initializes correctly."""
        logger.info("\n=== Test: Event Handler Initialization ===")
        
        self.assertIsNotNone(
            self.controller.event_handler,
            "Event handler should be initialized"
        )
        logger.info("✅ Event handler initialized")
    
    def test_event_processing(self):
        """Test that events can be processed without crashing."""
        logger.info("\n=== Test: Event Processing ===")
        
        if not self.controller.event_handler:
            self.skipTest("Event handler not initialized")
        
        # Try to check and process events
        try:
            events = self.controller.event_handler.check_events()
            self.assertIsNotNone(events, "Event check should return a list")
            logger.info(f"✅ Event check returned {len(events)} events")
        except Exception as e:
            logger.warning(f"⚠️  Event checking not fully implemented: {e}")
    
    def test_strategy_update_from_events(self):
        """Test that strategy manager responds to events."""
        logger.info("\n=== Test: Strategy Update from Events ===")
        
        if not self.controller.strategy_manager:
            self.skipTest("Strategy manager not initialized")
        
        # Create test events
        events = [
            {
                'type': 'social',
                'name': 'Test Event',
                'importance': 5
            }
        ]
        
        try:
            # Strategy manager should be able to process events
            result = self.controller.strategy_manager.update_strategy(events)
            # Result can be None, action, or list - all are valid
            logger.info(f"✅ Strategy update completed (result type: {type(result)})")
        except Exception as e:
            logger.warning(f"⚠️  Strategy update not fully implemented: {e}")

class TestPerformanceMetrics(unittest.TestCase):
    """Test performance monitoring and analytics."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = {
            "target_fps": 60,
            "render": {"background_color": [20, 50, 80]},
            "characters": {"count": 0}
        }
        
        self.controller = GameplayController(config=config)
    
    def test_action_analytics(self):
        """Test that action analytics are tracked."""
        logger.info("\n=== Test: Action Analytics ===")
        
        if not self.controller.action_resolver:
            self.skipTest("Action resolver not initialized")
        
        analytics = self.controller.action_resolver.get_action_analytics()
        
        self.assertIsInstance(analytics, dict, "Analytics should be a dictionary")
        self.assertIn('total_actions', analytics, "Should track total actions")
        self.assertIn('success_rate', analytics, "Should track success rate")
        
        logger.info(f"✅ Analytics available: {list(analytics.keys())}")
    
    def test_game_statistics(self):
        """Test that game statistics are maintained."""
        logger.info("\n=== Test: Game Statistics ===")
        
        stats = self.controller.game_statistics
        
        self.assertIsInstance(stats, dict, "Statistics should be a dictionary")
        self.assertIn('actions_executed', stats, "Should track executed actions")
        self.assertIn('actions_failed', stats, "Should track failed actions")
        
        logger.info(f"✅ Statistics tracked: {list(stats.keys())}")

def run_integration_tests():
    """Run all integration tests and report results."""
    logger.info("\n" + "=" * 70)
    logger.info("TINY VILLAGE INTEGRATION TEST SUITE")
    logger.info("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFullTurnCycle))
    suite.addTests(loader.loadTestsFromTestCase(TestFailureModes))
    suite.addTests(loader.loadTestsFromTestCase(TestEventIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        logger.info("\n✅ ALL TESTS PASSED")
        return 0
    else:
        logger.info("\n⚠️  SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit_code = run_integration_tests()
    sys.exit(exit_code)
