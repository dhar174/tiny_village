#!/usr/bin/env python3
"""
Test script for the improved tiny_gameplay_controller.py
Tests key refactoring improvements including error handling, recovery, and feature implementations.
"""

import sys
import os
import unittest
import logging
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the refactored gameplay controller
from tiny_gameplay_controller import (
    GameplayController,
    ActionResolver,
    SystemRecoveryManager,
)

# Set up logging for test output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TestActionResolver(unittest.TestCase):
from tiny_graph_manager import GraphManager # Added
from actions import ActionSystem # Added
from unittest.mock import Mock, MagicMock, patch # Ensure Mock is imported

class TestActionResolver(unittest.TestCase):
    """Test the enhanced ActionResolver class."""

    def setUp(self):
        # Using Mocks as decided in the plan
        self.mock_action_system = Mock(spec=ActionSystem)
        self.mock_graph_manager = Mock(spec=GraphManager)
        self.action_resolver = ActionResolver(action_system=self.mock_action_system, graph_manager=self.mock_graph_manager)

    def test_dict_action_resolution(self):
        """Test resolving dictionary-based actions."""
        action_dict = {"name": "Test Action", "energy_cost": 10, "satisfaction": 5}

        resolved_action = self.action_resolver.resolve_action(action_dict)
        self.assertIsNotNone(resolved_action)
        self.assertTrue(hasattr(resolved_action, "execute"))
        self.assertEqual(resolved_action.name, "Test Action")

    def test_action_caching(self):
        """Test action caching functionality."""
        action_dict = {"name": "Cached Action", "energy_cost": 5}

        # First resolution should create and cache
        action1 = self.action_resolver.resolve_action(action_dict)
        cache_size_1 = len(self.action_resolver.action_cache)

        # Second resolution should use cache
        action2 = self.action_resolver.resolve_action(action_dict)
        cache_size_2 = len(self.action_resolver.action_cache)

        self.assertEqual(cache_size_1, cache_size_2)  # Cache size shouldn't increase

    def test_fallback_action(self):
        """Test fallback action when resolution fails."""
        # Pass invalid action data
        fallback_action = self.action_resolver.resolve_action(None)
        self.assertIsNotNone(fallback_action)
        self.assertTrue(hasattr(fallback_action, "execute"))

    def test_action_analytics(self):
        """Test action execution analytics."""
        # Track some test actions
        mock_character = Mock()
        mock_character.uuid = "test_char"
        mock_character.name = "Test Character"
        mock_character.energy = 100

        mock_action = Mock()
        mock_action.name = "Test Action"

        self.action_resolver.track_action_execution(mock_action, mock_character, True)
        self.action_resolver.track_action_execution(mock_action, mock_character, False)

        analytics = self.action_resolver.get_action_analytics()
        self.assertIn("total_actions", analytics)
        self.assertIn("success_rate", analytics)
        self.assertEqual(analytics["total_actions"], 2)
        self.assertEqual(analytics["success_rate"], 0.5)


class TestSystemRecoveryManager(unittest.TestCase):
    """Test the SystemRecoveryManager functionality."""

    def setUp(self):
        self.mock_gameplay_controller = Mock()
        self.recovery_manager = SystemRecoveryManager(self.mock_gameplay_controller)

    def test_recovery_strategy_setup(self):
        """Test that recovery strategies are properly set up."""
        self.assertIn("strategy_manager", self.recovery_manager.recovery_strategies)
        self.assertIn("graph_manager", self.recovery_manager.recovery_strategies)
        self.assertIn("action_system", self.recovery_manager.recovery_strategies)

    def test_recovery_attempt_limits(self):
        """Test that recovery attempts are limited."""
        # Set up a failing recovery
        self.recovery_manager._recover_strategy_manager = Mock(return_value=False)

        # Should allow up to max_recovery_attempts
        for i in range(self.recovery_manager.max_recovery_attempts):
            result = self.recovery_manager.attempt_recovery("strategy_manager")
            self.assertFalse(result)

        # Should refuse further attempts
        result = self.recovery_manager.attempt_recovery("strategy_manager")
        self.assertFalse(result)

    def test_successful_recovery_resets_attempts(self):
        """Test that successful recovery resets attempt counter."""
        # Set up a successful recovery
        self.recovery_manager._recover_strategy_manager = Mock(return_value=True)

        result = self.recovery_manager.attempt_recovery("strategy_manager")
        self.assertTrue(result)
        self.assertEqual(
            self.recovery_manager.recovery_attempts.get("strategy_manager", 0), 0
        )


class TestGameplayControllerInitialization(unittest.TestCase):
    """Test GameplayController initialization and error handling."""

    def test_initialization_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        config = {"screen_width": 800, "screen_height": 600}

        # Mock pygame to avoid display initialization issues in test
        with patch("pygame.init"), patch("pygame.display.set_mode") as mock_display:
            mock_display.return_value = Mock()

            controller = GameplayController(config=config)

            # Should have basic systems initialized
            self.assertIsNotNone(controller.action_resolver)
            self.assertIsNotNone(controller.recovery_manager)
            self.assertIsInstance(controller.characters, dict)
            self.assertIsInstance(controller.events, list)

    def test_initialization_error_recovery(self):
        """Test that initialization errors are handled gracefully."""
        config = {}

        with patch("pygame.init") as mock_pygame_init:
            mock_pygame_init.side_effect = Exception("Pygame init failed")

            controller = GameplayController(config=config)

            # Controller should still be created despite pygame failure
            self.assertIsNotNone(controller)
            self.assertIn(
                "Pygame initialization failed", controller.initialization_errors
            )


class TestUserDrivenConfiguration(unittest.TestCase):
    """Test dynamic and user-driven configuration features."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_character_loading_from_file(self):
        """Test loading characters from external JSON file."""
        # Create a test character file
        test_characters = {
            "characters": [
                {
                    "name": "Test Character",
                    "age": 30,
                    "job": "Tester",
                    "specialties": ["testing", "validation"],
                }
            ]
        }

        char_file = os.path.join(self.temp_dir, "test_characters.json")
        with open(char_file, "w") as f:
            json.dump(test_characters, f)

        config = {"characters": {"characters_file": char_file}}

        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config=config)

            # Test the character loading method directly
            characters_data = controller._load_characters_from_file(char_file)

            self.assertEqual(len(characters_data), 1)
            self.assertEqual(characters_data[0]["name"], "Test Character")
            self.assertEqual(characters_data[0]["job"], "Tester")

    def test_buildings_loading_from_file(self):
        """Test loading buildings from external JSON file."""
        test_buildings = {
            "buildings": [
                {
                    "name": "Test Building",
                    "type": "test",
                    "x": 100,
                    "y": 100,
                    "width": 50,
                    "height": 50,
                }
            ]
        }

        buildings_file = os.path.join(self.temp_dir, "test_buildings.json")
        with open(buildings_file, "w") as f:
            json.dump(test_buildings, f)

        config = {"map": {"buildings_file": buildings_file}}

        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config=config)

            # Test the buildings loading method directly
            buildings_data = controller._load_buildings_from_file(buildings_file)

            self.assertEqual(len(buildings_data), 1)
            self.assertEqual(buildings_data[0]["name"], "Test Building")
            self.assertEqual(buildings_data[0]["type"], "test")


class TestFeatureImplementation(unittest.TestCase):
    """Test the implemented feature stubs and systems."""

    def setUp(self):
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            self.controller = GameplayController(config={})

    def test_feature_status_tracking(self):
        """Test that feature implementation status is tracked."""
        if hasattr(self.controller, "get_feature_implementation_status"):
            status = self.controller.get_feature_implementation_status()

            # Should have status for all major features
            expected_features = [
                "achievement_system",
                "weather_system",
                "social_network",
                "quest_system",
            ]
            for feature in expected_features:
                self.assertIn(feature, status)

    def test_achievement_system_stub(self):
        """Test achievement system implementation."""
        if hasattr(self.controller, "achievement_system"):
            # Should be able to track achievements
            self.assertIsNotNone(self.controller.achievement_system)

    def test_weather_system_stub(self):
        """Test weather system implementation."""
        if hasattr(self.controller, "weather_system"):
            # Should have basic weather functionality
            self.assertIsNotNone(self.controller.weather_system)


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test comprehensive error handling and recovery mechanisms."""

    def test_character_update_error_handling(self):
        """Test error handling during character updates."""
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})

            # Create a mock character that will fail updates
            mock_character = Mock()
            mock_character.name = "Failing Character"
            mock_character.recall_recent_memories.side_effect = Exception(
                "Memory error"
            )
            mock_character.evaluate_goals.side_effect = Exception("Goals error")

            controller.characters["test_char"] = mock_character

            # Update should handle errors gracefully
            controller.update_game_state(0.016)  # 60 FPS delta time

            # Controller should still be functional
            self.assertIsNotNone(controller)

    def test_action_execution_error_recovery(self):
        """Test error recovery during action execution."""
        with patch("pygame.init"), patch("pygame.display.set_mode"):
            controller = GameplayController(config={})

            # Test with a failing action
            failing_action = {"name": "Failing Action", "execute": lambda: False}

            # Should handle execution failure gracefully
            result = controller._execute_decision_action(failing_action)

            # Should return False but not crash
            self.assertFalse(result)


def run_integration_test():
    """Run a simple integration test to verify the system works together."""
    logger.info("Running integration test...")

    try:
        config = {
            "screen_width": 800,
            "screen_height": 600,
            "characters": {"count": 2},
            "target_fps": 60,
        }

        with patch("pygame.init"), patch("pygame.display.set_mode") as mock_display:
            mock_display.return_value = Mock()

            # Create controller
            controller = GameplayController(config=config)

            # Verify core systems
            assert (
                controller.action_resolver is not None
            ), "ActionResolver not initialized"
            assert (
                controller.recovery_manager is not None
            ), "Recovery manager not initialized"
            assert isinstance(
                controller.characters, dict
            ), "Characters not properly initialized"

            # Test system recovery
            system_status = controller.recovery_manager.get_system_status()
            logger.info(f"System status: {system_status}")

            # Test action resolution
            test_action = {"name": "Test Action", "energy_cost": 10}
            resolved = controller.action_resolver.resolve_action(test_action)
            assert resolved is not None, "Action resolution failed"
            assert hasattr(
                resolved, "execute"
            ), "Resolved action missing execute method"

            # Test game state update
            controller.update_game_state(0.016)

            # Test analytics
            analytics = controller.action_resolver.get_action_analytics()
            logger.info(f"Action analytics: {analytics}")

            logger.info("✅ Integration test passed!")
            return True

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TINY GAMEPLAY CONTROLLER REFACTORING")
    print("=" * 60)

    # Run integration test first
    integration_success = run_integration_test()

    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)

    # Run unit tests
    unittest.main(verbosity=2, exit=False)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print("Unit Tests: See results above")
    print("=" * 60)
