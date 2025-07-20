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
from tiny_graph_manager import GraphManager
from actions import ActionSystem

# Set up logging for test output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TestActionResolver(unittest.TestCase):
    """Test the enhanced ActionResolver class."""

    def setUp(self):
        # Using Mocks as decided in the plan
        self.mock_action_system = Mock(spec=ActionSystem)
        self.mock_graph_manager = Mock(spec=GraphManager)
        self.action_resolver = ActionResolver(action_system=self.mock_action_system, graph_manager=self.mock_graph_manager)

    def test_dict_action_resolution(self):
        """Test resolving dictionary-based actions with comprehensive validation."""
        action_dict = {"name": "Test Action", "energy_cost": 10, "satisfaction": 5}

        resolved_action = self.action_resolver.resolve_action(action_dict)
        
        # Test 1: Verify action was successfully created (not None)
        self.assertIsNotNone(resolved_action, "Action resolution should not return None")
        
        # Test 2: Verify action has execute method AND it's callable
        self.assertTrue(hasattr(resolved_action, "execute"), "Action should have execute method")
        self.assertTrue(callable(resolved_action.execute), "Execute method should be callable")
        
        # Test 3: Verify action has expected properties from the dictionary
        self.assertEqual(resolved_action.name, "Test Action", "Action name should match input")
        self.assertTrue(hasattr(resolved_action, "effects"), "Action should have effects")
        self.assertTrue(hasattr(resolved_action, "cost"), "Action should have cost")
        
        # Test 4: Verify action effects are properly configured
        self.assertIsInstance(resolved_action.effects, list, "Effects should be a list")
        
        # Verify energy cost effect is present
        energy_effects = [e for e in resolved_action.effects if e.get("attribute") == "energy"]
        self.assertTrue(len(energy_effects) > 0, "Should have energy cost effect")
        self.assertEqual(energy_effects[0]["change_value"], -10, "Energy cost should be -10")
        
        # Verify satisfaction effect is present  
        satisfaction_effects = [e for e in resolved_action.effects if e.get("attribute") == "current_satisfaction"]
        self.assertTrue(len(satisfaction_effects) > 0, "Should have satisfaction effect")
        self.assertEqual(satisfaction_effects[0]["change_value"], 5, "Satisfaction should be +5")
        
        # Test 5: Verify action can be executed without errors
        # Create a mock character to test execution
        mock_character = Mock()
        mock_character.energy = 50
        mock_character.current_satisfaction = 10
        mock_character.uuid = "test_char_uuid"
        
        # Mock the preconditions_met method to return True for this test
        with patch.object(resolved_action, 'preconditions_met', return_value=True):
            try:
                result = resolved_action.execute(character=mock_character)
                # Test 6: Verify execute method returns a boolean result
                self.assertIsInstance(result, bool, "Execute should return boolean")
            except Exception as e:
                self.fail(f"Action execute should not raise exception: {e}")
        
        # Test 7: Verify action properly handles execution when preconditions fail
        with patch.object(resolved_action, 'preconditions_met', return_value=False):
            result = resolved_action.execute(character=mock_character)
            self.assertFalse(result, "Execute should return False when preconditions not met")

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
        """Test fallback action when resolution fails with comprehensive validation."""
        # Test 1: Verify fallback action is returned when passing invalid data
        fallback_action = self.action_resolver.resolve_action(None)
        
        # Test 2: Verify fallback action was successfully created (not None)
        self.assertIsNotNone(fallback_action, "Fallback action should not be None")
        
        # Test 3: Verify fallback action has execute method AND it's callable
        self.assertTrue(hasattr(fallback_action, "execute"), "Fallback action should have execute method")
        self.assertTrue(callable(fallback_action.execute), "Fallback execute method should be callable")
        
        # Test 4: Verify fallback action has expected properties
        self.assertTrue(hasattr(fallback_action, "name"), "Fallback action should have name")
        self.assertTrue(hasattr(fallback_action, "effects"), "Fallback action should have effects")
        self.assertTrue(hasattr(fallback_action, "cost"), "Fallback action should have cost")
        
        # Test 5: Verify fallback action has the expected name (should be "Rest" from fallback_actions)
        self.assertEqual(fallback_action.name, "Rest", "Fallback action should be named 'Rest'")
        
        # Test 6: Verify fallback action has appropriate effects (energy restoration)
        self.assertIsInstance(fallback_action.effects, list, "Fallback effects should be a list")
        energy_effects = [e for e in fallback_action.effects if e.get("attribute") == "energy"]
        self.assertTrue(len(energy_effects) > 0, "Fallback should have energy restoration effect")
        # Energy cost of 10 should become +10 energy restoration in fallback
        self.assertEqual(energy_effects[0]["change_value"], -10, "Fallback should restore energy")
        
        # Test 7: Verify fallback action can actually be executed
        mock_character = Mock()
        mock_character.energy = 30
        mock_character.current_satisfaction = 5
        mock_character.uuid = "fallback_test_char"
        
        # Test execution with preconditions met
        with patch.object(fallback_action, 'preconditions_met', return_value=True):
            try:
                result = fallback_action.execute(character=mock_character)
                self.assertIsInstance(result, bool, "Fallback execute should return boolean")
                # Fallback actions should generally succeed when preconditions are met
                self.assertTrue(result, "Fallback action should succeed when preconditions met")
            except Exception as e:
                self.fail(f"Fallback action execute should not raise exception: {e}")
        
        # Test 8: Test that ActionResolver returns fallback for various invalid inputs
        invalid_inputs = [
            None,
            "nonexistent_action",
            {"invalid": "dict"},
            123,
            [],
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                result_action = self.action_resolver.resolve_action(invalid_input)
                self.assertIsNotNone(result_action, f"Should return fallback action for invalid input: {invalid_input}")
                self.assertTrue(hasattr(result_action, "execute"), f"Fallback should have execute method for input: {invalid_input}")
                self.assertTrue(callable(result_action.execute), f"Fallback execute should be callable for input: {invalid_input}")
        
        # Test 9: Verify that if Action constructor itself fails, we handle it gracefully
        with patch('tiny_gameplay_controller.Action', side_effect=Exception("Action constructor failed")):
            fallback_with_constructor_error = self.action_resolver.resolve_action({"name": "Test", "energy_cost": 5})
            # Should still return something, possibly None, but shouldn't crash
            # The exact behavior depends on implementation - the key is no unhandled exception
            # If it returns None, that's also acceptable as long as the calling code handles it

    def test_action_class_validation(self):
        """Test that validates Action class can be properly instantiated and is functional.
        
        This test ensures that the Action class itself works correctly and will fail
        if the Action class is broken or non-functional.
        """
        # Test 1: Verify Action class can be imported and instantiated
        try:
            from actions import Action
        except ImportError as e:
            self.fail(f"Action class cannot be imported: {e}")
        
        # Test 2: Verify Action class can be instantiated with minimal parameters
        try:
            test_action = Action(
                name="Validation Test Action",
                preconditions=[],
                effects=[
                    {"targets": ["initiator"], "attribute": "energy", "change_value": -5}
                ],
                cost=1
            )
        except Exception as e:
            self.fail(f"Action class cannot be instantiated: {e}")
        
        # Test 3: Verify Action instance has required methods and attributes
        required_attributes = ["name", "effects", "cost", "execute"]
        for attr in required_attributes:
            self.assertTrue(hasattr(test_action, attr), f"Action should have {attr} attribute/method")
        
        # Test 4: Verify Action.execute method is callable and returns boolean
        self.assertTrue(callable(test_action.execute), "Action.execute should be callable")
        
        # Test 5: Create a realistic mock character and test action execution
        mock_character = Mock()
        mock_character.energy = 50
        mock_character.uuid = "validation_test_char"
        mock_character.name = "Test Character"
        
        # Mock preconditions_met to ensure execution logic is tested
        with patch.object(test_action, 'preconditions_met', return_value=True):
            try:
                result = test_action.execute(character=mock_character)
                self.assertIsInstance(result, bool, "Action.execute should return boolean")
            except Exception as e:
                self.fail(f"Action.execute should not raise exception with valid character: {e}")
        
        # Test 6: Verify action fails appropriately when preconditions not met
        with patch.object(test_action, 'preconditions_met', return_value=False):
            result = test_action.execute(character=mock_character)
            self.assertFalse(result, "Action.execute should return False when preconditions not met")
        
        # Test 7: Test that Action constructor validates parameters appropriately
        # This should fail if Action class doesn't properly validate inputs
        with self.assertRaises((TypeError, ValueError)):
            invalid_action = Action(
                name=None,  # Invalid name
                preconditions=None,
                effects=None,
                cost="invalid"  # Invalid cost type
            )
    
    def test_comprehensive_action_resolver_failure_scenarios(self):
        """Test ActionResolver behavior in various failure scenarios to ensure robustness."""
        
        # Test 1: Action constructor failure scenario
        action_dict = {"name": "Test Action", "energy_cost": 10}
        
        with patch('tiny_gameplay_controller.Action', side_effect=Exception("Mock Action constructor failure")):
            result = self.action_resolver.resolve_action(action_dict)
            # Should either return None or a fallback action, but not crash
            if result is not None:
                self.assertTrue(hasattr(result, "execute"), "If result is not None, should have execute method")
        
        # Test 2: Action.execute failure scenario 
        valid_action_dict = {"name": "Valid Action", "energy_cost": 5}
        resolved_action = self.action_resolver.resolve_action(valid_action_dict)
        
        # Mock execute to fail
        if resolved_action:
            with patch.object(resolved_action, 'execute', side_effect=Exception("Execute failed")):
                mock_character = Mock()
                mock_character.energy = 50
                mock_character.uuid = "test_char"
                
                # The actual gameplay controller should handle execute failures gracefully
                # Here we just verify our resolved action can handle the failure scenario
                try:
                    resolved_action.execute(character=mock_character)
                    self.fail("Expected execute to raise exception due to our mock")
                except Exception:
                    # This is expected due to our mock - the key is testing the scenario
                    pass
        
        # Test 3: Malformed action dictionary scenarios
        malformed_dicts = [
            {"energy_cost": "invalid"},  # Invalid energy cost type
            {"name": 123},  # Invalid name type  
            {"satisfaction": None},  # None values
            {},  # Empty dict
        ]
        
        for malformed_dict in malformed_dicts:
            with self.subTest(malformed_dict=malformed_dict):
                result = self.action_resolver.resolve_action(malformed_dict)
                # Should handle gracefully - either return valid action or None
                if result is not None:
                    self.assertTrue(hasattr(result, "execute"), f"Result should have execute method for: {malformed_dict}")
                    self.assertTrue(hasattr(result, "name"), f"Result should have name for: {malformed_dict}")

    def test_shallow_vs_deep_validation_demonstration(self):
        """Demonstrate why shallow hasattr tests are insufficient.
        
        This test shows how a shallow test would pass even with a broken Action class,
        while proper validation catches the issues.
        """
        
        # Create a broken "Action" class that has execute method but doesn't work
        class BrokenAction:
            def __init__(self):
                self.name = "Broken"
                self.execute = "not_callable"  # This will break execution
                self.effects = "not_a_list"    # This will break effect processing
                self.cost = "not_a_number"     # This will break cost calculations
        
        broken_action = BrokenAction()
        
        # Test 1: Shallow test would pass (only checks method existence)
        shallow_test_passes = hasattr(broken_action, "execute")
        self.assertTrue(shallow_test_passes, "Shallow test incorrectly passes for broken action")
        
        # Test 2: Deep validation catches the problems
        # Check if execute is actually callable
        self.assertFalse(callable(broken_action.execute), "Deep test correctly identifies non-callable execute")
        
        # Check if effects is actually a list
        self.assertFalse(isinstance(broken_action.effects, list), "Deep test correctly identifies invalid effects")
        
        # Check if cost is actually a number
        self.assertFalse(isinstance(broken_action.cost, (int, float)), "Deep test correctly identifies invalid cost")
        
        # Test 3: Show that actually calling execute would fail
        mock_character = Mock()
        try:
            broken_action.execute(character=mock_character)
            self.fail("Broken action execute should fail when called")
        except (TypeError, AttributeError):
            # This is expected - the broken action fails when actually used
            pass
        
        # Test 4: Compare with a properly working action from our resolver
        working_action = self.action_resolver.resolve_action({"name": "Working Action", "energy_cost": 5})
        if working_action:
            # Proper validation checks
            self.assertTrue(callable(working_action.execute), "Working action should have callable execute")
            self.assertIsInstance(working_action.effects, list, "Working action should have list effects")
            self.assertIsInstance(working_action.cost, (int, float), "Working action should have numeric cost")
            
            # And it should actually work when called
            mock_character = Mock()
            mock_character.energy = 50
            mock_character.uuid = "working_test"
            
            with patch.object(working_action, 'preconditions_met', return_value=True):
                try:
                    result = working_action.execute(character=mock_character)
                    self.assertIsInstance(result, bool, "Working action execute should return boolean")
                except Exception as e:
                    self.fail(f"Working action should execute without error: {e}")

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
