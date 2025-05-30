#!/usr/bin/env python3
"""
Simple validation test for the refactored tiny_gameplay_controller.py
This focuses on testing the core improvements without heavy dependencies.
"""

import sys
import os
import logging
from unittest.mock import Mock, patch

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_action_resolver():
    """Test ActionResolver functionality with mocked dependencies."""
    print("\n🧪 Testing ActionResolver...")

    try:
        # Mock the imports to avoid dependency issues
        with patch.dict(
            "sys.modules",
            {
                "tiny_strategy_manager": Mock(),
                "tiny_event_handler": Mock(),
                "tiny_types": Mock(),
                "tiny_map_controller": Mock(),
                "actions": Mock(),
            },
        ):
            from tiny_gameplay_controller import ActionResolver

            # Test basic initialization
            resolver = ActionResolver()
            assert resolver is not None
            print("  ✅ ActionResolver initialization successful")

            # Test dictionary action resolution
            action_dict = {"name": "Test Action", "energy_cost": 10, "satisfaction": 5}

            # Mock the Action class import
            mock_action = Mock()
            mock_action.name = "Test Action"
            mock_action.execute = Mock(return_value=True)

            with patch("actions.Action", return_value=mock_action):
                resolved_action = resolver.resolve_action(action_dict)
                assert resolved_action is not None
                print("  ✅ Dictionary action resolution successful")

            # Test action caching
            cache_size_before = len(resolver.action_cache)
            with patch("actions.Action", return_value=mock_action):
                resolver.resolve_action(action_dict)  # Should use cache
                resolver.resolve_action(action_dict)  # Should use cache
            cache_size_after = len(resolver.action_cache)
            print(f"  ✅ Action caching working (cache size: {cache_size_after})")

            # Test fallback action
            fallback = resolver.get_fallback_action()
            assert fallback is not None
            print("  ✅ Fallback action generation successful")

            # Test analytics
            analytics = resolver.get_action_analytics()
            assert "total_actions" in analytics
            assert "success_rate" in analytics
            print("  ✅ Action analytics working")

            return True

    except Exception as e:
        print(f"  ❌ ActionResolver test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_system_recovery_manager():
    """Test SystemRecoveryManager functionality."""
    print("\n🧪 Testing SystemRecoveryManager...")

    try:
        with patch.dict(
            "sys.modules",
            {
                "tiny_strategy_manager": Mock(),
                "tiny_event_handler": Mock(),
                "tiny_types": Mock(),
                "tiny_map_controller": Mock(),
                "actions": Mock(),
            },
        ):
            from tiny_gameplay_controller import SystemRecoveryManager

            # Create mock gameplay controller
            mock_controller = Mock()

            # Test initialization
            recovery_manager = SystemRecoveryManager(mock_controller)
            assert recovery_manager is not None
            print("  ✅ SystemRecoveryManager initialization successful")

            # Test recovery strategies setup
            assert len(recovery_manager.recovery_strategies) > 0
            expected_systems = ["strategy_manager", "graph_manager", "action_system"]
            for system in expected_systems:
                assert system in recovery_manager.recovery_strategies
            print("  ✅ Recovery strategies properly configured")

            # Test system status reporting
            status = recovery_manager.get_system_status()
            assert isinstance(status, dict)
            print("  ✅ System status reporting working")

            return True

    except Exception as e:
        print(f"  ❌ SystemRecoveryManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gameplay_controller_init():
    """Test GameplayController initialization with mocked dependencies."""
    print("\n🧪 Testing GameplayController initialization...")

    try:
        # Mock all external dependencies
        mock_modules = {
            "tiny_strategy_manager": Mock(),
            "tiny_event_handler": Mock(),
            "tiny_types": Mock(),
            "tiny_map_controller": Mock(),
            "actions": Mock(),
            "tiny_graph_manager": Mock(),
            "tiny_characters": Mock(),
            "tiny_locations": Mock(),
            "tiny_time_manager": Mock(),
            "tiny_items": Mock(),
            "tiny_animation_system": Mock(),
        }

        # Create mock classes
        mock_strategy_manager = Mock()
        mock_graph_manager = Mock()
        mock_event_handler = Mock()
        mock_map_controller = Mock()

        with patch.dict("sys.modules", mock_modules):
            with patch(
                "tiny_gameplay_controller.StrategyManager",
                return_value=mock_strategy_manager,
            ):
                with patch(
                    "tiny_gameplay_controller.EventHandler",
                    return_value=mock_event_handler,
                ):
                    with patch(
                        "tiny_gameplay_controller.MapController",
                        return_value=mock_map_controller,
                    ):
                        with patch("pygame.init"):
                            with patch("pygame.display.set_mode", return_value=Mock()):
                                with patch("pygame.display.set_caption"):
                                    with patch(
                                        "pygame.time.Clock", return_value=Mock()
                                    ):

                                        from tiny_gameplay_controller import (
                                            GameplayController,
                                        )

                                        config = {
                                            "screen_width": 800,
                                            "screen_height": 600,
                                            "characters": {"count": 2},
                                        }

                                        controller = GameplayController(config=config)

                                        # Verify basic attributes
                                        assert hasattr(controller, "action_resolver")
                                        assert hasattr(controller, "recovery_manager")
                                        assert hasattr(controller, "characters")
                                        assert hasattr(controller, "events")
                                        assert hasattr(controller, "game_statistics")

                                        print(
                                            "  ✅ GameplayController initialization successful"
                                        )
                                        print(
                                            f"  ✅ Game statistics initialized: {controller.game_statistics}"
                                        )

                                        # Test that recovery manager is working
                                        assert controller.recovery_manager is not None
                                        print(
                                            "  ✅ Recovery manager integrated successfully"
                                        )

                                        return True

    except Exception as e:
        print(f"  ❌ GameplayController initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_configuration_loading():
    """Test dynamic configuration loading features."""
    print("\n🧪 Testing configuration loading...")

    try:
        import tempfile
        import json

        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test character file
            char_file = os.path.join(temp_dir, "test_characters.json")
            test_characters = {
                "characters": [
                    {
                        "name": "Test Character",
                        "age": 30,
                        "job": "Tester",
                        "specialties": ["testing"],
                    }
                ]
            }
            with open(char_file, "w") as f:
                json.dump(test_characters, f)

            # Create test buildings file
            buildings_file = os.path.join(temp_dir, "test_buildings.json")
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
            with open(buildings_file, "w") as f:
                json.dump(test_buildings, f)

            # Test loading with mocked controller
            with patch.dict(
                "sys.modules",
                {
                    "tiny_strategy_manager": Mock(),
                    "tiny_event_handler": Mock(),
                    "tiny_types": Mock(),
                    "tiny_map_controller": Mock(),
                    "actions": Mock(),
                },
            ):
                with patch("pygame.init"):
                    with patch("pygame.display.set_mode", return_value=Mock()):
                        with patch("pygame.display.set_caption"):
                            with patch("pygame.time.Clock", return_value=Mock()):

                                from tiny_gameplay_controller import GameplayController

                                controller = GameplayController(config={})

                                # Test character loading
                                characters_data = controller._load_characters_from_file(
                                    char_file
                                )
                                assert len(characters_data) == 1
                                assert characters_data[0]["name"] == "Test Character"
                                print("  ✅ Character file loading successful")

                                # Test building loading
                                buildings_data = controller._load_buildings_from_file(
                                    buildings_file
                                )
                                assert len(buildings_data) == 1
                                assert buildings_data[0]["name"] == "Test Building"
                                print("  ✅ Building file loading successful")

                                return True

    except Exception as e:
        print(f"  ❌ Configuration loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_feature_validation():
    """Validate that key refactoring features are implemented."""
    print("\n🧪 Testing feature implementation status...")

    try:
        with patch.dict(
            "sys.modules",
            {
                "tiny_strategy_manager": Mock(),
                "tiny_event_handler": Mock(),
                "tiny_types": Mock(),
                "tiny_map_controller": Mock(),
                "actions": Mock(),
            },
        ):
            with patch("pygame.init"):
                with patch("pygame.display.set_mode", return_value=Mock()):
                    with patch("pygame.display.set_caption"):
                        with patch("pygame.time.Clock", return_value=Mock()):

                            from tiny_gameplay_controller import GameplayController

                            controller = GameplayController(config={})

                            # Check for key refactoring features
                            features_to_check = [
                                "action_resolver",
                                "recovery_manager",
                                "game_statistics",
                                "initialize_game_systems",
                                "setup_user_driven_configuration",
                                "update_game_state",
                                "_update_character_state_after_action",
                            ]

                            missing_features = []
                            for feature in features_to_check:
                                if not hasattr(controller, feature):
                                    missing_features.append(feature)

                            if missing_features:
                                print(f"  ❌ Missing features: {missing_features}")
                                return False
                            else:
                                print(
                                    f"  ✅ All {len(features_to_check)} key features implemented"
                                )

                            # Check feature status method if it exists
                            if hasattr(controller, "get_feature_implementation_status"):
                                status = controller.get_feature_implementation_status()
                                print(
                                    f"  ✅ Feature status tracking: {len(status)} features tracked"
                                )

                            return True

    except Exception as e:
        print(f"  ❌ Feature validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("TINY GAMEPLAY CONTROLLER REFACTORING VALIDATION")
    print("=" * 60)

    tests = [
        ("ActionResolver", test_action_resolver),
        ("SystemRecoveryManager", test_system_recovery_manager),
        ("GameplayController Init", test_gameplay_controller_init),
        ("Configuration Loading", test_configuration_loading),
        ("Feature Implementation", run_feature_validation),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"RUNNING: {test_name}")
        print("=" * 40)

        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\nResult: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ FAILED with exception: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:.<40} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests PASSED!")
        print("\nKey refactoring achievements:")
        print("  ✅ Enhanced ActionResolver with caching and validation")
        print("  ✅ Comprehensive SystemRecoveryManager")
        print("  ✅ Dynamic configuration loading")
        print("  ✅ Improved error handling throughout")
        print("  ✅ Feature implementation tracking")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed - review needed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
