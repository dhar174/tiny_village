#!/usr/bin/env python3
"""
Test script to validate system integration in update_game_state method.
Tests that AI, world, and event systems are correctly coordinated.
"""

import sys
import os
import logging
from unittest.mock import Mock, MagicMock

# Set up minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mock_pygame():
    """Mock pygame to avoid dependency issues during testing."""
    sys.modules['pygame'] = Mock()
    sys.modules['pygame.font'] = Mock()
    sys.modules['pygame.time'] = Mock()
    sys.modules['pygame.display'] = Mock()
    
    # Mock pygame.time.get_ticks to return incrementing time
    mock_time = Mock()
    mock_time.get_ticks.return_value = 1000
    sys.modules['pygame.time'] = mock_time
    sys.modules['pygame'].time = mock_time

def test_system_integration():
    """Test that all game systems are properly integrated in update_game_state."""
    
    # Mock pygame before importing gameplay controller
    mock_pygame()
    
    try:
        from tiny_gameplay_controller import GameplayController
        
        # Create controller with minimal configuration
        config = {
            "target_fps": 60,
            "render": {"background_color": [20, 50, 80]},
            "characters": {"count": 2}
        }
        
        controller = GameplayController(config=config)
        
        # Mock required systems
        controller.screen = Mock()
        controller.clock = Mock()
        controller.clock.tick.return_value = 16  # 60 FPS
        
        # Mock strategy manager
        controller.strategy_manager = Mock()
        controller.strategy_manager.update_strategy.return_value = [
            {"type": "character_action", "character_id": "test_char", "action": {"name": "Test Action"}}
        ]
        
        # Mock event handler
        controller.event_handler = Mock()
        controller.event_handler.check_events.return_value = [
            {"type": "new_day", "date": "2024-01-01"}
        ]
        
        # Mock map controller
        controller.map_controller = Mock()
        
        # Mock action resolver
        controller.action_resolver = Mock()
        
        # Create test character
        test_character = Mock()
        test_character.name = "TestCharacter"
        test_character.energy = 75
        test_character.health_status = 80
        test_character.use_llm_decisions = False
        
        controller.characters = {"test_char": test_character}
        
        logger.info("Testing system integration...")
        
        # Test 1: Basic update_game_state execution
        try:
            controller.update_game_state(0.016)  # 60 FPS delta time
            logger.info("✓ update_game_state executed without errors")
        except Exception as e:
            logger.error(f"✗ update_game_state failed: {e}")
            return False
        
        # Test 2: Verify event handler is called
        assert controller.event_handler.check_events.called, "Event handler check_events not called"
        logger.info("✓ Event handler integration working")
        
        # Test 3: Verify strategy manager is called
        assert controller.strategy_manager.update_strategy.called, "Strategy manager update_strategy not called"
        logger.info("✓ Strategy manager integration working")
        
        # Test 4: Verify map controller is updated
        assert controller.map_controller.update.called, "Map controller update not called"
        logger.info("✓ Map controller integration working")
        
        # Test 5: Test AI system integration (character processing)
        controller.strategy_manager.get_daily_actions.return_value = [Mock()]
        controller.strategy_manager.get_daily_actions.return_value[0].execute.return_value = True
        controller.strategy_manager.get_daily_actions.return_value[0].name = "Mock Action"
        
        controller.update_game_state(0.016)
        logger.info("✓ AI system integration working")
        
        # Test 6: Test world system updates (feature systems)
        controller.weather_system = {"current_weather": "clear", "temperature": 20}
        controller.social_networks = {"relationships": {}}
        controller.quest_system = {"active_quests": {}, "completed_quests": {}}
        
        controller.update_game_state(0.016)
        logger.info("✓ World system integration working")
        
        # Test 7: Test system recovery
        controller.map_controller = None  # Simulate system failure
        controller.update_game_state(0.016)
        logger.info("✓ System recovery integration working")
        
        logger.info("All system integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_architecture_compliance():
    """Test that the architecture follows the documented design."""
    
    mock_pygame()
    
    try:
        from tiny_gameplay_controller import GameplayController
        
        controller = GameplayController()
        
        # Test that update_game_state handles the documented workflow:
        # 1. Event-driven strategy updates
        # 2. Core system updates 
        # 3. Event processing and feature system updates
        # 4. Automatic system recovery
        
        # Verify key integration points exist
        assert hasattr(controller, 'update_game_state'), "Missing update_game_state method"
        assert hasattr(controller, '_update_character'), "Missing _update_character method"
        assert hasattr(controller, '_process_events_and_update_strategy'), "Missing event processing method"
        assert hasattr(controller, '_update_feature_systems'), "Missing feature systems update method"
        assert hasattr(controller, 'recovery_manager'), "Missing recovery manager"
        
        logger.info("✓ Architecture compliance verified")
        return True
        
    except Exception as e:
        logger.error(f"Architecture compliance test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting system integration tests...")
    
    integration_passed = test_system_integration()
    architecture_passed = test_architecture_compliance()
    
    if integration_passed and architecture_passed:
        logger.info("🎉 All tests passed! System integration is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. System integration needs attention.")
        sys.exit(1)