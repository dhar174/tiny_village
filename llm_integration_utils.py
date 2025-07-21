"""Utilities for setting up and managing LLM integration in Tiny Village.

This module provides helper functions to configure characters and managers for LLM-based
decision making, bridging the gap between individual LLM components and strategic integration.
"""

import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def enable_llm_for_characters(characters: List, character_names: Optional[List[str]] = None):
    """Enable LLM decision-making for specific characters.
    
    Args:
        characters: List of character objects
        character_names: Optional list of character names to enable (all if None)
    """
    enabled_count = 0
    for character in characters:
        if character_names is None or character.name in character_names:
            character.use_llm_decisions = True
            enabled_count += 1
            logger.info(f"Enabled LLM decisions for {character.name}")
    
    logger.info(f"Enabled LLM decisions for {enabled_count} characters")
    return [char for char in characters if getattr(char, 'use_llm_decisions', False)]


def create_llm_enabled_strategy_manager(model_name: str = None):
    """Create a StrategyManager with LLM capabilities enabled.
    
    Args:
        model_name: Optional model name for the LLM
        
    Returns:
        StrategyManager instance with LLM enabled
    """
    try:
        from tiny_strategy_manager import StrategyManager
        
        manager = StrategyManager(
            use_llm=True, 
            model_name=model_name or "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2"
        )
        
        if manager.brain_io and manager.output_interpreter:
            logger.info(f"Created LLM-enabled StrategyManager with model: {model_name}")
            return manager
        else:
            logger.warning("LLM components not available, falling back to utility-only manager")
            return StrategyManager(use_llm=False)
            
    except ImportError as e:
        logger.error(f"Could not import StrategyManager: {e}")
        return None


def setup_character_llm_integration(character, enable_llm: bool = True):
    """Setup LLM integration for a single character.
    
    Args:
        character: Character object to configure
        enable_llm: Whether to enable LLM decisions
    """
    if not hasattr(character, 'use_llm_decisions'):
        character.use_llm_decisions = enable_llm
    else:
        character.use_llm_decisions = enable_llm
    
    logger.info(f"Set LLM decisions for {character.name}: {enable_llm}")
    return character


def setup_full_llm_integration(characters: List, llm_character_names: List[str], model_name: str = None):
    """Complete setup for LLM integration with characters and strategy manager.
    
    Args:
        characters: List of character objects
        llm_character_names: Names of characters to enable LLM for
        model_name: Optional model name for LLM
        
    Returns:
        Tuple of (enabled_characters, strategy_manager)
    """
    # Enable LLM for specified characters
    enabled_characters = enable_llm_for_characters(characters, llm_character_names)
    
    # Create LLM-enabled strategy manager
    strategy_manager = create_llm_enabled_strategy_manager(model_name)
    
    # Configure the strategy manager with the enabled characters
    if strategy_manager and hasattr(strategy_manager, 'enable_llm_for_characters'):
        strategy_manager.enable_llm_for_characters(enabled_characters)
    
    logger.info(f"Full LLM integration setup complete: {len(enabled_characters)} characters, "
                f"LLM manager: {strategy_manager is not None}")
    
    return enabled_characters, strategy_manager


def validate_llm_integration(character, strategy_manager):
    """Validate that LLM integration is properly configured.
    
    Args:
        character: Character to validate
        strategy_manager: StrategyManager to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'character_llm_enabled': getattr(character, 'use_llm_decisions', False),
        'strategy_manager_llm_enabled': getattr(strategy_manager, 'use_llm', False),
        'brain_io_available': hasattr(strategy_manager, 'brain_io') and strategy_manager.brain_io is not None,
        'output_interpreter_available': hasattr(strategy_manager, 'output_interpreter') and strategy_manager.output_interpreter is not None,
        'decide_action_with_llm_method': hasattr(strategy_manager, 'decide_action_with_llm'),
    }
    
    results['fully_integrated'] = all([
        results['character_llm_enabled'],
        results['strategy_manager_llm_enabled'], 
        results['brain_io_available'],
        results['output_interpreter_available'],
        results['decide_action_with_llm_method']
    ])
    
    logger.debug(f"LLM integration validation for {character.name}: {results}")
    return results


def create_llm_test_character(name: str, enable_llm: bool = True):
    """Create a simple test character for LLM integration testing.
    
    Args:
        name: Character name
        enable_llm: Whether to enable LLM decisions
        
    Returns:
        Test character object
    """
    class TestCharacter:
        def __init__(self, name):
            self.name = name
            self.id = name
            self.hunger_level = 5.0
            self.energy = 5.0
            self.wealth_money = 10.0
            self.social_wellbeing = 5.0
            self.mental_health = 5.0
            self.health_status = 5.0
            self.job = "test_job"
            self.use_llm_decisions = enable_llm
            self.recent_event = "joined village"  # Add missing attribute
            self.long_term_goal = "live peacefully"  # Add missing attribute
            
            # Mock location and inventory for compatibility
            self.location = type("Location", (), {"name": "TestLocation"})()
            
            class MockInventory:
                def get_food_items(self):
                    return [type("Food", (), {"name": "test_food", "calories": 100})()]
            
            self.inventory = MockInventory()
            
        def get_current_goal(self):
            return None
    
    return TestCharacter(name)