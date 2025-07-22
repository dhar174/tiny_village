# LLM Integration Usage Guide

This guide shows how to use the completed LLM integration in Tiny Village character decision-making.

## Quick Start

```python
from tiny_strategy_manager import StrategyManager
from llm_integration_utils import enable_llm_for_characters, create_llm_enabled_strategy_manager

# 1. Enable LLM for specific characters
characters = [character1, character2, character3]
enabled_characters = enable_llm_for_characters(characters, ["character1", "character2"])

# 2. Create LLM-enabled strategy manager
strategy_manager = create_llm_enabled_strategy_manager("your-model-name")

# 3. The integration works automatically in GameplayController
# Characters with use_llm_decisions=True will use LLM decision-making
# Characters without this flag will use utility-based decisions
```

## Character Configuration

```python
# Enable LLM for a character
character.use_llm_decisions = True

# Or use the strategy manager methods
strategy_manager.enable_llm_for_character(character)
strategy_manager.disable_llm_for_character(character)
```

## Strategy Manager Integration

The LLM integration is now part of the main strategy update loop:

```python
# This automatically uses LLM or utility decisions based on character configuration
strategy_manager.update_strategy(events, character)

# Direct LLM decision-making
actions = strategy_manager.decide_action_with_llm(character, time="morning", weather="sunny")
```

## Data Flow

The integration follows the documented data flow:

```
Character Context → StrategyManager → PromptBuilder → BrainIO → OutputInterpreter → Actions
```

1. **Character Context**: Current needs, state, inventory, location
2. **StrategyManager**: Orchestrates decision with utility-based action generation
3. **PromptBuilder**: Creates contextual prompts for the LLM
4. **BrainIO**: Sends prompts to LLM and receives responses
5. **OutputInterpreter**: Parses LLM responses into executable actions
6. **Actions**: Executed in the game world with fallback to utility decisions

## Error Handling

The integration includes robust fallback mechanisms:

- LLM service unavailable → Falls back to utility-based decisions
- LLM response parsing fails → Returns highest utility action
- Character missing LLM flag → Uses utility-based planning
- Any component failure → Graceful degradation to working systems

## Testing

```python
from test_llm_integration_pipeline import TestLLMIntegrationPipeline
import unittest

# Run the integration tests
unittest.main(module='test_llm_integration_pipeline')
```

## Validation

```python
from llm_integration_utils import validate_llm_integration

# Check if integration is working properly
results = validate_llm_integration(character, strategy_manager)
print(f"Fully integrated: {results['fully_integrated']}")
```

## Events and Strategy Updates

The integration handles different event types:

- **new_day**: Uses LLM for daily planning if character has LLM enabled
- **social**: Routes to social event handling with LLM awareness  
- **crisis**: Emergency response with LLM decision support
- **work**: Career and productivity decisions
- **weather**: Environmental adaptation strategies

All event types respect the character's LLM configuration and fall back to utility-based decisions as needed.