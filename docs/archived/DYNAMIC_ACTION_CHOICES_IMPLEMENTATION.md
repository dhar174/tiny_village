# Dynamic Action Choices Implementation

## Overview
This implementation replaces hardcoded action choices in prompt generation with dynamically generated, contextually relevant actions from GOAPPlanner and StrategyManager.

## Problem Statement
Previously, the `generate_daily_routine_prompt()` method in `tiny_prompt_builder.py` contained hardcoded action choices:
- "Go to the market to Buy_Food"
- "Work at your job to Improve_job_performance"
- "Visit a friend to Increase_Friendship"
- "Engage in a Leisure_Activity to improve Mental_Health"
- "Work on a personal project to Pursue_Hobby"

These static choices didn't adapt to character state, goals, or current needs.

## Solution
Integrated dynamic action generation using existing infrastructure:
1. `PromptBuilder.prioritize_actions()` calls `StrategyManager.get_daily_actions()`
2. `StrategyManager` uses `GOAPPlanner` to generate contextually relevant actions
3. Actions are sorted by utility score and formatted with their effects
4. Prompts now include dynamic, character-specific action choices

## Changes Made

### File: `tiny_prompt_builder.py`

#### 1. Added Logging Support (Line 13)
```python
import logging
logger = logging.getLogger(__name__)
```

#### 2. Added Constants to PromptBuilder Class (Lines 2195-2202)
```python
# Urgency thresholds for goal priorities
URGENCY_THRESHOLD_URGENT = 8.0
URGENCY_THRESHOLD_HIGH = 6.0

# Needs priority thresholds
NEEDS_PRIORITY_CRITICAL_THRESHOLD = 80.0
NEEDS_PRIORITY_HIGH_THRESHOLD = 60.0
```

#### 3. Fixed Exception Handling (Line 2430)
```python
# Before:
except (ValueError, TypeError):  # e not captured
    util = 0.0
    print(f"Error calculating utility for action {action}: {e}")

# After:
except (ValueError, TypeError) as e:  # Properly capture exception
    util = 0.0
    logger.warning(f"Error calculating utility for action {action}: {e}")
```

#### 4. Replaced Hardcoded Actions (Lines 2570-2596)
```python
# Before:
prompt += "Options:\n"
prompt += "1. Go to the market to Buy_Food.\n"
prompt += f"2. Work at your job to Improve_{getattr(self.character, 'job_performance', 'job_performance')}.\n"
prompt += "3. Visit a friend to Increase_Friendship.\n"
prompt += "4. Engage in a Leisure_Activity to improve Mental_Health.\n"
prompt += "5. Work on a personal project to Pursue_Hobby.\n"

# After:
prompt += "Options:\n"
dynamic_action_choices = self.prioritize_actions()

if dynamic_action_choices:
    # Use dynamically generated action choices with utility scores
    for choice in dynamic_action_choices:
        prompt += f"{choice}\n"
else:
    # Fallback to basic action prioritization if StrategyManager unavailable
    actions = self.action_options.prioritize_actions(self.character)
    for i, action in enumerate(actions[:5], 1):
        try:
            descriptor = descriptors.get_action_descriptors(action)
        except (KeyError, AttributeError):
            descriptor = action.replace("_", " ").title()
        action_name = action.replace("_", " ").title().replace(" ", "_")
        prompt += f"{i}. {descriptor} to {action_name}.\n"
```

## How It Works

### Action Generation Flow
1. **PromptBuilder** calls `self.prioritize_actions()`
2. **prioritize_actions()** creates a `StrategyManager` instance
3. **StrategyManager** calls `get_daily_actions(character)`
4. **get_daily_actions()** generates potential actions based on:
   - Character state (hunger, energy, health, etc.)
   - Current location and inventory
   - Character job and skills
5. Actions are evaluated using **utility functions** from `tiny_utility_functions`
6. Actions are sorted by utility score (highest first)
7. Top 5 actions are formatted with:
   - Action name and description
   - Utility score (0-10 scale)
   - Effects on character attributes

### Example Dynamic Action Format
```
1. Rest to regain energy (Utility: 7.5) - Effects: energy: +0.15
2. Work on current project (Utility: 6.8) - Effects: money: +20.0, energy: -0.30
3. Exercise to improve health (Utility: 5.2) - Effects: health: +0.10
4. Eat available food (Utility: 4.9) - Effects: hunger: -0.25
5. Social visit to friend (Utility: 4.3) - Effects: social_wellbeing: +0.15
```

## Benefits

### 1. Contextual Relevance
Actions adapt to character's current situation:
- Low energy → Rest and Sleep actions prioritized
- High hunger → Eat and Buy_Food actions prioritized
- At home location → Home-based actions available
- Low social_wellbeing → Social interaction actions prioritized

### 2. Transparent Decision Making
- Utility scores help characters make informed choices
- Action effects clearly shown
- Players/users can understand why actions are suggested

### 3. Better AI Behavior
- Characters make more realistic decisions
- Goals and needs directly influence action choices
- No more irrelevant hardcoded options

### 4. Integration with GOAP
- Leverages existing Goal-Oriented Action Planning
- Actions align with character goals
- Supports complex, multi-step planning

### 5. Backward Compatibility
- Fallback mechanism if StrategyManager unavailable
- Graceful degradation to ActionOptions.prioritize_actions()
- No breaking changes to existing code

## Testing

### Test Results
✅ **Dynamic Action Generation** - Verified working correctly
✅ **Hardcoded Actions Removed** - Confirmed absent from prompts
✅ **Utility Scores Included** - Displayed in action choices
✅ **Fallback Mechanism** - Exists and functions correctly
✅ **Code Review** - All issues addressed
✅ **Security Scan (CodeQL)** - No vulnerabilities found

### Example Test Output
```
=== Testing Dynamic Action Choices ===

Generated Prompt:
================================================================================
<|system|>...
<|user|>TestCharacter, it's morning, and it's a sunny day outside...
Options:
1. Rest to regain energy (Utility: 7.5) - Effects: energy: +0.15
2. Work on current project (Utility: 6.8) - Effects: money: +20.0
3. Exercise to improve health (Utility: 5.2) - Effects: health: +0.10
...
================================================================================

✅ SUCCESS: Dynamic action choices are working correctly!
   - Hardcoded actions removed
   - Dynamic actions from StrategyManager included
   - Utility scores displayed
```

## Architecture

### Component Interaction
```
┌─────────────────┐
│  PromptBuilder  │
└────────┬────────┘
         │ calls prioritize_actions()
         ▼
┌─────────────────┐
│ StrategyManager │
└────────┬────────┘
         │ calls get_daily_actions()
         ▼
┌─────────────────┐     ┌──────────────┐
│  GOAPPlanner    │────▶│ Character    │
└────────┬────────┘     │ State/Goals  │
         │              └──────────────┘
         │ generates actions
         ▼
┌─────────────────┐
│ Action Objects  │
│ with Effects    │
└────────┬────────┘
         │ calculate_action_utility()
         ▼
┌─────────────────┐
│ Utility Scores  │
│ Sorted Actions  │
└────────┬────────┘
         │ format for prompt
         ▼
┌─────────────────┐
│ Dynamic Prompt  │
│ Action Choices  │
└─────────────────┘
```

## Future Enhancements

### Potential Improvements
1. **Cached Action Plans** - Cache recent action generation to reduce computation
2. **Action Learning** - Track successful actions to improve future suggestions
3. **Personality-Based Filtering** - Further filter actions based on character personality
4. **Context-Aware Descriptions** - More varied action descriptions based on context
5. **Multi-Turn Planning** - Show consequences of action sequences

### Extensibility
The implementation is designed to be extensible:
- New action types can be added to StrategyManager
- GOAPPlanner can be enhanced without changing PromptBuilder
- Utility calculation can be refined independently
- Action effect system can be expanded

## Maintenance

### Code Locations
- **Main Implementation:** `tiny_prompt_builder.py` lines 2570-2596
- **Action Prioritization:** `tiny_prompt_builder.py` line 2396
- **Strategy Manager:** `tiny_strategy_manager.py`
- **GOAP Planner:** `tiny_goap_system.py`
- **Utility Functions:** `tiny_utility_functions.py`

### Key Methods
- `PromptBuilder.generate_daily_routine_prompt()` - Entry point
- `PromptBuilder.prioritize_actions()` - Gets dynamic actions
- `StrategyManager.get_daily_actions()` - Generates actions
- `GOAPPlanner.plan_actions()` - Plans action sequences
- `calculate_action_utility()` - Scores actions

## Troubleshooting

### Common Issues

**Issue:** Empty action list in prompt
**Solution:** Check StrategyManager initialization and fallback mechanism

**Issue:** Low utility scores for all actions
**Solution:** Review character state values and utility calculation parameters

**Issue:** Actions not relevant to character state
**Solution:** Verify GOAPPlanner goal setting and action effects configuration

**Issue:** Import errors for StrategyManager
**Solution:** Check graceful ImportError handling in prioritize_actions()

## Conclusion

This implementation successfully replaces hardcoded action choices with dynamic, contextually relevant options generated by GOAPPlanner and StrategyManager. The system now provides:

- ✅ Better AI decision-making
- ✅ Transparent action reasoning
- ✅ Character-specific action choices
- ✅ Integration with existing planning systems
- ✅ Backward compatibility and robustness

The change aligns with the goal-oriented design philosophy of the Tiny Village system and provides a more immersive, realistic character behavior experience.
