# GOAP Planner Integration with Character/World State

## Overview
This implementation addresses the requirement for the planner to dynamically receive the current world state and available actions for characters.

## Key Features Implemented

### 1. Dynamic World State Retrieval (`get_current_world_state()`)
- Automatically retrieves character's current state via `character.get_state()`
- Enriches state with world context from graph manager when available
- Provides robust fallback to basic character state if graph manager unavailable
- Returns unified `State` object combining character and world information

### 2. Dynamic Action Retrieval (`get_available_actions()`)
- Queries graph manager for context-specific actions when available
- Falls back to strategy manager for daily actions
- Provides basic fallback actions (Rest, Idle) if other sources unavailable
- Converts different action formats to consistent Action objects

### 3. Enhanced Planning (`plan_actions()` & `plan_for_character()`)
- Automatically retrieves current state and actions if not provided
- Maintains backwards compatibility with existing API
- Adds new `plan_for_character()` convenience method for simple usage
- Includes detailed logging for planning process

### 4. Strategy Manager Integration
- Updated to properly initialize GOAP planner with graph manager
- Enables full integration with existing game systems
- Maintains existing LLM and utility-based decision making

## Usage Examples

### Basic Dynamic Planning
```python
from tiny_goap_system import GOAPPlanner

planner = GOAPPlanner(graph_manager)
plan = planner.plan_for_character(character, goal)
```

### Advanced Planning with Manual State/Actions
```python
# Still supports existing API
plan = planner.plan_actions(character, goal, current_state, actions)

# Or let planner retrieve automatically
plan = planner.plan_actions(character, goal)  # Auto-retrieves state and actions
```

## Test Results
- ✅ Dynamic state retrieval working
- ✅ Dynamic action retrieval with fallbacks working  
- ✅ Enhanced planning generates valid action sequences
- ✅ Backwards compatibility maintained
- ✅ Integration with existing game systems verified

## Example Planning Output
For a character with energy=20 wanting energy≥50:
```
Generated plan with 3 actions:
1. Rest (+10 energy)
2. Rest (+10 energy) 
3. Rest (+10 energy)
Result: 20 + 30 = 50 energy (goal achieved)
```

## Error Handling
- Graceful fallback when graph manager unavailable
- Warning messages for missing dependencies
- Always provides basic actions as last resort
- Maintains system stability even with component failures

The planner now fully satisfies the requirement to "dynamically receive the current world state and available actions for the character."