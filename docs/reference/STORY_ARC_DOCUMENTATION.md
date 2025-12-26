# StoryArc System Documentation

## Overview

The StoryArc system provides narrative progression tracking for the Tiny Village simulation. It allows the game to manage story elements and track their progression through different narrative phases.

## Purpose

The StoryArc system was created to resolve the issue where references to `STARTING_THRESHOLD`, `DEVELOPING_THRESHOLD`, and `CLIMAX_THRESHOLD` constants were made but these constants were not defined, causing AttributeError at runtime.

## Constants

The StoryArc class defines three key constants for narrative progression:

- `STARTING_THRESHOLD = 0.2` - Marks the transition from setup to rising action
- `DEVELOPING_THRESHOLD = 0.6` - Marks the transition from rising action to climax
- `CLIMAX_THRESHOLD = 0.9` - Marks the transition from climax to resolution

## Usage

### Basic StoryArc Creation

```python
from tiny_story_arc import StoryArc

# Create a new story arc
arc = StoryArc("Village Festival Preparation", importance=7)

# Check current phase
print(arc.phase)  # "setup"
print(arc.progression)  # 0.0

# Advance the story
arc.advance_progression(0.3)
print(arc.phase)  # "rising_action"
```

### Using Constants

```python
# Access the threshold constants
if arc.progression < StoryArc.STARTING_THRESHOLD:
    print("Story is in setup phase")
elif arc.progression < StoryArc.DEVELOPING_THRESHOLD:
    print("Story is in rising action")
elif arc.progression < StoryArc.CLIMAX_THRESHOLD:
    print("Story is at climax")
else:
    print("Story is in resolution")
```

### Adding Events and Characters

```python
# Add events to progress the story
class MockEvent:
    def __init__(self, importance):
        self.importance = importance

event = MockEvent(importance=8)
arc.add_event(event)  # Automatically advances progression

# Add characters involved in the story
arc.add_character("Alice")
arc.add_character("Bob")
```

### StoryArcManager

```python
from tiny_story_arc import StoryArcManager

# Create manager
manager = StoryArcManager()

# Create multiple story arcs
festival_arc = manager.create_arc("Festival", importance=8)
mystery_arc = manager.create_arc("Missing Merchant", importance=6)

# Get arcs by phase
setup_arcs = manager.get_arcs_in_phase("setup")
climax_arcs = manager.get_arcs_in_phase("climax")

# Get statistics
stats = manager.get_statistics()
print(f"Active arcs: {stats['active_arcs']}")
print(f"Completed arcs: {stats['completed_arcs']}")
```

## Integration with Event System

The StoryArc system is designed to integrate with the existing event system:

```python
from tiny_event_handler import Event
from tiny_story_arc import StoryArc

# Create event and story arc
event = Event(
    name="Harvest Festival",
    date="2024-09-01",
    event_type="social",
    importance=7,
    impact=5
)

arc = StoryArc("Autumn Celebrations")

# Add event to story arc (automatically advances progression)
arc.add_event(event)
```

## Story Phases

1. **Setup (0.0 - 0.2)**: Introduction and establishing elements
2. **Rising Action (0.2 - 0.6)**: Building tension and developing plot
3. **Climax (0.6 - 0.9)**: Peak tension and critical events
4. **Resolution (0.9 - 1.0)**: Conclusion and aftermath

## Key Features

- **Automatic phase tracking**: Phase is automatically updated when progression changes
- **Event integration**: Events can be added to story arcs and automatically advance progression
- **Character tracking**: Keep track of which characters are involved in each story arc
- **Completion handling**: Story arcs can be marked as completed with resolution types
- **Statistics**: Generate comprehensive statistics about story progression
- **Flexible progression**: Stories can advance based on events, time, or manual progression

## Error Prevention

This implementation prevents the original AttributeError by ensuring:

1. All three constants (`STARTING_THRESHOLD`, `DEVELOPING_THRESHOLD`, `CLIMAX_THRESHOLD`) are properly defined
2. Constants are accessible both from the class and instances
3. Constants have reasonable values in logical order
4. Proper type checking and validation

## Future Enhancements

The StoryArc system is designed to be extensible and could support:

- Complex branching narratives
- Multiple concurrent story arcs with interactions
- Character-specific story progression
- Integration with AI-driven narrative generation
- Save/load functionality for persistent stories
- Visual story progression displays