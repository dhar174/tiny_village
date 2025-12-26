# Event-Driven Storytelling System Documentation

## Overview

The Event-Driven Storytelling System enhances TinyVillage with dynamic narrative capabilities that create meaningful story experiences based on character actions and world changes. This system enables AI characters to experience coherent storylines that emerge naturally from their autonomous behaviors.

## Core Components

### 1. StorytellingEventHandler (`tiny_storytelling_engine.py`)

The main storytelling engine that extends the existing EventHandler with narrative capabilities.

```python
from tiny_storytelling_engine import StorytellingEventHandler

# Create storytelling handler
story_handler = StorytellingEventHandler(graph_manager, time_manager)
```

**Key Features:**
- Character action monitoring and triggers
- World state change tracking
- Story template system
- Mini-storyline creation
- Narrative context management

### 2. Character Action Triggers

Monitor character actions and trigger story events based on configurable conditions.

```python
from tiny_storytelling_engine import CharacterActionTrigger

# Create a romance trigger
romance_trigger = CharacterActionTrigger(
    action_names={"talk", "flirt", "compliment"},
    character_conditions={
        "happiness": {"operator": ">=", "value": 50}
    },
    location_conditions={"location_name": "Village Square"},
    cooldown_hours=12,
    probability=0.7
)

# Add to handler
story_handler.add_character_action_monitor("romance", romance_trigger)
```

### 3. Story Templates and Types

Pre-defined story templates for different narrative scenarios:

```python
# Story types available
StoryEventType.ROMANCE
StoryEventType.ADVENTURE
StoryEventType.MYSTERY
StoryEventType.HEROIC_JOURNEY
StoryEventType.CONFLICT
StoryEventType.FRIENDSHIP
# ... and more

# Create event from template
festival_event = story_handler.create_story_event_from_template(
    "village_festival", 
    "Summer Festival", 
    [character1, character2]
)
```

### 4. Story Chains for Mini-Storylines

Create connected events that form coherent narratives:

```python
# Define story chain events
romance_chain = [
    {
        "delay_hours": 0,
        "importance": 5,
        "effects": [{"type": "attribute_change", "targets": ["participants"], 
                    "attribute": "romantic_interest", "change_value": 10}]
    },
    {
        "delay_hours": 24,
        "importance": 6,
        "effects": [{"type": "relationship_change", "targets": ["participants"],
                    "attribute": "relationship_depth", "change_value": 15}]
    }
]

# Create the story chain
story_events = story_handler.create_story_chain(
    "alice_bob_romance",
    romance_chain,
    [alice, bob]
)
```

## Integration Layer (`storytelling_integration.py`)

The integration layer provides easy connection with existing game systems.

### Basic Usage

```python
from storytelling_integration import StorytellingGameIntegration

# Create integration
integration = StorytellingGameIntegration(graph_manager, time_manager)

# Monitor character action
triggered_events = integration.monitor_character_action(
    character, 
    "explore", 
    {"target": "forest"}, 
    {"success": True, "discovery": "artifact"}
)

# Track world state change
world_events = integration.track_world_state_change(
    "economic", 
    [village, characters], 
    magnitude=75, 
    description="Market crash affects village economy"
)
```

### Character Development Arcs

Create multi-part storylines for character growth:

```python
# Available arc types: "romance", "hero_journey", "redemption", "friendship"
hero_arc = integration.create_character_arc_storyline(
    character, 
    "hero_journey", 
    duration_days=10
)
```

## Story Event Types and Narrative Impact

### Story Event Types
- **ROMANCE**: Love stories and relationships
- **ADVENTURE**: Exploration and discovery
- **MYSTERY**: Investigations and puzzles
- **HEROIC_JOURNEY**: Hero's journey narratives
- **CONFLICT**: Tensions and disputes
- **FRIENDSHIP**: Social bonds and cooperation
- **TRAGEDY**: Loss and sorrow
- **COMEDY**: Humor and lighthearted events
- **DISCOVERY**: Learning and revelations
- **BETRAYAL**: Trust breaking and deception
- **REDEMPTION**: Forgiveness and second chances
- **COMING_OF_AGE**: Growth and maturation

### Narrative Impact Levels
- **MINOR (1)**: Small character moments
- **MODERATE (2)**: Notable events affecting one character
- **SIGNIFICANT (3)**: Events affecting multiple characters
- **MAJOR (4)**: Village-wide or life-changing events
- **LEGENDARY (5)**: Epic events that become village lore

## Character Action Monitoring

The system monitors these character actions for story triggers:

### Romance Triggers
- Actions: talk, flirt, compliment, gift
- Conditions: happiness ≥ 50, energy ≥ 30
- Location: social areas
- Cooldown: 12 hours

### Adventure Triggers
- Actions: explore, investigate, search, venture
- Conditions: curiosity ≥ 60, bravery ≥ 40
- Cooldown: 24 hours

### Conflict Triggers
- Actions: argue, confront, challenge, disagree
- Conditions: stress ≥ 70
- Cooldown: 6 hours

### Friendship Triggers
- Actions: help, support, assist, collaborate
- Conditions: empathy ≥ 50
- Cooldown: 18 hours

## World State Change Tracking

Monitor significant world changes that trigger narrative events:

```python
from tiny_storytelling_engine import WorldStateChange

change = WorldStateChange(
    change_type="natural_disaster",
    affected_entities=[village, characters],
    magnitude=90,
    timestamp=datetime.now(),
    description="Severe storm threatens village"
)

story_handler.track_world_state_change(change)
```

## Example: Complete Romance Storyline

```python
# 1. Set up characters
alice = create_character("Alice", happiness=75, empathy=80)
bob = create_character("Bob", happiness=70, empathy=75)

# 2. Create integration
integration = StorytellingGameIntegration()

# 3. Monitor initial interaction
events = integration.monitor_character_action(
    alice, "talk", {"target": bob}, {"success": True}
)

# 4. Create romance arc
romance_arc = integration.create_character_arc_storyline(
    alice, "romance", duration_days=7
)

# 5. Continue monitoring actions
integration.monitor_character_action(
    alice, "compliment", {"target": bob}, {"success": True}
)

# 6. Get story summary
status = integration.get_character_story_status(alice)
print(f"Alice's romance story: {status['story_summary']['total_story_events']} events")
```

## Getting Story Information

### Character Story Summary
```python
summary = story_handler.get_character_story_summary(character)
# Returns: total events, event list, active threads
```

### Village Narrative Status
```python
village_status = integration.get_village_narrative_status()
# Returns: event statistics, narrative summary, active storylines
```

### Active Story Threads
```python
threads = story_handler.get_active_story_threads()
# Returns: all active story chains with status and participants
```

### Narrative Summary
```python
summary = story_handler.generate_narrative_summary(timeframe_days=7)
# Returns: story themes, character involvement, narrative momentum
```

## Best Practices

### 1. Character Attribute Setup
Ensure characters have the necessary attributes for story triggers:
```python
character.happiness = 60
character.curiosity = 70
character.bravery = 50
character.empathy = 65
character.intelligence = 75
character.stress = 30
```

### 2. Balanced Trigger Probabilities
- High frequency actions (talk): 0.3-0.5 probability
- Medium frequency actions (help): 0.5-0.7 probability  
- Rare actions (heroic): 0.8-1.0 probability

### 3. Appropriate Cooldowns
- Social interactions: 6-12 hours
- Adventure actions: 24-48 hours
- Major events: 72+ hours

### 4. Story Chain Design
- Start with low impact, build to climax
- Include variety in effects (positive and negative)
- Consider character development over time
- Plan for meaningful resolutions

## Testing

Run the comprehensive test suite:
```bash
python test_storytelling_engine.py
```

Run the interactive demo:
```bash
python demo_storytelling_system.py
```

## Integration with Existing Systems

### With Character Actions
```python
# In your action execution code:
def execute_character_action(character, action_name, params):
    # Execute the action normally
    result = original_action_execution(character, action_name, params)
    
    # Monitor for story triggers
    story_events = integration.monitor_character_action(
        character, action_name, params, result
    )
    
    return result, story_events
```

### With World Events
```python
# When significant world changes occur:
def handle_world_change(change_type, entities, magnitude, description):
    # Handle the change normally
    handle_original_change(change_type, entities, magnitude)
    
    # Track for story events
    story_events = integration.track_world_state_change(
        change_type, entities, magnitude, description
    )
    
    return story_events
```

### With Game Loop
```python
# In your main game loop:
def game_update():
    # Regular game updates
    update_characters()
    update_world()
    
    # Generate daily story events
    daily_story = integration.generate_daily_story_events()
    
    # Process any story events
    integration.story_handler.process_events()
```

This system creates dynamic, character-driven narratives that emerge naturally from gameplay, making each AI character's experience unique and meaningful within the village simulation.