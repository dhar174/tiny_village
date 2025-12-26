# Event Effect Schema v2 Documentation

## Overview

Event Effect Schema v2 provides a typed, validated system for defining and applying event effects in the Tiny Village game. This replaces the previous ad-hoc dictionary-based approach with a structured, type-safe system that ensures effects are well-formed and applied consistently.

## Key Components

### 1. EffectV2 Class

The core data structure for defining effects with full validation.

```python
from effect_schema import EffectV2, EffectType, EffectCondition, OperatorType

effect = EffectV2(
    type=EffectType.ATTRIBUTE_CHANGE,
    targets=["participants"],
    attribute="happiness",
    change_value=10,
    description="Increases happiness by 10"
)
```

### 2. Effect Types

Supported effect types (enum `EffectType`):
- `ATTRIBUTE_CHANGE`: Modify character attributes
- `RELATIONSHIP_CHANGE`: Modify relationships between characters
- `LOCATION_CHANGE`: Modify location attributes
- `WORLD_STATE_CHANGE`: Modify global world state

### 3. Operators

Supported operators for attribute modification (enum `OperatorType`):
- `ADD`: Add value to current value (default)
- `SUBTRACT`: Subtract value from current value
- `MULTIPLY`: Multiply current value by factor
- `SET`: Set value directly
- `MIN`: Set to minimum of current and new value
- `MAX`: Set to maximum of current and new value

### 4. EffectDispatcher

Central system for applying effects consistently.

```python
from effect_dispatcher import EffectDispatcher

dispatcher = EffectDispatcher(graph_manager)
success = dispatcher.apply_effect(effect, event)
```

## Canonical Examples

### Example 1: Simple Happiness Boost

```python
happiness_boost = EffectV2(
    type=EffectType.ATTRIBUTE_CHANGE,
    targets=["participants"],
    attribute="happiness",
    change_value=10,
    description="Increases participant happiness by 10"
)
```

**Use case**: Village festival, social gathering, receiving good news

### Example 2: Conditional Energy Drain

```python
from effect_schema import EffectCondition

energy_drain = EffectV2(
    type=EffectType.ATTRIBUTE_CHANGE,
    targets=["participants"],
    attribute="energy",
    change_value=-5,
    conditions=[EffectCondition("energy", ">=", 10)],
    description="Drains 5 energy from participants with at least 10 energy"
)
```

**Use case**: Work events, physical activities (only affects characters with sufficient energy)

### Example 3: Relationship Trust Boost with Chaining

```python
trust_boost = EffectV2(
    type=EffectType.RELATIONSHIP_CHANGE,
    targets=["participants"],
    attribute="trust",
    change_value=5,
    chain=["friendship_level", "loyalty"],
    description="Increases trust and chains to friendship and loyalty"
)
```

**Use case**: Shared experiences, helping each other, collaborative projects

### Example 4: Location Development

```python
location_development = EffectV2(
    type=EffectType.LOCATION_CHANGE,
    targets=["location"],
    attribute="development_level",
    change_value=2,
    operator=OperatorType.ADD,
    description="Increases location development level by 2"
)
```

**Use case**: Building projects, infrastructure improvements

### Example 5: World Economy Boost (Stacking)

```python
economy_boost = EffectV2(
    type=EffectType.WORLD_STATE_CHANGE,
    targets=["world"],
    attribute="economic_activity",
    change_value=15,
    stacking=True,
    description="Increases global economic activity by 15 (stacks)"
)
```

**Use case**: Market days, trade caravans, economic events

## Validation

Effects are automatically validated upon creation. Invalid effects will raise `ValueError` with descriptive error messages:

```python
# Missing required field
try:
    invalid_effect = EffectV2(
        type=EffectType.ATTRIBUTE_CHANGE,
        targets=[],  # Empty targets list - invalid!
        attribute="happiness"
    )
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: "Effect must have at least one target"
```

### Validation Rules

1. **Effect Type**: Must be one of the defined `EffectType` enum values
2. **Targets**: Must be a non-empty list of strings
3. **Attribute**: Must be a non-empty string
4. **Change Value**: Must be numeric for ADD, SUBTRACT, MULTIPLY operators
5. **Operator**: Must be one of the defined `OperatorType` enum values
6. **Conditions**: Must be a list of valid `EffectCondition` objects
7. **Chain**: Must be a list of attribute name strings

## Backward Compatibility

The system maintains backward compatibility with the old dictionary-based format:

```python
# Old format (still works)
old_effect = {
    "type": "attribute_change",
    "targets": ["participants"],
    "attribute": "happiness",
    "change_value": 10
}

# Automatically converted to EffectV2
effect = EffectV2.from_dict(old_effect)
```

## Usage in Event Handler

Effects are automatically processed by the `EventHandler`:

```python
from tiny_event_handler import Event, EventHandler

# Create event with effects (using dict or EffectV2)
event = Event(
    name="Village Festival",
    date=datetime.now(),
    event_type="social",
    importance=8,
    impact=5,
    effects=[
        {
            "type": "attribute_change",
            "targets": ["participants"],
            "attribute": "happiness",
            "change_value": 15
        }
    ]
)

# EventHandler automatically validates and applies effects
handler = EventHandler(graph_manager)
handler.add_event(event)
results = handler.process_events()
```

## Advanced Features

### Conditional Effects

Effects can have conditions that must be met for the effect to apply:

```python
conditional_effect = EffectV2(
    type=EffectType.ATTRIBUTE_CHANGE,
    targets=["participants"],
    attribute="productivity",
    change_value=10,
    conditions=[
        EffectCondition("energy", ">=", 50),
        EffectCondition("mood", "==", "motivated")
    ]
)
```

**Condition Operators**: `>=`, `>`, `<=`, `<`, `==`, `!=`

### Chained Effects

Effects can cascade to related attributes:

```python
chained_effect = EffectV2(
    type=EffectType.ATTRIBUTE_CHANGE,
    targets=["participants"],
    attribute="skill",
    change_value=5,
    chain=["experience", "confidence"]  # Also increases these
)
```

### Effect Priority

Effects with higher priority are applied first:

```python
high_priority_effect = EffectV2(
    type=EffectType.ATTRIBUTE_CHANGE,
    targets=["participants"],
    attribute="health",
    change_value=20,
    priority=10  # Applied before lower priority effects
)
```

## Error Handling

The system provides clear error messages for invalid effects:

```python
# Invalid effect type
try:
    EffectV2(
        type="invalid_type",
        targets=["participants"],
        attribute="happiness"
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Invalid effect type: invalid_type. Must be one of ['attribute_change', ...]"

# Invalid operator
try:
    EffectV2(
        type=EffectType.ATTRIBUTE_CHANGE,
        targets=["participants"],
        attribute="score",
        change_value=10,
        operator="invalid_operator"
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Invalid operator: invalid_operator. Must be one of ['add', ...]"
```

## Testing

The system includes comprehensive tests covering:
- Valid effect creation and validation
- Invalid inputs (missing fields, wrong types)
- Effect application to different entity types
- Conditional effects
- Chained effects
- Backward compatibility with old dict format

Run tests:
```bash
python tests/test_effect_schema_v2.py
```

## Migration Guide

### From Old Dict Format to EffectV2

**Old way:**
```python
effect = {
    "type": "attribute_change",
    "targets": ["participants"],
    "attribute": "happiness",
    "change_value": 10
}
```

**New way:**
```python
effect = EffectV2(
    type=EffectType.ATTRIBUTE_CHANGE,
    targets=["participants"],
    attribute="happiness",
    change_value=10
)
```

**Benefits:**
- Type safety and IDE auto-completion
- Automatic validation on creation
- Clear error messages for invalid configurations
- No need to remember dict key names

### Updating Event Templates

Event templates can now use EffectV2 directly or continue using dicts:

```python
# Both formats work
template = {
    "effects": [
        # Using EffectV2
        EffectV2(
            type=EffectType.ATTRIBUTE_CHANGE,
            targets=["participants"],
            attribute="happiness",
            change_value=10
        ),
        # Using dict (backward compatible)
        {
            "type": "relationship_change",
            "targets": ["participants"],
            "attribute": "trust",
            "change_value": 5
        }
    ]
}
```

## API Reference

### EffectV2

**Required Parameters:**
- `type` (EffectType): The type of effect
- `targets` (List[str]): List of target specifications
- `attribute` (str): The attribute to modify

**Optional Parameters:**
- `change_value` (Union[int, float]): Value to apply (default: 0)
- `operator` (OperatorType): How to apply the value (default: ADD)
- `conditions` (List[EffectCondition]): Conditions for effect application
- `stacking` (bool): Whether multiple instances stack (default: True)
- `chain` (List[str]): Attributes to cascade the effect to
- `description` (str): Human-readable description
- `priority` (int): Application priority (default: 0)

**Methods:**
- `validate()`: Validate the effect configuration
- `should_apply(entity)`: Check if conditions are met for an entity
- `to_dict()`: Convert to dictionary format
- `from_dict(data)`: Create from dictionary (class method)

### EffectDispatcher

**Constructor:**
- `__init__(graph_manager=None)`: Initialize with optional GraphManager

**Methods:**
- `apply_effect(effect, event, context=None)`: Apply an effect
- `get_applied_effects_summary()`: Get statistics on applied effects
- `clear_log()`: Clear the applied effects log

## Best Practices

1. **Use EffectType enum** instead of strings for type safety
2. **Add descriptions** to effects for debugging and documentation
3. **Use conditions** to make effects contextual and realistic
4. **Set appropriate priorities** for effects that depend on order
5. **Chain related attributes** to create realistic cascading effects
6. **Validate early** by creating EffectV2 objects when defining events
7. **Test thoroughly** with various entity states and conditions

## Support and Issues

For questions or issues with the Effect Schema v2 system:
1. Check this documentation first
2. Review the canonical examples
3. Check the test suite for usage patterns
4. Create an issue on the repository with:
   - Description of the problem
   - Example code demonstrating the issue
   - Expected vs actual behavior
