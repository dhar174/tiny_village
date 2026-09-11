# Memory Testing Best Practices

## Issue Description

The original issue (#432) identified a problematic testing pattern where `MagicMock` objects were used to simulate memory objects in PromptBuilder tests. This antipattern can hide integration bugs because MagicMock allows access to any attribute or method, even those that don't exist in real memory objects.

**Problematic Pattern (from issue):**
```python
pb = PromptBuilder(self.character)
mem1 = MagicMock(description="Met Bob yesterday", importance_score=5)
```

## Why MagicMock is Problematic for Memory Testing

1. **False Positives**: Tests pass even when memory processing logic is broken
2. **Missing Validation**: Doesn't validate actual memory object structure
3. **Hidden Bugs**: Allows access to non-existent attributes/methods
4. **Poor Test Coverage**: Doesn't catch integration failures between PromptBuilder and memory system

## Correct Testing Patterns

### 1. Simple Memory Mock (Current Good Pattern)

For basic tests that only need description:

```python
class MockMemory:
    def __init__(self, description):
        self.description = description

memories = [MockMemory("won a pie contest"), MockMemory("lost keys at market")]
```

### 2. Realistic Memory Objects (Recommended)

For comprehensive testing that validates memory integration:

```python
class RealSpecificMemory:
    def __init__(self, description, importance_score):
        self.description = description
        self.importance_score = importance_score
        self.creation_time = datetime.now()
        self.last_access_time = self.creation_time

class RealGeneralMemory:
    def __init__(self, description):
        self.description = description
        self.creation_time = datetime.now()
        # Note: No importance_score for general memories
```

## PromptBuilder Memory Integration

The PromptBuilder processes memories using this pattern (from `tiny_prompt_builder.py:2182-2186`):

```python
if memories:
    prompt += "\nRecent memories influencing you:\n"
    for mem in memories[:2]:
        desc = getattr(mem, "description", str(mem))
        prompt += f"- {desc}\n"
```

## Test Coverage Requirements

When testing memory integration, ensure coverage of:

1. **Basic Description Access**: `getattr(mem, "description", str(mem))`
2. **Importance Score Handling**: Both specific (has score) and general (no score) memories
3. **Attribute Fallbacks**: Graceful handling when attributes are missing
4. **Edge Cases**: Empty lists, None values, malformed objects
5. **Memory Limiting**: Only first 2 memories are used in prompts

## Example Test Implementation

```python
def test_prompt_builder_memory_integration(self):
    """Test PromptBuilder with realistic memory objects."""
    
    # Create realistic memory objects
    specific_mem = RealSpecificMemory("Met Bob yesterday", 5)
    general_mem = RealGeneralMemory("Weather was nice")
    
    memories = [specific_mem, general_mem]
    
    # Test PromptBuilder memory formatting
    prompt = self.prompt_builder.generate_decision_prompt(
        time="morning",
        weather="sunny", 
        action_choices=["1. Test action"],
        memories=memories
    )
    
    # Validate memory integration
    self.assertIn("Recent memories influencing you:", prompt)
    self.assertIn("Met Bob yesterday", prompt)
    self.assertIn("Weather was nice", prompt)
    
    # Test edge cases
    with self.assertRaises(AttributeError):
        _ = specific_mem.nonexistent_attribute  # Should fail properly
```

## Migration Guide

If you find tests using MagicMock for memory objects:

1. **Identify the Pattern**: Look for `MagicMock(description=..., importance_score=...)`
2. **Replace with Real Objects**: Use `RealSpecificMemory` or `RealGeneralMemory`
3. **Add Validation Tests**: Ensure invalid attribute access raises AttributeError
4. **Test Edge Cases**: Include broken/malformed memory object handling
5. **Verify Integration**: Test the actual PromptBuilder memory processing logic

## Files Demonstrating Correct Patterns

- `tests/test_prompt_builder_memory_integration.py` - Comprehensive memory testing examples
- `tests/test_decision_prompt_memories.py` - Simple MockMemory pattern (already good)
- `test_memory_magicmock_issue.py` - Demonstrates the antipattern and solution

## Benefits of Proper Memory Testing

1. **Accurate Integration Testing**: Tests actually validate memory-PromptBuilder interaction
2. **Bug Detection**: Catches attribute access errors and processing logic bugs  
3. **Better Coverage**: Tests both successful and failure scenarios
4. **Maintainability**: Changes to memory structure are reflected in tests
5. **Documentation**: Tests serve as examples of correct memory object usage

## Summary

Replace MagicMock memory objects with realistic memory classes that match the actual `tiny_memories.py` structure. This ensures tests accurately validate the memory integration functionality and catch bugs that MagicMock would hide.