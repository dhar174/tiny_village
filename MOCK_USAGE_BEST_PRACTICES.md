# Mock() Usage Best Practices for Action and Utility Testing

## Issue Summary

The issue #423 identified a problem with using `Mock()` objects for testing action objects in utility calculations. The mock objects might not behave the same way as actual action instances, potentially allowing tests to pass when they should fail.

## The Problem

When using `Mock()` for action objects in utility tests:

```python
# PROBLEMATIC APPROACH
action = Mock()
action.name = "Test Action"
# Mock() automatically creates mock objects for any attribute access
# action.cost and action.effects will be Mock objects, not real values

utility = calculate_action_utility(char_state, action, current_goal=goal)
```

### Why This Is Problematic

1. **False Positives**: Tests might pass even when the utility function expects specific data types
2. **No Interface Enforcement**: Mock() returns other Mock() objects for attribute access
3. **Misleading Results**: The test doesn't validate actual behavior with production code
4. **Hidden Bugs**: Real issues might be masked by overly permissive mocks

## The Solutions

### Option 1: Use Real Action Objects (Preferred)

```python
# CORRECT APPROACH - Real Action Object
action = Action(
    name="EatFood",
    preconditions={},
    effects=[{"attribute": "hunger", "change_value": -0.7}],
    cost=0.1
)

utility = calculate_action_utility(char_state, action, current_goal=goal)
```

**Benefits:**
- Tests actual production code behavior
- Catches interface mismatches
- Validates real integration between components

### Option 2: Use Proper Test Classes

When real classes have complex dependencies, create test classes that match the interface:

```python
# CORRECT APPROACH - Test Class with Proper Interface
from tests.test_tiny_utility_functions import MockAction

action = MockAction(
    name="EatFood",
    cost=0.1,
    effects=[{"attribute": "hunger", "change_value": -0.7}]
)

utility = calculate_action_utility(char_state, action, current_goal=goal)
```

**Benefits:**
- Avoids complex dependencies while maintaining interface fidelity
- Explicit about what attributes are expected
- Tests fail if interface changes

## When Mock() Is Appropriate

Mock() should be used for:

1. **External Dependencies**: Database connections, API clients, file systems
2. **Complex Systems**: Components that are not the focus of the test
3. **Side Effects**: When you need to verify method calls were made

```python
# APPROPRIATE Mock() usage - for dependencies, not main objects
mock_graph_manager = Mock(spec=GraphManager)
mock_character = Mock()
mock_character.uuid = "test_char"
mock_character.name = "Test Character"
```

## Guidelines

### DO:
- Use real objects for the main components being tested
- Use Mock() with `spec=` parameter to enforce interface
- Create proper test classes when real objects have complex dependencies
- Write tests that will fail if the expected behavior is broken

### DON'T:
- Use Mock() for the primary objects being tested (actions, goals, etc.)
- Rely on Mock()'s automatic attribute creation
- Write tests that pass regardless of actual functionality
- Over-mock components that could be tested with real objects

## Examples of Fixed Tests

See the following files for examples of proper testing approaches:

- `test_fix_validation.py` - Demonstrates the issue and correct solutions
- `test_utility_with_real_classes.py` - Uses real Action and Goal classes
- `test_tiny_utility_functions.py` - Defines proper MockAction and MockGoal classes
- `tests/test_gameplay_controller.py` - Fixed to use real Action objects

## Implementation Notes

The fixes in this repository include:

1. **Fixed `test_gameplay_controller.py`**: Replaced `Mock()` action with real `Action` object
2. **Created demonstration files**: Show the problem and solutions
3. **Added documentation**: Explains when and how to use Mock() properly
4. **Preserved existing good patterns**: MockAction and MockGoal classes are good examples

## Testing Your Changes

To verify your tests are using the correct approach:

1. Run the test with the real object
2. Temporarily break the functionality being tested
3. Verify the test fails as expected
4. If the test still passes, you're probably over-mocking

Remember: **Good tests fail when there's an error, not when they're designed to always pass!**