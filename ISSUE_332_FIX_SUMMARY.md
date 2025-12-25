# Fix for Issue #332: Remove Hardcoded Social Wellbeing Constants from TalkAction

## Problem Statement

The TalkAction class contained hardcoded constants for social wellbeing increments in its default effects:
- Target social_wellbeing increment: `1` (line 875)
- Initiator social_wellbeing increment: `0.5` (line 880)

This created tight coupling between the mock/test implementation and the actual `respond_to_talk` method, contradicting the goal of making tests less coupled to implementation details. If the real `respond_to_talk` method changed its increment values, the hardcoded mock values would become inconsistent, potentially masking bugs.

## Solution Implemented

### 1. Removed Hardcoded Default Effects
**Before:**
```python
_effects = (
    effects
    if effects is not None
    else [
        {
            "targets": ["target"],
            "attribute": "social_wellbeing",
            "change_value": 1,  # HARDCODED
        },
        {
            "targets": ["initiator"],
            "attribute": "social_wellbeing",
            "change_value": 0.5,  # HARDCODED
        }
    ]
)
```

**After:**
```python
_effects = (
    effects
    if effects is not None
    else [
        # No default social_wellbeing effects - let respond_to_talk method handle this
        # to avoid coupling between hardcoded values and actual implementation
    ]
)
```

### 2. Fixed Graph Manager Parameter Passing
Added `"graph_manager"` to the allowed parameters in TalkAction constructor to ensure proper graph updates when custom effects are used.

### 3. Updated Tests for Better Decoupling
- Modified existing tests to use explicit custom effects rather than relying on hardcoded defaults
- Added new test specifically validating the decoupling behavior
- Tests now verify that:
  - Default TalkAction applies no hardcoded effects
  - The `respond_to_talk` method is called and handles social wellbeing changes
  - Custom effects still work when explicitly provided

## Benefits

1. **Decoupling**: TalkAction is no longer coupled to specific social_wellbeing increment values
2. **Flexibility**: The `respond_to_talk` method can implement its own logic without conflicts
3. **Bug Prevention**: Changes to `respond_to_talk` won't be masked by inconsistent hardcoded values
4. **Maintainability**: Tests focus on behavior rather than specific numeric values

## Verification

The fix is verified through:
- Existing tests continue to pass
- New comprehensive tests validate the decoupling
- Demo script shows real-world usage scenarios
- All TalkAction functionality remains intact while removing the coupling

## Files Modified

1. `actions.py`: Removed hardcoded default effects, fixed graph_manager parameter
2. `test_actions.py`: Updated tests to be less coupled to hardcoded values
3. `test_talk_action_decoupling.py`: New comprehensive tests (created)
4. `demo_talk_action_fix.py`: Demo script showing the fix in action (created)

This change successfully resolves issue #332 by eliminating the tight coupling between TalkAction's hardcoded effects and the actual `respond_to_talk` implementation.