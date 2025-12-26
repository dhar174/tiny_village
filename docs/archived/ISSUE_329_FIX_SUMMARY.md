# Fix for Issue #329: MockMotives Hardcoded Values

## Problem
The nested MockMotives class in test files returned hardcoded values (50, 30, 70, 50) which could mask bugs in PromptBuilder logic. These fixed values didn't accurately test the PromptBuilder's behavior with different motive values, potentially allowing tests to pass even when the PromptBuilder had logical errors in processing character data.

## Solution
Replaced the hardcoded MockMotives implementation with a configurable system that calculates motive values based on the character's actual state.

### Key Changes

1. **State-based calculations**: Motive values now derive from character attributes:
   - Health motive based on `character.health_status`
   - Wealth motive based on `character.wealth_money`  
   - Mental health motive based on `character.mental_health`
   - Social motive based on `character.social_wellbeing`
   - And similar patterns for other motives

2. **Realistic relationships**: Characters with poor conditions have higher motives (more urgent needs):
   - Low health → High health motive
   - Low wealth → High wealth motive
   - High hunger level → High hunger motive

3. **Bounds checking**: All motive values constrained to reasonable range (10-100)

4. **No hardcoded fallbacks**: Even standalone mode avoids the problematic (50, 30, 70, 50) values

### Example Behavior

```python
# Unhealthy character
unhealthy = MockCharacter()
unhealthy.health_status = 1    # Poor health
unhealthy.wealth_money = 1     # Poor wealth
unhealthy.hunger_level = 10    # Very hungry

motives = unhealthy.get_motives()
print(motives.get_health_motive())  # ~92 (high urgency)
print(motives.get_wealth_motive())  # ~99 (high urgency)
print(motives.get_hunger_motive())  # ~90 (high urgency)

# Healthy character  
healthy = MockCharacter()
healthy.health_status = 10     # Perfect health
healthy.wealth_money = 100     # Rich
healthy.hunger_level = 1       # Well fed

motives = healthy.get_motives()
print(motives.get_health_motive())  # ~20 (low urgency)
print(motives.get_wealth_motive())  # ~10 (low urgency)  
print(motives.get_hunger_motive())  # ~18 (low urgency)
```

### Test Coverage

Created comprehensive tests that verify:
- ✅ Motives vary with character state
- ✅ No hardcoded values (50, 30, 70, 50) are returned
- ✅ All motive values within reasonable bounds (10-100)
- ✅ PromptBuilder handles varying motive scenarios correctly
- ✅ Character state changes affect motive calculations
- ✅ Standalone mode works without returning problematic values

### Benefits

1. **Better test coverage**: Tests will now fail if PromptBuilder doesn't handle varying inputs correctly
2. **Realistic scenarios**: Character state directly influences motive calculations
3. **Debugging capability**: Different character configurations produce different test scenarios
4. **Maintainability**: No magic numbers, calculations are explicit and documented

## Files Changed

- `test_crisis_prompt.py`: Enhanced with configurable MockMotives and MockCharacter
- `test_mock_motives_configurable.py`: Comprehensive test suite validating the fix

## Verification

All tests pass (9 test methods) and demonstrate that:
1. MockMotives no longer returns hardcoded values
2. Different character states produce different motive values  
3. PromptBuilder successfully processes varying motive scenarios
4. The fix addresses the original issue while maintaining backward compatibility