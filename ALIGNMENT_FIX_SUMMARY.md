# StrategyManager and GOAPPlanner Alignment Fix

## Summary
This fix addresses the alignment issue between StrategyManager and GOAPPlanner, ensuring that StrategyManager can properly request and receive valid plans from GOAPPlanner.

## Issues Addressed

### 1. Import Dependency Issues
**Problem**: StrategyManager had hard dependencies on modules that might not be available (networkx, numpy, etc.)
**Solution**: Made all heavy dependencies optional with graceful fallbacks
- GraphManager (requires networkx) → Optional with None fallback
- Character class (requires numpy) → Optional with simplified character handling
- LLM components → Optional with warning messages

### 2. Interface Misalignment
**Problem**: StrategyManager methods called GOAPPlanner with incorrect interfaces
**Solution**: Fixed all method calls to use proper signatures
- `plan_daily_activities()` now correctly calls `goap_planner.plan_actions(character, goal, state, actions)`
- `update_strategy()` now properly creates Goal and State objects
- `respond_to_job_offer()` now uses correct GOAP interface

### 3. Goal Object Creation
**Problem**: StrategyManager created Goal objects with wrong constructor parameters
**Solution**: Updated Goal instantiation to use correct parameters
- Changed from `completion_conditions` to `target_effects`
- Added support for `target_effects` in GOAP goal satisfaction checking

### 4. GOAP Planning Robustness
**Problem**: GOAP planner could hang in infinite loops when no plan exists
**Solution**: Added safety measures
- Maximum iteration limit (1000 iterations)
- Maximum plan length limit (10 actions)
- Better error handling and warnings

## Key Changes

### In `tiny_strategy_manager.py`:
1. **Optional imports** with graceful degradation
2. **Fixed method interfaces** for GOAP planning
3. **Proper Goal object creation** using `target_effects`
4. **Enhanced character state handling** for different input types
5. **Removed duplicate methods** and inconsistencies

### In `tiny_goap_system.py`:
1. **Added support for `target_effects`** in goal satisfaction checking
2. **Added loop prevention** with iteration and plan length limits
3. **Enhanced error handling** with warnings

## Validation

### Test Coverage:
- ✅ Basic import and instantiation tests
- ✅ Integration tests between StrategyManager and GOAPPlanner
- ✅ Goal creation and planning tests
- ✅ Realistic planning scenarios with achievable goals
- ✅ Edge cases (impossible goals, empty action lists)

### Results:
- **StrategyManager can successfully request plans** from GOAPPlanner
- **GOAPPlanner returns valid plans** when goals are achievable
- **Interface methods work correctly** with proper parameter passing
- **No infinite loops** or hanging issues
- **Graceful handling** of missing dependencies

## Interface Methods Working:

```python
# These methods now work correctly:
strategy_manager = StrategyManager()

# 1. Daily activity planning
plan = strategy_manager.plan_daily_activities(character)

# 2. Event-based strategy updates  
result = strategy_manager.update_strategy(events, character)

# 3. Job offer responses
decision = strategy_manager.respond_to_job_offer(character, job_details)

# 4. Direct GOAP planning
actions = strategy_manager.get_daily_actions(character)
goal = Goal(name="test", target_effects={"energy": 8}, priority=0.8)
state = State(strategy_manager.get_character_state_dict(character))
plan = strategy_manager.goap_planner.plan_actions(character, goal, state, actions)
```

## Backwards Compatibility
- All existing functionality is preserved
- Optional dependencies mean the system works even with minimal installations
- Graceful degradation ensures no functionality is completely broken

## Future Improvements
- Consider adding more sophisticated goal satisfaction checking
- Implement priority-based action selection in GOAP planning
- Add caching for repeated planning requests
- Enhanced error reporting and debugging information