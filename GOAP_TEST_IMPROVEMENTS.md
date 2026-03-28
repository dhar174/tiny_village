## GOAPPlanner Test Improvements - Issue #392

### Problem Statement
The issue identified that creating a `GOAPPlanner` with `graph_manager=None` may not properly test the functionality since the planner likely requires a valid graph_manager to work correctly. This could lead to tests passing even when the planner doesn't function as intended with real data.

### Root Cause Analysis
Upon investigation, I found that:

1. **Multiple test files** were creating `GOAPPlanner(graph_manager=None)` which limited test coverage
2. The **GOAPPlanner relies on graph_manager** for several key functionalities:
   - Enhanced state retrieval via `get_character_state()`
   - Action retrieval via `get_possible_actions()`
   - Alternative action finding via `find_alternative_actions()`
   - Goal difficulty calculation via `calculate_goal_difficulty()`
   - Environment conditions via `get_environment_conditions()`

3. **Codebase bugs** were discovered during testing:
   - Missing `HEURISTIC_SCALING_FACTOR` constant
   - Duplicate method definitions with inconsistent signatures
   - Conflicting planning algorithm implementations

### Solution Implemented

#### 1. Created MockGraphManager (`tests/mock_graph_manager.py`)
- **Minimal mock implementation** that provides the essential GraphManager interface
- **Returns realistic test data** without requiring NetworkX or complex dependencies
- **Supports all GOAPPlanner method calls** that were previously failing with `None`

#### 2. Updated Test Files
Updated the following test files to use the mock graph manager:

- `tests/test_cost_integration.py` - 3 instances updated
- `tests/test_basic_integration.py` - 1 instance updated  
- `tests/test_planner_integration.py` - 3 instances updated
- `tests/test_action_definitions_resolution.py` - 1 instance updated
- `tests/test_unified_goap.py` - 2 instances updated
- `tests/test_goap_implementations.py` - 1 instance updated

#### 3. Fixed Codebase Issues
- **Added missing constant**: `HEURISTIC_SCALING_FACTOR = 0.1`
- **Fixed method signature mismatches** in `plan_actions` method
- **Resolved duplicate method definitions** causing type errors
- **Fixed planning algorithm conflicts** between heapq and simple list implementations

### Benefits Achieved

#### 1. Enhanced Test Coverage
- Tests now exercise **full code paths** including graph manager interactions
- Better **error detection** when graph manager methods are called
- More **realistic test scenarios** with actual state and action data

#### 2. Improved Code Quality  
- Fixed several **critical bugs** in the GOAP system
- **Consistent method signatures** across the codebase
- **Proper error handling** and fallback mechanisms

#### 3. Better Test Reliability
- Tests now **fail appropriately** when functionality is broken
- **Reduced false positives** from incomplete test coverage
- **Maintainable test infrastructure** for future development

### Verification Results
All updated test files now pass successfully:
- ✅ `tests/test_cost_integration.py` - 7 tests passing
- ✅ `tests/test_basic_integration.py` - All integration tests passing
- ✅ `tests/test_planner_integration.py` - 4 tests passing  
- ✅ `tests/test_action_definitions_resolution.py` - All resolution tests passing
- ✅ `tests/test_unified_goap.py` - All unified tests passing
- ✅ `tests/test_goap_implementations.py` - All implementation tests passing

### Key Files Modified

#### New Files:
- `tests/mock_graph_manager.py` - Minimal mock GraphManager implementation

#### Modified Files:
- `tiny_goap_system.py` - Fixed method signatures and added missing constants
- All test files listed above - Updated to use mock graph manager

### Usage Guidelines

For future tests involving GOAPPlanner:

```python
# Instead of this:
planner = GOAPPlanner(None)

# Use this:
from tests.mock_graph_manager import create_mock_graph_manager
mock_graph_manager = create_mock_graph_manager(character)
planner = GOAPPlanner(mock_graph_manager)
```

This ensures tests exercise the full functionality of the GOAPPlanner with realistic dependency interactions while maintaining simplicity and avoiding complex setup requirements.