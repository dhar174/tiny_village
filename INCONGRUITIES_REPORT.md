# Incongruities Report

This document outlines the incongruities found between the markdown documentation and the actual codebase implementation in the Tiny Village repository.

## 1. Outdated or Misaligned Testing Documentation

### Issue: Overlapping/Duplicate Testing Files for Custom Buildings
- **Documentation**: `CUSTOM_BUILDINGS_IMPLEMENTATION.md` and `CUSTOM_BUILDINGS_GUIDE.md` extensively mention running `python -m unittest tests.test_building_loading_unit`.
- **Codebase**: Both `tests/test_building_loading_unit.py` and `tests/test_custom_building_loading.py` exist in the repository, containing redundant or conflicting test coverage for building loading.
- **Impact**: Minor. Having multiple test files for the same feature can cause confusion.

### Issue: Memory-Object Mocking in PromptBuilder Tests Conflicts with Memory Testing Best Practices
- **Documentation**: `docs/testing/MEMORY_TESTING_BEST_PRACTICES.md` explicitly warns against `MagicMock`-style stand-ins for memory objects because they can hide attribute and integration bugs.
- **Codebase**: `tests/test_enhanced_prompt_builder.py` uses `MagicMock` objects as memory-like inputs (for example `MagicMock(description="Test memory")` and `MagicMock(description="Legacy memory")`) instead of lightweight classes or realistic memory objects.
- **Impact**: Moderate. This directly contradicts the memory-specific guidance and can let PromptBuilder memory-formatting tests pass without validating real memory-object behavior.

### Issue: General Over-Mocking Guidance Is Inconsistent Across Building-Loader Tests
- **Documentation**: `MOCK_USAGE_BEST_PRACTICES.md` says to use real objects for the main component under test where possible and reserve `Mock()`/`MagicMock()` for boundaries or external dependencies.
- **Codebase**: `tests/test_building_loading_unit.py` replaces `pygame` with `MagicMock` and binds `GameplayController._load_buildings_from_file` onto a `Mock(spec=GameplayController)` instead of exercising a real controller instance, while `tests/test_custom_building_loading.py` covers the same loader through a real `GameplayController` with only display-boundary patching.
- **Impact**: Moderate. The repository currently offers two conflicting examples for the same feature area, which makes it harder for contributors to tell whether the preferred pattern is a real controller with patched boundaries or a partially mocked controller shell.

## 2. Incomplete or Outdated Project Status Claims

### Issue: Minimum Demo Status Was Historically Outdated
- **Documentation**: Previously, `docs/reference/MINIMUM_DEMO_STATUS.md` still described `main.py` as missing, `MapController` display initialization as a blocker, and `Action.execute()` compatibility as unfinished work.
- **Codebase**: `main.py` exists at the repository root and the related status items are already implemented, so the doc had drifted behind the current code.
- **Resolution**: The status document has been updated to mark those items as resolved and to remove later sections that still treated them as active blockers.
- **Impact**: Major. While this drift remained in place, new contributors could conclude the runtime was in a more broken state than it actually is.

### Issue: Archived Docs Referencing Root
- **Documentation**: Files in `docs/archived/` (e.g., `missing_demo_elements.md`) list `main.py` as missing.
- **Codebase**: As noted above, `main.py` exists. The archived README (`docs/archived/README.md`) notes this, but the individual files retain the outdated claims, which is confusing if searched globally.
- **Impact**: Minor.

## 3. Discrepancies in Architecture and Integration Docs

### Issue: BuildingManager Initialization
- **Documentation**: `BUILDING_SYSTEM_DOCUMENTATION.md` mentions `self.building_manager = BuildingManager()` inside `GameplayController.initialize_game_systems()`.
- **Codebase**: `tiny_gameplay_controller.py` conditionally creates `self.building_manager` but the documentation does not accurately reflect the fallback/error handling structure seen in the code.
- **Impact**: Minor. The documentation simplifies the implementation details.

### Issue: Undocumented Methods in CheckpointManager
- **Documentation**: `CHECKPOINT_SYSTEM_DOCUMENTATION.md` outlines methods like `create_checkpoint`, `restore_checkpoint`, `should_checkpoint`, etc.
- **Codebase**: `CheckpointManager` in `tiny_gameplay_controller.py` contains all these methods, but also has internal helper methods and error recovery parameters that are not documented, meaning the documentation doesn't fully capture the robustness of the system.
- **Impact**: Minor.

## 4. Minor Fixes Applied
- Verified that `main.py` exists and is the correct entry point.
- Confirmed that the public method names for `CheckpointManager`, `BuildingManager`, and `StrategyManager` remain in sync with their markdown references. The remaining discrepancies noted above are about undocumented behaviors and simplified implementation descriptions rather than missing or renamed methods, so no code changes were necessary for that alignment.
