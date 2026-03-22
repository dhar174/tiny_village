# Incongruities Report

This document outlines the incongruities found between the markdown documentation and the actual codebase implementation in the Tiny Village repository.

## 1. Outdated or Misaligned Testing Documentation

### Issue: Overlapping/Duplicate Testing Files for Custom Buildings
- **Documentation**: `CUSTOM_BUILDINGS_IMPLEMENTATION.md` and `CUSTOM_BUILDINGS_GUIDE.md` extensively mention running `python -m unittest tests.test_building_loading_unit`.
- **Codebase**: Both `tests/test_building_loading_unit.py` and `tests/test_custom_building_loading.py` exist in the repository, containing redundant or conflicting test coverage for building loading.
- **Impact**: Minor. Having multiple test files for the same feature can cause confusion.

### Issue: Testing Best Practices Tools
- **Documentation**: `docs/testing/MEMORY_TESTING_BEST_PRACTICES.md` implies certain testing strategies and tools (such as avoiding over-mocking).
- **Codebase**: Some newer test files use aggressive mocking patterns that contradict the strict guidelines described in the best practices document.

## 2. Incomplete or Outdated Project Status Claims

### Issue: Minimum Demo Status Outdated
- **Documentation**: `docs/reference/MINIMUM_DEMO_STATUS.md` claims that `main.py` is missing and needs to be created, and that the `MapController` initialization is broken.
- **Codebase**: `main.py` clearly exists at the root of the repository and the MapController is fully integrated, making the "Minimum Demo Status" documentation significantly outdated.
- **Impact**: Major. New developers reading the status document will believe the project is in a more broken state than it actually is.

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
- Validated that `CheckpointManager`, `BuildingManager`, and `StrategyManager` methods exactly match their markdown references. No fixes were necessary in the code for these, as they align well.
