## What Works

- The repository has established architecture and documentation for the core
  simulation loop: events, strategy updates, GOAP planning, action execution,
  graph state updates, and memory recording.
- Current docs describe working integration across core systems: gameplay
  control, event handling, strategy coordination, graph management,
  storytelling, checkpoints, and analytics.
- The repo has current contributor guidance (`AGENTS.md`), organized docs under
  `docs/`, a `tests/` directory with organized test coverage, and testing
  guidance that emphasizes realistic test objects over permissive mocking.
- The `tests/` directory and root-level `test_*.py` files provide integration
  and regression test coverage across many subsystems.

## What Is Still Incomplete

- Several Memory Bank core files contained stale content from an unrelated
  project (`langgraph_system_generator`) and are being corrected as part of
  TASK001.
- Task-level Memory Bank tracking has only just begun with TASK001; the index
  and task file need to be committed.
- Some current status docs still describe demo/runtime gaps such as display
  initialization edge cases or action execution compatibility concerns.

## Current Status

- As of 2026-10-13, TASK002 / #718 has a reviewed, rebased fix in draft PR #706.
  Cold utility/GOAP imports, production fallback construction, and 5 GOAP tests
  pass. Maintainer approval and merge remain pending.
- Bounded validation executed 42 tests: 30 passed, 2 failures, 10 errors, and
  0 skipped. The minimal demo exits 0 but executes zero actions. See
  `docs/testing/issue-718-goal-fallback-validation.md` for exact commands and scope.
- Graph lock re-entry still hangs (#719). Separate issues #773 and #774 retain
  integration fixture and utility Goal/API failures; they are blocked by #718.

- TinyVillage is documented as a modular Python simulation project with
  root-level runtime modules and optional advanced AI/LLM features.
- The Memory Bank is in the process of being corrected from stale template
  content into repo-specific, TinyVillage-accurate documentation.
- The current canonical contributor instructions emphasize minimal, reality-
  based changes and conservative test design.

## Known Issues and Limitations

- Documentation may contain inconsistencies across time; newer docs such as the
  root `README.md` and root `AGENTS.md` should generally be prioritized over
  older or archived status notes when they conflict.
- Optional LLM/NLP features require additional dependencies and should degrade
  gracefully when those dependencies are absent.
- Some testing history in the repo shows over-mocking pitfalls, especially
  around memory-related tests; Memory Bank guidance should preserve the
  repository preference for tests that fail on real behavior mismatches.
