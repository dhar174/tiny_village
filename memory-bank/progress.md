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

- As of 2026-10-14, #718 is complete: #706 merged and the issue closed.
- TASK003 / #719 is complete: #777 merged and the issue closed. Post-merge
  verification passes 54/54 tests, no skips/timeouts, plus cold-getter, actual
  Goal fallback and minimal probes.
- TASK004 / #720 repairs shared and machine-specific cache configuration;
  73/73 tests pass in bounded batches (19 cache); two review findings are fixed.
  Single-module initialization exceeded its aggregate deadline; details retained
  in docs/testing/issue-720-owned-cache-validation.md. Draft PR #778 awaits human review/merge.
- Existing integration tests remain 9/11 and utility tests 16/26. #773/#774 retain
  those failures. #775/#776 retain missing Door effects and optional spaCy
  initialization limits; full Character/simulation readiness is not established.
- Evidence: `docs/testing/issue-719-graph-initialization-validation.md`.

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
