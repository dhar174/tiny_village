# TASK012 — Lightweight runtime and basic memory (#654/#776)

**Status:** In progress
**Added/Updated:** 2026-10-20

## Request and approved plan

Implement the reviewed [plan](../../docs/superpowers/plans/2026-10-20-issue-654-lightweight-runtime.md).
AI remains a required final-product capability (D015). Basic memory and bounded
startup/event/memory checks are selected; gameplay proof stays #751/#658.

## Progress ledger

| Task | Status | Evidence |
| --- | --- | --- |
| 1 Baseline | In progress | Branch from merged main 8a89965; D015 edits preserved |
| 2 Lazy services | Pending | |
| 3 Basic memory / real Characters | Pending | |
| 4 Headless controller | Pending | |
| 5 Minimal runner | Pending | |
| Verification / docs / PR | Pending | |

## Pre-flight

- Memory mode produced by Task 3 is consumed by Tasks 4/5; default full, minimal
  explicitly basic. Semantic initialization must not occur during construction.
- Task 4 preserves controller constructor shape; Task 5 supplies configuration
  and uses real registration/memory APIs rather than a parallel simulation.
- Evidence directory: /workspace/scratch/issue-654/. Preserve all failures.
- Ruling: use the existing clone with a feature branch, without a worktree —
  managed-cloud onboarding constraint; no main-branch implementation.
- Ruling: keep the bounded, approved verification matrix rather than bare pytest
  over unrelated live-model and known-broken legacy suites — the approved scope
  controls verification; no whole-repository-green claim will be made.

## Log

### 2026-10-20

- User approved the complete plan. Created codex/issue-654-lightweight-runtime
  from origin/main 8a899650; all four D015 edits carried without loss.
- Implement inline using executing-plans; a fresh whole-branch review is required
  at the end. No per-task delegation or independent implementation agents.
