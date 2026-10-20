# TASK012 — Lightweight runtime and basic memory (#654/#776)

**Status:** Implemented — PR #788 ready for human review; issues open pending merge
**Added/Updated:** 2026-10-20

## Request and approved plan

Implement the reviewed [plan](../../docs/superpowers/plans/2026-10-20-issue-654-lightweight-runtime.md).
AI remains a required final-product capability (D015). Basic memory and bounded
startup/event/memory checks are selected; gameplay proof stays #751/#658.

## Progress ledger

| Task | Status | Evidence |
| --- | --- | --- |
| 1 Baseline | Complete | Branch from merged main 8a89965; D015 edits preserved |
| 2 Lazy services | Complete | Cold imports RED→GREEN; owned-cache boundary test passes |
| 3 Basic memory / real Characters | Complete | 8 fresh-process checks pass, including complete Characters and basic restore |
| 4 Headless controller | Complete | Four core-only checks plus real Pygame owned/borrowed cleanup |
| 5 Minimal runner | Complete | Real seven-Character graph/event/memory probe and CLI validation |
| Verification / docs / PR | Complete | 111 distinct cases / 112 profile executions; all five review findings fixed; PR #788 open |

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

- Task 1 complete: baseline.json retains both original core-only failures.
- Task 2: removed eager imports/model/corpus initialization. Cold imports and
  non-LLM strategy tests pass 2/2 after retained RED and intermediate failure
  exposed a further eager NMF import. Cache-boundary check recorded separately.

- Tasks 2/3 complete: 8/8 fresh-process checks pass in basic-green3.log. Real
  Characters finish construction with no AI imports; memory fields, 50-record
  eviction, equal importance, independent stores and explicit errors verified.
- Retained intermediate decorator-order failure and restore-data loss failure;
  corrected helper placement and preserved recent records during restoration.
  New feature tests were observed RED before implementation.

- Tasks 4/5: 15 core-only tests and one real visual lifecycle test pass.
  Source/config copies prevent generated visual preferences mutating caller data.
- Ruling: minimal CLI requires at least one Character, while controller component
  tests can still request zero — memory acceptance cannot pass vacuously; cost
  is rejecting a previously meaningless zero-Character minimal invocation.
- Ruling: replace unsupported sample jobs Blacksmith/Guard/Healer with catalog
  roles Craftsman/Security Guard/Doctor — real construction exposed process exit
  12; cost is fixture wording only, no job-system feature removal. Invalid jobs
  now raise ValueError instead of terminating the process; retained RED probe.
- Ruling: the startup Event uses the supported neutral numeric impact 0 — dict
  impacts cannot currently be hashed as graph nodes. Keep this distinct legacy
  event defect as a follow-up rather than changing event equality/identity in
  dependency work; cost is leaving dict-impact graph registration unsupported.
- Retained visual fixture CWD/import failure, config-mutation failure, and the
  legacy smoke test's obsolete turn/banner assertions. Updated the latter to
  actual startup/event/memory claims, retaining its 120-second parent limit.

- Verification matrix completed: three fresh verified installs/pip checks; 106
  distinct cases / 107 executions across core, dev, visual, NLP and CPU-LLM.
  Initial cache import-order subcases expected eager mapper loading; retained
  failures and updated only that test to explicitly load the real mapper after
  proving lazy import. Corrected cache selection passes 27/27; runtime unchanged.
- Drafted validation report and user-facing guidance; all dependency locks remain
  unchanged. No live model, gameplay-turn or whole-suite readiness claim.

- Independent whole-branch review found five Important issues, no Critical or
  actionable Minor findings. Accepted all five: lazy helper enablement, cached
  controller provider use/status, restored-store ownership, theme analysis guard,
  and independently owned Pygame display/font cleanup. One RED→GREEN fix pass
  in ccc2469; no findings deferred and no second review dispatch.
- Final complete scoped matrix on ccc2469 passes 111 distinct cases / 112 profile
  executions, no skips/timeouts, plus expected standalone CLI exit codes 0/0/2.
  Earlier failures/matrices preserved separately; final sources stayed unchanged.
  Published baseline and machine-readable final receipt in docs/testing/evidence/.
- Separate observed legacy defects now have issues: #786 for dictionary Event
  impacts/hashing and #787 for job-catalog count/missing-role diagnostics. These
  are scope limits, not deferred review findings. Next ordered task remains #726.

- Published ready PR #788 targeting main, with closing references for #654/#776
  and explicit human review requests for boundaries, retained AI paths, failures
  and guidance. Both issue bodies/checklists and evidence comments synchronized;
  issues remain open until merge. No human phase decision was recorded.
- PR: https://github.com/dhar174/tiny_village/pull/788
