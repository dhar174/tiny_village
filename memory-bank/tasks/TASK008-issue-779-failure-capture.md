# [TASK008] - Plan durable failure capture for #779

**Status:** Pending plan approval

**Added:** 2026-10-16

**Updated:** 2026-10-16

## Original Request

"Yes - please begin on #779," following the proposed failure-capture plan
and subsequent implementation after review.

## Approach

Inspect the current issue, completed investigation, existing subprocess helpers
and original verification supervisor. Propose a minimal shared capture helper
that preserves emitted output through abrupt exits and retains current deadlines,
assertions, fixture behavior and unittest failure/error distinctions.

## Implementation Plan

[Proposed plan](../../docs/superpowers/plans/2026-10-16-issue-779-failure-capture.md).
Approval and all human decisions remain in
[docs/decisions.md](../../docs/decisions.md).

1. Implement a standard-library capture helper and focused real-process checks.
2. Integrate only graph and model-cache integration launch helpers.
3. Run one bounded 81-method matrix plus three original probes; report all
   outcomes and stop on a startup timeout, incomplete batch or unsafe cleanup.

## Progress Tracking

| ID | Description | Status | Updated | Notes |
| --- | --- | --- | --- | --- |
| 8.1 | Read issue, existing helper code and investigation evidence | Complete | 2026-10-16 | No new observation runs |
| 8.2 | Draft and self-review scoped implementation plan | Complete | 2026-10-16 | Durable output, process ownership, exact deadlines and bounded validation |
| 8.3 | Human plan review | Pending | 2026-10-16 | Canonical approval status in docs/decisions.md |
| 8.4 | Implementation and bounded verification | Not started | 2026-10-16 | No test, runtime, dependency or cloud configuration edits |

## Progress Log

### 2026-10-16

- Confirmed #779 is open; #780 is open, ready for review and unmerged at
  29063c3. Created a separate local planning branch to preserve that PR.
- Inspected both existing run_python helpers and their caller/fixture contracts.
  Ownership inherits the graph helper; cache fixture TMPDIR is ephemeral.
- Planned direct file output, unique artifact directories, phase markers,
  explicit outcomes and one bounded resource observation. Preserve the outer
  supervisor's process-group ownership; avoid a signal-handshake subsystem.
- Defined a single nine-batch/three-probe application observation window after
  approval. It does not reopen the old exhausted 16-case diagnostic budget.
- No tests or experiments executed during planning. #779 remains unresolved;
  #721 and #722 dispositions are unchanged. Plan review is the next step.
