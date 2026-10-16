# [TASK008] - Plan durable failure capture for #779

**Status:** Completed — implementation/verification published; #779 unresolved

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
| 8.3 | Human plan review | Complete | 2026-10-16 | Canonical approval status in docs/decisions.md |
| 8.4 | Implementation and bounded verification | Complete | 2026-10-16 | 11 new checks + 81 existing methods + three probes pass; draft PR #781 |

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

- Approval recorded as D008. Begin sequential implementation and the approved
  nine-batch/three-probe verification window. No extra diagnostic observations.
- Task 1: new helper passes 11/11 focused checks (2.658s, no skips/errors),
  including abrupt-parent evidence and timeout reaping. Integration not yet run.
- Task 2: two launch helpers integrated; all existing test bodies/fixtures
  compare unchanged. Code committed as 31e8cdb, equivalent to tested sources.
- Task 3: one approved nine-batch/three-probe window passes 81/81 existing
  methods and all probes, no failures/errors/skips/timeouts. All 42 child
  artifact sets verified; 390 source hashes stable; all owned groups reaped.
- Minimal mode still executes zero actions and logs existing map/sentiment/
  missing-node limitations. No full-gameplay or live-model claim.
- Separate author review found no blocking defect; independent agent review was
  excluded by the approved execution method. Publish a stacked PR above #780.
- Published/attached draft PR #781 above codex/issue-721-review; #780 remains
  ready, open and unchanged at 29063c3. No automatic closing references.
- Posted and read back findings on #779 (comment6205767601) and #721
  (comment6205768056). Updated #779's dedicated status block while preserving
  all original acceptance text and its open state. #721/#722 remain open.
- The approved observation budget is consumed. Capture tooling is delivered;
  historical cause remains unknown. Report: docs/testing/issue-779-failure-capture.md.
- Review follow-up: reproduced P2 deadline regression; start the parent wait
  budget after process creation/STARTED metadata and keep launch time separate.
  Two regression cases fail before the change; all 13 focused checks pass
  afterward (3.352s). Original real-process delayed-launch reproduction now
  exits 0 for both old and corrected launchers with unchanged timeouts.
- Prior 81-method/probe results remain historical; no new application window.
  #780 has merged; #781 now targets main. Push correction and resolve the
  addressed review finding; keep #779/#721/#722 dispositions unchanged.
