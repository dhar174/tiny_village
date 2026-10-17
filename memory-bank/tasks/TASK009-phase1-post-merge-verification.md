# [TASK009] - Verify merged Phase 1 capture work

**Status:** Completed with supervisor gap — human dispositions D010–D012 recorded; Phase 1 complete

**Added:** 2026-10-16

**Updated:** 2026-10-17 (America/New_York)

## Original Request

"Proceed," following the proposed single post-merge verification and subsequent
human closure/review walkthrough.

## Approach

Freeze main at `77e4b9437cc00970def933d992140e6da37eae8e`, including merged
#780 and #781. Reuse the prepared offline Linux environment and existing
supervisor, adding the 13 capture checks before the nine existing test batches
and three probes. Stop on a failing/incomplete batch; do not retry or expand.

## Implementation Plan

1. Record the exact SHA, commands, dependencies and source hashes.
2. Run 13 capture checks, 81 existing methods and three probes once, retaining
   20/25-second graph, 25/30-second cache and 110/120-second batch limits.
3. Retain complete evidence and publish a concise report on #779/#721.
4. Present the #779 closure/risk decision, then #721 and #722 separately.

All human decisions belong in [docs/decisions.md](../../docs/decisions.md),
including verification authorization D009. No closure or gate approval is inferred.

## Progress Tracking

| ID | Description | Status | Updated | Notes |
| --- | --- | --- | --- | --- |
| 9.1 | Freeze merged-main inputs | Complete | 2026-10-16 | main 77e4b94; no runtime/test edits |
| 9.2 | Run one authorized window | Complete with gap | 2026-10-16 | 94 successful method summaries; three probes; lifecycle supervisor result unknown |
| 9.3 | Verify and publish evidence | Complete | 2026-10-16 | Draft PR #782; #779/#721 comments and status blocks verified |
| 9.4 | Handoff to human decisions | Complete | 2026-10-17 | D010 accepts #779 closure; D011 approves #721; D012 separately approves #722 and completes #717 |

## Progress Log

### 2026-10-16

- Created a documentation branch from current merged main; application source
  remains identical to 77e4b94 during the window.
- Recorded authorization D009; it does not accept risk or close an issue.
- Evidence root: /workspace/scratch/phase1-post-merge/2026-10-16-main-77e4b94/.
  Exact commands, source hashes and all attempt outcomes are retained there.
- Execution-server transport disconnected after the lifecycle suite. Preserved
  its 12-test OK summary and 15 child zero-exits; final supervisor exit/cleanup
  remain unknown. The original results snapshot remains untouched.
- Recovery ruling: the tests had completed, so continue only NOT_RUN targets
  once, in a detached supervisor. No completed/interrupted target was repeated,
  no deadline changed and no application diagnosis was added.
- Completed all remaining targets. All 94 methods report success and all three
  probes exit 0; lifecycle keeps SUPERVISOR_INTERRUPTED, not PASS. 82 methods
  have confirmed supervisor completion; 12 have the recorded supervisor gap.
- Verified all 42 child artifact sets and unchanged 390 source/manifest hashes.
  Minimal mode still executes zero actions and retains map/sentiment/node limits.
- Report: docs/testing/phase-1-post-merge-verification.md. Prepare the #779
  risk/closure question; #721 Deferred and #722 open remain unchanged.
- Published/attached draft evidence PR #782. Posted and read back comments
  6227420596 (#779) and 6227421138 (#721). Updated dedicated status blocks
  while preserving original acceptance text and open states. No issue closed.
- Human handoff is ready: ask whether the maintainer accepts the unknown
  historical cause and supervisor limitation for the proposed #779 closure.

### 2026-10-17 UTC

- Continued the walkthrough after the maintainer's explicit approval. The
  decision is maintained only in [D010](../../docs/decisions.md#d010--accept-residual-uncertainty-and-close-investigation-779).
- Published the updated evidence on PR #782 and synchronized #779/#721 status
  blocks without changing their original acceptance text. #779 is closed as
  non-reproducible with diagnostics added; its cause remains unknown.
- No application changes, new tests, repeated observations or new phase work.
  #721 remains Deferred and #722 remains open for separate decisions.
- Read-back receipts: [#779 closure comment](https://github.com/dhar174/tiny_village/issues/779#issuecomment-6227664671)
  and [#721 handoff](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6227665778).
  #779 state is closed / not_planned; #717/#721/#722 are open. Original issue
  acceptance tails match their pre-update snapshots byte for byte.

### 2026-10-16 America/New_York — subsequent review disposition

- [D011](../../docs/decisions.md#d011--approve-startup-engineering-review-721)
  records the maintainer's explicit #721 approval. #721 is closed as completed;
  #722 and parent #717 remain open. #779's accepted-risk closure is unchanged.
- No new observations, runtime changes or broader readiness claims. The
  next walkthrough item is #722's separate Phase 1 decision.

### 2026-10-17 America/New_York — Phase 1 gate disposition

- The maintainer explicitly approved the Phase 1 gate. The canonical
  [D012](../../docs/decisions.md#d012--approve-phase-1-gate-722-and-complete-phase-717)
  records scope, rationale, evidence and the exact confirmation.
- Published the decision in PR #782, recorded approval on #722 and closed
  #722/#717 as completed. Retained the native hierarchy/dependencies and all
  earlier failures and accepted limitations. Refreshed linked issue status.
- Phase 2 is unblocked, starting with #724's MVP scenario/support matrix/budgets.
  #723/#724 remain open; subsequent review and phase-gate decisions are required.
- No runtime changes, installs or new test/application observations. Project
  synchronization remains outstanding from the earlier HTTP 403; no repeat
  request through the unchanged credential or successful update is claimed.
- Publication receipts: [#722 approval](https://github.com/dhar174/tiny_village/issues/722#issuecomment-6229527606)
  and [#717 completion](https://github.com/dhar174/tiny_village/issues/717#issuecomment-6229528445).
  Read-back confirms both closed/completed, all #718–#722 sub-issues closed,
  #724's native #722 dependency closed, and #723/#724 still open. Original
  issue text is retained; completed gate/phase checklist items are checked,
  with the unavailable Project Decision synchronization left unchecked.
