# [TASK009] - Verify merged Phase 1 capture work

**Status:** Completed with supervisor gap — #779 accepted-risk closure; #721/#722 decisions pending

**Added:** 2026-10-16

**Updated:** 2026-10-17

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
| 9.4 | Handoff to human decisions | In Progress | 2026-10-17 | D010 accepts #779 closure; next #721, then #722 separately |

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
