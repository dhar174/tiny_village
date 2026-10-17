# [TASK005] - Verify merged startup repairs and guide human review (#721)

**Status:** Completed — human Approved under D011; Project synchronization outstanding

**Added:** 2026-10-15

**Updated:** 2026-10-16 (America/New_York)

## Original Request

"Let's do it, together. After you verify things, you can walk me through the decisions step by step."

## Approach

Verify merged main rather than reuse pre-merge results. Retain failed and
incomplete runs, identify the limits of each passing check, and prepare a
human-readable review on #721. A human must explicitly record the disposition;
#722 remains a separate phase decision. No application changes are authorized
by this verification and walkthrough alone.

## Implementation Plan

1. Confirm merge/prerequisites and freeze the reviewed revision.
2. Execute existing focused regression suites and startup probes.
3. Retain failures and run targeted diagnostics where useful.
4. Publish evidence and guide the human through scope, limits and disposition.
5. Record the explicit human decision before any issue closure or phase advance.

## Progress Tracking

**Overall Status:** Completed — D011 approval recorded and #721 closed; #722 remains a separate pending decision; Project synchronization outstanding.

| ID | Description | Status | Updated | Notes |
| --- | --- | --- | --- | --- |
| 5.1 | Confirm merge and prerequisites | Complete | 2026-10-15 | PR #778 merged at 93079fd; #718/#719/#720 closed; source matches f796b33 |
| 5.2 | Focused merged verification | Complete | 2026-10-15 | 30 tests in complete suites pass; cache 26/27; import timing failures and incomplete lifecycle/ownership retained |
| 5.3 | Review evidence and diagnostics | Complete | 2026-10-15 | Both affected import orders pass instrumented diagnostics; cause unproven; investigation #779 |
| 5.4 | Human decision walkthrough | Complete | 2026-10-16 | D011 explicitly approves Linux/offline startup at 77e4b94; prior Deferred retained |
| 5.5 | Record decision and phase ordering | Complete with external sync limit | 2026-10-16 | #721 closed; #722/#717 open; Project Decision update remains unconfirmed after 403 |

## Progress Log

### 2026-10-15

- Confirmed PR #778 merge 93079fd6ef7ef7607153b95e2b66a48a957a9ee6,
  exact tree match with reviewed f796b33, and all three closed prerequisites.
- Both merged-main CodeQL language checks succeed; these are static analysis,
  not startup verification.
- Reused prepared Python 3.12 CPU environment and previous bounded commands,
  unchanged assertions/deadlines; no dependency/model installation.
- Eight import-order subcases: six pass, graph-manager-first reaches the fatal
  timer in pandas imports, Character-first reaches the parent deadline.
- Lifecycle/ownership batches reach aggregate deadlines after partial passes
  and timing failures. Do not count incomplete batches as passing suites.
- No cause established for timing variability; no graph-lock regression claim.
- #773–#776 remain open; earlier outcomes are historical and not rerun here.
- Verification completes with failures: 30/30 in five complete suites, cache
  26/27, two import-order timeouts and incomplete lifecycle/ownership batches.
  Goal fallback, cold getter and minimal probes pass; minimal executes no actions.
- Instrumented graph-first/Character-first diagnostics pass unchanged limits in
  10.19s/5.11s. Import timing varies; no root cause established and no failed run
  erased. Recommendation is Deferred, pending explicit human disposition.
- Created #779 to retain the scoped timing investigation; no Project dependency
  change, runtime implementation or human decision supplied by that issue.
- Review packet: docs/testing/issue-721-startup-review.md. Raw evidence and
  corrected interpretation of incomplete-suite counts retained in scratch.
- Published the review evidence on #721:
  https://github.com/dhar174/tiny_village/issues/721#issuecomment-6194577005.
  Human disposition, gate/dependency changes and source repairs remain pending.
- dhar174 answered “Yes” to recording Deferred pending #779. Recorded the
  explicit decision in the issue body and comment, including reviewer, date,
  revision, rationale and evidence:
  https://github.com/dhar174/tiny_village/issues/721#issuecomment-6194738536.
- Added and read back native dependency #779 → #721; preserved existing #720
  dependency. Both #721 and #722 remain open. No runtime implementation or
  downstream phase advancement authorized by this disposition.
- Project 10 read returned HTTP 403 “Resource not accessible by integration”
  through the configured GH_PROJECT_TOKEN binding. Decision field remains
  unsynchronized; issue decision is recorded and verified. No credential requested.
- User confirmed the proposed scope and requested a single Markdown decision
  record. Consolidated the decisions and outstanding actions in
  [docs/decisions.md](../../docs/decisions.md); future decisions belong there.
  Evidence remains in the validation report; no runtime changes or new checks.

### 2026-10-17 UTC

- Latest evidence is the 77e4b94 post-merge report in PR #782: 94 successful
  method summaries, three zero-exit probes, and one lifecycle supervisor gap.
- See [D010](../../docs/decisions.md#d010--accept-residual-uncertainty-and-close-investigation-779)
  for the maintainer's accepted-risk disposition of #779. Its closed state
  preserves the native historical dependency; it does not approve this review.
- Next: obtain #721's explicit disposition using the updated evidence, then
  consider #722 separately. Existing later-phase scope decisions still apply.

### 2026-10-16 America/New_York — subsequent approval

- The maintainer explicitly approved #721's engineering review. See the
  canonical [D011](../../docs/decisions.md#d011--approve-startup-engineering-review-721)
  for the reviewed SHA, rationale, accepted limits and scope.
- Recorded approval and closed #721 as completed. Refreshed linked issue
  status without granting #722 or #717 approval. Native dependencies remain.
- Updated PR #782 with the decision and evidence pointers. No new tests or
  application runs; this records the human's disposition of retained evidence.
- Project synchronization remains outstanding from the earlier HTTP 403;
  no credential change or successful Project field update is claimed.
