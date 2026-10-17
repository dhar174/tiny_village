# Project decisions

This is the single maintained record of human decisions from the project review
walkthrough. Add future decisions here. Evidence reports and task summaries
should link to this file rather than maintain separate decision records.

## Recorded decisions

### D001 — Defer startup engineering review #721

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Disposition:** Deferred
- **Reviewed revision:** `93079fd6ef7ef7607153b95e2b66a48a957a9ee6`
  (merged PR #778).
- **Decision:** Keep #721 open until #779 explains the startup timing failures
  and required verification completes with retained evidence. Keep phase gate
  #722 open and downstream phase advancement blocked.
- **Reason:** Merged verification has import timeouts, incomplete lifecycle and
  ownership batches, and one cache-loader integration timeout. Passing targeted
  diagnostics show variability without establishing a cause or replacing the
  failed and incomplete runs.
- **Confirmation:** The user answered “Yes” to recording Deferred pending #779.
- **Evidence:** [Merged verification](testing/issue-721-startup-review.md) and
  [recorded GitHub disposition](https://github.com/dhar174/tiny_village/issues/721#issuecomment-6194738536).
- **Follow-up:** [#779](https://github.com/dhar174/tiny_village/issues/779) is a
  verified native blocker of [#721](https://github.com/dhar174/tiny_village/issues/721).

### D002 — Keep wider repairs in their existing phases

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep the following issues in their existing phases and preserve
  their existing dependencies and review requirements:

  | Issue | Scope retained |
  | --- | --- |
  | [#773](https://github.com/dhar174/tiny_village/issues/773) | Integration fixtures and fallback action behavior |
  | [#774](https://github.com/dhar174/tiny_village/issues/774) | Goal contracts and explicit falsy action IDs |
  | [#775](https://github.com/dhar174/tiny_village/issues/775) | Missing Open Door effects and production action behavior |
  | [#776](https://github.com/dhar174/tiny_village/issues/776) | Character memory behavior when spaCy is unavailable |

- **Boundary:** These issues remain open follow-ups; this scope decision does
  not claim their behavior works or their acceptance criteria are complete.
  Require #779 before approving #721. Any newly diagnosed prerequisite must be
  made explicit rather than silently expanding or relaxing review scope.
- **Reason:** Separate the focused startup investigation from the broader
  integration, contract and feature repairs already tracked in the roadmap.
- **Confirmation:** The user replied “I agree” to keeping #773–#776 in their
  existing phases while requiring #779 before approving #721.

### D003 — Maintain decisions in one Markdown file

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Maintain review decisions in this file, including their scope,
  rationale, confirmation, evidence and outstanding actions.
- **Confirmation:** “And, just put all decisions in a single MD file.”
- **Existing records:** Preserve historical GitHub comments and validation
  evidence as provenance. Use this file for subsequent decision updates.

### D004 — Defer native Windows validation to platform work

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep native Windows execution and cache-permission validation
  in the later platform work, rather than add it as a prerequisite for #721.
  The current review remains focused on the Linux startup evidence and #779.
- **Boundary:** Windows support and Windows cache isolation remain unverified.
  Require native Windows validation before claiming Windows support; Linux
  tests and simulated Windows metadata do not establish it.
- **Reason:** Resolve the current startup verification failures within their
  existing environment and address platform-specific guarantees separately.
- **Confirmation:** “Yes, save it for later,” in response to deferring Windows
  validation to the later platform work.

### D005 — Defer live model validation to ML/LLM work

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep real model downloads and inference validation in the
  later ML/LLM work. Use offline startup checks for #721, including the
  existing cache integration checks with controlled model-loading boundaries.
- **Boundary:** Offline checks do not establish live download, transport,
  artifact verification or inference behavior. Preserve existing verification
  settings; this decision does not permit disabling TLS or checksum checks.
  The cache-loader timeout remains part of required investigation #779.
- **Reason:** Validate startup and cache configuration within the prepared
  environment without expanding this review into live model integration.
- **Confirmation:** The user answered “Yes” to keeping real downloads and
  inference in later ML/LLM work while #721 uses offline startup checks.

### D006 — Defer real gameplay-turn validation to integration work

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Keep completion of a real gameplay turn in the later demo and
  integration work, rather than make it a prerequisite for Phase 1 approval.
  Phase 1 remains focused on reliable startup within the agreed environment.
- **Boundary:** The minimal-mode probe exits successfully but executes zero
  actions. Its result establishes no completed gameplay turn or full simulation
  readiness. Later integration validation must demonstrate actual behavior.
- **Reason:** Separate the startup repairs from end-to-end simulation behavior,
  retaining the latter as explicit follow-up work.
- **Confirmation:** “Later, yes,” in response to scheduling gameplay-turn
  validation in later demo/integration work.

### D007 — Approve the bounded #779 investigation

- **Date:** 2026-10-15
- **Decision-maker:** dhar174
- **Decision:** Execute the [investigation plan](superpowers/plans/2026-10-15-issue-779-startup-investigation.md)
  sequentially within its diagnostic budget and unchanged verification limits.
- **Boundary:** Investigation only. Bring findings back before applying a
  runtime, dependency, persistent configuration or deadline-contract remedy.
  #721 remains Deferred; #722 remains open.
- **Confirmation:** “Approved, proceed,” in response to the investigation-only
  plan approval question.

### D008 — Approve durable failure capture and bounded verification for #779

- **Date:** 2026-10-16
- **Decision-maker:** dhar174
- **Decision:** Implement the [failure-capture plan](superpowers/plans/2026-10-16-issue-779-failure-capture.md)
  sequentially, including its focused real-process checks and single nine-batch,
  three-probe application observation window.
- **Boundary:** Test-harness diagnostics only; preserve original assertions,
  deadlines, runtime, dependency and cache behavior. No automatic retries,
  issue closure, phase approval or delegation. Bring any unexplained timeout
  or proposed runtime/environment correction back with evidence.
- **Confirmation:** “Looks good. Approved.” in response to the saved plan.

### D009 — Approve one post-merge verification window

- **Date:** 2026-10-16
- **Decision-maker:** dhar174
- **Decision:** Verify merged `main` once with the 13 capture checks,
  81 existing startup-matrix methods and three original probes. Preserve all
  original assertions, child deadlines and 110/120-second batch limits.
- **Boundary:** One sequential window, no automatic retries or expanded
  diagnosis. Bring results back for the #779 closure/risk decision and the
  separate human reviews of #721 and #722. This execution approval does not
  accept the unexplained timeout risk, close issues or approve either gate.
- **Confirmation:** “Proceed,” following the ordered post-merge next steps.

### D010 — Accept residual uncertainty and close investigation #779

- **Date:** 2026-10-17 UTC (2026-10-16 EDT)
- **Decision-maker:** dhar174
- **Decision:** Accept the unknown historical timeout cause and the D009
  lifecycle supervisor completion gap; authorize closing #779 as **unable to
  reproduce; diagnostics added**. This is an accepted-risk disposition, not
  a demonstrated root-cause fix or fulfillment of every original criterion.
- **Evidence:** [Post-merge report](testing/phase-1-post-merge-verification.md)
  at main `77e4b9437cc00970def933d992140e6da37eae8e`: 94 successful method
  summaries (82 with confirmed supervisor completion, 12 with the documented
  supervisor gap), three zero-exit probes and 42 verified child artifact sets.
  Diagnostics merged in #781; evidence is published in #782.
- **Boundary:** Preserve the original failures and missing terminal record.
  Reopen #779 if the timeout recurs and retain the diagnostic artifacts before
  proposing further investigation. This decision supplies no approval of #721
  or #722 and authorizes no additional observation window or phase advancement.
- **Confirmation:** “yes” to: “Do you accept the unknown historical timeout
  cause and this reporting gap, and authorize closing #779 as ‘unable to
  reproduce; diagnostics added’?”

### D011 — Approve startup engineering review #721

- **Date:** 2026-10-16 (America/New_York)
- **Decision-maker:** dhar174
- **Disposition:** Approved
- **Reviewed revision:** `77e4b9437cc00970def933d992140e6da37eae8e`
  (merged startup repairs and capture diagnostics).
- **Decision:** Approve #721 for the agreed Linux/offline startup scope and
  close the engineering-review issue. This supersedes D001's Deferred
  disposition for the current review; the original failed runs remain evidence.
- **Rationale and evidence:** Startup fallback, graph lifecycle/ownership and
  cache repairs are merged. The [post-merge report](testing/phase-1-post-merge-verification.md)
  retains 94 successful method summaries and three zero-exit probes. 82 methods
  have confirmed supervisor completion; the other 12 retain successful summaries
  and child exits with the supervisor reporting gap accepted under D010.
  The historical timeout cause remains unknown; #779's accepted-risk closure
  does not establish a root-cause fix.
- **Boundary:** Native Windows, live models and real gameplay remain deferred
  under D004–D006; #773–#776 retain their existing scope under D002. This does
  not approve phase gate #722, close parent #717 or advance downstream work.
- **Confirmation:** “Yes, approved. **#721’s engineering review**,” explicitly
  for the agreed Linux/offline startup scope and documented limitations.

## Review status

The current scope questions have been answered (D002, D004–D006). Review #721
is **Approved** under D011 and closed for the agreed Linux/offline startup
scope. D010 accepts #779's residual uncertainty and closure. D001 remains the
historical disposition of the earlier failed verification. Phase gate #722
still needs its separate human decision; parent #717 remains open.

The [#779 investigation plan](superpowers/plans/2026-10-15-issue-779-startup-investigation.md)
was approved under D007, and its bounded cycle has concluded. Its original
16-case diagnostic budget remains exhausted; no production repair is approved.

The [failure-capture follow-up](superpowers/plans/2026-10-16-issue-779-failure-capture.md)
was approved under D008. Its implementation merged and its bounded verification
window concluded. This did not change #721's disposition or approve a runtime
correction.

The [D008 implementation and verification report](testing/issue-779-failure-capture.md)
records 11 new capture checks, 81 existing methods and three original probes
passing in the single approved window. Its nine-batch/three-probe budget is
consumed. The historical timeout did not recur, so its cause remains unknown;
#779 stayed open at that handoff; D010 subsequently accepts closure with the
cause still unknown. That handoff did not change #721's disposition; its later
approval is recorded separately under D011.
Merged [PR #781](https://github.com/dhar174/tiny_village/pull/781) contains this
follow-up; merging it is not a human approval or issue closure.

The [D009 post-merge report](testing/phase-1-post-merge-verification.md) on merged
main `77e4b94` contains 94 successful test-method summaries and three zero-exit
probes. One lifecycle supervisor completion record is missing after a transport
disconnect; its successful 12-test summary and all child exits are retained.
Only previously unstarted targets were continued, once each. No target was
rerun, and the missing supervisor result is not labeled PASS. The authorized
window is consumed. D010 subsequently accepts the residual risk and #779
closure. D011 subsequently approves #721; #722's decision remains pending.
Neither merging the PRs nor "Proceed" supplied those dispositions.
The evidence is published in [draft PR #782](https://github.com/dhar174/tiny_village/pull/782)
and linked on #779/#721; their original acceptance text is preserved.

The [approved observation cycle](testing/issue-779-startup-investigation.md)
has concluded: 51 affected regression methods pass and all 16 predeclared
diagnostic cases exit successfully. The historical timeout cause remains
unconfirmed. No remedy or new review disposition is approved by those results;
the diagnostic budget is exhausted. The later D010 closure accepts this
uncertainty without claiming a proven remedy.

## Outstanding actions and limits

- #779 is closed under D010 as unable to reproduce; diagnostics added. Reopen
  if the timeout recurs and retain failure artifacts. Further observations
  require a new scope; no root cause or production fix has been established.
- Include native Windows execution and cache-permission checks when planning
  the later platform work, before any Windows-support claim (D004).
- Include live model download and inference checks in the later ML/LLM work
  before making readiness claims for those features (D005).
- Demonstrate a real gameplay turn during the later demo/integration work;
  retain the zero-action limitation of the current minimal-mode probe (D006).
- Project 10's #721 **Decision** field still needs synchronization to **Approved**
  under D011; #722 has no approval to synchronize.
  The configured credential received HTTP 403, “Resource not accessible by
  integration,” on the last attempt. The native issue dependency and recorded
  issue disposition were verified; a Project field update was not.
- #721 is approved and closed under D011 using the current evidence and D010's
  accepted limitations. No additional observation window is authorized.
- Consider #722 separately afterward. No Phase 1 completion decision has been
  made, and no platform, gameplay or model-inference readiness is implied.

## Issue closure and roadmap references

- #718, #719 and #720 remain closed: their scoped implementation repairs merged.
- #779 is closed with reason **not planned** under D010's non-reproducible,
  accepted-risk disposition. Original root-cause criteria remain unfulfilled.
- #721 is closed as completed under the explicit D011 human approval.
- #717 and #722 remain open. Engineering-review approval does not supply the
  outstanding phase-gate approval.
- #773–#776 remain open in their existing phases; deferral is not completion.
- Native Windows support/validation is carried into support-matrix and CI work
  (#724/#726); core/optional dependency profiles remain in #725. Live-model
  integration evidence remains separate in #585. Later lightweight-demo,
  real-turn scenario and canonical demo work remain in #654/#751/#658.
- GitHub hierarchy: #717 is the Phase 1 parent of #718–#722. #722 is a gate,
  not the parent issue. #779 is now a native child and required blocker of #721;
  #721 remains a required blocker of #722.

### Repository synchronization — 2026-10-15

Updated 18 existing issue bodies with targeted status/scope references to this
decision record: #717–#722, #779, #773–#776, #724–#726, #654, #751, #658 and
#585. Preserved their previous bodies, acceptance checklists and issue states;
no issues were closed or reopened. Read-back verification confirms the native
relationships and the open review/gate/remediation issues.

The decision file and evidence are published in
[documentation PR #780](https://github.com/dhar174/tiny_village/pull/780).
That PR closes no issues and supplies no new human review or phase approval.
Project Decision synchronization remains outstanding as described above.
