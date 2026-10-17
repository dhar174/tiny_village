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

## Review status

The current scope questions have been answered (D002, D004–D006). Review #721
remains **Deferred** under D001. Revisit its disposition after #779 supplies
the required startup evidence; this walkthrough does not approve Phase 1.

The [#779 investigation plan](superpowers/plans/2026-10-15-issue-779-startup-investigation.md)
was approved under D007, and its bounded cycle has concluded. Its original
16-case diagnostic budget remains exhausted; no production repair is approved.

The [failure-capture follow-up](superpowers/plans/2026-10-16-issue-779-failure-capture.md)
is approved under D008. Its test-harness implementation and single bounded
verification window may proceed. This does not change #721's disposition or
approve a runtime correction.

The [D008 implementation and verification report](testing/issue-779-failure-capture.md)
records 11 new capture checks, 81 existing methods and three original probes
passing in the single approved window. Its nine-batch/three-probe budget is
consumed. The historical timeout did not recur, so its cause remains unknown;
#779 stays open and #721's Deferred disposition is unchanged.
Merged [PR #781](https://github.com/dhar174/tiny_village/pull/781) contains this
follow-up; merging it is not a human approval or issue closure.

The [D009 post-merge report](testing/phase-1-post-merge-verification.md) on merged
main `77e4b94` contains 94 successful test-method summaries and three zero-exit
probes. One lifecycle supervisor completion record is missing after a transport
disconnect; its successful 12-test summary and all child exits are retained.
Only previously unstarted targets were continued, once each. No target was
rerun, and the missing supervisor result is not labeled PASS. The authorized
window is consumed; #779 risk acceptance/closure and #721/#722 decisions remain
pending. Neither merging the PRs nor "Proceed" supplied those dispositions.

The [approved observation cycle](testing/issue-779-startup-investigation.md)
has concluded: 51 affected regression methods pass and all 16 predeclared
diagnostic cases exit successfully. The historical timeout cause remains
unconfirmed. No remedy or new review disposition is approved by those results;
the diagnostic budget is exhausted and #779 remains open.

## Outstanding actions and limits

- #779 remains open: decide whether to accept the unknown historical cause and
  D009's supervisor limitation, or require further evidence. Closure on the
  proposed non-reproducible rationale requires an explicit human decision;
  no such acceptance has been inferred. Further observations require a new scope.
- Include native Windows execution and cache-permission checks when planning
  the later platform work, before any Windows-support claim (D004).
- Include live model download and inference checks in the later ML/LLM work
  before making readiness claims for those features (D005).
- Demonstrate a real gameplay turn during the later demo/integration work;
  retain the zero-action limitation of the current minimal-mode probe (D006).
- Project 10's **Decision** field still needs synchronization to **Deferred**.
  The configured credential received HTTP 403, “Resource not accessible by
  integration,” on the last attempt. The native issue dependency and recorded
  issue disposition were verified; a Project field update was not.
- Revisit #721 after the required evidence is complete. An explicit human
  **Approved** decision is required before closing it.
- Consider #722 separately afterward. No Phase 1 completion decision has been
  made, and no platform, gameplay or model-inference readiness is implied.

## Issue closure and roadmap references

- #718, #719 and #720 remain closed: their scoped implementation repairs merged.
- #717, #721, #722 and #779 remain open. The implementation merges and passing
  observation cycle do not supply the outstanding review/gate approval or
  resolve the unexplained timeout issue.
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
