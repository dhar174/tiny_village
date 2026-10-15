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

## Outstanding actions and limits

- Investigate #779 and return the results to #721. These decisions do not
  establish a cause, approve an unreviewed repair, or change test deadlines.
- Include native Windows execution and cache-permission checks when planning
  the later platform work, before any Windows-support claim (D004).
- Project 10's **Decision** field still needs synchronization to **Deferred**.
  The configured credential received HTTP 403, “Resource not accessible by
  integration,” on the last attempt. The native issue dependency and recorded
  issue disposition were verified; a Project field update was not.
- Revisit #721 after the required evidence is complete. An explicit human
  **Approved** decision is required before closing it.
- Consider #722 separately afterward. No Phase 1 completion decision has been
  made, and no platform, gameplay or model-inference readiness is implied.
