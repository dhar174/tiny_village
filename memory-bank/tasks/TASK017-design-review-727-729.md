# TASK017 — Remaining design review for #727–#729

**Status:** In progress; written dispositions pending
**Added/Updated:** 2026-10-21 (UTC)

## Request and scope

Walk through the outstanding UX, visual and UI proposals one decision at a
time, explaining options and recommendations. Record human selections only in
[D021](../../docs/decisions.md#d021--review-remaining-design-proposals-for-727729).
No runtime implementation or automatic issue/phase approval.

## Approach

Continue the architectural design review from merged main `3e1c7ef` (#793).
Reuse D020's explicit selections for overlapping #727 proposals. Review the
remaining method, asset/font/token and UI defaults; reflect selected changes
in the specifications, then obtain final written dispositions before #730.

## Progress

| Step | Status | Evidence |
| --- | --- | --- |
| Reconcile existing choices and remaining proposals | Complete | #727 section 5, #728 section 5 and #729 section 5; D020 resolves three earlier UX proposals. |
| Review remaining proposals | In progress | D021 R01 recorded; remaining visual/UI defaults pending. |
| Final written-specification dispositions | Pending | #727/#728/#729 remain open. |
| Combined human design review #730 | Pending | Separate decision after planning acceptance. |

## Log

- 2026-10-21 UTC: user requested the step-through. Created documentation branch
  from merged main `3e1c7ef`. Recorded the first submitted choice in D021.
  Existing specification snapshots remain unchanged until consolidation. A
  static check caught an attempted edit to #727's hashed research source;
  reverted that edit to retain the original evidence. No runtime walkthrough
  or tests executed.
