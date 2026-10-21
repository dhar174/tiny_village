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
| Review remaining proposals | In progress | D021 R01–R11 recorded; terminal-save/window-close proposals pending. |
| Final written-specification dispositions | Pending | #727/#728/#729 remain open. |
| Combined human design review #730 | Pending | Separate decision after planning acceptance. |

## Log

- 2026-10-21 UTC: user requested the step-through. Created documentation branch
  from merged main `3e1c7ef`. Recorded the first submitted choice in D021.
  Existing specification snapshots remain unchanged until consolidation. A
  static check caught an attempted edit to #727's hashed research source;
  reverted that edit to retain the original evidence. No runtime walkthrough
  or tests executed.
- 2026-10-21 UTC: recorded the submitted environment-foundation selection in
  D021 R02. Palette/font and remaining UI proposals are still awaiting choices;
  no production asset import or implementation started.
- 2026-10-21 (America/New_York): recorded acceptance of the existing-color trial
  approach in D021 R03. Font pairing is the next choice. No rendered contrast
  test, palette modification or runtime implementation performed.
- 2026-10-21 (America/New_York): recorded the submitted font-pairing choice in
  D021 R04. Type sizes/spacing are next. No font binary imported or native
  rendering/character-coverage test executed.
- 2026-10-21 (America/New_York): recorded the submitted type-size/spacing
  baseline in D021 R05. Animation/layer production scope is the next proposal;
  no font-fit or runtime behavior has been tested.
- 2026-10-21 (America/New_York): recorded the submitted animation/layer scope
  in D021 R06. Remaining #728 proposals now have selections; complete written
  acceptance is still pending. Next: #729 dimensions. No artwork or rendering
  implementation performed.
- 2026-10-21 (America/New_York): recorded the submitted initial panel dimensions
  in D021 R07. Next: quicksave return policy outside the session menu. No layout
  implementation or native font-fit test performed.
- 2026-10-21 (America/New_York): recorded the submitted quicksave return policy
  in D021 R08. Next: keyboard focus/Page Up/Down precedence. No runtime save or
  input behavior implemented or tested.
- 2026-10-21 (America/New_York): recorded the submitted Page Up/Down focus
  precedence in D021 R09. Next: routine-confirmation duration/placement. No
  runtime keyboard behavior implemented or tested.
- 2026-10-21 (America/New_York): recorded the submitted routine-confirmation
  policy in D021 R10. Next: displayed event-history buffer. No notification
  behavior implemented or tested.
- 2026-10-21 (America/New_York): recorded the submitted UI history-buffer
  policy in D021 R11. Next: terminal saves, then window-close handling. No
  history storage/display behavior implemented or tested.
