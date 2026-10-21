# TASK016 — Pygame UI structure, wireframes and states (#729)

**Status:** Approved under D023; GitHub publication/closure pending authentication
**Added/Updated:** 2026-10-21 (America/New_York)

## Request and scope

Begin #729 together with options, visual comparisons and recommendations,
following the reviewed #727/#728 choices. The user explicitly selected 12
interface choices. The single authoritative record is
[D020](../../docs/decisions.md#d020--select-interface-behavior-for-729), under D003.
This task defines the observer UI; no product implementation or phase approval.

## Approach and artifact

Base main 4c9bbfd (#792 merged). Research current input/panels/persistence,
compare layouts, review choices, consolidate dimensioned wireframes and state/
focus requirements, then request review of the exact written artifact.

[Draft UI specification](../../docs/superpowers/specs/2026-10-20-issue-729-ui-design.md)
contains the selected right-sidebar structure, all required states and the
written-review proposals. Existing #727/#728 final dispositions remain pending;
the user explicitly authorized this preliminary design walkthrough.

## Progress

| Step | Status | Evidence |
| --- | --- | --- |
| Inspect existing layout, inputs and planning contracts | Complete | Base/source hashes and current gaps in draft/receipt. |
| Compare main-screen layouts | Complete | Generated A/B/C board; user chose A; reference hash retained. |
| Review interface choices | Complete | D020 records all 12 explicit selections and alternatives. |
| Consolidate geometry/state/input specification | Complete | Two dimensioned layouts, state matrix, focus/shortcut and failure requirements. |
| Self-review bounds/links/claims | Complete | 18 region rectangles and 18 header slots checked; local links, choice counts and documentation-only scope. |
| Complete written review | Approved | [D023](../../docs/decisions.md#d023--approve-design-planning-727729-and-combined-review-730); issue synchronization pending authentication |

## Log

- 2026-10-20: user selected bottom-drawer Summary/Events and compact collapsed
  outcome. Created documentation branch from merged main 4c9bbfd. #791/#792
  merges and open #727/#728/#729/#730 states verified.
- Preserved generated comparison unchanged; sample inventory/bars/text are
  illustrative, not scenario state. Explicit rectangles at 1280×720 and
  1024×768 fit, with main regions non-overlapping in collapsed/expanded modes.
  No rendered font-fit, behavior, performance or usability-pass claim.
- Input proposals distinguish text/modal/widget/global ownership and one-layer
  Escape. Exact geometry, quicksave return, Page Up/Down focus precedence,
  confirmation timing, history buffer and terminal-save/close behavior remain
  visibly proposed for written review, not silently approved by the 12 answers.
- No product code, runtime prototype, production assets, locks, dependencies
  or workflows changed. #730 is the next combined review after written
  dispositions and planning completion; no automatic approval or merge.
- PR #793 review follow-through: defined the visible, focusable Quicksave (S)
  session-menu action separately from named Save. Both quicksave routes share
  the command, dedicated slot and state checks; menu invocation retains focus
  and paused playback. Added mouse/keyboard acceptance coverage. The proposed
  return-to-prior-playback policy applies only to S outside the menu and still
  needs written review. Static documentation checks passed; no runtime behavior
  was implemented or tested.

- 2026-10-21 (America/New_York): TASK017 consolidates the remaining selections
  from D021 into this specification. Complete written disposition remains
  pending; no runtime validation or later design/phase approval is inferred.

- 2026-10-21 (America/New_York): complete planning approval recorded in D023.
  #729 closure is authorized; no closure is claimed while GitHub authentication
  returns 401. No new runtime or phase approval. TASK019 prepares #731.
