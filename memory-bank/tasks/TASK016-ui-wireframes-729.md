# TASK016 — Pygame UI structure, wireframes and states (#729)

**Status:** Awaiting written-specification review; #729 open
**Added/Updated:** 2026-10-20 (America/New_York)

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
| Human written review | Pending | #729 remains open; #730/#732 and implementation are not approved. |

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
