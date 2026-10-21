# TASK014 — MVP journeys, controls and simulation feedback (#727)

**Status:** Awaiting written-specification review; #727 open
**Added/Updated:** 2026-10-20 (America/New_York)

## Request and scope

Review #727 together, presenting viable options, tradeoffs and recommendations
for each decision. Preserve the accepted #724 contract and expansive TinyVillage
vision. Record all human choices only in docs/decisions.md under D003.
The user explicitly selected 15 choices; their single authoritative record is
[D018](../../docs/decisions.md#d018--select-mvp-ux-behavior-for-727).

## Approach and artifact

Architectural UX specification path: research existing behavior and roadmap,
review choices one at a time, consolidate written journeys and proposed
walkthrough, then obtain human review of the exact written artifact. No product
implementation or automatic approval follows conversational choices.

[Draft specification](../../docs/superpowers/specs/2026-10-20-issue-727-ux-design.md)
is based on merged main 969f6b9 (#790). #727 defines planning acceptance;
#728/#729 refine aesthetics/UI, #730 reviews design before UI implementation,
and #731/#732 retain engineering and phase-gate decisions. Runtime/persistence
and release acceptance stay with their mapped downstream issues.

## Progress

| Step | Status | Evidence |
| --- | --- | --- |
| Inspect contract, runtime and issue dependencies | Complete | Exact base and static findings in draft section 1. |
| Review interaction alternatives | Complete | D018 records all 15 explicit selections, alternatives and rationale. |
| Consolidate journeys/controls/walkthrough | Complete | Draft sections 2–4; proposed connective defaults separated in section 5. |
| Self-review consistency and scope | Complete | Cross-checked choices, links, ownership and planning-versus-runtime claims. |
| Human review of written specification | Pending | #727 remains open; no final written-spec acceptance recorded. |

## Log

- 2026-10-20: user selected the final guidance option. Created a focused
  documentation branch from merged main 969f6b9, preserving #726 history.
  Inspected relevant runtime paths, #724 and existing #727–#732/persistence/
  Phase 8 issues. Current minimal demo still explicitly exercises zero actions;
  existing save/load is incomplete. Neither is relabeled as working UX.
- Consolidated selected behavior, failure/unsaved-state paths and native
  keyboard/mouse walkthrough. Future commands, timing targets and human review
  are requirements rather than executed evidence. No runtime tests, benchmarks,
  dependency installs, product code, locks or CI changes for this draft.
- Proposed defaults are visibly separated for written review. Draft publication
  is for human review; keep #727 and later human gates open.
