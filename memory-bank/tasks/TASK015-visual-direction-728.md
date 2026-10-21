# TASK015 — Pixel-art direction and aesthetic constraints (#728)

**Status:** Awaiting written-specification review; #728 open
**Added/Updated:** 2026-10-20 (America/New_York)

## Request and scope

Review #728's options and recommendations together. The user requested pixel
art and explicitly selected 13 design choices, then requested side-by-side
palette concepts and chose A. The authoritative decision record is
[D019](../../docs/decisions.md#d019--select-pixel-art-direction-for-728), in the
single file required by D003. Preserve D014/D015 and the #724/#727 contracts.

## Approach and artifact

Architectural design workflow: inspect existing art/rendering, review choices,
generate palette references, research primary sources, consolidate a written
specification and request review of the exact artifact. No product implementation.

[Draft visual specification](../../docs/superpowers/specs/2026-10-20-issue-728-visual-design.md)
is based on merged main 7391861 (#791). Its generated comparison is a reference,
not a production sprite or approved UI layout. Asset/font candidates and exact
tokens/type/spacing values are labeled proposed defaults for written review.

## Progress

| Step | Status | Evidence |
| --- | --- | --- |
| Inspect source, constraints and prior decisions | Complete | Base 7391861; current placeholder/flat-map boundaries documented. |
| Review pixel-art choices | Complete | D019 contains 13 explicit selections, alternatives and rationale. |
| Compare palettes | Complete | Generated reference preserved unchanged; user selected A; provenance/hash in receipt. |
| Research asset/font candidates | Complete | Primary pages/OFL metadata, sheet inspection, hashes and fit/coverage gaps retained. |
| Consolidate and self-review draft | Complete | Visual spec, static token ratios, local links and scope/choice checks. |
| Human written-specification review | Pending | #728 remains open; no final written disposition or phase approval recorded. |

## Log

- 2026-10-20: user selected contextual icons for the selected villager. Created
  a documentation branch from merged main 7391861, preserving all prior work.
  #791 merge verified; #727 remains open without a recorded final disposition.
  Continue the explicitly requested preliminary #728 design; do not infer
  prerequisite closure or phase advancement from the merge.
- Verified candidate metadata: Slates CC-BY 4.0/32×32; Buch and Eldiran CC0
  with coverage/style/animation gaps; Kenney 16×16 mismatch; Cainos published
  no-redistribution term; LPC mixed author/license obligations. Font OFL files
  and metadata inspected for Atkinson Hyperlegible, Noto Sans and Pixelify Sans.
  No third-party art/font binary added to runtime assets.
- Eight proposed solid-token contrast checks pass their stated arithmetic
  thresholds. No rendered-UI, font-coverage, animation, performance or general
  accessibility-pass claim. Generated board copied byte-for-byte to docs with
  SHA-256/provenance. Research copies remain in /workspace/scratch/issue-728.
- Draft publication is for written review. #729/#730 and later gates remain
  separate; no product code, runtime assets, locks or workflows changed.

- 2026-10-20 follow-through: #792 merge verified at 4c9bbfd. #728 remains
  open without a final written disposition; specific production assets/fonts/
  tokens are not adopted. The user explicitly requested the preliminary #729
  walkthrough, tracked in TASK016; no prerequisite closure or phase approval.
