# #738 walkthrough preparation

**Not yet ready for the gate decision:** authenticated initial/final board receipts
and #806/#737 completion are still missing. This packet is preparation only.

1. Verify prerequisites: Phase2 approval D025; scoped734 triage (D027),735 utility
   consolidation (D028),736 reconciliation/carry-forward (D030),737 dispositions
   and publication (D031–D033). Recheck live states and immutable evidence at the
   walkthrough rather than relying on this preparation date.
2. Review737's current332-issue/24-PR ownership/dependency matrix, reviewed source
   heads, actual CI/mergeability and activation gates. Historical green heads are
   not current integration acceptance. Retained dependent PRs refresh only after
   their prerequisites merge.709/711/715 stay open pending807/808/809 replacements.
3. Inspect the approved initial live plan, apply checkpoint and independent receipt;
   then806/737 closure evidence and the independently verified final readiness
   plan/receipt. Review all12 fields and six views, preservation of human edits and
   the no-remote-CAS limitation. Missing proof remains a blocker.
4. Walk through retained runtime gaps:799 identity/provenance,800 location lifecycle,
   801 animation lifecycle,802 Goal/Plan contracts,803 motive contribution,741
   replanning and774 utility/falsy-ID debt. Their native owners/blockers and unchecked
   requirements remain.735/736 consolidation does not repair these runtime gaps.
5. Confirm D023's accepted727/728/729/730 planning and later recorded refinements:
   headless+Pygame product scope, pixel-art direction, readable panels, explicit
   playback/failure/save policies and separate design acceptance. These establish
   approved design intent, not implemented UI/aesthetic or gameplay readiness.
6. Present an evidence-based recommendation after the actual readbacks. Options:
   Approved accepts the scoped Phase3 work with stated limits; Changes requested
   creates separately owned required remediations/blockers; Deferred records why
   review cannot yet conclude. Keep738 open for the latter two decisions.

Human reviewer: dhar174, or an explicitly delegated maintainer. Record the actual
reviewer, local date/timezone, evidence, rationale, limits and one decision only in
docs/decisions.md. Never infer approval from a checklist, bot, passing tests or
elapsed time. Phase4 implementation remains blocked until separate738 approval.

Evidence entry points (recheck at the walkthrough):
- `../issue-734-publication/README.md` and `../issue-734-publication-review/README.md`.
- `../../testing/issue-735-utility-consolidation.md` and `../../testing/issue-736-post-merge-verification.md`.
- `../../testing/phase-2-gate-readiness.md` and `../../testing/issue-730-design-consistency-review.md`.
- `../../superpowers/specs/2026-10-20-issue-727-ux-design.md`, `2026-10-20-issue-728-visual-design.md` and `2026-10-20-issue-729-ui-design.md`.
