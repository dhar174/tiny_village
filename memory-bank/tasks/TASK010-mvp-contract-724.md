# [TASK010] — Define the MVP scenario, support matrix and release budgets (#724)

**Status:** In Progress — draft PR #783 published; human agreement pending

**Added:** 2026-10-20

**Updated:** 2026-10-20 (America/New_York)

## Original Request

“Lets do it,” following Phase 1 completion and the proposed next task #724.

## Approach and scope

Prepare #724's planning deliverable against actual runtime and issue evidence.
Separate proposed behavior/support/budgets from current capabilities; preserve
D002/D004–D006 and D010–D012. Product implementation belongs to the existing
later tasks. Human decisions stay in docs/decisions.md.

PR #782 remains open at 68da3af. This task is on a documentation branch based
on that head so its decision record is present. Publish a stacked draft PR
against codex/phase1-post-merge-review, with #782 as a merge prerequisite.

## Plan and progress

| ID | Step | Status | Updated | Evidence / next step |
| --- | --- | --- | --- | --- |
| 10.1 | Verify repository and prerequisites | Complete | 2026-10-20 | main 77e4b94; #722 Approved/closed under D012; #724 open |
| 10.2 | Research runtime and related issues | Complete | 2026-10-20 | Static reads of CLI/demo, strategy/actions, time, UI/performance, dependencies and roadmap tasks |
| 10.3 | Draft scenario/support/budgets | Complete | 2026-10-20 | docs/superpowers/specs/2026-10-20-issue-724-mvp-contract.md; all choices proposed |
| 10.4 | Review and publish draft | Complete | 2026-10-20 | Draft PR #783 stacked on #782; #724 comment and additive status block verified |
| 10.5 | Obtain human agreement | Pending | 2026-10-20 | Approve exact spec revision or revise; #724 acceptance criteria remain unchecked |

## Progress log

### 2026-10-20

- Found the new executor checkout was stale (610b4f8), fetched main 77e4b94
  and verified current GitHub #722/#724 and PR #782 status before working.
- Created codex/issue-724-mvp-contract from remote #782 head 68da3af, without
  merging #782 or changing runtime code. No worktree or delegation used.
- Read canonical Memory Bank, contributor guidance, decisions and Phase 1
  report; inspected actual main/demo, actions/strategy, calendar, controls and
  performance collection. Cached issue research under /workspace/scratch/issue-724/.
- Confirmed no scenario/seed/tick/result CLI contract exists, minimal demo
  patches modules, unit scales differ, and frame timing includes FPS limiting.
  Earlier startup success does not demonstrate real actions or release budgets.
- Asked the presentation preference question; the maintainer explicitly chose
  headless proof plus a Pygame view. D013 in docs/decisions.md records that
  choice only; detailed fixture/support/budgets still need agreement. No runtime
  implementation or benchmark window was requested/run.
- Drafted a deterministic three-villager economic loop, explicit conserved
  totals, failure cases, proposed CLI/result interfaces, support matrix, UX/UI
  constraints, numeric budgets and a bounded repeatable ten-actor soak.
- Mapped implementation and design/engineering/release reviews to existing
  roadmap issues. Drafting does not close #724, approve Phase 2's gate, modify
  native dependency edges or claim Project custom fields are synchronized.
- Published and attached [draft PR #783](https://github.com/dhar174/tiny_village/pull/783),
  base codex/phase1-post-merge-review. Its focused diff contains six Markdown
  files; #782 must merge first, followed by retargeting/rechecking this PR.
- Posted and read back the [#724 draft handoff](https://github.com/dhar174/tiny_village/issues/724#issuecomment-6296391132)
  and added a dedicated draft-status block while preserving the entire original
  issue body. #724 remains open with original acceptance checkboxes unchecked.
- Checked document link targets, explicit future-command labels, fixture
  arithmetic and scope/measurement consistency. No tests/lint or runtime runs.
  Publication receipts are in /workspace/scratch/issue-724/publication-receipt.json.
