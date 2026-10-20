# [TASK010] — Define the MVP scenario, support matrix and release budgets (#724)

**Status:** In Progress — D014 acceptance recorded; integration to main pending

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
| 10.5 | Obtain human agreement | Complete | 2026-10-20 | D014 accepts the contract with explicit original-vision preservation guardrails |
| 10.6 | Integrate approved contract to main | In Progress | 2026-10-20 | Ready PR #784 targets main and closes #724 on merge; source/issue read-back verified |

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

### 2026-10-20 — accepted contract and merge-target repair

- The maintainer accepted the contract on the condition that the original
  TinyVillage vision remains expansive. D014 records the exact confirmation;
  section 1 now makes fixture boundaries and feature-preservation explicit.
- Verified #782 merged as 79e5683, then #783 merged as a9ed214 into #782's old
  branch. Main did not contain the MVP specification. Created a branch from
  current main and cherry-picked the two exact #783 commits before recording
  the new decision; no runtime changes or feature removals.
- Publish a focused PR to main with a closing reference for #724. Keep the
  issue open until integration; its human agreement is supplied by D014.
- Next ordered task: #725, core/development dependency locks and optional
  visual/NLP/LLM/GPU profiles, including review of #704 and verified installation.
  #727 UX planning can follow #724 separately; #654 then #726 follow #725.
- Published and attached [integration PR #784](https://github.com/dhar174/tiny_village/pull/784),
  targeting main with a closing reference for #724. Its diff is six Markdown
  files and contains no runtime edits. Local specification links resolve.
- Read back the [D014/integration handoff on #724](https://github.com/dhar174/tiny_village/issues/724#issuecomment-6297463774).
  The issue remains open pending integration; agreement criteria are checked
  and the status block identifies the merge target repair. No dependency edges
  or Project custom fields were changed.
