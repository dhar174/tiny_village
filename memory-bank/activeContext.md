## Current Work Focus

- TASK021 / #734 complete under D027: #797 merged at 2c3bc57, approved
  publication independently verified and #734 closed. All 307 annotations
  preserve original bodies/assignees; 80 missing-parent links, seven narrow
  closures, #799/#800/#801 and seven blocker edges verified, zero cycles.
  #602/#609 and #733/#738 remain open. PR #798 retains current publication
  receipts separately from the frozen historical review package. No Project
  writes; configuration remains #737's pending handoff after retained HTTP 403.
- TASK022 / #735 complete: #703 merged at 323dad6 and #735 closed; #714
  remains superseded under D028. #798 merged at 11b6e93 with review fixes and
  retained publication receipts. Required merged-main checks pass.
- TASK023 / #736: eight dispositions selected under D029; #393 is the last
  pending choice. Review baseline is main 11b6e93 and merged #349. #420's five
  original methods and one selected #417 utility method pass; other findings
  retain their documented source-review/probe limits. Finish the walkthrough
  and scoped consolidation plan before execution. No implementation, rebase
  or closure; #737 and human gate #738 remain later steps.

- TASK020 / #732 is complete: dhar174 explicitly approved Phase 2 under D025.
  #732 and parent #723 are closed/completed with immutable decision links
  verified; all native Phase 2 children were closed first. #734's native H2
  dependency is closed. PR #795 merged at a7dea4c; Project fields remain
  unsynchronized after HTTP 403. Gate #738 retains its own later human decision.

- TASK019 / #731: Approved under D024 within the reviewed Python 3.12/Linux
  foundation, with all documented evidence limits. No new blocking remediation
  identified; current hosted/retained source evidence stands. Published in #795,
  #731 is closed/completed with its immutable approval link verified. See TASK019.

- D023 records dhar174's explicit approval of #727/#728/#729 planning and #730
  combined design review on 2026-10-21. TASK014–TASK018 are complete for their
  planning scope. All four issues are closed/completed; immutable decision
  links and states were verified after repository authentication recovered.
  Existing PR #795 carries the record and engineering handoff. Project 10
  fields remain unsynchronized after HTTP 403. #731/#732 and #723 stay open;
  do not repeat settled design choices or infer further human approval.

- TASK013 / #726 is complete and closed. #789 merged at ac0e25d; main and
  both manual ML profiles pass 132 cases + three CLI probes, all five artifact
  digests verified. Owner activated tiny_ruleset; API readback confirms the
  effective required check, GitHub Actions source and strict/up-to-date policy.
  Existing rules/reviews/bypasses are preserved. Evidence PR #790 merged at
  969f6b9. Next: review #727 UX journeys and plan #728 aesthetic direction, with
  separate human review. No phase gate is approved. See TASK013/validation report.

- Product requirement clarification (2026-10-20): apply [D015](../docs/decisions.md#d015--require-ai-capabilities-in-the-final-product)
  when scoping #654/#776 and later AI work. Lightweight testing profiles do
  not establish readiness of the final AI-enabled product.

- TASK012 / #654 and #776 are complete through merged #788 at eb5f4c9.
  Real basic memory and headless startup are verified; gameplay and live inference
  remain separate. Historical full-profile and PR-fix evidence is retained in
  docs/testing/issue-654-lightweight-runtime-validation.md; TASK013 supplies the
  newer merged-source baseline. Next ordered implementation is #726.

- TASK011 / #725 merged via #785 at 8a899650 and #725 closed. Its eight hash
  locks and legacy manifest are unchanged. Original dependency validation is
  historical; TASK012 supplies the newer runtime evidence. GPU/model inference
  and gameplay acceptance remain separate. No phase-gate approval supplied.

- TASK010 / #724 is complete: #784 integrated the accepted contract to main
  at 8d6b254 and #724 closed. D014 preserves the original expansive vision;
  scenario limits are fixture data. TASK011 now implements #725; UX planning
  #727 also follows #724 with its own later design approval requirements.

- #718 is complete: PR #706 merged and the issue closed on 2026-10-14.
- TASK003 / #719 is complete: PR #777 merged as `0ba9061`; #719 closed.
- Post-merge verification passes 54/54 tests plus cold-getter/fallback/minimal
  probes. Reduced minimal mode still executes zero actions.
- TASK004 / #720 is complete: PR #778 merged as `93079fd`; #720 closed.
- TASK005 / #721 is approved and closed under D011 for Linux/offline startup
  at main 77e4b94. D010 accepts the unknown historical timeout cause and one
  lifecycle supervisor reporting gap. Historical failed runs remain evidence;
  Windows, live models and real gameplay retain their later agreed scope.
- Evidence and limits: `docs/testing/issue-719-graph-initialization-validation.md`.

## Recent Changes Reflected in the Repository

- `memory-bank/memory-bank-instructions.md` was updated with TinyVillage-
  specific guidance and documentation sources.
- The repository's canonical contributor guidance lives in root `AGENTS.md`,
  with `docs/reference/AGENTS.md` treated as supplemental background.
- Current docs distinguish between current documentation and archived historical
  material under `docs/archived/`.

## Earlier handoff history (superseded by Current Work Focus)

- TASK009 verifies merged main 77e4b94 after #780/#781. All 94 test methods
  report success and three probes exit 0. Lifecycle has a preserved missing
  supervisor terminal record after transport loss; its 12-test OK summary and
  all child zero-exits survive. Only unstarted targets were continued once.
  No blanket clean-supervisor claim. See docs/testing/phase-1-post-merge-verification.md.
  Evidence published in draft PR #782 and on #779/#721; D009 budget consumed.
  D010 now accepts the unknown cause and supervisor gap; #779 is closed as
  unable to reproduce, diagnostics added (2026-10-17 UTC). D011 subsequently
  approves #721. D012 now separately approves #722 and completes parent #717
  (2026-10-17 America/New_York). Phase 2 is unblocked; the next ordered task
  is #724: define the MVP scenario, support matrix and release budgets.
  TASK010 records the accepted contract; integrate it to main, then begin #725.
- TASK008 implements the scoped #779 failure-capture follow-up on a separate
  branch, published as PR #781 (now merged with #780). See docs/superpowers/plans/2026-10-16-issue-779-failure-capture.md.
  D008 approves implementation and one bounded observation window. The shared
  capture helper and both integrations are verified: 11 new checks, 81 existing
  methods and three probes pass. See docs/testing/issue-779-failure-capture.md.
  D008 observation budget consumed. Deadline review correction passes 13 focused
  capture checks; the prior application matrix was not rerun at that handoff.
  #779 stayed open then; the subsequent D010 closure accepts its unknown cause.
  Findings posted on #779/#721.
  #780/#781 merged. Deadline fix a32cdb2 is included; review updated and thread resolved.
- TASK007 synchronized 18 actual GitHub issue bodies with the decision record,
  preserving original checklists and states. #779 is now a native child/blocker
  of #721; #721 blocks #722. Docs PR #780 publishes decisions/evidence.
  That synchronization kept #718–#720 closed and #717/#721/#722/#779 open.
  Subsequent #779 closure is recorded under D010 and #721 approval under D011;
  Subsequent D012 separately approves #722 and completes #717.
- TASK006 / #779's approved investigation plan is in
  `docs/superpowers/plans/2026-10-15-issue-779-startup-investigation.md`.
  Approval is D007; the bounded cycle is complete. All 51 affected methods pass
  and all 16 diagnostic cases exit 0, but historical timeouts do not recur and
  their cause remains unconfirmed. See docs/testing/issue-779-startup-investigation.md.
  No runtime remedy or further observation budget was approved by D007;
  subsequent D010 accepts closure without establishing a cause.
- Merged-main verification completes with timing failures/incomplete batches;
  cache passes 26/27, and 30 tests in five other complete suites pass. Both
  affected import orders pass instrumented diagnostics, but cause is unproven.
  dhar174 explicitly confirmed Deferred; the decision is recorded on #721 and
  #779 is a verified native blocker. The user also confirmed scope; maintain
  all current and future review decisions in `docs/decisions.md`.
  Project Decision synchronization was denied (403); issue disposition is
  authoritative, with Project field update still outstanding.
  Source repair, approval and downstream phase advancement are separate decisions.
- D011 records #721 engineering approval; D012 separately records #722 gate
  approval and Phase 1 completion. Later phase reviews/gates remain required.
  Project 10 needs #721/#722 Decision Approved and corresponding readiness
  synchronization; unchanged credentials previously returned HTTP 403.
- Retain #773/#774 integration/utility failures and #775/#776 Door-definition/
  optional-spaCy limits, scheduled under existing parent tasks and human gates.

## Active Considerations

- Some current docs may disagree with older status notes; root `README.md` and
  root `AGENTS.md` should generally be prioritized over older archived docs
  when they conflict.
- Optional dependencies for LLM/NLP features should be documented without
  making them sound mandatory for every workflow.
- Testing guidance strongly favors realistic objects and conservative use of
  mocks, especially for memory-related tests.
