## Current Work Focus

- #718 is complete: PR #706 merged and the issue closed on 2026-10-14.
- TASK003 / #719 is complete: PR #777 merged as `0ba9061`; #719 closed.
- Post-merge verification passes 54/54 tests plus cold-getter/fallback/minimal
  probes. Reduced minimal mode still executes zero actions.
- TASK004 / #720 is complete: PR #778 merged as `93079fd`; #720 closed.
- TASK005 / #721 verifies merged main and prepares a human walkthrough on
  `codex/issue-721-review`. Historical cold-import timing failures and incomplete
  lifecycle/ownership batches prevent a blanket readiness claim. The earlier
  81 pre-merge checks remain historical evidence.
- Evidence and limits: `docs/testing/issue-719-graph-initialization-validation.md`.

## Recent Changes Reflected in the Repository

- `memory-bank/memory-bank-instructions.md` was updated with TinyVillage-
  specific guidance and documentation sources.
- The repository's canonical contributor guidance lives in root `AGENTS.md`,
  with `docs/reference/AGENTS.md` treated as supplemental background.
- Current docs distinguish between current documentation and archived historical
  material under `docs/archived/`.

## Immediate Next Steps

- TASK008 implements the scoped #779 failure-capture follow-up on a separate
  branch, published as draft PR #781 stacked on #780. See docs/superpowers/plans/2026-10-16-issue-779-failure-capture.md.
  D008 approves implementation and one bounded observation window. The shared
  capture helper and both integrations are verified: 11 new checks, 81 existing
  methods and three probes pass. See docs/testing/issue-779-failure-capture.md.
  D008 observation budget consumed; review draft PR #781 next. #779 stays open
  because the historical cause remains unknown. Findings posted on #779/#721.
  #780 is now ready for review, open and unmerged at 29063c3.
- TASK007 synchronized 18 actual GitHub issue bodies with the decision record,
  preserving original checklists and states. #779 is now a native child/blocker
  of #721; #721 blocks #722. Docs PR #780 publishes decisions/evidence.
  #718–#720 remain closed; #717/#721/#722/#779 remain open. No new approval.
- TASK006 / #779's approved investigation plan is in
  `docs/superpowers/plans/2026-10-15-issue-779-startup-investigation.md`.
  Approval is D007; the bounded cycle is complete. All 51 affected methods pass
  and all 16 diagnostic cases exit 0, but historical timeouts do not recur and
  their cause remains unconfirmed. See docs/testing/issue-779-startup-investigation.md.
  No runtime remedy or further observation budget is approved; #779 stays open.
- Merged-main verification completes with timing failures/incomplete batches;
  cache passes 26/27, and 30 tests in five other complete suites pass. Both
  affected import orders pass instrumented diagnostics, but cause is unproven.
  dhar174 explicitly confirmed Deferred; the decision is recorded on #721 and
  #779 is a verified native blocker. The user also confirmed scope; maintain
  all current and future review decisions in `docs/decisions.md`.
  Project Decision synchronization was denied (403); issue disposition is
  authoritative, with Project field update still outstanding.
  Source repair, approval and downstream phase advancement are separate decisions.
- Human startup review #721 and gate #722 follow; passing scoped tests does
  not approve either decision.
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
