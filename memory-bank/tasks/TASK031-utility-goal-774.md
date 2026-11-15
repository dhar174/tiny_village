# TASK031 — Explicit utility Goal interface (#774)

Status: In Progress. Updated: 2026-11-15 America/New_York.
User approved the plan at ../../docs/superpowers/plans/2026-11-15-issue-774-utility-goal.md.
Worktree /workspace/scratch/issue774-utility; branchcodex/774-utility-goal; base46461b0.

## Execution ledger
- Fresh bounded26-method baseline FAIL:16pass/1failure/9errors/0skips. Retained
  /workspace/scratch/774-baseline.json and .log. Failure is explicit action_id0;
  nine errors are full Goal construction using lightweight kwargs.
- New contract baseline:4methods,5failure records (three import-order subcases),
  0errors. The full native-Goal bridge already passed and characterizes preserved
  behavior; the other3 methods expose missing UtilityGoal.
- Implemented explicit UtilityGoal plus Goal alias, metadata defaults and no Character
  import guessing. Full Goal class untouched. Removed fixture Goal doubles; fixture
  action IDs now distinguish None from explicit falsy values. Registered both modules.
-26/26 tests PASS; new4+existing13 full-Goal methods PASS17/17. Reviewed exactdiff.
- Ruling: use this canonical Memory Bank task as the execution ledger; cloud skill
  scripts have no installed local path. No design deviation from approved plan.

- Canonical source a99310f passed core159+3 CLI probes/dev20/visual69, no failures,
  errors or skips. Supplementary planner/strategy selection passed11 tests.
- Independent read-only source review found one regression:3 consumers imported the
  removed MockGoal. Reproduced the validation-suite ImportError, then migrated those
  consumers to production UtilityGoal and registered its7 validation methods in core.
  Focused26+4+7 methods PASS37/37; existing simple utility script and advanced demo
  both exit0. No other blocking finding was reported. These consumer migrations are
  required compatibility repairs within the approved Goal-fixture reconciliation.
- Draft PR #822: https://github.com/dhar174/tiny_village/pull/822 (attached).

- Final source fb55bef passes core166+3 CLI probes/dev20/visual69, zero failures,
  errors or skips, actual exits0. Fresh60s bounded original26 PASS26/26. Independent
  final delta review resolved the sole finding with no remaining findings; it did
  not execute commands. Durable receipt: docs/testing/evidence/issue-774-local-results.json.

## Remaining
Exact-head hosted CI and automated review; leave PR unmerged for user handoff. Leave774 open
until repair merge and main verification. #218/#741/#821 remain separate; no broader
readiness or full live-loop Goal migration claim. No mem0ry4ai tools available.
