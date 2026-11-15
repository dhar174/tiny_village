# TASK030 — Production Goal contract (#802)

Status: In Progress. Updated: 2026-11-14 (America/New_York).
User approved the chat plan; implementation uses codex/802-goal-contract based on
main a3e39f3. Goal: reconcile full production Goal conditions, completion state,
Plan evaluation and bounded real-Goal feasibility. See the approved plan at
../../docs/superpowers/plans/2026-11-14-issue-802-goal-contract.md.

## Progress / implementation ledger
- Baseline: seven new real-world tests fail against unchanged #819 source. Original
  #736 blocked sources/captures remain untouched. Local /workspace/scratch/802-clean-red.log.
- Implemented boolean-bucket normalization, common condition iteration/extraction,
  native graph-first/live vs explicit actor state, partner isolation, contextual errors,
  production planner/graph delegation, and bounded feasibility with optional actions.
- Nine final targeted tests pass (5.582s). Initial partner fixture lacked required personality
  and time-manager arguments; corrected before counting product baseline failures.
- Ruling: repair the food inventory predicate's sum(integer) defect because real
  generated food conditions otherwise cannot evaluate. Scope is one count expression;
  other inventory categories and economic semantics remain separate. Cost if wrong:
  inventory presence predicates could report incorrect food availability.
- Ruling: use this Memory Bank task as the execution ledger, avoiding a competing
  status source; cloud skill scripts are not installed as filesystem resources.
- gh view802 failed: installed CLI has no top-level view; gh issue view802 --json
  successfully read the issue. No mem0ry4ai tools available; unrelated Notion not used.

- Independent review found stale non-character state (P2) and non-contextual literal
  syntax errors (P3); both repaired with observed red→green regressions in the single
  integration pass. The reviewer did not run tests or inspect those unpublished fixes.
- Draft PR #820 created and attached. Source checkpoint 2eb3598; final core125+3CLI probes passes at2eb3598.
  Dev20/visual69 pass at 7b22725 (before the diagnostic-only syntax correction).
- Ruling: include the P3 malformed-syntax correction because contextual errors are
  part of the approved interface; bare SyntaxError bypasses callers catching ValueError.
  Cost if wrong: callers could lose parser-specific exception handling.

## Remaining
Local verification and independent review complete; final hosted CI/review and merge remain pending.
#802 remains open pending merge and merged-source verification. No broader Goal
catalog, graph path scoring, full Plan execution/replanning, or readiness claim.
