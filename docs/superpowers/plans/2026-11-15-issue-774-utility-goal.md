# Resolve #774 utility Goal contract

User approved this plan in chat (“Please implement the plan”). Base main46461b0.

## Goal and constraints
Give utility scoring an explicit UtilityGoal metadata type, preserving Goal as its
compatibility alias. Leave tiny_characters.Goal construction/completion unchanged.
Preserve native #740 boundaries, existing scoring/default-priority behavior, locks,
search bounds and timeouts. This does not complete #218/#741 or #821's catalog work.

## Tasks
- [x] Reproduce the 26-case baseline:16 passes,1 failure,9 errors,0 skips.
- [x] Add explicit UtilityGoal(name=None,target_effects=None,priority=.5,score=None,**kwargs),
  independent of Character imports. Keep priority/score/urgency/deadline/description/
  attributes metadata; score defaults to priority, urgency defaults to priority if None.
- [x] Replace fixture Goal import guessing and fixture Goal substitutes with explicit
  production metadata. Preserve0/empty-string/False Action IDs; only None generates one.
- [x] Register all26 cases plus new role/real-world bridge coverage in core CI.
- [x] Run core/dev/visual, independent review, publish focused PR #822 and exact local receipts.
- [ ] Verify hosted checks; close774 only after merge+merged-source verification.

## Verification
Require26/26 passes without weakening assertions. Fresh character/utility/strategy-first
imports and reload must retain the same exported role/interface/results (not object
identity across reload). Check29.1 golden, urgency multiplier, plan simulation and input
preservation. A real full Goal/Character/GraphManager/Action must use native scoring,
leave state unchanged while evaluating, execute synchronously20→30 and complete.
Keep full-domain default utility priority. No fallback Goal may substitute for that proof.

Next roadmap: #218 real multi-step planning, review/rework candidate689; then #741
active-plan cursors/tick execution/replanning. Later reviews and gates remain separate.
