# Complete #821 implementation plan

Execution outline of the approved conversational plan. User authorization:
“Please implement the plan.” Full issue scope and the explicit-completion and
unhashable-mutable-object policies were chosen by the user before execution.

1. Reproduce defects with real domain objects; unify completion through the utility
   dispatcher. Preserve full Goal Conditions and empty-condition True; empty
   metadata is False. Remove priority/prose completion and retain native units,
   finite target checks, aliases and planner tolerance.
2. Fix all five nonfood counters. Match public inventory data across current
   positive-quantity items, with optional category filtering. Preserve both Cook
   requirements and require cooked food, not cooked nonfood objects.
3. Derive employment from graph job or explicit hypothetical actor job, with no
   live fallback. Project supported JobRoles/Job names into scalar planning data;
   retain unsupported-data handling for unrelated goals. Track actor and partner
   job dependencies in request fingerprints. Preserve typed identity and reject
   ambiguous bare names.
4. Add Goal.evaluate_utility(), keeping the constructor callback and exact native
   three-argument convention, including bootstrap. Contextualize condition/scoring
   errors with original causes; Strategy cannot swallow required errors, and
   plan_request retains explicit ERROR rather than NO_PLAN/SATISFIED. Preserve the
   separate five-argument utility helper.
5. Fix the Goal equality operand; retain its existing structural fields. Make
   Goal/Plan unhashable and remove completion from Plan observation/comparison.
6. Verify real-object regression controls, existing Goal contracts and canonical
   core/dev/visual profiles. Independent repository-role specialists review supplied
   source/evidence; coordinator alone writes/runs. Record exact receipts and known
   limitations, update Memory Bank, then deliver a reviewed PR.

No automatic motive-catalog activation, inventory/location simulation, investment
engine, score-heuristic retuning, later human-gate approval or AI readiness claim.
Use current dependency locks/deadlines. Preserve D035 and #740/#218/#741/#602.
