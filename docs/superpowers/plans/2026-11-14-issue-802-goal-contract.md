# Production Goal contract implementation plan

Approved in chat: “Proceed with that plan.” Base: merged #819, a3e39f3.

## Contract
Preserve Goal construction arguments. Normalize boolean condition buckets to copied
lists, accepting singular Conditions and tuples; validate entries and retain their
identities. All requirements in both buckets are reevaluated, without mutation.
Empty conjunctions remain true; target_effects are auxiliary, not completion.
Add iter_completion_conditions() and check_completion(state=None). Explicit native
State/mapping snapshots apply only to the actor; other targets use their own graph-first
state. Resolve existing entity/attribute aliases. Missing required fields, invalid
containers and unresolved targets raise errors rather than ordinary no-plan outcomes.
Keep exact Condition comparisons and #740 units.

## Tasks
- [x] Retain baseline failures; add real Character/Goal/Condition/GraphManager checks.
- [x] Implement normalized containers, shared extraction, graph-first completion and
  safe literal parsing for existing nested/inventory predicates.
- [x] Delegate production completion in planner, Plan evaluation and graph achievement.
- [x] Add bounded production feasibility (optional actions keyword): an empty successful
  plan is feasible; None is not. Preserve 1000 iterations / ten-action search bounds.
- [ ] Run core/dev/visual canonical checks and independent review; fix genuine defects.
- [ ] Publish a focused PR and evidence signed Codex GPT-6.1 Sol.

## Verification and delivery
Prove all/empty/malformed buckets, repeated checks, graph authority, explicit snapshots,
partner isolation, aliases, missing fields, inventory/boolean predicates, import orders,
already-satisfied [], reachable energy20→30, blocked/unreachable None with valid positive
controls, equal-effect costs5 vs1 in both orders, actual synchronized Action execution,
Plan false→true and bounded feasibility without mutation. Keep original #736 sources
and failed captures unchanged; authoritative state updates replace stale object-only
writes in maintained tests. No global-optimality claim, cost rebalance, full active-plan
execution/replanning, graph path evaluator repair, lightweight Goal replacement or
broader product readiness. Close #802 only after merge and merged-source verification;
leave #218/#741/#774/#803/#816 and economic/catalog scope intact.
