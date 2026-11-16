# #218 — executable StrategyManager plans

Approved by dhar174 on 2026-11-15 America/New_York: "Please implement the plan."
The approval follows the detailed chat plan, the explicit choice to complete #218
without #741 tick integration, the request for named repository subagents, and the
requirement to start from latest main. See D036 in ../../decisions.md.

## Baseline and ownership

Fetch origin and branch from its current main, verifying exact identity before
editing. This execution starts at 570cd89280b68491e90bb866ae93fd6b5428f07e on
codex/issue-218-goal-reaching-plans. Refresh main before final handoff. Preserve
merged #220/#740/#802/#822 contracts. Review #689 selectively; do not apply its
older planner implementation wholesale.

The coordinator is the sole writer and command runner under
../../agents/team-coordination.md. Adapt the read-only repository definitions:
1. .zcode/agents/tiny-village-architecture-scout.md: baseline/ownership handoff.
2. tiny-village-planning-actions, tiny-village-world-state, tiny-village-test-ci:
   independent contract/state/test investigations after the scout, concurrently.
3. tiny-village-llm-contracts: immediate-candidate and fallback compatibility.
4. tiny-village-ui-simulation only if production controller code changes.
5. tiny-village-reviewer: independent actual-patch and evidence review; no fixes.
All definitions are under .zcode/agents at the repository root. No specialist
writes, commands, network calls, model I/O, or nested delegation. Supply exact
source/evidence when compatible read-only tools are unavailable; distinguish this
adaptation from native Zcode discovery or installation.

## Implementation contract

- Add StrategyManager.plan_for_goal(character, goal, actions=None,
  current_state=None), preserving full production Goal identity and UtilityGoal
  metadata compatibility. Return ordered list, [] satisfied, or None within
  bounds. Required malformed Goal/state errors propagate.
- Route daily and GOAP event planning through one implementation. Remove
  evaluate_utility sequence collapse and fallback to a different goal. Preserve
  existing optional LLM immediate-decision compatibility.
- Separate complete candidates from currently applicable ranked get_daily_actions.
  Include future-applicable Work and enabling daily actions in event planning;
  remove name-only and arbitrary top-N reachability restrictions.
- Reconcile native 0-100 needs, fractional happiness, credits and aliases in one
  event-goal builder, including generic70/65 and outdoors happiness+.10.
- Share pure target resolution with real Action execution. Model supported scalar
  actor effects, concrete state-aware Conditions, shared clamps and aliases.
  Exclude unmodelled custom execution, callbacks and multi-entity writes explicitly.
  Never execute actions or mutate live bindings/state during planning.
- Copy supported data recursively and use structural exact state/depth keys.
  Uniform-cost search retains depth10 and1000-expansion defaults, finite positive
  costs, no-op skipping and replay validation of preconditions plus terminal goal.
- Keep result-cache capacity zero. Failure handoff reads fresh state and performs
  one bounded search excluding the exact supplied failed instance. #741 owns
  live scheduling and invocation.

## Verification and boundaries

Use actual registered Characters, full Goals, Actions and graph state. Execute
returned plans via the real controller single-action path and independently check
terminal values, graph/object parity, actor identity and exactly-once bookkeeping.
Cover prerequisite chains, default daily recovery, event units, satisfied/no-plan
outcomes, malformed contracts, isolation, target parity, limits, cost/depth routes,
partial execution failures, alternatives and optional-provider boundaries.

Register maintained coverage in core. Run focused checks, then core/dev/visual
through python -B scripts/run_ci_checks.py --profile PROFILE --artifacts FRESH_PATH
under the appropriate locked Python3.12/Linux environments. Inspect exits, counts,
logs and structured receipts. Retain failures and verify resulting source hashes.

#741 autonomous ticks, #821 remaining inventory/catalog behavior, live inference,
native Windows/GPU support, later gates and unrelated job-role parsing defects
are separate. No merge or issue closure is authorized by passing local checks.

## Verified implementation refinement

A retained default-state failure required an explicit1.0 non-progress effort cost
for UtilityGoal metadata. The planning/actions specialist and independent reviewer
reviewed this goal-dependent cost policy; prerequisites remain eligible, production
Goal costs and original search bounds are unchanged. This does not claim raw-cost
optimality or bounded completeness. See the implementation/evidence report at
../../testing/issue-218-goal-reaching-plans.md for the actual RED/GREEN outcome,
final counts, traces, review and retained stderr limitations.
