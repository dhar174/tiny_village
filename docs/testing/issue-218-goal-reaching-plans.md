# Executable StrategyManager plans — #218

Local implementation authorized by [D036](../decisions.md#d036--approve-218-executable-planning-implementation),
2026-11-15 America/New_York. Branch `codex/issue-218-goal-reaching-plans` starts at
latest `origin/main` `570cd89280b68491e90bb866ae93fd6b5428f07e`. A final fetch
confirmed main unchanged. These results were collected on the **uncommitted working patch**
and subsequently matched to committed source by all eight Python hashes.
See the publication record below; no merged implementation is claimed.

## Result and interface

StrategyManager returns a complete ordered sequence without selecting one utility
winner from it. Daily and GOAP-backed event paths share `plan_for_goal`; currently
blocked actions remain available when an earlier step can enable them.
`get_daily_actions` still supplies immediately applicable, utility-ranked choices
for existing controller/LLM consumers.

| Interface | Contract |
| --- | --- |
| `StrategyManager.plan_for_goal(character, goal, actions=None, current_state=None)` | Complete ordered `list[Action]`, `[]` already satisfied, or `None` within bounds |
| `GOAPPlanner.plan_actions(...)` | Same outcomes, preserving the supplied production Goal and checking actor/environment identity |
| `GOAPPlanner.validate_plan(character, goal, plan, current_state=None)` | Privately replay every precondition/effect and require the terminal goal |
| `StrategyManager.handle_plan_execution_failure(character, goal, failed_action, actions=None)` | Fresh state, one bounded search, excluding the exact supplied failed instance, same original goal |

Malformed required state/Goal contracts propagate errors. No utility action or
another goal's plan substitutes for failure. Empty sequences stay empty through
`update_strategy`. Existing LLM-selected immediate choices remain distinct from
validated GOAP plans; no model is initialized by goal planning.

The production full Goal/Condition completion contract remains separate from
UtilityGoal target metadata. No motive-generated inventory/catalog migration is
claimed. Production Goal actor ownership is checked in both planner entry points,
including initially satisfied requests.

## Search and execution parity

- Uniform-cost search stores cumulative effective cost independently, with exact
  structural `(state, depth)` dominance keys, stable ties and unchanged-successor
  skipping. Defaults remain ten actions and1000 expansions. There is no plan-result
  caching; the legacy `plans` mapping remains unused.
- Scalar transitions share `_AttributeEffect` parsing, clamps, native aliases and
  pure target resolution with `Action.execute`. Initiator/implicit-target identity
  and per-descriptor deduplication agree with actual execution.
- Admission rejects no-op wrappers, custom execution/preconditions, callbacks,
  unmodelled dynamic methods, multi-entity writes and incompatible numeric
  inventory/location effects. Missing targets do not create simulated actor gains.
- Private snapshots copy supported nested containers and omit opaque domain
  references. Required unsupported reads fail rather than traversing/copying a live
  Character, graph, inventory or lock. Planning does not rebind actions or mutate
  source state, siblings, Conditions or world objects.
- Needs/targets use native points, happiness fractions and money credits. The
  event builder fixes generic70/65, social happiness`.75`, positive weather`.80`
  plus fractional impact and outdoors happiness+.10. Metadata comparisons use1e-9;
  full production Condition comparisons retain their existing semantics.
- A failed action may retain earlier synchronized writes. Replanning therefore
  reads fresh graph state rather than assuming whole-action rollback.

### Default-state search correction

An additional real default-state probe found `None` after1000 expansions at
energy20/satisfaction0 although an eight-step daily plan exists. The retained RED
worker report has1failure/0errors. Native immediate utility rewarded unrelated
food/work branches at the minimum search cost.

The planning/actions specialist reviewed an explicit **1.0 non-progress effort
cost** for UtilityGoal metadata: canonical signed remaining distance is normalized
through existing native-unit conversion. A transition that does not decrease this
distance incurs the cost. All candidates remain eligible, including prerequisites
that do not immediately progress a money goal. Production full Goal costs remain
unchanged. This is a goal-dependent effective-cost policy, not a claim of minimizing
raw `Action.cost` or finding every possible plan within the fixed bounds.

The covering RED-to-GREEN regression executes the default plan. A metadata
Rest-to-Work prerequisite regression separately prevents converting the cost into
pruning. Depth/cost selection controls still execute to the final goal.

## Actual execution evidence

Both traces invoke the real controller single-action pipeline, check each
Character/graph value, and finish with a real production Goal completion check.

| Scenario | Executed sequence | Authoritative outcome |
| --- | --- | --- |
| Earn credits, energy20/money5 | Rest, Work as Baker | Rest: energy35/money5; Work: energy5/money25; complete |
| Default wellbeing, energy20/satisfaction0 | Rest four times, Leisure Activity four times | energy80/satisfaction80; complete in eight actions |
| Partial synchronization failure | Energy write survives; money write fails; two alternative recoveries | Fresh replan starts at energy30 and finishes at60; original Goal complete |

The full per-step traces, commands and stderr diagnostics are retained in
[the local receipt](evidence/issue-218-local-results.json), with raw artifacts under
`/workspace/scratch/issue-218-traces/`. Run `python -B demo_goap_integration.py` in
the supported core environment for the real two-step demonstration.

## Verification and review

| Final profile | Executed tests | CLI probes | Test failures/errors/skips |
| --- | ---: | ---: | --- |
| Core | 186 | 3 | 0/0/0 |
| Dev | 20 | 0 | 0/0/0 |
| Visual | 69 | 0 | 0/0/0 |

All final runner and supervised-batch process exits were0; every structured batch
reported PASS. Focused acceptance ran19 methods:12 integration plus7 real event/
compatibility cases, including preservation of three prior state-conversion
scenarios. Existing Goal, scale, action-target and execution regressions remain in
core. The older no-target Half fixture now declares its intended initiator target,
with a separate negative test proving omitted targets make no real/projected gain.

Canonical commands, each in its corresponding locked Python3.12/Linux environment:

```sh
python -B scripts/run_ci_checks.py --profile core --artifacts /workspace/scratch/issue-218-final-core
python -B scripts/run_ci_checks.py --profile dev --artifacts /workspace/scratch/issue-218-final-dev
python -B scripts/run_ci_checks.py --profile visual --artifacts /workspace/scratch/issue-218-final-visual
```

Exact interpreter paths, lock digests, source hashes, reports and counts are in the
receipt. Earlier pre-amendment profile results (core182+3/dev20/visual69) are retained
separately and do not replace the final changed-source verification.

Repository-defined architecture-scout, planning-actions, world-state, test-ci,
llm-contracts and reviewer roles were adapted to Codex with supplied source and
execution evidence. They performed no commands, network calls or edits; the
coordinator alone wrote and executed. Native Zcode installation/discovery was not
exercised. No production controller edit triggered ui-simulation.

Independent source review found and resolved three defects: direct planner actor
ownership, malformed descriptor admission and incompatible scalar effects aborting
unrelated search. The corrected source and cost amendment were reviewed with no
remaining blocker. Final test-ci/reviewer evidence review reconciled the supplied
counts/traces. Review did not independently inspect disk or execute tests.

## Retained limits

**Stderr is not clean.** Real-world construction still logs the pre-existing job
parser errors (117 roles counted,116 parsed; missing-role diagnostics). These also
appear in earlier #802 captures; `tiny_jobs.py` and `job_roles.json` are unchanged.
The receipt retains their exact lines. Passing test summaries do not relabel those
runtime diagnostics as fixed. Intentional negative/fault-injection cases also retain
expected failure logs.

#741 still owns active-plan cursors, autonomous execution across ticks, world-change
invalidation and live replacement scheduling. #821 retains remaining inventory/
Goal-catalog repairs. The legacy job-offer selection API is not part of the new
validated planning guarantee. Live inference, native Windows/GPU behavior, phase
approval and broader gameplay readiness remain unverified. The original verification preceded publication; no merge or issue closure
has been performed.

## Publication

The user subsequently authorized commit/push/PR publication. Implementation commit
`52e39cb5b4095966e6cd667872e8670d15b38862` preserves every tested Python source hash.
Ready-for-review [PR #823](https://github.com/dhar174/tiny_village/pull/823) targets `main` from
`codex/issue-218-goal-reaching-plans`; remote branch SHA and PR metadata were read
back and the PR attached to the chat. This follow-up updates documentation only.
Hosted checks were in progress at initial readback; their success is not inferred
from local verification. #741 and all recorded limitations remain separate.
