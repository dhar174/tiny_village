# Issue #736 Functional-Test Consolidation Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task after approval. This plan does not require delegation.

**Goal:** Reconcile nine stale test PRs through traceable, behavior-focused coverage in maintained locations, retaining genuine runtime failures and superseding candidates only when their assertions are accounted for.

**Architecture:** Port assertions, not branches. Use production State, Action, Condition, memory, Character and GraphManager objects at the boundary each test claims to exercise. Keep state-level planning, graph integration and world execution evidence distinct. No production repair is included.

**Tech stack:** Existing Python 3.12 locks, unittest/pytest, NetworkX MultiDiGraph, the durable subprocess capture helper and existing CI runner.

**Spec:** [#736](https://github.com/dhar174/tiny_village/issues/736), [D029](../../decisions.md#d029--review-736-functional-test-pr-dispositions-step-by-step), and the [pinned walkthrough](../../planning/issue-736/README.md). D029 is the single decision record; all nine choices are selected. **This implementation plan awaits review.**

## Global constraints

- Baseline: main `11b6e93595a70ce9f24cfb239a31ac873847ff56`, containing merged #703 and #798. Refresh before execution and reassess changed source or PR heads.
- Preserve the current root layout. Do not restore `tests/test_completed_implementations.py` or wholesale demonstration rewrites.
- No runtime-module, dependency-lock, model-download, UI, gameplay, graph-identity or AI-readiness changes. AI remains a final-product requirement under D015.
- No copied planner, precondition, effect or happiness implementation in fixtures. Do not replace a missing required result with a warning, exception-as-success, skip or expected failure.
- New cold-process cases use a 20-second diagnostic deadline and 25-second parent deadline. Preserve existing deadlines and failures; retries never erase earlier results.
- #799 remains after human gate #738. Do not add a reverse blocker or silently move its runtime repairs into #736. #737 owns the subsequent coordination/Project handoff.
- No merge or phase-gate approval is implied. Completion and closures require the evidence rules below.

## Review focus

1. **Already satisfied versus unreachable goals:** require an empty successful plan only for the former; assert the documented no-plan result for the latter.
2. **Silent fallbacks:** graph-state/action tests must prove graph-derived inputs were used; a fallback Rest action or character-only state cannot satisfy graph integration acceptance.
3. **Missing relationships:** no partner, no family and empty relationships are legitimate inputs. Applicable motive/social/romantic/family contributions must also change the real result as required.
4. **Swallowed errors:** a returned happiness number, `None` plan or successful banner cannot substitute for the expected contribution or achieved goal.
5. **Duplicate and stale evidence:** map assertions at pinned heads, verify actual test collection, and recheck candidate heads before closures. A renamed file or old CI success is insufficient.

## Research and evidence limits

Retained probes ran five original #420 methods and one selected #417 method successfully against the baseline. A separate bounded diagnostic rejected #393's reduced argument lists against current signatures and exposed #397's negative passed count for failing subtests. It did not run whole candidate suites. No additional product tests ran while writing this plan.

Current `tests/test_goap_planner_integration_real.py` uses CharacterStub, GraphManagerStub and an EnergyAction overriding execution. It supplies useful existing coverage, but its name does not establish real graph integration.

Source inspection identifies additional risks to reproduce during execution: `GOAPPlanner._goal_satisfied` calls `goal.check_completion(state)`, while the production Character Goal accepts no state argument; its constructor iterates lists of completion conditions while `check_completion` calls each dictionary value directly. `calculate_happiness` also catches errors in contribution paths and calls motive getters that must be checked against real PersonalMotives. These are source findings, not newly executed failure claims.

## File responsibilities

| File | Planned responsibility |
| --- | --- |
| `docs/planning/issue-736/assertion-map.md` | Candidate SHA, original method/assertion, maintained destination or exclusion, exact validation, and unresolved blocker for every relevant assertion. |
| `tests/test_action_contracts.py` (new) | Only unique State/Action/Condition assertions missing from existing maintained tests. Omit this file if mapping proves no gap. |
| `tests/test_utility_contracts.py` (new) | Strict real-Action utility and plan edge cases without changing existing mock-based utility suites. |
| `tests/test_memory_runtime.py` | Minimal public API coverage missing from existing lazy/basic-memory tests. |
| `tests/test_character_happiness_contracts.py` (new) | Real Character happiness outcomes and legitimate absent-relationship cases. |
| `tests/test_goap_graph_contracts.py` (new) | Strict planner and graph contracts, clearly labeling state-level versus graph/world integration. |
| `test_completed_implementations.py` | Replace only superseded happiness source checks once their required behavior is covered; retain building, pause and unrelated checks. |
| `scripts/run_ci_checks.py` | Add validated consolidated selectors to existing core batches; preserve runner behavior, profiles and deadlines. |
| `docs/testing/issue-736-functional-test-consolidation.md` | Commands, revisions, counts, traces, exclusions, blockers and closure receipts. |
| `docs/planning/issue-736/evidence/` | Existing review evidence remains intact; retain new source snapshots/captures in a separately identified execution set. |
| `docs/decisions.md`, TASK023 and Memory Bank summaries | Approval/status and resumable handoff; decisions remain solely in D029. |

## Ordered execution

### Task 1: Build the assertion-level map

**Depends on:** plan approval. **Produces:** the reviewed port/exclusion inventory used by every later task.

- [ ] Refresh main and all nine PR heads, states and relevant discussions. Record differences from the pinned heads; reassess changed assertions before porting them. Preserve existing local documentation commits.
- [ ] Retain the exact original changed test sources needed for #393/#395/#391/#406 and remaining #417/#408 comparisons as text with hashes; do not make old sources discoverable tests.
- [ ] Map each candidate assertion to an existing maintained method, a new test below, a justified non-product demonstration exclusion, or a named unresolved requirement. Include exact source references and distinguish inherited tests from new PR deltas.
- [ ] Deduplicate against `tests/test_actions.py`, `tests/test_action_definitions_resolution.py`, `tests/test_tiny_utility_functions.py`, `tests/test_utility_with_real_classes.py`, `tests/test_issue_425_fix.py`, `tests/test_memory_runtime.py`, `tests/test_ci_checks.py` and current planner suites. A script's assertions count as maintained execution only when the chosen command actually invokes them.
- [ ] Commit the map and provenance before test ports.

### Task 2: Consolidate Action/State and utility assertions

**Depends on:** Task 1. **Consumes:** `State(data)`, `Action(...).apply_effects(state)`, `Condition(...).check_condition()`, `calculate_action_utility(state, action, current_goal=...)`, and `calculate_plan_utility(state, plan, current_goal=..., simulate_effects=...)`.

- [ ] For #420, preserve unique dict-backed State read/write/default behavior, Action application of energy `5 + 2 = 7`, empty preconditions, and genuine Condition pass/fail outcomes. Use real Character targets for any target-dependent boundary; no StatefulTarget replacement is evidence of full Character integration.
- [ ] For #417, directly call required utility functions without `hasattr` guards. Use its four edge states `(hunger, energy, happiness)`: `(0,1,0.5)`, `(1,0,0)`, `(0.5,0.5,1)`, `(0.999,0.001,0.001)`. Assert finite numeric action/plan results and unchanged caller state where the API promises simulation isolation.
- [ ] Retain the unique golden real-Action calculation `12.6 + 17.5 - 1.0 = 29.1` if current suites lack equivalent coverage. Label the lightweight production utility Goal as such; it is not proof of the full Character Goal contract.
- [ ] Verify existing no-simulation versus simulated plan coverage before adding cases. Exclude generic arithmetic, JSON/file examples, made-up missing names and copied UtilityTestAction logic.
- [ ] Apply #408's map against these tests and existing coverage, retaining any additional unique product assertion discovered.
- [ ] Run the new unittest modules and the affected existing Action/utility suites. Record actual collection and failures; do not repair utility production code here. Commit validated ports separately.

### Task 3: Preserve useful memory API coverage

**Depends on:** Task 1; may follow Task 2. **Consumes:** real `tiny_memories` exports and the existing TestBasicMemory/TestLazyMemory contracts.

- [ ] Map #406's three added MemoryManager/SpecificMemory/GeneralMemory export checks separately from its inherited memory tests. Preserve direct public imports/class availability without initializing models; label this API-surface evidence accurately.
- [ ] Reuse existing real record/recall, timestamps/importance, 50-record eviction, independent stores, lazy initialization and explicit capability-failure tests. Add only a demonstrably missing API assertion to `tests/test_memory_runtime.py`.
- [ ] Run that module and `tests.test_lightweight_runtime.TestMinimalRunner` in the locked core environment. Required imports must fail visibly if broken; no dependency-based skip.
- [ ] Preserve legacy semantic memory suites and their unresolved requirements. Full SpecificMemory/GeneralMemory construction or live pretrained inference is outside this API port; do not call those features verified from class imports.
- [ ] Commit the minimal port and map updates.

### Task 4: Replace happiness keyword evidence with behavior

**Depends on:** Task 1. **Consumes:** real `Character.calculate_happiness()`, real PersonalMotives and basic-memory Character construction.

- [ ] Construct complete real Characters using the existing headless controller configuration: `headless=True`, `memory.mode='basic'`, `characters.use_llm=False`, automatic checkpoints disabled. Use fresh child processes and `finally: controller.close()`; do not stop constructors early.
- [ ] Add paired deterministic cases with all unrelated Character inputs fixed: supported motive scores `2` versus `8`; social relationship quality `0.2` versus `0.8`; a partner with relationship quality `0.8` versus the same relationship without a partner; family membership with quality `0.8` versus the same relationship without family membership. Assert a finite result and the required contribution direction, not implementation keyword presence or a copy of the formula.
- [ ] Add empty relationships, no partner and no family cases. They must calculate a finite baseline without requiring irrelevant relationships to exist. Use non-saturated valid values so rounding/saturation does not disguise a contribution; retain the values in the test.
- [ ] If real motive representation or swallowed exceptions prevent a required contribution, retain the failing case and trace as a blocker. Do not invent motive getters, bypass a real object with a fake, lower a feature-count threshold, or fix production code under this task.
- [ ] Replace the root `test_happiness_calculation` source scan only after all its required features are covered by passing behavior tests or a separately reviewed disposition of a demonstrated blocker. Keep unrelated root tests intact.
- [ ] Run the new module and relevant existing root checks, recording each case. Commit the validated behavior coverage; retain blocked cases under Task 6's rules.

### Task 5: Consolidate strict planner and real graph contracts

**Depends on:** Tasks 1–2. **Consumes:** `GOAPPlanner(graph)`, `plan_actions(character, goal, current_state=None, actions=None)`, `get_current_world_state(character)`, `get_available_actions(character)`, real GraphManager/Character/Action/Condition/Goal objects.

- [ ] From #395, retain unique scenarios for an already satisfied goal, a reachable goal, an unreachable goal and blocked preconditions. Require `[]` for an initially satisfied goal, a nonempty usable plan for a reachable goal, and the documented no-plan result for an unreachable one. An unexpected exception or `None` for a reachable goal fails.
- [ ] For reachable cases, assert an independent final predicate such as `energy >= target_energy` before also checking the production completion API. Apply returned actions through production effect/execution paths; do not count `plan is not None` or a positive-effect action alone as success.
- [ ] From #393, retain dynamic graph state/action retrieval and useful cost/utility selection scenarios. For cost comparisons use two actions reaching the same **initially unsatisfied** energy target; place the expensive alternative first and verify both selected cost and achieved predicate. Do not reuse a hunger-direction example that is already satisfied under current threshold semantics.
- [ ] Test full Character Goal construction/completion and planner integration using their current public interfaces. Reproduce signature/container incompatibilities without monkeypatching Goal.check_completion; a local PlannerGoal substitute cannot satisfy that acceptance.
- [ ] Use an actual GraphManager/MultiDiGraph with real registered Characters for graph cases. Observe calls with wrapping spies only; do not replace return values. Check authoritative graph values and returned action provenance so silent character-state/fallback-action paths cannot pass as graph retrieval.
- [ ] Keep supported explicit-state planner tests distinct from graph-fed and executed-world tests. An explicit-state pass does not clear a failing graph lookup or #799 action-target/identity defect.
- [ ] Preserve current `_calculate_action_cost(action, state, character, goal)` and `_estimate_cost_to_goal(state, goal, character)` interfaces. Exclude #393's simulated planner/graph and production signature reductions.
- [ ] Run new cases plus existing GOAP error-handling and planner integration suites. Retain failures before assessing their relation to #799/#220/#602 or a new scoped follow-up. No runtime repair, rebase or PR closure. Commit validated tests and blocker evidence separately.

### Task 6: Validate, integrate selectors and report gaps honestly

**Depends on:** Tasks 2–5. **Produces:** reviewable test consolidation with a complete evidence map, or an explicitly partial result awaiting blocker disposition.

- [ ] Collect new unittest tests before execution and retain exact method counts. Use existing pytest collection for existing pytest suites; do not assume a green run executed standalone `main()` assertions.
- [ ] Run each new module in locked core and development environments, with durable captures. Reuse `FreshPythonTestCase`/`capture_python` for cold children and preserve original failing runs. Add no new capture infrastructure.
- [ ] Run affected existing Action/utility/planner/root tests, plus current `tests.test_ci_checks` for #397 and maintained full-output error evidence for #404. Scope broader runtime checks to memory/headless behavior touched by fixtures; preserve unrelated legacy failures as such.
- [ ] Add passing new selectors to the existing core component/runtime batches. Keep workflow jobs, profile locks and runner semantics unchanged; run `scripts/run_ci_checks.py --profile core --artifacts <absolute-evidence-directory>` with the core interpreter and the dev profile with its own interpreter. This uses local resources and does not require paid remote CI.
- [ ] For any required contract that fails, retain a directly rerunnable failing source as a text artifact with its command, expectation, actual result, profile and issue mapping. Do not disguise it as a passing test, quietly exclude it from the map or add xfail/skip merely to merge. New discovered requirements need an explicit scoped follow-up; no automatic runtime fix.
- [ ] A blocked required port prevents a blanket “#736 complete” claim. Present the concrete gap and proposed follow-up for human review before closing its candidate or #736. Do not create a dependency cycle with #738/#799.
- [ ] Publish a report separating passed cases, expected diagnostic injections, existing failures, blocked contracts, skips, timeouts and unexecuted work. Include exact revisions, commands, environments, source hashes and input/final-state traces. Self-review each #736 criterion against that map.

Planned commands run from the repository root, after the named files exist. Use
the current locked interpreters `/workspace/.venvs/issue726-core/bin/python`,
`/workspace/.venvs/issue726-dev/bin/python` and, for renderer-dependent root
coverage, `/workspace/.venvs/issue726-visual/bin/python`. Use existing SDL dummy
drivers for the visual environment. Capture stdout, stderr, exit codes and
collected counts; these commands are not claims of results already obtained.

| Coverage | Command arguments to the appropriate Python interpreter |
| --- | --- |
| New behavior tests, core and dev | `-m unittest -v tests.test_action_contracts tests.test_utility_contracts tests.test_character_happiness_contracts tests.test_goap_graph_contracts` (omit the action module only if Task 1 documents zero unique gaps) |
| Existing Action/utility | `-m unittest -v tests.test_actions tests.test_tiny_utility_functions tests.test_utility_with_real_classes tests.test_issue_425_fix` |
| Memory, startup and error behavior | `-m unittest -v tests.test_memory_runtime tests.test_lightweight_runtime.TestMinimalRunner tests.test_goap_error_handling` |
| Existing pytest planner integration, dev | `-m pytest -q tests/test_goap_planner_integration_real.py` |
| Maintained runner counting/error evidence, dev | `-m unittest -v tests.test_ci_checks` |
| Root regression baseline, visual | `-m unittest -v test_completed_implementations` |
| Integrated local profile, core/dev respectively | `scripts/run_ci_checks.py --profile core --artifacts /workspace/scratch/issue-736/validation/core` and `scripts/run_ci_checks.py --profile dev --artifacts /workspace/scratch/issue-736/validation/dev` |

Run each verification target once after its relevant change; rerun only for a
subsequent change or diagnosis, retaining both results. New contract modules
must collect nonzero tests and have zero failures, errors or skips to count as
passing coverage. Existing-suite failures remain separately reported and cannot
be labeled new regressions without baseline evidence. Text-only blocked
reproductions remain explicitly unfulfilled coverage.

### Task 7: Review, publish and supersede with verified crosslinks

**Depends on:** Task 6 and acceptance of any unresolved required-port disposition.

- [ ] Open one focused consolidation PR against current main, carrying separate Action/utility, memory, happiness, planner/graph and evidence commits where those changes exist. Link #736 and each candidate without automatic issue-closing keywords until acceptance is complete. Attach the created PR to this task.
- [ ] Request human review of assertion quality, actual graph/goal outcomes, exclusions, blocked-contract handling and original contributor attribution. No UI/aesthetic implementation is included; earlier approved design reviews remain intact.
- [ ] For #404/#397, link maintained traceback/counting coverage and retained diagnostics. For all other candidates, link exact replacement tests or specifically reviewed exclusions/follow-ups, with validation and contribution credit.
- [ ] After the replacement is merged and current heads are rechecked, apply each selected superseding closure only when its map is complete and required assertions pass or a specific blocker disposition has been reviewed. Leave changed/unaccounted-for candidates open. Closing an old PR does not close its linked runtime issues.
- [ ] Verify closure states/comments and replacement links by readback. Update #736's acceptance evidence without replacing its original requirements; close it only when all three criteria are accounted for. Keep #737/#738 and unresolved runtime issues open, and hand merge order/ownership/Project synchronization to #737.

## Acceptance checklist

- [ ] All nine candidates and merged #349 are mapped at retained SHAs to current locations, with the D029 choice and exact validation references.
- [ ] Unique real behavior, actual goal outcomes and graph assertions are preserved; no copied logic or fabricated green integration result.
- [ ] Memory API presence is separated from semantic-memory readiness; state-level planning from real graph/world execution.
- [ ] Every required failure remains visible with a rerunnable artifact and accountable follow-up; no unreviewed acceptance weakening.
- [ ] New passing regressions are collected by supported local CI profiles, with actual counts and retained failures/timeouts.
- [ ] Replacement publication and candidate closures occur in dependency order with verified crosslinks and original issues preserved.

**Handoff:** Review this plan before implementation. The nine dispositions authorize the intended direction; they do not themselves establish that the ports pass or that any candidate is ready to close.
