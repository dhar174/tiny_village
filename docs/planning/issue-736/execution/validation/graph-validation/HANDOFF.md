# Task 5 graph/planner handoff

Task 1 map prerequisite was committed at a7d187a. No production source was edited. No commits were created by this worker. Supported behavior remains separate from required failed integration contracts.

## Files

- tests/test_goap_graph_contracts.py: 5 unittest methods, each using FreshPythonTestCase, complete real Character/GraphManager/Goal/Action/Condition objects, headless basic memory, disabled LLM/checkpoints and finally controller.close(). Goal completion is explicitly not claimed by passing primitives.
- docs/planning/issue-736/execution/blocked/{planner_,graph_}*.py.txt: 15 standalone stdin-rerunnable reproductions; never collected, skipped, xfailed or counted as passing regression coverage.
- final-source-hashes.json: exact hashes for the final new module and 15 reproductions.

## Exact commands and results

Run from /workspace/tiny_village. Driver files reside in this handoff directory and use the existing capture_python helper without changing infrastructure.

    /workspace/.venvs/issue726-core/bin/python /workspace/scratch/issue-736/execution/graph-validation/run_graph.py core passing
    /workspace/.venvs/issue726-dev/bin/python /workspace/scratch/issue-736/execution/graph-validation/run_graph.py dev passing

The driver loads tests.test_goap_graph_contracts with unittest.defaultTestLoader, retains collection IDs in <profile>/passing-collection.json, and runs TextTestRunner(verbosity=2). Equivalent normal selector: python -m unittest -v tests.test_goap_graph_contracts. Both final runs: collected/run 5, failures 0, errors 0, skips 0, exit 0. See <profile>/passing-result.json and passing-stderr.log. Each child has a durable capture under <profile>/captures with diagnostic deadline 20 seconds and parent deadline 25 seconds.

One earlier core attempt failed all five methods because the fixture indexed controller.characters by name, although its public registration uses UUID keys. The fixture now selects actual values. Prior attempt's result/log/collection remain under core/fixture-invalid-* and its five captures remain intact. This was a fixture error, not a production failure, and it was not counted as a valid product run. The fixed core module and final dev module were each run once.

    /workspace/.venvs/issue726-core/bin/python /workspace/scratch/issue-736/execution/graph-validation/run_graph.py core blocked
    /workspace/.venvs/issue726-dev/bin/python /workspace/scratch/issue-736/execution/graph-validation/run_graph.py dev blocked

Each command executes the 15 named graph/planner text sources using capture_python. Both profiles: 15 exited 1, zero timeouts. These are 15 reproductions of overlapping contract gaps plus candidate misuse, not 15 distinct runtime defects. Exact command/source/hash/interpreter/capture directory for every case is in <profile>/blocked-results.json. Direct rerun example:

    /workspace/.venvs/issue726-core/bin/python < docs/planning/issue-736/execution/blocked/planner_reachable.py.txt

All blocked children used diagnostic/parent deadlines 20/25 seconds. Full stdout, stderr, exit status, event streams and run metadata remain in their capture directories. Both interpreters report Python 3.12.14.

## Classifications

- planner_goal_container: Condition itself correctly returns False at energy20/target30; Goal accepts the list container in construction but check_completion() raises AttributeError: list has no check_condition. Required completion false/true transition remains unfulfilled.
- planner_goal_signature: full Goal has bound signature (); planner calls it with state and swallows TypeError. Independently satisfied state energy30 yields False. This is a production integration blocker.
- planner_already_satisfied / planner_reachable: actual plan None for satisfied energy30/target30 and reachable energy20 + actual Action(+10)/target30. Required []/nonempty usable plan fail. Reproductions contain independent final-energy and real completion checks after any returned plan; those final assertions are not reached.
- planner_cost_selection: expensive-first equally effective real Actions cost5 versus1, initially unsatisfied energy20/target30; actual plan None. Cost selection and achieved final predicate remain unfulfilled because full Goal completion blocks planning. No fallback Goal was forced or adapter invented.
- planner_unreachable / planner_blocked_preconditions: actual None meets the negative no-plan expectation. Blocked action checks its real precondition once, does not change State or Character energy20, and direct blocked execution also passes in the maintained module. These text reproductions fail ONLY their satisfied positive completion controls. Do not label the negative no-plan behavior broken; attribute missing acceptance to the full Goal completion blocker.
- graph_state_provenance: actual registered Character has energy20; authoritative existing node attribute set to77. Wrapping spy confirms actual get_character_state(character.name) call. Planner returns Character-backed energy20, failing graph authority acceptance.
- graph_action_provenance: actual registered neighbor with actual GraphRecovery interaction and actual MultiDiGraph edge. Wrapping spy confirms actual get_possible_actions(character.name) call. Planner returns fallback Rest/Idle, failing graph action provenance. No name alias nodes or patched return values.
- graph_executed_world: production Action.execute returns True and Character energy20→30, but stored node remains20. Actual graph update receives UUID that is absent as node key. This is existing #799 territory, not repaired here.
- graph_character_identity: changing energy changes Character hash; the same registered object remains visible through iteration but graph.has_node(character) becomes False. Existing #799 stable identity defect reproduced, no reverse blocker introduced.
- planner_replan_boolean: fallback Plan.replan retains actual GraphRecovery but returns None. Required explicit boolean contract from #395 remains unfulfilled.
- planner_plan_evaluate: evaluating Plan with actual incomplete Goal raises the same list-container error. This is an additional affected API, not a separate root cause.
- planner_goal_feasibility: real Goal's boolean-key/list-value completion representation is incompatible with planner state-threshold handling; fails in State.get(False) with bool.split AttributeError. Required real-Goal feasibility is unresolved.
- planner_plan_utility: EXCLUDED CANDIDATE INTERFACE MISUSE. #395 calls evaluate_utility(Plan, character), expecting numeric utility. Current API expects a sequence of Actions and selects an Action. Candidate call raises TypeError: Plan not subscriptable. Retain as historical diagnostic; do not call it a production defect. Maintained new test passes correct API using a real Action sequence and selects HigherBenefit (utility9 versus0).

## Passing scope

Plan name/goal/queue identities and deterministic counter/dependency retention; blocked real Condition and Action execution; planner effect simulation caller/Character isolation and production Action single-effect application; numeric positive contextual cost and monotonic finite heuristic (60→20→0); correct real Action-sequence utility selection. No passing full Goal outcome, graph-fed plan, executed-world synchronization, gameplay or readiness claim.

## Existing-suite duplicates

Before the parent's no-repeat message arrived, this worker ran:

    /workspace/.venvs/issue726-dev/bin/python /workspace/scratch/issue-736/execution/graph-validation/run_existing.py

This uses two separately captured sources equivalent to -m unittest -v tests.test_goap_error_handling and -m pytest -q tests/test_goap_planner_integration_real.py. Results: 5/5 and 6/6, exit0, no skips. See dev/existing-results.json, two .source.txt files and associated captures. These duplicate parent checks; no need to rerun. The pytest suite remains stub-based and is not real graph outcome evidence. Existing job-role ERROR logs and uu deprecation warning remain visible in captures.

## Limits

No runtime fixes, changed identity rules, dependency changes, hidden retries, skip/xfail markers, fabricated planner logic, remote CI or publication. Task 5 acceptance remains partial pending explicit disposition of required failed contracts. Parent owns report/map/CI selector updates, commits, and follow-up issue classification.
