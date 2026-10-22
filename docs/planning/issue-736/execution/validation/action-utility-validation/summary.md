# Task 2 validation

From /workspace/tiny_village:

- /workspace/.venvs/issue726-core/bin/python -m unittest -v tests.test_action_contracts tests.test_utility_contracts: collected8, passed8, exit0, 1.525s. Zero failures/errors/skips/timeouts.
- /workspace/.venvs/issue726-dev/bin/python -m unittest -v tests.test_action_contracts tests.test_utility_contracts: collected8, passed8, exit0, 1.573s. Zero failures/errors/skips/timeouts.
- /workspace/.venvs/issue726-core/bin/python -m unittest -v tests.test_actions tests.test_tiny_utility_functions tests.test_utility_with_real_classes tests.test_issue_425_fix: collected40, passed30, failure1/errors9, exit1, 0.005s. Zero skips/timeouts.

TINY_TEST_ARTIFACT_DIR points to this directory's respective core-captures/dev-captures/existing-core-captures subdirectory. Raw stdout/stderr, command/collection JSON and original child sources are retained. New tests were run once per profile, existing affected suites once in core. No product code changed, no full-suite execution, no test skips/xfails.

New observed traces in both profiles: energy5+2=7; real Character energy12→true and5→false; real Action golden29.1; four edge utility scores7.5/13.5/10.5/13.494; single repeated eating12.9, no-simulation plan25.8, simulated plan24.0; hunger/alignment outcomes are retained in each child stdout. Utility Goal provenance is tiny_utility_functions.Goal (lightweight production fallback), not full tiny_characters.Goal.

Unsuccessful existing methods (all tests.test_tiny_utility_functions.TestTinyUtilityFunctions):

- test_action_priority_in_goal_contexts: Goal(priority=...) TypeError.
- test_calculate_action_utility_goal_effect_opposite_direction: same TypeError.
- test_calculate_action_utility_with_goal_match: same TypeError.
- test_calculate_action_utility_with_goal_no_match_effect: same TypeError.
- test_enhanced_mock_action_utility_calculation_compatibility: same TypeError.
- test_enhanced_mock_goal_with_urgency: same TypeError.
- test_plan_utility_with_goal: same TypeError.
- test_realistic_action_scenario_with_enhanced_mocks: same TypeError.
- test_utility_evaluator_with_urgent_goal: same TypeError.
- test_mock_action_preserves_explicit_falsy_action_ids: generated id != explicit0.

The same40-selector command was run once with the same core interpreter from an exact git archive of origin/main11b6e93595a70ce9f24cfb239a31ac873847ff56 at /workspace/scratch/issue-736/execution/baseline-main. It also reports40 cases,30 passed,1 failure,9 errors,zero skips/timeouts. All10 unsuccessful test IDs and failure/error classifications match the working-tree run exactly. Retained baseline-main.stdout.log, baseline-main.stderr.log and baseline-main.run.json provide exact revision/command/source hashes/comparison. This establishes the failures on baseline independently of the new contracts.

Existing Action5/real-class utility5/issue4254 all pass; unchanged tiny-utility suite passes16 of26. No baseline/new passing contract reruns were performed. Parent owns final disposition/report.
