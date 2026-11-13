# TASK029 — Reconcile need and resource units (#740)

**Updated** 2026-11-13

**Status** Implemented on `codex/740-runtime-units` at `fc5286e` (parent `56d36c2` = main);
worktree `C:\Users\darf3\Documents\tiny_village-740`. Issue #740 remains open/Todo/Ready.
Before-merge gate: hosted Linux CI (core+dev+visual on the PR head). Push/PR is the maintainer's call.

The approved plan reconciles the runtime's mixed numeric scales. Native 0-100 units for
energy, hunger_level, health_status, mental_health, social_wellbeing, job_performance, and
current_satisfaction across Character, graph state, action conditions/effects, and GOAP
search. Happiness stays a 0-1 fraction; wealth_money stays credits. Conversion happens only
at documented boundaries: ÷100 into normalized utility scoring (state values and effective
deltas; happiness/credits pass through) and the named village-economy-v1 fixture conversion
(x10 on entry, /10 on reporting) in actions.py.

Scalar effects clamp finite energy/hunger/health results to 0-100 (signed deltas, `add:`,
numeric `set:`); invalid descriptors still reject before writes. Registered Character event
writes for the three core needs route through the shared synchronized updater
(actions.apply_attribute_change), restoring graph coherence for the D035 stale-graph gap on
those fields; mapper bounds/defaults are native (0,100) for needs rows. Strategy producers,
gates, and absolute targets use native values; snapshots are graph-first with the standalone
path preserved; GOAP preconditions evaluate against the simulated state, effect projection
shares the scalar calculation, and targets are sign-directional (positive = minimum,
negative = maximum). Prompts render /100 labels and credits units with translated thresholds.

Independent review (tiny-village-reviewer): no blocking findings; six non-blocking findings,
five addressed in the diff (docstring, alias-key emission at the cost boundary, mock
mental default, dispatcher-gate documentation, receipt precision); the satisfaction reader
asymmetry is recorded for the #816 alias audit.

Verification receipts: `.test-artifacts/windows-local/20261113-740/` and
`.test-artifacts/740-review/` (gitignored). Windows-local runs are non-canonical
(run_ci_checks refuses Windows); core-profile modules all green at HEAD; pre-existing red
modules (test_tiny_graph_manager 32 DummyGoal errors, event-planning 4F+1E,
decision-prompt ImportError, prompt-builder 1F+6E, test_ci_checks Windows env issue) are
unchanged and CI-invisible. The strategic-llm module carries one documented delta
(novelty routing-reason semantics on its in-file mock). Behavior changes carried: Sleep
gate effectively 3->30 native, Work newly gated at energy >= 30 (Work-as-Unemployed case
bug deliberately untouched), get_daily_actions ordering (Sleep > Work > NoOp > Eat),
mock defaults x10 across test files.

Deferred follow-ups: #802 broader Goal completion, #741 replanning, #816 direct-writer
audit + alias asymmetry + satisfaction reader asymmetry; calc_distance_cost (1-energy)
heuristic; HUNGER_OFFSET undefined in the UI hunger display; dead tests/test_cost_integration.py;
Goal import-order fragility; output-interpreter default effect magnitudes; other demos with
0-10 seeds; test_enhanced_decision_prompt deeper rot; goal Priority Score /10 + motive /10
scales.
