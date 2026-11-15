# Utility Goal contract and #774 verification

Date: 2026-11-15 America/New_York. Base main46461b0 (merged #820).

## Supported roles and compatibility

`tiny_utility_functions.UtilityGoal` is production data-only metadata. The existing
`tiny_utility_functions.Goal` export is its compatibility alias. Importing Character,
utility or strategy first, or reloading utility after imports settle, no longer selects
the full Character Goal constructor. Python still creates a new class object on module
reload; compatibility means the same role/interface/results, not identity with stale
references held by other modules.

Constructor: `UtilityGoal(name=None, target_effects=None, priority=0.5, score=None,
**kwargs)`. A missing or empty name defaults to UnnamedGoal; score defaults to priority only when None;
urgency defaults to priority when omitted/None. Explicit priority/score0 and urgency0
remain explicit values. Optional deadline defaults to None, description to
`Goal: <name>`. Omitted or explicit `attributes=None` creates a fresh dictionary;
a supplied mapping is retained, including an empty dictionary. Existing metadata kwargs
remain accepted. No new completion API, graph ownership or invented successful completion
predicate is supplied by metadata.

`tiny_characters.Goal` remains the full domain Goal with its unchanged constructor,
Condition containers and graph-first completion contract. Callers needing that role
must import it explicitly. The existing native action/plan utility adapters continue
to accept full Goals and convert native absolute targets to remaining normalized
changes. A full Goal without a utility priority retains the existing0.5 default;
its domain score is not silently rescaled or aliased to priority.

Metadata has no automatic unit guessing: raw utility inputs/effects/goals use the
existing normalized delta scoring contract; native planner producers use absolute
native target specs and the named adapters. #740 native units, scoring constants,
urgency multiplier, plan projection, dependencies and search limits are unchanged.
The compatibility alias keeps existing StrategyManager imports working; this does
not migrate motive-generated full Goals into the live loop or complete #218.

## Fixture reconciliation and CI ownership

The26 existing utility methods now import actual production UtilityGoal explicitly,
removing the RealGoal-or-substitute selection and fixture Goal doubles. Existing
MockAction cases remain labeled fixture behavior; the fixture now generates an ID
only for None and preserves0, empty string andFalse. Numeric assertions are retained.
The original26 methods, four new contract methods and seven existing fixture-validation
methods are in canonical core CI. Three consumers of the removed MockGoal were migrated
to actual UtilityGoal; the simple utility script and advanced demo both run successfully.

The new coverage uses actual Actions for utility golden/urgency tests and complete
real Character, full Goal, Condition, GraphManager and Action objects for the native
bridge. These checks do not treat metadata as a full Goal substitute.

| Scenario | Independent asserted result |
| --- | --- |
| Metadata defaults/explicit zero | Alias role, score/urgency defaults, deadline/description/attributes; fresh default dictionaries |
| Three import orders and reload | Explicit UtilityGoal constructor in all cases, distinct from full Goal,29.1 golden and unchanged caller state |
| Urgency | .9 urgency yields exactly twice the .5-urgency score with identical priority/action/state |
| Full native Goal bridge | Graph/object energy20; target30; native utility12.7 without mutation; real Action executes both stores to30; completionTrue; satisfied-target score.05 |
| Action fixture IDs |0/empty-string/False preserved; None generates id(instance) |

Fresh base26-method execution reports16 passes,1failure,9errors,0skips. The failure
is falsy ID0, and all nine errors are the full Goal constructor receiving metadata
kwargs. The revised26 report26passes, zero failures/errors/skips. The new4-method
baseline reports5 failure records across3 failing methods (three import-order subcases),
zero errors; the already-working native bridge is a preserved-behavior control.
Four new plus13 full-Goal methods pass17/17 after the change. Baseline/result receipts
retain exact IDs, commands, revisions and counts; no import failure is hidden by a test
fallback and no weakened assertion establishes completion.

## Review, evidence and closure

Canonical core/dev/visual results and source hashes are retained in
[evidence/issue-774-local-results.json](evidence/issue-774-local-results.json).
Source `fb55befcaf0c7186e2aebfa7229e927eb4d04d6c` passes core 166 tests plus 3 CLI probes,
dev 20 and visual 69, all with zero failures/errors/skips and actual process exit0.
A fresh subprocess bounded to 60 seconds passes all 26 original methods at that source.
The supplementary real planner/strategy selection passed11 tests before the consumer
import repair. Hosted run links and final-head results are recorded on PR #822 after
completion. Profile environments use existing Python3.12
Linux hash locks; no new lint/typecheck configuration, timeout changes or model runs.
Independent source review found an import regression in three consumers of the removed
MockGoal. The failing validation import was reproduced; consumers were migrated and its
seven tests registered in core. All 37 affected methods then passed. Independent final delta review confirmed the
fix and identified no remaining findings;
the reviewer did not execute commands. Hosted checks must match the
final head; local evidence alone does not establish merged-main completion.

Close #774 only after merge and verification of main, completing its three original
criteria and reading them back. Keep historical16/1/9 evidence. #821 remains open:
this resolves the utility metadata boundary, not its wider inventory/catalog or full
live-loop integration requirements. Next is #218 planning integration (candidate689
requires selective review/rework), then #741 active-plan execution/replanning. Neither
those issues nor later phase gates close through these utility tests.

## GLM review reconciliation

GLM-5.3-Flash reviewed head d703e71 after main9505076 (agent configuration) was merged
into the branch; the nine source hashes matched the original local evidence revision.
The review used four read-only tiny-village specialists and reported no blockers.

Two P3 concerns were reproduced before repair: the Action fixture raised NameError
while logging exceptions from callable and check_condition preconditions; explicit
attributes=None stored None. The fixture now imports logging and returns False with
warnings, and metadata uses an explicit None default without discarding a supplied
empty mapping. A new validation method exercises both raising precondition forms;
the existing metadata method covers independent None defaults, supplied empty mapping
identity, explicit zero urgency and the preserved empty-name default. Original26 method
identities and numeric assertions remain intact. Retained red output is recorded in
the evidence receipt; the earlier166/20/69 result above remains historical.

The reviewer also noted pre-existing event-target units and legacy completion/evaluator
policy/arity concerns. These remain with #218/#821; source checks and a signed PR reply
record their disposition. The alias consumer suites/demos listed by the reviewer are
not all in canonical CI; no blanket coverage or full live-loop readiness is claimed.

Final GLM-fix source `9f36cfb607d4cfa25e13b3c61c9e431445a0cf70` passes core167 tests plus3 CLI probes,
dev20 and visual69 (all actual exits0; zero failures/errors/skips). The bounded
original26 and focused51 selections pass. Final independent delta review reports
no remaining defects; the reviewer ran no commands. See PR #822 for exact-head
hosted results and subsequent merge verification.
