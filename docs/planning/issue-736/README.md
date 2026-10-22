# #736 — Functional-test PR walkthrough

**Review in progress; Eight dispositions selected under D029, not yet applied.** Baseline: merged
main `11b6e93595a70ce9f24cfb239a31ac873847ff56`, including #703/#798. #735 is
closed; #736's prerequisites are closed. #737 and human gate #738 remain later
steps. No test/runtime changes, PR closures, rebases or GitHub metadata writes
have been performed for this walkthrough.

The user requested a step-by-step review of all nine candidates. Preserve
choices in the single docs/decisions.md file after the user answers. Current
[heads](heads.md), [PR descriptions/reviews/discussion](evidence/prs.json) and
bounded captures are retained. Inspect current heads again before any future
mutation; old branch check results alone do not establish compatibility.

## First decision — #420

Head `264d07e660bb`; 55 main-only commits and seven PR-only commits at review.
It changes tests/verify_genuine_failures.py, a path absent from current main.
All **five existing candidate test methods pass** against current main in the
locked core environment, with no errors, failures, skips or timeout. The full
original candidate source executes in an isolated module with its expected
repository-relative path, importing the actual current production systems.
No repository test file was created or changed for that probe.

Two methods contain useful production assertions: State read/write/defaults,
and Action effect/precondition behavior using real Action/Condition objects.
They also include unrelated nonexistent-name checks. Three other methods focus
on Python arithmetic, file existence or absence of a made-up source string.
These are not evidence of gameplay, graph ownership or goal completion.
Current tests/test_actions.py and tests/test_action_definitions_resolution.py
already cover related action/precondition/State behavior; any port should keep
only unique contract assertions and avoid duplication. A green five-test demo
is not a reason to restore the whole demonstration script.

Options:

1. **Split/salvage (recommended):** map unique useful assertions into maintained
   targeted modules, validate, then supersede #420 with a precise crosslink.
2. Rebase and retain the full PR, accepting extra demonstration/test maintenance.
3. Supersede without porting; retain the gap analysis so omitted assertions are
   deliberate rather than silently lost.
4. Keep open/defer; this postpones the disposition and leaves #736 incomplete.

**User choice: split/salvage, then supersede after validation**, recorded in
[D029](../../decisions.md#d029--review-736-functional-test-pr-dispositions-step-by-step).
No implementation or closure yet; finish the walkthrough before execution.

## Remaining review queue — preliminary source findings only

#417: user selected split/salvage unique utility assertions, then supersede after
validation (D029). #408 is selected for superseding through the verified
consolidation, preserving any additional unique useful assertion found during
mapping. #406 is selected for salvage of useful public memory API assertions,
then superseding after validation; existing memory tests and unresolved
requirements remain intact. #404 is selected for superseding as an obsolete fix
with no current caller. #397 is selected for superseding with maintained
counting-coverage links, retaining its diagnostic evidence. #395 is selected for
salvage of strict planner state/goal-completion assertions, then superseding after
validation. #391 is selected for behavior-focused happiness checks, including
applicable and absent relationship cases, then superseding after validation.
Broken required behavior remains a failure or tracked blocker. Only #393
remains pending.

| PR | Source finding; selected dispositions above, #393 pending |
| --- | --- |
| #417 | Mixed real utility/Action checks, source-text happiness checks and generic Python demonstrations across four files. One selected original utility method passes against current main; that narrow result does not validate the entire PR. Inspect useful assertions against existing utility/#425 coverage before choosing a split/carrier. |
| #408 | Overlapping earlier test_fix_validation and demonstration files; several checks inspect names/source rather than behavior. Compare against #417 and maintained suites before superseding. |
| #406 | Nine illustrative tests include Python operations and locally invented UserRepository/DataProcessingService implementations, not TinyVillage domain behavior. Also touches the old memory suite and an agent session log; separate any useful memory assertion from unrelated examples. |
| #404 | Adds format_traceback only to tests/test_completed_implementations.py, absent in current main. The current root-level file from #349 has no such wrapper/call. A repository search finds no current Python format_traceback caller; the maintained runner preserves stderr/full failure output. No current caller requires the old helper. |
| #397 | Adds a counting helper and competing test_fix_validation file. A controlled two-failing-subtest probe makes its helper report minus one passed test for one executed method. The current bounded CI runner already has explicit subtest-counting coverage. Examine any unique counting case without restoring the obsolete wrapper or weaker happiness checks. |
| #395 | More planner calls, but explicitly treats some None results/exceptions as successful bug detection. Those paths must fail on broken behavior; map useful scenarios to maintained planner tests and retained #799 defects. |
| #391 | Replaces four happiness source-keyword requirements with a three-feature threshold and warnings. Neither source keyword counts nor relaxed thresholds prove happiness behavior; discuss replacing them with concrete behavior criteria rather than weakening acceptance. |
| #393 | Introduces a dict-backed MockGraphManager and changes production planner call signatures. Signature binding confirms its reduced-argument calls do not match current production methods; required character/state/goal context would be lost. The claimed heuristic constant already exists in its diff context. Do not treat this as a test-only merge; evaluate graph/planner assertions separately from those production regressions. |

## #349 baseline and verification limits

The current root test_completed_implementations.py is byte-for-byte identical
to #349's final root file. Its deterministic building collision assertions were
retained. Its other tests still include manual pause-attribute toggling, happiness
source scans and planner method-existence checks. The stale candidates generally
modify a different tests/ path; they must not revive competing layouts or weaken
assertions merely because #349 is merged. #349 is a comparison baseline, not
proof that all current functional coverage is sufficient.

#420's five methods and #417's one selected utility method pass on current main.
A third bounded capture checks #393's reduced argument counts against current
production signatures and exercises #397's original counter with one method
and two controlled failing subtests: total=1, failures=2, passed=-1. Those two
injected subtest failures are expected evidence of the counter defect, not passing
product tests. None of these runs validates an entire other PR. All three captures
use the established 20s diagnostic / 25s parent deadline, with zero timeouts.
Remaining findings are source-reviewed, not executed. Production/graph/gameplay acceptance and existing recorded failures
remain unchanged. Continue the walkthrough before any consolidation plan or
specific superseding action is approved.
