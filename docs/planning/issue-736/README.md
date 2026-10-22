# #736 — Functional-test PR walkthrough

**Scoped reconciliation complete under D030; #804 merged/verified, #736 and nine superseded candidates closed. Runtime requirements remain open.** Baseline: merged
main `11b6e93595a70ce9f24cfb239a31ac873847ff56`, including #703/#798. #735 is
closed; #736's prerequisites are closed. #737 and human gate #738 remain later
steps. Test ports and local selectors are implemented; runtime files remain unchanged.
All nine candidates were closed as superseded after #804 merged and its merged
source passed the agreed checks, under D029/D030. No candidate was rebased or
merged. Runtime follow-ups #802/#803
track demonstrated gaps; see the [validation report](../../testing/issue-736-functional-test-consolidation.md).

The completed step-by-step review and confirmed choices are recorded in the
single docs/decisions.md file under D029; D030 accepts the scoped reconciliation
with runtime requirements carried forward. Pinned [heads](heads.md),
[PR descriptions/reviews/discussion](evidence/prs.json) and bounded captures are
retained. The sections below preserve the historical walkthrough at review
baseline `11b6e93`; the final closeout is in the linked post-merge report.

## Historical first decision — #420

Head `264d07e660bb`; 55 main-only commits and seven PR-only commits at review.
It changes tests/verify_genuine_failures.py, a path absent from review baseline main.
All **five existing candidate test methods passed** against `11b6e93` in the
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
The selected salvage was implemented and validated in #804. #420 is now closed
as superseded; implementation evidence and retained runtime gaps are in the
validation and post-merge reports.

## Historical review queue — preliminary source findings at `11b6e93`

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
Broken required behavior remains a failure or tracked blocker. #393 is selected
for salvage of real graph/goal contract tests, then superseding after validation;
the latest answer replaces the earlier rework/rebase choice (D029). The
[consolidation plan](../../superpowers/plans/2026-10-22-issue-736-functional-test-consolidation.md)
was approved, implemented and merged in #804. All nine selected superseding
closures have been applied. Required failed runtime contracts remain open in
#802/#803/#799/#741/#774 under the explicit D030 carry-forward acceptance.

| PR | Source finding; selected dispositions recorded above |
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

At baseline main `11b6e93`, root test_completed_implementations.py is byte-for-byte
identical to #349's final root file. This branch retains its deterministic building
collision assertions and adds an explicit assertion that the supplied map object
is retained. Its other tests still include manual pause-attribute toggling, happiness
source scans and planner method-existence checks. The stale candidates generally
modify a different tests/ path; they must not revive competing layouts or weaken
assertions merely because #349 is merged. #349 is a comparison baseline, not
proof that all current functional coverage is sufficient.

#420's five methods and #417's one selected utility method passed on the review
baseline `11b6e93`. A third bounded capture checked #393's reduced argument counts
against that revision's production signatures and exercised #397's original counter with one method
and two controlled failing subtests: total=1, failures=2, passed=-1. Those two
injected subtest failures are expected evidence of the counter defect, not passing
product tests. None of these runs validates an entire other PR. All three captures
use the established 20s diagnostic / 25s parent deadline, with zero timeouts.
Those initial source findings and three captures were the historical review
inputs, not the later implementation's complete validation. The validation report
records the subsequent maintained tests and failed runtime reproductions; the
post-merge report records the passing merged-source checks, all nine superseding
closures and scoped #736 completion. Runtime/graph/gameplay requirements and
recorded failures remain outstanding in their owning issues under D030.

[Post-merge verification and closure readbacks](../../testing/issue-736-post-merge-verification.md) record the final outcome. Next task: #737; no #738 approval is supplied.
