# #821 Goal, utility and inventory contracts

Base: main a600aee. Implementation authorized by D041; final repairs and ready
merge authorized by D042. Current verification: all known retained failures and
the parser diagnostic are addressed; see the final merge gate below.
Execution outline:
[approved plan](../superpowers/plans/2026-11-25-issue-821-goal-contracts.md).

## Supported boundaries

`tiny_utility_functions.Goal` permanently aliases data-only `UtilityGoal`, distinct
from `tiny_characters.Goal` in either import order. Existing lightweight
construction sites remain metadata. Full Goals retain Conditions as completion
requirements; an empty full Goal is deliberately complete. Empty metadata is
incomplete. Utility `is_goal_achieved`, GOAP `_goal_satisfied`, and GraphManager
`is_goal_achieved` now share this distinction. Priority/description cannot prove
completion. Numeric metadata uses absolute native targets, negative upper bounds,
canonical aliases and the existing 1e-9 tolerance, with required finite state.

`Goal.evaluate_utility()` invokes the configured public callback once with the
Character, Goal and environment. The actual factory convention already used these
three arguments; no five-argument factory bug was reproduced. The separate utility
helper retains its five-argument signature. Missing/invalid evaluators or scores
raise contextual `GoalEvaluationError` with their original cause. Raw state and
bucket contract errors keep their existing types. Strategy cannot reinterpret a
required evaluation error as a priority fallback; request APIs retain ERROR.

Inventory category checks count quantities, not stack records. Attribute queries
inspect public stored data on current positive-quantity items, continue after
missing/nonmatching attributes, and can restrict an existing category through the
optional fourth argument. Invalid operators/arguments/comparisons raise explicitly.
Cook retains both Conditions and restricts cooked matching to food.

Employment derives from canonical graph job, or solely from an explicit actor
snapshot. Construction can read local job. JobRoles (including Job subclasses)
project to names for scalar planning; missing job differs from explicit None.
Actor/partner request fingerprints track job dependencies. Unsupported job objects
remain excluded from unrelated scalar planning; employment checks fail explicitly.
No cached authoritative has_job field or arbitrary bound-method fallback is added.

Goal equality retains its field list with the corrected goal_type operand. Mutable
Goal/Plan objects are unhashable; Plan inspection/comparison does not call evaluate.
Typed actor lookup and existing ambiguity errors remain intact.

## Historical implementation verification

Eight new real-object tests and thirteen existing production Goal tests pass.
All local canonical profiles pass: core **253 tests + 3 CLI probes**, development
**20 tests**, visual **70 tests**. Every batch exits0, with zero assertion failures,
errors or skips. All eight committed source hashes match the tested snapshot.
Full commands, profile summaries, locks, source hashes, review scope and retained
failures are in the [verification receipt](evidence/issue-821-review.json).
The implementation head6744d37 also passed all eight hosted checks; final-head
check and PR delivery state remain authoritative on
[PR #828](https://github.com/dhar174/tiny_village/pull/828).
The initial32-test regression run included duplicated inherited controls and had
six failing new groups. Discovery was corrected before final verification.
The first implementation run failed bootstrap by accepting only Job rather than
its actual JobRoles base; this was corrected before the passing runs.

## Historical limits before the final merge gate

The scalar planner still excludes inventory/location simulation. Catalog Goals
are not automatically enabled in live selection; missing authoritative sources
for unsupported predicates such as investment remain explicit errors. These
repairs do not implement a portfolio engine or establish full catalog, live-model,
Windows/GPU, visual-product or later human-gate readiness.

The separately retained102-test building/event suites have three known exact-base
failures; existing job-parser ERROR output remains. Passing assertion summaries do
not mean clean stderr. Reviewer conclusions are based on supplied source and
execution evidence; no native Zcode or independent command execution is claimed.

## Requested specialist review and re-review

Review of head4952ce6 found an Important authority defect: deleting registered
graph job data could revive a stale object job in actor snapshots and partner
fingerprints. The correction preserves authoritative key presence in actor
snapshots and reads registered partner job dependencies only from graph copies.
Present None remains different from missing; unrelated fields and bootstrap
retain their existing behavior. The regression was observed RED then GREEN; all
10 focused real-world tests pass (16.005s). World-state and planning/actions
specialists re-reviewed the corrective delta and found no remaining blocker in
the supplied excerpts. Corrected-source canonical checks pass: **255 core tests + 3 CLI probes**, **20 development**, **70 visual**, every batch exit0 and no
assertion failures/errors/skips. All eight committed source hashes match the
verified snapshot. All eight hosted checks passed on corrective head8088196.
The253-test receipt above remains historical. The additional102-check suite was
rerun and retains the same3 baseline failures (exit1); parser ERROR output persists.
See the [re-review receipt](evidence/issue-821-rereview.json). Final documentation
commit checks and open PR delivery state remain authoritative on GitHub.

## Final merge gate repairs (D042)

The user required failures/errors to be addressed before merging. The three
previously retained failures are now repaired: non-native dispatch uses its
existing mapper compatibility path, and the lifecycle test exercises public
processing/bookkeeping at the event occurrence time before cleanup. Native
Character synchronization remains intact. Job diagnostics count actual parsed
entries instead of an inconsistent strict-name comparison; all117 roles parse
without the false117/116 ERROR. A positive event fixture now carries concrete
numeric state and asserts the effect; the invalid-schema negative control
explicitly checks its expected error log. The combined113-check focused suite
passes with no uncaptured ERROR records. Final canonical verification passes:
**256 core tests + 3 CLI probes**, **20 development**, **70 visual**; every batch
exits0 with zero assertion failures/errors/skips. All12 committed source/test
hashes match the verified snapshot. All8 hosted checks passed on repair
commit96e9768. The [final verification receipt](evidence/issue-821-premerge.json)
retains commands, complete profile summaries, fixes and review evidence. Earlier
baseline-failure limitations above describe historical revisions and are resolved.
D042 authorizes normal exact-head merge once the final evidence commit's hosted
checks pass; GitHub records the actual integration outcome.
