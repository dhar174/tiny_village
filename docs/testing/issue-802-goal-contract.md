# Production Goal contract — #802

Approved implementation, 2026-11-14 America/New_York. Base main `a3e39f3` includes
#819's native runtime units. #802 remains open pending merge and merged verification.

## Interface and ownership

`Goal` retains all constructor parameters. `completion_conditions` is a bool-keyed
dictionary of Condition lists. Singular Conditions and tuple values normalize to
copied lists; Condition/target identities remain intact. Both buckets are requirements,
not truth assertions or negation. Every condition is checked on every evaluation;
empty containers are true. `target_effects` remain scoring/search metadata and do
not substitute for full-domain completion. Invalid containers fail explicitly.

`Goal.iter_completion_conditions()` is shared by completion and required-item
extraction. `Goal.check_completion(state=None)` returns bool. With no argument it
reads each resolved entity's graph-first native snapshot. With a State or mapping,
only actor conditions use that supplied snapshot, including an explicit empty mapping;
required missing attributes cannot fall back to live actor values. Other targets
use their own registered node, including non-character entities. Standalone entity
attributes remain the fallback when not registered. Existing field/entity aliases
are resolved; exact Condition comparisons apply, without utility scaling or 0.1 slack.

Missing attributes, malformed literals and unresolved entities raise errors with
Goal/Condition context. Malformed containers raise TypeError. Errors propagate through
production planner and graph checks rather than being counted as ordinary False/None.
Legacy lightweight-goal branches keep their prior behavior; they do not establish
production acceptance. No goal buckets, snapshots, Characters or graph values change
while checking completion/planning. Shared inventory references are only read.

`Plan.evaluate()` retains its no-argument interface and delegates to Goal completion.
`GraphManager.is_goal_achieved(character, goal)` uses the same predicate and rejects
an actor mismatch. Production `evaluate_feasibility_of_goal(goal, state, *, actions=None)`
is True for an already-completed goal or any plan found by existing bounded search.
It checks `is not None` so an empty successful plan counts. Explicit empty action
lists remain empty. False means no plan found within 1000 iterations / ten actions,
not global impossibility. No action is executed by feasibility.

## Real-object acceptance

Nine maintained methods in `tests/test_production_goal_contract.py` enter core CI.
They construct complete real Characters and the actual MultiDiGraph-backed world.

| Scenario | Asserted outcome |
| --- | --- |
| Singular/tuple buckets, multiple requirements | Copied lists; both buckets reevaluated; threshold 29.95 fails, 30 passes; hunger 11 fails maximum10 |
| Graph20/object90 actor | Live completion/Plan/graph achievement false; explicit energy30 true without changing live state |
| Actual recovery Action | Graph20 and stale object90 become30 together; Goal, Plan and graph achievement become true |
| Already complete / reachable | Already-complete state returns []; recovery from20 returns usable action, independent final energy30 |
| Unreachable / blocked | Empty actions and energy21 precondition at20 yield None/False; repeated reachable positive control still succeeds |
| Equivalent effects / different costs | Cost1 action selected ahead of cost5 in both input orders |
| Feasibility bounds | Two +5 actions cannot reach30 with depth1, can with depth2; explicit mappings also work |
| Partner / location ownership | Actor snapshot cannot satisfy partner; graph partner30 beats object90; location graph open=True beats object=False |
| Errors / inventory / imports | Missing and invalid inputs propagate, food predicates support single/double quotes and true/false controls, both import orders construct full Goal |

## Failures retained and review

Seven initial new cases failed against the unchanged #819 source. Setup failures
from initially missing partner construction arguments were corrected before the clean
baseline rerun and are not product evidence. Independent review of PR #820's initial
head found stale non-character reads (P2) and bare malformed-literal errors (P3).
Each now has a real failing-before/passing-after regression. All nine acceptance
methods pass after the integration fixes. Review did not independently execute tests
or review the unpublished integration patches; the covering tests verify those patches.

Real inventory evaluation exposed `sum(count_food_items_total())` where the count
is already an integer. The one food predicate expression is corrected; other inventory
categories and economic/catalog behavior are not claimed complete.

Original #736 sources/captures are unchanged. Re-running the old container source
still exits1 at its final assertion: it directly sets Character.energy=30 while the
registered graph remains20. It no longer crashes on the list container. Maintained
coverage uses actual synchronized Action execution and separately proves graph/object
precedence; this old fixture failure is not relabeled passing coverage. Existing job
role parsing error logs also appear in that old reproduction, as before.

## Verification and closure

Local core at `2eb3598`: **125 tests + 3 CLI probes**, zero failures/errors/skips.
Dev at `7b22725`: **20 tests**; visual at `7b22725`: **69 tests**, zero failures/errors/skips.
Those dev/visual runs precede the diagnostic-only malformed-literal correction;
core and the nine final acceptance cases cover that correction. The
[local receipt](evidence/issue-802-local-results.json) retains exact revisions.


The compact local receipt records exact source revisions, commands, profile counts,
lock hashes and source hashes. Canonical commands are `python -B scripts/run_ci_checks.py
--profile core|dev|visual --artifacts PATH` under the existing Python3.12/Linux locked
profile environments. No lock, lint/typecheck configuration, search limit or timeout
was changed. Supplementary real planner integration and strategy modules pass11 tests
with one existing uu deprecation warning. Final hosted receipts must be published on the PR after the current head passes.

Close #802 only after repair merge and merged-source identity/required-check verification,
then update both original criteria and read the issue back. #218 broader integration,
#741 active plans/replanning, #774 utility fixtures, #803 graph interaction limitations,
#816 other writers, specialized graph path/cost evaluators and later catalog/economic
work remain separate. Passing these cases does not establish global plan optimality,
complete motive catalogs, gameplay episodes, live AI inference or platform readiness.
