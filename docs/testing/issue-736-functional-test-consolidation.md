# #736 functional-test consolidation — validation and handoff

The approved test consolidation is implemented, with **21 new maintained test methods passing in core and development environments**. Required full-Goal, graph/world and motive-satisfaction contracts remain unfulfilled. **D030 accepts the scoped reconciliation with these requirements carried into their runtime issues.** #804 subsequently merged at6cc576e and passed the agreed post-merge checks; #736 and the nine superseded candidates are now closed for the accepted scope. See [post-merge closeout](issue-736-post-merge-verification.md). No runtime repair or phase gate is approved.

The nine human dispositions and execution approval are in [D029](../decisions.md#d029--review-736-functional-test-pr-dispositions-step-by-step). The [assertion map](../planning/issue-736/assertion-map.md) identifies retained, redundant, obsolete and blocked assertions. [The approved plan](../superpowers/plans/2026-10-22-issue-736-functional-test-consolidation.md) supplies the scope and closure boundaries.

## Revisions and environment

- Baseline production/main: `11b6e93595a70ce9f24cfb239a31ac873847ff56`; all nine candidate heads remain the pinned heads. Refreshed GitHub metadata reports all OPEN, eight conflicting, #408 mergeable. That does not validate any stale branch.
- Implementation commits: memory `757db0b`, Action/utility `b1b85b6`, happiness `a8aa2a4`, planner/reproductions `bde0331`, local selectors/building assertion `5c48ade`. No production runtime modules or dependency locks changed.
- Python 3.12.14/Linux x86_64, existing locked `/workspace/.venvs/issue726-core`, `issue726-dev` and `issue726-visual` interpreters. No installs, live models, downloads or remote CI validation were performed for this task.
- Integrated profile summaries record HEAD `a8aa2a4`: the planner file and selector/building edits were already present in the working tree and committed during the runs, without content changes. The checked source content is committed in `5c48ade`; [validated-source-hashes.json](../planning/issue-736/execution/validation/validated-source-hashes.json) identifies those files. The recorded revision alone does not describe the full then-uncommitted tree.
- New cold children use the existing durable helper's 20-second diagnostic / 25-second parent deadlines. Existing suite/CI supervisor deadlines are retained. No timeout was observed. Passing constructors still emit existing job-role parsing ERROR logs; these are retained, not described as pristine output.

## Executed results

Counts below describe executions, with overlap explicitly retained. They must not be summed as unique tests. New tests cover existing correct behavior; no production code was changed merely to manufacture a RED/GREEN sequence.

| Command/selection | Environment | Result |
| --- | --- | --- |
| `python -m unittest -v tests.test_action_contracts tests.test_utility_contracts` | core, then dev | 8/8 pass each; 16 cold children, no skips/timeouts. |
| `python -m unittest -v tests.test_memory_runtime tests.test_lightweight_runtime.TestMinimalRunner` | core, then dev | 14/14 pass each, including one new public API test. |
| `python -m unittest -v tests.test_character_happiness_contracts` | core, then dev | 7/7 pass each; required added motive contribution is separate and fails. |
| `python -m unittest -v tests.test_goap_graph_contracts` | core, then dev | Final 5/5 pass each; primitive contracts only. First core run had five fixture failures, described below. |
| `python scripts/run_ci_checks.py --profile core --artifacts /workspace/scratch/issue-736/execution/profiles/core` | core | **72/72 tests + 3/3 CLI probes PASS**, no skips/errors/failures. All 21 new methods are included. |
| `python scripts/run_ci_checks.py --profile dev --artifacts /workspace/scratch/issue-736/execution/profiles/dev` | dev | **20/20 PASS**, including maintained runner coverage; no skips/errors/failures. |
| `python -m unittest -v tests.test_actions tests.test_tiny_utility_functions tests.test_utility_with_real_classes tests.test_issue_425_fix` | core, worktree | 40 methods: 30 pass, 1 failure, 9 errors. |
| Same preceding command on exact `git archive origin/main` source | core, baseline | Identical 40 methods and all 10 unsuccessful IDs/classifications; independently reproduces the existing failures. |
| Existing `tests/test_goap_planner_integration_real.py` via bounded pytest worker | dev | 6/6 pass; this existing suite uses stubs and does not prove real graph planning. |
| Root `test_completed_implementations` via bounded unittest worker | visual with SDL dummy drivers | 4/4 pass, including retained building map identity assertion. Happiness source scan is still present. |
| Extra worker checks before coordination stop arrived | dev | GOAP error handling 5/5 and the same stub planner suite 6/6 pass; duplicate evidence, not additional unique coverage. |

The ten unchanged utility failures are nine `Goal(priority=...)` constructor errors and `test_mock_action_preserves_explicit_falsy_action_ids`. All occur in `tests.test_tiny_utility_functions.TestTinyUtilityFunctions`; exact method names and baseline equality are retained in [the Action/utility summary](../planning/issue-736/execution/validation/action-utility-validation/summary.md). Existing #774 retains that work. Action 5/5, real-class utility 5/5 and #425 tests 4/4 pass in the affected suite.

The initial planner module used a mistaken name lookup in the controller's UUID-keyed registry. All five children failed at fixture setup. Only that fixture lookup changed to registry values; original failures and their source remain in the archive under `graph-validation/core/fixture-invalid-*` and the associated captures. These five failures are neither production defects nor passing evidence.

## What was preserved

- Three State/Action tests: dict reads/default/write, direct effect `energy 5→7`, and complete real Character Condition/precondition pass at 12 and fail at 5 against threshold 10.
- Five real-Action utility tests: golden `29.1`, four explicit boundary states, finite results, caller-state preservation, hunger/alignment ordering and diminishing value under repeated simulated eating. The public utility Goal is the lightweight production export in these fresh imports; it is not the full Character Goal.
- One memory API test: direct concrete MemoryManager/SpecificMemory/GeneralMemory class imports without service initialization. Existing real record/recall tests remain. This proves neither semantic memory construction nor inference.
- Seven happiness tests: supported motive caps, social quality, partner/family contributions and legitimate absent relationships, all on complete real Characters. Character-local relationship input is not presented as graph-derived relationship integration.
- Five planner primitives: real Goal/action queue identities, blocked real action execution, isolated simulated effects, contextual cost/heuristic behavior and real Action-sequence utility selection. **None proves successful full-Goal planning or synchronized world execution.**
- #395's alternative-action None-or-dict/action/name shape check is explicitly excluded: it cannot distinguish successful replacement from an implementation returning None forever. Actual alternative selection/replanning remains unverified under #741; no new test or diagnostic is misrepresented as having exercised it.
- The unique old building map-retention assertion is now explicit in the existing deterministic root building test. Generic Python demonstrations, fabricated missing names and copied production implementations were not ported.

## Strict reproductions and classification

The [blocked source directory](../planning/issue-736/execution/blocked) contains directly rerunnable text files, outside discovery by design. They are **unfulfilled requirements or explicitly classified candidate misuse**, not passing tests or silent exclusions. The validation archive retains exact code, commands, return codes and child artifacts.

| Finding | Observed result | Tracking / acceptance |
| --- | --- | --- |
| Basic motive-satisfaction pair | Scores 2 and 8 both yield `0.3610225808639045`; absent real motive getters are swallowed. One core reproduction fails. | [#803](https://github.com/dhar174/tiny_village/issues/803), related #139. Root source scan retained pending disposition. |
| Full Goal completion container | Constructor accepts `{False: [Condition]}`; completion calls `.check_condition()` on the list and errors. | [#802](https://github.com/dhar174/tiny_village/issues/802), related #167/#218. |
| Full Goal/planner state interface | Planner passes a state to a no-state method, catches the mismatch, and reports False. Satisfied/reachable/cost-selection cases return None. | #802. Independent final predicates remain required; no substitute Goal adapter. |
| Negative planner cases | Unreachable/blocked cases return the expected None; their positive completion controls fail. | Attribute the control failure to #802; do not claim the negative no-plan observation is broken. |
| Plan evaluation / real-Goal feasibility | Plan evaluation hits the same container error; feasibility mishandles boolean/list completion representation. | #802, coordinated with #741/#218. |
| Graph state provenance | Actual node energy is set to 77; planner returns Character-backed 20 after real name lookup fails. | Existing [#799](https://github.com/dhar174/tiny_village/issues/799); no alias nodes or mocked return values. |
| Graph action provenance | Real registered graph interaction is bypassed; returned actions are fallback Rest/Idle. | #799, with remaining planner integration under #218/#602. |
| Executed world synchronization | Action reports True and Character changes 20→30, while the graph stays 20; UUID update cannot find a node. | #799; no successful world-execution acceptance. |
| Mutable Character identity | State mutation changes hash and breaks graph lookup for the registered object. | #799, already scheduled after separate human gate #738. |
| Replan result | Fallback preserves the queued action but returns None instead of the candidate's explicit boolean expectation. | Existing [#741](https://github.com/dhar174/tiny_village/issues/741); return/failure contract needs review before repair. |
| Candidate `evaluate_utility(Plan, character)` | TypeError; current API takes an Action sequence and selects an Action. | **Excluded candidate interface misuse**, not a production defect. Correct API use passes in maintained coverage. |

Fifteen graph/planner sources each exit 1 in core and dev, with no timeouts. Two of those 30 executions are the excluded candidate misuse; negative cases fail their positive controls as described. Together with the one motive reproduction, this is **31 failed diagnostic executions**, not 31 distinct defects. No skip/xfail or exception-as-success converts these into acceptance.

Separately, #404's replacement evidence injects one controlled worker error through nested `outer_736`/`inner_736` frames. The worker correctly reports FAIL and retains the full traceback; the outer validation of that retention exits zero. This is an expected-error diagnostic, not a passing product test. Existing #397 subtest-counting diagnostics and current runner counting tests remain distinguished the same way.

Example replay from repository root:

```sh
PYTHONPATH=. /workspace/.venvs/issue726-core/bin/python docs/planning/issue-736/execution/blocked/happiness-motive-satisfaction.py.txt
/workspace/.venvs/issue726-core/bin/python < docs/planning/issue-736/execution/blocked/planner_reachable.py.txt
```

## Retained evidence

The [validation index](../planning/issue-736/execution/validation/README.md) explains the complete raw archive and its 772 member hashes. Original captures retain absolute scratch paths; the index maps them to portable extracted paths. Expanded summaries include core/dev profile reports, source hashes, baseline comparison, per-module results and [graph classifications](../planning/issue-736/execution/validation/graph-validation/HANDOFF.md). Candidate source snapshots have a separate immutable source index/hash manifest. The original walkthrough's 21 evidence hashes are unchanged.

## Review and closure handoff

The test ports and local profile integration were accepted under D030 and merged. Required runtime behavioral acceptance remains unfulfilled in the owner issues. #802/#803 record specific later runtime work; existing #799/#741/#774 remain open. Proposed scheduling is after separate gate #738, with no new reverse dependency, runtime design approval or automatic Project change. #737 remains responsible for subsequent ownership/order/Project synchronization.

There are three reasonable dispositions for the remaining #736 gap:

1. **Recommended: accept the scoped reconciliation with the failing requirements explicitly carried into their runtime issues**, after reviewing these reproductions. That would need a separate recorded human acceptance of the limitation before #736 or affected candidates close; it claims no successful real-Goal/graph runtime behavior.
2. Keep #736 and its candidate closures pending while planning narrowly scoped runtime repairs before completing this acceptance. That changes the current after-#738 runtime order and needs an explicit ordering/design decision.
3. Defer this consolidation's merge and leave the current roadmap blocked until that ordering/acceptance question is settled.

The options above were presented for review. dhar174 explicitly selected option 1 under [D030](../decisions.md#d030--accept-736-reconciliation-and-retain-runtime-requirements) on 2026-10-22, requiring the runtime gaps to remain referenced in their correct issues. This records scoped reconciliation acceptance with retained runtime requirements; options 2/3 were not selected. The replacement must merge before any selected superseding closures. No linked runtime issue, PR review or phase gate has been automatically approved or closed.

## Independent review

A fresh reviewer inspected branch `a259efe` against main `11b6e93`, including the
seven changed test/selector files, retained source and all772 archive hashes.
It found no Critical issue and judged publication as a partial draft safe.
One Important gap was the missing explicit disposition of #395's conditional
alternative-action shape assertion; the map/report now explain its exclusion
and retain real alternative behavior under #741. A stale current/baseline
byte-identity statement was also corrected. These are documentation corrections;
validated test/runner source and all captured outcomes are unchanged. No second
review or test rerun is claimed for this correction.

## Publication

Published as [draft PR #804](https://github.com/dhar174/tiny_village/pull/804).
Evidence was linked on #736, #802/#803, #799 and #741, with candidate-specific
handoff comments on all nine old PRs. Publication receipts are retained under
execution/. No candidate, runtime issue, #736 or human gate is closed or approved.

## Accepted carry-forward — D030

Runtime tracking is explicit: #802 under #218 for full Goal contracts; #803 under
#139 for motive satisfaction; #799 for graph identity/authority/synchronization;
#741 for replanning/alternative behavior; #774 for existing utility test debt.
Original acceptance text, failed captures and existing prerequisites remain.
#802/#803 follow separate gate #738; their parent acceptance includes these child
requirements. #736 was subsequently completed after #804 merged and was verified, as recorded in the post-merge report. Scoped
acceptance does not mean any failed runtime outcome has passed.

### Carry-forward tracking readback

The eight owning/parent/reconciliation issue bodies retain their original text
and now include D030/evidence links and unchecked runtime requirements. Native
parents are #218→#802 and #139→#803; both children depend on #738 and each parent
on its child. The affected28-node prerequisite ancestry was checked for cycles.
Existing #799/#741/#774 relationships remain unchanged. #737 and #738 receive
explicit handoffs so Project synchronization and later gate review retain the
gaps. Readback and body-preservation receipts are in
[execution/option1](../planning/issue-736/execution/option1).
