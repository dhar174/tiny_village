# Issue #740 completion investigation

Date: 2026-11-14 (America/New_York).

## Revision and decision

Investigated merged PR #818 at
`c8f0f17ac2d0851ad667b0088f81df4c57b6eb43`. The issue was still OPEN and its
Memory Bank entry still described pre-merge status. #220 is CLOSED and #738 is
CLOSED under the explicit human approval in D034; those decisions are preserved.

**Baseline verdict: not complete.** Green PR/main CI and the prior review did
not cover the additional runtime unit paths described below. The user authorized
the closure investigation and narrowly scoped repairs, with re-verification.

The verified repair source checkpoint is
`675301460354bc341f5c526673c729eeaebecee2`. The issue must remain open until the final repair is merged
and merged-source verification passes. This report is an engineering assessment,
not a new human phase-gate approval or a claim that Phase 4 is complete.

## Acceptance and ownership

| #740 criterion | Required evidence | Implementation/evidence |
| --- | --- | --- |
| Document one unit per canonical field and explicit boundaries | Field table, source/consumer inventory, fixture exception | [Runtime unit reference](../reference/runtime-units.md); canonical native snapshots and named utility adapters |
| Validate clamps, thresholds, costs, targets on real scenarios | Executed real Character/Action/Graph/controller cases, including failures | Existing scale/execution contracts plus `tests/test_runtime_units.py`; canonical core/dev/visual checks |
| Record representative before/after state | Exact values in both stores and normalized consumers; retained failures | Reproduction table below, maintained assertions and JSON check reports |

Small native values remain small native values. This is not a global field or
saved-game migration, nor permission to rebalance every action by multiplying
its constants. Related #184 and #187 remain open.

## Findings and falsifiers

| Path | Observed at baseline | Required/repaired outcome |
| --- | --- | --- |
| Dictionary → strategy utility snapshot | Energy 77 stayed 77; object input became .77 | Both yield .77; hunger 60 becomes .60; 100 credits stay 100 |
| Real Character → GOAP action utility | Rest at graph energy 20 fell through to −.1 | Graph-first native bridge yields .8; useful Rest outranks idle |
| Generic interpreted action | execute returned True but the singular `target` key applied no satisfaction effect | Supported `targets` descriptor writes +1 to both stores |
| Strategy then building-style Action effects | Rest wrote `satisfaction=8`; a +5 current_satisfaction effect wrote 5; consumer still saw .08 | Generic +1 → Rest +8 → service descriptor +5 yields 14 in both canonical/compatibility views and .14 in utility |
| Catalog hobby | Happiness .5 became 2.5 | Fractional +.2 yields .7; scalar bounds cap fractions at 1 and point needs at 100 |
| Native event fields beyond core three | Object-only mapper path could leave registered graph values stale | Graph 70/object 10 +10 yields 80 in both, then +200 caps both at 100 |
| Canonical need effect scoring | Canonical field names could miss consumer alias scoring | Hunger 80, Eat −20, cost .1 scores 2.2; two projected meals score 3.6 for canonical and alias state/effect keys, without changing input |
| Prompt effect snapshot | Stale seeded satisfaction alias could mask native current_satisfaction | Graph current_satisfaction 80 beats alias 10/object 20; set:25 yields −.55 utility delta |
| Few-shot matching | Native examples were compared directly with fractional consumer input | {.7 energy, .3 social wellbeing, .5 mental health} selects SocialVisit instead of the baseline Sleep mismatch |
| Selected-character UI | Actual visual render raised NameError for undefined HUNGER_OFFSET | Real dummy-display render shows graph Energy 77, Hunger 60, Health 90, each /100, despite stale object values |

Four original core falsifiers failed on the baseline (4 failures, no errors or
skips). After binding the real rendering provider, the UI baseline reproduction
failed specifically with `NameError: HUNGER_OFFSET`; an earlier test setup had
failed before reaching that line and is not counted as the product reproduction.
All new checks use the real world harness. The visual check records text while
delegating to real Pygame fonts and a real surface; no mock Character or graph
establishes acceptance. Pure scalar assertions supplement those world scenarios.

PR #819's automated review found a further projection gap at `0ba397c`:
alias-only initial snapshots were not canonicalized before plan effects were
simulated. The expanded existing core case failed before the correction (one
failure, no errors/skips). Both native scoring entry points now canonicalize
their copied input before projecting; all four canonical/alias state-and-effect
combinations assert identical one-action and two-action scores, with unchanged
input dictionaries. This correction is included in the source checkpoint above.

Existing maintained traces also cover Rest 20→35 and 95→100; hunger 2→0;
registered health 80−20→60; Sleep/Work at 29/30; add/set projection parity;
per-attribute rollback; graph-first energy 77 → utility .77 and prompt /100;
fixture 7→70, −6→−60, and native 10→fixture 1. The earlier PR review's
money set:25 from graph credits 100 → −75 delta remains a maintained check.

## Inspection coverage

- Producers: Character defaults, catalog effects, generated strategy actions,
  interpreter factories, event mapping, building/service descriptors and controller
  effects. Fixed happiness catalog defaults; retained intentionally small point
  deltas and unscaled credit/item amounts.
- Shared state: scalar validation/bounds, dynamic result bounds, synchronized
  canonical writes, seeded alias rollback, object compatibility property, graph
  overlays, dictionary inputs and standalone fallback.
- Consumers: strategy ordering, simulated preconditions/projection, numeric goal
  direction/alias reads, GOAP action cost and post-plan action scoring, normalized
  plan scoring, prompt scoring/examples/labels, actual needs rendering.
- Distance cost: corrected the native energy fraction expression. Its sole
  production caller computes a newly added relationship's distance before the
  edge exists, taking the unchanged no-edge branch (distance 10 → cost 10).
- Demo data: shared lightweight factory defaults and explicitly `/10` runtime
  example inputs/labels moved to native points. Historical isolated copied-class
  demonstrations are identified separately in the unit reference and do not
  establish runtime behavior or live model readiness.

## Verification receipts

Commands run from the repository root with offline model flags set:

```bash
python -B scripts/run_ci_checks.py --profile core --artifacts <core-receipts>
python -B scripts/run_ci_checks.py --profile dev --artifacts <dev-receipts>
python -B scripts/run_ci_checks.py --profile visual --artifacts <visual-receipts>
```

Python 3.12/Linux; core omits optional rendering/ML packages. Dev and visual
were installed into separate fresh environments with `--require-hashes
--only-binary=:all:` from their checked-in locks; `pip check` returned
`No broken requirements found.` Neither lockfiles nor timeouts were changed.

All three canonical profiles completed on that source checkpoint:

| Profile | Tests | Failures/errors/skips | CLI probes | Result |
| --- | ---: | --- | ---: | --- |
| core | 116 | 0 / 0 / 0 | 3 | PASS |
| dev | 20 | 0 / 0 / 0 | 0 | PASS |
| visual | 69 | 0 / 0 / 0 | 0 | PASS |

These are profile execution counts, not a claim that all methods are unique
across profiles. New closure coverage is seven core cases and one visual case.
Supplementary pytest execution of the real planner integration and strategy
manager modules passed 11 tests (one existing `uu` deprecation warning).
The compact [local receipt](evidence/issue-740-local-results.json) retains the
actual report counts, source revision, platform, package/lock evidence and hashes
of changed Python sources. Detailed local artifacts are
`/workspace/scratch/740-review-verified-{core,dev,visual}`. Hosted artifacts and their
immutable run links must also be published on the PR/issue; the local paths
alone are not sufficient publication evidence. Hosted and merged-source status
remain pending for the final repair head until verified and merged. All eight
checks passed on the earlier `0ba397c` head ([Python CI](https://github.com/dhar174/tiny_village/actions/runs/43605917099),
[CodeQL](https://github.com/dhar174/tiny_village/actions/runs/43605914118)); those
receipts do not verify the subsequent alias projection correction.

One intermediate core run had 113 tests, one failure, and three passing probes:
it exposed legacy no-effects `Action.satisfaction` utility-rating compatibility.
That fallback was preserved; its existing regression passes alongside the new
native scoring regression. This intermediate failure is not reported as PASS.

## Remaining limitations and why they are separate

- **#802 / #218:** Re-ran the retained full production Goal signature source.
  It exited 1: `completion_signature=()`, energy 30, expected satisfaction True,
  actual False after the state-argument mismatch. This remains a failure, never
  positive completion coverage. #740's scalar/target unit proofs do not repair
  Goal containers, feasibility or full Goal/Plan evaluation.
- **#741:** Active-plan cursors across ticks, replanning and alternative-action
  outcomes remain separate. Correct units do not establish those behaviors.
- **#816:** Direct writers outside the synchronized action/event paths still
  require their broader audit. The specific satisfaction field duplication and
  registered native need event writes encountered here were repaired rather
  than waved away as deferrals.
- **#187 / relationship-cost compatibility:** The older weighted distance-cost
  branch indexes a MultiDiGraph edge bundle as a flat attribute record. With two
  real Characters and an existing edge it raises `KeyError('strength')` on both
  baseline and candidate. This is independent of the energy unit correction,
  is not reached by the sole production add-edge call site, and is not claimed
  as verified relationship behavior or general cost balancing.
- **#184 and later economic work:** Complete catalog preconditions, domain
  inventory/location descriptors, wage payer conservation and the full economy
  episode are not completed by these unit checks.
- Live ML/LLM inference, Windows support, full visual gameplay and save/load
  migrations were not executed or established. Lightweight tutorial substitutes
  and a passing startup probe do not prove them.

## Conditional closeout

Publish the repair PR and this evidence; verify final hosted checks and merged
source identity. Only then mark #740's three original criteria complete, post
a signed summary, close as completed, and read the issue back. Keep all separate
issues, their original requirements and later human gates intact. If the repair
remains unmerged or a required check fails, #740 stays open.
