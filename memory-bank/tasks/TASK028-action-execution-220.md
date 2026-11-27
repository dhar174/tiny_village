# TASK028 — Graph-backed action execution (#220)

**Updated** 2026-11-27 America/New_York

**Status** PR [#817](https://github.com/dhar174/tiny_village/pull/817) merged at `84cbbdb`, and #220 is closed. The P2 readiness follow-up is on `codex/pr-817-action-readiness`.

The approved scope selectively ports draft [#696](https://github.com/dhar174/tiny_village/pull/696)
at `8016a3f467719afa4b795207c14f40502b161410` onto
`67a8abd20d497397cd3c48006a85287d5f082ee2`, the [#815](https://github.com/dhar174/tiny_village/pull/815)
merge. [D035](../../docs/decisions.md#d035--preserve-object-graph-nodes-with-immutable-entity-uuids-799)
continues to govern identity and rollback.

Action resolves registered actors and targets, evaluates concrete conditions
against graph state and applies validated scalar changes in native units.
Controller and Plan invoke the selected action once. The controller owns its
success bookkeeping. Weather drain shares the synchronized attribute write,
which restores the failed attribute and preserves earlier committed effects.

The candidate disposition and local receipts are retained under
`.test-artifacts/`. The comparison covers the six original #220 regression
failures, #799 compatibility controls, and the two canonical Linux profiles.
The #816 weather case checks exact values of 20, 15 and 25 and two writes.
The wider #816 direct-writer audit and the #740/#802 planner changes remain
separate follow-ups. #741 retains the replanning work and #602 retains the
integrated gameplay acceptance gate.

The Zcode review follow-up restores the warning for an empty daily action list
and records a failed attempt when a selected action cannot execute. GitHub
confirmed the pinned #696 source commit above. Plan execution still stops
on failure. The retained recovery methods belong to the #741 replanning work.


## Precondition readiness follow-up

Approved with “Please implement the plan.” Base: main `5c2eae0`.
[PR #817 P2 finding](https://github.com/dhar174/tiny_village/pull/817#discussion_r4565495408).

- On the baseline, both native queue loops invoked an action blocked at energy
  20 or 21 by an energy-30 requirement, counted its rejection as a failure,
  and never reached the independent recovery action.
- Live readiness shares execution's typed actor, graph and implicit-target
  binding. The preview restores all three references on every exit. Both loops
  skip rejected entries without credit or failure accounting. An eligible
  action is invoked once through the existing adapter, with its execution-time
  guard and rollback unchanged. Legacy readiness and existing replan limits
  remain intact.
- The two loop paths, explicit partner and exceptional preview regressions
  failed against the baseline. Six targeted tests passed on the corrected
  source, including the partial-write/no-retry, repeated dependency and
  stale live-plan tail controls. The follow-up PR records canonical results.
- Planning/actions, test/CI and independent reviewer roles reviewed supplied
  excerpts and coordinator evidence. The coordinator alone wrote files and ran
  checks. No native Zcode invocation or independent checkout access is claimed.
- PR #815 needs no code change. Merge and Phase 4 closure remain separate.
  Resolve the original P2 thread only after verifying the repair on main.
